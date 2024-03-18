#================================#
# Canonize model: split and fuse #
#================================#

"""
    canonize(model)

Canonize model by flattening it and fusing BatchNorm layers into preceding Dense and Conv
layers with linear activation functions.
"""
function canonize(model::Chain)
    model = canonize_split(model)
    model = canonize_fuse(flatten_model(model))
    return model
end

#==============#
# Split layers #
#==============#

canonize_split(model::Chain) = Chain(canonize_split(model.layers)...)
canonize_split(p::Parallel) = Parallel(p.connection, canonize_split.(p.layers))
canonize_split(s::SkipConnection) = SkipConnection(canonize_split(s.layers), s.connection)
canonize_split(layer) = layer
canonize_split(layers::Tuple) = canonize_split.(layers)
canonize_split(layers::AbstractArray) = canonize_split.(layers)

# Don't split LayerNorm if the affine part is already the identity
canonize_split(l::LayerNorm{F,D}) where {F,D<:typeof(identity)} = l

function canonize_split(l::LayerNorm)
    layer_norm = LayerNorm(identity, identity, l.ϵ, l.size, false)
    if l.diag isa Scale
        diag = l.diag
    else # LayerNorm does not contain an affine transformation, but an activation
        diag = Scale(1, l.λ; bias=false)
        diag.scale .= 1.0
    end
    return Chain(layer_norm, diag)
end

#=============#
# Fuse layers #
#=============#

function canonize_fuse(model::Chain)
    model = Chain(canonize_fuse.(model.layers)) # recursively canonize Parallel layers

    i = 1
    while i < length(model)
        l1, l2 = model[i:(i + 1)]

        if is_fuseable(l1, l2)
            fused = canonize_fuse(l1, l2)
            model = Chain(model[1:(i - 1)]..., fused, model[(i + 2):end]...)
            # if fused, don't increment i,
            # instead try fusing the new layer with the next one
        else
            i += 1
        end
    end
    return model
end

function canonize_fuse(p::Parallel)
    return Parallel(p.connection, canonize_fuse.(p.layers))
end

function canonize_fuse(s::SkipConnection)
    return SkipConnection(canonize_fuse(s.layers), s.connection)
end

canonize_fuse(layer) = layer

# If two layers satisfy `is_fuseable`, the two-argument `canonize_fuse(l1, l2)` is called.

is_fuseable(l::Union{Dense,Conv}, bn::BatchNorm) = activation_fn(l) == identity
is_fuseable(l1, l2) = false

# Fuse BatchNorm layers into Dense and Conv layers

function canonize_fuse(d::Dense, bn::BatchNorm)
    d.σ != identity &&
        throw(ArgumentError("Can't fuse Dense layer with activation $(d.σ)."))
    scale = safedivide(bn.γ, sqrt.(bn.σ²))
    W = scale .* d.weight
    b = if d.bias != false
        scale .* (d.bias - bn.μ) + bn.β
    else
        -scale .* bn.μ + bn.β
    end
    return Dense(W, b, bn.λ)
end

function canonize_fuse(c::Conv, bn::BatchNorm)
    c.σ != identity && throw(ArgumentError("Can't fuse Conv layer with activation $(c.σ)."))
    scale = safedivide(bn.γ, sqrt.(bn.σ²))
    W = c.weight .* reshape(scale, 1, 1, 1, :)
    b = if c.bias != false
        scale .* (c.bias - bn.μ) + bn.β
    else
        -scale .* bn.μ + bn.β
    end
    return Conv(bn.λ, W, b, c.stride, c.pad, c.dilation, c.groups)
end
