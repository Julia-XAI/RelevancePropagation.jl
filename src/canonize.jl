#=============================#
# Functions to fuse layers    #
#=============================#

function fuse_batchnorm(d::Dense, bn::BatchNorm)
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

function fuse_batchnorm(c::Conv, bn::BatchNorm)
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

is_fuseable(l::Union{Dense,Conv}, bn::BatchNorm) = activation_fn(l) == identity
is_fuseable(l1, l2) = false

#=============================#
# Functions to split layers   #
#=============================#

function split_layer(l::LayerNorm)
    layer_norm = LayerNorm(identity, identity, l.ϵ, l.size, false)
    if l.diag isa Scale
        diag = l.diag
    else # in case the LayerNorm layer does not contain an affine transformation, but an activation
        diag = Scale(1, l.λ; bias=false)
        diag.scale .= 1.0
    end
    return (layer_norm, diag)
end

is_splittable(l::LayerNorm) = true
is_splittable(l::LayerNorm{F,D,T,N}) where {F,D<:typeof(identity),T,N} = false # don't split any further if the affine part is already the identity
is_splittable(l) = false

#=================================#
# Canonize model (split and fuse) #
#=================================#

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

function canonize_split(model::Chain)
    model = Chain(canonize_split.(model.layers)) # recursively canonize Parallel layers

    i = 1
    while i <= length(model)
        l = model[i]

        if is_splittable(l)
            splitted = split_layer(l)
            model = Chain(model[1:(i - 1)]..., splitted..., model[(i + 1):end]...)
            # if fused, don't increment i,
            # instead try fusing the new layer with the next one
        else
            i += 1
        end
    end
    return model
end

function canonize_split(p::Parallel)
    return Parallel(p.connection, canonize_split.(p.layers))
end

canonize_split(layer) = layer

function canonize_fuse(model::Chain)
    model = Chain(canonize_fuse.(model.layers)) # recursively canonize Parallel layers

    i = 1
    while i < length(model)
        l1, l2 = model[i:(i + 1)]

        if is_fuseable(l1, l2)
            fused = fuse_batchnorm(l1, l2)
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

canonize_fuse(layer) = layer
