#================================#
# Canonize model: split and fuse #
#================================#

"""
    canonize(model, ps, st)

Canonize a model by flattening it and fusing BatchNorm layers into preceding
Dense and Conv layers with linear activation functions.
Returns a `(model, ps, st)` triple, since fusing parameters and flattening
re-key `ps` and `st`.

BatchNorm layers are fused using the running statistics in `st`
(collected by applying the model in train mode).
LayerNorm layers containing an affine transformation or an activation
function are split into a normalization-only LayerNorm followed by a
`Scale` layer carrying both.
"""
function canonize(model::Chain, ps, st)
    return canonize_fuse(flatten_model(canonize_split(model, ps, st)...)...)
end

#==============#
# Split layers #
#==============#

function canonize_split(model::Union{Chain,Parallel}, ps, st)
    ks = keys(model.layers)
    triples = map(k -> canonize_split(model.layers[k], ps[k], st[k]), ks)
    layers = NamedTuple{ks}(map(first, triples))
    split_ps = NamedTuple{ks}(map(t -> t[2], triples))
    split_st = NamedTuple{ks}(map(t -> t[3], triples))
    return setproperties(model, (; layers)), split_ps, split_st
end
function canonize_split(s::SkipConnection, ps, st)
    # `SkipConnection` is an `AbstractLuxWrapperLayer`:
    # its `ps`/`st` pass through to the wrapped layer directly.
    inner, split_ps, split_st = canonize_split(s.layers, ps, st)
    return setproperties(s, (; layers=inner)), split_ps, split_st
end
canonize_split(layer, ps, st) = layer, ps, st

function canonize_split(l::LayerNorm, ps, st)
    affine = haskey(ps, :scale)
    # Don't split LayerNorm if the affine part is already the identity
    !affine && l.activation === identity && return l, ps, st

    norm = LayerNorm(l.shape, identity; dims=l.dims, epsilon=l.epsilon, affine=false)
    if affine
        # Lux sizes LayerNorm's affine parameters `(shape..., 1)`
        scale = Scale(l.shape, l.activation)
        scale_ps = (; weight=reshape(ps.scale, l.shape), bias=reshape(ps.bias, l.shape))
    else # LayerNorm contains no affine transformation, only an activation
        scale = Scale(l.shape, l.activation; use_bias=false)
        scale_ps = (; weight=ones(Float32, l.shape))
    end
    split = Chain(norm, scale)
    # The nested `Chain` is spliced into its parent by `flatten_model`
    return split,
    (; layer_1=NamedTuple(), layer_2=scale_ps),
    (; layer_1=st, layer_2=NamedTuple())
end

#=============#
# Fuse layers #
#=============#

function canonize_fuse(model::Chain, ps, st)
    # Recursively canonize Parallel and SkipConnection layers first
    layers, pss, sts = [], [], []
    for k in keys(model.layers)
        layer, p, s = canonize_fuse(model.layers[k], ps[k], st[k])
        push!(layers, layer)
        push!(pss, p)
        push!(sts, s)
    end

    i = 1
    while i < length(layers)
        if is_fuseable(layers[i], layers[i + 1], sts[i + 1])
            layers[i], pss[i] = canonize_fuse(
                layers[i], pss[i], layers[i + 1], pss[i + 1], sts[i + 1]
            )
            deleteat!(layers, i + 1)
            deleteat!(pss, i + 1)
            deleteat!(sts, i + 1)
            # if fused, don't increment i,
            # instead try fusing the new layer with the next one
        else
            i += 1
        end
    end
    fused_model = Chain(layers...)
    ks = keys(fused_model.layers)
    return fused_model, NamedTuple{ks}(Tuple(pss)), NamedTuple{ks}(Tuple(sts))
end

function canonize_fuse(p::Parallel, ps, st)
    ks = keys(p.layers)
    triples = map(k -> canonize_fuse(p.layers[k], ps[k], st[k]), ks)
    layers = NamedTuple{ks}(map(first, triples))
    fused_ps = NamedTuple{ks}(map(t -> t[2], triples))
    fused_st = NamedTuple{ks}(map(t -> t[3], triples))
    return setproperties(p, (; layers)), fused_ps, fused_st
end
function canonize_fuse(s::SkipConnection, ps, st)
    inner, fused_ps, fused_st = canonize_fuse(s.layers, ps, st)
    return setproperties(s, (; layers=inner)), fused_ps, fused_st
end
canonize_fuse(layer, ps, st) = layer, ps, st

# If two layers satisfy `is_fuseable`, the five-argument `canonize_fuse` is called.
function is_fuseable(l::Union{Dense,Conv}, bn::BatchNorm, st_bn)
    return activation_fn(l) === identity && haskey(st_bn, :running_mean)
end
is_fuseable(l1, l2, st2) = false

"""
    canonize_fuse(layer, ps_layer, bn, ps_bn, st_bn)

Fuse a BatchNorm layer into a preceding Dense or Conv layer
with identity activation function.
Returns the fused `(layer, ps)` pair;
the fused layer takes the BatchNorm's activation function.
"""
function canonize_fuse(layer::Union{Dense,Conv}, ps_layer, bn::BatchNorm, ps_bn, st_bn)
    activation_fn(layer) !== identity &&
        throw(ArgumentError("Can't fuse layer with activation $(activation_fn(layer))."))
    μ, σ² = st_bn.running_mean, st_bn.running_var
    γ = haskey(ps_bn, :scale) ? ps_bn.scale : one.(μ)  # BatchNorm(...; affine=false)
    β = haskey(ps_bn, :bias) ? ps_bn.bias : zero.(μ)
    scale = γ ./ sqrt.(σ² .+ bn.epsilon)

    weight = fuse_weight(layer, ps_layer.weight, scale)
    bias = if haskey(ps_layer, :bias)
        scale .* (ps_layer.bias .- μ) .+ β
    else
        β .- scale .* μ
    end
    fused = setproperties(layer, (; activation=bn.activation, use_bias=static(true)))
    return fused, (; weight, bias)
end

# Scale the weight along its output dimension:
# rows for Dense, the trailing output-channel dimension for Conv.
fuse_weight(::Dense, w::AbstractMatrix, scale) = scale .* w
function fuse_weight(::Conv, w::AbstractArray, scale)
    return w .* reshape(scale, ntuple(Returns(1), ndims(w) - 1)..., :)
end
