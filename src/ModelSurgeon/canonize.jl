#================================#
# Canonize model: split and fuse #
#================================#

"""
    canonize(model, ps, st; exclude, unwrap)

Canonize a model by splitting fused layers ([`canonize_split`](@ref)),
flattening it ([`flatten_model`](@ref)) and fusing BatchNorm layers into
preceding Dense and Conv layers with linear activation functions
([`canonize_fuse`](@ref)).
Returns a `(model, ps, st)` triple, since fusing parameters and flattening
re-key `ps` and `st`.

BatchNorm layers are fused using the running statistics in `st`
(collected by applying the model in train mode).
LayerNorm layers containing an affine transformation or an activation
function are split into a normalization-only LayerNorm followed by a
`Scale` layer carrying both.

The `exclude` and `unwrap` keyword arguments are documented in
[`flatten_model`](@ref); excluded layers are kept intact by all three passes.
"""
function canonize(model, ps, st; exclude=Returns(false), unwrap=Returns(false))
    split_model, split_ps, split_st = canonize_split(model, ps, st; exclude)
    flat = flatten_model(split_model, split_ps, split_st; exclude, unwrap)
    return canonize_fuse(flat...)
end

#==============#
# Split layers #
#==============#

"""
    canonize_split(model, ps, st; exclude)

Split layers that bundle several operations into separate layers, using
[`split_activation`](@ref): currently `LayerNorm` layers with an affine
transformation or activation function. Returns a `(model, ps, st)` triple
whose nested `Chain`s are spliced into their parents by
[`flatten_model`](@ref).
"""
function canonize_split(model, ps, st; exclude=Returns(false))
    return map_triple(split_norm, model, ps, st; exclude)
end

split_norm(layer, ps, st, kp::KeyPath) = layer, ps, st
split_norm(l::LayerNorm, ps, st, kp::KeyPath) = split_activation(l, ps, st)

"""
    split_activation(layer, ps, st)

Split a layer into a `Chain` of the layer with its activation function
removed, followed by a separate layer applying the activation.
Returns a `(layer, ps, st)` triple; the nested `Chain` is spliced into its
parent by [`flatten_model`](@ref).

Layers without an activation function or with `identity` activation are
returned unchanged. For `LayerNorm`, the affine transformation is split out
together with the activation into a `Scale` layer, leaving a
normalization-only `LayerNorm`.

To support custom layers, add methods on your own layer types
(this is piracy-free):
```julia
ModelSurgeon.split_activation(l::MyLayer, ps, st) = ...
```
"""
function split_activation(layer, ps, st)
    σ = activation_fn(layer)
    (isnothing(σ) || σ === identity) && return layer, ps, st
    split = Chain(remove_activation(layer), WrappedFunction(Base.Fix1(broadcast, σ)))
    return split, (; layer_1=ps, layer_2=NamedTuple()), (; layer_1=st, layer_2=NamedTuple())
end

function split_activation(l::LayerNorm, ps, st)
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
    return split,
    (; layer_1=NamedTuple(), layer_2=scale_ps),
    (; layer_1=st, layer_2=NamedTuple())
end

#=============#
# Fuse layers #
#=============#

"""
    canonize_fuse(model, ps, st)

Fuse adjacent layers in all `Chain`s of a model wherever
[`is_fuseable`](@ref) holds, using the five-argument
`canonize_fuse(layer1, ps1, layer2, ps2, st2)` method for the fused pair.
Returns a `(model, ps, st)` triple whose `Chain`s are re-keyed to
`layer_1, ..., layer_N`.

[`is_fuseable`](@ref) and the five-argument `canonize_fuse` form the fusion
extension API: to add fusions beyond Dense/Conv+BatchNorm, add methods for
your own layer types (this is piracy-free).
"""
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

"""
    is_fuseable(layer1, layer2, st2)

Determine whether two adjacent layers can be fused by
[`canonize_fuse`](@ref), given the states `st2` of the second layer.
"""
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
