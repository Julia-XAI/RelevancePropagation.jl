#=====================#
# Model tree helpers  #
#=====================#

# Generic helpers for walking Lux models.
# `Chain` and `Parallel` store their children in a `layers` NamedTuple that
# `ps` and `st` mirror; `SkipConnection` is an `AbstractLuxWrapperLayer` whose
# `ps`/`st` pass through to the wrapped layer directly.
children_layers(c::Chain) = values(c.layers)
children_layers(p::Parallel) = values(p.layers)
children_layers(s::SkipConnection) = (s.layers,)

"""
    map_layers(f, model)

Apply `f` to each layer of a Lux model, mirroring the model structure
as nested `NamedTuple`s keyed like the model's `ps` and `st`.
"""
function map_layers(f, model::Union{Chain,Parallel})
    layers = model.layers
    return NamedTuple{keys(layers)}(map(l -> map_layers(f, l), values(layers)))
end
map_layers(f, model::SkipConnection) = map_layers(f, model.layers)
map_layers(f, layer) = f(layer)

"""
    chainall(f, model)

Determines whether `f` returns `true` for all layers in a Lux model.
"""
chainall(f, layer) = f(layer)
chainall(f, model::DataflowLayer) = all(chainall(f, l) for l in children_layers(model))

"""
    first_element(model)

Returns first layer of a Lux `Chain`, descending into nested `Chain`s.
"""
first_element(c::Chain) = first_element(first(values(c.layers)))
first_element(layer) = layer

"""
    last_element(model)

Returns last layer of a Lux `Chain`, descending into nested `Chain`s.
"""
last_element(c::Chain) = last_element(last(values(c.layers)))
last_element(layer) = layer

# Bundle the children of a `Chain` or `Parallel` with their `ps`/`st` into a
# NamedTuple of `FrozenLayer`s mirroring `model.layers`.
function frozen_children(f::FrozenLayer{<:Union{Chain,Parallel}})
    layers = f.layer.layers
    return NamedTuple{keys(layers)}(
        map(FrozenLayer, values(layers), values(f.ps), values(f.st))
    )
end

# `SkipConnection` is an `AbstractLuxWrapperLayer`:
# its `ps`/`st` pass through to the wrapped layer directly.
frozen_inner(f::FrozenLayer{<:SkipConnection}) = FrozenLayer(f.layer.layers, f.ps, f.st)

# Compute activations of all layers, including the input.
# Returns a tuple `(input, a¹, a², ..., aᴺ)` of length `length(layers) + 1`.
get_activations(layers::NamedTuple, input) = (input, _activations(values(layers), input)...)

function _activations(layers::Tuple, x)
    isempty(layers) && return ()
    y = first(layers)(x)
    return (y, _activations(Base.tail(layers), y)...)
end

#===============#
# Flatten model #
#===============#

"""
    flatten_model(model, ps, st)

Flatten a Lux `Chain` containing nested `Chain`s.
Returns a `(model, ps, st)` triple whose layers are re-keyed to
`layer_1, ..., layer_N`; since this re-keys `ps` and `st` as well,
the transformation is joint over the Lux triple.

`Parallel` and `SkipConnection` layers keep their container,
but their branches are flattened internally.

Generic `AbstractLuxWrapperLayer`s (e.g. Boltz.jl model wrappers) pass `ps`
and `st` through to the layer they wrap; `flatten_model` unwraps them, both
at the top level and inside `Chain`s.
"""
function flatten_model(model::AbstractLuxWrapperLayer{field}, ps, st) where {field}
    return flatten_model(getfield(model, field), ps, st)
end
function flatten_model(model::Chain, ps, st)
    layers, pss, sts = flatten_chain(model, ps, st)
    flat_model = Chain(layers...)
    ks = keys(flat_model.layers)
    return flat_model, NamedTuple{ks}(Tuple(pss)), NamedTuple{ks}(Tuple(sts))
end

# Return vectors of layer, ps and st entries with nested `Chain`s spliced in.
function flatten_chain(c::Chain, ps, st)
    layers, pss, sts = [], [], []
    for k in keys(c.layers)
        layer, p, s = flatten_layer(c.layers[k], ps[k], st[k])
        if layer isa Chain
            append!(layers, values(layer.layers))
            append!(pss, values(p))
            append!(sts, values(s))
        else
            push!(layers, layer)
            push!(pss, p)
            push!(sts, s)
        end
    end
    return layers, pss, sts
end

# Flatten the insides of a layer.
flatten_layer(layer, ps, st) = layer, ps, st
flatten_layer(c::Chain, ps, st) = flatten_model(c, ps, st)
function flatten_layer(p::Parallel, ps, st)
    ks = keys(p.layers)
    branches = map(k -> flatten_layer(p.layers[k], ps[k], st[k]), ks)
    layers = NamedTuple{ks}(map(first, branches))
    flat_ps = NamedTuple{ks}(map(b -> b[2], branches))
    flat_st = NamedTuple{ks}(map(b -> b[3], branches))
    return setproperties(p, (; layers)), flat_ps, flat_st
end
function flatten_layer(s::SkipConnection, ps, st)
    # `SkipConnection` is an `AbstractLuxWrapperLayer`:
    # its `ps`/`st` pass through to the wrapped layer directly.
    inner, flat_ps, flat_st = flatten_layer(s.layers, ps, st)
    return setproperties(s, (; layers=inner)), flat_ps, flat_st
end
# Generic wrapper layers (e.g. Boltz.jl model wrappers) pass `ps`/`st` through
# to the layer they wrap — unwrap them so wrapped `Chain`s get spliced in.
function flatten_layer(l::AbstractLuxWrapperLayer{field}, ps, st) where {field}
    return flatten_layer(getfield(l, field), ps, st)
end
# Lux pooling layers are `AbstractLuxWrapperLayer`s around internal pooling
# ops and must stay intact — rules and composites dispatch on `PoolingLayer`.
flatten_layer(l::PoolingLayer, ps, st) = l, ps, st
