#====================#
# Model tree helpers #
#====================#

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

#==================#
# Triple traversal #
#==================#

# Traversal policy: which layers a traversal descends into.
# `Chain`, `Parallel` and `SkipConnection` are always descended. All other
# `AbstractLuxWrapperLayer`s are kept intact unless opted in via `unwrap`:
# the trait only guarantees `ps`/`st` transparency, not application
# transparency, so descending unknown wrappers by default would corrupt
# layers like `Maxout` (which wraps a raw `NamedTuple` of branches) and
# `RepeatedLayer` (which applies its inner model several times).
descends(::DataflowLayer, unwrap) = true
descends(layer::AbstractLuxWrapperLayer, unwrap) = unwrap(layer)::Bool
descends(layer, unwrap) = false

"""
    map_triple(f, model, ps, st; exclude, unwrap)

Walk a Lux `(model, ps, st)` triple jointly, applying
`f(layer, ps, st, kp::Functors.KeyPath) -> (layer, ps, st)` to each leaf layer
and rebuilding the containers around the results.
Returns the mapped `(model, ps, st)` triple.

The `KeyPath` passed to `f` follows the structure of `ps` and `st`:
wrapper layers like `SkipConnection` are transparent and do not add a key.
`f` may return a replacement subtree, e.g. turn a single layer into a `Chain`
of several layers (see [`split_activation`](@ref)).

# Keyword arguments
- `exclude(kp::KeyPath, layer)::Bool`: layers for which `exclude` returns
  `true` are treated as leaves and passed to `f` whole, even if the traversal
  could descend into them. Defaults to `Returns(false)`.
- `unwrap(layer)::Bool`: opt-in descent into `AbstractLuxWrapperLayer`s beyond
  `Chain`, `Parallel` and `SkipConnection`. Opted-in wrappers are recursed
  transparently (no `KeyPath` key) and rebuilt around the mapped contents.
  Defaults to `Returns(false)`, since unknown wrapper layers are only
  guaranteed to be `ps`/`st`-transparent, not application-transparent.
"""
function map_triple(f, model, ps, st; exclude=Returns(false), unwrap=Returns(false))
    isleaf(kp::KeyPath, layer) = exclude(kp, layer)::Bool || !descends(layer, unwrap)
    # Tied layers can occur several times in a model, but their `ps`/`st`
    # entries are per-occurrence — disable Functors' object-identity cache.
    return fmap_with_path(
        model, ps, st; exclude=isleaf, walk=TripleWalk(), cache=nothing
    ) do kp, layer, ps_layer, st_layer
        f(layer, ps_layer, st_layer, kp)
    end
end

struct TripleWalk <: Functors.AbstractWalk end

# Containers with a `layers` NamedTuple: recurse per child, extending the
# `KeyPath` by the child's key, and re-key `ps`/`st` alongside.
function (::TripleWalk)(recurse, kp::KeyPath, layer::Union{Chain,Parallel}, ps, st)
    ks = keys(layer.layers)
    triples = map(k -> recurse(KeyPath(kp, k), layer.layers[k], ps[k], st[k]), ks)
    layers = NamedTuple{ks}(map(t -> t[1], triples))
    new_ps = NamedTuple{ks}(map(t -> t[2], triples))
    new_st = NamedTuple{ks}(map(t -> t[3], triples))
    return setproperties(layer, (; layers)), new_ps, new_st
end

# Wrapper layers (`SkipConnection`, wrappers opted in via `unwrap`) pass
# `ps`/`st` through to the wrapped layer: recurse with an unchanged `KeyPath`
# and rebuild the wrapper around the result.
function (::TripleWalk)(
    recurse, kp::KeyPath, layer::AbstractLuxWrapperLayer{field}, ps, st
) where {field}
    inner, new_ps, new_st = recurse(kp, getfield(layer, field), ps, st)
    return setproperties(layer, NamedTuple{(field,)}((inner,))), new_ps, new_st
end
