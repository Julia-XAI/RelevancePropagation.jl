#===============#
# Flatten model #
#===============#

"""
    flatten_model(model, ps, st; exclude, unwrap)

Flatten a Lux `Chain` containing nested `Chain`s.
Returns a `(model, ps, st)` triple whose layers are re-keyed to
`layer_1, ..., layer_N`; since this re-keys `ps` and `st` as well,
the transformation is joint over the Lux triple.

`Parallel` and `SkipConnection` layers keep their container,
but their branches are flattened internally.

# Keyword arguments
- `exclude(kp::KeyPath, layer)::Bool`: layers for which `exclude` returns
  `true` are kept intact, taking precedence over `unwrap`.
  Defaults to `Returns(false)`.
- `unwrap(layer)::Bool`: opt-in unwrapping of `AbstractLuxWrapperLayer`s whose
  application is a plain pass-through to the layer they wrap (e.g. Boltz.jl
  model wrappers). The wrapper is dropped and a wrapped `Chain` is spliced into
  its parent. Defaults to `Returns(false)`: the `AbstractLuxWrapperLayer` trait
  only guarantees `ps`/`st` transparency, not application transparency, so
  unwrapping unknown wrappers (e.g. `Maxout`, `RepeatedLayer`, the Lux pooling
  wrappers) would corrupt the model.
"""
function flatten_model(model, ps, st; exclude=Returns(false), unwrap=Returns(false))
    root = KeyPath()
    exclude(root, model)::Bool && return model, ps, st
    return flatten_layer(root, model, ps, st, exclude, unwrap)
end

# Recursion entry for non-root layers: `exclude` takes precedence over the
# per-container methods below.
function flatten_child(kp::KeyPath, layer, ps, st, exclude, unwrap)
    exclude(kp, layer)::Bool && return layer, ps, st
    return flatten_layer(kp, layer, ps, st, exclude, unwrap)
end

# Flatten the insides of a layer. Only called on non-excluded layers.
flatten_layer(kp::KeyPath, layer, ps, st, exclude, unwrap) = layer, ps, st

function flatten_layer(kp::KeyPath, c::Chain, ps, st, exclude, unwrap)
    layers, pss, sts = [], [], []
    for k in keys(c.layers)
        child_kp = KeyPath(kp, k)
        child = c.layers[k]
        if exclude(child_kp, child)::Bool
            # Excluded children stay intact, even excluded nested `Chain`s.
            push!(layers, child)
            push!(pss, ps[k])
            push!(sts, st[k])
            continue
        end
        layer, p, s = flatten_layer(child_kp, child, ps[k], st[k], exclude, unwrap)
        if layer isa Chain # splice nested (already flattened) `Chain`s in
            append!(layers, values(layer.layers))
            append!(pss, values(p))
            append!(sts, values(s))
        else
            push!(layers, layer)
            push!(pss, p)
            push!(sts, s)
        end
    end
    flat_model = Chain(layers...)
    ks = keys(flat_model.layers)
    return flat_model, NamedTuple{ks}(Tuple(pss)), NamedTuple{ks}(Tuple(sts))
end

function flatten_layer(kp::KeyPath, p::Parallel, ps, st, exclude, unwrap)
    ks = keys(p.layers)
    branches = map(
        k -> flatten_child(KeyPath(kp, k), p.layers[k], ps[k], st[k], exclude, unwrap), ks
    )
    layers = NamedTuple{ks}(map(b -> b[1], branches))
    flat_ps = NamedTuple{ks}(map(b -> b[2], branches))
    flat_st = NamedTuple{ks}(map(b -> b[3], branches))
    return setproperties(p, (; layers)), flat_ps, flat_st
end

function flatten_layer(kp::KeyPath, s::SkipConnection, ps, st, exclude, unwrap)
    # `SkipConnection` is an `AbstractLuxWrapperLayer`:
    # its `ps`/`st` pass through to the wrapped layer directly.
    inner, flat_ps, flat_st = flatten_child(kp, s.layers, ps, st, exclude, unwrap)
    return setproperties(s, (; layers=inner)), flat_ps, flat_st
end

# Wrapper layers pass `ps`/`st` through to the layer they wrap. Opted-in
# wrappers are dropped so wrapped `Chain`s get spliced into their parents;
# all others are kept intact (see the `unwrap` docstring above).
function flatten_layer(
    kp::KeyPath, l::AbstractLuxWrapperLayer{field}, ps, st, exclude, unwrap
) where {field}
    unwrap(l)::Bool || return l, ps, st
    return flatten_child(kp, getfield(l, field), ps, st, exclude, unwrap)
end
