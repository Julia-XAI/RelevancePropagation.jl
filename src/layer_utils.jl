"""
    activation_fn(layer)

Return activation function of a Lux layer.
In case the layer is unknown or has no activation function, `nothing` is returned.
"""
activation_fn(layer) = hasfield(typeof(layer), :activation) ? layer.activation : nothing
activation_fn(f::FrozenLayer) = activation_fn(f.layer)

"""
    remove_activation(layer)

Return a copy of the Lux layer with its activation function set to `identity`.
Layers without an activation function are returned unchanged.
"""
function remove_activation(layer)
    isnothing(activation_fn(layer)) && return layer
    return setproperties(layer, (; activation=identity))
end

# Parameters live in `ps`, whose entries Lux names uniformly:
# `weight` and `bias` for all layers modified by LRP rules (`Dense`, `Scale`,
# convolutions). Layers constructed with `use_bias=false` have no `bias` key.
has_weight(f::FrozenLayer) = haskey(f.ps, :weight)
has_bias(f::FrozenLayer) = haskey(f.ps, :bias)

# Structure helpers for walking Lux models.
# `Chain` and `Parallel` store their children in a `layers` NamedTuple that
# `ps` and `st` mirror; `SkipConnection` is an `AbstractLuxWrapperLayer` whose
# `ps`/`st` pass through to the wrapped layer directly.
children_layers(c::Chain) = values(c.layers)
children_layers(p::Parallel) = values(p.layers)
children_layers(s::SkipConnection) = (s.layers,)

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
