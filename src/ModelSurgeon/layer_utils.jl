"""
    activation_fn(layer)

Return activation function of a Lux layer.
In case the layer is unknown or has no activation function, `nothing` is returned.
"""
activation_fn(layer) = hasfield(typeof(layer), :activation) ? layer.activation : nothing

"""
    remove_activation(layer)

Return a copy of the Lux layer with its activation function set to `identity`.
Layers without an activation function are returned unchanged.
"""
function remove_activation(layer)
    isnothing(activation_fn(layer)) && return layer
    return setproperties(layer, (; activation=identity))
end
