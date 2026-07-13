#==========================#
# Strip output activations #
#==========================#

has_output_softmax(model::Chain) = has_output_softmax(last_element(model))
has_output_softmax(x) = is_softmax(x) || is_softmax(activation_fn(x))

is_softmax(x) = x isa SoftmaxActivation
is_softmax(l::WrappedFunction) = is_softmax(l.func)

"""
    strip_softmax(model)

Remove softmax activation on the model output if it exists.

An output layer with softmax activation has its activation replaced by
`identity`; a bare output softmax (a `WrappedFunction`) is replaced by a
`NoOpLayer`, preserving the length of the chain.
Since activation functions are part of the layer configuration in Lux,
`ps` and `st` are unaffected.
"""
function strip_softmax(model::Chain)
    has_output_softmax(model) || return model
    return strip_output_softmax(model)
end

# Descend into the last entry of nested `Chain`s, preserving layer keys.
function strip_output_softmax(c::Chain)
    ks = keys(c.layers)
    vals = values(c.layers)
    stripped = (Base.front(vals)..., strip_output_softmax(last(vals)))
    return Chain(NamedTuple{ks}(stripped))
end
strip_output_softmax(l::WrappedFunction) = is_softmax(l) ? NoOpLayer() : l
strip_output_softmax(l) = is_softmax(activation_fn(l)) ? remove_activation(l) : l
