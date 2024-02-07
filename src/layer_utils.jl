"""
    activation_fn(layer)

Return activation function of the layer.
In case the layer is unknown or no activation function is found, `nothing` is returned.
"""
activation_fn(layer) = nothing
activation_fn(l::Dense)         = l.σ
activation_fn(l::Conv)          = l.σ
activation_fn(l::CrossCor)      = l.σ
activation_fn(l::ConvTranspose) = l.σ
activation_fn(l::BatchNorm)     = l.λ
activation_fn(l::LayerNorm)     = l.λ
activation_fn(l::InstanceNorm)  = l.λ
activation_fn(l::GroupNorm)     = l.λ

has_weight(layer) = hasproperty(layer, :weight)
has_bias(layer) = hasproperty(layer, :bias)
has_weight_and_bias(layer) = has_weight(layer) && has_bias(layer)

"""
    copy_layer(layer, W, b, [σ=identity])

Copy layer using weights `W` and `b`. The activation function `σ` can also be set,
defaulting to `identity`.
"""
copy_layer(::Dense, W, b; σ=identity) = Dense(W, b, σ)
function copy_layer(l::Conv, W, b; σ=identity)
    return Conv(W, b, σ; stride=l.stride, pad=l.pad, dilation=l.dilation, groups=l.groups)
end
function copy_layer(l::ConvTranspose, W, b; σ=identity)
    return ConvTranspose(
        W, b, σ; stride=l.stride, pad=l.pad, dilation=l.dilation, groups=l.groups
    )
end
function copy_layer(l::CrossCor, W, b; σ=identity)
    return CrossCor(W, b, σ; stride=l.stride, pad=l.pad, dilation=l.dilation)
end
