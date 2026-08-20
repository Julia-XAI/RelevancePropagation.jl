"""Union type for dataflow layers."""
const DataflowLayer = Union{Chain,Parallel,SkipConnection}

"""Union type for convolutional layers.

Cross-correlation layers are constructed in Lux via `Conv(...; cross_correlation=true)`
and are therefore covered by `Conv`.
"""
const ConvLayer = Union{Conv,ConvTranspose}

"""Union type for dropout layers."""
const DropoutLayer = Union{Dropout,AlphaDropout,VariationalHiddenDropout}

"""Union type for reshaping layers such as `FlattenLayer`."""
const ReshapingLayer = Union{FlattenLayer,ReshapeLayer}

"""Union type for max pooling layers."""
const MaxPoolLayer = Union{MaxPool,AdaptiveMaxPool,GlobalMaxPool}

"""Union type for mean pooling layers."""
const MeanPoolLayer = Union{MeanPool,AdaptiveMeanPool,GlobalMeanPool}

"""Union type for pooling layers."""
const PoolingLayer = Union{MaxPoolLayer,MeanPoolLayer}

"""Union type for LP-norm pooling layers.

Kept separate from `PoolingLayer` so consumers can decide whether to treat
LP-norm pooling like other pooling layers.
"""
const LPPoolLayer = Union{LPPool,AdaptiveLPPool,GlobalLPPool}

"""Union type for normalization layers."""
const NormalizationLayer = Union{BatchNorm}

"""Union type for softmax activation functions."""
const SoftmaxActivation = Union{typeof(softmax),typeof(softmax!)}

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
