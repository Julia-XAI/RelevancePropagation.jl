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

# LP pooling layers are not covered by the exported `PoolingLayer` union used
# for rule dispatch (LRP has no rules for them), but they must stay intact
# when flattening — see `keep_pooling` in `model_surgery.jl`.
"""Union type for LP-norm pooling layers."""
const LPPoolLayer = Union{LPPool,AdaptiveLPPool,GlobalLPPool}

"""Union type for normalization layers."""
const NormalizationLayer = Union{BatchNorm}

# Activation functions
"""Union type for ReLU-like activation functions."""
const ReluLikeActivation = Union{typeof(relu),typeof(gelu),typeof(swish),typeof(mish)}

# Layers & activation functions supported by LRP
"""Union type for activation functions that are allowed by default in "deep rectifier networks"."""
const LRPSupportedActivation = Union{typeof(identity),ReluLikeActivation}

"""Union type for layers that are allowed by default in "deep rectifier networks".
This includes the usage of allowed activation functions as layers,
which Lux wraps in `WrappedFunction`.
"""
const LRPSupportedLayer = Union{
    DataflowLayer,
    Dense,
    Scale,
    LayerNorm,
    ConvLayer,
    DropoutLayer,
    NormalizationLayer,
    ReshapingLayer,
    PoolingLayer,
    NoOpLayer,
    LRPSupportedActivation,
}
