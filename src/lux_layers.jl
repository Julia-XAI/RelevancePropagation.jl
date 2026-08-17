# Generic Lux layer-type unions (`ConvLayer`, `PoolingLayer`, ...) live in the
# `ModelSurgeon` submodule and are imported in `RelevancePropagation.jl`.
# This file holds the LRP-specific policy unions built on top of them.

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

# Introspection on Lux parameter NamedTuples, consumed by the LRP rules
# (see `rules.jl`).
has_weight(ps) = haskey(ps, :weight)
has_bias(ps) = haskey(ps, :bias)
