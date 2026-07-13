"""Union type for dataflow layers."""
const DataflowLayer = Union{Chain,Parallel,SkipConnection}

"""Union type for softmax activation functions."""
const SoftmaxActivation = Union{typeof(softmax),typeof(softmax!)}
