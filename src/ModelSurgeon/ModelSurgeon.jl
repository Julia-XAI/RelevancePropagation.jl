"""
    ModelSurgeon

Structural rewrites of the Lux `(model, ps, st)` triple:

- [`map_triple`](@ref): joint traversal of model, parameters and states
- [`flatten_model`](@ref): splice nested `Chain`s into their parents
- [`canonize`](@ref): canonize models by splitting ([`canonize_split`](@ref))
  and fusing ([`canonize_fuse`](@ref)) layers
- [`strip_softmax`](@ref): remove softmax activations on the model output

The module is self-contained and holds no automatic differentiation machinery:
it owns model *structure*; consumers own what happens to the rewritten triple.
Policies are injected as function arguments (`exclude`, `unwrap`, the
[`is_fuseable`](@ref)/[`canonize_fuse`](@ref) pair and
[`split_activation`](@ref)) instead of overloadable traits on foreign types,
keeping consumers piracy-free.
"""
module ModelSurgeon

using Lux: Chain, Parallel, SkipConnection
using Lux: Dense, Scale, Conv, ConvTranspose, BatchNorm, LayerNorm
using Lux:
    MaxPool, MeanPool, AdaptiveMaxPool, AdaptiveMeanPool, GlobalMaxPool, GlobalMeanPool
using Lux: LPPool, AdaptiveLPPool, GlobalLPPool
using Lux: Dropout, AlphaDropout, VariationalHiddenDropout
using Lux: FlattenLayer, ReshapeLayer, NoOpLayer, WrappedFunction
using LuxCore: AbstractLuxWrapperLayer
using ConstructionBase: setproperties
using Functors: Functors, KeyPath, fmap_with_path
using NNlib: softmax, softmax!
using Static: static

export DataflowLayer, ConvLayer, DropoutLayer, ReshapingLayer
export MaxPoolLayer, MeanPoolLayer, PoolingLayer, LPPoolLayer, NormalizationLayer
export SoftmaxActivation
export activation_fn, remove_activation
export children_layers, map_layers, chainall, first_element, last_element
export map_triple
export flatten_model
export canonize, canonize_split, canonize_fuse, is_fuseable, split_activation
export strip_softmax, has_output_softmax

include("lux_layers.jl")
include("traverse.jl")
include("flatten.jl")
include("canonize.jl")
include("softmax.jl")

end # module
