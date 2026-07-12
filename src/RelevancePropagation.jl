module RelevancePropagation

using Reexport: @reexport
import XAIBase: call_analyzer
using XAIBase: XAIBase, AbstractXAIMethod, Explanation
using XAIBase: AbstractOutputSelector

using Lux: Lux, Chain, Parallel, SkipConnection
using Lux: Dense, Scale, Conv, ConvTranspose
using Lux: BatchNorm, LayerNorm
using Lux:
    MaxPool, MeanPool, AdaptiveMaxPool, AdaptiveMeanPool, GlobalMaxPool, GlobalMeanPool
using Lux: Dropout, AlphaDropout, VariationalHiddenDropout
using Lux: FlattenLayer, ReshapeLayer, NoOpLayer, WrappedFunction
using Lux: apply, testmode

using Enzyme: autodiff_thunk, ReverseSplitWithPrimal, ReverseSplitWidth
using Enzyme: Const, Duplicated, BatchDuplicated, make_zero

using ConstructionBase: setproperties
using NNlib: relu, gelu, swish, mish, softmax, softmax!
using Markdown: @md_str

@reexport using XAIBase

include("bibliography.jl")
include("autodiff.jl")
include("layer_types.jl")
include("layer_utils.jl")
include("utils.jl")
include("checks.jl")
include("rules.jl")
include("lrp.jl")
# Not yet ported to Lux/Enzyme (v4.0.0 rewrite, see PLAN.md):
# include("composite.jl")         # phase 4
# include("show.jl")              # phase 4
# include("composite_presets.jl") # phase 4
# include("canonize.jl")          # phase 5
# include("crp.jl")               # phase 6

export LRP

# LRP rules
export AbstractLRPRule
export LRP_CONFIG
export ZeroRule, EpsilonRule, GammaRule, WSquareRule, FlatRule

# Useful type unions
export ConvLayer, PoolingLayer, DropoutLayer, ReshapingLayer, NormalizationLayer

end # module
