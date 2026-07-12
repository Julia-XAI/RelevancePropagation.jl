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
using Functors: KeyPath
using NNlib: relu, gelu, swish, mish, softmax, softmax!
using Markdown: @md_str
using Statistics: mean, var

@reexport using XAIBase

include("bibliography.jl")
include("autodiff.jl")
include("layer_types.jl")
include("layer_utils.jl")
include("utils.jl")
include("checks.jl")
include("rules.jl")
include("composite.jl")
include("lrp.jl")
include("show.jl")
include("composite_presets.jl") # uses show.jl
# Not yet ported to Lux/Enzyme (v4.0.0 rewrite, see PLAN.md):
# include("canonize.jl")          # phase 5
# include("crp.jl")               # phase 6

export LRP

# LRP rules
export AbstractLRPRule
export LRP_CONFIG
export ZeroRule, EpsilonRule, GammaRule, WSquareRule, FlatRule
export ZBoxRule, ZPlusRule, AlphaBetaRule, GeneralizedGammaRule
export PassRule, LayerNormRule

# LRP composites
export Composite, AbstractCompositePrimitive
export LayerMap, GlobalMap, RangeMap, FirstLayerMap, LastLayerMap
export GlobalTypeMap, RangeTypeMap, FirstLayerTypeMap, LastLayerTypeMap
export FirstNTypeMap
export lrp_rules, show_layer_indices

# Default composites
export EpsilonGammaBox, EpsilonPlus, EpsilonAlpha2Beta1, EpsilonPlusFlat
export EpsilonAlpha2Beta1Flat

# Model utilities
export strip_softmax

# Useful type unions
export ConvLayer, PoolingLayer, DropoutLayer, ReshapingLayer, NormalizationLayer

end # module
