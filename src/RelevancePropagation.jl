module RelevancePropagation

using Reexport: @reexport
import XAIBase: call_analyzer
using XAIBase: XAIBase, AbstractXAIMethod, Explanation
using XAIBase: AbstractOutputSelector
using XAIBase: AbstractFeatureSelector, number_of_features

using Lux: Chain, Parallel, SkipConnection
using Lux: Dense, Scale, LayerNorm, Conv, ConvTranspose
using Lux: NoOpLayer, WrappedFunction
using LuxCore: LuxCore, apply, testmode # owned by LuxCore, not Lux (ExplicitImports)
using LuxCore: AbstractLuxWrapperLayer

using Enzyme: autodiff, autodiff_thunk, Reverse, ReverseSplitWithPrimal, Active
using Enzyme: Const, Duplicated, DuplicatedNoNeed
using Enzyme.EnzymeRules: EnzymeRules, RevConfig, AugmentedReturn
using Enzyme.EnzymeRules: needs_primal, needs_shadow, overwritten

using Functors: KeyPath
using LinearAlgebra: dot
using NNlib: relu, gelu, swish, mish
using NNlib: conv, ∇conv_data, DenseConvDims
using NNlib: maxpool, ∇maxpool, meanpool, ∇meanpool
using Markdown: @md_str
using Static: known
using Statistics: mean, var

@reexport using XAIBase

# The `ModelSurgeon` submodule owns the structural rewrites of the Lux
# `(model, ps, st)` triple. It is self-contained and included before all
# other source files so it cannot reference RP types; RP-specific policy
# (e.g. keeping pooling layers intact) is injected from outside in
# `model_surgery.jl`.
include("ModelSurgeon/ModelSurgeon.jl")
using .ModelSurgeon: ModelSurgeon
using .ModelSurgeon: DataflowLayer, ConvLayer, DropoutLayer, ReshapingLayer
using .ModelSurgeon: PoolingLayer, MaxPoolLayer, MeanPoolLayer
using .ModelSurgeon: LPPoolLayer, NormalizationLayer
using .ModelSurgeon: children_layers, map_layers, chainall
using .ModelSurgeon: activation_fn, remove_activation
using .ModelSurgeon: strip_softmax, has_output_softmax

include("bibliography.jl")
include("autodiff.jl")
include("lux_layers.jl")
include("model_surgery.jl")
include("utils.jl")
include("checks.jl")
include("rules.jl")
include("composite.jl")
include("lrp.jl")
include("show.jl")
include("composite_presets.jl") # uses show.jl
include("crp.jl")

export LRP
export CRP

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
export strip_softmax, flatten_model, canonize

# Useful type unions
export ConvLayer, PoolingLayer, DropoutLayer, ReshapingLayer, NormalizationLayer

end # module
