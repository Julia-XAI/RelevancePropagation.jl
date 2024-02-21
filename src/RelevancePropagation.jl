module RelevancePropagation

using Reexport
@reexport using XAIBase

using XAIBase: AbstractFeatureSelector, number_of_features
using Base.Iterators
using MacroTools: @forward
using Flux
using Flux: Scale, normalise
using Zygote
using Markdown
using Statistics: mean, std

include("compat.jl")
include("bibliography.jl")
include("layer_types.jl")
include("layer_utils.jl")
include("chain_utils.jl")
include("modelindex.jl")
include("utils.jl")
include("canonize.jl")
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
export PassRule, ZBoxRule, ZPlusRule, AlphaBetaRule, GeneralizedGammaRule
export LayerNormRule

# LRP composites
export Composite, AbstractCompositePrimitive
export ChainTuple, ParallelTuple, SkipConnectionTuple
export LayerMap, GlobalMap, RangeMap, FirstLayerMap, LastLayerMap
export GlobalTypeMap, RangeTypeMap, FirstLayerTypeMap, LastLayerTypeMap
export FirstNTypeMap
export lrp_rules, show_layer_indices

# Default composites
export EpsilonGammaBox, EpsilonPlus, EpsilonAlpha2Beta1, EpsilonPlusFlat
export EpsilonAlpha2Beta1Flat
# Useful type unions
export ConvLayer, PoolingLayer, DropoutLayer, ReshapingLayer, NormalizationLayer

# utils
export strip_softmax, flatten_model, canonize

end # module
