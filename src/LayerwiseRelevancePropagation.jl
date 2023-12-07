module LayerwiseRelevancePropagation

using Reexport
@reexport using XAIBase

using Base.Iterators
using MacroTools: @forward
using Flux
using Zygote
using Markdown

include("bibliography.jl")
include("flux_types.jl")
include("flux_layer_utils.jl")
include("flux_chain_utils.jl")
include("utils.jl")
include("canonize.jl")
include("checks.jl")
include("rules.jl")
include("composite.jl")
include("analyzer.jl")
include("show.jl")
include("composite_presets.jl") # uses show.jl
include("crp.jl")

export LRP

# LRP rules
export AbstractLRPRule
export LRP_CONFIG
export ZeroRule, EpsilonRule, GammaRule, WSquareRule, FlatRule
export PassRule, ZBoxRule, ZPlusRule, AlphaBetaRule, GeneralizedGammaRule

# LRP composites
export Composite, AbstractCompositePrimitive
export ChainTuple, ParallelTuple
export LayerMap, GlobalMap, RangeMap, FirstLayerMap, LastLayerMap
export GlobalTypeMap, RangeTypeMap, FirstLayerTypeMap, LastLayerTypeMap
export FirstNTypeMap
export lrp_rules, show_layer_indices

# Default composites
export EpsilonGammaBox, EpsilonPlus, EpsilonAlpha2Beta1, EpsilonPlusFlat
export EpsilonAlpha2Beta1Flat
# Useful type unions
export ConvLayer, PoolingLayer, DropoutLayer, ReshapingLayer, NormalizationLayer

# CRP
export CRP, TopNConcepts, IndexedConcepts

# utils
export strip_softmax, flatten_model, canonize

end # module
