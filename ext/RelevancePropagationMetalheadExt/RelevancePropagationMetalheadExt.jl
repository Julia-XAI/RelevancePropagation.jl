module RelevancePropagationMetalheadExt
using RelevancePropagation, Flux
using RelevancePropagation:
    SelfAttentionRule, SelectClassToken, PositionalEmbeddingRule, modify_layer # all used types have to be used explicitely
import RelevancePropagation: canonize, is_compatible, lrp! # all functions to which you want to add methods have to be imported
using Metalhead:
    ViPosEmbedding, ClassTokens, MultiHeadSelfAttention, chunk, ViT, seconddimmean
using Metalhead.Layers: _flatten_spatial

include("utils.jl")
include("rules.jl")
end
