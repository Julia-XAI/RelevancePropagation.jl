module VisionTransformerExt
using RelevancePropagation, Flux
using RelevancePropagation: SelfAttentionRule, SelectClassToken, PositionalEmbeddingRule, modify_layer # all used types have to be used explicitely
import RelevancePropagation: prepare_vit, is_compatible, lrp! # all functions to which you want to add methods have to be imported
using Metalhead: ViPosEmbedding, ClassTokens, MultiHeadSelfAttention, chunk, ViT, seconddimmean
using Metalhead.Layers: _flatten_spatial
using NNlib: split_heads, join_heads

include("rules.jl")
include("utils.jl")
end
