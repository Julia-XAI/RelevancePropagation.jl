module VisionTransformerExt
using RelevancePropagation, Metalhead, Flux
using RelevancePropagation:
    SelectClassToken, SelfAttentionRule, PositionalEmbeddingRule, REF_ALI_TRANSFORMER
import RelevancePropagation: prepare_vit
using Metalhead: ViPosEmbedding, ClassTokens, MultiHeadSelfAttention, chunk
using Metalhead.Layers: _flatten_spatial
using NNlib: split_heads, join_heads

include("rules.jl")
include("utils.jl")
end
