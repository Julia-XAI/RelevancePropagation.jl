module VisionTransformerExt
    using RelevancePropagation, Metalhead, Flux
    using Metalhead: ViPosEmbedding, ClassTokens, _flatten_spatial, MultiHeadSelfAttention
    using NNlib: split_heads, join_heads

    include("layers.jl")
    include("rules.jl")
    include("utils.jl")

    export prepare_vit
end