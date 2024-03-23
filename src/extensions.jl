# since package extensions should/can not define new types or functions, 
# we have to define them here and add the relevant methods in the extension

#==================================#
# RelevancePropagationMetalheadExt #
#==================================#

# layers
"""Flux layer to select the first token, for use with Metalhead.jl's vision transformer."""
struct SelectClassToken end
Flux.@functor SelectClassToken
(::SelectClassToken)(x) = x[:, 1, :]

# rules
"""
    SelfAttentionRule(value_rule=ZeroRule(), out_rule=ZeroRule)

LRP-AH rule. Used on `MultiHeadSelfAttention` layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_i^k = \\sum_j\\frac{\\alpha_{ij} a_i^k}{\\sum_l \\alpha_{lj} a_l^k} R_j^{k+1}
```
where ``alpha_{ij}`` are the attention weights.
Relevance through the value projection (before attention) and the out projection (after attention) is by default propagated using the [`ZeroRule`](@ref).

# Optional arguments
- `value_rule`: Rule for the value projection, defaults to `ZeroRule()`
- `out_rule`: Rule for the out projection, defaults to `ZeroRule()`

# References
- $REF_ALI_TRANSFORMER
"""
struct SelfAttentionRule{V,O} <: AbstractLRPRule
    value_rule::V
    out_rule::O
end

"""
    PositionalEmbeddingRule()

To be used with Metalhead.jl`s `ViPosEmbedding` layer. Treats the positional embedding like a bias term.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_i^k = \\frac{a_i^k}{a_i^k + e^i} R_i^{k+1}
```
where ``e^i`` is the learned positional embedding.
"""
struct PositionalEmbeddingRule <: AbstractLRPRule end

# utils
"""Prepare the vision transformer model for the use with `RelevancePropagation.jl`"""
function prepare_vit end
