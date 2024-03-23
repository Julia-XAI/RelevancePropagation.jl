# attention
SelfAttentionRule() = SelfAttentionRule(ZeroRule(), ZeroRule())
LRP_CONFIG.supports_layer(::MultiHeadSelfAttention) = true
is_compatible(::SelfAttentionRule, ::MultiHeadSelfAttention) = true

function lrp!(
    Rᵏ, rule::SelfAttentionRule, mha::MultiHeadSelfAttention, _modified_layer, aᵏ, Rᵏ⁺¹
)
    # query, key, value projections
    qkv = mha.qkv_layer(aᵏ)
    q, k, v = chunk(qkv, 3; dims=1)
    Rᵥ = similar(v)
    # attention
    nheads = mha.nheads
    fdrop = mha.attn_drop
    bias = nothing
    # reshape to merge batch dimensions
    batch_size = size(q)[3:end]
    batch_size == size(k)[3:end] == size(v)[3:end] ||
        throw(ArgumentError("Batch dimensions have to be the same."))
    q, k, v = map(x -> reshape(x, size(x, 1), size(x, 2), :), (q, k, v))
    # add head dimension
    q, k, v = split_heads.((q, k, v), nheads)
    # compute attention scores
    αt = dot_product_attention_scores(q, k, bias; fdrop)
    # move head dimension to third place
    vt = permutedims(v, (1, 3, 2, 4))
    xt = vt ⊠ αt
    # remove head dimension
    x = permutedims(xt, (1, 3, 2, 4))
    x = join_heads(x)
    # restore batch dimensions
    x = reshape(x, size(x, 1), size(x, 2), batch_size...)
    Rₐ = similar(x)

    # lrp pass
    ## forward: aᵏ   ->(v_proj)      v  ->(attention) x  ->(out_proj) out
    ## lrp:     Rᵏ   <-(value_rule)  Rᵥ <-(AH-Rule)   Rₐ <-(out_rule) Rᵏ⁺¹
    ## output projection
    lrp!(
        Rₐ,
        rule.out_rule,
        mha.projection[1],
        modify_layer(rule.out_rule, mha.projection[1]),
        x,
        Rᵏ⁺¹,
    )
    ## attention
    lrp_attention!(Rᵥ, xt, αt, vt, nheads, batch_size, Rₐ)
    ## value projection
    _, _, w = chunk(mha.qkv_layer.weight, 3; dims=1)
    _, _, b = chunk(mha.qkv_layer.bias, 3; dims=1)
    proj = Dense(w, b)
    lrp!(Rᵏ, rule.value_rule, proj, modify_layer(rule.value_rule, proj), aᵏ, Rᵥ)
end

function lrp_attention!(Rᵥ, x, α, v, nheads, batch_size, Rₐ)
    # input dimensions:
    ## Rₐ: [embedding x token x batch...]
    ## x : [embedding x token x head x batch]
    ## α : [token x token x head x batch]
    ## v : [embedding x token x head x batch]
    ## Rᵥ: [embedding x token x batch...]

    # reshape Rₐ: combine batch dimensions, split heads, move head dimension
    Rₐ = permutedims(
        split_heads(reshape(Rₐ, size(Rₐ, 1), size(Rₐ, 2), :), nheads), (1, 3, 2, 4)
    )
    # compute relevance term
    s = Rₐ ./ x
    # add extra dimensions for broadcasting
    s = reshape(s, size(s, 1), size(s, 2), 1, size(s)[3:end]...)
    α = reshape(permutedims(α, (2, 1, 3, 4)), 1, size(α)...)
    # compute relevances, broadcasting over extra dimensions
    R = α .* s
    R = dropdims(sum(R; dims=2); dims=2)
    R = R .* v
    # reshape relevances (drop extra dimension, move head dimension, join heads, split batch dimensions)
    R = join_heads(permutedims(R, (1, 3, 2, 4)))
    Rᵥ .= reshape(R, size(R, 1), size(R, 2), batch_size...)
end

#=========================#
# Special ViT layers      #
#=========================#

# reshaping image -> token
LRP_CONFIG.supports_layer(::typeof(_flatten_spatial)) = true
function lrp!(Rᵏ, ::ZeroRule, ::typeof(_flatten_spatial), _modified_layer, aᵏ, Rᵏ⁺¹)
    Rᵏ .= reshape(permutedims(Rᵏ⁺¹, (2, 1, 3)), size(Rᵏ)...)
end

# ClassToken layer: adds a Class Token; we ignore this token for the relevances
LRP_CONFIG.supports_layer(::ClassTokens) = true
function lrp!(Rᵏ, ::ZeroRule, ::ClassTokens, _modified_layer, aᵏ, Rᵏ⁺¹)
    Rᵏ .= Rᵏ⁺¹[:, 2:end, :]
end

# Positional Embedding (you can also use the PassRule)
LRP_CONFIG.supports_layer(::ViPosEmbedding) = true
is_compatible(::PositionalEmbeddingRule, ::ViPosEmbedding) = true
function lrp!(
    Rᵏ, ::PositionalEmbeddingRule, layer::ViPosEmbedding, _modified_layer, aᵏ, Rᵏ⁺¹
)
    Rᵏ .= aᵏ ./ layer(aᵏ) .* Rᵏ⁺¹
end

# class token selection: only the class token is used for the final predictions, 
# so it gets all the relevance
LRP_CONFIG.supports_layer(::SelectClassToken) = true
function lrp!(Rᵏ, ::ZeroRule, ::SelectClassToken, _modified_layer, aᵏ, Rᵏ⁺¹)
    fill!(Rᵏ, zero(eltype(Rᵏ)))
    Rᵏ[:, 1, :] .= Rᵏ⁺¹
end
