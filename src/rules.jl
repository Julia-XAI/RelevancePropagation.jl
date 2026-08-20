# https://julia-xai.github.io/RelevancePropagation.jl/dev/lrp/developer/
"""
    AbstractLRPRule

Abstract supertype of all LRP rules.
Custom rules must subtype `AbstractLRPRule` and can customize their behavior by
extending [`modify_input`](@ref), [`modify_denominator`](@ref),
[`modify_parameters`](@ref) and [`is_compatible`](@ref),
or by implementing a custom [`propagate`](@ref) method.
"""
abstract type AbstractLRPRule end

# Default parameters
const LRP_DEFAULT_GAMMA = 0.25f0
const LRP_DEFAULT_EPSILON = 1.0f-6
const LRP_DEFAULT_STABILIZER = 1.0f-9
const LRP_DEFAULT_ALPHA = 2.0f0
const LRP_DEFAULT_BETA = 1.0f0

"""
    propagate(rule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)

Propagate the relevance `Rᵏ⁺¹` at the output of a layer
to the relevance `Rᵏ` at its input, according to the LRP rule.
This is the redefined VJP the Enzyme reverse pass calls
for each rule-carrying node ([`lrp_node`](@ref)),
and it is a pure function: `Rᵏ` is returned.

# Arguments
- `rule`: LRP rule to apply.
- `layer`: The Lux layer relevance is propagated through.
  The wrap-time activation split guarantees this layer carries no activation
  function: activation-bearing leaves are split into an affine node and a
  `PassRule` activation node when the model is wrapped.
- `aᵏ`: Layer input activation.
- `zᵏ`: Layer output, cached on the Enzyme tape. For split layers this is the
  pre-activation of the original fused layer.
- `ps`, `st`: Unmodified layer parameters and states.
- `Rᵏ⁺¹`: Relevance at the layer output.
"""
function propagate(rule::AbstractLRPRule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    ãᵏ = modify_input(rule, aᵏ)
    ρps = modify_params(rule, ps)
    # If neither input nor parameters are modified, the cached pre-activation
    # is the rule's z̃ and only a one-shot VJP is needed.
    if ρps === ps && ãᵏ === aᵏ
        s = Rᵏ⁺¹ ./ modify_denominator(rule, zᵏ)
        c = input_vjp(layer, ãᵏ, ρps, st, s)
        return ãᵏ .* c
    end
    # Otherwise the modified forward pass and its VJP share one prepared
    # pullback, so the fallback's reverse pass reuses the tape of the
    # forward pass that computed z̃ — one modified forward instead of two.
    z̃, pullback = prepare_vjp(layer, ãᵏ, ρps, st)
    s = Rᵏ⁺¹ ./ modify_denominator(rule, z̃)
    return ãᵏ .* pullback(s)
end

#===================================#
# Functions used to implement rules #
#===================================#

# The functions that follow define the default fallbacks used by LRP rules
# when calling the generic `propagate` implementation above.
# Rule types are used to dispatch on rule-specific implementations.

# To implement a new rule, extend the following functions for your rule type:
# - modify_input
# - modify_denominator
# - modify_parameters OR (modify_weight and modify_bias)
# - is_compatible

const LRP_PARAMETER_MODIFICATION_DIAGRAM = """
Rules modify the `weight` and `bias` entries of a layer's `ps` NamedTuple
via `modify_weight` and `modify_bias`, which by default both call
`modify_parameters`:
```
┌─────────────────────────────────────────┐
│              modify_params              │
└─────────┬─────────────────────┬─────────┘
          │ calls               │ calls
┌─────────▼─────────┐ ┌─────────▼─────────┐
│   modify_weight   │ │    modify_bias    │
└─────────┬─────────┘ └─────────┬─────────┘
          │ calls               │ calls
┌─────────▼─────────┐ ┌─────────▼─────────┐
│ modify_parameters │ │ modify_parameters │
└───────────────────┘ └───────────────────┘
```
"""

"""
    modify_input(rule, input)

Modify input activation before computing relevance propagation.
"""
modify_input(rule, input) = input

"""
    modify_denominator(rule, d)

Modify denominator ``z`` for numerical stability on the forward pass.
"""
modify_denominator(rule, d) = stabilize_denom(d, LRP_DEFAULT_STABILIZER)

"""
    is_compatible(rule, layer, ps)

Check compatibility of an LRP rule with a Lux `layer` and its parameters `ps`.
By default, a rule is compatible with layers that have a `weight` parameter.
"""
is_compatible(rule, layer, ps) = has_weight(ps)

struct LRPCompatibilityError <: Exception
    rule::String
    layer::String
    LRPCompatibilityError(rule, layer) = new("$rule", "$layer")
end
function Base.showerror(io::IO, e::LRPCompatibilityError)
    return print(io, "LRP rule ", e.rule, " isn't compatible with layer ", e.layer)
end

"""
    modify_parameters(rule, parameter)

Modify parameters before computing the relevance.

## Note
$LRP_PARAMETER_MODIFICATION_DIAGRAM
"""
modify_parameters(rule, param) = param

"""
    modify_weight(rule, weight)

Modify layer weights before computing the relevance.

## Note
$LRP_PARAMETER_MODIFICATION_DIAGRAM
"""
modify_weight(rule, w) = modify_parameters(rule, w)

"""
    modify_bias(rule, bias)

Modify layer bias before computing the relevance.

## Note
$LRP_PARAMETER_MODIFICATION_DIAGRAM
"""
modify_bias(rule, b) = modify_parameters(rule, b)

"""
    modify_params(rule, ps; keep_bias=true)

Return the parameter NamedTuple `ps` with `weight` and `bias` modified via
[`modify_weight`](@ref) and [`modify_bias`](@ref).
Setting `keep_bias=false` zeroes the bias instead.

Parameters without a `weight` entry are returned unchanged. If no
modification is applied, `ps` itself is returned (`===`), which signals to
[`propagate`](@ref) that the cached pre-activation can be reused.

Modified parameters are computed lazily on each call: rules only hold their
hyperparameters, no copies of model parameters.

## Note
$LRP_PARAMETER_MODIFICATION_DIAGRAM
"""
function modify_params(rule, ps; keep_bias::Bool=true)
    has_weight(ps) || return ps
    weight = modify_weight(rule, ps.weight)
    if !has_bias(ps)
        weight === ps.weight && return ps
        return merge(ps, (; weight))
    end
    bias = keep_bias ? modify_bias(rule, ps.bias) : zero(ps.bias)
    weight === ps.weight && bias === ps.bias && return ps
    return merge(ps, (; weight, bias))
end

# Useful presets, used e.g. in AlphaBetaRule, ZBoxRule & ZPlusRule:
modify_parameters(::Val{:keep_positive}, p) = keep_positive(p)
modify_parameters(::Val{:keep_negative}, p) = keep_negative(p)

#===========#
# LRP Rules #
#===========#

# The following LRP rules use the generic `propagate` implementation at the
# top of this file.

"""
    ZeroRule()

LRP-``0`` rule. Commonly used on upper layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i \\frac{W_{ij}a_j^k}{\\sum_l W_{il}a_l^k+b_i} R_i^{k+1}
```

# References
- $REF_BACH_LRP
"""
struct ZeroRule <: AbstractLRPRule end
is_compatible(::ZeroRule, layer, ps) = true # compatible with all layer types

"""
    EpsilonRule([epsilon=$(LRP_DEFAULT_EPSILON)])

LRP-``ϵ`` rule. Commonly used on middle layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{W_{ij}a_j^k}{\\epsilon +\\sum_{l}W_{il}a_l^k+b_i} R_i^{k+1}
```

# Optional arguments
- `epsilon`: Optional stabilization parameter, defaults to `$(LRP_DEFAULT_EPSILON)`.

# References
- $REF_BACH_LRP
"""
struct EpsilonRule{T<:Real} <: AbstractLRPRule
    ϵ::T
    EpsilonRule(epsilon=LRP_DEFAULT_EPSILON) = new{eltype(epsilon)}(epsilon)
end
modify_denominator(r::EpsilonRule, d) = stabilize_denom(d, r.ϵ)
is_compatible(::EpsilonRule, layer, ps) = true # compatible with all layer types

"""
    GammaRule([gamma=$(LRP_DEFAULT_GAMMA)])

LRP-``γ`` rule. Commonly used on lower layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{(W_{ij}+\\gamma W_{ij}^+)a_j^k}
    {\\sum_l(W_{il}+\\gamma W_{il}^+)a_l^k+(b_i+\\gamma b_i^+)} R_i^{k+1}
```

# Optional arguments
- `gamma`: Optional multiplier for added positive weights, defaults to `$(LRP_DEFAULT_GAMMA)`.

# References
- $REF_MONTAVON_OVERVIEW
"""
struct GammaRule{T<:Real} <: AbstractLRPRule
    γ::T
    GammaRule(gamma=LRP_DEFAULT_GAMMA) = new{eltype(gamma)}(gamma)
end
function modify_parameters(r::GammaRule, param::AbstractArray)
    γ = convert(eltype(param), r.γ)
    return @. param + γ * keep_positive(param)
end

# Internally used for GeneralizedGammaRule:
struct NegativeGammaRule{T<:Real} <: AbstractLRPRule
    γ::T
    NegativeGammaRule(gamma=LRP_DEFAULT_GAMMA) = new{eltype(gamma)}(gamma)
end
function modify_parameters(r::NegativeGammaRule, param::AbstractArray)
    γ = convert(eltype(param), r.γ)
    return @. param + γ * keep_negative(param)
end

"""
    WSquareRule()

LRP-``w²`` rule. Commonly used on the first layer when values are unbounded.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{W_{ij}^2}{\\sum_l W_{il}^2} R_i^{k+1}
```

# References
- $REF_MONTAVON_DTD
"""
struct WSquareRule <: AbstractLRPRule end
modify_input(::WSquareRule, input) = ones_like(input)
modify_weight(::WSquareRule, w) = w .^ 2
modify_bias(::WSquareRule, b) = zero(b)

"""
    FlatRule()

LRP-Flat rule. Similar to the [`WSquareRule`](@ref), but with all weights set to one
and all bias terms set to zero.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{1}{\\sum_l 1} R_i^{k+1} = \\sum_i\\frac{1}{n_i} R_i^{k+1}
```
where ``n_i`` is the number of input neurons connected to the output neuron at index ``i``.

# References
- $REF_LAPUSCHKIN_CLEVER_HANS
"""
struct FlatRule <: AbstractLRPRule end
modify_input(::FlatRule, input) = ones_like(input)
modify_weight(::FlatRule, w) = ones_like(w)
modify_bias(::FlatRule, b) = zero(b)

#===================#
# Complex LRP Rules #
#===================#

# The following rules use custom `propagate` implementations.
# Rules that require several modified parameter variants construct them
# lazily inside their `propagate` body via `modify_params`.

"""
    PassRule()

Pass-through rule. Passes relevance through to the lower layer.

Supports layers with constant input and output shapes, e.g. reshaping layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = R_j^{k+1}
```
"""
struct PassRule <: AbstractLRPRule end
function propagate(::PassRule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    return reshape_relevance(aᵏ, Rᵏ⁺¹)
end
is_compatible(::PassRule, layer, ps) = true

reshape_relevance(aᵏ, Rᵏ⁺¹) = reshape(Rᵏ⁺¹, size(aᵏ))

"""
    ZBoxRule(low, high)

LRP-``zᴮ``-rule. Commonly used on the first layer for pixel input.

The parameters `low` and `high` should be set to the lower and upper bounds
of the input features, e.g. `0.0` and `1.0` for raw image data.
It is also possible to provide two arrays of that match the input size.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k=\\sum_i \\frac{W_{ij}a_j^k - W_{ij}^{+}l_j - W_{ij}^{-}h_j}
    {\\sum_l W_{il}a_l^k+b_i - \\left(W_{il}^{+}l_l+b_i^{+}\\right) - \\left(W_{il}^{-}h_l+b_i^{-}\\right)} R_i^{k+1}
```

All three terms propagate through the affine part of the layer,
matching the formula above.
This is an intentional divergence from v3,
which routed the ``z``- and ``c``-terms
through the layer's activation function
(see the changelog for version 4.0).

# References
- $REF_MONTAVON_OVERVIEW
"""
struct ZBoxRule{T} <: AbstractLRPRule
    low::T
    high::T
end

function propagate(rule::ZBoxRule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    l = zbox_input(aᵏ, rule.low)
    h = zbox_input(aᵏ, rule.high)
    ps⁺ = modify_params(Val(:keep_positive), ps)
    ps⁻ = modify_params(Val(:keep_negative), ps)

    z⁺, pullback⁺ = prepare_vjp(layer, l, ps⁺, st)
    z⁻, pullback⁻ = prepare_vjp(layer, h, ps⁻, st)

    s = Rᵏ⁺¹ ./ modify_denominator(rule, zᵏ - z⁺ - z⁻)
    c = input_vjp(layer, aᵏ, ps, st, s) # z arrives as the cached zᵏ
    c⁺ = pullback⁺(s)
    c⁻ = pullback⁻(s)
    return @. aᵏ * c - l * c⁺ - h * c⁻
end

zbox_input(in::AbstractArray{T}, c::Real) where {T} = fill!(similar(in), convert(T, c))
function zbox_input(in::AbstractArray{T}, A::AbstractArray) where {T}
    @assert size(A) == size(in)
    return copyto!(similar(in, T), A)
end

"""
    ZPlusRule()

LRP-``z⁺`` rule. Commonly used on lower layers.

Equivalent to `AlphaBetaRule(1.0f0, 0.0f0)`, but slightly faster.
See also [`AlphaBetaRule`](@ref).

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac{\\left(W_{ij}a_j^k\\right)^+}{\\sum_l\\left(W_{il}a_l^k+b_i\\right)^+} R_i^{k+1}
```

# References
- $REF_BACH_LRP
- $REF_MONTAVON_DTD
"""
struct ZPlusRule <: AbstractLRPRule end

function propagate(rule::ZPlusRule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    aᵏ⁺ = keep_positive(aᵏ)
    aᵏ⁻ = keep_negative(aᵏ)
    ps⁺ = modify_params(Val(:keep_positive), ps)
    ps⁻ = modify_params(Val(:keep_negative), ps; keep_bias=false)

    # The seed needs both primals, so both forwards are prepared first and
    # both reverses run after the seed is known.
    z⁺, pullback⁺ = prepare_vjp(layer, aᵏ⁺, ps⁺, st)
    z⁻, pullback⁻ = prepare_vjp(layer, aᵏ⁻, ps⁻, st)

    s = Rᵏ⁺¹ ./ modify_denominator(rule, z⁺ + z⁻)
    c⁺ = pullback⁺(s)
    c⁻ = pullback⁻(s)
    return @. aᵏ⁺ * c⁺ + aᵏ⁻ * c⁻
end

"""
    AlphaBetaRule([alpha=$(LRP_DEFAULT_ALPHA), beta=$(LRP_DEFAULT_BETA)])

LRP-``αβ`` rule. Weights positive and negative contributions according to the
parameters `alpha` and `beta` respectively. The difference ``α-β`` must be equal to one.
Commonly used on lower layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\left(
    \\alpha\\frac{\\left(W_{ij}a_j^k\\right)^+}{\\sum_l\\left(W_{il}a_l^k+b_i\\right)^+}
    -\\beta\\frac{\\left(W_{ij}a_j^k\\right)^-}{\\sum_l\\left(W_{il}a_l^k+b_i\\right)^-}
\\right) R_i^{k+1}
```

# Optional arguments
- `alpha`: Multiplier for the positive output term, defaults to `$(LRP_DEFAULT_ALPHA)`.
- `beta`: Multiplier for the negative output term, defaults to `$(LRP_DEFAULT_BETA)`.

# References
- $REF_BACH_LRP
- $REF_MONTAVON_OVERVIEW
"""
struct AlphaBetaRule{T<:Real} <: AbstractLRPRule
    α::T
    β::T
    function AlphaBetaRule(alpha=LRP_DEFAULT_ALPHA, beta=LRP_DEFAULT_BETA)
        alpha < 0 && throw(ArgumentError("Parameter `alpha` must be ≥0."))
        beta < 0 && throw(ArgumentError("Parameter `beta` must be ≥0."))
        !isone(alpha - beta) && throw(ArgumentError("`alpha - beta` must be equal one."))
        return new{eltype(alpha)}(alpha, beta)
    end
end

function propagate(rule::AlphaBetaRule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    aᵏ⁺ = keep_positive(aᵏ)
    aᵏ⁻ = keep_negative(aᵏ)
    # ᵅ/ᵝ: parameter variants of the positive and negative term. The α- and
    # β-variants share weights and only differ in their biases, so the
    # β-VJPs are computed through the α-variants.
    psᵅ⁺ = modify_params(Val(:keep_positive), ps)
    psᵅ⁻ = modify_params(Val(:keep_negative), ps; keep_bias=false)
    psᵝ⁻ = modify_params(Val(:keep_negative), ps)
    psᵝ⁺ = modify_params(Val(:keep_positive), ps; keep_bias=false)

    # The α-variant forwards are prepared so their VJPs reuse the tapes;
    # the crossed bias variants only enter the β-denominator, two plain
    # forwards. The second seed sᵝ shares the α-points' two-seed pullbacks
    # (one width-2 batched reverse per tape on the Enzyme fallback),
    # since the α- and β-variants agree there: Wᵝ⁺ = Wᵅ⁺ and Wᵝ⁻ = Wᵅ⁻.
    zᵅ⁺, pullbackᵅ⁺ = prepare_vjp2(layer, aᵏ⁺, psᵅ⁺, st)
    zᵅ⁻, pullbackᵅ⁻ = prepare_vjp2(layer, aᵏ⁻, psᵅ⁻, st)
    zᵝ⁺ = first(apply(layer, aᵏ⁻, psᵝ⁺, st))
    zᵝ⁻ = first(apply(layer, aᵏ⁺, psᵝ⁻, st))

    sᵅ = Rᵏ⁺¹ ./ modify_denominator(rule, zᵅ⁺ + zᵅ⁻)
    sᵝ = Rᵏ⁺¹ ./ modify_denominator(rule, zᵝ⁺ + zᵝ⁻)
    cᵅ⁺, cᵝ⁺ = pullbackᵅ⁺(sᵅ, sᵝ)
    cᵅ⁻, cᵝ⁻ = pullbackᵅ⁻(sᵅ, sᵝ)

    T = eltype(aᵏ)
    α = convert(T, rule.α)
    β = convert(T, rule.β)
    return @. α * (aᵏ⁺ * cᵅ⁺ + aᵏ⁻ * cᵅ⁻) - β * (aᵏ⁺ * cᵝ⁻ + aᵏ⁻ * cᵝ⁺)
end

"""
    GeneralizedGammaRule([gamma=$(LRP_DEFAULT_GAMMA)])

Generalized LRP-``γ`` rule. Can be used on layers with `leakyrelu` activation functions.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_j^k = \\sum_i\\frac
    {(W_{ij}+\\gamma W_{ij}^+)a_j^+ +(W_{ij}+\\gamma W_{ij}^-)a_j^-}
    {\\sum_l(W_{il}+\\gamma W_{il}^+)a_j^+ +(W_{il}+\\gamma W_{il}^-)a_j^- +(b_i+\\gamma b_i^+)}
I(z_k>0) \\cdot R^{k+1}_i
+\\sum_i\\frac
    {(W_{ij}+\\gamma W_{ij}^-)a_j^+ +(W_{ij}+\\gamma W_{ij}^+)a_j^-}
    {\\sum_l(W_{il}+\\gamma W_{il}^-)a_j^+ +(W_{il}+\\gamma W_{il}^+)a_j^- +(b_i+\\gamma b_i^-)}
I(z_k<0) \\cdot R^{k+1}_i
```

# Optional arguments
- `gamma`: Optional multiplier for added positive weights, defaults to `$(LRP_DEFAULT_GAMMA)`.

# References
- $REF_ANDEOL_DOMAIN_INVARIANT
"""
struct GeneralizedGammaRule{T<:Real} <: AbstractLRPRule
    γ::T
    GeneralizedGammaRule(gamma=LRP_DEFAULT_GAMMA) = new{eltype(gamma)}(gamma)
end

function propagate(rule::GeneralizedGammaRule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    aᵏ⁺ = keep_positive(aᵏ)
    aᵏ⁻ = keep_negative(aᵏ)
    # ˡ/ʳ: LHS/RHS of the generalized Gamma-rule equation. The ˡ/ʳ-variants
    # share weights, so the ʳ-VJPs are computed through the ˡ-variants.
    rule⁺ = GammaRule(rule.γ)
    rule⁻ = NegativeGammaRule(rule.γ)
    psˡ⁺ = modify_params(rule⁺, ps)
    psˡ⁻ = modify_params(rule⁻, ps; keep_bias=false)
    psʳ⁻ = modify_params(rule⁻, ps)
    psʳ⁺ = modify_params(rule⁺, ps; keep_bias=false)

    # The ˡ-variant forwards are prepared so their VJPs reuse the tapes;
    # the crossed bias variants only enter the ʳ-denominator, two plain
    # forwards. The second seed sʳ shares the ˡ-points' two-seed pullbacks
    # (one width-2 batched reverse per tape on the Enzyme fallback),
    # since the ˡ- and ʳ-variants agree there: Wʳ⁺ = Wˡ⁺ and Wʳ⁻ = Wˡ⁻.
    zˡ⁺, pullbackˡ⁺ = prepare_vjp2(layer, aᵏ⁺, psˡ⁺, st)
    zˡ⁻, pullbackˡ⁻ = prepare_vjp2(layer, aᵏ⁻, psˡ⁻, st)
    zʳ⁺ = first(apply(layer, aᵏ⁻, psʳ⁺, st))
    zʳ⁻ = first(apply(layer, aᵏ⁺, psʳ⁻, st))

    # The indicator masks read the cached pre-activation. The rule targets
    # `leakyrelu` layers, whose activation preserves sign, so the masks agree
    # with v3's masks on the layer output.
    sˡ = masked_copy(Rᵏ⁺¹, zᵏ .> 0) ./ modify_denominator(rule, zˡ⁺ + zˡ⁻)
    sʳ = masked_copy(Rᵏ⁺¹, zᵏ .< 0) ./ modify_denominator(rule, zʳ⁺ + zʳ⁻)
    cˡ⁺, cʳ⁺ = pullbackˡ⁺(sˡ, sʳ)
    cˡ⁻, cʳ⁻ = pullbackˡ⁻(sˡ, sʳ)
    return @. aᵏ⁺ * (cˡ⁺ + cʳ⁻) + aᵏ⁻ * (cˡ⁻ + cʳ⁺)
end

"""
    LayerNormRule()

LRP-LN rule. Used on `LayerNorm` layers.

# Definition
Propagates relevance ``R^{k+1}`` at layer output to ``R^k`` at layer input according to
```math
R_i^k = \\sum_j\\frac{a_i^k\\left(\\delta_{ij} - 1/N\\right)}{\\sum_l a_l^k\\left(\\delta_{lj}-1/N\\right)} R_j^{k+1}
```
Relevance through the affine transformation is by default propagated using the [`ZeroRule`](@ref).

If you would like to assign a special rule to the affine transformation inside of the `LayerNorm` layer,
call `canonize` on your model.
This will split the `LayerNorm` layer into
1. a `LayerNorm` layer without affine transformation
2. a `Scale` layer implementing the affine transformation
You can then assign separate rules to these two layers.

## Note
Statistics are computed over the dimensions the layer normalizes over:
Lux's default `dims=Colon()` normalizes over all dimensions, including the batch
dimension. Set `dims=1:length(shape)` on the `LayerNorm` layer for the per-sample
normalization v3 (Flux) applied.

# References
- $REF_ALI_TRANSFORMER
"""
struct LayerNormRule <: AbstractLRPRule end
is_compatible(::LayerNormRule, ::LayerNorm, ps) = true

function propagate(::LayerNormRule, layer::LayerNorm, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    dims = layer.dims # Colon() means statistics over all dimensions
    μₐ = mean(aᵏ; dims=dims)
    z = aᵏ .- μₐ

    R = if has_bias(ps) # affine LayerNorm: Lux stores its parameters as ps.scale/ps.bias
        # Forward pass through the normalization part, matching Lux's formula
        # activation.(scale .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ bias):
        ϵ = convert(float(eltype(aᵏ)), layer.epsilon)
        σ² = var(aᵏ; dims=dims, mean=μₐ, corrected=false)
        aᵏₙ = @. z / sqrt(σ² + ϵ)
        # Propagate through the affine part with the ZeroRule as a fallback
        # when the model is not canonized. The wrap-time split guarantees the
        # layer carries no activation, so the Scale layer is affine.
        scale = Scale(layer.shape)
        ps_scale = (; weight=ps.scale, bias=ps.bias)
        zₙ = first(apply(scale, aᵏₙ, ps_scale, NamedTuple()))
        propagate(ZeroRule(), scale, aᵏₙ, zₙ, ps_scale, NamedTuple(), Rᵏ⁺¹)
    else
        Rᵏ⁺¹
    end
    # LRP pass through the normalization
    s = @. R / stabilize_denom(z, LRP_DEFAULT_STABILIZER)
    μₛ = mean(s; dims=dims)
    return @. aᵏ * (s - μₛ)
end

#==========================#
# Performance improvements #
#==========================#

# The following methods aren't strictly necessary – tests still pass when removing them.
# However they improve performance on specific combinations of rule and layer types.

# Rules that don't require layer information:
for R in (ZeroRule, EpsilonRule)
    for L in (DropoutLayer, ReshapingLayer)
        @eval function propagate(_rule::$R, _layer::$L, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
            return reshape_relevance(aᵏ, Rᵏ⁺¹)
        end
    end
end

function propagate(_rule::FlatRule, _layer::Dense, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    n = size(aᵏ, 1) # number of input neurons connected to each output neuron
    return similar(aᵏ) .= sum(Rᵏ⁺¹; dims=1) ./ n
end
