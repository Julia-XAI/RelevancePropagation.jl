using RelevancePropagation
using Test
using ReferenceTests

using RelevancePropagation: propagate, modify_params
using RelevancePropagation: modify_input, modify_denominator
using RelevancePropagation: is_compatible, modify_weight, modify_bias
using RelevancePropagation: modify_parameters, NegativeGammaRule
using RelevancePropagation: activation_fn, remove_activation
using RelevancePropagation: wrap_rules, LayerWithRule, SplitActivationNode
using RelevancePropagation: stabilize_denom
using Lux
using LuxCore: LuxCore
using LinearAlgebra: I
using Random: randn
using StableRNGs: StableRNG

# Fixed pseudo-random numbers
T = Float32
pseudorandn(dims...) = randn(StableRNG(123), T, dims...)

forward(layer, ps, st, x) = first(LuxCore.apply(layer, x, ps, st))
setup_testmode(layer) = (l=Lux.setup(StableRNG(123), layer); (l[1], Lux.testmode(l[2])))

const RULES = Dict(
    "ZeroRule"             => ZeroRule(),
    "EpsilonRule"          => EpsilonRule(),
    "GammaRule"            => GammaRule(),
    "ZBoxRule"             => ZBoxRule(0.0f0, 1.0f0),
    "AlphaBetaRule"        => AlphaBetaRule(2.0f0, 1.0f0),
    "WSquareRule"          => WSquareRule(),
    "FlatRule"             => FlatRule(),
    "ZPlusRule"            => ZPlusRule(),
    "GeneralizedGammaRule" => GeneralizedGammaRule(),
    "LayerNormRule"        => LayerNormRule(),
)

## Hand-written tests
@testset "ZeroRule analytic" begin
    rule = ZeroRule()

    ## Simple dense layer
    Rᵏ⁺¹ = reshape([1 / 3 2 / 3], 2, 1)
    aᵏ = reshape([1.0 2.0], 2, 1)
    W = [3.0 4.0; 5.0 6.0]
    b = [7.0, 8.0]
    Rᵏ = reshape([17 / 90, 316 / 675], 2, 1) # expected output

    layer = Dense(2 => 2, relu)
    ps = (; weight=W, bias=b)
    st = NamedTuple()
    # The wrap-time split assigns the rule to the activation-stripped layer
    node = wrap_rules(layer, rule)
    @test node isa SplitActivationNode
    affine = node.affine.layer
    @test activation_fn(affine) == identity
    @test modify_params(rule, ps) === ps # ZeroRule doesn't modify parameters

    zᵏ = forward(affine, ps, st, aᵏ)
    R̂ᵏ = @inferred propagate(rule, affine, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ

    ## Pooling layer
    Rᵏ⁺¹ = Float32.([1 2; 3 4]//30)
    aᵏ = Float32.([1 2 3; 10 5 6; 7 8 9])
    Rᵏ = Float32.([0 0 0; 4 0 2; 0 0 4]//30) # expected output

    # Repeat in color channel dim and add batch dim
    Rᵏ⁺¹ = reshape(repeat(Rᵏ⁺¹, 1, 3), 2, 2, 3, 1)
    aᵏ = reshape(repeat(aᵏ, 1, 3), 3, 3, 3, 1)
    Rᵏ = reshape(repeat(Rᵏ, 1, 3), 3, 3, 3, 1)

    layer = MaxPool((2, 2); stride=(1, 1))
    ps, st = setup_testmode(layer)
    # Activation-free layers are wrapped without a split
    @test wrap_rules(layer, rule) isa LayerWithRule

    zᵏ = forward(layer, ps, st, aᵏ)
    R̂ᵏ = @inferred propagate(rule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ

    ## Scale layer
    Rᵏ⁺¹ = reshape([1 / 3 2 / 3], 2, 1)
    aᵏ = reshape([1.0 2.0], 2, 1)
    w = [-2.0, 2.0]
    b = [1.0, -3.0]
    Rᵏ = reshape([2 / 3, 8 / 3], 2, 1) # expected output

    layer = Scale(2, relu)
    ps = (; weight=w, bias=b)
    st = NamedTuple()

    affine = remove_activation(layer)
    zᵏ = forward(affine, ps, st, aᵏ)
    R̂ᵏ = @inferred propagate(rule, affine, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ
end

@testset "modify_params" begin
    W = [1.0 -1.0; 2.0 0.0]
    b = [-1.0, 1.0]
    ps = (; weight=W, bias=b)

    # ZeroRule and EpsilonRule don't modify parameters,
    # signalled by returning `ps` itself
    for rule in (ZeroRule(), EpsilonRule())
        @test modify_params(rule, ps) === ps
    end
    # Rules propagate through the activation-stripped layer: the wrap-time
    # split assigns the rule to the affine child and routes the activation
    # through a PassRule node.
    node = wrap_rules(Dense(2 => 2, relu), ZeroRule())
    @test node isa SplitActivationNode
    @test activation_fn(node.affine.layer) == identity
    @test node.activation.rule == PassRule()

    # keep_bias=false zeroes the bias
    ρps = modify_params(ZeroRule(), ps; keep_bias=false)
    @test ρps.weight == W
    @test iszero(ρps.bias)

    # Parameters without bias entries stay bias-free
    ρps = modify_params(ZeroRule(), (; weight=W); keep_bias=false)
    @test ρps.weight == W
    @test !haskey(ρps, :bias)
end

@testset "EpsilonRule denominator" begin
    rule = EpsilonRule(1.0f-2)
    @test modify_denominator(rule, [1.0f0, 0.0f0]) ≈ [1.01f0, 1.0f-2]
    @test modify_input(rule, [1.0f0, 2.0f0]) == [1.0f0, 2.0f0]
end

@testset "Parallel and SkipConnection analytic" begin
    W = [3.0 4.0; 5.0 6.0]
    b = [7.0, 8.0]
    aᵏ = reshape([1.0 2.0], 2, 1)
    # Lux `Parallel` doesn't wrap bare functions, so `NoOpLayer` replaces
    # v3's `identity` branch.
    composite = Composite(GlobalTypeMap(NoOpLayer => PassRule(), Dense => ZeroRule()))

    dense = Dense(2 => 2, relu)
    ps_dense = (; weight=W, bias=b)
    model_p = Chain(Parallel(+, NoOpLayer(), dense))
    ps_p = (; layer_1=(; layer_1=NamedTuple(), layer_2=ps_dense))
    st_p = (; layer_1=(; layer_1=NamedTuple(), layer_2=NamedTuple()))
    model_s = Chain(SkipConnection(dense, +))
    ps_s = (; layer_1=ps_dense)
    st_s = (; layer_1=NamedTuple())
    analyzer_p = LRP(model_p, ps_p, st_p, composite)
    analyzer_s = LRP(model_s, ps_s, st_s, composite)

    # aᵏ⁺¹₁ = identity(aᵏ) = [1 2]
    # aᵏ⁺¹₂ = Dense(aᵏ) = [3*1 + 4*2 + 7,  5*1 + 6*2 + 8] = [18 25]
    # aᵏ⁺¹ = [19 27]

    # For output neuron 1:
    # Rᵏ⁺¹ = [1 0]
    # Rᵏ⁺¹₁ = [1 0] .* [ 1  2] ./ [19 27] = [ 1/19 0]
    # Rᵏ⁺¹₂ = [1 0] .* [18 25] ./ [19 27] = [18/19 0]
    # The identity function is trivial:
    # Rᵏ₁ = Rᵏ⁺¹₁ = [1/19 0]
    # The Dense layer requires computation of LRP:
    # [Rᵏ₂]ⱼ = ∑ᵢ ([W]ᵢⱼ * [aᵏ]ⱼ / [aᵏ⁺¹₂]ᵢ *  [Rᵏ⁺¹₂]ᵢ)
    # [Rᵏ₂]₁ = 3*1/18*(18/19) + 5*1/25*0 = 3/19
    # [Rᵏ₂]₂ = 4*2/18*(18/19) + 6*2/25*0 = 8/19
    # Rᵏ₂ = [3/19 8/19]
    # Rᵏ = Rᵏ₁ + Rᵏ₂ = [4/19 8/19]
    e1_p = analyze(aᵏ, analyzer_p, 1)
    e1_s = analyze(aᵏ, analyzer_s, 1)
    @test e1_p.val ≈ reshape([4 / 19 8 / 19], 2, 1)
    @test e1_s.val ≈ reshape([4 / 19 8 / 19], 2, 1)

    # Analogous for output neuron 2:
    # Rᵏ⁺¹ = [0 1]
    # Rᵏ⁺¹₁ = [0 1] .* [ 1  2] ./ [19 27] = [0  2/27]
    # Rᵏ⁺¹₂ = [0 1] .* [18 25] ./ [19 27] = [0 25/27]
    # Identity function:
    # Rᵏ₁ = Rᵏ⁺¹₁ = [0 2/27]
    # Dense layer:
    # [Rᵏ₂]ⱼ = ∑ᵢ ([W]ᵢⱼ * [aᵏ]ⱼ / [aᵏ⁺¹₂]ᵢ *  [Rᵏ⁺¹₂]ᵢ)
    # [Rᵏ₂]₁ = 3*1/18*0 + 5*1/25*(25/27) =  5/27
    # [Rᵏ₂]₂ = 4*2/18*0 + 6*2/25*(25/27) = 12/27
    # Rᵏ₂ = [5/27 12/27]
    # Rᵏ = Rᵏ₁ + Rᵏ₂ = [5/27 14/27]
    e2_p = analyze(aᵏ, analyzer_p, 2)
    e2_s = analyze(aᵏ, analyzer_s, 2)
    @test e2_p.val ≈ reshape([5 / 27 14 / 27], 2, 1)
    @test e2_s.val ≈ reshape([5 / 27 14 / 27], 2, 1)
end

@testset "AlphaBetaRule analytic" begin
    aᵏ = [1.0f0, 1.0f0]
    W = [1.0f0 -1.0f0]
    b = [-1.0f0]
    layer = Dense(2 => 1)
    ps = (; weight=W, bias=b)
    st = NamedTuple()
    zᵏ = forward(layer, ps, st, aᵏ)
    Rᵏ⁺¹ = zᵏ

    # Expected outputs
    Rᵏ_α1β0 = [-1.0f0, 0.0f0]
    Rᵏ_α2β1 = [-2.0f0, 0.5f0]

    R̂ᵏ = @inferred propagate(AlphaBetaRule(1.0f0, 0.0f0), layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ_α1β0

    R̂ᵏ = @inferred propagate(AlphaBetaRule(2.0f0, 1.0f0), layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ_α2β1

    R̂ᵏ = @inferred propagate(ZPlusRule(), layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ_α1β0
end

@testset "GeneralizedGammaRule analytic" begin
    a = [-1.0, 1.0]
    a⁺ = [0.0, 1.0]
    a⁻ = [-1.0, 0.0]
    W = [1.0 -4.0; 2.0 0.0]
    b = [-2.0, 3.0]
    layer = Dense(2 => 2, leakyrelu) # leakyrelu defaults to a=0.01
    ps = (; weight=W, bias=b)
    st = NamedTuple()
    Rᵏ⁺¹ = [-0.07; 1.0]
    Rᵏ⁺¹⁺ = [0.0; 1.0]
    Rᵏ⁺¹⁻ = [-0.07; 0.0]
    affine = remove_activation(layer)
    zᵏ = forward(affine, ps, st, a)
    @test Rᵏ⁺¹ ≈ forward(layer, ps, st, a)

    W⁺ = [1.25 -4.0; 2.5 0.0] # W + γW⁺
    b⁺ = [-2.0, 3.75]         # b + γb⁺
    W⁻ = [1.0 -5.0; 2.0 0.0]  # W + γW⁻
    b⁻ = [-2.5, 3.0]          # b + γb⁻
    sˡ = Rᵏ⁺¹⁺ ./ stabilize_denom(W⁺ * a⁺ + W⁻ * a⁻ + b⁺, 1.0e-9)
    sʳ = Rᵏ⁺¹⁻ ./ stabilize_denom(W⁺ * a⁻ + W⁻ * a⁺ + b⁻, 1.0e-9)
    Rᵏ =
        a⁺ .* (transpose(W⁺) * sˡ + transpose(W⁻) * sʳ) +
        a⁻ .* (transpose(W⁻) * sˡ + transpose(W⁺) * sʳ)

    rule = GeneralizedGammaRule(0.25)
    # ˡ/ʳ: LHS/RHS of the generalized Gamma-rule equation
    psˡ⁺ = modify_params(GammaRule(0.25), ps)
    psˡ⁻ = modify_params(NegativeGammaRule(0.25), ps; keep_bias=false)
    psʳ⁻ = modify_params(NegativeGammaRule(0.25), ps)
    psʳ⁺ = modify_params(GammaRule(0.25), ps; keep_bias=false)
    @test psˡ⁺.weight == W⁺
    @test psˡ⁻.weight == W⁻
    @test psʳ⁻.weight == W⁻
    @test psʳ⁺.weight == W⁺
    @test psˡ⁺.bias == b⁺
    @test psʳ⁻.bias == b⁻
    @test iszero(psˡ⁻.bias)
    @test iszero(psʳ⁺.bias)

    R̂ᵏ = @inferred propagate(rule, affine, a, zᵏ, ps, st, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ
end

@testset "LayerNormRule analytic" begin
    rule = LayerNormRule()

    Rᵏ⁺¹ = reshape(repeat([1/3 1/3; 2/3 2/3], 4), 2, 2, 2, 2)
    aᵏ = reshape(repeat([1.0, 2.0]; inner=(2, 2, 2)), 2, 2, 2, 2)
    w = [-2.0, 2.0]
    b = [1.0, -3.0]
    Rᵏ = reshape(hcat([[2/15 2/45; 172/45 -188/45]' for _ in 1:4]...), 2, 2, 2, 2) # expected output

    # Lux folds LayerNorm's affine parameters into `ps` as `scale`/`bias`
    # of size `(shape..., 1)`; v3's Flux LayerNorm stored them in an inner
    # `.diag::Scale`. The affine parameters below replicate the v3 test.
    ps_affine = (;
        scale=reshape(repeat(Float64.(w), 1, 2), 2, 2, 1),
        bias=reshape(repeat(Float64.(b), 1, 2), 2, 2, 1),
    )

    # LayerNorm can be constructed in different ways (w/o relu, w/o affine)
    # and used either without canonizing the model (using the default ZeroRule()
    # as a fallback for the affine part) or canonized, splitting the LayerNorm
    # in two parts (normalization and affine transformation).
    # In the following, we test different combinations of this.

    ###################
    # relu activation #
    ###################
    layer = LayerNorm((2, 2), relu; epsilon=0.0f0)

    # not canonized. Under the wrap-time split, `propagate` sees the
    # activation-stripped layer. The LayerNormRule computes its own
    # statistics, so the cached pre-activation `zᵏ` is unused and `nothing`
    # is passed instead: these v3-replicating affine parameters aren't valid
    # for the Lux forward.
    affine = remove_activation(layer)
    R̂ᵏ = @inferred propagate(rule, affine, aᵏ, nothing, ps_affine, NamedTuple(), Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ

    # canonized: LayerNorm splits into normalization and affine Scale part
    model = Chain(LayerNorm((2, 2), relu; epsilon=0.0f0))
    model, ps, st = canonize(model, (; layer_1=ps_affine), (; layer_1=NamedTuple()))
    aₙ = forward(model[1], ps.layer_1, st.layer_1, aᵏ) # normalization-only part
    affine₂ = remove_activation(model[2]) # Scale layer carrying the relu
    z₂ = forward(affine₂, ps.layer_2, st.layer_2, aₙ)

    R = @inferred propagate(ZeroRule(), affine₂, aₙ, z₂, ps.layer_2, st.layer_2, Rᵏ⁺¹)
    R̂ᵏ = @inferred propagate(rule, model[1], aᵏ, aₙ, ps.layer_1, st.layer_1, R)
    @test R̂ᵏ ≈ Rᵏ

    ############################
    # no affine transformation #
    ############################
    layer = LayerNorm((2, 2); affine=false, epsilon=0.0f0)

    # not canonized
    zᵏ = forward(layer, NamedTuple(), NamedTuple(), aᵏ)
    R̂ᵏ = @inferred propagate(rule, layer, aᵏ, zᵏ, NamedTuple(), NamedTuple(), R)
    @test R̂ᵏ ≈ Rᵏ

    # canonized: a LayerNorm without affine part and activation stays unsplit
    model = Chain(LayerNorm((2, 2); affine=false, epsilon=0.0f0))
    model, ps, st = canonize(model, (; layer_1=NamedTuple()), (; layer_1=NamedTuple()))
    @test length(model.layers) == 1
    z₁ = forward(model[1], ps.layer_1, st.layer_1, aᵏ)

    R̂ᵏ = @inferred propagate(rule, model[1], aᵏ, z₁, ps.layer_1, st.layer_1, R)
    @test R̂ᵏ ≈ Rᵏ

    ######################################
    # no affine transformation, but relu #
    ######################################
    layer = LayerNorm((2, 2), relu; affine=false, epsilon=0.0f0)

    # not canonized
    affine = remove_activation(layer)
    zᵏ = forward(affine, NamedTuple(), NamedTuple(), aᵏ)
    R̂ᵏ = @inferred propagate(rule, affine, aᵏ, zᵏ, NamedTuple(), NamedTuple(), R)
    @test R̂ᵏ ≈ Rᵏ

    # canonized: splits into normalization and a bias-free Scale carrying relu
    model = Chain(LayerNorm((2, 2), relu; affine=false, epsilon=0.0f0))
    model, ps, st = canonize(model, (; layer_1=NamedTuple()), (; layer_1=NamedTuple()))
    aₙ = forward(model[1], ps.layer_1, st.layer_1, aᵏ)
    affine₂ = remove_activation(model[2])
    z₂ = forward(affine₂, ps.layer_2, st.layer_2, aₙ)

    Rₙ = @inferred propagate(ZeroRule(), affine₂, aₙ, z₂, ps.layer_2, st.layer_2, R)
    R̂ᵏ = @inferred propagate(rule, model[1], aᵏ, aₙ, ps.layer_1, st.layer_1, Rₙ)
    @test R̂ᵏ ≈ Rᵏ
end

## Test individual rules
@testset "modify_parameters" begin
    rule = GammaRule(0.42)

    # Dense layer
    W, b = [1.0 -1.0; 2.0 0.0], [-1.0, 1.0]
    ps = (; weight=W, bias=b)

    ρps = @inferred modify_params(rule, ps)
    @test ρps.weight ≈ [1.42 -1.0; 2.84 0.0]
    @test ρps.bias ≈ [-1.0, 1.42]
    @test ps.weight ≈ W # original parameters are not mutated
    @test ps.bias ≈ b

    ρps = modify_params(Val(:keep_positive), ps)
    @test ρps.weight ≈ [1.0 0.0; 2.0 0.0]
    @test ρps.bias ≈ [0.0, 1.0]

    ρps = modify_params(Val(:keep_positive), ps; keep_bias=false)
    @test ρps.weight ≈ [1.0 0.0; 2.0 0.0]
    @test ρps.bias ≈ [0.0, 0.0]

    ρps = modify_params(Val(:keep_negative), ps)
    @test ρps.weight ≈ [0.0 -1.0; 0.0 0.0]
    @test ρps.bias ≈ [-1.0, 0.0]

    ρps = modify_params(Val(:keep_negative), ps; keep_bias=false)
    @test ρps.weight ≈ [0.0 -1.0; 0.0 0.0]
    @test ρps.bias ≈ [0.0, 0.0]

    W = @inferred modify_weight(rule, W)
    b = @inferred modify_bias(rule, b)
    @test W ≈ [1.42 -1.0; 2.84 0.0]
    @test b ≈ [-1.0, 1.42]

    # Scale layer: Lux uniformly names the parameter `weight`, not `scale`
    w, b = [1.0, -1.0], [-1.0, 1.0]
    ps = (; weight=w, bias=b)

    ρps = @inferred modify_params(rule, ps)
    @test ρps.weight ≈ [1.42, -1.0]
    @test ρps.bias ≈ [-1.0, 1.42]
end

## Reference tests over all (rule, layer) combinations.
# The JLD2 reference values from v3 stay valid: all test weights are explicit
# StableRNG(123) draws that are injected into the Lux `ps` NamedTuples.
# Under the wrap-time activation split, `propagate` only ever sees affine
# layers, so the harness strips the activation before calling; the relevance
# is still seeded with the fused layer output. This preserves all reference
# values except `ZBoxRule`'s on activation-bearing layers, whose affine-only
# semantics are an intentional, documented divergence from v3.
function run_rule_tests(rule, layer, ps, st, rulename, layername, aᵏ)
    if is_compatible(rule, layer, ps)
        affine = remove_activation(layer)
        zᵏ = forward(affine, ps, st, aᵏ)
        Rᵏ⁺¹ = forward(layer, ps, st, aᵏ)
        Rᵏ = propagate(rule, affine, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
        @test typeof(Rᵏ) == typeof(aᵏ)
        @test size(Rᵏ) == size(aᵏ)
        @test_reference "references/rules/$rulename/$layername.jld2" Dict("R" => Rᵏ) by =
            (r, a) -> isapprox(r["R"], a["R"]; atol=1e-5, rtol=0.02)
    end
end

## Test Dense layer
# Define Dense test input
din = 4 # input dimension
dout = 3 # output dimension
batchsize = 2
aᵏ_dense = pseudorandn(din, batchsize)

layers = Dict(
    "Dense_relu" => (
        Dense(din => dout, relu),
        (; weight=pseudorandn(dout, din), bias=pseudorandn(dout)),
        NamedTuple(),
    ),
    "Dense_identity" => (
        Dense(din => dout; use_bias=false),
        (; weight=Matrix{Float32}(I, dout, din)),
        NamedTuple(),
    ),
)
@testset "Dense" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, (layer, ps, st)) in layers
                @testset "$layername" begin
                    run_rule_tests(rule, layer, ps, st, rulename, layername, aᵏ_dense)
                end
            end
        end
    end
end

## Test Scale layer
# Define Scale test input
d = 4 # input + output dimension
batchsize = 2
aᵏ_scale = pseudorandn(d, batchsize)

layers = Dict(
    "Scale_relu" =>
        (Scale(d, relu), (; weight=pseudorandn(d), bias=pseudorandn(d)), NamedTuple()),
    "Scale_identity" =>
        (Scale(d; use_bias=false), (; weight=ones(Float32, d)), NamedTuple()),
)
@testset "Scale" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, (layer, ps, st)) in layers
                @testset "$layername" begin
                    run_rule_tests(rule, layer, ps, st, rulename, layername, aᵏ_scale)
                end
            end
        end
    end
end

## Test ConvLayers and others
cin, cout = 3, 4
insize = (6, 6, 3, batchsize)
aᵏ = pseudorandn(insize...)
layers = Dict(
    "Conv"           => (Conv((3, 3), cin => cout), (; weight=pseudorandn(3, 3, cin, cout), bias=pseudorandn(cout)), NamedTuple()),
    "Conv_relu"      => (Conv((3, 3), cin => cout, relu), (; weight=pseudorandn(3, 3, cin, cout), bias=pseudorandn(cout)), NamedTuple()),
    "MaxPool"        => (MaxPool((3, 3)), setup_testmode(MaxPool((3, 3)))...),
    "MeanPool"       => (MeanPool((3, 3)), setup_testmode(MeanPool((3, 3)))...),
    "GlobalMaxPool"  => (GlobalMaxPool(), setup_testmode(GlobalMaxPool())...),
    "GlobalMeanPool" => (GlobalMeanPool(), setup_testmode(GlobalMeanPool())...),
    "flatten"        => (FlattenLayer(), setup_testmode(FlattenLayer())...),
    "Dropout"        => (Dropout(0.2f0), setup_testmode(Dropout(0.2f0))...),
)
@testset "Other Layers" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, (layer, ps, st)) in layers
                @testset "$layername" begin
                    run_rule_tests(rule, layer, ps, st, rulename, layername, aᵏ)
                end
            end
        end
    end
end

# Test equivalence of ZPlusRule() and AlphaBetaRule(1.0f0, 0.0f0)
layer, ps, st = layers["Conv"]
zᵏ = forward(layer, ps, st, aᵏ) # identity activation: output == pre-activation
Rᵏ⁺¹ = zᵏ
Rᵏ_z⁺ = propagate(ZPlusRule(), layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
Rᵏ_αβ = propagate(AlphaBetaRule(1.0f0, 0.0f0), layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
@test Rᵏ_z⁺ ≈ Rᵏ_αβ
