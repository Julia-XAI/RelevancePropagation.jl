using RelevancePropagation
using Test
using ReferenceTests

using RelevancePropagation: FrozenLayer, lrp!, modify_input, modify_denominator
using RelevancePropagation: is_compatible, modify_layer, modify_weight, modify_bias
using RelevancePropagation: modify_parameters
using RelevancePropagation: activation_fn
using RelevancePropagation: stabilize_denom
using Lux
using LinearAlgebra: I
using Random: randn
using StableRNGs: StableRNG

# Fixed pseudo-random numbers
T = Float32
pseudorandn(dims...) = randn(StableRNG(123), T, dims...)

function frozen_testmode(layer)
    ps, st = Lux.setup(StableRNG(123), layer)
    return FrozenLayer(layer, ps, Lux.testmode(st))
end

# Rules ported in later phases are skipped until they land (see PLAN.md).
# The guards below make the restored v3 testsets run automatically once
# the corresponding rule is defined.
ported(s::Symbol) = isdefined(RelevancePropagation, s)

## Hand-written analytic tests
@testset "ZeroRule analytic" begin
    rule = ZeroRule()

    ## Simple dense layer
    Rᵏ⁺¹ = reshape([1 / 3 2 / 3], 2, 1)
    aᵏ = reshape([1.0 2.0], 2, 1)
    W = [3.0 4.0; 5.0 6.0]
    b = [7.0, 8.0]
    Rᵏ = reshape([17 / 90, 316 / 675], 2, 1) # expected output

    layer = FrozenLayer(Dense(2 => 2, relu), (; weight=W, bias=b), NamedTuple())
    modified_layer = modify_layer(rule, layer)
    @test activation_fn(modified_layer) == identity
    @test modified_layer.ps.weight == W
    @test modified_layer.ps.bias == b

    R̂ᵏ = similar(aᵏ) # will be inplace updated
    lrp!(R̂ᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ

    ## Pooling layer
    Rᵏ⁺¹ = Float32.([1 2; 3 4]//30)
    aᵏ = Float32.([1 2 3; 10 5 6; 7 8 9])
    Rᵏ = Float32.([0 0 0; 4 0 2; 0 0 4]//30) # expected output

    # Repeat in color channel dim and add batch dim
    Rᵏ⁺¹ = reshape(repeat(Rᵏ⁺¹, 1, 3), 2, 2, 3, 1)
    aᵏ = reshape(repeat(aᵏ, 1, 3), 3, 3, 3, 1)
    Rᵏ = reshape(repeat(Rᵏ, 1, 3), 3, 3, 3, 1)

    pool = MaxPool((2, 2); stride=(1, 1))
    layer = FrozenLayer(pool, Lux.setup(StableRNG(123), pool)...)
    modified_layer = modify_layer(rule, layer)
    @test modified_layer === layer # layers without weights are not modified

    R̂ᵏ = similar(aᵏ) # will be inplace updated
    lrp!(R̂ᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ

    ## Scale layer
    Rᵏ⁺¹ = reshape([1 / 3 2 / 3], 2, 1)
    aᵏ = reshape([1.0 2.0], 2, 1)
    w = [-2.0, 2.0]
    b = [1.0, -3.0]
    Rᵏ = reshape([2 / 3, 8 / 3], 2, 1) # expected output

    layer = FrozenLayer(Scale(2, relu), (; weight=w, bias=b), NamedTuple())
    modified_layer = modify_layer(rule, layer)

    R̂ᵏ = similar(aᵏ) # will be inplace updated
    lrp!(R̂ᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ
end

@testset "modify_layer" begin
    W = [1.0 -1.0; 2.0 0.0]
    b = [-1.0, 1.0]
    layer = FrozenLayer(Dense(2 => 2, relu), (; weight=W, bias=b), NamedTuple())

    # ZeroRule and EpsilonRule don't modify parameters
    for rule in (ZeroRule(), EpsilonRule())
        modified_layer = modify_layer(rule, layer)
        @test modified_layer.ps.weight == W
        @test modified_layer.ps.bias == b
        @test activation_fn(modified_layer) == identity
    end

    # keep_bias=false zeroes the bias
    modified_layer = modify_layer(ZeroRule(), layer; keep_bias=false)
    @test modified_layer.ps.weight == W
    @test iszero(modified_layer.ps.bias)

    # Layers without bias parameters stay bias-free
    layer_nobias = FrozenLayer(
        Dense(2 => 2, relu; use_bias=false), (; weight=W), NamedTuple()
    )
    modified_layer = modify_layer(ZeroRule(), layer_nobias)
    @test modified_layer.ps.weight == W
    @test !haskey(modified_layer.ps, :bias)
end

@testset "EpsilonRule denominator" begin
    rule = EpsilonRule(1.0f-2)
    @test modify_denominator(rule, [1.0f0, 0.0f0]) ≈ [1.01f0, 1.0f-2]
    @test modify_input(rule, [1.0f0, 2.0f0]) == [1.0f0, 2.0f0]
end

#=============================================================================#
# The testsets below are restored from v3 (Flux/Zygote) and adapted to the    #
# Lux/FrozenLayer API. They cover rules that are not ported yet and are       #
# skipped until their port lands (phase 3, see PLAN.md).                      #
#=============================================================================#

@testset "Parallel and SkipConnection analytic" begin
    if !(ported(:Composite) && ported(:PassRule))
        @test_skip false # needs PassRule (phase 3) and Composite (phase 4)
    else
        W = [3.0 4.0; 5.0 6.0]
        b = [7.0, 8.0]
        aᵏ = reshape([1.0 2.0], 2, 1)
        composite = Composite(
            GlobalTypeMap(NoOpLayer => PassRule(), Dense => ZeroRule())
        )

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
end

@testset "AlphaBetaRule analytic" begin
    if !(ported(:AlphaBetaRule) && ported(:ZPlusRule))
        @test_skip false # AlphaBetaRule and ZPlusRule land in phase 3
    else
        aᵏ = [1.0f0, 1.0f0]
        W = [1.0f0 -1.0f0]
        b = [-1.0f0]
        layer = FrozenLayer(Dense(2 => 1), (; weight=W, bias=b), NamedTuple())
        Rᵏ⁺¹ = layer(aᵏ)

        # Expected outputs
        Rᵏ_α1β0 = [-1.0f0, 0.0f0]
        Rᵏ_α2β1 = [-2.0f0, 0.5f0]

        R̂ᵏ = similar(aᵏ) # will be inplace updated
        rule = AlphaBetaRule(1.0f0, 0.0f0)
        modified_layers = modify_layer(rule, layer)
        lrp!(R̂ᵏ, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
        @test R̂ᵏ ≈ Rᵏ_α1β0

        rule = AlphaBetaRule(2.0f0, 1.0f0)
        modified_layers = modify_layer(rule, layer)
        lrp!(R̂ᵏ, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
        @test R̂ᵏ ≈ Rᵏ_α2β1

        rule = ZPlusRule()
        modified_layers = modify_layer(rule, layer)
        lrp!(R̂ᵏ, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
        @test R̂ᵏ ≈ Rᵏ_α1β0
    end
end

@testset "GeneralizedGammaRule analytic" begin
    if !ported(:GeneralizedGammaRule)
        @test_skip false # GeneralizedGammaRule lands in phase 3
    else
        a = [-1.0, 1.0]
        a⁺ = [0.0, 1.0]
        a⁻ = [-1.0, 0.0]
        W = [1.0 -4.0; 2.0 0.0]
        b = [-2.0, 3.0]
        layer = FrozenLayer(
            Dense(2 => 2, leakyrelu), (; weight=W, bias=b), NamedTuple()
        ) # leakyrelu defaults to a=0.01
        Rᵏ⁺¹ = [-0.07; 1.0]
        Rᵏ⁺¹⁺ = [0.0; 1.0]
        Rᵏ⁺¹⁻ = [-0.07; 0.0]
        @test Rᵏ⁺¹ ≈ layer(a)

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
        ml = modify_layer(rule, layer)
        @test ml.layerˡ⁺.ps.weight == W⁺
        @test ml.layerˡ⁻.ps.weight == W⁻
        @test ml.layerʳ⁻.ps.weight == W⁻
        @test ml.layerʳ⁺.ps.weight == W⁺
        @test ml.layerˡ⁺.ps.bias == b⁺
        @test ml.layerʳ⁻.ps.bias == b⁻
        @test iszero(ml.layerˡ⁻.ps.bias)
        @test iszero(ml.layerʳ⁺.ps.bias)

        R̂ᵏ = similar(Rᵏ)
        lrp!(R̂ᵏ, rule, layer, ml, a, Rᵏ⁺¹)
        @test R̂ᵏ ≈ Rᵏ
    end
end

@testset "LayerNormRule analytic" begin
    if !ported(:LayerNormRule)
        @test_skip false # LayerNormRule lands in phase 3
    else
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

        ###################
        # relu activation #
        ###################
        layer = FrozenLayer(
            LayerNorm((2, 2), relu; epsilon=0.0f0), ps_affine, NamedTuple()
        )

        # not canonized
        modified_layer = modify_layer(rule, layer)
        R̂ᵏ = similar(aᵏ) # will be inplace updated
        lrp!(R̂ᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
        @test R̂ᵏ ≈ Rᵏ

        # canonized: LayerNorm splits into normalization and affine Scale part
        if !ported(:canonize)
            @test_skip false # canonize lands in phase 5
        else
            model = Chain(LayerNorm((2, 2), relu; epsilon=0.0f0))
            model, ps, st = canonize(model, (; layer_1=ps_affine), (; layer_1=NamedTuple()))
            layer_1 = FrozenLayer(model[1], ps.layer_1, st.layer_1)
            layer_2 = FrozenLayer(model[2], ps.layer_2, st.layer_2)
            modified_layer_1 = modify_layer(LayerNormRule(), layer_1)
            modified_layer_2 = modify_layer(ZeroRule(), layer_2)

            R̂ᵏ = zero(aᵏ) # will be inplace updated
            aₙ = layer_1(aᵏ)
            R = similar(aₙ)

            lrp!(R, ZeroRule(), layer_2, modified_layer_2, aₙ, Rᵏ⁺¹)
            lrp!(R̂ᵏ, rule, layer_1, modified_layer_1, aᵏ, R)
            @test R̂ᵏ ≈ Rᵏ
        end

        ############################
        # no affine transformation #
        ############################
        layer = FrozenLayer(
            LayerNorm((2, 2); affine=false, epsilon=0.0f0), NamedTuple(), NamedTuple()
        )

        # not canonized
        modified_layer = modify_layer(rule, layer)
        R̂ᵏ = similar(aᵏ) # will be inplace updated
        R = Rᵏ⁺¹ # relevance of the normalization output equals Rᵏ⁺¹ here
        lrp!(R̂ᵏ, rule, layer, modified_layer, aᵏ, R)
        @test size(R̂ᵏ) == size(aᵏ)

        ######################################
        # no affine transformation, but relu #
        ######################################
        layer = FrozenLayer(
            LayerNorm((2, 2), relu; affine=false, epsilon=0.0f0),
            NamedTuple(),
            NamedTuple(),
        )

        # not canonized
        modified_layer = modify_layer(rule, layer)
        R̂ᵏ = zero(aᵏ) # will be inplace updated
        lrp!(R̂ᵏ, rule, layer, modified_layer, aᵏ, R)
        @test size(R̂ᵏ) == size(aᵏ)
    end
end

## Test individual rules
@testset "modify_parameters" begin
    if !ported(:GammaRule)
        @test_skip false # GammaRule lands in phase 3
    else
        rule = GammaRule(0.42)

        # Dense layer
        W, b = [1.0 -1.0; 2.0 0.0], [-1.0, 1.0]
        layer = FrozenLayer(Dense(2 => 2, relu), (; weight=W, bias=b), NamedTuple())

        modified_layer = modify_layer(rule, layer)
        @test modified_layer.ps.weight ≈ [1.42 -1.0; 2.84 0.0]
        @test modified_layer.ps.bias ≈ [-1.0, 1.42]
        @test layer.ps.weight ≈ W
        @test layer.ps.bias ≈ b

        modified_layer = modify_layer(Val(:keep_positive), layer)
        @test modified_layer.ps.weight ≈ [1.0 0.0; 2.0 0.0]
        @test modified_layer.ps.bias ≈ [0.0, 1.0]

        modified_layer = modify_layer(Val(:keep_positive), layer; keep_bias=false)
        @test modified_layer.ps.weight ≈ [1.0 0.0; 2.0 0.0]
        @test modified_layer.ps.bias ≈ [0.0, 0.0]

        modified_layer = modify_layer(Val(:keep_negative), layer)
        @test modified_layer.ps.weight ≈ [0.0 -1.0; 0.0 0.0]
        @test modified_layer.ps.bias ≈ [-1.0, 0.0]

        modified_layer = modify_layer(Val(:keep_negative), layer; keep_bias=false)
        @test modified_layer.ps.weight ≈ [0.0 -1.0; 0.0 0.0]
        @test modified_layer.ps.bias ≈ [0.0, 0.0]

        W = modify_weight(rule, W)
        b = modify_bias(rule, b)
        @test W ≈ [1.42 -1.0; 2.84 0.0]
        @test b ≈ [-1.0, 1.42]

        # Scale layer: Lux uniformly names the parameter `weight`, not `scale`
        w, b = [1.0, -1.0], [-1.0, 1.0]
        layer = FrozenLayer(Scale(2, relu), (; weight=w, bias=b), NamedTuple())

        modified_layer = modify_layer(rule, layer)
        @test modified_layer.ps.weight ≈ [1.42, -1.0]
        @test modified_layer.ps.bias ≈ [-1.0, 1.42]
    end
end

## Reference tests over all (rule, layer) combinations.
# The JLD2 reference values from v3 stay valid: all test weights are explicit
# StableRNG(123) draws that are injected into the Lux `ps` NamedTuples.
ALL_RULES_PORTED = all(
    ported,
    (
        :GammaRule,
        :ZBoxRule,
        :AlphaBetaRule,
        :WSquareRule,
        :FlatRule,
        :ZPlusRule,
        :GeneralizedGammaRule,
        :LayerNormRule,
    ),
)

@testset "Reference tests" begin
    if !ALL_RULES_PORTED
        @test_skip false # remaining rules land in phase 3
    else
        RULES = Dict(
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

        function run_rule_tests(rule, layer, rulename, layername, aᵏ)
            if is_compatible(rule, layer)
                Rᵏ⁺¹ = layer(aᵏ)
                Rᵏ = similar(aᵏ)
                modified_layer = modify_layer(rule, layer)
                lrp!(Rᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
                @test typeof(Rᵏ) == typeof(aᵏ)
                @test size(Rᵏ) == size(aᵏ)
                @test_reference "references/rules/$rulename/$layername.jld2" Dict(
                    "R" => Rᵏ
                ) by = (r, a) -> isapprox(r["R"], a["R"]; atol=1e-5, rtol=0.02)
            end
        end

        ## Test Dense layer
        din = 4 # input dimension
        dout = 3 # output dimension
        batchsize = 2
        aᵏ_dense = pseudorandn(din, batchsize)

        dense_layers = Dict(
            "Dense_relu" => FrozenLayer(
                Dense(din => dout, relu),
                (; weight=pseudorandn(dout, din), bias=pseudorandn(dout)),
                NamedTuple(),
            ),
            "Dense_identity" => FrozenLayer(
                Dense(din => dout; use_bias=false),
                (; weight=Matrix{Float32}(I, dout, din)),
                NamedTuple(),
            ),
        )
        @testset "Dense" begin
            for (rulename, rule) in RULES
                @testset "$rulename" begin
                    for (layername, layer) in dense_layers
                        @testset "$layername" begin
                            run_rule_tests(rule, layer, rulename, layername, aᵏ_dense)
                        end
                    end
                end
            end
        end

        ## Test Scale layer
        d = 4 # input + output dimension
        aᵏ_scale = pseudorandn(d, batchsize)

        scale_layers = Dict(
            "Scale_relu" => FrozenLayer(
                Scale(d, relu),
                (; weight=pseudorandn(d), bias=pseudorandn(d)),
                NamedTuple(),
            ),
            "Scale_identity" => FrozenLayer(
                Scale(d; use_bias=false), (; weight=ones(Float32, d)), NamedTuple()
            ),
        )
        @testset "Scale" begin
            for (rulename, rule) in RULES
                @testset "$rulename" begin
                    for (layername, layer) in scale_layers
                        @testset "$layername" begin
                            run_rule_tests(rule, layer, rulename, layername, aᵏ_scale)
                        end
                    end
                end
            end
        end

        ## Test ConvLayers and others
        cin, cout = 3, 4
        insize = (6, 6, 3, batchsize)
        aᵏ = pseudorandn(insize...)
        other_layers = Dict(
            "Conv" => FrozenLayer(
                Conv((3, 3), cin => cout),
                (; weight=pseudorandn(3, 3, cin, cout), bias=pseudorandn(cout)),
                NamedTuple(),
            ),
            "Conv_relu" => FrozenLayer(
                Conv((3, 3), cin => cout, relu),
                (; weight=pseudorandn(3, 3, cin, cout), bias=pseudorandn(cout)),
                NamedTuple(),
            ),
            "MaxPool"        => frozen_testmode(MaxPool((3, 3))),
            "MeanPool"       => frozen_testmode(MeanPool((3, 3))),
            "GlobalMaxPool"  => frozen_testmode(GlobalMaxPool()),
            "GlobalMeanPool" => frozen_testmode(GlobalMeanPool()),
            "flatten"        => frozen_testmode(FlattenLayer()),
            "Dropout"        => frozen_testmode(Dropout(0.2f0)),
        )
        @testset "Other Layers" begin
            for (rulename, rule) in RULES
                @testset "$rulename" begin
                    for (layername, layer) in other_layers
                        @testset "$layername" begin
                            run_rule_tests(rule, layer, rulename, layername, aᵏ)
                        end
                    end
                end
            end
        end

        # Test equivalence of ZPlusRule() and AlphaBetaRule(1.0f0, 0.0f0)
        @testset "ZPlusRule ≡ AlphaBetaRule(1, 0)" begin
            layer = other_layers["Conv"]
            Rᵏ⁺¹ = layer(aᵏ)
            Rᵏ_z⁺ = similar(aᵏ)
            Rᵏ_αβ = similar(aᵏ)
            rule = ZPlusRule()
            modified_layers = modify_layer(rule, layer)
            lrp!(Rᵏ_z⁺, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
            rule = AlphaBetaRule(1.0f0, 0.0f0)
            modified_layers = modify_layer(rule, layer)
            lrp!(Rᵏ_αβ, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
            @test Rᵏ_z⁺ ≈ Rᵏ_αβ
        end
    end
end
