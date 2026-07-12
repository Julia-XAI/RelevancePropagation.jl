using RelevancePropagation
using Test
using ReferenceTests

using Lux
using JLD2
using Random: rand
using StableRNGs: StableRNG

pseudorand(dims...) = rand(StableRNG(123), Float32, dims...)

input_size = (32, 32, 3, 1)
input = pseudorand(input_size)

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; pad=1),
        Conv((3, 3), 16 => 16, relu; pad=1),
        MaxPool((2, 2)),
    ),
    Chain(
        FlattenLayer(), Dense(1024 => 512, relu), Dropout(0.5f0), Dense(512 => 100, relu)
    ),
)
ps, st = Lux.setup(StableRNG(123), model)
flat_model, flat_ps, flat_st = flatten_model(model, ps, st)

# v3's `flatten` LRP kwarg became the explicit `flatten_model` transform
# of the Lux triple; composites with positional primitives apply to the
# flattened triple, matching v3's `flatten=true` default.
const LRP_ANALYZERS = Dict(
    "LRPZero" => () -> LRP(flat_model, flat_ps, flat_st),
    "LRPZero_COC" => () -> LRP(model, ps, st), # chain of chains
    "LRPEpsilonAlpha2Beta1Flat" =>
        () -> LRP(flat_model, flat_ps, flat_st, EpsilonAlpha2Beta1Flat()),
)

function test_cnn(name, method)
    @testset "$name" begin
        @testset "Max activation" begin
            # Reference test explanation
            analyzer = method()
            println("Timing $name...")
            print("cold:")
            @time expl = analyze(input, analyzer)

            @test size(expl.val) == size(input)
            @test_reference "references/cnn/$(name)_max.jld2" Dict("expl" => expl.val) by =
                (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        end
        @testset "Neuron selection" begin
            analyzer = method()
            print("warm:")
            @time expl = analyze(input, analyzer, 1)

            @test size(expl.val) == size(input)
            @test_reference "references/cnn/$(name)_ns1.jld2" Dict("expl" => expl.val) by =
                (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        end
    end
end

# Run analyzers
@testset "LRP analyzers" begin
    for (name, method) in LRP_ANALYZERS
        test_cnn(name, method)
    end
end

@testset "CRP" begin
    composite = EpsilonPlus()
    layer_index = 5 # last Conv layer in the flattened model
    n_features = 2
    features = TopNFeatures(n_features)
    analyzer = CRP(LRP(flat_model, flat_ps, flat_st, composite), layer_index, features)

    @testset "Max activation" begin
        println("Timing CRP...")
        print("cold:")
        @time expl = analyze(input, analyzer)

        @test size(expl.val) == size(input) .* (1, 1, 1, n_features)
        @test_reference "references/cnn/CRP_max.jld2" Dict("expl" => expl.val) by =
            (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
    end
    @testset "Neuron selection" begin
        print("warm:")
        @time expl = analyze(input, analyzer, 1)

        @test size(expl.val) == size(input) .* (1, 1, 1, n_features)
        @test_reference "references/cnn/CRP_ns1.jld2" Dict("expl" => expl.val) by =
            (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
    end
end

# Layerwise relevances in LRP methods
@testset "Layerwise relevances" begin
    analyzer1 = LRP(flat_model, flat_ps, flat_st)
    analyzer2 = LRP(model, ps, st)
    e1 = analyze(input, analyzer1; layerwise_relevances=true)
    e2 = analyze(input, analyzer2; layerwise_relevances=true)
    lwr1 = e1.extras.layerwise_relevances
    lwr2 = e2.extras.layerwise_relevances

    @test length(lwr1) == 11 # 10 layers in flattened VGG11
    @test length(lwr2) == 3 # 2 chains in unflattened VGG11
    @test lwr1[1] ≈ lwr2[1]
    @test lwr1[end] ≈ lwr2[end]
end

@testset "Normalized output relevance" begin
    analyzer1 = LRP(flat_model, flat_ps, flat_st)
    analyzer2 = LRP(flat_model, flat_ps, flat_st; normalize_output_relevance=false)

    e1 = analyze(input, analyzer1)
    e2 = analyze(input, analyzer2)
    v1, v2 = e1.val, e2.val

    # Conservation is approximate: bias terms absorb relevance, and the amount
    # depends on the parameter draw (0.90 for this Lux.setup seed).
    @test isapprox(sum(v1), 1, atol=0.15)
    @test !isapprox(sum(v2), 1; atol=0.15)

    ratio = first(v1) / first(v2)
    @test v1 ≈ v2 * ratio
end
