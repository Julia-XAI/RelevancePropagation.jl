using RelevancePropagation
using Test
using ReferenceTests

using Flux
using JLD2
using Random: rand
using StableRNGs: StableRNG

const LRP_ANALYZERS = Dict(
    "LRPZero"                   => LRP,
    "LRPZero_COC"               => m -> LRP(m; flatten=false), # chain of chains
    "LRPEpsilonAlpha2Beta1Flat" => m -> LRP(m, EpsilonAlpha2Beta1Flat()),
)

pseudorand(dims...) = rand(StableRNG(123), Float32, dims...)

input_size = (32, 32, 3, 1)
input = pseudorand(input_size)

init(dims...) = Flux.glorot_uniform(StableRNG(123), dims...)

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1, init=init),
        Conv((3, 3), 8 => 8, relu; pad=1, init=init),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; pad=1, init=init),
        Conv((3, 3), 16 => 16, relu; pad=1, init=init),
        MaxPool((2, 2)),
    ),
    Chain(
        Flux.flatten,
        Dense(1024 => 512, relu; init=init),
        Dropout(0.5),
        Dense(512 => 100, relu; init=init),
    ),
)
Flux.testmode!(model, true)

function test_cnn(name, method)
    @testset "$name" begin
        @testset "Max activation" begin
            # Reference test explanation
            analyzer = method(model)
            println("Timing $name...")
            print("cold:")
            @time expl = analyze(input, analyzer)

            @test size(expl.val) == size(input)
            @test_reference "references/cnn/$(name)_max.jld2" Dict("expl" => expl.val) by =
                (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        end
        @testset "Neuron selection" begin
            analyzer = method(model)
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
    layer_index = 5 # last Conv layer
    n_features = 2
    features = TopNFeatures(n_features)
    analyzer = CRP(LRP(model, composite), layer_index, features)

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
    analyzer1 = LRP(model)
    analyzer2 = LRP(model; flatten=false)
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
    analyzer1 = LRP(model)
    analyzer2 = LRP(model; normalize_output_relevance=false)

    e1 = analyze(input, analyzer1)
    e2 = analyze(input, analyzer2)
    v1, v2 = e1.val, e2.val

    @test isapprox(sum(v1), 1, atol=0.05)
    @test !isapprox(sum(v2), 1; atol=0.05)

    ratio = first(v1) / first(v2)
    @test v1 ≈ v2 * ratio
end
