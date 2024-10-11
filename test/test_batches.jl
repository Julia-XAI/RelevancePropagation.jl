using RelevancePropagation
using Test

using Flux
using Random: rand
using StableRNGs: StableRNG

pseudorand(dims...) = rand(StableRNG(123), Float32, dims...)

## Test `fuse_batchnorm` on Dense and Conv layers
ins = 20
outs = 10
batchsize = 15

model = Chain(Dense(ins, outs, relu; init=pseudorand))

# Input 1 w/o batch dimension
input1_no_bd = rand(StableRNG(1), Float32, ins)
# Input 1 with batch dimension
input1_bd = reshape(input1_no_bd, ins, 1)
# Input 2 with batch dimension
input2_bd = rand(StableRNG(2), Float32, ins, 1)
# Batch containing inputs 1 & 2
input_batch = cat(input1_bd, input2_bd; dims=2)

ANALYZERS = Dict(
    "LRPZero"     => LRP,
    "LRPZero_COC" => m -> LRP(m; flatten=false),  # chain of chains
)

for (name, method) in ANALYZERS
    @testset "$name" begin
        # Analyzing a batch should have the same result
        # as analyzing inputs in batch individually
        analyzer = method(model)
        expl2_bd = analyzer(input2_bd)
        analyzer = method(model)
        expl_batch = analyzer(input_batch)
        @test expl2_bd.val ≈ expl_batch.val[:, 2]
    end
end

@testset "Normalized output relevance" begin
    analyzer1 = LRP(model)
    analyzer2 = LRP(model; normalize_output_relevance=false)

    e1 = analyze(input_batch, analyzer1)
    e2 = analyze(input_batch, analyzer2)
    v1_bd1 = e1.val[:, 1]
    v1_bd2 = e1.val[:, 2]
    v2_bd1 = e2.val[:, 1]
    v2_bd2 = e2.val[:, 2]

    @test isapprox(sum(v1_bd1), 1, atol=0.05)
    @test isapprox(sum(v1_bd2), 1, atol=0.05)
    @test !isapprox(sum(v2_bd1), 1; atol=0.05)
    @test !isapprox(sum(v2_bd2), 1; atol=0.05)

    ratio_bd1 = first(v1_bd1) / first(v2_bd1)
    ratio_bd2 = first(v1_bd2) / first(v2_bd2)
    @test !isapprox(ratio_bd1, ratio_bd2)
    @test v1_bd1 ≈ v2_bd1 * ratio_bd1
    @test v1_bd2 ≈ v2_bd2 * ratio_bd2
end
