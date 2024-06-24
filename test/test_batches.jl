using RelevancePropagation
using Test

using Flux
using Random: rand, MersenneTwister

pseudorand(dims...) = rand(MersenneTwister(123), Float32, dims...)

## Test `fuse_batchnorm` on Dense and Conv layers
ins = 20
outs = 10
batchsize = 15

model = Chain(Dense(ins, outs, relu; init=pseudorand))

# Input 1 w/o batch dimension
input1_no_bd = rand(MersenneTwister(1), Float32, ins)
# Input 1 with batch dimension
input1_bd = reshape(input1_no_bd, ins, 1)
# Input 2 with batch dimension
input2_bd = rand(MersenneTwister(2), Float32, ins, 1)
# Batch containing inputs 1 & 2
input_batch = cat(input1_bd, input2_bd; dims=2)

ANALYZERS = Dict(
    "LRPZero"     => LRP,
    "LRPZero_COC" => m -> LRP(m; flatten=false),  # chain of chains
)

for (name, method) in ANALYZERS
    @testset "$name" begin
        # Using `add_batch_dim=true` should result in same explanation
        # as input reshaped to have a batch dimension
        analyzer = method(model)
        expl1_no_bd = analyzer(input1_no_bd; add_batch_dim=true)
        analyzer = method(model)
        expl1_bd = analyzer(input1_bd)
        @test expl1_bd.val ≈ expl1_no_bd.val

        # Analyzing a batch should have the same result
        # as analyzing inputs in batch individually
        analyzer = method(model)
        expl2_bd = analyzer(input2_bd)
        analyzer = method(model)
        expl_batch = analyzer(input_batch)
        @test expl1_bd.val ≈ expl_batch.val[:, 1]
    end
end
