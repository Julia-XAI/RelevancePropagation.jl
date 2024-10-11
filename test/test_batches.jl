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
