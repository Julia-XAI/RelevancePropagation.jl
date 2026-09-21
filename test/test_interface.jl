using RelevancePropagation
using Test

using Flux
using StableRNGs: StableRNG

pseudorand(dims...) = rand(StableRNG(123), Float32, dims...)

model = Chain(
    Conv((3, 3), 3 => 4, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(36 => 5, relu),
    Dense(5 => 3),
)
input = pseudorand(8, 8, 3, 2)

lrp = LRP(model)
crp = CRP(lrp, 4, TopNFeatures(2)) # concepts of the hidden Dense layer

@testset "LRP" begin
    @test XAIBase.test_interface(lrp, input)
    @test XAIBase.test_interface(lrp, input; output_selection=2)
    @test XAIBase.test_interface(LRP(model, EpsilonPlusFlat()), input)
    @test analyze(input, lrp).pooling isa SumPooling
end

# CRP attributes concepts instead of input features:
# the attributions of all concepts are concatenated along the batch dimension,
# so the batch-size checks of `XAIBase.test_interface` don't apply.
@testset "CRP" begin
    attr = analyze(input, crp)
    @test attr isa Attribution
    @test attr.pooling isa SumPooling
    @test size(attr.val) == size(input) .* (1, 1, 1, 2)
end
