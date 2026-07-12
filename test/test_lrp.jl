using RelevancePropagation
using Test

using RelevancePropagation: FrozenLayer, lrp!, modify_layer
using Lux
using StableRNGs: StableRNG

model = Chain(Dense(10 => 8, relu), Dense(8 => 4, relu), Dense(4 => 3))
ps, st = Lux.setup(StableRNG(123), model)

input = rand(StableRNG(1), Float32, 10, 1)
batch = cat(input, rand(StableRNG(2), Float32, 10, 1); dims=2)

@testset "Analyzer construction" begin
    analyzer = LRP(model, ps, st)
    @test analyzer.rules ==
        (layer_1=ZeroRule(), layer_2=ZeroRule(), layer_3=ZeroRule())

    rules = [ZeroRule(), EpsilonRule(), ZeroRule()]
    analyzer = LRP(model, ps, st, rules)
    @test analyzer.rules ==
        (layer_1=ZeroRule(), layer_2=EpsilonRule(), layer_3=ZeroRule())
    @test_throws ArgumentError LRP(model, ps, st, [ZeroRule(), ZeroRule()])

    analyzer = LRP(model, ps, st, (; layer_1=ZeroRule(), layer_2=ZeroRule(), layer_3=ZeroRule()))
    @test analyzer.rules.layer_2 == ZeroRule()
    @test_throws ArgumentError LRP(
        model, ps, st, (; foo=ZeroRule(), bar=ZeroRule(), baz=ZeroRule())
    )
end

@testset "Model checks" begin
    # Softmax on output layer is not allowed
    model_softmax = Chain(Dense(10 => 8, relu), Dense(8 => 3), softmax)
    ps_s, st_s = Lux.setup(StableRNG(123), model_softmax)
    @test_throws ArgumentError LRP(model_softmax, ps_s, st_s)

    # Unknown layers are caught by the model checks unless skipped
    unknown_function(x) = x
    model_unknown = Chain(Dense(10 => 8, relu), WrappedFunction(unknown_function))
    ps_u, st_u = Lux.setup(StableRNG(123), model_unknown)
    @test_throws ErrorException LRP(model_unknown, ps_u, st_u; verbose=false)
    @test LRP(model_unknown, ps_u, st_u; skip_checks=true) isa LRP
end

@testset "End-to-end MLP" begin
    analyzer = LRP(model, ps, st)
    expl = analyze(input, analyzer)

    @test size(expl.val) == size(input)
    @test expl.analyzer == :LRP
    @test isnothing(expl.extras)

    # Neuron selection
    expl1 = analyze(input, analyzer, 1)
    @test size(expl1.val) == size(input)

    # Layerwise relevances
    expl = analyze(input, analyzer; layerwise_relevances=true)
    lwr = expl.extras.layerwise_relevances
    @test length(lwr) == 4 # input + 3 layers
    @test first(lwr) == expl.val
    @test size(lwr[2]) == (8, 1)

    # The analyzer applies the same rules as a manual backward pass
    layers = analyzer.layers
    as = (input, layers[1](input), layers[2](layers[1](input)))
    zs = layers[3](as[3])
    R3 = zero(zs)
    R3[argmax(zs)] = 1
    Rs = (similar(as[1]), similar(as[2]), similar(as[3]))
    for k in 3:-1:1
        Rᵏ⁺¹ = k == 3 ? R3 : Rs[k + 1]
        lrp!(Rs[k], ZeroRule(), layers[k], modify_layer(ZeroRule(), layers[k]), as[k], Rᵏ⁺¹)
    end
    expl = analyze(input, analyzer)
    @test expl.val ≈ Rs[1]
end

@testset "Batches" begin
    analyzer = LRP(model, ps, st)
    expl_single = analyze(input, analyzer)
    expl_batch = analyze(batch, analyzer)
    @test expl_single.val ≈ expl_batch.val[:, 1]
end

@testset "Output relevance normalization" begin
    analyzer1 = LRP(model, ps, st)
    analyzer2 = LRP(model, ps, st; normalize_output_relevance=false)

    e1 = analyze(batch, analyzer1)
    e2 = analyze(batch, analyzer2)

    for i in axes(batch, 2)
        v1 = e1.val[:, i]
        v2 = e2.val[:, i]
        ratio = first(v1) / first(v2)
        @test !isapprox(ratio, 1)
        @test v1 ≈ v2 * ratio
    end
end
