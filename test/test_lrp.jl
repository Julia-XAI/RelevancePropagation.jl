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

@testset "Nested dataflow layers" begin
    # Analytic Parallel and SkipConnection test with explicit rule NamedTuples;
    # the same values are tested via a Composite in test_rules.jl.
    W = [3.0 4.0; 5.0 6.0]
    b = [7.0, 8.0]
    aᵏ = reshape([1.0 2.0], 2, 1)
    dense = Dense(2 => 2, relu)
    ps_dense = (; weight=W, bias=b)

    model_p = Chain(Parallel(+, NoOpLayer(), dense))
    ps_p = (; layer_1=(; layer_1=NamedTuple(), layer_2=ps_dense))
    st_p = (; layer_1=(; layer_1=NamedTuple(), layer_2=NamedTuple()))
    rules_p = (; layer_1=(; layer_1=PassRule(), layer_2=ZeroRule()))
    analyzer_p = LRP(model_p, ps_p, st_p, rules_p)

    model_s = Chain(SkipConnection(dense, +))
    ps_s = (; layer_1=ps_dense)
    st_s = (; layer_1=NamedTuple())
    rules_s = (; layer_1=ZeroRule()) # rule on a SkipConnection targets the wrapped layer
    analyzer_s = LRP(model_s, ps_s, st_s, rules_s)

    # See test_rules.jl for the derivation of the expected values.
    e1_p = analyze(aᵏ, analyzer_p, 1)
    e1_s = analyze(aᵏ, analyzer_s, 1)
    @test e1_p.val ≈ reshape([4 / 19 8 / 19], 2, 1)
    @test e1_s.val ≈ reshape([4 / 19 8 / 19], 2, 1)

    e2_p = analyze(aᵏ, analyzer_p, 2)
    e2_s = analyze(aᵏ, analyzer_s, 2)
    @test e2_p.val ≈ reshape([5 / 27 14 / 27], 2, 1)
    @test e2_s.val ≈ reshape([5 / 27 14 / 27], 2, 1)

    # A nested Chain yields the same relevances as its flat equivalent
    model_flat = Chain(Dense(10 => 8, relu), Dense(8 => 8, relu), Dense(8 => 4, relu), Dense(4 => 3))
    ps_flat, st_flat = Lux.setup(StableRNG(456), model_flat)
    model_nested = Chain(
        Dense(10 => 8, relu), Chain(Dense(8 => 8, relu), Dense(8 => 4, relu)), Dense(4 => 3)
    )
    ps_nested = (;
        layer_1=ps_flat.layer_1,
        layer_2=(; layer_1=ps_flat.layer_2, layer_2=ps_flat.layer_3),
        layer_3=ps_flat.layer_4,
    )
    st_nested = (;
        layer_1=st_flat.layer_1,
        layer_2=(; layer_1=st_flat.layer_2, layer_2=st_flat.layer_3),
        layer_3=st_flat.layer_4,
    )
    rules_nested = (;
        layer_1=ZeroRule(),
        layer_2=(; layer_1=EpsilonRule(), layer_2=ZeroRule()),
        layer_3=EpsilonRule(),
    )
    analyzer_flat = LRP(
        model_flat, ps_flat, st_flat, [ZeroRule(), EpsilonRule(), ZeroRule(), EpsilonRule()]
    )
    analyzer_nested = LRP(model_nested, ps_nested, st_nested, rules_nested)
    e_flat = analyze(batch, analyzer_flat)
    e_nested = analyze(batch, analyzer_nested)
    @test e_nested.val ≈ e_flat.val

    # SkipConnection wrapping a Chain takes a nested rules NamedTuple
    model_sc = Chain(
        Dense(10 => 10, relu), SkipConnection(Chain(Dense(10 => 8, relu), Dense(8 => 10)), +)
    )
    ps_sc, st_sc = Lux.setup(StableRNG(789), model_sc)
    rules_sc = (; layer_1=ZeroRule(), layer_2=(; layer_1=ZeroRule(), layer_2=EpsilonRule()))
    analyzer_sc = LRP(model_sc, ps_sc, st_sc, rules_sc)
    e_sc = analyze(batch, analyzer_sc)
    @test size(e_sc.val) == size(batch)
    @test !any(isnan, e_sc.val)
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
