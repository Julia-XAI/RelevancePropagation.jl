using Test

using RelevancePropagation: ModelSurgeon # RP exports a conflicting `flatten_model`
using Lux
using LuxCore: AbstractLuxWrapperLayer
using Functors: KeyPath
using StableRNGs: StableRNG

# flatten_model: a joint transformation of (model, ps, st),
# since flattening nested Chains re-keys `ps` and `st`.
function ms_flat_triple(model; kwargs...)
    ModelSurgeon.flatten_model(model, Lux.setup(StableRNG(123), model)...; kwargs...)
end
@test first(ms_flat_triple(Chain(Chain(Chain(abs)), sqrt, Chain(relu)))) ==
    Chain(abs, sqrt, relu)
@test first(ms_flat_triple(Chain(abs, sqrt, relu))) == Chain(abs, sqrt, relu)
@test first(
    ms_flat_triple(
        Chain(Chain(Parallel(+, Chain(Chain(NoOpLayer())), Chain(Chain(NoOpLayer())))))
    ),
) == Chain(Parallel(+, Chain(NoOpLayer()), Chain(NoOpLayer())))
@test first(ms_flat_triple(Chain(Chain(SkipConnection(Chain(Chain(NoOpLayer())), +))))) ==
    Chain(SkipConnection(Chain(NoOpLayer()), +))

# ps/st are re-keyed to the flattened layer_1..layer_N structure
let model = Chain(Chain(Dense(5 => 5), BatchNorm(5)))
    ps, st = Lux.setup(StableRNG(123), model)
    flat_model, flat_ps, flat_st = ModelSurgeon.flatten_model(model, ps, st)
    @test flat_model == Chain(Dense(5 => 5), BatchNorm(5))
    @test keys(flat_model.layers) == (:layer_1, :layer_2)
    @test flat_ps.layer_1 == ps.layer_1.layer_1
    @test flat_ps.layer_2 == ps.layer_1.layer_2
    @test flat_st.layer_1 == st.layer_1.layer_1
    @test flat_st.layer_2 == st.layer_1.layer_2

    x = randn(StableRNG(1), Float32, 5, 4)
    y, _ = Lux.apply(model, x, ps, Lux.testmode(st))
    y_flat, _ = Lux.apply(flat_model, x, flat_ps, Lux.testmode(flat_st))
    @test y ≈ y_flat
end

# Unwrapping `AbstractLuxWrapperLayer`s (the pattern used by Boltz.jl model
# wrappers), whose `ps`/`st` pass through to the wrapped layer, is opt-in via
# the `unwrap` keyword argument — both at the top level and inside `Chain`s.
struct FlattenTestWrapper{L} <: AbstractLuxWrapperLayer{:inner}
    inner::L
end
unwrap_flatten_test_wrapper = Base.Fix2(isa, FlattenTestWrapper)

let model = FlattenTestWrapper(
        Chain(Dense(2 => 3, relu), Chain(Dense(3 => 2)), FlattenTestWrapper(Dense(2 => 2)))
    )
    ps, st = Lux.setup(StableRNG(123), model)
    flat_model, flat_ps, flat_st = ModelSurgeon.flatten_model(
        model, ps, st; unwrap=unwrap_flatten_test_wrapper
    )
    @test flat_model == Chain(Dense(2 => 3, relu), Dense(3 => 2), Dense(2 => 2))
    @test keys(flat_ps) == (:layer_1, :layer_2, :layer_3)

    x = randn(StableRNG(1), Float32, 2, 4)
    y, _ = Lux.apply(model, x, ps, st)
    y_flat, _ = Lux.apply(flat_model, x, flat_ps, flat_st)
    @test y ≈ y_flat
end

# Nested wrappers unwrap recursively; wrapped `Chain`s are spliced in.
@test first(
    ms_flat_triple(
        FlattenTestWrapper(FlattenTestWrapper(Chain(NoOpLayer())));
        unwrap=unwrap_flatten_test_wrapper,
    ),
) == Chain(NoOpLayer())
@test first(
    ms_flat_triple(
        Chain(abs, FlattenTestWrapper(Chain(sqrt, relu)));
        unwrap=unwrap_flatten_test_wrapper,
    ),
) == Chain(abs, sqrt, relu)

# By default, wrapper layers are kept intact: the `AbstractLuxWrapperLayer`
# trait only guarantees `ps`/`st` transparency, not application transparency.
@test first(ms_flat_triple(Chain(abs, FlattenTestWrapper(Chain(sqrt, relu))))).layers.layer_2 isa
    FlattenTestWrapper

# Lux pooling layers are `AbstractLuxWrapperLayer`s around internal pooling
# ops and stay intact when flattening, including the LP family.
@test first(ms_flat_triple(Chain(Chain(Conv((3, 3), 1 => 2, relu)), MaxPool((2, 2))))) ==
    Chain(Conv((3, 3), 1 => 2, relu), MaxPool((2, 2)))
@test first(ms_flat_triple(Chain(GlobalMeanPool(), Chain(FlattenLayer())))) ==
    Chain(GlobalMeanPool(), FlattenLayer())
let flat = first(
        ms_flat_triple(Chain(Chain(LPPool((2, 2))), GlobalLPPool(), AdaptiveLPPool((1, 1))))
    )
    @test flat.layers.layer_1 isa LPPool
    @test flat.layers.layer_2 isa GlobalLPPool
    @test flat.layers.layer_3 isa AdaptiveLPPool
end

# `Maxout` and `RepeatedLayer` are wrapper layers that are *not*
# application-transparent; flattening must keep them intact.
let model = Chain(
        Chain(Dense(2 => 4)),
        Maxout(Dense(4 => 4), Dense(4 => 4)),
        RepeatedLayer(Dense(4 => 4); repeats=Val(2)),
    )
    ps, st = Lux.setup(StableRNG(123), model)
    flat_model, flat_ps, flat_st = ModelSurgeon.flatten_model(model, ps, st)
    @test flat_model.layers.layer_2 isa Maxout
    @test flat_model.layers.layer_3 isa RepeatedLayer

    x = randn(StableRNG(1), Float32, 2, 4)
    y, _ = Lux.apply(model, x, ps, st)
    y_flat, _ = Lux.apply(flat_model, x, flat_ps, flat_st)
    @test y ≈ y_flat
end

# `exclude` keeps layers intact and takes precedence over splicing and
# `unwrap`; the KeyPath passed to it follows the `ps`/`st` structure.
let model = Chain(Chain(Chain(abs), sqrt), FlattenTestWrapper(Chain(relu)))
    keep(kp, l) = kp == KeyPath(:layer_1, :layer_1) || l isa FlattenTestWrapper
    flat = first(ms_flat_triple(model; exclude=keep, unwrap=unwrap_flatten_test_wrapper))
    @test flat == Chain(Chain(abs), sqrt, FlattenTestWrapper(Chain(relu)))
end
