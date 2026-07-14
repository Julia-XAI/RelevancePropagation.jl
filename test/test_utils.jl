using RelevancePropagation
using Test

using RelevancePropagation: StaticLayer, activation_fn
using RelevancePropagation: has_weight, has_bias
using RelevancePropagation: check_output_softmax
using RelevancePropagation: stabilize_denom, drop_batch_index, masked_copy

using Lux
using LuxCore: AbstractLuxWrapperLayer
using StableRNGs: StableRNG

static_layer(layer) = StaticLayer(layer, Lux.setup(StableRNG(123), layer)...)

# RP extends ModelSurgeon's `activation_fn` with a `StaticLayer` method
@test activation_fn(static_layer(Dense(5 => 2, gelu))) == gelu
@test isnothing(activation_fn(static_layer(MaxPool((2, 2)))))

# has_weight / has_bias on StaticLayer
@test has_weight(static_layer(Dense(2 => 2)))
@test has_bias(static_layer(Dense(2 => 2)))
@test has_weight(static_layer(Dense(2 => 2; use_bias=false)))
@test !has_bias(static_layer(Dense(2 => 2; use_bias=false)))
@test has_weight(static_layer(Scale(2)))
@test has_bias(static_layer(Scale(2)))
@test has_weight(static_layer(Conv((3, 3), 3 => 2)))
@test !has_weight(static_layer(MaxPool((2, 2))))
@test !has_weight(static_layer(BatchNorm(2))) # BatchNorm ps are (scale, bias)

# check_output_softmax
@test_throws ArgumentError check_output_softmax(Chain(Dense(2 => 2), softmax))
@test check_output_softmax(Chain(Dense(2 => 2), relu)) isa Chain

# strip_softmax is re-exported from the ModelSurgeon submodule
# (mechanics are tested in modelsurgeon/test_strip_softmax.jl)
@test strip_softmax(Chain(Dense(2 => 2), softmax)) == Chain(Dense(2 => 2), NoOpLayer())

# stabilize_denom
A = [1.0 0.0 1.0e-25; -1.0 -0.0 -1.0e-25]
S = @inferred stabilize_denom(A, 1e-3)
@test S ≈ [1.001 1e-3 1e-3; -1.001 1e-3 -1e-3]
S = @inferred stabilize_denom(Float32.(A), 1e-2)
@test S ≈ [1.01 1.0f-2 1.0f-2; -1.01 1.0f-2 -1.0f-2]

# drop_batch_index
I1 = CartesianIndex(5, 3, 2)
I2 = @inferred drop_batch_index(I1)
@test I2 == CartesianIndex(5, 3)
I1 = CartesianIndex(5, 3, 2, 6)
I2 = @inferred drop_batch_index(I1)
@test I2 == CartesianIndex(5, 3, 2)

# masked_copy
A    = [4  9  9; 9  6  9; 1  7  8]
mask = Matrix{Bool}([0  1  1; 0  1  0; 1  1  1])
mc   = @inferred masked_copy(A, mask)
@test mc == [0  9  9; 0  6  0; 1  7  8]

#================================#
# RP's ModelSurgeon policy       #
#================================#

# RP's exported `flatten_model` and `canonize` bake in the pooling leaf
# policy: rules and composites dispatch on intact Lux pooling wrappers, so
# all nine pooling types (including the LP family) stay intact — even under
# a custom `unwrap` (mechanics are tested in modelsurgeon/test_flatten.jl).
function flat_triple(model; kwargs...)
    flatten_model(model, Lux.setup(StableRNG(123), model)...; kwargs...)
end
@test first(flat_triple(Chain(Chain(Conv((3, 3), 1 => 2, relu)), MaxPool((2, 2))))) ==
    Chain(Conv((3, 3), 1 => 2, relu), MaxPool((2, 2)))
@test first(flat_triple(Chain(GlobalMeanPool(), Chain(FlattenLayer())))) ==
    Chain(GlobalMeanPool(), FlattenLayer())

struct TestWrapper{L} <: AbstractLuxWrapperLayer{:inner}
    inner::L
end

let model = Chain(
        TestWrapper(Chain(Conv((3, 3), 1 => 2, relu), MaxPool((2, 2)))), LPPool((2, 2))
    )
    # Even the broadest possible `unwrap` splices only the custom wrapper;
    # pooling layers are protected by RP's baked-in policy.
    flat = first(flat_triple(model; unwrap=Returns(true)))
    @test flat == Chain(Conv((3, 3), 1 => 2, relu), MaxPool((2, 2)), LPPool((2, 2)))
end

# RP's exported `canonize` fuses through the same policy
let model = Chain(Chain(Conv((3, 3), 1 => 2), BatchNorm(2)), MaxPool((2, 2)))
    ps, st = Lux.setup(StableRNG(123), model)
    x = randn(StableRNG(1), Float32, 8, 8, 1, 4)
    _, st = Lux.apply(model, x, ps, Lux.trainmode(st))
    st = Lux.testmode(st)
    model_canonized, ps_canonized, st_canonized = canonize(model, ps, st)
    @test length(model_canonized) == 2 # Conv and BatchNorm fused
    @test model_canonized[1] isa Conv
    @test model_canonized[2] isa MaxPool
    @test first(Lux.apply(model_canonized, x, ps_canonized, st_canonized)) ≈
        first(Lux.apply(model, x, ps, st))
end
