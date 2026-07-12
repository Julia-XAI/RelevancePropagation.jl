using RelevancePropagation
using Test

using RelevancePropagation: FrozenLayer, activation_fn, remove_activation
using RelevancePropagation: has_weight, has_bias
using RelevancePropagation: chainall, first_element, last_element
using RelevancePropagation: has_output_softmax, check_output_softmax
using RelevancePropagation: stabilize_denom, drop_batch_index, masked_copy

using Lux
using StableRNGs: StableRNG

frozen_layer(layer) = FrozenLayer(layer, Lux.setup(StableRNG(123), layer)...)

# Test `activation_fn`
@test activation_fn(Dense(5 => 2, gelu)) == gelu
@test activation_fn(BatchNorm(5, selu)) == selu
@test activation_fn(InstanceNorm(5, selu)) == selu
@test activation_fn(GroupNorm(4, 2, selu)) == selu
@test activation_fn(LayerNorm((5,), selu)) == selu
@test activation_fn(Conv((5, 5), 3 => 2, softplus)) == softplus
@test activation_fn(ConvTranspose((5, 5), 3 => 2, softplus)) == softplus
# v3's CrossCor is Lux's Conv with cross_correlation=true (same layer type)
@test activation_fn(Conv((5, 5), 3 => 2, softplus; cross_correlation=true)) == softplus
@test activation_fn(Scale(3, relu)) == relu
@test isnothing(activation_fn(FlattenLayer()))
@test isnothing(activation_fn(MaxPool((2, 2))))
@test isnothing(activation_fn(WrappedFunction(relu)))

# remove_activation (replaces v3's `copy_layer`: parameters live in `ps`
# NamedTuples now, so rule-modified copies are covered by `modify_layer`
# tests in test_rules.jl)
@test activation_fn(remove_activation(Dense(5 => 2, gelu))) == identity
@test remove_activation(FlattenLayer()) == FlattenLayer()
let l = remove_activation(Conv((3, 3), 3 => 2, relu; stride=2, pad=1))
    @test activation_fn(l) == identity
    @test l.stride == (2, 2) # other layer configuration is preserved
end

# has_weight / has_bias on FrozenLayer
@test has_weight(frozen_layer(Dense(2 => 2)))
@test has_bias(frozen_layer(Dense(2 => 2)))
@test has_weight(frozen_layer(Dense(2 => 2; use_bias=false)))
@test !has_bias(frozen_layer(Dense(2 => 2; use_bias=false)))
@test has_weight(frozen_layer(Scale(2)))
@test has_bias(frozen_layer(Scale(2)))
@test has_weight(frozen_layer(Conv((3, 3), 3 => 2)))
@test !has_weight(frozen_layer(MaxPool((2, 2))))
@test !has_weight(frozen_layer(BatchNorm(2))) # BatchNorm ps are (scale, bias)

# chainall, first_element, last_element
d = Dense(2 => 2)
d_relu = Dense(2 => 3, relu)
model = Chain(d, Chain(d, d_relu))
@test chainall(l -> l isa Dense, model)
@test !chainall(l -> activation_fn(l) == relu, model)
@test chainall(l -> l isa Dense, Chain(d, Parallel(+, d, d), SkipConnection(d, +)))
@test !chainall(l -> l isa Dense, Chain(d, Parallel(+, d, MaxPool((2, 2)))))
@test first_element(model) == d
@test last_element(model) == d_relu

# has_output_softmax
@test has_output_softmax(Chain(Dense(2 => 2), softmax)) == true
@test has_output_softmax(Chain(Dense(2 => 2, softmax))) == true
@test has_output_softmax(Chain(Dense(2 => 2), Chain(Chain(softmax)))) == true
@test has_output_softmax(Chain(Dense(2 => 2, softmax), Dense(2 => 2, relu))) == false
@test has_output_softmax(Chain(Dense(2 => 2), tanh)) == false

# check_output_softmax
@test_throws ArgumentError check_output_softmax(Chain(Dense(2 => 2), softmax))
@test check_output_softmax(Chain(Dense(2 => 2), relu)) isa Chain

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

#=============================================================================#
# The tests below are restored from v3 (Flux/Zygote) and adapted to Lux.     #
# They cover model utilities that are not ported yet and are skipped until   #
# their port lands (phase 5, see PLAN.md).                                   #
#=============================================================================#

# flatten_model: a joint transformation of (model, ps, st),
# since flattening nested Chains re-keys `ps` and `st`.
flat_triple(model) = flatten_model(model, Lux.setup(StableRNG(123), model)...)
@test first(flat_triple(Chain(Chain(Chain(abs)), sqrt, Chain(relu)))) ==
    Chain(abs, sqrt, relu)
@test first(flat_triple(Chain(abs, sqrt, relu))) == Chain(abs, sqrt, relu)
@test first(
    flat_triple(
        Chain(Chain(Parallel(+, Chain(Chain(NoOpLayer())), Chain(Chain(NoOpLayer())))))
    ),
) == Chain(Parallel(+, Chain(NoOpLayer()), Chain(NoOpLayer())))
@test first(flat_triple(Chain(Chain(SkipConnection(Chain(Chain(NoOpLayer())), +))))) ==
    Chain(SkipConnection(Chain(NoOpLayer()), +))

# ps/st are re-keyed to the flattened layer_1..layer_N structure
let model = Chain(Chain(Dense(5 => 5), BatchNorm(5)))
    ps, st = Lux.setup(StableRNG(123), model)
    flat_model, flat_ps, flat_st = flatten_model(model, ps, st)
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

# strip_softmax: model-only in v4, `ps` stays untouched.
# A bare output softmax is replaced by `NoOpLayer`, preserving chain length.
@test strip_softmax(Chain(Dense(2 => 2), softmax)) == Chain(Dense(2 => 2), NoOpLayer())
@test strip_softmax(Chain(Dense(2 => 2, softmax))) == Chain(Dense(2 => 2, identity))
@test strip_softmax(Chain(Chain(Dense(2 => 2)), Chain(Chain(softmax)))) ==
    Chain(Chain(Dense(2 => 2)), Chain(Chain(NoOpLayer())))
@test strip_softmax(Chain(Dense(2 => 2, relu), Chain(Dense(2 => 2, softmax)))) ==
    Chain(Dense(2 => 2, relu), Chain(Dense(2 => 2, identity)))
# don't do anything if there is no softmax at the end
@test strip_softmax(Chain(Chain(Dense(2 => 2)), Chain(Chain(softmax)), Dense(2 => 2))) ==
    Chain(Chain(Dense(2 => 2)), Chain(Chain(softmax)), Dense(2 => 2))
@test strip_softmax(Chain(Dense(2 => 2, softmax), Chain(Dense(2 => 2, relu)))) ==
    Chain(Dense(2 => 2, softmax), Chain(Dense(2 => 2, relu)))
# Ignore output softmax if in Parallel or SkipConnection dataflow layer
# (unlike `Chain`, they require explicit `WrappedFunction` wrapping)
@test strip_softmax(
    Chain(
        Dense(2 => 2, softmax),
        Chain(Dense(2 => 2, relu)),
        Parallel(+, WrappedFunction(softmax), WrappedFunction(softmax)),
    ),
) == Chain(
    Dense(2 => 2, softmax),
    Chain(Dense(2 => 2, relu)),
    Parallel(+, WrappedFunction(softmax), WrappedFunction(softmax)),
)
@test strip_softmax(
    Chain(
        Dense(2 => 2, softmax),
        Chain(Dense(2 => 2, relu)),
        SkipConnection(WrappedFunction(softmax), +),
    ),
) == Chain(
    Dense(2 => 2, softmax),
    Chain(Dense(2 => 2, relu)),
    SkipConnection(WrappedFunction(softmax), +),
)
