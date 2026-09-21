using Test

using RelevancePropagation.ModelSurgeon: activation_fn, remove_activation
using Lux

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
