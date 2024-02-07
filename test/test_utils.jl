using Flux
using Flux: flatten
using RelevancePropagation: activation_fn, copy_layer, flatten_model
using RelevancePropagation: has_output_softmax, check_output_softmax
using RelevancePropagation: stabilize_denom, drop_batch_index, masked_copy
using Random

# Test `activation_fn`
@test activation_fn(Dense(5, 2, gelu)) == gelu
for T in (BatchNorm, LayerNorm, InstanceNorm)
    @test activation_fn(T(5, selu)) == selu
end
@test activation_fn(GroupNorm(4, 2, selu)) == selu
for T in (Conv, ConvTranspose, CrossCor)
    @test activation_fn(T((5, 5), 3 => 2, softplus)) == softplus
end
@test isnothing(activation_fn(flatten))

# copy_layer
for T in (Conv, ConvTranspose, CrossCor)
    l1 = T((3, 3), 3 => 2, relu)
    l2 = copy_layer(l1, 2 * l1.weight, 0.1 * l1.bias; σ=gelu)
    @test l2.weight ≈ 2 * l1.weight
    @test l2.bias ≈ 0.1 * l1.bias
    @test activation_fn(l2) == gelu
end

# flatten_model
@test flatten_model(Chain(Chain(Chain(abs)), sqrt, Chain(relu))) == Chain(abs, sqrt, relu)
@test flatten_model(Chain(abs, sqrt, relu)) == Chain(abs, sqrt, relu)
@test flatten_model(
    Chain(Chain(Parallel(+, Chain(Chain(identity)), Chain(Chain(identity)))))
) == Chain(Parallel(+, Chain(identity), Chain(identity)))
@test flatten_model(Chain(Chain(SkipConnection(Chain(Chain(identity)), +)))) ==
    Chain(SkipConnection(Chain(identity), +))

# has_output_softmax
@test has_output_softmax(Chain(abs, sqrt, relu, softmax)) == true
@test has_output_softmax(Chain(abs, sqrt, relu, tanh)) == false
@test has_output_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax)))) == true
@test has_output_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) == false
@test has_output_softmax(Chain(Dense(5, 5, softmax), Dense(5, 5, softmax))) == true
@test has_output_softmax(Chain(Dense(5, 5, softmax), Dense(5, 5, relu))) == false
@test has_output_softmax(Chain(Dense(5, 5, softmax), Chain(Dense(5, 5, softmax)))) == true
@test has_output_softmax(Chain(Dense(5, 5, softmax), Chain(Dense(5, 5, relu)))) == false

# check_output_softmax
@test_throws ArgumentError check_output_softmax(Chain(abs, sqrt, relu, softmax))

# strip_softmax
d_softmax  = Dense(2, 2, softmax; init=pseudorand)
d_softmax2 = Dense(2, 2, softmax; init=pseudorand)
d_relu     = Dense(2, 2, relu; init=pseudorand)
d_identity = Dense(2, 2; init=pseudorand)
# flatten to remove softmax
m = strip_softmax(Chain(Chain(abs), sqrt, Chain(Chain(softmax))))
@test m == Chain(Chain(abs), sqrt, Chain(Chain(identity)))
m1 = strip_softmax(Chain(d_relu, Chain(d_softmax)))
m2 = Chain(d_relu, Chain(d_identity))
x = rand(Float32, 2, 10)
@test typeof(m1) == typeof(m2)
@test m1(x) == m2(x)
# don't do anything if there is no softmax at the end
@test strip_softmax(Chain(Chain(abs), Chain(Chain(softmax)), sqrt)) ==
    Chain(Chain(abs), Chain(Chain(softmax)), sqrt)
@test strip_softmax(Chain(d_softmax, Chain(d_relu))) == Chain(d_softmax, Chain(d_relu))
@test strip_softmax(Chain(Parallel(+, softmax, softmax), d_softmax, Chain(d_relu))) ==
    Chain(Parallel(+, softmax, softmax), d_softmax, Chain(d_relu))
@test strip_softmax(Chain(SkipConnection(softmax, +), d_softmax, Chain(d_relu))) ==
    Chain(SkipConnection(softmax, +), d_softmax, Chain(d_relu))
# Ignore output softmax if in Parallel or SkipConnection dataflow layer
@test strip_softmax(Chain(d_softmax, Chain(d_relu), Parallel(+, softmax, softmax))) ==
    Chain(d_softmax, Chain(d_relu), Parallel(+, softmax, softmax))
@test strip_softmax(Chain(d_softmax, Chain(d_relu), SkipConnection(softmax, +))) ==
    Chain(d_softmax, Chain(d_relu), SkipConnection(softmax, +))

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
