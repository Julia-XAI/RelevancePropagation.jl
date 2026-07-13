using Test

using RelevancePropagation.ModelSurgeon: strip_softmax, has_output_softmax
using Lux

# has_output_softmax
@test has_output_softmax(Chain(Dense(2 => 2), softmax)) == true
@test has_output_softmax(Chain(Dense(2 => 2, softmax))) == true
@test has_output_softmax(Chain(Dense(2 => 2), Chain(Chain(softmax)))) == true
@test has_output_softmax(Chain(Dense(2 => 2, softmax), Dense(2 => 2, relu))) == false
@test has_output_softmax(Chain(Dense(2 => 2), tanh)) == false

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
