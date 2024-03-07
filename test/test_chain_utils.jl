using RelevancePropagation: ChainTuple, ParallelTuple, SkipConnectionTuple
using RelevancePropagation: ModelIndex, chainmap, chainindices, chainzip
using RelevancePropagation: activation_fn
using Flux

x = rand(Float32, 2, 5)
d1 = Dense(2, 2, relu)
d2 = Dense(2, 2, selu)
d3 = Dense(2, 2, gelu)
d4 = Dense(2, 2, celu)

c1 = Chain(d1)
c2 = Chain(d1, d2)
c3 = Chain(Chain(d1, d1), d2)
c4 = Chain(d1, Chain(d2, d2))
c5 = Chain(d1, Chain(d2, d2), d3)
c6 = Chain(Parallel(+, d1, d1))
c7 = Chain(d1, Parallel(+, d2, d2, Chain(d3, d3)), d4)
c8 = Chain(SkipConnection(d1, +))
c9 = Chain(SkipConnection(SkipConnection(d1, +), +))
c10 = Chain(d1, SkipConnection(d2, +))
c11 = Chain(d1, SkipConnection(Chain(d2, d3), +), d4)

# pre-compute occuring hidden activations, where hXYZ = dX(dY(dZ(x))) = dX(hYZ)
h1 = d1(x)
h11 = d1(h1)
h21 = d2(h1)
h31 = d3(h1)
h211 = d2(h11)
h221 = d2(h21)
h3221 = d3(h221)
h331 = d3(d3(h1))
h4p1 = d4(2 * h21 + h331) # output of Chain c6

# Test chainmap
@test chainmap(activation_fn, c1) == ChainTuple(relu)
@test chainmap(activation_fn, c2) == ChainTuple(relu, selu)
@test chainmap(activation_fn, c3) == ChainTuple(ChainTuple(relu, relu), selu)
@test chainmap(activation_fn, c4) == ChainTuple(relu, ChainTuple(selu, selu))
@test chainmap(activation_fn, c5) == ChainTuple(relu, ChainTuple(selu, selu), gelu)
@test chainmap(activation_fn, c6) == ChainTuple(ParallelTuple(relu, relu))
@test chainmap(activation_fn, c7) ==
    ChainTuple(relu, ParallelTuple(selu, selu, ChainTuple(gelu, gelu)), celu)
@test chainmap(activation_fn, c8) == ChainTuple(SkipConnectionTuple(relu))
@test chainmap(activation_fn, c9) ==
    ChainTuple(SkipConnectionTuple(SkipConnectionTuple(relu)))
@test chainmap(activation_fn, c10) == ChainTuple(relu, SkipConnectionTuple(selu))
@test chainmap(activation_fn, c11) ==
    ChainTuple(relu, SkipConnectionTuple(ChainTuple(selu, gelu)), celu)

const MI = ModelIndex
@test chainindices(c1) == ChainTuple(MI(1))
@test chainindices(c2) == ChainTuple(MI(1), MI(2))
@test chainindices(c3) == ChainTuple(ChainTuple(MI(1, 1), MI(1, 2)), MI(2))
@test chainindices(c4) == ChainTuple(MI(1), ChainTuple(MI(2, 1), MI(2, 2)))
@test chainindices(c5) == ChainTuple(MI(1), ChainTuple(MI(2, 1), MI(2, 2)), MI(3))
@test chainindices(c6) == ChainTuple(ParallelTuple(MI(1, 1), MI(1, 2)))
@test chainindices(c7) == ChainTuple(
    MI(1), ParallelTuple(MI(2, 1), MI(2, 2), ChainTuple(MI(2, 3, 1), MI(2, 3, 2))), MI(3)
)
@test chainindices(c8) == ChainTuple(SkipConnectionTuple(MI(1, 1)))
@test chainindices(c9) == ChainTuple(SkipConnectionTuple(SkipConnectionTuple(MI(1, 1, 1))))
@test chainindices(c10) == ChainTuple(MI(1), SkipConnectionTuple(MI(2, 1)))
@test chainindices(c11) ==
    ChainTuple(MI(1), SkipConnectionTuple(ChainTuple(MI(2, 1, 1), MI(2, 1, 2))), MI(3))

@test ModelIndex(1) ∈ ModelIndex(1)
@test ModelIndex(1) ∉ ModelIndex(2)
@test ModelIndex(1, 2) ∈ ModelIndex(1)
@test ModelIndex(1, 2) ∉ ModelIndex(2)
@test ModelIndex(1, 2) ∈ ModelIndex(1, 2)
@test ModelIndex(1, 2, 3) ∈ ModelIndex(1, 2)
@test ModelIndex(1, 2) ∉ ModelIndex(1, 2, 3)

# Test chainzip
t1 = ChainTuple(1, 2, 3)
t2 = ChainTuple(4, 5, 6)
t3 = ChainTuple(7, 8, 9)
@test chainzip(+, t1, t2) == ChainTuple(5, 7, 9)
@test chainzip(+, t1, t2, t3) == ChainTuple(12, 15, 18)
@test chainzip(*, t1, t2, t3) == ChainTuple(28, 80, 162)

@test chainzip(
    +,
    ChainTuple(1, ChainTuple(ParallelTuple(2, ChainTuple(3)))),
    ChainTuple(4, ChainTuple(ParallelTuple(5, ChainTuple(6)))),
) == ChainTuple(5, ChainTuple(ParallelTuple(7, ChainTuple(9))))

@test_throws ErrorException chainzip(+, t1, ChainTuple(1, 2))
@test_throws ErrorException chainzip(+, t1, ParallelTuple(1, 2, 3))
