# Layer indexing (was: chainindices/ModelIndex) used by composite machinery.
# `layer_indices` mirrors the model structure with KeyPath leaves addressing
# each layer like `ps`/`st`; SkipConnection stays transparent.
using RelevancePropagation
using Test

using RelevancePropagation: layer_indices, keypath_in
using Lux
using Functors: KeyPath

d1 = Dense(2 => 2, relu)
d2 = Dense(2 => 2, selu)
d3 = Dense(2 => 2, gelu)
d4 = Dense(2 => 2, celu)

c2 = Chain(d1, d2)
c3 = Chain(Chain(d1, d1), d2)
c7 = Chain(d1, Parallel(+, d2, d2, Chain(d3, d3)), d4)
c11 = Chain(d1, SkipConnection(Chain(d2, d3), +), d4)

@test layer_indices(c2) == (; layer_1=KeyPath(:layer_1), layer_2=KeyPath(:layer_2))
@test layer_indices(c3) == (;
    layer_1=(; layer_1=KeyPath(:layer_1, :layer_1), layer_2=KeyPath(:layer_1, :layer_2)),
    layer_2=KeyPath(:layer_2),
)
@test layer_indices(c7) == (;
    layer_1=KeyPath(:layer_1),
    layer_2=(;
        layer_1=KeyPath(:layer_2, :layer_1),
        layer_2=KeyPath(:layer_2, :layer_2),
        layer_3=(;
            layer_1=KeyPath(:layer_2, :layer_3, :layer_1),
            layer_2=KeyPath(:layer_2, :layer_3, :layer_2),
        ),
    ),
    layer_3=KeyPath(:layer_3),
)
@test layer_indices(c11) == (;
    layer_1=KeyPath(:layer_1),
    layer_2=(; layer_1=KeyPath(:layer_2, :layer_1), layer_2=KeyPath(:layer_2, :layer_2)),
    layer_3=KeyPath(:layer_3),
)

# Prefix-matching semantics used by LayerMap (was: `Base.in` on ModelIndex)
@test keypath_in(KeyPath(:layer_1, :layer_2), KeyPath(:layer_1))
@test keypath_in(KeyPath(:layer_1, :layer_2), KeyPath(:layer_1, :layer_2))
@test !keypath_in(KeyPath(:layer_1), KeyPath(:layer_1, :layer_2))
@test !keypath_in(KeyPath(:layer_1, :layer_2), KeyPath(:layer_2))
