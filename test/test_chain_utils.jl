# v3's ChainTuple/ParallelTuple/SkipConnectionTuple machinery is replaced by
# NamedTuples mirroring the Lux `ps`/`st` structure.
# - `chainmap`  → `map_layers`, returning nested NamedTuples keyed like `ps`
# - `chainzip`  → zipping NamedTuples along the model tree; its user-facing
#   behavior (rules zipped over layers, key-mismatch errors) is covered in
#   test_lrp.jl via `get_modified_layers`
# - `chainindices`/`ModelIndex` → `layer_indices` on `Functors.KeyPath`
using RelevancePropagation
using Test

using RelevancePropagation: map_layers, activation_fn, layer_indices, keypath_in
using Lux
using Functors: KeyPath

d1 = Dense(2 => 2, relu)
d2 = Dense(2 => 2, selu)
d3 = Dense(2 => 2, gelu)
d4 = Dense(2 => 2, celu)

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

# Test map_layers (was: chainmap).
# SkipConnection is an `AbstractLuxWrapperLayer`: its `ps`/`st` pass through to
# the wrapped layer directly, so it is transparent in the mapped NamedTuples.
@test map_layers(activation_fn, c1) == (; layer_1=relu)
@test map_layers(activation_fn, c2) == (; layer_1=relu, layer_2=selu)
@test map_layers(activation_fn, c3) ==
    (; layer_1=(; layer_1=relu, layer_2=relu), layer_2=selu)
@test map_layers(activation_fn, c4) ==
    (; layer_1=relu, layer_2=(; layer_1=selu, layer_2=selu))
@test map_layers(activation_fn, c5) ==
    (; layer_1=relu, layer_2=(; layer_1=selu, layer_2=selu), layer_3=gelu)
@test map_layers(activation_fn, c6) == (; layer_1=(; layer_1=relu, layer_2=relu))
@test map_layers(activation_fn, c7) == (;
    layer_1=relu,
    layer_2=(; layer_1=selu, layer_2=selu, layer_3=(; layer_1=gelu, layer_2=gelu)),
    layer_3=celu,
)
@test map_layers(activation_fn, c8) == (; layer_1=relu)
@test map_layers(activation_fn, c9) == (; layer_1=relu)
@test map_layers(activation_fn, c10) == (; layer_1=relu, layer_2=selu)
@test map_layers(activation_fn, c11) ==
    (; layer_1=relu, layer_2=(; layer_1=selu, layer_2=gelu), layer_3=celu)

# Layer indexing (was: chainindices/ModelIndex).
# `layer_indices` mirrors the model structure with KeyPath leaves addressing
# each layer like `ps`/`st`; SkipConnection stays transparent.
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
