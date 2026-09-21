# v3's ChainTuple/ParallelTuple/SkipConnectionTuple machinery is replaced by
# NamedTuples mirroring the Lux `ps`/`st` structure.
# - `chainmap`  → `map_layers`, returning nested NamedTuples keyed like `ps`
# - `chainzip`  → zipping NamedTuples along the model tree; its user-facing
#   behavior (rules zipped over layers, key-mismatch errors) is covered in
#   test_lrp.jl via `wrap_rules`/`check_rule_compat`
# - `chainindices`/`ModelIndex` → `layer_indices` on `Functors.KeyPath`,
#   tested RP-side in test_layer_indices.jl
using Test

using RelevancePropagation.ModelSurgeon:
    map_layers, chainall, first_element, last_element, map_triple, activation_fn
using Lux
using LuxCore: AbstractLuxWrapperLayer
using Functors: KeyPath
using StableRNGs: StableRNG

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

#=============#
# map_triple  #
#=============#

# map_triple jointly walks the `(model, ps, st)` triple, applying
# `f(layer, ps, st, kp)` at the leaves. KeyPaths follow the `ps`/`st`
# structure: SkipConnection is transparent and adds no key.
triple(model) = (model, Lux.setup(StableRNG(123), model)...)

let (model, ps, st) = triple(
        Chain(d1, SkipConnection(Chain(d2, d2), +), Parallel(+, d3, d3))
    )
    kps = KeyPath[]
    mapped_model, mapped_ps, mapped_st = map_triple(model, ps, st) do l, p, s, kp
        push!(kps, kp)
        l, (; weight=zero(p.weight), bias=zero(p.bias)), s
    end
    # Tied layers (d2, d3 occur twice) are visited once per occurrence,
    # since their `ps`/`st` entries are per-occurrence.
    @test kps == [
        KeyPath(:layer_1),
        KeyPath(:layer_2, :layer_1),
        KeyPath(:layer_2, :layer_2),
        KeyPath(:layer_3, :layer_1),
        KeyPath(:layer_3, :layer_2),
    ]
    # Containers are rebuilt around the mapped leaves
    @test mapped_model == model
    @test keys(mapped_ps) == keys(ps)
    @test all(iszero, mapped_ps.layer_2.layer_1.weight)
    @test all(iszero, mapped_ps.layer_3.layer_2.weight)
    @test mapped_st == st
end

# `exclude` marks additional layers as leaves: the excluded subtree is passed
# to `f` whole instead of being descended.
let (model, ps, st) = triple(c5) # Chain(d1, Chain(d2, d2), d3)
    seen = []
    map_triple(model, ps, st; exclude=(kp, l) -> kp == KeyPath(:layer_2)) do l, p, s, kp
        push!(seen, kp => l)
        l, p, s
    end
    @test first.(seen) == [KeyPath(:layer_1), KeyPath(:layer_2), KeyPath(:layer_3)]
    @test seen[2].second isa Chain
end

# A leaf `f` may return a replacement subtree, e.g. a `Chain` of two layers.
let (model, ps, st) = triple(Chain(d1, d2))
    split_model, split_ps, split_st = map_triple(model, ps, st) do l, p, s, kp
        Chain(l, NoOpLayer()),
        (; layer_1=p, layer_2=NamedTuple()),
        (; layer_1=s, layer_2=NamedTuple())
    end
    @test split_model == Chain(Chain(d1, NoOpLayer()), Chain(d2, NoOpLayer()))
    @test split_ps.layer_1.layer_1 == ps.layer_1
    @test split_ps.layer_2.layer_1 == ps.layer_2
end

# Unknown wrapper layers are leaves by default (the `AbstractLuxWrapperLayer`
# trait only guarantees `ps`/`st` transparency, not application transparency);
# `unwrap` opts them into transparent descent, rebuilding the wrapper.
struct MapWrapper{L} <: AbstractLuxWrapperLayer{:inner}
    inner::L
end

let (model, ps, st) = triple(Chain(MapWrapper(Chain(d1, d2)), d3))
    # Default: the wrapper is a leaf
    leaves = []
    map_triple(model, ps, st) do l, p, s, kp
        push!(leaves, l)
        l, p, s
    end
    @test length(leaves) == 2
    @test leaves[1] isa MapWrapper

    # Opted in: descended transparently (no extra KeyPath key), then rebuilt
    kps = KeyPath[]
    mapped_model, mapped_ps, _ = map_triple(
        model, ps, st; unwrap=Base.Fix2(isa, MapWrapper)
    ) do l, p, s, kp
        push!(kps, kp)
        l, (; weight=zero(p.weight), bias=zero(p.bias)), s
    end
    @test kps ==
        [KeyPath(:layer_1, :layer_1), KeyPath(:layer_1, :layer_2), KeyPath(:layer_2)]
    @test mapped_model.layers.layer_1 isa MapWrapper
    @test all(iszero, mapped_ps.layer_1.layer_2.weight)
end
