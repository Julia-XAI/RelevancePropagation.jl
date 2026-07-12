# Restored from v3 and adapted to the v4 Lux port (see PLAN.md):
# - rules are NamedTuples mirroring the model's `ps`/`st` instead of ChainTuples
# - `LayerMap` addresses layers via `Functors.KeyPath` instead of `ModelIndex`
# - v3 used Metalhead's VGG11; rule assignment only depends on layer types and
#   positions, so slim VGG11-shaped Lux models are used instead. (Boltz's Lux
#   VGG arrives with the docs port in phase 7.)
# - v3 tested composites on `flatten_model(model)`; the equivalent flat model
#   is constructed directly here until `flatten_model` lands in phase 5.
using RelevancePropagation
using Test
using ReferenceTests

using Lux
using Functors: KeyPath
using StableRNGs: StableRNG

# VGG11-shaped models with slim channels: Chain(features, classifier) like
# Metalhead's `VGG(11).layers`, and its 19-layer flat equivalent.
features = Chain(
    Conv((3, 3), 1 => 2, relu; pad=1),
    MaxPool((2, 2)),
    Conv((3, 3), 2 => 3, relu; pad=1),
    MaxPool((2, 2)),
    Conv((3, 3), 3 => 4, relu; pad=1),
    Conv((3, 3), 4 => 4, relu; pad=1),
    MaxPool((2, 2)),
    Conv((3, 3), 4 => 5, relu; pad=1),
    Conv((3, 3), 5 => 5, relu; pad=1),
    MaxPool((2, 2)),
    Conv((3, 3), 5 => 5, relu; pad=1),
    Conv((3, 3), 5 => 5, relu; pad=1),
    MaxPool((2, 2)),
)
classifier = Chain(
    FlattenLayer(),
    Dense(5 => 8, relu),
    Dropout(0.5f0),
    Dense(8 => 8, relu),
    Dropout(0.5f0),
    Dense(8 => 10),
)
model = Chain(features, classifier)
model_flat = Chain(features.layers..., classifier.layers...)

# This composite is non-sensical, but covers many composite primitives
composite1 = Composite(
    ZeroRule(), # default rule
    GlobalMap(PassRule()), # override default rule
    GlobalTypeMap(
        ConvLayer    => AlphaBetaRule(2.0f0, 1.0f0),
        Dense        => EpsilonRule(1.0f-6),
        PoolingLayer => EpsilonRule(1.0f-6),
    ),
    FirstNTypeMap(7, Conv => FlatRule()),
    RangeTypeMap(4:10, PoolingLayer => EpsilonRule(1.0f-5)),
    LayerMap(9, AlphaBetaRule(1.0f0, 0.0f0)),
    FirstLayerMap(ZBoxRule(-3.0f0, 3.0f0)),
    RangeMap(18:19, ZeroRule()),
    LastLayerMap(PassRule()),
)
rules1 = lrp_rules(model_flat, composite1)
@test rules1 == (;
    layer_1=ZBoxRule(-3.0f0, 3.0f0),
    layer_2=EpsilonRule(1.0f-6),
    layer_3=FlatRule(),
    layer_4=EpsilonRule(1.0f-5),
    layer_5=FlatRule(),
    layer_6=FlatRule(),
    layer_7=EpsilonRule(1.0f-5),
    layer_8=AlphaBetaRule(2.0f0, 1.0f0),
    layer_9=AlphaBetaRule(1.0f0, 0.0f0),
    layer_10=EpsilonRule(1.0f-5),
    layer_11=AlphaBetaRule(2.0f0, 1.0f0),
    layer_12=AlphaBetaRule(2.0f0, 1.0f0),
    layer_13=EpsilonRule(1.0f-6),
    layer_14=PassRule(),
    layer_15=EpsilonRule(1.0f-6),
    layer_16=PassRule(),
    layer_17=EpsilonRule(1.0f-6),
    layer_18=ZeroRule(),
    layer_19=PassRule(),
)

model2 = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    FlattenLayer(),
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
)
composite2 = Composite(
    LastLayerTypeMap(Dense => EpsilonRule(2.0f-5), Conv => EpsilonRule(2.0f-4)),
    FirstLayerTypeMap(
        Dense => AlphaBetaRule(1.0f0, 0.0f0), Conv => AlphaBetaRule(2.0f0, 1.0f0)
    ),
)
rules2 = lrp_rules(model2, composite2)
@test rules2 == (;
    layer_1=AlphaBetaRule(2.0f0, 1.0f0),
    layer_2=ZeroRule(),
    layer_3=ZeroRule(),
    layer_4=ZeroRule(),
    layer_5=ZeroRule(),
    layer_6=ZeroRule(),
    layer_7=ZeroRule(),
    layer_8=EpsilonRule(2.0f-5),
)

composite3 = Composite(
    GlobalTypeMap(
        ConvLayer      => ZPlusRule(),
        Dense          => EpsilonRule(),
        DropoutLayer   => PassRule(),
        ReshapingLayer => PassRule(),
    ),
    FirstLayerTypeMap(ConvLayer => FlatRule(), Dense => FlatRule()),
    LastLayerMap(EpsilonRule(1.0f-5)),
)
rules3 = lrp_rules(model, composite3)
@test rules3 == (;
    layer_1=(;
        layer_1=FlatRule(),
        layer_2=ZeroRule(),
        layer_3=ZPlusRule(),
        layer_4=ZeroRule(),
        layer_5=ZPlusRule(),
        layer_6=ZPlusRule(),
        layer_7=ZeroRule(),
        layer_8=ZPlusRule(),
        layer_9=ZPlusRule(),
        layer_10=ZeroRule(),
        layer_11=ZPlusRule(),
        layer_12=ZPlusRule(),
        layer_13=ZeroRule(),
    ),
    layer_2=(;
        layer_1=PassRule(),
        layer_2=EpsilonRule(),
        layer_3=PassRule(),
        layer_4=EpsilonRule(),
        layer_5=PassRule(),
        layer_6=EpsilonRule(1.0f-5),
    ),
)

# LayerMap addresses nested layers via KeyPath (integer/tuple conveniences map
# to Lux's default `layer_i` naming) and matches all layers below the path.
model4 = Chain(Dense(2 => 2), Chain(Dense(2 => 2), Dense(2 => 2)), Dense(2 => 2))
composite4 = Composite(
    LayerMap((2, 1), EpsilonRule()), LayerMap(KeyPath(:layer_3), GammaRule())
)
@test lrp_rules(model4, composite4) == (;
    layer_1=ZeroRule(),
    layer_2=(; layer_1=EpsilonRule(), layer_2=ZeroRule()),
    layer_3=GammaRule(),
)
composite5 = Composite(LayerMap(2, EpsilonRule())) # prefix matches the whole sub-chain
@test lrp_rules(model4, composite5) == (;
    layer_1=ZeroRule(),
    layer_2=(; layer_1=EpsilonRule(), layer_2=EpsilonRule()),
    layer_3=ZeroRule(),
)

# Bare functions in a Chain are wrapped in `WrappedFunction`;
# type maps match the wrapped function itself.
model6 = Chain(Dense(2 => 2, relu), identity)
composite6 = Composite(GlobalTypeMap(typeof(identity) => PassRule(), Dense => EpsilonRule()))
@test lrp_rules(model6, composite6) == (; layer_1=EpsilonRule(), layer_2=PassRule())

# Show reference tests require show.jl and the default composites from
# composite_presets.jl, which land later in phase 4 (see PLAN.md).
if !isdefined(RelevancePropagation, :EpsilonGammaBox)
    @test_skip false
else
    DEFAULT_COMPOSITES = Dict(
        "EpsilonGammaBox"        => EpsilonGammaBox(-3.0f0, 3.0f0),
        "EpsilonPlus"            => EpsilonPlus(),
        "EpsilonAlpha2Beta1"     => EpsilonAlpha2Beta1(),
        "EpsilonPlusFlat"        => EpsilonPlusFlat(),
        "EpsilonAlpha2Beta1Flat" => EpsilonAlpha2Beta1Flat(),
    )
    for (name, c) in DEFAULT_COMPOSITES
        @test_reference "references/show/$name.txt" repr("text/plain", c)
    end

    @test_reference "references/show/show_layer_indices.txt" repr(
        "text/plain", show_layer_indices(model)
    )

    @test_reference "references/show/composite1.txt" repr("text/plain", composite1)
    @test_reference "references/show/composite2.txt" repr("text/plain", composite2)

    # Analyzer show tests on the slim VGG11 models
    ps_flat, st_flat = Lux.setup(StableRNG(123), model_flat)
    analyzer1 = LRP(model_flat, ps_flat, st_flat, composite1)
    @test analyzer1.rules == rules1
    @test_reference "references/show/lrp1.txt" repr("text/plain", analyzer1)

    ps2, st2 = Lux.setup(StableRNG(123), model2)
    analyzer2 = LRP(model2, ps2, st2, composite2)
    @test analyzer2.rules == rules2
    @test_reference "references/show/lrp2.txt" repr("text/plain", analyzer2)

    ps, st = Lux.setup(StableRNG(123), model)
    analyzer3 = LRP(model, ps, st, composite3)
    @test analyzer3.rules == rules3
    @test_reference "references/show/lrp3.txt" repr("text/plain", analyzer3)
end
