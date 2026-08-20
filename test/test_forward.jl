# Forward-pass equivalence of the wrap-time activation split.
#
# The split rests on one numerical assumption: applying the activation-
# stripped layer and then broadcasting σ reproduces the fused layer's output
# bit-for-bit. Lux routes activations through LuxLib's fused kernels, exactly
# the kind of code that could legally reassociate — this testset is the
# tripwire. If a fused LuxLib path ever breaks bit-equality, relaxing the
# assertion for a specific layer/activation pair is a deliberate, documented
# decision, not a default.
using RelevancePropagation
using Test

using RelevancePropagation: wrap_rules, map_layers
using Lux
using LuxCore: LuxCore
using StableRNGs: StableRNG

fwd(layer, x, ps, st) = first(LuxCore.apply(layer, x, ps, st))
setup_testmode(layer) = (l=Lux.setup(StableRNG(123), layer); (l[1], Lux.testmode(l[2])))

x_dense = randn(StableRNG(1), Float32, 4, 2)
x_img = randn(StableRNG(2), Float32, 6, 6, 3, 2)

# Every activation-bearing layer type the engine splits, plus activation-free
# layers whose nodes must stay plain pass-through wrappers.
SPLIT_LAYERS = [
    ("Dense identity", Dense(4 => 3), x_dense),
    ("Dense relu", Dense(4 => 3, relu), x_dense),
    ("Dense gelu", Dense(4 => 3, gelu), x_dense),
    ("Dense leakyrelu", Dense(4 => 3, leakyrelu), x_dense),
    ("Scale relu", Scale(4, relu), x_dense),
    ("Scale gelu", Scale(4, gelu), x_dense),
    ("Scale leakyrelu", Scale(4, leakyrelu), x_dense),
    ("Conv relu", Conv((3, 3), 3 => 4, relu), x_img),
    ("Conv gelu", Conv((3, 3), 3 => 4, gelu), x_img),
    ("Conv leakyrelu", Conv((3, 3), 3 => 4, leakyrelu), x_img),
    ("ConvTranspose relu", ConvTranspose((3, 3), 3 => 4, relu), x_img),
    ("BatchNorm relu testmode", BatchNorm(3, relu), x_img),
    ("LayerNorm relu", LayerNorm((6, 6, 3), relu), x_img),
    ("LayerNorm relu no affine", LayerNorm((6, 6, 3), relu; affine=false), x_img),
    ("MaxPool", MaxPool((2, 2)), x_img),
    ("FlattenLayer", FlattenLayer(), x_img),
    ("Dropout testmode", Dropout(0.5f0), x_img),
]

@testset "Single wrapped layers" begin
    for (name, layer, x) in SPLIT_LAYERS
        @testset "$name" begin
            ps, st = setup_testmode(layer)
            node = wrap_rules(layer, ZeroRule())
            @test fwd(node, x, ps, st) == fwd(layer, x, ps, st)
        end
    end
end

MODELS = [
    ("MLP", Chain(Dense(4 => 4, relu), Dense(4 => 4, gelu), Dense(4 => 2)), x_dense),
    (
        "CNN",
        Chain(
            Conv((3, 3), 3 => 4, relu),
            MaxPool((2, 2)),
            FlattenLayer(),
            Dense(16 => 2, relu),
        ),
        x_img,
    ),
    (
        "nested Chain",
        Chain(
            Dense(4 => 4, relu),
            Chain(Dense(4 => 4, relu), Dense(4 => 4, gelu)),
            Dense(4 => 2),
        ),
        x_dense,
    ),
    (
        "Parallel",
        Chain(
            Dense(4 => 4, relu),
            Parallel(+, Dense(4 => 4, relu), Dense(4 => 4)),
            Dense(4 => 2),
        ),
        x_dense,
    ),
    (
        "SkipConnection",
        Chain(Dense(4 => 4, relu), SkipConnection(Dense(4 => 4, relu), +), Dense(4 => 2)),
        x_dense,
    ),
    (
        "un-canonized BatchNorm",
        Chain(Dense(4 => 4, relu), BatchNorm(4, relu), Dense(4 => 2)),
        x_dense,
    ),
    ("LayerNorm", Chain(LayerNorm((4,), relu), Dense(4 => 2)), x_dense),
]

@testset "Wrapped models" begin
    for (name, model, x) in MODELS
        @testset "$name" begin
            ps, st = setup_testmode(model)
            wrapped = wrap_rules(model, map_layers(Returns(ZeroRule()), model))
            @test fwd(wrapped, x, ps, st) == fwd(model, x, ps, st)
        end
    end
end

@testset "Container as one unit" begin
    # A single rule on a sub-model is exempt from the split; its forward must
    # still match the fused model exactly.
    model = Chain(Dense(4 => 4, relu), Chain(Dense(4 => 4, relu), Dense(4 => 4, gelu)))
    ps, st = setup_testmode(model)
    wrapped = wrap_rules(model, (; layer_1=ZeroRule(), layer_2=ZeroRule()))
    @test fwd(wrapped, x_dense, ps, st) == fwd(model, x_dense, ps, st)
end

@testset "Explanation.output" begin
    # The engine's captured output falls under the same assertion, covering
    # the end-to-end reverse pass, the tap-inserted model and CRP's forward
    # loop.
    model = Chain(Dense(4 => 4, relu), Dense(4 => 4, gelu), Dense(4 => 2))
    ps, st = setup_testmode(model)
    y = fwd(model, x_dense, ps, st)

    analyzer = LRP(model, ps, st)
    @test analyze(x_dense, analyzer).output == y
    @test analyze(x_dense, analyzer; layerwise_relevances=true).output == y

    crp = CRP(LRP(model, ps, st), 1, TopNFeatures(1))
    @test analyze(x_dense, crp).output == y
end
