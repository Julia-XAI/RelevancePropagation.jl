using RelevancePropagation:
    input_vjp, prepare_vjp, seeded_pullback, remove_activation, PoolingLayer
using Test

using Lux
using LuxCore: LuxCore
using StableRNGs: StableRNG
using Zygote: Zygote

x_dense = randn(StableRNG(1), Float32, 4, 2)
x_img = randn(StableRNG(2), Float32, 6, 6, 3, 2)
x_img4 = randn(StableRNG(3), Float32, 6, 6, 4, 2)

# The nested Enzyme VJPs are cross-checked against Zygote.
# `seeded_pullback` covers the whole differentiation surface of LRP, including
# the nonlinear pass-through layers (pooling, testmode BatchNorm, activations).
LAYERS = [
    ("Dense identity", Dense(4 => 3), x_dense),
    ("Dense relu", Dense(4 => 3, relu), x_dense),
    ("Dense gelu", Dense(4 => 3, gelu), x_dense),
    ("Dense no bias", Dense(4 => 3, relu; use_bias=false), x_dense),
    ("Scale relu", Scale(4, relu), x_dense),
    ("Conv", Conv((3, 3), 3 => 4), x_img),
    ("Conv relu", Conv((3, 3), 3 => 4, relu), x_img),
    ("Conv cross-correlation", Conv((3, 3), 3 => 4; cross_correlation=true), x_img),
    ("Conv stride", Conv((3, 3), 3 => 4, relu; stride=2), x_img),
    ("Conv pad", Conv((3, 3), 3 => 4, relu; pad=1), x_img),
    ("Conv dilation", Conv((2, 2), 3 => 4, relu; dilation=2), x_img),
    ("Conv groups", Conv((3, 3), 4 => 4, relu; groups=2), x_img4),
    ("ConvTranspose", ConvTranspose((3, 3), 3 => 4), x_img),
    ("ConvTranspose stride", ConvTranspose((3, 3), 3 => 4, relu; stride=2), x_img),
    ("ConvTranspose outpad", ConvTranspose((3, 3), 3 => 4; stride=2, outpad=1), x_img),
    (
        "ConvTranspose cross-correlation",
        ConvTranspose((3, 3), 3 => 4; cross_correlation=true),
        x_img,
    ),
    ("MaxPool", MaxPool((2, 2)), x_img),
    ("MeanPool", MeanPool((2, 2)), x_img),
    ("GlobalMaxPool", GlobalMaxPool(), x_img),
    ("GlobalMeanPool", GlobalMeanPool(), x_img),
    ("AdaptiveMaxPool", AdaptiveMaxPool((2, 2)), x_img),
    ("AdaptiveMeanPool", AdaptiveMeanPool((2, 2)), x_img),
    ("BatchNorm testmode", BatchNorm(3), x_img),
    ("BatchNorm relu testmode", BatchNorm(3, relu), x_img),
    ("BatchNorm no affine", BatchNorm(3; affine=false), x_img),
    ("BatchNorm no track_stats", BatchNorm(3; track_stats=false), x_img),
    ("LayerNorm", LayerNorm((6, 6, 3)), x_img),
    ("LayerNorm relu", LayerNorm((6, 6, 3), relu), x_img),
    ("LayerNorm no affine", LayerNorm((6, 6, 3); affine=false), x_img),
    ("FlattenLayer", FlattenLayer(), x_img),
    ("Dropout testmode", Dropout(0.5f0), x_img),
]

setup_testmode(layer) = (l=Lux.setup(StableRNG(123), layer); (l[1], Lux.testmode(l[2])))

@testset "seeded_pullback vs Zygote" begin
    for (name, layer, x) in LAYERS
        @testset "$name" begin
            ps, st = setup_testmode(layer)
            z_ref, back_ref = Zygote.pullback(
                x -> first(LuxCore.apply(layer, x, ps, st)), x
            )
            s = randn(StableRNG(17), Float32, size(z_ref)...)
            dx_ref = only(back_ref(s))
            @test seeded_pullback(layer, x, ps, st, s) ≈ dx_ref
        end
    end
end

# `prepare_vjp` is the two-phase primitive the rule bodies consume: both the
# primal it returns (the rule's z̃) and its single-use pullback must agree
# with the plain forward pass and Zygote — on the fast paths and on the
# split-mode Enzyme fallback (the layers with activations) alike.
@testset "prepare_vjp vs Zygote" begin
    for (name, layer, x) in LAYERS
        @testset "$name" begin
            ps, st = setup_testmode(layer)
            z_ref, back_ref = Zygote.pullback(
                x -> first(LuxCore.apply(layer, x, ps, st)), x
            )
            s = randn(StableRNG(17), Float32, size(z_ref)...)
            dx_ref = only(back_ref(s))
            z̃, pullback = prepare_vjp(layer, x, ps, st)
            @test z̃ ≈ z_ref
            @test pullback(s) ≈ dx_ref
        end
    end
end

# The hand-written fast-path VJPs must agree with the nested-AD fallback on
# the activation-stripped layers `propagate` uses them on
# (the wrap-time split guarantees rules only see affine layers).
@testset "input_vjp fast paths vs nested AD" begin
    for (name, layer, x) in LAYERS
        f = remove_activation(layer)
        f isa Union{Dense,Scale,Conv,ConvTranspose,PoolingLayer,BatchNorm} || continue
        @testset "$name" begin
            ps, st = setup_testmode(layer)
            z = first(LuxCore.apply(f, x, ps, st))
            s = randn(StableRNG(17), Float32, size(z)...)
            @test input_vjp(f, x, ps, st, s) ≈ seeded_pullback(f, x, ps, st, s)
        end
    end
end
