using RelevancePropagation: FrozenLayer, layer_pullback
using Test

using Lux
using StableRNGs: StableRNG
using Zygote: Zygote

x_dense = randn(StableRNG(1), Float32, 4, 2)
x_img = randn(StableRNG(2), Float32, 6, 6, 3, 2)

# Enzyme split-mode pullbacks are cross-checked against Zygote VJPs.
# `layer_pullback` covers the whole differentiation surface of LRP, including
# the nonlinear pass-through layers (pooling, testmode BatchNorm, activations).
LAYERS_1SEED = [
    ("Dense identity", Dense(4 => 3), x_dense),
    ("Dense relu", Dense(4 => 3, relu), x_dense),
    ("Dense gelu", Dense(4 => 3, gelu), x_dense),
    ("Dense no bias", Dense(4 => 3, relu; use_bias=false), x_dense),
    ("Scale relu", Scale(4, relu), x_dense),
    ("Conv", Conv((3, 3), 3 => 4), x_img),
    ("Conv relu", Conv((3, 3), 3 => 4, relu), x_img),
    ("Conv cross-correlation", Conv((3, 3), 3 => 4; cross_correlation=true), x_img),
    ("ConvTranspose", ConvTranspose((3, 3), 3 => 4), x_img),
    ("MaxPool", MaxPool((2, 2)), x_img),
    ("MeanPool", MeanPool((2, 2)), x_img),
    ("GlobalMaxPool", GlobalMaxPool(), x_img),
    ("GlobalMeanPool", GlobalMeanPool(), x_img),
    ("AdaptiveMaxPool", AdaptiveMaxPool((2, 2)), x_img),
    ("AdaptiveMeanPool", AdaptiveMeanPool((2, 2)), x_img),
    ("BatchNorm testmode", BatchNorm(3), x_img),
    ("BatchNorm relu testmode", BatchNorm(3, relu), x_img),
    ("LayerNorm", LayerNorm((6, 6, 3)), x_img),
    ("LayerNorm relu", LayerNorm((6, 6, 3), relu), x_img),
    ("LayerNorm no affine", LayerNorm((6, 6, 3); affine=false), x_img),
    ("FlattenLayer", FlattenLayer(), x_img),
    ("Dropout testmode", Dropout(0.5f0), x_img),
]

frozen(layer) = FrozenLayer(layer, Lux.setup(StableRNG(123), layer)...)

@testset "layer_pullback vs Zygote" begin
    for (name, layer, x) in LAYERS_1SEED
        @testset "$name" begin
            f = frozen(layer)
            z_ref, back_ref = Zygote.pullback(f, x)
            s = randn(StableRNG(17), Float32, size(z_ref)...)
            dx_ref = only(back_ref(s))

            z, back = layer_pullback(f, x)
            @test z ≈ z_ref
            @test back(s) ≈ dx_ref
        end
    end
end
