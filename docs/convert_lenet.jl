# One-time conversion of the docs' pre-trained Flux LeNet-5 (`src/model.bson`,
# removed from the repository alongside this conversion) to a Lux parameter
# NamedTuple (`src/model.jld2`, key `"ps"`).
#
# Run from this directory in a temporary environment containing
# Flux (≥ 0.16), BSON, Lux, JLD2 and StableRNGs. The resulting logits were
# verified to match the Flux model bit-exactly.
using Flux: Flux
using BSON: BSON
using Lux: Lux
using JLD2: JLD2
using Lux: Chain, Conv, MaxPool, FlattenLayer, Dense, relu
using StableRNGs: StableRNG

flux_model = BSON.load(joinpath(@__DIR__, "src", "model.bson"), Main)[:model]

lux_model = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    FlattenLayer(),
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
)
ps, st = Lux.setup(StableRNG(123), lux_model)

# Flux and Lux agree on convolution semantics and weight layout,
# so parameters can be transferred directly.
wb(l) = (; weight=l.weight, bias=copy(l.bias))
ps = (;
    layer_1=wb(flux_model[1]),
    layer_2=NamedTuple(),
    layer_3=wb(flux_model[3]),
    layer_4=NamedTuple(),
    layer_5=NamedTuple(),
    layer_6=wb(flux_model[6]),
    layer_7=wb(flux_model[7]),
    layer_8=wb(flux_model[8]),
)

# sanity: identical logits
x = rand(StableRNG(1), Float32, 28, 28, 1, 4)
y_flux = flux_model(x)
y_lux = first(lux_model(x, ps, st))
@assert y_flux ≈ y_lux
println("max abs deviation: ", maximum(abs, y_flux - y_lux))

# check ps shapes match a fresh Lux.setup (guards against layout drift)
ps_ref, _ = Lux.setup(StableRNG(123), lux_model)
for k in keys(ps_ref), p in keys(ps_ref[k])
    @assert size(ps_ref[k][p]) == size(ps[k][p]) "$k.$p size mismatch"
end

JLD2.jldsave(joinpath(@__DIR__, "src", "model.jld2"); ps=ps)
println("saved model.jld2")
