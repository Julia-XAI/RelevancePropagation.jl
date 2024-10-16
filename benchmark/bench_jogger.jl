using BenchmarkTools
using Flux
using RelevancePropagation
using RelevancePropagation: lrp!, modify_layer

on_CI = haskey(ENV, "GITHUB_ACTIONS")

T = Float32
input_size = (32, 32, 3, 1)
input = rand(T, input_size)

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; pad=1),
        Conv((3, 3), 16 => 16, relu; pad=1),
        MaxPool((2, 2)),
    ),
    Chain(
        Flux.flatten,
        Dense(1024 => 512, relu),         # 102_764_544 parameters
        Dropout(0.5),
        Dense(512 => 100, relu),
    ),
)
Flux.testmode!(model, true)

# Use one representative algorithm of each type
algs = Dict("LRP" => LRP, "LREpsilonPlusFlat" => model -> LRP(model, EpsilonPlusFlat()))

# Define benchmark
_alg(alg, model) = alg(model) # for use with @benchmarkable macro

suite = BenchmarkGroup()
suite["CNN"] = BenchmarkGroup([k for k in keys(algs)])
for (name, alg) in algs
    analyzer = alg(model)
    suite["CNN"][name] = BenchmarkGroup(["construct analyzer", "analyze"])
    suite["CNN"][name]["construct analyzer"] = @benchmarkable _alg($(alg), $(model))
    suite["CNN"][name]["analyze"] = @benchmarkable analyze($(input), $(analyzer))
end

# generate input for conv layers
insize = (32, 32, 3, 1)
in_dense = 64
out_dense = 10
aᵏ = rand(T, insize)

layers = Dict(
    "Conv"  => (Conv((3, 3), 3 => 2), aᵏ),
    "Dense" => (Dense(in_dense, out_dense, relu), randn(T, in_dense, 1)),
)
rules = Dict(
    "ZeroRule"      => ZeroRule(),
    "EpsilonRule"   => EpsilonRule(),
    "GammaRule"     => GammaRule(),
    "WSquareRule"   => WSquareRule(),
    "FlatRule"      => FlatRule(),
    "AlphaBetaRule" => AlphaBetaRule(),
    "ZPlusRule"     => ZPlusRule(),
    "ZBoxRule"      => ZBoxRule(zero(T), oneunit(T)),
)

layernames = String.(keys(layers))
rulenames  = String.(keys(rules))

suite["modify layer"] = BenchmarkGroup(rulenames)
suite["apply rule"]   = BenchmarkGroup(rulenames)
for rname in rulenames
    suite["modify layer"][rname] = BenchmarkGroup(layernames)
    suite["apply rule"][rname] = BenchmarkGroup(layernames)
end

for (lname, (layer, aᵏ)) in layers
    Rᵏ = similar(aᵏ)
    Rᵏ⁺¹ = layer(aᵏ)
    for (rname, rule) in rules
        modified_layer = modify_layer(rule, layer)
        suite["modify layer"][rname][lname] = @benchmarkable modify_layer($(rule), $(layer))
        suite["apply rule"][rname][lname] = @benchmarkable lrp!(
            $(Rᵏ), $(rule), $(layer), $(modified_layer), $(aᵏ), $(Rᵏ⁺¹)
        )
    end
end
