using BenchmarkTools
using Lux
using StableRNGs: StableRNG
using RelevancePropagation
using RelevancePropagation: propagate, node_forward, modify_params

T = Float32
input_size = (32, 32, 3, 1)
input = rand(StableRNG(1), T, input_size)

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
        FlattenLayer(), Dense(1024 => 512, relu), Dropout(0.5f0), Dense(512 => 100, relu)
    ),
)
# Composites with positional primitives apply to the flattened triple
# (v3 benchmarks used the flatten=true LRP default).
model, ps, st = flatten_model(model, Lux.setup(StableRNG(123), model)...)

# Use one representative algorithm of each type
algs = Dict(
    "LRP" => LRP,
    "LRPEpsilonPlusFlat" => (model, ps, st) -> LRP(model, ps, st, EpsilonPlusFlat()),
)

# Define benchmark
_alg(alg, model, ps, st) = alg(model, ps, st) # for use with @benchmarkable macro

suite = BenchmarkGroup()
suite["CNN"] = BenchmarkGroup([k for k in keys(algs)])
for (name, alg) in algs
    analyzer = alg(model, ps, st)
    analyze(input, analyzer) # fill the per-input-type thunk cache before timing
    suite["CNN"][name] = BenchmarkGroup(["construct analyzer", "analyze"])
    suite["CNN"][name]["construct analyzer"] = @benchmarkable _alg(
        $(alg), $(model), $(ps), $(st)
    )
    suite["CNN"][name]["analyze"] = @benchmarkable analyze($(input), $(analyzer))
end

# generate input for conv layers
insize = (32, 32, 3, 1)
in_dense = 64
out_dense = 10

conv = Conv((3, 3), 3 => 2)
dense = Dense(in_dense => out_dense, relu)
layers = Dict(
    "Conv" => (conv, Lux.setup(StableRNG(123), conv)..., rand(StableRNG(2), T, insize)),
    "Dense" =>
        (dense, Lux.setup(StableRNG(123), dense)..., randn(StableRNG(3), T, in_dense, 1)),
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

suite["modify params"] = BenchmarkGroup(rulenames)
suite["propagate"] = BenchmarkGroup(rulenames)
for rname in rulenames
    suite["modify params"][rname] = BenchmarkGroup(layernames)
    suite["propagate"][rname] = BenchmarkGroup(layernames)
end

for (lname, (layer, ps, st, aᵏ)) in layers
    # Seed the relevance with the layer output, like the rule tests do
    zᵏ, Rᵏ⁺¹ = node_forward(layer, aᵏ, ps, st)
    for (rname, rule) in rules
        suite["modify params"][rname][lname] = @benchmarkable modify_params($(rule), $(ps))
        suite["propagate"][rname][lname] = @benchmarkable propagate(
            $(rule), $(layer), $(aᵏ), $(zᵏ), $(ps), $(st), $(Rᵏ⁺¹)
        )
    end
end
