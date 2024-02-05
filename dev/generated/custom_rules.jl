using RelevancePropagation
using Flux
using MLDatasets
using ImageCore
using BSON

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

model = BSON.load("../model.bson", @__MODULE__)[:model] # load pre-trained LeNet-5 model

struct MyGammaRule <: AbstractLRPRule end

import RelevancePropagation: modify_parameters

modify_parameters(::MyGammaRule, param) = param + 0.25f0 * relu.(param)

rules = [
    ZPlusRule(),
    EpsilonRule(),
    MyGammaRule(), # our custom GammaRule
    EpsilonRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
]
analyzer = LRP(model, rules)
heatmap(input, analyzer)

rules = [
    ZPlusRule(),
    EpsilonRule(),
    GammaRule(), # XAI.jl's GammaRule
    EpsilonRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
]
analyzer = LRP(model, rules)
heatmap(input, analyzer)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
