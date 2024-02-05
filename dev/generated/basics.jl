using RelevancePropagation
using Flux

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16; pad=1),
        BatchNorm(16, relu),
        Conv((3, 3), 16 => 8, relu; pad=1),
        BatchNorm(8, relu),
    ),
    Chain(Flux.flatten, Dense(2048 => 512, relu), Dropout(0.5), Dense(512 => 100, softmax)),
);

model = strip_softmax(model)

model_canonized = canonize(model)

model_flat = flatten_model(model)

analyzer = LRP(model)
analyzer.model

using BSON

model = BSON.load("../model.bson", @__MODULE__)[:model] # load pre-trained LeNet-5 model

using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

analyzer = LRP(model)

heatmap(input, analyzer)

composite = EpsilonPlusFlat() # using composite preset EpsilonPlusFlat

analyzer = LRP(model, composite)

heatmap(input, analyzer)

expl = analyze(input, analyzer; layerwise_relevances=true)
expl.extras.layerwise_relevances

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
