using RelevancePropagation
using Flux

using BSON # hide
model = BSON.load("../model.bson", @__MODULE__)[:model] # load pre-trained LeNet-5 model

using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

composite = EpsilonPlusFlat()
lrp_analyzer = LRP(model, composite)

feature_layer = 3    # index of relevant layer in model
model[feature_layer] # show layer

features = TopNFeatures(5)

features = IndexedFeatures(1, 2, 10)

using VisionHeatmaps

analyzer = CRP(lrp_analyzer, feature_layer, features)
heatmap(input, analyzer)

x, y = MNIST(Float32, :test)[10:11]
batch = reshape(x, 28, 28, 1, :)

heatmap(batch, analyzer)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
