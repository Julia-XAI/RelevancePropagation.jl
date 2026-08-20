# # Concept Relevance Propagation
# In [*From attribution maps to human-understandable explanations through Concept Relevance Propagation*](https://www.nature.com/articles/s42256-023-00711-8) (CRP),
# Achtibat et al. propose the conditioning of LRP relevances on individual features of a model.

#md # !!! note
#md #     This package is part the [Julia-XAI ecosystem](https://github.com/Julia-XAI)
#md #     and builds on the basics shown in the [*Getting started* guide](https://julia-xai.github.io/XAIDocs/).

# We start out by loading the same pre-trained LeNet-5 model and MNIST input data:
using RelevancePropagation
using Lux
using JLD2
using StableRNGs: StableRNG

model = Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    FlattenLayer(),
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
);

ps = load("../model.jld2", "ps"); # load pre-trained parameters
_, st = Lux.setup(StableRNG(123), model); # all layers in LeNet-5 are stateless
#-
using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

# ## Step 1: Create LRP analyzer
# To create a CRP analyzer, first define an LRP analyzer with your desired rules:
composite = EpsilonPlusFlat()
lrp_analyzer = LRP(model, ps, st, composite)

# Note that CRP assumes flat models like our LeNet-5;
# use [`flatten_model`](@ref) on nested models.

# ## Step 2: Define concepts
# Then, specify the index of the layer on the outputs of which you want to condition the explanation.
# In this example, we are interested in the outputs of the last convolutional layer, layer 3:
feature_layer = 3          # index of relevant layer in model
model.layers.layer_3       # show layer

# Then, specify the concepts / features you are interested in.
# To automatically select the $n$ most relevant features, use [`TopNFeatures`](@ref).
#
# Note that for convolutional layers,
# a feature corresponds to an entire output channel of the layer.
features = TopNFeatures(5)

# To manually specify features, use [`IndexedFeatures`](@ref).
features = IndexedFeatures(1, 2, 10)

# ## Step 3: Use CRP analyzer
# We can now create a [`CRP`](@ref) analyzer
# and use it like any other analyzer from RelevancePropagation.jl:
using VisionHeatmaps

analyzer = CRP(lrp_analyzer, feature_layer, features)
heatmap(input, analyzer)

# ## Using CRP on input batches
# Note that `CRP` uses the batch dimension to return explanations.
# When using CRP on batches, the explanations are first sorted by features, then inputs,
# e.g. `[c1_i1, c1_i2, c2_i1, c2_i2, c3_i1, c3_i2]` in the following example:
x, y = MNIST(Float32, :test)[10:11]
batch = reshape(x, 28, 28, 1, :)

heatmap(batch, analyzer)
