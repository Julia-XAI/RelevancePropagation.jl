# # Creating an LRP Analyzer

#md # !!! note
#md #     This package is part the [Julia-XAI ecosystem](https://github.com/Julia-XAI).
#md #     For an introduction to the ecosystem, please refer to the
#md #     [*Getting started* guide](https://julia-xai.github.io/XAIDocs/).

# We start out by loading a small convolutional neural network:
using RelevancePropagation
using Lux
using StableRNGs: StableRNG

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
    Chain(
        FlattenLayer(), Dense(2048 => 512, relu), Dropout(0.5), Dense(512 => 100, softmax)
    ),
);

# This model contains two chains: the convolutional layers and the fully connected layers.
#
# Lux models are stateless: parameters `ps` and states `st` live outside of the model
# and are initialized by `Lux.setup` with an explicit random number generator:
ps, st = Lux.setup(StableRNG(123), model);

# ## Model preparation

#md # !!! note "TLDR"
#md #
#md #     1. Use [`strip_softmax`](@ref) to strip the output softmax from your model.
#md #        Otherwise [model checks](@ref model-checks) will fail.
#md #     1. Use [`canonize`](@ref) to fuse linear layers.
#md #     1. Don't just call `LRP(model, ps, st)`, instead use a [`Composite`](@ref)
#md #        to apply LRP rules to your model.
#md #        Read [*Assigning rules to layers*](@ref composites) for more information.
#md #     1. Use [`flatten_model`](@ref) to flatten your model.
#md #        This reduces computational overhead.

# ### Stripping the output softmax
# When using LRP, it is recommended to explain output logits instead of probabilities.
# This can be done by stripping the output softmax activation from the model
# using the [`strip_softmax`](@ref) function:
model = strip_softmax(model)

# Since activation functions are part of the layer configuration in Lux,
# `strip_softmax` only transforms the model; `ps` and `st` remain valid.
#
# If you don't remove the output softmax,
# [model checks](@ref model-checks) will fail.

# ### [Model canonization](@id canonization)
# LRP is not invariant to a model's implementation.
# Applying the [`GammaRule`](@ref) to two linear layers in a row will yield different results
# than first fusing the two layers into one linear layer and then applying the rule.
# This fusing is called "canonization" and can be done using the [`canonize`](@ref) function.
#
# Since fusing layers changes the structure of the parameters and states,
# `canonize` transforms the entire Lux triple `(model, ps, st)`:
model_canonized, ps_canonized, st_canonized = canonize(model, ps, st);
model_canonized

# After canonization, the first `BatchNorm` layer has been fused into the preceding `Conv` layer.
# The second `BatchNorm` layer wasn't fused
# since its preceding `Conv` layer has a ReLU activation function.

# ### [Flattening the model](@id flatten-model)
# RelevancePropagation.jl's LRP implementation supports nested Lux Chains and Parallel layers.
# However, it is recommended to flatten the model before analyzing it.
#
# LRP is implemented by first running a forward pass through the model,
# keeping track of the intermediate activations, followed by a backward pass
# that computes the relevances.
#
# To keep the LRP implementation simple and maintainable,
# RelevancePropagation.jl does not pre-compute "nested" activations.
# Instead, for every internal chain, a new forward pass is run to compute activations.
#
# By "flattening" a model, this overhead can be avoided.
# For this purpose, RelevancePropagation.jl provides the function [`flatten_model`](@ref).
# Like `canonize`, it transforms the entire Lux triple,
# since splicing nested chains re-keys `ps` and `st`:
model_flat, ps_flat, st_flat = flatten_model(model, ps, st);
model_flat

#md # !!! note "Flattening is explicit"
#md #
#md #     Unlike previous (Flux-based) versions of this package,
#md #     the `LRP` constructor does not flatten models automatically,
#md #     since `ps` and `st` have to be transformed alongside the model.

# ## LRP rules
# The following examples will be run on a pre-trained LeNet-5 model:
using JLD2

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

# We also load the MNIST dataset:
using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

# By default, the `LRP` constructor will assign the [`ZeroRule`](@ref) to all layers.
analyzer = LRP(model, ps, st)

# This analyzer will return heatmaps that look identical to the `InputTimesGradient` analyzer
# from [ExplainableAI.jl](https://github.com/Julia-XAI/ExplainableAI.jl).
# We can visualize `Explanation`s by computing a `heatmap` using either
# [VisionHeatmaps.jl](https://julia-xai.github.io/XAIDocs/VisionHeatmaps/stable/) or
# [TextHeatmaps.jl](https://julia-xai.github.io/XAIDocs/TextHeatmaps/stable/),
# either for images or text, respectively.
using VisionHeatmaps

heatmap(input, analyzer)

# LRP's strength lies in assigning different rules to different layers,
# based on their functionality in the neural network[^1].
# RelevancePropagation.jl [implements many LRP rules out of the box](@ref rules),
# but it is also possible to [*implement custom rules*](@ref custom-rules).
#
# To assign different rules to different layers,
# use one of the [composites presets](@ref api-composite-presets),
# or create your own composite, as described in
# [*Assigning rules to layers*](@ref composites).

composite = EpsilonPlusFlat() # using composite preset EpsilonPlusFlat
#-
analyzer = LRP(model, ps, st, composite)
#-
heatmap(input, analyzer)

# ## Computing layerwise relevances
# If you are interested in computing layerwise relevances,
# call `analyze` with an LRP analyzer and the keyword argument
# `layerwise_relevances=true`.
#
# The layerwise relevances can be accessed in the `extras` field
# of the returned `Explanation`:

expl = analyze(input, analyzer; layerwise_relevances=true)
expl.extras.layerwise_relevances

# Note that the layerwise relevances are only kept for layers in the outermost `Chain` of the model.
# Since our LeNet-5 model is flat, we obtained all relevances.
#
# [^1]: G. Montavon et al., [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
