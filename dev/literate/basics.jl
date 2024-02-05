# # [Creating an LRP Analyzer](@id docs-lrp-basics)

#md # !!! note
#md #     This package is part the [Julia-XAI ecosystem](https://github.com/Julia-XAI).
#md #     For an introduction to the ecosystem, please refer to the
#md #     [*Getting started* guide](https://julia-xai.github.io/XAIDocs/).

# We start out by loading a small convolutional neural network:
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
# This model contains two chains: the convolutional layers and the fully connected layers.

# ## [Model preparation](@id docs-lrp-model-prep)

#md # !!! note "TLDR"
#md #
#md #     1. Use [`strip_softmax`](@ref) to strip the output softmax from your model.
#md #        Otherwise [model checks](@ref docs-lrp-model-checks) will fail.
#md #     1. Use [`canonize`](@ref) to fuse linear layers.
#md #     1. Don't just call `LRP(model)`, instead use a [`Composite`](@ref)
#md #        to apply LRP rules to your model.
#md #        Read [*Assigning rules to layers*](@ref docs-composites) for more information.
#md #     1. By default, `LRP` will call [`flatten_model`](@ref) to flatten your model.
#md #        This reduces computational overhead.

# ### [Stripping the output softmax](@id docs-lrp-strip-softmax)
# When using LRP, it is recommended to explain output logits instead of probabilities.
# This can be done by stripping the output softmax activation from the model
# using the [`strip_softmax`](@ref) function:
model = strip_softmax(model)

# If you don't remove the output softmax,
# [model checks](@ref docs-lrp-model-checks) will fail.

# ### [Canonizing the model](@id docs-lrp-canonization)
# LRP is not invariant to a model's implementation.
# Applying the [`GammaRule`](@ref) to two linear layers in a row will yield different results
# than first fusing the two layers into one linear layer and then applying the rule.
# This fusing is called "canonization" and can be done using the [`canonize`](@ref) function:
model_canonized = canonize(model)

# After canonization, the first `BatchNorm` layer has been fused into the preceding `Conv` layer.
# The second `BatchNorm` layer wasn't fused
# since its preceding `Conv` layer has a ReLU activation function.

# ### [Flattening the model](@id docs-lrp-flatten-model)
# RelevancePropagation.jl's LRP implementation supports nested Flux Chains and Parallel layers.
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
# For this purpose, RelevancePropagation.jl provides the function [`flatten_model`](@ref):
model_flat = flatten_model(model)

# This function is called by default when creating an LRP analyzer.
# Note that we pass the unflattened model to the analyzer, but `analyzer.model` is flattened:
analyzer = LRP(model)
analyzer.model

# If this flattening is not desired, it can be disabled
# by passing the keyword argument `flatten=false` to the `LRP` constructor.

# ## LRP rules
# The following examples will be run on a pre-trained LeNet-5 model:
using BSON

model = BSON.load("../model.bson", @__MODULE__)[:model] # load pre-trained LeNet-5 model

# We also load the MNIST dataset:
using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

# By default, the `LRP` constructor will assign the [`ZeroRule`](@ref) to all layers.
analyzer = LRP(model)

# This ana lyzer will return heatmaps that look identical to the `InputTimesGradient` analyzer
# from [ExplainableAI.jl](https://github.com/Julia-XAI/ExplainableAI.jl):

heatmap(input, analyzer)

# LRP's strength lies in assigning different rules to different layers,
# based on their functionality in the neural network[^1].
# RelevancePropagation.jl [implements many LRP rules out of the box](@ref api-lrp-rules),
# but it is also possible to [*implement custom rules*](@ref docs-custom-rules).
#
# To assign different rules to different layers,
# use one of the [composites presets](@ref api-composite-presets),
# or create your own composite, as described in
# [*Assigning rules to layers*](@ref docs-composites).

composite = EpsilonPlusFlat() # using composite preset EpsilonPlusFlat
#-
analyzer = LRP(model, composite)
#-
heatmap(input, analyzer)

# ## [Computing layerwise relevances](@id docs-lrp-layerwise)
# If you are interested in computing layerwise relevances,
# call `analyze` with an LRP analyzer and the keyword argument
# `layerwise_relevances=true`.
#
# The layerwise relevances can be accessed in the `extras` field
# of the returned `Explanation`:

expl = analyze(input, analyzer; layerwise_relevances=true)
expl.extras.layerwise_relevances

# Note that the layerwise relevances are only kept for layers in the outermost `Chain` of the model.
# Since we used a flattened model, we obtained all relevances.

# ## [Performance tips](@id docs-lrp-performance)
# ### [Using LRP with a GPU](@id gpu-docs)
# All LRP analyzers support GPU backends,
# building on top of [Flux.jl's GPU support](https://fluxml.ai/Flux.jl/stable/gpu/).
# Using a GPU only requires moving the input array and model weights to the GPU.
#
# For example, using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl):

# ```julia
# using CUDA, cuDNN
# using Flux
# using RelevancePropagation
#
# # move input array and model weights to GPU
# input = input |> gpu # or gpu(input)
# model = model |> gpu # or gpu(model)
#
# # analyzers don't require calling `gpu`
# analyzer = LRP(model)
#
# # explanations are computed on the GPU
# expl = analyze(input, analyzer)
# ```

# Some operations, like saving, require moving explanations back to the CPU.
# This can be done using Flux's `cpu` function:

# ```julia
# val = expl.val |> cpu # or cpu(expl.val)
#
# using BSON
# BSON.@save "explanation.bson" val
# ```
#
# ### Using LRP without a GPU
# Using Julia's package extension mechanism,
# RelevancePropagation.jl's LRP implementation can optionally make use of
# [Tullio.jl](https://github.com/mcabbott/Tullio.jl) and
# [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl)
# for faster LRP rules on dense layers.
#
# This only requires loading the packages before loading RelevancePropagation.jl:
# ```julia
# using LoopVectorization, Tullio
# using RelevancePropagation
# ```
#
# [^1]: G. Montavon et al., [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
