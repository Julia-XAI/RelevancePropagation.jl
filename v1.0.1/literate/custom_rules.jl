# # [Custom LRP Rules](@id docs-custom-rules)
# One of the design goals of RelevancePropagation.jl is to combine ease of use and
# extensibility for the purpose of research.

# This example will show you how to implement custom LRP rules.

#md # !!! note
#md #     This package is part the [Julia-XAI ecosystem](https://github.com/Julia-XAI)
#md #     and builds on the basics shown in the [*Getting started* guide](https://julia-xai.github.io/XAIDocs/).

# We start out by loading the same pre-trained LeNet-5 model and MNIST input data:
using RelevancePropagation
using Flux
using MLDatasets
using ImageCore
using BSON

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

model = BSON.load("../model.bson", @__MODULE__)[:model] # load pre-trained LeNet-5 model

# ## Implementing a custom rule
# ### Step 1: Define rule struct
# Let's define a rule that modifies the weights and biases of our layer on the forward pass.
# The rule has to be of supertype `AbstractLRPRule`.
struct MyGammaRule <: AbstractLRPRule end

# ### [Step 2: Implement rule behavior](@id docs-custom-rules-impl)
# It is then possible to dispatch on the following four utility functions
# with the rule type `MyCustomLRPRule` to define custom rules without writing boilerplate code.
#
# 1. [`modify_input(rule::MyGammaRule, input)`](@ref RelevancePropagation.modify_input)
# 1. [`modify_parameters(rule::MyGammaRule, parameter)`](@ref RelevancePropagation.modify_parameters)
# 1. [`modify_denominator(rule::MyGammaRule, denominator)`](@ref RelevancePropagation.modify_denominator)
# 1. [`is_compatible(rule::MyGammaRule, layer)`](@ref RelevancePropagation.is_compatible)
#
# By default:
# 1. `modify_input` doesn't change the input
# 1. `modify_parameters` doesn't change the parameters
# 1. `modify_denominator` avoids division by zero by adding a small epsilon-term (`1.0f-9`)
# 1. `is_compatible` returns `true` if a layer has fields `weight` and `bias`
#
# To extend internal functions, import them explicitly:
import RelevancePropagation: modify_parameters

modify_parameters(::MyGammaRule, param) = param + 0.25f0 * relu.(param)

# Note that we didn't implement three of the four functions.
# This is because the defaults are sufficient to implement the `GammaRule`.

# ### Step 3: Use rule in LRP analyzer
# We can directly use our rule to make an analyzer!
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

# We just implemented our own version of the ``γ``-rule in 2 lines of code.
# The heatmap perfectly matches the pre-implemented `GammaRule`:
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

# ## Performance tips
# 1. Make sure functions like `modify_parameters` don't promote the type of weights
#    (e.g. from `Float32` to `Float64`).
# 2. If your rule `MyRule` doesn't modify weights or biases,
#    defining `modify_layer(::MyRule, layer) = nothing`
#    can provide reduce memory allocations and improve performance.

# ## [Advanced layer modification](@id docs-custom-rules-advanced)
# For more granular control over weights and biases,
# [`modify_weight`](@ref RelevancePropagation.modify_weight) and
# [`modify_bias`](@ref RelevancePropagation.modify_bias) can be used.
#
# If the layer doesn't use weights (`layer.weight`) and biases (`layer.bias`),
# RelevancePropagation provides a lower-level variant of
# [`modify_parameters`](@ref RelevancePropagation.modify_parameters) called
# [`modify_layer`](@ref RelevancePropagation.modify_layer).
# This function is expected to take a layer and return a new, modified layer.
# To add compatibility checks between rule and layer types, extend
# [`is_compatible`](@ref RelevancePropagation.is_compatible).

#md # !!! warning "Extending modify_layer"
#md #
#md #     Use of a custom function `modify_layer` will overwrite functionality of `modify_parameters`,
#md #     `modify_weight` and `modify_bias` for the implemented combination of rule and layer types.
#md #     This is due to the fact that internally, `modify_weight` and `modify_bias` are called
#md #     by the default implementation of `modify_layer`.
#md #     `modify_weight` and `modify_bias` in turn call `modify_parameters` by default.
#md #
#md #     The default call structure looks as follows:
#md #     ```
#md #     ┌─────────────────────────────────────────┐
#md #     │              modify_layer               │
#md #     └─────────┬─────────────────────┬─────────┘
#md #               │ calls               │ calls
#md #     ┌─────────▼─────────┐ ┌─────────▼─────────┐
#md #     │   modify_weight   │ │    modify_bias    │
#md #     └─────────┬─────────┘ └─────────┬─────────┘
#md #               │ calls               │ calls
#md #     ┌─────────▼─────────┐ ┌─────────▼─────────┐
#md #     │ modify_parameters │ │ modify_parameters │
#md #     └───────────────────┘ └───────────────────┘
#md #     ```
#md #
#md #     Therefore `modify_layer` should only be extended for a specific rule
#md #     and a specific layer type.

# ## Advanced LRP rules
# To implement custom LRP rules that require more than `modify_layer`, `modify_input`
# and `modify_denominator`, take a look at the [LRP developer documentation](@ref lrp-dev-docs).
