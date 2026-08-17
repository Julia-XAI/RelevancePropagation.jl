# # [Custom LRP Rules](@id custom-rules)
# One of the design goals of RelevancePropagation.jl is to combine ease of use and
# extensibility for the purpose of research.

# This example will show you how to implement custom LRP rules.

#md # !!! note
#md #     This package is part the [Julia-XAI ecosystem](https://github.com/Julia-XAI)
#md #     and builds on the basics shown in the [*Getting started* guide](https://julia-xai.github.io/XAIDocs/).

# We start out by loading the same pre-trained LeNet-5 model and MNIST input data:
using RelevancePropagation
using VisionHeatmaps
using Lux
using MLDatasets
using ImageCore
using JLD2
using StableRNGs: StableRNG

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

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

# ## Implementing a custom rule
# ### Step 1: Define rule struct
# Let's define a rule that modifies the weights and biases of our layer on the forward pass.
# The rule has to be of supertype `AbstractLRPRule`.
struct MyGammaRule <: AbstractLRPRule end

# ### Step 2: Implement rule behavior
# It is then possible to dispatch on the following four utility functions
# with the rule type `MyCustomLRPRule` to define custom rules without writing boilerplate code.
#
# 1. [`modify_input(rule::MyGammaRule, input)`](@ref RelevancePropagation.modify_input)
# 1. [`modify_parameters(rule::MyGammaRule, parameter)`](@ref RelevancePropagation.modify_parameters)
# 1. [`modify_denominator(rule::MyGammaRule, denominator)`](@ref RelevancePropagation.modify_denominator)
# 1. [`is_compatible(rule::MyGammaRule, layer, ps)`](@ref RelevancePropagation.is_compatible)
#
# By default:
# 1. `modify_input` doesn't change the input
# 1. `modify_parameters` doesn't change the parameters
# 1. `modify_denominator` avoids division by zero by adding a small epsilon-term (`1.0f-9`)
# 1. `is_compatible` returns `true` if the layer's parameters `ps` have a `weight` entry
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
analyzer = LRP(model, ps, st, rules)

heatmap(input, analyzer) # using VisionHeatmaps.jl

# We just implemented our own version of the ``γ``-rule in 2 lines of code.
# The heatmap perfectly matches the pre-implemented `GammaRule`:
rules = [
    ZPlusRule(),
    EpsilonRule(),
    GammaRule(), # RelevancePropagation.jl's GammaRule
    EpsilonRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
]
analyzer = LRP(model, ps, st, rules)
heatmap(input, analyzer)

# ## Performance tips
# Make sure functions like `modify_parameters` don't promote the type of weights
# (e.g. from `Float32` to `Float64`).
# Rules whose `modify_parameters` (or `modify_weight` and `modify_bias`)
# is the identity get a fast path for free:
# the pre-activation cached by the Enzyme reverse pass is reused,
# skipping the modified forward pass entirely.

# ## [Advanced parameter modification](@id custom-rules-advanced)
# For more granular control over weights and biases,
# [`modify_weight`](@ref RelevancePropagation.modify_weight) and
# [`modify_bias`](@ref RelevancePropagation.modify_bias) can be used.
# These operate on the arrays in a layer's `ps` NamedTuple;
# rules never hold copies of model parameters —
# modified parameters are computed lazily via
# [`modify_params`](@ref RelevancePropagation.modify_params) on each call.
# Parameters without a `weight` entry are returned unmodified.
# To add compatibility checks between rule and layer types, extend
# [`is_compatible`](@ref RelevancePropagation.is_compatible).

#md # !!! warning "Extending modify_weight and modify_bias"
#md #
#md #     `modify_weight` and `modify_bias` overwrite the functionality of
#md #     `modify_parameters` for the implemented rule type, since they call
#md #     `modify_parameters` by default.
#md #
#md #     The default call structure looks as follows:
#md #     ```
#md #     ┌─────────────────────────────────────────┐
#md #     │              modify_params              │
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

# ## Advanced LRP rules
# To implement custom LRP rules that require more than `modify_parameters`, `modify_input`
# and `modify_denominator`, take a look at the [LRP developer documentation](@ref developer).
