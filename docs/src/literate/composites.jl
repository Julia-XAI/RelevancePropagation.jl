# # [Assigning LRP Rules to Layers](@id composites)
# In this example, we will show how to assign LRP rules to specific layers.
# For this purpose, we first define a small VGG-like convolutional neural network:
using RelevancePropagation
using Lux
using StableRNGs: StableRNG

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; pad=1),
        Conv((3, 3), 16 => 16, relu; pad=1),
        MaxPool((2, 2)),
    ),
    Chain(FlattenLayer(), Dense(1024 => 512, relu), Dropout(0.5), Dense(512 => 100, relu)),
);
ps, st = Lux.setup(StableRNG(123), model);

# ## [Manually assigning rules](@id composites-manual)
# When creating an LRP analyzer, we can assign individual rules to each layer.
# As we can see above, our model is a `Chain` of two Lux `Chain`s.
# Using [`flatten_model`](@ref), we can flatten the model into a single `Chain`.
# Since this re-keys the parameters and states,
# `flatten_model` transforms the entire Lux triple:
model_flat, ps_flat, st_flat = flatten_model(model, ps, st);
model_flat

# This allows us to define an LRP analyzer using an array of rules
# matching the length of the flattened chain:
rules = [
    FlatRule(),
    ZPlusRule(),
    ZeroRule(),
    ZPlusRule(),
    ZPlusRule(),
    ZeroRule(),
    PassRule(),
    EpsilonRule(),
    PassRule(),
    EpsilonRule(),
];

# The `LRP` analyzer will show a summary of how layers and rules got matched:
LRP(model_flat, ps_flat, st_flat, rules)

# However, this approach only works for models that can be fully flattened.
# For nested models and models containing `Parallel` and `SkipConnection` layers,
# rules are assigned as a `NamedTuple` that mirrors the structure of the model —
# the same structure Lux uses for `ps` and `st`.
# The `layer_i` names are Lux's own keys for the children of a `Chain`
# (children live in a `NamedTuple`, not a vector),
# which is what allows rules to line up with the entries of `ps` and `st`:
rules = (;
    layer_1=(;
        layer_1=FlatRule(),
        layer_2=ZPlusRule(),
        layer_3=ZeroRule(),
        layer_4=ZPlusRule(),
        layer_5=ZPlusRule(),
        layer_6=ZeroRule(),
    ),
    layer_2=(;
        layer_1=PassRule(), layer_2=EpsilonRule(), layer_3=PassRule(), layer_4=EpsilonRule()
    ),
)

analyzer = LRP(model, ps, st, rules)

# ## Custom composites
# Instead of manually defining rules, we can also define a [`Composite`](@ref).
# A composite constructs a set of LRP-rules by sequentially applying the
# [composite primitives](@ref api-composite-primitives) it contains.
#
# To obtain the same set of rules as in the previous example, we can define
composite = Composite(
    GlobalTypeMap( # the following maps of layer types to LRP rules are applied globally
        Conv         => ZPlusRule(),   # apply ZPlusRule on all Conv layers
        Dense        => EpsilonRule(), # apply EpsilonRule on all Dense layers
        Dropout      => PassRule(),    # apply PassRule on all Dropout layers
        MaxPool      => ZeroRule(),    # apply ZeroRule on all MaxPool layers
        FlattenLayer => PassRule(),    # apply PassRule on all flatten layers
    ),
    FirstLayerMap( # the following rule is applied to the first layer
        FlatRule(),
    ),
);

#md # !!! note "Bare functions in Chains"
#md #
#md #     Lux wraps bare functions in `Chain`s in `WrappedFunction` layers.
#md #     Type maps match on the wrapped function,
#md #     so `typeof(myfunction) => rule` works as expected.

# We now construct an LRP analyzer from `composite`
analyzer = LRP(model, ps, st, composite)

# As you can see, this analyzer contains the same rules as our previous one.
# To compute rules for a model without creating an analyzer, use [`lrp_rules`](@ref):
lrp_rules(model, composite)

# ## Composite primitives
# The following [Composite primitives](@ref api-composite-primitives) can used to construct a [`Composite`](@ref).
#
# To apply a single rule, use:
# * [`LayerMap`](@ref) to apply a rule to a layer at a given index
# * [`GlobalMap`](@ref) to apply a rule to all layers
# * [`RangeMap`](@ref) to apply a rule to a positional range of layers
# * [`FirstLayerMap`](@ref) to apply a rule to the first layer
# * [`LastLayerMap`](@ref) to apply a rule to the last layer
#
# To apply a set of rules to layers based on their type, use:
# * [`GlobalTypeMap`](@ref) to apply a dictionary that maps layer types to LRP-rules
# * [`RangeTypeMap`](@ref) for a `TypeMap` on generalized ranges
# * [`FirstLayerTypeMap`](@ref) for a `TypeMap` on the first layer of a model
# * [`LastLayerTypeMap`](@ref) for a `TypeMap` on the last layer
# * [`FirstNTypeMap`](@ref) for a `TypeMap` on the first `n` layers
#
# Primitives are called sequentially in the order the `Composite` was created with
# and overwrite rules specified by previous primitives.
#
# Positional primitives ([`RangeMap`](@ref), [`RangeTypeMap`](@ref), [`FirstNTypeMap`](@ref))
# use *top-level* positions in the model:
# in our example model, position `1` refers to the entire first `Chain`
# of convolutional layers.

# ## Assigning a rule to a specific layer
# To assign a rule to a specific layer, we can use [`LayerMap`](@ref),
# which maps an LRP-rule to all layers in the model at the given index.
#
# To display indices, use the [`show_layer_indices`](@ref) helper function:
show_layer_indices(model)

# Indices are `KeyPath`s from [Functors.jl](https://github.com/FluxML/Functors.jl)
# addressing layers like the keys of the model's `ps` and `st`.
# Besides `KeyPath`s, `LayerMap` also accepts integers and tuples of integers,
# which are converted to `KeyPath`s:
# `LayerMap((1, 5), rule)` is equivalent to `LayerMap(KeyPath(:layer_1, :layer_5), rule)`.
#
# Let's demonstrate `LayerMap` by assigning a specific rule to the last `Conv` layer
# at index `(1, 5)`:
composite = Composite(LayerMap((1, 5), EpsilonRule()))

LRP(model, ps, st, composite)

# This approach also works with `Parallel` and `SkipConnection` layers.

# ## [Composite presets](@id composites-presets)
# RelevancePropagation.jl provides a set of default composites.
# A list of all implemented default composites can be found
# [in the API reference](@ref api-composite-presets),
# e.g. the [`EpsilonPlusFlat`](@ref) composite:
composite = EpsilonPlusFlat()
#-
analyzer = LRP(model, ps, st, composite)
