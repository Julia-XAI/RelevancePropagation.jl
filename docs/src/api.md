# API Reference
## Basic API
All methods in RelevancePropagation.jl work by calling `analyze` on an input and an analyzer:
```@docs
analyze
Explanation
heatmap
```

## LRP analyzer
```@docs
LRP
```

### Model preparation
```@docs
strip_softmax
canonize
flatten_model
```

## [LRP rules](@id api-lrp-rules)

!!! note "Notation"
    We use the following notation for LRP rules: 

    *  $W$ is the weight matrix of the layer
    *  $b$ is the bias vector of the layer
    *  $a^k$ is the activation vector at the input of layer $k$
    *  $a^{k+1}$ is the activation vector at the output of layer $k$
    *  $R^k$ is the relevance vector at the input of layer $k$
    *  $R^{k+1}$ is the relevance vector at the output of layer $k$
    *  $\rho$ is a function that modifies parameters (what we call [`modify_parameters`](@ref docs-custom-rules-impl))
    *  $\epsilon$ is a small positive constant to avoid division by zero

    Subscript characters are used to index vectors and matrices 
    (e.g. $b_i$ is the $i$-th entry of the bias vector), 
    while the superscripts $^k$ and $^{k+1}$ indicate the relative positions 
    of activations $a$ and relevances $R$ in the model.
    For any $k$, $a^k$ and $R^k$ have the same shape. 

    Note that all terms in the following equations are scalar value,
    which removes the need to differentiate between matrix and element-wise operations.
    For more information, refer to the [developer documentation](@ref lrp-dev-docs).

```@docs
ZeroRule
EpsilonRule
GammaRule
WSquareRule
FlatRule
AlphaBetaRule
ZPlusRule
ZBoxRule
PassRule
GeneralizedGammaRule
LayerNormRule
```

## Composites
### Applying composites
```@docs
Composite
lrp_rules
```

### [Composite primitives](@id api-composite-primitives)
#### Mapping layers to rules
Composite primitives that apply a single rule:
```@docs
LayerMap
GlobalMap
RangeMap
FirstLayerMap
LastLayerMap
```

To apply `LayerMap` to nested Flux Chains or `Parallel` layers, 
make use of `show_layer_indices`:
```@docs
show_layer_indices
```

#### Mapping layers to rules based on type
Composite primitives that apply rules based on the layer type:
```@docs
GlobalTypeMap
RangeTypeMap
FirstLayerTypeMap
LastLayerTypeMap
FirstNTypeMap
```

#### Union types for composites
The following exported union types types can be used to define TypeMaps:
```@docs
ConvLayer
PoolingLayer
DropoutLayer
ReshapingLayer
NormalizationLayer
```

### [Composite presets](@id api-composite-presets)
```@docs
EpsilonGammaBox
EpsilonPlus
EpsilonAlpha2Beta1
EpsilonPlusFlat
EpsilonAlpha2Beta1Flat
```

### Manual rule assignment
For [manual rule assignment](@ref docs-composites-manual), use `ChainTuple`, 
`ParallelTuple` and `SkipConnectionTuple`, matching the model structure:
```@docs
ChainTuple
ParallelTuple
SkipConnectionTuple
```

## Custom rules 
These utilities can be used to define custom rules without writing boilerplate code.
To extend these functions, explicitly `import` them: 
```@docs
RelevancePropagation.modify_input
RelevancePropagation.modify_denominator
RelevancePropagation.modify_parameters
RelevancePropagation.modify_weight
RelevancePropagation.modify_bias
RelevancePropagation.modify_layer
RelevancePropagation.is_compatible
```
Compatibility settings:
```@docs
LRP_CONFIG.supports_layer
LRP_CONFIG.supports_activation
```

## CRP
```@docs
CRP
TopNFeatures
IndexedFeatures
```

# Index
```@index
```
