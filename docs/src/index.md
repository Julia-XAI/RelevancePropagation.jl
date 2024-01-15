```@meta
CurrentModule = RelevancePropagation
```

# RelevancePropagation.jl

Julia implementation of [Layerwise Relevance Propagation][paper-lrp] (LRP) 
and [Concept Relevance Propagation][paper-crp] (CRP) 
for use with [Flux.jl](https://fluxml.ai) models.

This package is part of the [Julia-XAI ecosystem](https://github.com/Julia-XAI) and compatible with
[ExplainableAI.jl](https://github.com/Julia-XAI/ExplainableAI.jl).

## Installation 
To install this package and its dependencies, open the Julia REPL and run 
```julia-repl
julia> ]add RelevancePropagation
```

## Manual
```@contents
Pages = [
    "generated/basics.md",
    "generated/composites.md",
    "generated/custom_layer.md",
    "generated/custom_rules.md",
    "developer.md",
]
Depth = 3
```

## API reference
```@contents
Pages = ["api.md"]
Depth = 2
```