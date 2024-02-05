```@meta
CurrentModule = RelevancePropagation
```

# RelevancePropagation.jl

Julia implementation of [Layerwise Relevance Propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) (LRP) 
and [Concept Relevance Propagation](https://www.nature.com/articles/s42256-023-00711-8) (CRP) 
for use with [Flux.jl](https://fluxml.ai) models.

!!! note
    This package is part the [Julia-XAI ecosystem](https://github.com/Julia-XAI).
    For an introduction to the ecosystem, please refer to the 
    [*Getting started* guide](https://julia-xai.github.io/XAIDocs/).

## Installation 
To install this package and its dependencies, open the Julia REPL and run 
```julia-repl
julia> ]add RelevancePropagation
```

## Index
### Manual
#### Basic Usage
```@contents
Pages = [
    "generated/basics.md",
    "generated/composites.md",
    "generated/crp.md",
]
Depth = 3
```
#### Advanced Usage
```@contents
Pages = [
    "generated/custom_layer.md",
    "generated/custom_rules.md",
    "developer.md",
]
Depth = 3
```

### API Reference
```@contents
Pages = ["api.md"]
Depth = 4
```
