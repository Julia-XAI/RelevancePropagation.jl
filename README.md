# LayerwiseRelevancePropagation.jl

[![Build Status](https://github.com/Julia-XAI/LayerwiseRelevancePropagation.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Julia-XAI/LayerwiseRelevancePropagation.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Julia-XAI/LayerwiseRelevancePropagation.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Julia-XAI/LayerwiseRelevancePropagation.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Julia implementation of [Layerwise Relevance Propagation][paper-lrp] (LRP) 
and [Concept Relevance Propagation][paper-crp] (CRP) 
for use with [Flux.jl](https://fluxml.ai) models.

This package is part of the [Julia-XAI ecosystem](https://github.com/Julia-XAI) and compatible with
[ExplainableAI.jl](https://github.com/Julia-XAI/ExplainableAI.jl).

## Installation 
This package supports Julia ≥1.6. To install it, open the Julia REPL and run 
```julia-repl
julia> ]add LayerwiseRelevancePropagation
```

## Example
Let's use LRP to explain why an image of a castle gets classified as such 
using a pre-trained VGG16 model from [Metalhead.jl](https://github.com/FluxML/Metalhead.jl):

![][castle]

```julia
using LayerwiseRelevancePropagation
using Flux
using Metalhead                  # pre-trained vision models

# Load & prepare model
model = VGG(16, pretrain=true).layers
model = strip_softmax(model)
model = canonize(model)

# Load input
input = ...                      # input in WHCN format

# Run XAI method
composite = EpsilonPlusFlat()
analyzer = LRP(model, composite)
expl = analyze(input, analyzer)  # or: expl = analyzer(input)
heatmap(expl)                    # Show heatmap

```

We can also get an explanation for the activation of the output neuron 
corresponding to the "street sign" class by specifying the corresponding output neuron position `920`:

```julia
analyze(input, analyzer, 920) 
```

Heatmaps for all implemented analyzers are shown in the following table. 
Red color indicate regions of positive relevance towards the selected class, 
whereas regions in blue are of negative relevance.

| **Analyzer**                                  | **Heatmap for class "castle"** |**Heatmap for class "street sign"** |
|:--------------------------------------------- |:------------------------------:|:----------------------------------:|
| `LRP` with `EpsilonPlus` composite            | ![][castle-lrp-ep]             | ![][streetsign-lrp-ep]              |
| `LRP` with `EpsilonPlusFlat` composite        | ![][castle-lrp-epf]            | ![][streetsign-lrp-epf]            |
| `LRP` with `EpsilonAlpha2Beta1` composite     | ![][castle-lrp-eab]            | ![][streetsign-lrp-eab]            |
| `LRP` with `EpsilonAlpha2Beta1Flat` composite | ![][castle-lrp-eabf]           | ![][streetsign-lrp-eabf]           |
| `LRP` with `EpsilonGammaBox` composite        | ![][castle-lrp-egb]            | ![][streetsign-lrp-egb]            |
| `LRP`                                         | ![][castle-lrp]                | ![][streetsign-lrp]                |



## Acknowledgements
> Adrian Hill acknowledges support by the Federal Ministry of Education and Research (BMBF) 
> for the Berlin Institute for the Foundations of Learning and Data (BIFOLD) (01IS18037A).

[paper-lrp]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
[paper-crp]: https://www.nature.com/articles/s42256-023-00711-8


[castle]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg
[castle-lrp]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRP.png
[castle-lrp-egb]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonGammaBox.png
[castle-lrp-ep]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonPlus.png
[castle-lrp-epf]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonPlusFlat.png
[castle-lrp-eab]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonAlpha2Beta1.png
[castle-lrp-eabf]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle_LRPEpsilonAlpha2Beta1Flat.png
[streetsign-lrp]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRP.png
[streetsign-lrp-egb]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonGammaBox.png
[streetsign-lrp-ep]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonPlus.png
[streetsign-lrp-epf]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonPlusFlat.png
[streetsign-lrp-eab]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonAlpha2Beta1.png
[streetsign-lrp-eabf]: https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/streetsign_LRPEpsilonAlpha2Beta1Flat.png