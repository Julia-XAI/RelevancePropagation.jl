# RelevancePropagation.jl

| **Documentation** | **Build Status** |
|:----------------- |:---------------- |
| [![Stable documentation][docs-stab-img]][docs-stab-url] [![Latest documentation][docs-dev-img]][docs-dev-url] | [![Build Status][ci-img]][ci-url] [![Coverage Status][codecov-img]][codecov-url] [![Aqua QA][aqua-img]][aqua-url] |

Julia implementation of [Layerwise Relevance Propagation][paper-lrp] (LRP) 
and [Concept Relevance Propagation][paper-crp] (CRP) 
for use with [Flux.jl](https://fluxml.ai) models.

This package is part of the [Julia-XAI ecosystem](https://github.com/Julia-XAI) and compatible with
[ExplainableAI.jl](https://github.com/Julia-XAI/ExplainableAI.jl).

## Installation 
This package supports Julia ≥1.10. To install it, open the Julia REPL and run 
```julia-repl
julia> ]add RelevancePropagation
```

## Example
Let's use LRP to explain why an image of a castle gets classified as such 
using a pre-trained VGG16 model from [Metalhead.jl](https://github.com/FluxML/Metalhead.jl):

![][castle]

```julia
using RelevancePropagation
using VisionHeatmaps         # visualization of explanations as heatmaps
using Flux, Metalhead        # pre-trained vision models in Flux
using DataAugmentation       # input preprocessing
using HTTP, FileIO, ImageIO  # load image from URL
using ImageInTerminal        # show heatmap in terminal


# Load & prepare model
model = VGG(16, pretrain=true).layers
model = strip_softmax(model)
model = canonize(model)

# Load input
url = HTTP.URI("https://raw.githubusercontent.com/Julia-XAI/ExplainableAI.jl/gh-pages/assets/heatmaps/castle.jpg")
img = load(url) 

# Preprocess input
mean = (0.485f0, 0.456f0, 0.406f0)
std  = (0.229f0, 0.224f0, 0.225f0)
tfm = CenterResizeCrop((224, 224)) |> ImageToTensor() |> Normalize(mean, std)
input = apply(tfm, Image(img))               # apply DataAugmentation transform
input = reshape(input.data, 224, 224, 3, :)  # unpack data and add batch dimension

# Run XAI method
composite = EpsilonPlusFlat()
analyzer = LRP(model, composite)
expl = analyze(input, analyzer)  # or: expl = analyzer(input)
heatmap(expl)                    # show heatmap using VisionHeatmaps.jl
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
| `LRP` with `EpsilonPlus` composite            | ![][castle-lrp-ep]             | ![][streetsign-lrp-ep]             |
| `LRP` with `EpsilonPlusFlat` composite        | ![][castle-lrp-epf]            | ![][streetsign-lrp-epf]            |
| `LRP` with `EpsilonAlpha2Beta1` composite     | ![][castle-lrp-eab]            | ![][streetsign-lrp-eab]            |
| `LRP` with `EpsilonAlpha2Beta1Flat` composite | ![][castle-lrp-eabf]           | ![][streetsign-lrp-eabf]           |
| `LRP` with `EpsilonGammaBox` composite        | ![][castle-lrp-egb]            | ![][streetsign-lrp-egb]            |
| `LRP` with `ZeroRule` (discouraged)           | ![][castle-lrp]                | ![][streetsign-lrp]                |

## Acknowledgements
> Adrian Hill acknowledges support by the Federal Ministry of Education and Research (BMBF) 
> for the Berlin Institute for the Foundations of Learning and Data (BIFOLD) (01IS18037A).

<!-- References -->
[paper-lrp]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
[paper-crp]: https://www.nature.com/articles/s42256-023-00711-8

<!-- Images -->
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

<!-- Shields / Badges -->
[docs-stab-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stab-url]: https://julia-xai.github.io/RelevancePropagation.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://julia-xai.github.io/RelevancePropagation.jl/dev
[ci-img]: https://github.com/Julia-XAI/RelevancePropagation.jl/actions/workflows/CI.yml/badge.svg?branch=main
[ci-url]: https://github.com/Julia-XAI/RelevancePropagation.jl/actions/workflows/CI.yml?query=branch%3Amain
[codecov-img]: https://codecov.io/gh/Julia-XAI/RelevancePropagation.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Julia-XAI/RelevancePropagation.jl
[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl