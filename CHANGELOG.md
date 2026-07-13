# RelevancePropagation.jl

## Version `v4.0.0`
This release rewrites the package from Flux.jl/Zygote.jl to
[Lux.jl](https://lux.csail.mit.edu) and
[Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).
Zygote.jl is unmaintained, and Lux's explicit-parameter design is the model
framework with first-class Enzyme support.
Flux.jl models are no longer supported.

### Breaking API changes
* ![BREAKING][badge-breaking] `LRP` analyzers are constructed from the Lux
  triple: `LRP(model, ps, st[, rules])` instead of `LRP(model[, rules])`.
  States are converted once via `Lux.testmode` at construction
  (LRP is inference-only).
* ![BREAKING][badge-breaking] Rules for nested models are assigned as
  `NamedTuple`s mirroring the structure of the model's `ps` and `st`.
  `ChainTuple`, `ParallelTuple` and `SkipConnectionTuple` were removed.
  A plain `AbstractVector` of rules is still accepted for flat models.
* ![BREAKING][badge-breaking] `flatten_model` and `canonize` take and return
  the triple `(model, ps, st)`: splicing nested `Chain`s and fusing BatchNorm
  re-key `ps` and `st`. `strip_softmax` stays model-only.
* ![BREAKING][badge-breaking] The `LRP` constructor no longer flattens models
  automatically and the `flatten` keyword argument was removed —
  call `flatten_model` explicitly.
* ![BREAKING][badge-breaking] `LayerMap` addresses layers by
  `Functors.KeyPath` (as used by Lux's `ps`/`st` trees) instead of
  `ModelIndex`; integers and tuples of integers are converted for
  convenience. `show_layer_indices` returns `KeyPath` structures.
* ![BREAKING][badge-breaking] Custom layers must subtype
  `Lux.AbstractLuxLayer` and follow the Lux layer interface.
  `modify_layer` operates on the internal `FrozenLayer` wrapper bundling a
  layer with its `ps` and `st`.
* ![BREAKING][badge-breaking] GPU support is out of scope for this release
  (Enzyme runs on the CPU).

### Behavioral changes
* ![BREAKING][badge-breaking] Lux `LayerNorm` differs from Flux `LayerNorm`:
  its default `dims=Colon()` normalizes over all dimensions including the
  batch dimension, and epsilon is placed inside the square root
  (`(x - μ) / √(σ² + ϵ)`). `LayerNormRule` follows the layer's configuration,
  so relevances for "the same" architecture can differ from v3.
* ![Enhancement][badge-enhancement] BatchNorm fusion in `canonize` is now
  exact: it uses the running statistics in `st` and includes `epsilon`
  (v3 ignored it). `affine=false` BatchNorm and `use_bias=false` layers
  are handled.
* ![Feature][badge-feature] `flatten_model` unwraps generic
  `AbstractLuxWrapperLayer`s, e.g. Boltz.jl model wrappers like
  `Vision.VGG`.

### Automatic differentiation
* ![Maintenance][badge-maintenance] The AD fallback computes vector-Jacobian
  products with Enzyme split-mode thunks instead of Zygote pullbacks.
  All Enzyme-specific code is confined to `src/autodiff.jl`.
* ![Maintenance][badge-maintenance] Enzyme compiles differentiation code per
  combination of rule, layer, and input type on the first call to `analyze`.
  Cold-start latency grows compared to v3 (VGG16: ~32 s vs ~9 s on an Apple
  M3 Pro; v3 measured on Julia 1.11, v4 on Julia 1.12), while warm calls
  stay comparable (~1.8 s vs ~1.2 s). Compiled code is cached per input
  element type and dimensionality.

### Dependencies
* ![Maintenance][badge-maintenance] Dropped Flux, Zygote, MacroTools and
  MLUtils; added Lux, LuxCore, Enzyme, Functors, ConstructionBase and Static.
  Zygote remains a test-only dependency to cross-check Enzyme pullbacks.

## Version `v3.0.0`
* ![BREAKING][badge-breaking] Update XAIBase interface to `v4`. 
  This adds a field to the `Explanation` return type and removes the `add_batch_dim` keyword argument.
  Refer to the [XAIBase.jl changelog](https://github.com/Julia-XAI/XAIBase.jl/blob/main/CHANGELOG.md#version-v400) for more information ([#19])
* ![Feature][badge-feature] Add option to skip normalization of output layer relevance ([#22])

## Version `v2.0.1`
* ![Bugfix][badge-bugfix] Fix model canonization and flattening on `SkipConnection` and `Parallel` ([#14][#14])

## Version `v2.0.0`
This release removes the automatic reexport of heatmapping functionality.
Users are now required to manually load 
[VisionHeatmaps.jl][VisionHeatmaps] and/or [TextHeatmaps.jl][TextHeatmaps].

This reduces the maintenance burden for new heatmapping features 
and the amount of dependencies for users who don't require heatmapping functionality.

* ![BREAKING][badge-breaking] Removed reexport of heatmapping functionality by updating XAIBase dependency to `v3.0.0` ([#13][#13]).
* ![Feature][badge-feature] Add support for `LayerNorm` and `Scale` layers ([#9][#9])
* ![Feature][badge-feature] Add `LayerNormRule` ([#9][#9])
* ![Documentation][badge-docs] Add LRP rule overview to docs ([#12][#12])

Some internal improvements were made as well:
* ![Maintenance][badge-maintenance] update `canonize` mechanism to include model splitting pass `canonize_split` ([#9][#9])
* ![Maintenance][badge-maintenance] improve `modify_layer` by introducing `get_weight` and `get_bias` abstractions to handle varying field names ([#9][#9])
* ![Maintenance][badge-maintenance] Update `LayerMap` to use `ModelIndex` ([#10][#10])
* ![Maintenance][badge-maintenance] Make `chainzip` more robust ([#11][#11])

## Version `v1.1.0`
* ![Feature][badge-feature] Support `SkipConnection` layers ([#8][#8])
* ![Documentation][badge-docs] Document LRP rule notation in API reference 
  ([e11c234](https://github.com/Julia-XAI/RelevancePropagation.jl/commit/e11c234c09b7c5232acc5f254379ea5bd01d1e7c))

## Version `v1.0.1`
* ![Documentation][badge-docs] Reorganize documentation ([#7][#7])

## Version `v1.0.0`
Initial release of RelevancePropagation.jl.

<!--
# Badges
![BREAKING][badge-breaking]
![Deprecation][badge-deprecation]
![Feature][badge-feature]
![Enhancement][badge-enhancement]
![Bugfix][badge-bugfix]
![Experimental][badge-experimental]
![Maintenance][badge-maintenance]
![Documentation][badge-docs]
-->

[#22]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/22
[#19]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/19
[#14]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/14
[#13]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/13
[#12]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/12
[#11]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/11
[#10]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/10
[#9]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/9
[#8]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/8
[#7]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/7

[VisionHeatmaps]: https://julia-xai.github.io/XAIDocs/VisionHeatmaps/stable/
[TextHeatmaps]: https://julia-xai.github.io/XAIDocs/TextHeatmaps/stable/

[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-security]: https://img.shields.io/badge/security-black.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg
