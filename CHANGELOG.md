# RelevancePropagation.jl

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
