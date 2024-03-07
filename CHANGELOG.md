# RelevancePropagation.jl
## Version `v2.0.0`
This release removes the automatic reexport of heatmapping functionality.
Users are now required to manually load 
[VisionHeatmaps.jl][VisionHeatmaps] and/or [TextHeatmaps.jl][TextHeatmaps].

This reduces the maintenance burden for new heatmapping features 
and the amount of dependencies for users who don't require heatmapping functionality.

* ![BREAKING][badge-breaking] Removed reexport of heatmapping functionality by updating XAIBase dependency to `v3.0.0` ([#13][pr-13]).
* ![Feature][badge-feature] Add support for `LayerNorm` and `Scale` layers ([#9][pr-9])
* ![Feature][badge-feature] Add `LayerNormRule` ([#9][pr-9])
* ![Documentation][badge-docs] Add LRP rule overview to docs ([#12][pr-12])

Some internal improvements were made as well:
* ![Maintenance][badge-maintenance] update `canonize` mechanism to include model splitting pass `canonize_split` ([#9][pr-9])
* ![Maintenance][badge-maintenance] improve `modify_layer` by introducing `get_weight` and `get_bias` abstractions to handle varying field names ([#9][pr-9])
* ![Maintenance][badge-maintenance] Update `LayerMap` to use `ModelIndex` ([#10][pr-10])
* ![Maintenance][badge-maintenance] Make `chainzip` more robust ([#11][pr-11])

## Version `v1.1.0`
* ![Feature][badge-feature] Support `SkipConnection` layers ([#8][pr-8])
* ![Documentation][badge-docs] Document LRP rule notation in API reference 
  ([e11c234](https://github.com/Julia-XAI/RelevancePropagation.jl/commit/e11c234c09b7c5232acc5f254379ea5bd01d1e7c))

## Version `v1.0.1`
* ![Documentation][badge-docs] Reorganize documentation ([#7][pr-7])

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

[pr-13]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/13
[pr-12]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/12
[pr-11]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/11
[pr-10]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/10
[pr-9]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/9
[pr-8]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/8
[pr-7]: https://github.com/Julia-XAI/RelevancePropagation.jl/pull/7

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
