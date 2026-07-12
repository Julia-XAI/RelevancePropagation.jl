# Porting notes: v3 (Flux/Zygote) → v4 (Lux/Enzyme)

Differences discovered while porting, recorded for reviewers and as raw
material for the v4.0.0 CHANGELOG. The migration plan itself lives in
`PLAN.md`.

## API

- The analyzer is constructed from the Lux triple:
  `LRP(model, ps, st[, rules])` instead of `LRP(model[, rules])`.
  States are converted once via `Lux.testmode(st)` at construction
  (LRP is inference-only).
- Rules are assigned as a `NamedTuple` mirroring the keys of `model.layers`
  (`layer_1`, `layer_2`, …). A plain `AbstractVector` of rules is still
  accepted for flat models and matched positionally.

## Behavioral differences

### Lux `LayerNorm` is not Flux `LayerNorm`

This is the one place where analyzing "the same" architecture in v3 and v4
can give different relevances, because the layer itself computes something
different:

- **Normalized dimensions:** Lux's default `dims=Colon()` normalizes over
  *all* dimensions **including the batch dimension**. Flux normalized
  per-sample over `1:length(shape)`. `LayerNormRule` follows the layer's
  `dims` field; to reproduce Flux behavior, construct the Lux layer with
  `dims=1:length(shape)`-style explicit dims.
- **Epsilon placement:** Lux computes `(x - μ) / sqrt(σ² + ϵ)`;
  Flux computed `(x - μ) / (σ + ϵ)`. Results differ for any `ϵ > 0`.
- Variance is uncorrected (`corrected=false`) in both, but Lux applies the
  affine transform as `scale .* x̂ .+ bias` with `ps.scale`/`ps.bias` sized
  `(shape..., 1)`.
- The affine flag is a `Static.True`/`Static.False` *type parameter*, not a
  `Bool` field — `setproperties(ln, (; affine=false))` fails. With
  `affine=false` the layer has empty `ps`.
- There is no inner `.diag::Scale` like in Flux; the `LayerNormRule`
  fallback and `canonize` build the affine part from `ps.scale`/`ps.bias`
  as a Lux `Scale` on the fly.

### Rule compatibility for no-bias layers

Lux constructs `Dense(...; use_bias=false)` by *omitting* the `:bias` key
from `ps` (Flux stored the sentinel `bias = false`). v3's
`has_bias(layer) = hasproperty(layer, :bias)` was therefore `true` even for
no-bias Flux layers; v4's rule compatibility defaults to requiring only a
weight (`haskey(ps, :weight)`), which matches v3's *effective* behavior —
no-bias `Dense`/`Conv` remain compatible with all weight-bias rules.

## Framework differences (Flux → Lux)

- **Parameter naming is uniform:** every parameterized layer LRP touches
  (`Dense`, `Scale`, `Conv`, `ConvTranspose`) names its parameters
  `ps.weight` / `ps.bias`. Flux's `Scale` used `scale`, which required a
  `get_weight` special case in v3 — dropped.
- **Activation field is uniformly `activation`** (Flux used `σ`/`λ`), so
  removing activations is one generic
  `ConstructionBase.setproperties(layer, (; activation=identity))` instead
  of per-layer-type methods.
- **`CrossCor` doesn't exist as a type:** cross-correlation is
  `Conv(...; cross_correlation=true)` — the *same* `Conv` type. Rules
  dispatching on `Conv` cover it automatically;
  `ConvLayer = Union{Conv,ConvTranspose}`.
- **`SkipConnection` is an `AbstractLuxWrapperLayer{:layers}`:** its
  `ps`/`st` pass through to the wrapped layer *directly*, with no
  intermediate `:layers` key. `Chain`/`Parallel` store children in a
  `layers::NamedTuple` that `ps`/`st` mirror key-for-key.
- **Bare functions in a `Chain` become `WrappedFunction{typeof(f)}`** —
  softmax and activation checks must handle both the raw function and the
  wrapper.
- **Lux `Chain` does not auto-flatten nested chains**, so `flatten_model`
  stays necessary (and must remap `ps`/`st` keys jointly).
- **`BatchNorm` state lives in `st`** (`running_mean`, `running_var`,
  `training`); inference mode is `Lux.testmode(st)` on the state tree, not
  `Flux.testmode!` on the layer.
- **Reshaping:** `Flux.flatten`/`MLUtils.flatten` → `Lux.FlattenLayer`
  (plus `ReshapeLayer`) in the `ReshapingLayer` union.

## AD differences (Zygote → Enzyme)

- All Enzyme code is confined to `src/autodiff.jl`. The only AD primitive
  is a VJP w.r.t. a layer's *input*, via split-mode thunks
  (`ReverseSplitWithPrimal`) so the seed can depend on the primal output,
  matching Zygote's `pullback` semantics with one true forward pass.
- **Pullbacks are single-use.** Zygote's `back` could be re-invoked with
  different seeds; Enzyme reverse thunks cannot re-run a consumed tape.
  `AlphaBetaRule` and `GeneralizedGammaRule` (which seed the same pullback
  twice in v3) were restructured onto `layer_pullback_2seeds`: one width-2
  `BatchDuplicated` shadow evaluates both seeds in a single
  forward+reverse pass.
- **Width-2 thunks are restricted to weight-bias layers.** Compiling
  width-2 thunks through pooling/normalization layers nondeterministically
  aborts Julia with an Enzyme assertion (`AdjointGenerator.h:6478`).
  The two rules that need two seeds only apply to weight-bias layers, so
  the restriction costs nothing — but it must never be violated.
- **Layers are differentiated as immutable `FrozenLayer(layer, ps, st)`
  bundles** annotated `Enzyme.Const`. `Lux.StatefulLuxLayer` was rejected:
  it mutates itself on every call, which is unsafe to annotate `Const`
  across split-mode fwd/rev boundaries. LRP never uses updated states, so
  `FrozenLayer` discards them.
- **Enzyme accumulates (`+=`) into shadow buffers** — any reuse of shadows
  across calls requires `make_zero!` in between, or relevances silently
  accumulate.
- `back(s)` returns the input gradient directly, so v3's
  `c = only(back(s))` became `c = back(s)` throughout the rules.

## Structural code changes

- `ChainTuple`/`ParallelTuple`/`SkipConnectionTuple` and
  `chainmap`/`chainzip` (`chain_utils.jl`, ~280 lines + MacroTools) are
  gone: rules and modified layers are stored as nested `NamedTuple`s
  mirroring the Lux `ps` tree; `map_layers` covers the mapping use case.
- `ModelIndex` → `Functors.KeyPath` (what Lux's own `layer_map` uses) for
  `LayerMap`/`show_layer_indices`.
- `copy_layer` is gone: rules modify `ps` NamedTuples and wrap them in new
  `FrozenLayer`s; the activation swap is the one generic `setproperties`
  call above.
- Dependencies: dropped Flux, Zygote, MacroTools, MLUtils,
  DifferentiationInterface (never added — raw Enzyme by design); added
  Lux, Enzyme, Functors, ConstructionBase. Zygote survives as a
  *test-only* dependency to cross-check `layer_pullback`.

## Test suite / reference values

- **Rule-level JLD2 references from v3 stay valid**: the test layers use
  explicit `StableRNG(123)` weights injected into Lux `ps` NamedTuples,
  and Flux/Lux agree on convolution semantics and weight layout.
- **Model-level (CNN) references need regeneration** in phase 7:
  `Lux.setup` draws parameters in a different order than Flux's
  initialization, even with the same seed.
- v3 tests whose feature isn't ported yet are adapted to the Lux API and
  kept as `@test_skip`/`@test_broken`, guarded on
  `isdefined(RelevancePropagation, :SymbolName)` where possible so they
  activate automatically when the port lands (never deleted).
