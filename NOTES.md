# Porting notes: v3 (Flux/Zygote) → v4 (Lux/Enzyme)

Differences discovered while porting, recorded for reviewers and as
background for the v4.0.0 CHANGELOG.

## API

- The analyzer is constructed from a model together with its parameters and
  states: `LRP(model, ps, st[, rules])` instead of `LRP(model[, rules])`.
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

### Model utilities transform the Lux triple

- `flatten_model` and `canonize` take and return `(model, ps, st)`:
  splicing nested `Chain`s and fusing BatchNorm re-key `ps`/`st` to the
  flattened `layer_1..layer_N` structure. `strip_softmax` stays model-only
  (activation functions are layer configuration in Lux, `ps` untouched).
- `strip_softmax` replaces a bare output softmax with `NoOpLayer()`,
  *preserving chain length* (v3 swapped the Flux function for `identity`,
  which also preserved length). The v3 implementation found the output
  layer by `!=` comparison against `last_element` — unsafe in Lux, where
  immutable layers compare structurally (`Dense(2 => 2) == Dense(2 => 2)`
  is `true`, unlike mutable Flux layers) — so v4 descends positionally.
- `canonize` fuses BatchNorm using the running statistics in `st` and
  *includes `epsilon`* — the fusion is exact (v3 ignored `ϵ` and papered
  over it with `safedivide`). `affine=false` BatchNorm fuses with
  `γ=1, β=0`; `track_stats=false` (no `running_mean` in `st`) blocks
  fusion. Fusing into a `use_bias=false` layer flips the static flag via
  `setproperties(layer, (; use_bias=Lux.static(true)))` — a plain `Bool`
  fails, the field is `Static.True`/`Static.False`.
- Lux `Parallel` does **not** wrap bare functions in `WrappedFunction`
  (`Chain` does); it also has no `getindex`, so tests address branches as
  `p.layers.layer_i`.

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
  `Flux.testmode!` on the layer. Under LRP, testmode `BatchNorm` is
  differentiated like any other layer; `canonize` can fuse it into a
  preceding linear layer.
- **Reshaping:** `Flux.flatten`/`MLUtils.flatten` → `Lux.FlattenLayer`
  (plus `ReshapeLayer`) in the `ReshapingLayer` union.

## AD differences (Zygote → Enzyme)

- All Enzyme code is confined to `src/autodiff.jl`. The only AD primitive
  is a VJP w.r.t. a layer's *input*, via split-mode thunks
  (`ReverseSplitWithPrimal`) so the seed can depend on the primal output,
  matching Zygote's `pullback` semantics with one true forward pass.
- **Pullbacks are single-use.** Zygote's `back` could be re-invoked with
  different seeds; an Enzyme reverse thunk consumes the tape recorded by
  its forward thunk. Verified empirically: re-running a reverse thunk on an
  already-consumed tape happens to return correct values for some layers
  (`Dense`) but aborts the Julia process for others (`Scale`), so it must
  never be done. Rules that need VJPs with two different seeds through the
  same layer (`AlphaBetaRule`, `GeneralizedGammaRule`) construct one
  single-use pullback per seed; the cost is one extra forward pass per
  layer, and the thunk is compiled only once per layer and input type.
- **Layers are differentiated as immutable `StaticLayer(layer, ps, st)`
  bundles** (named to avoid confusion with `Lux.Experimental.FrozenLayer`) annotated `Enzyme.Const`. `Lux.StatefulLuxLayer` was rejected:
  it mutates itself on every call, which is unsafe to annotate `Const`
  across split-mode fwd/rev boundaries. LRP never uses updated states, so
  `StaticLayer` discards them.
- **Enzyme accumulates (`+=`) into shadow buffers** — any reuse of shadows
  across calls requires `make_zero!` in between, or relevances silently
  accumulate.
- `back(s)` returns the input gradient directly, so v3's
  `c = only(back(s))` became `c = back(s)` throughout the rules.

## Structural code changes

- `ChainTuple`/`ParallelTuple`/`SkipConnectionTuple` and
  `chainmap`/`chainzip` (~280 lines + MacroTools) are gone: rules and
  modified layers are stored as nested `NamedTuple`s mirroring the Lux `ps`
  tree. The generic model-walking helpers that survive (`map_layers`,
  `chainall`, `first_element`, `last_element`, …) live in the self-contained
  `ModelSurgeon` submodule (`src/ModelSurgeon/`) together with
  `flatten_model`, `canonize` and `strip_softmax`; RP's exported
  `flatten_model`/`canonize` are thin policy wrappers in
  `src/model_surgery.jl` that keep pooling layers intact
  (see `PLAN_MODELSURGEON.md`).
- `ModelIndex` → `Functors.KeyPath` (what Lux's own `layer_map` uses) for
  `LayerMap`/`show_layer_indices`.
- `copy_layer` is gone: rules modify `ps` NamedTuples and wrap them in new
  `StaticLayer`s; the activation swap is the one generic `setproperties`
  call above.
- Dependencies: dropped Flux, Zygote, MacroTools, MLUtils,
  DifferentiationInterface (never added — raw Enzyme by design); added
  Lux, Enzyme, Functors, ConstructionBase. Zygote survives as a
  *test-only* dependency to cross-check `layer_pullback`.
- CRP kept the v3 algorithm unchanged: the analyzer's
  `rules`/`layers`/`modified_layers` NamedTuples are unpacked positionally
  with `values()` for the `k`-indexed backward loops, and activations come
  from `get_activations` on the `StaticLayer` NamedTuple. The flat-model
  assumption (positional `layer::Int`) is now documented in the docstring.

## Compilation latency

Enzyme compiles one thunk per (rule-variant × layer type × input type) on
the *first* `analyze` call. On a random-init VGG16 (224×224×3×1 input,
`EpsilonPlusFlat()` composite, Apple M3 Pro, Julia 1.12) the first
`analyze` takes ~32 s; warm calls take ~1.8 s. Thunks are cached per input
element type and dimensionality, so batch-size changes of the same
eltype/ndims do not recompile.

## Test suite / reference values

- **Rule-level JLD2 references from v3 stay valid**: the test layers use
  explicit `StableRNG(123)` weights injected into Lux `ps` NamedTuples,
  and Flux/Lux agree on convolution semantics and weight layout.
- **Model-level (CNN) references were regenerated**: `Lux.setup` draws
  parameters in a different order than Flux's initialization, even with
  the same seed.
