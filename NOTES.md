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

## AD architecture (Zygote → Enzyme)

The v4 engine implements LRP by **redefining Enzyme's VJPs**: LRP is reverse-mode AD
in which each layer's true VJP is replaced by the rule's relevance map, with
the relevance `Rᵏ` carried as the cotangent at `aᵏ`. (An earlier iteration of
the port hand-rolled its own backward-pass engine — `get_activations`,
`lrp_backward_pass!`, per-layer split-mode `layer_pullback` thunks, and
structural `lrp!` methods for `Chain`/`Parallel`/`SkipConnection` — and used
Enzyme only as a per-layer Zygote substitute; it was scrapped in review.)

- All Enzyme code is confined to `src/autodiff.jl`. One
  `Enzyme.autodiff` call per `analyze` differentiates the scalar loss
  `dot(mask, model(x))` over the *wrapped* model; the input shadow `dx` is
  the explanation. The mask (output relevance seed) is built by an
  `EnzymeRules.inactive` function, detaching it from differentiation.
- **Each rule is an `EnzymeRules` custom rule** on `lrp_node`: the
  augmented forward splits weight layers into affine part + activation and
  caches the pre-activation `z` on the tape; the reverse calls the pure
  rule body `propagate` and accumulates into the input shadow. Rules whose
  parameter modification is the identity (Zero/Epsilon, the dominant case)
  reuse the cached `z` — no modified forward pass at all.
- **Branch routing needs no structural code**: Lux's `apply` plumbing is
  differentiated as-is; shadow accumulation (`.+=`) implements "sum branch
  relevances", and a tiny vararg custom rule on the wrapped `connection` of
  `Parallel`/`SkipConnection` implements the proportional relevance split
  from the tape, with no re-forwarding of branches.
- **Inner VJPs**: rules pull seeds back through (modified) layers via
  `input_vjp`. Activation-free `Dense`/`Scale`/`Conv`/`ConvTranspose` use
  hand-written fast paths (`Wᵀs`, broadcast, `∇conv_data`, `conv`) —
  cross-checked against the AD fallback in `test_autodiff.jl`; everything
  else uses `seeded_pullback`, one *combined-mode* nested `autodiff` over
  `dot(layer(x), s)`. Combined mode re-runs the layer forward, but split
  thunks (single-use tapes, see git history: re-running a consumed tape
  aborts Julia for some layers) and their two-phase bookkeeping are gone;
  multi-seed rules (`AlphaBetaRule`, `GeneralizedGammaRule`) just call
  `input_vjp` once per seed, which with fast paths is FLOP-optimal
  (4 forwards + 4 transposes for αβ).
- **Wrappers take explicit `ps`/`st` arguments, mirroring Lux.**
  `LayerWithRule <: AbstractLuxWrapperLayer{:layer}` is parameter- and
  state-transparent, so the user's `ps`/`st` trees apply to the wrapped
  model unchanged; `ps`/`st` enter the custom rules as `Enzyme.Const`
  arguments. Benchmarked against a `StaticLayer`-style capturing wrapper:
  runtime-identical, so the explicit-argument design costs nothing (see
  "Single-layer benchmarks" below).
- **The wrapper returns an empty state.** Returning the real `st` threads
  `Const` state arrays (BatchNorm running statistics) into the active
  return structure and triggers `EnzymeRuntimeActivityError`. LRP is
  inference-only; state updates were always discarded.
- **Custom-rule return activity must cover `DuplicatedNoNeed`**: methods
  are declared `::Type{<:Union{Duplicated,DuplicatedNoNeed}}` and branch on
  `needs_primal`/`needs_shadow` (config params are static, branches fold).
- `layerwise_relevances` is implemented with `TappedLayer` wrappers between
  the outermost children (identity primal, recording reverse), rebuilt per
  call when requested.
- LuxLib warns once (`training is set to Val{false} but is being used
  within an autodiff call`) when testmode BatchNorm is differentiated —
  harmless for LRP, which intentionally differentiates the inference graph.

## Single-layer benchmarks (GammaRule)

Design variants measured on `Dense(512 => 512, relu)` (batch 64) and
`Conv((3,3), 32 => 32, relu; pad=1)` (32×32×32×8), Apple M3 Pro, warm
(`.handoff/bench_gamma.jl`, deleted with the handoff; numbers kept here).
GammaRule is the simplest parameter-modifying rule, so it isolates the cost
of lazy `modify_params` and of the VJP strategy:

| variant (rule body)               | Dense min | Conv min |
|-----------------------------------|-----------|----------|
| old engine (split thunks, precomputed modified layer) | 212 µs | 6.5 ms |
| new engine, lazy ρps, nested combined AD | 319 µs | 9.6 ms |
| new engine, lazy ρps, fast-path VJP | 199 µs | 5.9 ms |
| new engine, precomputed ρps, fast-path VJP | 173 µs | 6.0 ms |

- The nested combined-AD fallback pays a redundant forward (the rule body
  already computed `z̃`) — 1.5–1.6× slower than split thunks. The
  **fast-path VJPs are what make the new engine a net win**; the AD
  fallback only remains for layers without weights, where
  parameter-modifying rules don't apply.
- Lazy `modify_params` costs ~27 µs / 1 MiB per call on the 512×512 Dense
  and nothing measurable on Conv; precomputation would keep a permanent
  modified copy of all weights for a ≤5% end-to-end gain — not worth it.
- A ps-capturing wrapper (StaticLayer-style) is runtime-identical to the
  ps-transparent explicit-argument wrapper (602 vs 624 µs min end-to-end on
  the Dense chain); the explicit design wins on structure alone.

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
- `copy_layer` is gone: rules modify `ps` NamedTuples lazily inside the
  reverse pass (`modify_params`, returning `ps` itself when the rule
  doesn't modify parameters); the activation swap is the one generic
  `setproperties` call above (`remove_activation`).
- Dependencies: dropped Flux, Zygote, MacroTools, MLUtils,
  DifferentiationInterface (never added — raw Enzyme by design); added
  Lux, Enzyme, Functors, ConstructionBase. Zygote survives as a
  *test-only* dependency to cross-check `seeded_pullback` and the
  `input_vjp` fast paths.
- CRP kept the v3 algorithm unchanged: masking individual concepts at
  layer `l` needs explicit control over intermediate relevances, so CRP
  does not reuse the end-to-end reverse pass. Instead it runs its own
  `k`-indexed positional loops over the children of the analyzer's wrapped
  model — a forward pass collecting activations and pre-activations via
  `node_forward`, then per-concept backward passes through `propagate`
  (containers fall back to `seeded_pullback`). The flat-model assumption
  (positional `layer::Int`) is now documented in the docstring.

## Compilation latency

Enzyme compiles the reverse pass on the *first* `analyze` call: one
whole-model thunk specialized on the wrapped model type (rules are type
parameters, so a different rule assignment recompiles), plus nested
combined-mode thunks for layers taking the `seeded_pullback` fallback.
On a random-init VGG16 (224×224×3×1 input, `EpsilonPlusFlat()` composite,
Apple M3 Pro, Julia 1.12) the first `analyze` takes ~40 s (the scrapped
split-thunk engine took ~32 s); warm calls take ~1.8 s, runtime-identical
to the old engine. Thunks are cached per input element type and
dimensionality, so batch-size changes of the same eltype/ndims do not
recompile.

## Test suite / reference values

- **Rule-level JLD2 references from v3 stay valid**: the test layers use
  explicit `StableRNG(123)` weights injected into Lux `ps` NamedTuples,
  and Flux/Lux agree on convolution semantics and weight layout.
- **Model-level (CNN) references were regenerated**: `Lux.setup` draws
  parameters in a different order than Flux's initialization, even with
  the same seed.
