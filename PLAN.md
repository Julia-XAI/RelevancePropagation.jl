# Plan: Migrate RelevancePropagation.jl to Lux.jl + Enzyme.jl (v4.0.0)

**Branch:** `ah/enzyme` · **Status:** Phases 1–6 complete; Phase 7 (tests,
docs, release) next — see HANDOFF.md.

Rewrite the package from Flux/Zygote to Lux/Enzyme as a breaking v4.0.0 release,
simplifying the codebase along the way.

## Motivation

- **Zygote is unmaintained and deprecated** — staying on it is a dead end, which
  motivates the switch to Enzyme as the AD backend.
- **Flux doesn't fully support Enzyme**, which in turn motivates the switch to
  Lux: its explicit-parameter design is the model framework with first-class
  Enzyme support (and it simplifies the LRP machinery, see below).

## Target architecture

Public API becomes the Lux triple:

```julia
analyzer = LRP(model, ps, st, composite)
expl = analyze(input, analyzer)
```

- `st` is converted once via `Lux.testmode(st)` at construction (LRP is inference-only).
- The only AD primitive in the whole package is a **VJP w.r.t. a layer's input**
  (never w.r.t. parameters), implemented with **raw Enzyme** — no
  DifferentiationInterface middle layer (decided: the port is Lux+Enzyme specific,
  and raw Enzyme gives more control).
- The VJP is **not always through a linearized layer**: for layers without
  weight/bias, `modify_layer` returns the original layer unchanged
  (`src/rules.jl:129`), so under Zero/Epsilon rules the pullback runs through
  max-pooling, testmode BatchNorm, and bare activation functions *including their
  nonlinearity* (which is what makes ReLU behave as pass-through). Enzyme's
  differentiation surface includes these nonlinear layers — hence the Phase 1
  spike list.

### AD core: one Enzyme helper

Keep the `(z, back)` pullback *shape* every rule in `src/rules.jl` is written
against, but implement it once with Enzyme split-mode thunks. All Enzyme
incantations live in a single file; rules stay readable.

```julia
# Sketch — the only place Enzyme appears in the package
struct FrozenLayer{L,P,S}   # immutable → genuinely safe to annotate `Const`
    layer::L; ps::P; st::S
end
(f::FrozenLayer)(x) = first(Lux.apply(f.layer, x, f.ps, f.st))

function layer_pullback(f::F, x::AbstractArray) where {F<:FrozenLayer}
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{F}, Duplicated, Duplicated{typeof(x)})
    dx = make_zero(x)
    tape, z, dz = fwd(Const(f), Duplicated(x, dx))
    back(s) = (dz .= s; rev(Const(f), Duplicated(x, dx), tape); dx)   # call once per pullback
    return z, back
end
```

Design points:

1. **One true forward pass.** `ReverseSplitWithPrimal` matches Zygote's `pullback`
   semantics: primal out, seed later. Necessary because the seed
   `s = Rᵏ⁺¹ ./ modify_denominator(rule, z)` depends on the primal `z`.
   (The naive combined-`Reverse` + mutating-wrapper pattern runs the forward twice.)
2. **`BatchDuplicated` for multi-seed rules.** `AlphaBetaRule` and
   `GeneralizedGammaRule` reuse one pullback with two seeds
   (`back⁺(sᵅ)`, `back⁺(sᵝ)` — `src/rules.jl:413-416`). Re-invoking a reverse thunk
   on the same tape is unsupported; a width-2 `BatchDuplicated` shadow does both
   seeds in one forward+reverse. Details:
   - The width lives in the *mode*, not just the annotations:
     `ReverseSplitWidth(ReverseSplitWithPrimal, Val(2))`.
   - This is a **second helper with its own shape** — `z, back2 =
     layer_pullback_2seeds(f, x)` with `back2(s₁, s₂) -> (dx₁, dx₂)` — because both
     seeds depend on primals from *two different* pullbacks (`sᵅ` needs
     `zᵅ⁺ + zᵅ⁻`, `src/rules.jl:411`); split mode supports this (seed after fwd,
     before rev), but one helper shape doesn't serve all rules.
   - `ZBoxRule` needs **no** batching: its three backs are each called once with
     the same seed (`src/rules.jl:346-348`).
3. **Preallocated shadows / first-call-cached thunks.** `LRP` precomputes modified
   layers at construction. Caveats:
   - Enzyme *accumulates* (`+=`) into `dx` — reused shadow buffers must be
     re-zeroed with `make_zero!` between calls or relevances silently accumulate.
   - Thunks are keyed on `typeof(x)` (eltype + ndims), unknown at `LRP`
     construction — so thunk caching happens on first `analyze`, not in the
     constructor.
   - Measure with the existing PkgJogger suite.
4. **Activity is trivial — via `FrozenLayer`, not `StatefulLuxLayer`.**
   `StatefulLuxLayer` mutates itself on every call (`set_state!` caches the
   returned `st`); annotating a self-mutating wrapper `Const` is only safe while
   every written value is provably inactive, and mutation between the fwd and rev
   thunk calls interacts with split-mode `ModifiedBetween` assumptions. LRP never
   uses the returned state, so the immutable `FrozenLayer` above discards it —
   zero mutation, same `Const` activity story. NNlib ships an EnzymeCore extension
   with custom rules for `conv!`/`pool!`, so CNN hot paths don't rely on Enzyme
   differentiating im2col.
5. **Fallback** if split mode misbehaves on some layer: combined `Reverse` with a
   mutating wrapper (redundant forward, identical correctness), per-layer, behind
   the same helper interface.
6. **Rule-code delta is mechanical:** the helper returns `dx` directly, so every
   `c = only(back(s))` becomes `c = back(s)` — touches all rules, changes nothing
   semantically.

### Simplification wins from Lux

1. **`copy_layer` shrinks to one generic function** (`src/layer_utils.jl:34-46`).
   Parameters live in `ps`, so the per-rule part of `modify_layer` collapses into
   `modify_parameters(rule, ps)` returning a modified `ps` NamedTuple. The
   activation swap still requires reconstructing the immutable Lux layer struct,
   but it becomes *one* generic call —
   `ConstructionBase.setproperties(layer; activation=identity)` (Lux layers
   uniformly name the field `activation`) — instead of five hand-written methods,
   and it's rule-independent, so it runs once per layer rather than per
   rule-variant. Layers without an `activation` field (pooling, dropout, reshape)
   are never modified.
2. **`ChainTuple`/`ParallelTuple`/`SkipConnectionTuple` die**
   (`src/chain_utils.jl`, 277 lines, plus `@forward`/MacroTools). Lux's `ps`/`st`
   NamedTuples already mirror the model tree; rules and pre-modified parameters are
   stored as NamedTuples of the same shape. Dataflow dispatch in
   `src/lrp.jl:104-163` already dispatches on the layer type, so the wrapper tuples
   were redundant.
3. **`ModelIndex` (83 lines) → `Functors.KeyPath`** (what Lux's own `layer_map`
   uses). `LayerMap`/`show_layer_indices` keep their semantics.
4. **Bias special-casing shrinks.** Lux omits the `bias` field when
   `use_bias=false` → `haskey(ps, :bias)` replaces the `bias == false` branches in
   `modify_layer` and `canonize_fuse`.
5. **`Flux.activations` → ~10-line loop** over `model.layers` threading `st`.
6. **`strip_softmax` gets trivial** — activation is layer config in Lux; `ps`
   untouched.
7. **Dependency diet.** Drop: Flux, Zygote, MacroTools, MLUtils (only used for
   `MLUtils.flatten` in the `ReshapingLayer` union and one composite preset →
   `Lux.FlattenLayer`).
   Final set: Lux, Enzyme, Functors, NNlib, XAIBase, Reexport (stays — reexports
   XAIBase, part of the public API), ConstructionBase (tiny, already in Lux's dep
   tree, for the activation swap) + stdlib (Statistics, Markdown, Random).
   Zygote becomes a *test-only* dep (cross-check `layer_pullback` vs
   `Zygote.pullback` on sample layers).

### What gets harder (known costs)

- **`lrp!` signatures grow** to carry `(layer, ps, st)`. Bundle each layer's
  `(layer, ps_modified, st)` into a `StatefulLuxLayer` (or small internal struct)
  to keep rule implementations clean.
- **`canonize` and `flatten_model` must transform `(model, ps, st)` jointly** —
  fusing BatchNorm or flattening nested Chains changes `ps` key structure
  (`layer_1`, `layer_2`, …). Most fiddly part of the port; isolated in its own
  phase. Same applies to the `flatten=true` LRP kwarg that `RangeMap`/
  `FirstNTypeMap` indexing semantics rely on.
- **Reference JLD2 values need regeneration** — Lux's `setup` draws parameters in a
  different order than Flux's `init`, even with StableRNG.
- **Docs VGG example moves Metalhead → Boltz.jl** (`Vision.VGG`).
- **Lux `LayerNorm` differs structurally from Flux's.** The non-canonized
  `LayerNormRule` fallback (`src/rules.jl:558-573`) and `canonize_split` both
  assume Flux's inner `.diag::Scale`; Lux folds affine params into
  `ps.scale`/`ps.bias` and configures normalized dims differently. Needs a
  Lux-native rewrite, structure verified in the Phase 1 spike.
- **Enzyme × Julia version compat.** Enzyme historically lags new Julia minors —
  pin compat bounds and decide the CI matrix explicitly.
- **TTFX.** Per-layer thunks are Enzyme-friendly, but first `analyze` on a
  VGG-scale composite compiles one thunk per (rule-variant × layer type). Budget
  an explicit before/after latency measurement, not just runtime benchmarks.

## Phases

### Phase 1 — Spike (de-risk first)
- [x] Scratch script: `autodiff_thunk`/`ReverseSplitWithPrimal` through
      `FrozenLayer`-wrapped `Dense`, `Conv`, `MaxPool`, testmode `BatchNorm`,
      `LayerNorm` on CPU; compare against Zygote VJPs. Include the nonlinear
      surface (pooling, testmode BN, activation functions) — see AD core notes.
- [x] Two-seed variant: `ReverseSplitWidth(ReverseSplitWithPrimal, Val(2))` +
      width-2 `BatchDuplicated`; validate the `layer_pullback_2seeds` shape.
- [x] Check whether `set_runtime_activity` is needed anywhere (ideally not).
- [x] Verify Lux API details assumed in this plan: `Scale` parameter names;
      whether Lux 1.x `Chain` auto-flattens nested chains; `CrossCor` ↔
      `Conv(cross_correlation=true)` type-level implications for `ConvLayer` union;
      Lux `LayerNorm` internals (`ps.scale`/`ps.bias`, normalized-dims semantics)
      for `LayerNormRule` and `canonize_split`.

**Spike findings (Enzyme v0.13.181, Lux v1.31.4, Julia 1.12.6):**

1. Width-1 split-mode `layer_pullback` matches Zygote VJPs for *every* layer
   type in one session: `Dense` (identity/relu/gelu/no-bias), `Scale`, `Conv`
   (incl. `cross_correlation=true`), `ConvTranspose`, `MaxPool`, `MeanPool`,
   `Global*Pool`, `Adaptive*Pool`, testmode `BatchNorm` (incl. relu),
   `LayerNorm` (incl. relu, `affine=false`), `FlattenLayer`, testmode
   `Dropout`, `WrappedFunction`.
2. Width-2 `layer_pullback_2seeds` matches Zygote on weight-bias layers
   (`Dense`, `Scale`, `Conv`, `ConvTranspose`) — the only layers that need it
   (`AlphaBetaRule`/`GeneralizedGammaRule` domain). **Compiling width-2 thunks
   for pooling/BatchNorm layers nondeterministically aborts Julia** with an
   Enzyme assertion (`AdjointGenerator.h:6478`, boxed-Float32 shadow) — the
   implementation must never request width-2 thunks for non-weight layers.
3. `set_runtime_activity` is not needed anywhere.
4. Lux API details: parameter naming is uniformly `weight`/`bias` (also for
   `Scale`, so the Flux `get_weight` special case dies); `use_bias=false` omits
   the `bias` key from `ps`; `Conv(cross_correlation=true)` is the *same*
   `Conv` type (`ConvLayer = Union{Conv,ConvTranspose}`); Lux `Chain` does
   **not** auto-flatten nested chains (joint `flatten_model` stays necessary);
   `LayerNorm` has fields `(shape, activation, epsilon, dims, affine)` and
   `ps = (scale, bias)` sized `(shape..., 1)`, `affine=false` gives empty `ps`;
   `BatchNorm` `ps = (scale, bias)`, `st = (running_mean, running_var,
   training)`; `ConstructionBase.setproperties(layer, (; activation=identity))`
   works on `Dense`/`Conv`/`Scale`/`BatchNorm`/`LayerNorm` and the result
   differentiates fine; `Parallel` has fields `(connection, layers, name)` with
   `ps` mirroring `layers` keys; `SkipConnection` is an
   `AbstractLuxWrapperLayer{:layers}`, so its `ps`/`st` pass through to the
   wrapped layer *directly* (no `:layers` key); pooling layers are distinct
   wrapper types (unions work); bare functions in a `Chain` become
   `WrappedFunction{typeof(f)}` (softmax checks must handle this).

### Phase 2 — Core skeleton
- [x] Swap deps in Project.toml (add Lux, Enzyme, Functors; drop Flux, Zygote,
      MacroTools, MLUtils).
- [x] Lux layer-type unions (`src/layer_types.jl`).
- [x] `layer_utils` on `ps` (activation field, `haskey(ps, :weight/:bias)`).
- [x] Activations collector (loop over `model.layers`, threading `st`).
- [x] New `LRP` struct: `model, ps, st_test, rules::NamedTuple, modified_ps::NamedTuple`.
- [x] `FrozenLayer` + `layer_pullback` Enzyme helper; generic `lrp!` using it
      (incl. the mechanical `only(back(s))` → `back(s)` change in all rules).
- [x] `ZeroRule` + `EpsilonRule`; one end-to-end MLP test green.

### Phase 3 — All rules
- [x] Port `modify_*` family to operate on `ps` NamedTuples.
- [x] Simple rules: `GammaRule`, `WSquareRule`, `FlatRule`.
- [x] Multi-variant rules (named-tuple-of-variants convention kept): `ZBoxRule`
      and `ZPlusRule` (single-seed pullbacks only); `AlphaBetaRule` and
      `GeneralizedGammaRule` via `layer_pullback_2seeds`.
- [x] `LayerNormRule` against Lux `LayerNorm`; `PassRule`.
      Lux `LayerNorm` note: default `dims=Colon()` normalizes over *all* dims
      (incl. batch), unlike Flux's per-sample `1:length(shape)`; the rule
      follows the layer's `dims` and documents the difference.
- [x] Keep fast paths: Zero/Epsilon on dropout/reshaping layers, FlatRule on
      Dense.
      The v3 rule reference JLD2 values stay valid (explicit StableRNG(123)
      weights injected into Lux `ps`).

### Phase 4 — Dataflow + composites
- [x] `Chain`/`Parallel`/`SkipConnection` `lrp!` recursion with per-branch `ps`/`st`.
      `SkipConnection` is an `AbstractLuxWrapperLayer{:layers}`, transparent in
      `ps`/`st`/rules — plain rules paired with `FrozenLayer{<:SkipConnection}`
      are ambiguous against the rule-specific `lrp!` methods, resolved with
      `@eval`-generated routing methods onto `lrp_skip_connection!`.
- [x] Composites: TypeMaps on Lux layer types; `LayerMap` via `KeyPath`.
      v3 semantics preserved exactly: positional primitives (`RangeMap`,
      `RangeTypeMap`, `FirstNTypeMap`) use *top-level* positions, `LayerMap`
      matches by `KeyPath` prefix, last matching primitive wins,
      `WrappedFunction` is unwrapped for type matching. `lrp_rules(model,
      composite)` needs no `ps`/`st`.
- [x] Rewrite `show.jl` for NamedTuple rules (shrinks).
      Show references regenerated; presets gained `NoOpLayer => PassRule()`.

### Phase 5 — Model utilities
- [x] Model checks / `LRP_CONFIG` on Lux types (`AbstractLuxLayer`, `WrappedFunction`).
- [x] `strip_softmax` (model-only, `ps` untouched).
      Bare output softmax becomes `NoOpLayer()` (length-preserving).
- [x] Joint `(model, ps, st)` version of `flatten_model` (remap `ps` keys).
- [x] Joint `(model, ps, st)` version of `canonize`: LayerNorm split, BatchNorm
      fusion into Dense/Conv.
      Fusion includes `epsilon` (exact, unlike v3) and handles `affine=false`
      BatchNorm and `use_bias=false` layers (flipped via `Lux.static(true)`).

### Phase 6 — CRP
- [x] Port `crp.jl` (near-mechanical once backward pass is ported).
      NamedTuples unpacked positionally via `values()`; flat-model
      assumption now documented in the `CRP` docstring.

### Phase 7 — Tests, docs, release
- [ ] Port test models to Lux (StableRNG via `Lux.setup`); regenerate JLD2
      references.
- [x] Zygote-vs-Enzyme consistency testset for `layer_pullback` (Zygote test-only dep).
      Landed with phase 2 (`test/test_autodiff.jl`, additive-first ordering).
- [ ] Keep Aqua, ExplicitImports, JuliaFormatter tests.
- [ ] Port PkgJogger benchmarks; measure shadow/thunk preallocation (remember
      `make_zero!` on reused shadows; thunk cache is per input type, first-call).
- [ ] TTFX measurement: first-`analyze` latency on a VGG-scale composite,
      before/after comparison against v3.
- [ ] Rewrite Literate docs with Lux; VGG composites example via Boltz.jl; README.
- [ ] Remove stale Tullio/LoopVectorization doc content: `basics.jl` advertises a
      Tullio/LV package extension that no longer exists (stale since the
      ExplainableAI.jl split); rewrite the `@tullio` custom-rule example in
      `developer.md` with plain broadcasting/matmul. (The package itself has no
      Tullio/LV deps — nothing to drop from Project.toml.)
- [ ] CHANGELOG, compat bounds (julia ≥ 1.10; pin Enzyme, decide CI matrix —
      Enzyme lags new Julia minors), tag v4.0.0.

## Commit strategy

Optimize the branch history for commit-by-commit review (merge without squash):

1. **One commit per plan checkbox**, subject prefixed with the phase:
   `phase 3: port GammaRule/WSquareRule/FlatRule to ps NamedTuples`. Reviewers can
   diff a commit against its checkbox.
2. **Isolate mechanical noise from semantic changes.** Pure-mechanical sweeps get
   their own commits with no logic edits mixed in: the Project.toml dep swap, the
   `only(back(s))` → `back(s)` sweep, JuliaFormatter runs.
3. **Deletions stand alone.** Dropping `chain_utils.jl`, `modelindex.jl`,
   `copy_layer` etc. are pure-removal commits — an all-red diff reviews in
   seconds and keeps the replacement commit small.
4. **Additive-first ordering.** New machinery (`FrozenLayer`, `layer_pullback` +
   its Zygote cross-check tests) lands before the commits that wire it in.
5. **Tests ride with the feature they cover**, so every commit is green with
   respect to the suite *as it exists at that commit*. Unavoidably-red
   intermediate commits are confined to Phase 2 and marked `[red]` in the
   subject.
   **Never delete tests** (user directive): v3 tests whose feature isn't ported
   yet are adapted to the Lux API and marked `@test_skip`/`@test_broken` instead
   — either guarded on `isdefined(RelevancePropagation, :SymbolName)` so they
   activate automatically when the port lands (test_rules.jl), or as plain
   `@test_skip` lines to un-skip in the porting commit (test_utils.jl,
   test_chain_utils.jl). Whole v3 files that can't even be `include`d without
   Flux keep `@test_broken` markers in runtests.jl until their phase.
6. **Generated artifacts in dedicated commits.** JLD2 reference regeneration is a
   binary blob — commit the generating script first, the blobs second, never
   mixed with logic changes.
7. **Docs commits are code-free** (Literate rewrite, README, Tullio cleanup).

## Decisions

**Locked:**
- Raw Enzyme, no DifferentiationInterface (user decision: Lux+Enzyme specific,
  more control). Consequence: third-party rules / other ADs out of scope for v4;
  the docs' custom-rules guide teaches the generic `lrp!` in terms of the internal
  helper, which keeps the door open.
- Enzyme is a hard dependency; no `backend` kwarg.

**Assumed (veto if wrong):**
- Flux support dropped entirely — no dual-framework package extensions.
- `canonize` is ported, not dropped (needed for BN-VGG workflows), scheduled last
  among utilities.
- GPU out of scope for v4 (Enzyme CPU; Reactant later).
