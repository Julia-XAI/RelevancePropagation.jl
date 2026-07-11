# Plan: Migrate RelevancePropagation.jl to Lux.jl + Enzyme.jl (v4.0.0)

**Branch:** `ah/enzyme` · **Status:** planning complete, no code written yet.

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

### AD core: one Enzyme helper

Keep the `(z, back)` pullback *shape* every rule in `src/rules.jl` is written
against, but implement it once with Enzyme split-mode thunks. All Enzyme
incantations live in a single file; rules stay readable.

```julia
# Sketch — the only place Enzyme appears in the package
function layer_pullback(f::F, x::AbstractArray) where {F}   # f = StatefulLuxLayer(layer, ps_modified, st)
    fwd, rev = autodiff_thunk(ReverseSplitWithPrimal, Const{F}, Duplicated, Duplicated{typeof(x)})
    dx = make_zero(x)
    tape, z, dz = fwd(Const(f), Duplicated(x, dx))
    back(s) = (dz .= s; rev(Const(f), Duplicated(x, dx), tape); dx)
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
   seeds in one forward+reverse.
3. **Preallocated shadows / cached thunks.** `LRP` precomputes modified layers at
   construction; `dx` shadow buffers and thunks can be preallocated per layer.
   Measure with the existing PkgJogger suite.
4. **Activity is trivial.** Wrap `(layer, ps_modified, st)` in a `StatefulLuxLayer`
   annotated `Const` — ps/st constant by construction; same pattern as Lux's own
   Enzyme docs. NNlib ships an EnzymeCore extension with custom rules for
   `conv!`/`pool!`, so CNN hot paths don't rely on Enzyme differentiating im2col.
5. **Fallback** if split mode misbehaves on some layer: combined `Reverse` with a
   mutating wrapper (redundant forward, identical correctness), per-layer, behind
   the same helper interface.

### Simplification wins from Lux

1. **`copy_layer` dies** (`src/layer_utils.jl:34-46`). Parameters live in `ps`, so
   `modify_layer` collapses into `modify_parameters(rule, ps)` returning a modified
   `ps` NamedTuple. No per-layer-type reconstruction (Dense/Conv/ConvTranspose/
   CrossCor/Scale). The only remaining layer modification is swapping the
   activation for `identity` once (the linearized layer) — uniform across Lux
   layers, which all carry an `activation` field.
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
7. **Dependency diet.** Drop: Flux, Zygote, MacroTools, MLUtils.
   Final set: Lux, Enzyme, Functors, NNlib, XAIBase (+ stdlib).
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

## Phases

### Phase 1 — Spike (de-risk first)
- [ ] Scratch script: `autodiff_thunk`/`ReverseSplitWithPrimal` + `BatchDuplicated`
      through `StatefulLuxLayer`-wrapped `Dense`, `Conv`, `MaxPool`, testmode
      `BatchNorm`, `LayerNorm` on CPU; compare against Zygote VJPs.
- [ ] Check whether `set_runtime_activity` is needed anywhere (ideally not).
- [ ] Verify Lux API details assumed in this plan: `Scale` parameter names;
      whether Lux 1.x `Chain` auto-flattens nested chains; `CrossCor` ↔
      `Conv(cross_correlation=true)` type-level implications for `ConvLayer` union.

### Phase 2 — Core skeleton
- [ ] Swap deps in Project.toml (add Lux, Enzyme, Functors; drop Flux, Zygote,
      MacroTools, MLUtils).
- [ ] Lux layer-type unions (`src/layer_types.jl`).
- [ ] `layer_utils` on `ps` (activation field, `haskey(ps, :weight/:bias)`).
- [ ] Activations collector (loop over `model.layers`, threading `st`).
- [ ] New `LRP` struct: `model, ps, st_test, rules::NamedTuple, modified_ps::NamedTuple`.
- [ ] `layer_pullback` Enzyme helper; generic `lrp!` using it.
- [ ] `ZeroRule` + `EpsilonRule`; one end-to-end MLP test green.

### Phase 3 — All rules
- [ ] Port `modify_*` family to operate on `ps` NamedTuples.
- [ ] Simple rules: `GammaRule`, `WSquareRule`, `FlatRule`.
- [ ] Multi-variant rules (named-tuple-of-variants convention kept): `ZBoxRule`,
      `AlphaBetaRule`, `ZPlusRule`, `GeneralizedGammaRule` — use `BatchDuplicated`
      for the two-seed pullbacks.
- [ ] `LayerNormRule` against Lux `LayerNorm`; `PassRule`.
- [ ] Keep fast paths (`src/rules.jl:583-597`): Zero/Epsilon on dropout/reshaping
      layers, FlatRule on Dense.

### Phase 4 — Dataflow + composites
- [ ] `Chain`/`Parallel`/`SkipConnection` `lrp!` recursion with per-branch `ps`/`st`.
- [ ] Composites: TypeMaps on Lux layer types; `LayerMap` via `KeyPath`.
- [ ] Rewrite `show.jl` for NamedTuple rules (shrinks).

### Phase 5 — Model utilities
- [ ] Model checks / `LRP_CONFIG` on Lux types (`AbstractLuxLayer`, `WrappedFunction`).
- [ ] `strip_softmax` (model-only, `ps` untouched).
- [ ] Joint `(model, ps, st)` version of `flatten_model` (remap `ps` keys).
- [ ] Joint `(model, ps, st)` version of `canonize`: LayerNorm split, BatchNorm
      fusion into Dense/Conv.

### Phase 6 — CRP
- [ ] Port `crp.jl` (near-mechanical once backward pass is ported).

### Phase 7 — Tests, docs, release
- [ ] Port test models to Lux (StableRNG via `Lux.setup`); regenerate JLD2
      references.
- [ ] Zygote-vs-Enzyme consistency testset for `layer_pullback` (Zygote test-only dep).
- [ ] Keep Aqua, ExplicitImports, JuliaFormatter tests.
- [ ] Port PkgJogger benchmarks; measure shadow/thunk preallocation.
- [ ] Rewrite Literate docs with Lux; VGG composites example via Boltz.jl; README.
- [ ] Remove stale Tullio/LoopVectorization doc content: `basics.jl` advertises a
      Tullio/LV package extension that no longer exists (stale since the
      ExplainableAI.jl split); rewrite the `@tullio` custom-rule example in
      `developer.md` with plain broadcasting/matmul. (The package itself has no
      Tullio/LV deps — nothing to drop from Project.toml.)
- [ ] CHANGELOG, compat bounds (julia ≥ 1.10), tag v4.0.0.

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
