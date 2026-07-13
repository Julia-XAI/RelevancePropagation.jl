# Handoff — v4.0.0 Lux/Enzyme port

**Branch:** `ah/enzyme` · **Plan:** PLAN.md (phases 1–6 complete; phase 7
partially done) · **Porting notes:** NOTES.md

## Current state

- Phase 7 test/lint/benchmark work is committed through `9110c58`
  (`phase 7: restore JET tests`). Working tree clean.
- Last full suite run: **584 pass, 0 failed, 0 broken** in ~4.5 min,
  including CNN reference tests, batch tests, linting
  (JuliaFormatter/Aqua/JET/ExplicitImports) and the PkgJogger benchmark
  suite. No "Not yet ported" markers remain in runtests.jl.
- Phase 7 commits so far, in order: stale CNN ref deletion → CNN/batch
  test port → regenerated CNN JLD2 refs → JuliaFormatter sweep →
  `static` import fix (Static.jl is a new direct dep) → linting
  re-enabled → benchmark port → JET restoration.

## Remaining: TTFX, docs, release

### TTFX measurement (measured — write up in NOTES.md)

- Scripts: `<scratch>/ttfx_v4.jl`, `<scratch>/ttfx_v3.jl` where
  `<scratch>` = `/private/tmp/claude-501/-Users-hill-Developer-Julia-XAI-RelevancePropagation-jl/2d61daed-43ac-4721-a9cd-a1c8601451e4/scratchpad`.
  Both build a random-init VGG16 (224×224 input) and time load /
  construct / first analyze / second analyze with `EpsilonPlusFlat()`.
- Results (also in `<scratch>/ttfx_v{3,4}_result.txt`):

  | | v3 (Flux/Zygote, Julia 1.11.9) | v4 (Lux/Enzyme, Julia 1.12.6) |
  |---|---|---|
  | load package | 1.0 s | 1.2 s |
  | construct analyzer | 2.9 s | 1.4 s |
  | **first analyze (TTFX)** | **9.2 s** | **32.3 s** |
  | second analyze (warm) | 1.2 s | 1.8 s |

  v4 TTFX is ~3.5× v3 (Enzyme thunk compilation dominates). Caveat: the
  v3 baseline runs on Julia 1.11 because v3.0.0 pins Flux 0.14, which
  crashes on 1.12 (`Core.Compiler._return_type`) — not an
  apples-to-apples Julia version. Remaining work: record these numbers +
  caveat in NOTES.md (feeds the CHANGELOG); optionally investigate warm
  `analyze` (1.8 s vs 1.2 s) with the PkgJogger suite before release.

### `flatten_model` unwrapping of wrapper layers (small src feature, do first)

Needed so the docs/README VGG example via Boltz.jl works. Verified facts:

- Boltz model wrappers (`Vision.VGG <: AbstractBoltzModel <:
  AbstractLuxWrapperLayer{:layer}`, `VGGFeatureExtractor <:
  AbstractLuxWrapperLayer{:model}`, similarly `VGGClassifier`,
  `Layers.ConvNormActivation`, `Layers.ConvNormActivationBlock`) are all
  pure pass-through: `ps`/`st` flow to the wrapped layer unchanged, and
  the wrapped layers are plain Lux `Chain`s/`Conv`/`Dense` etc.
  Boltz VGG16 ends in `Dense(4096 => 1000)` — **no output softmax, no
  BatchNorm** (default), so neither `strip_softmax` nor `canonize` is
  needed in the example.
- Implementation: in `src/layer_utils.jl` add
  `flatten_layer(l::AbstractLuxWrapperLayer{field}, ps, st) where {field} =
  flatten_layer(getfield(l, field), ps, st)` plus the analogous top-level
  `flatten_model` method (Boltz models are wrappers at top level, not
  `Chain`s). `SkipConnection` keeps its container via its existing, more
  specific method — dispatch handles it.
- `AbstractLuxWrapperLayer` is owned by LuxCore ⇒ `using LuxCore:
  AbstractLuxWrapperLayer` and add LuxCore (v1.5.3 in Manifest) to
  Project.toml deps + compat, or ExplicitImports fails like `static` did.
- Test with a custom wrapper defined in test_utils.jl (no Boltz test
  dep); e2e Boltz check can run in `<scratch>/boltzenv` (has Boltz + Lux;
  `Pkg.develop(path=<repo>)` the package into it first).
- Pretrained weights: `Vision.VGG(16; pretrained=true)` loads them during
  `Lux.setup` (verified via Boltz docs).

### Docs rewrite (next big chunk; code-free commits except the JLD2 blob)

Groundwork done:

- **Pre-trained LeNet-5 converted**: `<scratch>/model.jld2` holds the
  Lux-style `ps` NamedTuple (key `"ps"`), converted from
  `docs/src/model.bson` by `<scratch>/convert_lenet.jl` — logits match
  the Flux model **bit-exactly**. Commit script + blob per commit
  strategy (script first, blob second); delete `docs/src/model.bson` in a
  standalone removal commit. Docs then do
  `ps = load("../model.jld2", "ps"); st = Lux.setup(rng, model)[2]` with
  the LeNet architecture written out (Conv(5,5,1=>6,relu), MaxPool,
  Conv(5,5,6=>16,relu), MaxPool, FlattenLayer, Dense(256=>120,relu),
  Dense(120=>84,relu), Dense(84=>10)).
- File-by-file plan:
  - `docs/Project.toml`: drop Flux, BSON; add Lux, JLD2, StableRNGs,
    Boltz (only if the composites example actually runs VGG; the README
    example is not executed). MLDatasets/Image*/VisionHeatmaps stay.
    Check whether ColorSchemes/Distributions are still used.
  - `basics.jl`: Lux triple API (`Lux.setup`, `LRP(model, ps, st, …)`);
    `strip_softmax(model)` stays model-only; `canonize(model, ps, st)`
    and `flatten_model(model, ps, st)` return triples; **no `flatten`
    kwarg on `LRP` anymore** — v3 flattened by default, v4 is explicit.
    Drop the GPU section (GPU out of scope for v4) and the stale
    Tullio/LV section; consider a TTFX note instead (Enzyme thunks
    compile on first `analyze` per input type).
  - `composites.jl`: rules vector on the flattened triple; nested rules
    are plain **NamedTuples mirroring `ps`** (ChainTuple/ParallelTuple
    are gone); `GlobalTypeMap` with Lux types (`FlattenLayer` instead of
    `typeof(Flux.flatten)`); `show_layer_indices` returns KeyPath
    NamedTuples; `LayerMap` accepts `KeyPath`, integer, or tuple
    (`LayerMap((1, 5), rule)` → `KeyPath(:layer_1, :layer_5)`);
    positional primitives use top-level positions.
  - `crp.jl`: LeNet triple; `CRP(LRP(...), 3, features)`; model already
    flat. CRP assumes flat models (docstring documents this).
  - `custom_layer.jl`: custom layers must subtype
    `Lux.AbstractLuxLayer` to live in a `Chain` (see test_checks.jl for
    the exact pattern); bare functions become `WrappedFunction`;
    registration API unchanged (`LRP_CONFIG.supports_layer(::MyLayer) =
    true`, `supports_activation`, `skip_checks=true`).
  - `custom_rules.jl`: `modify_parameters` unchanged in spirit (operates
    on arrays); `is_compatible` doc line becomes "layer `ps` has a
    `:weight` key"; the LeNet rules vector has 8 entries. `modify_layer`
    operates on `FrozenLayer`s and *returns the layer unchanged* for
    weight-less layers (not `nothing` — update the perf-tip text).
  - `developer.md`: AD fallback section rewritten around
    `layer_pullback` (`src/autodiff.jl`, split-mode Enzyme, seed after
    primal, `back` single-use, `c = back(s)` not `only(back(s))`);
    LRP-struct section: rules/modified layers are NamedTuples mirroring
    `ps`, `FrozenLayer` bundles (layer, ps, st); replace the `@tullio`
    Dense example with the actual generic `lrp!` from src/rules.jl:12-20
    plus a plain-matmul Dense variant.
  - `index.md`, `README.md`: "for use with Lux.jl models"; README
    example: Boltz `Vision.VGG(16; pretrained=true)` → `Lux.setup` →
    `flatten_model` → `LRP(model, ps, st, EpsilonPlusFlat())`. Keep the
    existing heatmap asset links.
  - `rules.md`, `api.md`: drop the ChainTuple/ParallelTuple/
    SkipConnectionTuple "Manual rule assignment" section; check rules.md
    for Flux references.
- Docs build check: `docs/make.jl` runs Literate with **execution**, so
  the examples must actually run with the docs env. MNIST via MLDatasets
  needs `ENV["DATADEPS_ALWAYS_ACCEPT"]` in CI (check existing workflow).

### Release prep

- CHANGELOG from NOTES.md (API: triple constructor, NamedTuple rules,
  explicit flatten; behavioral: Lux LayerNorm differences, exact BN
  fusion; deps; TTFX numbers).
- Compat: julia ≥ 1.10 already set; pin Enzyme (currently 0.13.181);
  decide CI matrix (Enzyme lags new Julia minors — 1.10/1.11/1.12?).
  Bump version 4.0.0-DEV → 4.0.0. Check `.github/workflows` for the CI
  matrix and docs-build env vars. Tagging/registration is the user's call.

## Cross-task insights

- **The linting testset formats the entire repo** (`JuliaFormatter.format`
  on the package dir: src, test, benchmark, docs `.jl` files). Never edit
  *any* repo `.jl` file while a suite runs — a mid-run write caused the
  only failure in an otherwise green run. Draft in the scratchpad and move
  files in after the run.
- **PkgJogger 0.6.0 (latest) is incompatible with BenchmarkTools ≥ 1.6**:
  its `@test_benchmarks` calls the internal 2-arg `samplefunc`.
  test/test_benchmarks.jl collects the suite via `PkgJogger.@jog` and runs
  each leaf with `run(bench; samples=1, evals=1)` instead. Don't revert.
- **JET + Aqua are user-mandated** in the linting suite (JET was re-added
  by explicit request after its 2024 removal). JET's kwarg is
  `target_modules=(RelevancePropagation,)` — `target_defined_modules` is
  deprecated.
- **ExplicitImports owner rule**: import symbols from their defining
  package (Static.jl for `static`; LuxCore for `AbstractLuxWrapperLayer`
  when the flatten feature lands), adding the package as a direct dep.
- **LRP-0 conservation is draw-dependent**: sums were 0.90/1.07 for the
  Lux seeds vs ≈1 for v3's Flux draws; tests use `atol=0.15` with a
  comment. If reference regeneration changes seeds, expect to revisit.
- **Flux/Lux LeNet/VGG semantics match**: both use true convolution with
  (k,k,in,out) weights, same flatten order — direct `ps` transfer works
  (LeNet conversion was bit-exact). `Dense`/`Conv` bias is a plain vector
  in Lux 1.x.
- **Scratch envs** (all under `/private/tmp/claude-501/-Users-hill-…`):
  `1a40a751-…/scratchpad/env` = dev'ed pkg + test deps (re-resolved for
  Static.jl; good for ~30 s smoke runs from `test/`);
  `2d61daed-…/scratchpad/`: `lint-env` (pkg + JuliaFormatter/Aqua/
  ExplicitImports/JET), `fmt-env` (JuliaFormatter only), `boltzenv`
  (Boltz + Lux), `convenv` (Flux 0.16 + Lux + BSON + JLD2), `v3env11`
  (v3 + Flux 0.14 under Julia 1.11), plus TTFX scripts/results and
  `model.jld2` + `convert_lenet.jl`.
- **Background `Pkg.test()`**: don't pipe through `tail` — the failure
  detail gets cut; redirect full output to a file.

## Testing infrastructure

- Full suite: `julia --startup-file=no -e 'using Pkg;
  Pkg.activate("<repo>"); Pkg.test()'` (~4.5 min). `--startup-file=no` is
  **required**.
- ReferenceTests: missing reference files are created (test passes with
  `@info`); mismatches fail non-interactively. To regenerate: delete, run,
  inspect the new file, commit separately.
- JuliaFormatter (blue style, repo `.JuliaFormatter.toml`): format new
  files before committing; formatting-only changes go in dedicated commits.

## Standing directives (from the user; do not violate)

- **Never delete tests.** Adapt v3 tests to the Lux API and mark
  `@test_broken`/`@test_skip`, guarded on
  `isdefined(RelevancePropagation, :Sym)` where possible so they
  auto-activate when the feature lands.
- **Functions with randomness take an explicit `rng` argument** — never
  mutate the global (or Lux global) RNG.
- **Raw Enzyme, no DifferentiationInterface.**
- **Width-2 Enzyme thunks crash on pooling/normalization layers** — only
  ever request `layer_pullback_2seeds` for weight-bias layers (Dense,
  Scale, Conv, ConvTranspose).
- **Keep JET and Aqua in the linting tests.**
- Commit strategy (PLAN.md §Commit strategy): one commit per checkbox,
  phase-prefixed subject, every commit green, deletions and generated
  artifacts stand alone, trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## After Phase 7

Tag v4.0.0 — the port is complete. Out of scope for v4 (per PLAN
decisions): GPU/Reactant, dual-framework Flux extensions, third-party AD
backends.
