# Handoff — v4.0.0 Lux/Enzyme port

**Branch:** `ah/enzyme` · **Plan:** PLAN.md (all phases complete) ·
**Porting notes:** NOTES.md

## Current state: port complete, ready to tag

Phase 7 is done. The branch contains the full Flux/Zygote → Lux/Enzyme
rewrite, tests (591 pass, 0 fail, ~5 min suite), linting
(JuliaFormatter/Aqua/JET/ExplicitImports), benchmarks, rewritten docs,
CHANGELOG, and the version bump to 4.0.0.

Remaining actions are the **user's call**:

1. **Merge `ah/enzyme`** (history is structured for commit-by-commit
   review; merge without squash).
2. **Tag and register v4.0.0.**
3. Optional follow-ups (out of scope for v4, per PLAN decisions):
   GPU/Reactant support, third-party AD backends, warm-`analyze`
   performance investigation (1.8 s vs 1.2 s in v3, PkgJogger suite).

## What landed this session (in commit order)

- `docs: update PLAN and HANDOFF for phase 7 progress` — carried-over
  hand-off state from the previous session.
- `phase 7: flatten_model unwraps AbstractLuxWrapperLayers` — generic
  wrapper unwrapping (top level + inside Chains) for the Boltz VGG docs
  example; LuxCore is a new direct dep. **Critical detail:** Lux pooling
  layers are themselves `AbstractLuxWrapperLayer{:layer}`s around internal
  pooling ops — an explicit `flatten_layer(::PoolingLayer, ps, st)`
  pass-through keeps them intact (regression-tested in test_utils.jl).
  Verified e2e against Boltz `Vision.VGG(16)`: 24-layer flat Chain,
  outputs match, EpsilonPlusFlat analyze runs.
- `docs: record TTFX measurements in NOTES.md` — VGG16 table + caveats.
- LeNet conversion script (`docs/convert_lenet.jl`), `docs/src/model.jld2`
  blob, `docs/src/model.bson` removal.
- Full docs rewrite for Lux (all five Literate examples, developer.md,
  index.md, api.md, docs/Project.toml), README rewrite (Boltz VGG16
  example).
- CHANGELOG v4.0.0 entry, CI matrix update (dropped `pre` — Enzyme lags
  Julia minors; fixed stale codecov slug), version 4.0.0.

## Docs facts (for future edits)

- `docs/make.jl` runs Literate with **execution** (markdown via
  Documenter `@example` + executed `.ipynb`); build locally with
  `DATADEPS_ALWAYS_ACCEPT=true julia --project=docs docs/make.jl`
  after `Pkg.develop(path=".")` in the docs env. Full build ~10 min
  (MNIST download + Enzyme thunk compilation per example).
- Docs deps: dropped Flux/BSON/ColorSchemes/Distributions; added
  Lux/JLD2/StableRNGs. **Boltz is NOT a docs dep** — the executed
  examples use LeNet-5 and a VGG-like random-init CNN; only the
  (unexecuted) README shows Boltz.
- LeNet-5 params live in `docs/src/model.jld2` (key `"ps"`), loaded as
  `ps = load("../model.jld2", "ps")` relative to `docs/src/generated/`.
  States come from `Lux.setup(StableRNG(123), model)[2]` (all layers
  stateless). Conversion from the old model.bson was bit-exact
  (`docs/convert_lenet.jl`, needs a temp env with Flux 0.16 + BSON).
- Internal AD docstrings (`FrozenLayer`, `layer_pullback`,
  `layer_pullback_2seeds`) are included via `@docs` blocks in
  developer.md — removing them breaks `@ref`s in api.md docstrings
  (Documenter fails on :cross_references).
- GitHub source links in developer.md use `blob/main` (not `master`).
  Documenter `linkcheck=true` can hit GitHub 429 rate limits locally.

## Cross-task insights (kept from previous sessions)

- **The linting testset formats the entire repo** (src, test, benchmark,
  docs `.jl`). Never edit repo `.jl` files while a suite runs.
- **PkgJogger 0.6.0 `@test_benchmarks` is broken with BenchmarkTools ≥1.6**;
  test_benchmarks.jl runs leaves via the public API. Don't revert.
- **JET + Aqua are user-mandated**; JET kwarg is
  `target_modules=(RelevancePropagation,)`.
- **ExplicitImports owner rule**: import symbols from their defining
  package (Static.jl for `static`, LuxCore for `AbstractLuxWrapperLayer`).
- **LRP-0 conservation is draw-dependent**: tests use `atol=0.15`.
- **Scratch envs**: see `/private/tmp/claude-501/-Users-hill-…/`
  session dirs; `1a40a751-…/scratchpad/env` = dev'ed pkg + test deps
  (+ LuxCore), `2d61daed-…/scratchpad/boltzenv` has Boltz + dev'ed pkg.
- **Background `Pkg.test()`**: redirect full output to a file, don't pipe
  through `tail`.

## Testing infrastructure

- Full suite: `julia --startup-file=no -e 'using Pkg;
  Pkg.activate("<repo>"); Pkg.test()'` (~5 min). `--startup-file=no` is
  **required**.
- ReferenceTests: missing reference files are created on first run;
  regenerate by delete + run + inspect + commit separately.
- JuliaFormatter (blue style, repo `.JuliaFormatter.toml`): format new
  files before committing.

## Standing directives (from the user; do not violate)

- **Never delete tests.** Adapt and mark `@test_broken`/`@test_skip`.
- **Functions with randomness take an explicit `rng` argument.**
- **Raw Enzyme, no DifferentiationInterface.**
- **Width-2 Enzyme thunks crash on pooling/normalization layers** — only
  request `layer_pullback_2seeds` for weight-bias layers.
- **Keep JET and Aqua in the linting tests.**
- Commit strategy (PLAN.md §Commit strategy): one commit per checkbox,
  phase-prefixed subject, every commit green, deletions and generated
  artifacts stand alone, trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
