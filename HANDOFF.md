# Handoff — v4.0.0 Lux/Enzyme port

**Branch:** `ah/enzyme` · **Plan:** PLAN.md (phases 1–5 complete, ticked) ·
**Porting notes:** NOTES.md

## Current state

- Phases 1–5 are committed through `3c799dc` (`phase 5: JuliaFormatter sweep
  over phase 5 test files`). The working tree is clean apart from this file,
  PLAN.md checkbox updates and a NOTES.md phase-5 section.
- Last full suite run (at the canonize commit `2ce8867`, before the
  formatting-only sweep): **497 pass, 5 broken, 0 failures** in ~3 min.
  The 5 broken tests are the "Not yet ported" markers in runtests.jl:
  crp (phase 6), cnn / batches / benchmarks / linting (phase 7).
- All phase-5 utilities are exported: `strip_softmax` (model-only),
  `flatten_model(model, ps, st)` and `canonize(model, ps, st)` (joint
  triples, re-keyed to `layer_1..layer_N`). The LayerNorm canonize guard in
  test_rules.jl activated automatically (`ported(:canonize)`).

## Next: Phase 6 (CRP)

Port `src/crp.jl` (v3 file: `git show 2550998~1:src/crp.jl`; the on-disk
copy is identical). The port is near-mechanical:

- `CRP(lrp, layer::Int, features)` wraps an already-constructed `LRP`
  analyzer — no new Lux plumbing. `length(lrp.model)` works on Lux `Chain`s.
- v3 indexes `rules[k]`, `layers[k]`, `modified_layers[k]` **positionally**
  in the backward loops. In v4 these are NamedTuples on the `LRP` struct —
  unpack once via `values(lrp.rules)`, `values(lrp.layers)`,
  `values(lrp.modified_layers)` (see `lrp_backward_pass!` in `src/lrp.jl`),
  then the `k`-loops port unchanged.
- `get_activations(model, input)` → v4's
  `get_activations(lrp.layers, input)` (NamedTuple of `FrozenLayer`s;
  same `(input, a¹, …, aᴺ)` tuple contract).
- `AbstractFeatureSelector`/`TopNFeatures`/`IndexedFeatures` come from
  XAIBase and are framework-agnostic; the feature masking code
  (`R_feature[idx] .= …`) touches plain arrays only.
- CRP assumes a **flat** model (positional `layer::Int`); document that
  users should `flatten_model` first (v3 had the same implicit assumption).
- Port `test/test_crp.jl` (v3 on disk): Flux MLP/CNN → Lux triple via
  `Lux.setup(StableRNG(123), model)`; include in runtests.jl and remove the
  phase-6 marker. No JLD2 references involved.

## Cross-task insights (phases 6–7)

- **Lux `Parallel` does not wrap bare functions** (`Parallel(+, softmax, …)`
  is a `MethodError`) — wrap explicitly in `WrappedFunction`. `Chain` wraps
  automatically. `Parallel` also has **no `getindex`**: address branches as
  `p.layers.layer_i`.
- **Lux layers compare `==` structurally** (immutable): `Dense(2 => 2) ==
  Dense(2 => 2)` is `true`, unlike mutable Flux layers where `==` was
  identity. Never port v3 code that finds a layer by `==` against a model
  element (that's why v4 `strip_softmax` descends positionally).
- **Container rebuilds:** `setproperties(c, (; layers=nt))` works on
  `Chain`/`Parallel`/`SkipConnection` and preserves `connection`/`name`;
  `Chain(::NamedTuple)` preserves keys and `==`-matches varargs
  construction with default keys.
- **Static fields:** `use_bias`/`affine` are `Static.True`/`Static.False`
  type parameters. Flipping requires `Lux.static(true)` (imported as
  `using Lux: static`); a plain `Bool` in `setproperties` throws.
- **LuxLib warns** when a model is applied in train mode outside AD (used
  for collecting BatchNorm stats in canonize tests) — harmless, prints once.
- **Scratch env** at `<session-scratchpad>/env` (previous sessions used
  `/private/tmp/claude-501/-Users-hill-…/1a40a751-…/scratchpad/env`) has the
  package dev'ed plus Lux, Functors, StableRNGs, ConstructionBase,
  ReferenceTests, JLD2, LinearAlgebra, Random — enough to `include` most
  test files directly for ~30 s smoke runs (run from `test/` so reference
  paths resolve).
- **Background `Pkg.test()` needs the repo as the active project** — run
  `julia -e 'using Pkg; Pkg.activate("<repo>"); Pkg.test()'`; a bare
  `Pkg.test()` inherits whatever cwd the shell last used.
- Phase 7 note: `test/Manifest.toml` is a stale v3 leftover (Flux pins,
  Oct 2024) that `Pkg.test` ignores — delete it in the test-infrastructure
  commit to avoid confusion.

## Testing infrastructure

- Full suite: `julia --startup-file=no -e 'using Pkg;
  Pkg.activate("<repo>"); Pkg.test()'` (~3 min). `--startup-file=no` is
  **required**.
- **Never edit `test/` while a suite is running** — test files are
  `include`d at runtime mid-run. `src/` edits are safe once the test
  process has loaded the package (first `@info "Testing …"` in the log).
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
- Commit strategy (PLAN.md §Commit strategy): one commit per checkbox,
  phase-prefixed subject, every commit green, deletions and generated
  artifacts stand alone, trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## After Phase 6

Phase 7 (tests, docs, release) per PLAN.md — not yet started. Includes CNN
reference regeneration (JLD2), Zygote-vs-Enzyme consistency testset,
benchmarks/TTFX, Literate docs with Boltz.jl VGG, CHANGELOG and compat
bounds for the v4.0.0 tag.
