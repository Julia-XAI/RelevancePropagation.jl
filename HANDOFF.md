# Handoff — v4.0.0 Lux/Enzyme port

**Branch:** `ah/enzyme` · **Plan:** PLAN.md (phases 1–6 complete, ticked) ·
**Porting notes:** NOTES.md

## Current state

- Phases 1–6 are committed through `a87672d` (`phase 6: port CRP to Lux`).
- Last full suite run (at the CRP commit): **499 pass, 4 broken,
  0 failures** in ~3 min. The 4 broken tests are the "Not yet ported"
  markers in runtests.jl, all phase 7: cnn, batches, benchmarks, linting.
- `CRP` is exported and tested (`test/test_crp.jl`, analytic MLP). The
  port kept the v3 algorithm; NamedTuples are unpacked positionally via
  `values()`, and the flat-model assumption (positional `layer::Int`) is
  now documented in the docstring — users `flatten_model` first.

## Next: Phase 7 (tests, docs, release)

Per PLAN.md, roughly in commit order:

- **CNN tests + JLD2 regeneration** (`test/test_cnn.jl`,
  `test/test_batches.jl`; v3 files on disk). Port models to the Lux
  triple via `Lux.setup(StableRNG(123), model)`. Model-level JLD2
  references **must be regenerated** — `Lux.setup` draws parameters in a
  different order than Flux init, same seed or not. ReferenceTests
  creates missing reference files on first run (test passes with
  `@info`): delete the old refs, run, inspect, commit. Commit strategy:
  generating script/test port first, binary blobs in their own commit.
- **Zygote-vs-Enzyme consistency testset** — already done: landed with
  phase 2 as `test/test_autodiff.jl` (Zygote is test-only in
  `test/Project.toml`). PLAN checkbox ticked.
- **Linting** (`test_linting.jl`): re-enable JuliaFormatter/Aqua/
  ExplicitImports. Delete the stale `test/Manifest.toml` (v3 leftover
  with Flux pins, Oct 2024, ignored by `Pkg.test`) in the
  test-infrastructure commit.
- **Benchmarks** (`test_benchmarks.jl`, PkgJogger): port; measure
  shadow/thunk preallocation (remember `make_zero!` on reused shadows —
  Enzyme *accumulates* into `dx`; the thunk cache is keyed on input type
  and fills on first `analyze`, not at construction).
- **TTFX**: first-`analyze` latency on a VGG-scale composite,
  before/after comparison against v3.
- **Docs** (code-free commits): Literate rewrite with Lux; VGG composites
  example via Boltz.jl (`Vision.VGG`); README; remove stale Tullio/LV
  content (`basics.jl` advertises a package extension that no longer
  exists; rewrite the `@tullio` custom-rule example in `developer.md`
  with plain broadcasting/matmul).
- **Release**: CHANGELOG, compat bounds (julia ≥ 1.10; pin Enzyme —
  it lags new Julia minors; decide the CI matrix), tag v4.0.0.

## Cross-task insights

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
- **Scratch env** at
  `/private/tmp/claude-501/-Users-hill-…/1a40a751-…/scratchpad/env` has the
  package dev'ed plus Lux, Functors, StableRNGs, ConstructionBase,
  ReferenceTests, JLD2, LinearAlgebra, Random — enough to `include` most
  test files directly for ~30 s smoke runs (run from `test/` so reference
  paths resolve).
- **Background `Pkg.test()` needs the repo as the active project** — run
  `julia -e 'using Pkg; Pkg.activate("<repo>"); Pkg.test()'`; a bare
  `Pkg.test()` inherits whatever cwd the shell last used.

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

## After Phase 7

Tag v4.0.0 — the port is complete. Out of scope for v4 (per PLAN
decisions): GPU/Reactant, dual-framework Flux extensions, third-party AD
backends.
