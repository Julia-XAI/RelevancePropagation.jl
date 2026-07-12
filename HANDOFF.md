# Handoff — v4.0.0 Lux/Enzyme port

**Branch:** `ah/enzyme` · **Plan:** PLAN.md (phases 1–4 complete, ticked) ·
**Porting notes:** NOTES.md

## Current state

- Phases 1–4 are committed through `23a31ec` (`phase 4: regenerate show
  references for Lux output`). The working tree is clean apart from this file
  and PLAN.md checkbox updates.
- Last full suite run (at the show.jl commit `41f72b5`): **430 pass, 21 broken,
  0 failures**, `LRP composites | 9`. The broken tests are deliberate
  `@test_broken`/`@test_skip` markers for unported features.
- `runtests.jl` still carries 7 "Not yet ported" markers: checks + canonize
  (phase 5), crp (phase 6), cnn / batches / benchmarks / linting (phase 7).

## Next: Phase 5 (one commit per checkbox, in this order)

### 1. Model checks / `LRP_CONFIG` tests

`src/checks.jl` is already ported (phase 2): `LRP_CONFIG` module,
`check_lrp_compat`, `print_lrp_model_check`, `lrp_check_layer_type` with
`WrappedFunction` unwrapping. The remaining work is the **tests**:

- Port `test/test_checks.jl` (v3 version: `git show 2550998~1:test/test_checks.jl`):
  `check_lrp_compat` throws/passes, `print_lrp_model_check` reference test,
  `LRP_CONFIG.supports_layer`/`supports_activation` registration with a custom
  `MyLayer` struct and an `unknown_function`.
- Delete the stale v3 `test/references/show/check_lrp_compat.txt` before the
  run so ReferenceTests regenerates it (missing references are created and
  pass; mismatches fail non-interactively). Commit the regenerated file in a
  dedicated artifacts commit.
- v3 used Suppressor to swallow check output — either add it to
  test/Project.toml or replace with `redirect_stdout`.
- Include the file in runtests.jl and remove its "Not yet ported" marker.

**Open Lux API questions** are collected in a runnable probe script:
`<scratchpad>/probe_phase5.jl` (session scratchpad under
`/private/tmp/claude-501/-Users-hill-Developer-Julia-XAI-RelevancePropagation-jl/`;
may vanish on reboot). It probes: whether `Chain(MyLayer(...))` accepts a
non-`AbstractLuxLayer` struct and what `Lux.setup` does with it;
`Chain(::NamedTuple)` construction (needed to preserve keys in
`flatten_model`); `Chain` `getindex`/`length`; `BatchNorm` fields and
`epsilon`; flipping static `use_bias` via
`setproperties(d, (; use_bias=true))` vs `Lux.static(true)`; `Conv` fields for
fusion reconstruction; `WrappedFunction`/`NoOpLayer`/`Dense` equality (`==`)
for strip_softmax tests; the `LayerNorm(shape, identity; affine=false)` split
constructor. **Not yet run** — the scratch env first needs
`Pkg.add("ConstructionBase")`.

### 2. `strip_softmax` (model-only, in checks.jl)

v3 semantics, verified from `git show 2550998~1:src/chain_utils.jl` and
`...:test/test_utils.jl`:

- A bare output softmax (`WrappedFunction(softmax)` in Lux) is **replaced by an
  identity layer** (`NoOpLayer()`), preserving chain length — it is *not*
  removed from the chain.
- `Dense(n => m, softmax)` at the output gets its activation set to `identity`
  (via `remove_activation` in `src/layer_utils.jl`).
- Only the *last* element is touched, descending nested `Chain`s; a
  `Parallel`/`SkipConnection` at the output is left alone.
- `ps` is untouched (activation is layer config in Lux), so the signature is
  `strip_softmax(model)` — no joint tuple needed.
- Export it; un-skip the `@test_skip` strip_softmax block in
  `test/test_utils.jl`. **Warning:** the currently-skipped line
  `strip_softmax(Chain(Dense(2 => 2), softmax)) == Chain(Dense(2 => 2))` is
  **wrong** (contradicts v3 length-preserving semantics) — fix it to expect
  `Chain(Dense(2 => 2), NoOpLayer())` when un-skipping.

### 3. Joint `flatten_model(model, ps, st)`

- Returns `(flat_model, flat_ps, flat_st)`, re-keying everything to
  `layer_1, layer_2, …` (Lux `Chain` does not auto-flatten nested chains).
- Chains-of-chains are spliced; `Parallel`/`SkipConnection` keep their
  container but their branches flatten internally.
- Un-skip the flatten_model `@test_skip` block in `test/test_utils.jl`,
  adapting to the tuple API (build `ps`/`st` via `Lux.setup(StableRNG(...))`).
- Export it.

### 4. Joint `canonize(model, ps, st)`

Rewrite `src/canonize.jl` (v3 file still on disk for reference), include in
the module, export. v3 structure: `canonize_fuse(flatten_model(canonize_split(model)))`.

- **Split LayerNorm** when it is affine OR has activation ≠ identity:
  `LayerNorm(shape, identity; dims, epsilon, affine=false)` followed by
  `Scale(shape, activation)`. Lux LayerNorm `ps` is `(scale, bias)` sized
  `(shape..., 1)` — reshape to `shape` for the Scale layer's `ps`. Decide
  affine-ness via `haskey(ps, :scale)` (canonize has `ps`, unlike v3).
- **Fuse BatchNorm** into a preceding `Dense`/`Conv` whose activation is
  `identity`. Include epsilon (deliberate improvement over v3, which ignored
  it): `scale = γ ./ sqrt.(σ² .+ ϵ)`. Then `W′ = scale .* W` (row-wise for
  Dense; reshape scale to broadcast over the output-channel dim for Conv),
  `b′ = scale .* (b .- μ) .+ β`, or `b′ = β .- scale .* μ` for no-bias layers
  (requires flipping `use_bias` — see probe). The fused layer takes the
  BatchNorm's activation. Fuse repeatedly without incrementing the index
  (v3 behavior) so `Dense → BN → BN` collapses fully.
- Running stats live in `st` (`running_mean`, `running_var`); collect test
  stats via `_, st = Lux.apply(model, x, ps, st)` in train mode, then
  `testmode`.
- Port `test/test_canonize.jl` (v3: `@inferred` canonize_fuse for Dense/Conv +
  output ≈ checks, sequential model 15→9 layers, nested 4, Parallel/
  SkipConnection structure counts).
- Un-skip the LayerNorm canonize sub-guard in `test/test_rules.jl` (it is
  guarded on `isdefined(RelevancePropagation, :canonize)` and activates
  automatically once canonize is exported).

## Testing infrastructure

- Full suite: `julia --startup-file=no -e 'using Pkg; Pkg.test()'` from the
  repo root (~3–8 min). `--startup-file=no` is **required**.
- Fast smoke tests: scratch env at `<scratchpad>/env` with the package
  `Pkg.develop`ed plus Lux/Functors/StableRNGs (~30 s per script). Recreate if
  gone: `Pkg.activate(env); Pkg.develop(path=repo); Pkg.add.(["Lux",
  "Functors", "StableRNGs", "ConstructionBase"])`.
- **Never edit `test/` while a suite is running** — test files are `include`d
  at runtime mid-run. `src/` edits are safe ~30 s in (package precompiles at
  start). Confirm which state ran via the testset pass-counts.
- ReferenceTests: missing reference files are created (test passes with
  `@info`); mismatches fail non-interactively. To regenerate: delete, run,
  inspect the new file, commit separately.

## Standing directives (from the user; do not violate)

- **Never delete tests.** Adapt v3 tests to the Lux API and mark
  `@test_broken`/`@test_skip`, guarded on
  `isdefined(RelevancePropagation, :Sym)` where possible so they
  auto-activate when the feature lands.
- **Functions with randomness take an explicit `rng` argument** — never mutate
  the global (or Lux global) RNG.
- **Raw Enzyme, no DifferentiationInterface.**
- **Width-2 Enzyme thunks crash on pooling/normalization layers** — only ever
  request `layer_pullback_2seeds` for weight-bias layers (Dense, Scale, Conv,
  ConvTranspose).
- Commit strategy (PLAN.md §Commit strategy): one commit per checkbox,
  phase-prefixed subject, every commit green, deletions and generated
  artifacts stand alone, trailer
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## After Phase 5

Phase 6 (CRP) and Phase 7 (tests/docs/release) per PLAN.md — not yet started.
