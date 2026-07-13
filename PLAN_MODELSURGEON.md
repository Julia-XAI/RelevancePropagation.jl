# Plan: ModelSurgeon

**Status:** planned · **Decided:** 2026-07-13
**Canonical decision record:** `../REFACTOR.md`, section "Planned: ModelSurgeon".

Two stages:

1. **Now: a `ModelSurgeon` submodule in this package.** Consolidate the generic
   **structural rewrites of the Lux `(model, ps, st)` triple** into a submodule and
   iterate on the API in-repo, where breaking it costs nothing and the design can
   settle.
2. **Later, if it stands the test of time: break it out as `XAIModelSurgeon.jl`.**
   A standalone Julia-XAI package that "modified backprop" XAI methods add as a
   dependency. The trigger is the SmoothDiff Flux/Zygote → Lux/Enzyme port needing it
   as a real dependency — that port is the second consumer that validates the design.

Division of responsibility in both stages: **ModelSurgeon owns structure, RP owns AD.**
No pullback or Enzyme machinery goes in. XAIBase stays framework-free; ModelSurgeon is
the Lux-specific sibling, with no dependency between the two.

## Why shared

- SmoothDiff's Flux `prepare` already half-reimplements `canonize_split` (splitting
  fused activations out of `Dense`/`Conv` via `linear_copy` to insert accumulator
  layers), and its Lux port also needs BatchNorm fusion. In Lux its accumulator counts
  live in `st`, making `prepare` a joint triple rewrite.
- Upstream doesn't cover this — see the inventory below. Flattening and BatchNorm
  fusion are generic inference-graph transforms, so pieces may be worth offering to
  LuxDL once stable.

## Upstream inventory (checked against Lux v1.31.4)

What Lux and `Lux.Experimental` already provide, and where the gaps are:

- **`Lux.Experimental.layer_map(f, l, ps, st)`** is the only joint triple traversal
  upstream: `fmap_with_path(l, ps, st; walk=LayerWalkWithPath(), exclude=layer_map_leaf)`,
  calling `f(layer, ps, st, kp) -> (layer, ps, st)` at leaves
  (`src/contrib/map.jl`). Three findings:
  - Its leaf policy `layer_map_leaf` **hard-codes all 9 pooling wrappers as leaves**
    (Max/Mean/LP × plain/Global/Adaptive) — upstream hit exactly the problem behind
    RP's `PoolingLayer` carve-out and solved it the same inextensible way. The
    predicate is internal, not a kwarg; extending it from outside would be piracy.
    So `layer_map` cannot host RP's or SmoothDiff's policies as-is, and an
    `exclude` kwarg for `layer_map` is a concrete upstream offering once our design
    settles.
  - It is **structure-preserving per container** (`perform_layer_map` rebuilds each
    container with the same keys), but a leaf `f` may return a *replacement
    subtree* — so `canonize_split`'s `LayerNorm → Chain(norm, scale)` is
    expressible as a `layer_map`; splicing (`flatten_model`) and adjacent-sibling
    fusion (`canonize_fuse`) are not.
  - It lives under a **"No SemVer Guarantees"** banner (historically stable with
    deprecation periods, `@public` but unexported). Don't build the submodule's
    core on it; do mirror its conventions — the `LayerWalkWithPath` pattern is
    ~30 lines to own on top of `Functors.fmap_with_path`.
- **KeyPath convention matches ours**: `layer_map`'s `KeyPath` follows the `ps`/`st`
  structure (wrapper layers like `SkipConnection` are transparent) — the same
  convention as RP's `layer_indices`. Keep that alignment.
- **Nothing upstream** for chain splicing, BatchNorm fusion, or activation
  splitting — in `Lux`, `Lux.Experimental` (rest: `FrozenLayer`/`freeze`,
  `share_parameters`, `DebugLayer`), or `LuxCore`.
- **Wrapper-unwrapping is unsound as a default.** *Every* Lux container is an
  `AbstractLuxWrapperLayer` — including `Maxout` (`{:layers}`, wrapping a raw
  `NamedTuple`) and `RepeatedLayer` (`{:model}`). The trait guarantees `ps`/`st`
  *transparency*, not application transparency. RP's current generic
  `flatten_layer(::AbstractLuxWrapperLayer)` therefore silently corrupts both:
  `Maxout` unwraps to a bare `NamedTuple` posing as a layer, and `RepeatedLayer`
  unwraps to a single application of its inner model. (Chain/Parallel/
  SkipConnection escape only because specific methods take precedence.) See design
  question 2.
- **Name clashes to keep in mind** (matters once ModelSurgeon docs face Lux users):
  `Lux.Experimental.FrozenLayer` (parameter freezing) vs. RP's `FrozenLayer`
  (bundled callable triple), and Lux's internal `Lux.PoolingLayer` struct (the op
  all pooling wrappers wrap) vs. RP's exported `PoolingLayer` union.

## Stage 1: the submodule

The submodule is the package rehearsal — it must be self-contained, as if it already
lived in its own repo:

- `include` it **first** in `src/RelevancePropagation.jl`, before every other source
  file, so it physically cannot reference RP types (`FrozenLayer`, `PoolingLayer`,
  rules, composites).
- It does its own imports (Lux/LuxCore, the NNlib softmax functions,
  `ConstructionBase.setproperties`, `Static.static` — `canonize_fuse` sets
  `use_bias=static(true)` — and `Functors` for `fmap_with_path`/`KeyPath` once
  `map_triple` lands). RP pulls names in with `using .ModelSurgeon: ...`
  and keeps exporting `flatten_model`, `canonize`, `strip_softmax` unchanged — the
  user-facing API doesn't move, so this can land on `ah/enzyme` at any point, including
  within v4.0.0.
- RP-specific policy is injected from outside the submodule (see design questions) —
  exactly the extension mechanism the future package needs.

### What moves into the submodule

| From | What | Notes |
|---|---|---|
| `src/chain_utils.jl` | `children_layers`, `map_layers`, `chainall`, `first_element`, `last_element`, `flatten_model`/`flatten_chain`/`flatten_layer` | everything except the `FrozenLayer` helpers and `get_activations` |
| `src/canonize.jl` | `canonize`, `canonize_split`, `canonize_fuse`, `is_fuseable`, `fuse_weight` | moves wholesale |
| `src/checks.jl` | `strip_softmax`, `strip_output_softmax`, `has_output_softmax`, `is_softmax` | the LRP-check wrapper `check_output_softmax` stays in RP |
| `src/layer_types.jl` | `SoftmaxActivation` (line 34), `DataflowLayer` (line 2) | plain unions over Lux/NNlib types, needed by the movers |
| `src/layer_utils.jl` | `activation_fn(layer)`, `remove_activation` | generic single-layer property helpers; RP extends the submodule's `activation_fn` with its `FrozenLayer` method from outside |

### What stays out (RP proper)

- `FrozenLayer` and all per-layer Enzyme pullback machinery (`src/autodiff.jl`).
- `frozen_children` / `frozen_inner` from `chain_utils.jl` — relocate into
  `src/autodiff.jl` next to `FrozenLayer`.
- `get_activations` from `chain_utils.jl` — despite its generic look, it is an
  *execution* helper, not a structural rewrite: every call site (`lrp.jl`, `crp.jl`)
  passes a NamedTuple of callable `FrozenLayer`/modified-layer wrappers, and raw Lux
  layers aren't callable without `ps`/`st`, so it never operates on Lux structure.
  It also sits on the hot path (once per `analyze` call) and relies on tuple
  recursion for inferrability — relocate it next to `FrozenLayer` too.
- `has_weight` / `has_bias` (`layer_utils.jl:24–25`) — these have *only*
  `FrozenLayer` methods, so there is no generic function to extend; they stay in
  RP wholesale. Only `activation_fn` is a submodule function that RP extends
  (fine after extraction too: RP owns `FrozenLayer`, so it's not piracy).
- `PoolingLayer` and all rule/composite dispatch.
- `check_output_softmax` (LRP-specific error text).

### Tests

Existing coverage maps onto the split as follows; no tests are deleted:

- `test/test_canonize.jl` (199 lines) targets the submodule wholesale.
- `test/test_chain_utils.jl` (85 lines) is **mixed**: the `map_layers` testset
  targets the submodule, but the `layer_indices`/`keypath_in` half (lines 55–85)
  tests composite machinery from `src/composite.jl` and stays RP-targeted. Split
  the file along that line so stage 2 can move test files without untangling.
- `test/test_utils.jl`: both the `strip_softmax` testset and the
  `chainall`/`first_element`/`last_element` testset target the submodule; the rest
  (`stabilize_denom` etc.) stays RP.
- End-to-end coverage of flattening/canonization through `test_cnn.jl` /
  `test_lrp.jl` is unaffected.

## Rewrite license and performance ground rules

The move is a license to simplify, abstract and rewrite these utilities — not a 1:1
relocation. The constraint on any abstraction is how Julia performance works: **type
stability**. Concretely:

- Everything in the submodule is a **construction-time** transform, run once per
  model, never per `analyze` call. Dynamic accumulation like `flatten_chain`'s
  untyped `[], [], []` vectors is acceptable there — what matters is that the
  *returned* triple is concretely typed (it is: `Chain(layers...)` and
  `NamedTuple{ks}(Tuple(...))` land on concrete types), because it feeds RP's hot
  loop. Don't let a rewrite blur this line by moving hot-path helpers in
  (see `get_activations` above).
- Policy hooks must be **function arguments or dispatch** — both specialize and stay
  inferrable — never runtime flags or `Symbol` options. The linting testset runs JET;
  keep the module JET-clean.
- Per-call helpers (if any ever land here) use tuple recursion in the lispy
  `_activations` style, not loops over heterogeneous collections.

## API design questions (iterate here, in the submodule)

1. **One traversal primitive instead of three — decided: `map_triple(f, model, ps,
   st; exclude)`.** `flatten_layer`, `canonize_split` and `canonize_fuse` each
   hand-roll the same container recursion over `Chain`/`Parallel`/`SkipConnection`
   (rebuild `layers` NamedTuple, re-key `ps`/`st`, `setproperties`). Extract a
   single triple-walking primitive that all three become thin consumers of. This is
   the submodule's real product: SmoothDiff's `prepare` is then just another `f`,
   and the leaf policy (question 2) has exactly one place to live. Follow upstream
   conventions so a later LuxDL offering is a rename, not a redesign: build on
   `Functors.fmap_with_path` with our own walk (the `LayerWalkWithPath` pattern
   from `Lux.Experimental`, ~30 lines — don't depend on the no-SemVer module
   itself), name the leaf kwarg `exclude` with signature `exclude(kp::KeyPath,
   layer)` (Functors convention), and give `f` `layer_map`'s shape
   `f(layer, ps, st, kp) -> (layer, ps, st)`. Since a leaf `f` may return a
   replacement subtree, `canonize_split` reduces to one `map_triple` call; only
   splice (flatten) and sibling-fusion need logic beyond it.
2. **Leaf and unwrap policy for `flatten_model`.** The hard-coded
   `flatten_layer(l::PoolingLayer, ps, st) = l, ps, st` carve-out (`chain_utils.jl:144`)
   is RP policy — its rules dispatch on intact pooling wrappers. Constraints on the
   replacement hook:
   - It must take **precedence over** wrapper unwrapping — Lux pooling layers *are*
     wrapper layers, which is why the carve-out exists at all. Match Lux's own leaf
     table: all 9 pooling types incl. the LP family (RP's `PoolingLayer` union
     misses `LPPool`/`GlobalLPPool`/`AdaptiveLPPool`; today those get type-erased
     to the internal op by the generic unwrap).
   - The **unwrap default must flip** (see upstream inventory): unwrapping every
     unknown `AbstractLuxWrapperLayer` silently corrupts `Maxout` and
     `RepeatedLayer`. Unwrap needs to be opt-in for known application-transparent
     wrappers (the Boltz model wrappers that motivated it), with unknown wrappers
     kept intact as leaves — or made a second hook next to `exclude`.
   - It must survive extraction **without type piracy**: after stage 2, RP defining
     an overloadable-trait method like `XAIModelSurgeon.isleaf(::MaxPool) = true`
     pirates a foreign function on foreign (Lux) types — Aqua in every package's
     linting flags this. Prefer the predicate *argument* (specializes, stays
     inferrable). RP keeps its zero-config UX by exporting its own thin
     `flatten_model` wrapper that bakes in the pooling policy — shadowing, not
     extension.
3. **Fusion as an overloadable pair.** Document `is_fuseable(l1, l2, st2)` +
   five-argument `canonize_fuse` as the extension API so consumers can add fusions
   beyond Dense/Conv+BatchNorm. (Consumers extending these on their *own* layer
   types is piracy-free.)
4. **Generalized activation splitting.** `canonize_split` today only splits `LayerNorm`.
   SmoothDiff needs the same operation for `Dense`/`Conv` (activation out into a
   separate layer so it can be replaced). Find the shared shape — e.g. a
   `split_activation(layer, ps, st)` hook that both `canonize_split` and a future
   SmoothDiff `prepare` build on.
5. **Naming and exports.** Which traversal helpers are public API vs. internal, and
   whether `flatten_model` keeps its name outside the RP context. Note
   `first_element` has no remaining `src/` call sites (v3 leftover; composites now
   address layers via `KeyPath`) and `last_element`'s only consumer
   (`has_output_softmax`) moves along with it — decide whether `first_element`
   earns its keep as public API or is dropped in the rewrite.

## Stage 2: extraction to XAIModelSurgeon.jl (gated)

Only if the submodule design has stood the test of time. Preconditions:

- The API held up against a second consumer — at minimum a SmoothDiff `prepare`
  prototype written against the submodule.
- RP's v4 finish-line work is done, since it churns exactly these files: the GPU pass
  (`PLAN_GPU.md`; e.g. `canonize_split`'s CPU-only `ones(Float32, l.shape)` must become
  device-aware) and the Runic migration.

Extraction notes:

- Rename the module `ModelSurgeon` → `XAIModelSurgeon` (a package's root module must
  match the package name).
- New repo in the Julia-XAI org at `0.1.0-DEV`: Runic CI job, shared `test/linting.jl`
  (Aqua + JET, JET gated to Julia ≥ 1.12), standard CI matrix. Project.toml deps:
  Lux, LuxCore, Functors, ConstructionBase, Static, NNlib (with compat bounds). The
  submodule-targeting unit-test files move with the code (the RP-targeting halves
  identified under "Tests" stay); RP keeps a re-export smoke test.
- Not breaking for RP users: RP depends on XAIModelSurgeon and keeps exporting
  `flatten_model` / `canonize` / `strip_softmax` — as re-exports, or as thin
  policy-baking wrappers where design question 2 requires one. The extraction
  therefore doesn't need
  to join the v4/v5 release wave, but XAIModelSurgeon `0.1.0` must be registered before
  any RP release that depends on it.
- Docs: the moving docstrings are referenced in `docs/src/api.md` and the literate
  sources `basics.jl` / `composites.jl` / `crp.jl` — point them at XAIModelSurgeon and
  add the package to XAIDocs' MultiDocumenter aggregation.

## Checklist

Stage 1 (submodule):

- [ ] Create the `ModelSurgeon` submodule (`src/ModelSurgeon/`), included first in
      `RelevancePropagation.jl`; move the code per the table above
- [ ] Extract the shared triple-traversal primitive (design question 1) and rebuild
      flatten/split/fuse on it
- [ ] Replace the `PoolingLayer` flatten carve-out with the piracy-free leaf-policy
      hook (design question 2); RP's pooling policy lives outside the submodule;
      flip the wrapper-unwrap default (fixes `Maxout`/`RepeatedLayer` corruption)
      and cover the LP pooling family
- [ ] Relocate `frozen_children`/`frozen_inner`/`get_activations` to
      `src/autodiff.jl`; re-point RP's `FrozenLayer` method at the submodule's
      `activation_fn`; keep `has_weight`/`has_bias` in RP
- [ ] RP: `using .ModelSurgeon`, exports unchanged; split `test_chain_utils.jl`
      into submodule- and RP-targeting halves per "Tests"
- [ ] Iterate on design questions 1–5; sketch a SmoothDiff `prepare` against it

Stage 2 (extraction, gated on the above holding up):

- [ ] GPU pass (`PLAN_GPU.md`) and Runic migration landed, v4 shipped
- [ ] Create `Julia-XAI/XAIModelSurgeon.jl` (`0.1.0-DEV`) with ecosystem-standard
      CI/linting; move submodule + unit tests, rename module
- [ ] RP: depend on XAIModelSurgeon, re-export, add re-export smoke test
- [ ] Docs cross-references + XAIDocs entry
- [ ] Register `0.1.0` before any RP release depending on it; update `../REFACTOR.md`
      and this file as steps land
