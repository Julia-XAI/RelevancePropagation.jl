# Plan: GPU support — Metal and JLArrays now, CUDA next

GPU support is in scope for the upcoming "4.0.0" release of RP.jl.
Since this will be a breaking release anyway,
we should make the necessary changes to support GPUs now.

Companion to `PLAN_REACTANT.md`,
which records the discarded Reactant and "stop-gradient" approaches;
this document covers generic GPU-array support
and corrects the first-pass GPU assumptions that turned out to be wrong
(see "Corrections to the first-pass GPU notes" below).

**Design principle:** every in-package change must be array-type-generic —
correct for `Array`, `JLArray`, `MtlArray` and (by construction) `CuArray`.
Backend-specific code (Enzyme integration for a particular backend)
belongs upstream, not here.
JLArrays.jl is the enforcement mechanism:
it runs on standard CI with no GPU hardware,
so the generic test path can gate every PR.
CUDA compatibility is *assumed* to follow from this
(rationale under "Why this should transfer to CUDA")
and will be tested in follow-up work on real hardware.

**Status:**

- *2026-08-17 (Metal):* LRP produces correct relevances on Metal
  after four changes, three in this package and one upstream in Metal.jl.
  Verified end-to-end on a `Conv`/`MaxPool`/`FlattenLayer`/`Dense` CNN:
  Metal matches the CPU reference to `3.0e-7` relative
  (`2.4e-7` max absolute deviation, Float32).
- *2026-08-18 (JLArrays):* the same changes make JLArray work,
  and nothing less does — the current `dot`-seeded engine fails on JLArray
  with the same `EnzymeNonScalarReturnException` Metal hit,
  and split-mode seeding fixes it on both.
  Split-mode LRP on a `Dense` chain over `JLArray` matches CPU `analyze`
  **exactly** (max deviation `0.0`; JLArray computes on the CPU,
  so exact agreement is the expected outcome, and GPU testsets on JLArray
  can assert near-exact equality rather than a loose `rtol`).
- *2026-08-18 (design decision):* activations are split out of
  rule-carrying nodes at wrap time and handled by `PassRule` nodes;
  `ZBoxRule` adopts the affine-only semantics of its docstring —
  an intentional, documented divergence from v3
  (see "Activity patterns for the rule layer").
- *2026-08-18 (ordering + forward-equivalence tests):* the activation
  split moves to the front of the task list. It is the largest
  structural change, needs no GPU to test, and every later task then
  lands on the final architecture instead of being partially reworked
  by it. Its acceptance gains an explicit forward-equivalence testset:
  the split's core assumption — stripped forward plus broadcast σ
  reproduces the fused layer output exactly — becomes a tested
  invariant instead of an implicit one
  (see "Forward-pass equivalence").
- *2026-08-18 (task 1 LANDED):* the wrap-time activation split is
  implemented (`SplitActivationNode` pair-wrapper; use-site machinery
  deleted; `ZBoxRule` affine-only; CRP forward loop adapted;
  `PassRule` extended to activation-only layers in checks, composite
  matching, the presets and the no-rules constructor).
  All three acceptance criteria hold, and more tightly than required:
  the forward-equivalence testset (`test/test_forward.jl`) passes with
  `==` on every layer/model shape *including gelu* (no `isapprox`
  relaxation was needed); the CPU suite passes with **zero reference
  regeneration** — even `ZBoxRule`'s references are unchanged, because
  the relevance seeded from a ReLU output is zero exactly where the
  affine-only and σ-inclusive variants differ (the divergence is still
  real and documented for non-ReLU activations and nonzero relevance
  at negative pre-activations); and the new un-canonized
  `BatchNorm(…, relu)` end-to-end test matches a manual rule-body
  backward pass with max deviation `0.0`.
  Only the five composite-preset *show* references were regenerated
  (the presets now list `LRPSupportedActivation => PassRule()`).
- *2026-08-18 (tasks 2+3 LANDED):* `make_zero` is gone from the package
  (the four shadow allocations in `src/autodiff.jl` use `Base.zero`,
  and the import is removed), and `call_analyzer` seeds in split mode:
  `lrp_loss`, `detached_mask!` (with its `EnzymeRules.inactive` marker)
  and the `promote_op`-typed `output_ref` are deleted, the model output
  returns as the thunk's primal, and the seed is written into the
  return shadow between forward and reverse. The full cold CPU suite
  passes with zero reference regeneration. Joint acceptance held with
  *no* `EnzymeCore.make_zero` piracy anywhere in the test setup:
  on JLArray, `analyze` end to end is **exact** (dev `0.0`) for
  `ZeroRule`/`EpsilonRule`/`GammaRule`/`WSquareRule`/`FlatRule` on a
  `Dense` chain, and `layerwise_relevances` taps are exact too.
  The Metal re-check on the post-task-1 code also came back exact
  (dev `0.0`) for the `Dense` chain via plain `analyze`;
  CNN-on-Metal re-lands with task 4, which adds the pooling fast
  paths it needs.
- *2026-08-19 (task 4 LANDED):* pooling `input_vjp` fast paths,
  realized as two methods on the `MaxPoolLayer`/`MeanPoolLayer` unions,
  so the adaptive and global variants are covered along with
  `MaxPool`/`MeanPool`. The `PoolDims` come from calling the layer's
  nested pool mode (`layer.layer.mode(x)`) rather than reading its
  fields, exactly as the layer's own forward computes them —
  consistent by construction for the generic, global and adaptive modes.
  Cross-checked against `seeded_pullback` for all six pooling types in
  `test_autodiff.jl` (CPU); the full cold suite passes.
  CNN-on-Metal re-landed as promised: a
  `Conv`/`MaxPool`/`MeanPool`/`FlattenLayer`/`Dense` CNN via plain
  `analyze` matches CPU to `1.5e-8` abs / `1.2e-7` rel (`ZeroRule`)
  and `1.1e-8` abs / `2.0e-7` rel (`EpsilonPlus` composite) —
  Float32 rounding noise, in line with the earlier Metal measurements.
- *2026-08-19 (task 5 stage 1 LANDED):* the generic VJP fallback is
  split-mode. `prepare_vjp(layer, x, ps, st) -> (z̃, pullback)` is the
  two-phase primitive: fast-path layers apply the layer and close over
  their hand VJP, everything else runs the augmented forward of a
  `ReverseSplitWithPrimal` thunk (`thunk_vjp`) and closes over its
  tape; `seeded_pullback` survives as the one-shot wrapper
  (`last(thunk_vjp(...))(s)`), so its call sites (CRP's container
  fallback, the `input_vjp` activation guards, the Zygote cross-check
  testset) are unchanged. The rule bodies consume it per the taxonomy:
  the generic `propagate` shares one prepared pullback between z̃ and
  the VJP (1F+1R on the fallback, was 2F+1R), `ZPlusRule` prepares
  both points (2F+2R, was 4F+2R), `ZBoxRule` prepares the `l`/`h`
  points (3F+3R, was 5F+3R), and `AlphaBetaRule`/
  `GeneralizedGammaRule` prepare their α-/ˡ-points and route the
  second seed through one-shot `input_vjp`s (6F+4R, was 8F+4R;
  the 4F+2R target needs stage 2's width-2 `BatchDuplicated`).
  The combined-mode `dot` loss, the last `Active` scalar below the
  engine, is gone — with it the `autodiff`/`Reverse`/`Active` imports
  and the `LinearAlgebra` dependency. `model_output` moved to
  `src/autodiff.jl`, shared by the engine pass and `thunk_vjp`.
  A new `prepare_vjp vs Zygote` testset asserts primal and pullback
  on every layer in `test_autodiff.jl`; the full cold CPU suite
  passes with zero reference regeneration.
- *2026-08-19 (tasks 6–9 LANDED):* the three GPU-unfriendly spots in
  rule code are broadcast/device-generic —
  `masked_copy` via `ifelse.` (size check kept),
  `zbox_input` allocating through the input
  (`fill!(similar(in), …)` / `copyto!(similar(in, T), …)`),
  and the `FlatRule`/`Dense` fast path via one `sum(…; dims=1)`
  broadcast — all three bit-identical on CPU, zero test changes.
  `BatchNorm` gained `prepare_vjp`/`input_vjp` fast paths:
  testmode BatchNorm over tracked running statistics applies the
  channel-wise slope `γ ./ sqrt.(σ² .+ ϵ)` of its affine map
  (`affine=false` covered; trainmode and `track_stats=false` decline
  to the generic Enzyme fallback, which remains CPU-only).
  Cross-checked in `test_autodiff.jl` against the fallback and Zygote
  (two new BatchNorm variants in the layer list);
  the un-canonized `BatchNorm(…, relu)` end-to-end test now routes
  through the fast path and still passes;
  the full cold CPU suite passes with zero reference regeneration.

Nothing about the engine's design blocks GPU arrays —
the blockers are a missing upstream Enzyme extension,
one design decision in `call_analyzer` (the `dot` seed),
and three GPU-unfriendly spots in rule code.

## Measurement setup

Apple M3 Pro (18-core GPU), macOS Darwin 24.6.0, Julia 1.12.6.
Enzyme v0.13.199, EnzymeCore v0.8.21, Metal v1.10.0,
Lux v1.31.4, LuxLib v1.15.9, NNlib v0.9.44,
GPUArrays v11.5.10, JLArrays v0.3.2.
CUDA v5.8.2 was read for reference but not run (no NVIDIA hardware).

Reproduction environment: a scratch project that `Pkg.develop`s this package
and adds `Metal` (Metal runs) or `JLArrays` (generic runs),
plus `Lux`, `Enzyme`, `Functors`, `NNlib`.
All JLArray probes ran under `JLArrays.allowscalar(false)`
(scalar indexing is also disallowed by default in non-interactive sessions,
so CI gets this strictness for free).
Probe scripts are listed under "Reproducers" below.
Iterate with a DaemonicCabal session
(`--session=RelevancePropagation-metal`)
— Enzyme thunk compilation dominates cold runs
(a single stage cost ~5–6 min cold, seconds warm).
Note that a segfault kills the warm worker;
run crash-prone stages in a throwaway `julia` process
or restart with `juliaclient --project=$PWD --restart`
(the `--project` flag is required — a bare `--restart`
does not target the right worker).

## What already works

Confirmed by direct measurement, no changes required.
"JLArray dev `0.0`" means exact agreement with the CPU result.

| component | Metal | JLArray |
|---|---|---|
| Lux forward passes used by the engine | ✓ | ✓ `Dense`-family; ✗ conv/pool (NNlib, see below) |
| `propagate` on `Dense`: `ZeroRule`, `EpsilonRule`, `GammaRule`, `ZPlusRule`, `AlphaBetaRule`, `WSquareRule`, `FlatRule` | ✓ | ✓ dev `0.0` |
| `propagate` on `Dense`: `LayerNormRule` | ✓ | not measured |
| `input_vjp` fast paths: `Dense` (`Wᵀs`), `Scale` | ✓ | ✓ |
| `input_vjp` fast paths: `Conv` (`∇conv_data`), `ConvTranspose` (`conv`) | ✓ | ✗ NNlib scalar-indexes |
| `NNlib.∇maxpool` / `∇meanpool` (for the fast paths below) | ✓ | ✗ NNlib scalar-indexes |
| `stabilize_denom`, `keep_positive`, `keep_negative`, `ones_like` | ✓ | ✓ |
| `relevance_seed`, incl. `seed[idx] .= 1` / `.= y[idx]` with `idx::Vector{CartesianIndex{2}}` | ✓ | ✓ |
| Enzyme *plumbing*: thunk construction, `Chain` recursion, custom-rule dispatch, shadow accumulation | ✓ | ✓ (via the split-mode run) |

The JLArray ✗ entries are all one fact:
NNlib's conv and pooling kernels have real GPU implementations
only for backends with hardware kernels (cuDNN, LuxLib's Metal fallbacks);
the generic `AbstractArray` fallbacks scalar-index
and are rejected under `allowscalar(false)`.
**A JLArrays job therefore covers `Dense`-family models only** —
CNN paths need Metal (local) or CUDA (follow-up) hardware.

## The four blockers

### 1. `Enzyme.make_zero` aliases its argument on every GPU array

The worst of the four: it fails **silently**, returning wrong relevances
with no error.

```julia
y  = MtlArray(Float32[1, 2, 3])   # identical for JLArray (measured)
dy = Enzyme.make_zero(y)
dy .= 99
Array(y)   # Float32[99.0, 99.0, 99.0]   <- dy aliases y, and was never zeroed
```

`Enzyme`'s `make_zero` has fast methods only for `Base.Array` of
`AbstractFloat`/`Complex` (`Enzyme/src/typeutils/make_zero.jl:47`).
Anything else falls through to the generic mutable-struct recursion,
which rebuilds the wrapper struct while *sharing* the `data::DataRef` field.
The result is a new wrapper object over the original buffer,
still holding the original values.
Measured aliasing on both `MtlArray` and `JLArray`; `Base.Array` is fine.

CUDA.jl is unaffected because it ships `ext/EnzymeCoreExt.jl`,
which defines exactly the missing methods (`CUDA/ext/EnzymeCoreExt.jl:561`):

```julia
@inline EnzymeCore.make_zero(x::DenseCuArray{FT}) where {FT<:AbstractFloat} =
    Base.zero(x)
```

plus the `(::Type, seen::IdDict, prev, ::Val)` form and `make_zero!`.
Metal.jl and JLArrays.jl ship no Enzyme extension at all.

The package calls `make_zero` in five places, all of which are affected:
`src/lrp.jl:217` (the input shadow `dx`),
`src/autodiff.jl:122` (`dy` in `lrp_node`'s augmented primal),
`src/autodiff.jl:180` (`dz` in `lrp_connection`),
`src/autodiff.jl:216` (`dx` in `seeded_pullback`),
`src/autodiff.jl:320` (`dy` in `tap_relevance`).

**Fix (decided): switch the five call sites to `Base.zero`.**
All five arguments are plain numeric arrays, never nested structures,
so `Base.zero` is semantically identical and backend-agnostic —
one change fixes Metal, JLArrays and every future backend at once.
A `RelevancePropagationMetalExt` defining the three `EnzymeCore` methods
was also verified to work, but is per-backend and does nothing for JLArray;
reserve that shape for upstream.
File the Metal.jl issue either way, with the CUDA ext as the template.

Caveat to verify when landing this: the JLArray/Metal probes patched
`EnzymeCore.make_zero` globally, which also covers any *internal*
Enzyme call on the differentiated path.
The split-mode run's return shadow is allocated by this package's own
`augmented_primal` (call site 2 above), so switching call sites should
suffice — but re-run the JLArray end-to-end probe *without* the global
patch as the acceptance check for this task.

### 2. Enzyme cannot differentiate GPU-array operations — on either backend

Metal failure modes (all one root cause):

| expression | result |
|---|---|
| `autodiff(Reverse, x -> sum(abs2, x), Active, Duplicated(::MtlArray, …))` | **segfault** in LLVM's verifier inside `EnzymeCreateAugmentedPrimal` |
| `autodiff(Reverse, dot, Active, Duplicated(::MtlArray, …), Const(::MtlArray))` | `EnzymeNonScalarReturnException: … found nothing of type Nothing` |
| `autodiff_thunk(…)` over `x -> 2 .* x` | `LLVM error: … Number of arg operands != function parameters` on a call to `objc_msgSend` |

The third message names the Metal-specific cause: Metal.jl's array
operations bottom out in Objective-C message sends to the Metal API,
and Enzyme — differentiating at the LLVM IR level —
cannot rewrite `objc_msgSend`.
`set_runtime_activity` does not help.

JLArray reproduces the *structure* of the blocker
with different error messages (measured 2026-08-18):

| expression | result |
|---|---|
| `autodiff(Reverse, dot, Active, Duplicated(::JLArray, …), Const(::JLArray))` | works, correct gradient |
| `autodiff(Reverse, x -> sum(abs2, x), …)` | `LLVM error: function failed verification` |
| seeded `x -> 2 .* x` | `MethodError: no method matching mkcontext(::KernelAbstractions.Kernel{JLBackend, …})` — Enzyme's KernelAbstractions integration does not know `JLBackend` |
| `seeded_pullback` on `Dense`+`relu` | `EnzymeRuntimeActivityError` |
| the current engine, end to end (`make_zero` fixed, any rule set) | `EnzymeNonScalarReturnException`, same as Metal's `dot` case |

So standalone `dot` differentiates on JLArray,
but embedded in `lrp_loss` (behind `apply` and the inactive mask) it fails;
the exact trigger was not isolated because split mode removes the question.

On the `EnzymeRuntimeActivityError`s: this is the documented
"constant memory is stored (or returned) to a differentiable variable"
condition, with two documented mitigations —
rewrite the code to be activity-stable,
or opt into `set_runtime_activity(Reverse)`.
Neither is worth pursuing for `seeded_pullback` on GPU arrays:
on Metal `set_runtime_activity` was tried and the `objc_msgSend`
failure remains behind it, on JLArray the missing `JLBackend`
KernelAbstractions support sits behind it (untried),
and the fast paths of Change B remove the call from GPU paths entirely.

The conclusion is the same on both backends and is the design constraint:

**The engine's differentiable surface must be empty on GPU arrays.**

Consequences:

- **The top-level scalar loss is fatal.** `lrp_loss` (`src/lrp.jl:161`)
  ends in `dot(mask, y)`, and that `dot` is the one array operation
  Enzyme must differentiate itself. Dead on Metal, dead on JLArray.
- **`seeded_pullback` is fatal.** It differentiates
  `dot(first(apply(layer, x, ps, st)), s)` (`src/autodiff.jl:230`),
  so any layer without an `input_vjp` fast path is unusable on GPU arrays.
  Measured on Metal: `MaxPool`, `MeanPool` and `BatchNorm` all fail this way.
  Two call sites reach it even on fast-path layer types:
  `ZBoxRule`'s `c`-term, which today passes the layer *with* its
  activation (the fast paths guard on `activation === identity` and
  decline — resolved by the decided wrap-time activation split,
  see "Activity patterns for the rule layer" below),
  and sub-models treated as one differentiation unit,
  for which no fast path exists at all.
  CUDA is the one backend where `seeded_pullback` *may* work
  (its 773-line `EnzymeCoreExt` gives Enzyme rules for `cufunction`,
  `cudaconvert`, kernel launches) — but the design should not rely on it:
  treat "layer without a fast path" as CPU-only,
  and let CUDA support arrive as a bonus, not a requirement.

Everything else is already covered by the custom rules,
which Enzyme never differentiates — it only *calls* them.
That is why the engine is so close to working:
its differentiable surface is almost empty by construction.

### 3. `masked_copy` uses scalar indexing

`src/utils.jl:66-74` loops over `CartesianIndices(A)` with scalar
`getindex`/`setindex!`, which `GPUArraysCore` rejects
("Scalar indexing is disallowed") — measured on both Metal and JLArray
(on JLArray via `GeneralizedGammaRule`, the only user).

**Fix (verified on both; bit-identical to the loop on CPU):**

```julia
masked_copy(A::AbstractArray, mask::AbstractArray) =
    ifelse.(mask, A, zero(eltype(A)))
```

The size check should be kept.

### 4. `zbox_input` returns a CPU array

`src/rules.jl:374` builds the `low`/`high` bound arrays with
`fill(convert(T, c), size(in))`, which returns a `Matrix{Float32}`
even when `in` is a GPU array.
The `AbstractArray` method (`src/rules.jl:375`) has the same problem:
`convert.(T, A)` keeps `A` on whatever device it was already on.

Failure mode differs by backend, but both fail loudly (measured):

- Metal: `ArgumentError: Objects are on devices with different types:
  MetalDevice and CPUDevice` from Lux's device check.
- JLArray: `Illegal conversion of a JLArray to a Ptr`
  when the generic matmul meets the host-side bound array.
  (Plain host-`Matrix` ⊙ `JLArray` *broadcasts* silently fine —
  it is the matmul in `apply` that objects.
  So JLArrays does catch this bug, just later in the pipeline.)

**Fix (generic):** allocate through the input —

```julia
zbox_input(in::AbstractArray{T}, c::Real) where {T} =
    fill!(similar(in), convert(T, c))
function zbox_input(in::AbstractArray{T}, A::AbstractArray) where {T}
    @assert size(A) == size(in)
    return copyto!(similar(in, T), A)
end
```

`similar` inherits the device, `copyto!` is the canonical host→device
upload, and both lines stay correct for plain `Array`.

## Performance-only issue: `FlatRule` on `Dense`

`src/rules.jl:613-620` fills one view per batch sample and calls
`sum(view(Rᵏ⁺¹, :, i))`, which returns a host scalar —
one GPU→CPU synchronisation per sample.
It *works* on Metal, and on JLArray it runs under `allowscalar(false)`
with dev `0.0` — so this is confirmed performance-only, on both.

**Fix (verified, matches the loop form on CPU):**

```julia
function propagate(_rule::FlatRule, _layer::Dense, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    n = size(aᵏ, 1)
    return similar(aᵏ) .= sum(Rᵏ⁺¹; dims = 1) ./ n
end
```

This is worth doing regardless of GPU support.

## The verified path

Two changes turn the failures above into correct GPU results.
Both were measured; the numbers are from the runs described here.

### Change A: replace the `dot` seed with split-mode seeding

Instead of differentiating the scalar loss `dot(mask, model(x))`,
run Enzyme's augmented forward, write the relevance seed directly into
the returned output shadow, and run the reverse:

```julia
function lrp_split(model, x, ps, st, seed)
    dx = zero(x)
    fwd, rev = Enzyme.autodiff_thunk(
        ReverseSplitWithPrimal, Const{typeof(model_output)}, Duplicated,
        Const{typeof(model)}, Duplicated{typeof(x)},
        Const{typeof(ps)}, Const{typeof(st)},
    )
    tape, primal, shadow = fwd(
        Const(model_output), Const(model), Duplicated(x, dx), Const(ps), Const(st)
    )
    shadow .= seed
    rev(Const(model_output), Const(model), Duplicated(x, dx), Const(ps), Const(st), tape)
    return dx, primal
end
```

(The seed can be built from `primal` between `fwd` and `rev`,
since the primal output is available before the reverse runs —
that is how the probes computed `relevance_seed(primal, ns(primal), …)`.)

The annotations follow Enzyme's activity system
(FAQ ["Implementing pullbacks"](https://enzymead.github.io/Enzyme.jl/stable/faq/#Implementing-pullbacks),
[API reference](https://enzymead.github.io/Enzyme.jl/stable/api/)
for `autodiff_thunk`), and split mode is the *documented* shape
for array-valued pullbacks, not a workaround:

- **`Active` return is not an option for arrays.**
  `Active` is reserved for immutable values;
  combined-mode reverse `autodiff` therefore only differentiates
  functions returning a `Real` or `nothing`.
  The current `dot(mask, y)` loss exists purely to manufacture
  such a scalar — and that manufactured scalar is exactly the
  operation Enzyme cannot differentiate on GPU arrays.
  For thunks the API reference enumerates the return activity as
  "`Const` or `Duplicated` (or its variants `DuplicatedNoNeed`,
  `BatchDuplicated`, and `BatchDuplicatedNoNeed`)" —
  an array-returning `model_output` is `Duplicated`,
  and the seed enters as its shadow, with no loss function at all.
- **`ReverseSplitWithPrimal` + `Duplicated`** because both outputs
  are needed: `primal` becomes `Explanation.output`
  (and feeds `relevance_seed`), `shadow` receives the seed.
  The `NoPrimal`/`NoNeed` variants exist to skip the primal
  computation and do not apply here.
- **Shadows are accumulators.** Enzyme *adds* into shadow memory
  (per the FAQ, results are "added to" the shadow),
  so `dx` must be freshly zero-initialized on every call —
  which is also why blocker 1's aliasing `make_zero` is so poisonous:
  it violates exactly this zero-on-entry contract, silently.
- **The reverse thunk returns derivatives only for `Active`
  arguments**, of which there are none here: `model`, `ps`, `st`
  are `Const`, and the input relevance accumulates in place into
  `Duplicated`'s `dx`, so `rev`'s return value is ignored.
- **Why not the FAQ's mutating-accumulator pattern?**
  The documented combined-mode alternative for array outputs
  (a mutating `f!(y, x…)` returning `nothing`,
  with the seed passed up front as `Duplicated(y, seed)`)
  would also eliminate the scalar loss,
  but it moves a `copyto!`-style store *into* the differentiated
  region — on Metal that is again a device operation Enzyme would
  have to rewrite. The split thunk writes the seed into the shadow
  *between* the forward and reverse passes,
  outside anything Enzyme differentiates. (Reasoned, not measured.)

This removes the last array operation Enzyme had to differentiate,
and it is a strict improvement independent of GPU support:

- The model output comes back as `primal`, so the `Ref` capture
  (`output_ref`/`detached_mask!`, `src/lrp.jl:170-194`) can go —
  along with the `Base.promote_op` type over-approximation it needs.
- The seed is built outside the differentiated region,
  so `relevance_seed` no longer has to be `EnzymeRules.inactive`.
- On CPU it reproduces `analyze`'s current output **exactly**
  (`==`, not `isapprox`), on both the `Dense` chain and the CNN.

Note the objection recorded in `NOTES.md` ("re-running a consumed tape
aborts Julia for some layers") does not apply: that was about *reusing*
one tape across several seeds. Here the tape is used once per `analyze`,
exactly as in combined mode.

A custom `EnzymeRules` rule on a `seed_loss(mask, y)` function was tried
first and does **not** work: with a primal that ignores `y`,
Enzyme's activity analysis marks the whole chain `Const` and then looks
for `augmented_primal(…, ::Type{Const{Matrix{Float32}}}, …)` on `lrp_node`,
which no rule provides. Split mode avoids the question entirely.

### Change B: `input_vjp` fast paths for pooling layers

Needed so that no GPU code path reaches `seeded_pullback`.
Lux nests the configuration, so the fields are
`layer.layer.mode.{kernel_size, stride, pad, dilation}`:

```julia
function pool_dims(layer, x)
    m = layer.layer.mode
    return PoolDims(x, m.kernel_size; padding=m.pad, stride=m.stride, dilation=m.dilation)
end
function input_vjp(layer::MaxPool, x, ps, st, s)
    pdims = pool_dims(layer, x)
    return ∇maxpool(s, maxpool(x, pdims), x, pdims)
end
function input_vjp(layer::MeanPool, x, ps, st, s)
    pdims = pool_dims(layer, x)
    return ∇meanpool(s, meanpool(x, pdims), x, pdims)
end
```

Cross-checked against `seeded_pullback` on CPU: **exactly equal**
(max deviation `0.0`) for both layer types.
These belong in the existing `test_autodiff.jl` fast-path testset.
(The landed version, task 4, computes the `PoolDims` by calling the
nested pool mode instead of reading its fields, which extends the fast
paths to the adaptive and global pooling variants for free.)
On JLArray these paths cannot run at all
(NNlib pooling scalar-indexes there), which is fine:
the cross-check lives on CPU, the device test on Metal/CUDA.

### Measured results

With Change A, Change B, the `make_zero` fix from blocker 1,
and no other modifications:

| model / backend | vs CPU `analyze` |
|---|---|
| `Dense` chain, `ZeroRule`, Metal | max abs dev `6.0e-8` |
| `Conv`/`MaxPool`/`FlattenLayer`/`Dense` CNN, `ZeroRule`, Metal | max abs dev `2.4e-7`, relative `3.0e-7` |
| `Dense` chain, `ZeroRule`, JLArray | max abs dev `0.0` (exact) |
| CPU, split mode vs current engine | bit-identical (`==`) |

The Metal deviations are Float32 rounding noise.

## Activity patterns for the rule layer

The rules in `docs/src/rules.md` decompose into a handful of AD shapes,
and the activity system serves each differently.
The claims marked *measured* are from `probe5.jl` (CPU, `Dense`,
Enzyme v0.13.199): all comparisons are `==`, not `isapprox`.

### The primitive: a one-shot VJP that also returns the primal

On the fallback path (layers without an `input_vjp` fast path),
the generic `propagate` (`src/rules.jl:39-49`) runs the modified forward
**twice**: once in `apply` for the denominator z̃ (`src/rules.jl:45`),
and again inside `seeded_pullback`'s combined-mode `autodiff`.
A `ReverseSplitWithPrimal` thunk collapses this:
its augmented forward returns `(tape, primal, shadow)`,
the primal *is* z̃, the seed `s = Rᵏ⁺¹ ./ stab(z̃)` is written into the
shadow, and one reverse consumes the tape — one forward instead of two,
and no `Active` scalar anywhere (the same argument as Change A,
one level down).
*Measured:* primal `==` the separate `apply`,
pullback `==` `seeded_pullback`, for both identity and `relu` `Dense`.

Proposed shape — replace the raw `seeded_pullback` fallback with a
two-phase primitive that both fast paths and the thunk implement:

```julia
prepare_vjp(layer, x, ps, st) -> (z̃, pullback!)   # pullback!(s) -> c
```

Fast-path layers return `(apply(layer, x, ps, st), s -> hand_vjp(s))`;
generic layers run the augmented forward and close over the tape.
Each `pullback!` is **single-use** — one `rev` per tape,
honoring the `NOTES.md` finding that re-running a consumed tape
aborts Julia. Rules needing several VJPs at *the same point*
use batched shadows instead (below).

Backend reach, measured: on CPU this is exact and halves fallback
forwards; on JLArray the nested thunk still fails
(`EnzymeRuntimeActivityError`, the `JLBackend` hole behind it),
and Metal has `objc_msgSend` behind the same door —
so this primitive improves CPU now and plausibly unlocks CUDA's
fallback path, while fast paths remain the only mechanism
on Metal/JLArray. Nothing about it regresses any backend.

### Rule taxonomy

Costs are for the fallback path, in layer forwards (F) and reverses (R);
fast-path layers are unaffected (no Enzyme, already minimal).

| pattern | rules | today | with thunks |
|---|---|---|---|
| no AD at all | `PassRule`, `LayerNormRule`, `FlatRule`/`Dense`, reshape fast paths | — | — |
| cached z̃, one VJP | `ZeroRule`, `EpsilonRule` (z̃ = node-tape `zᵏ`) | 1F+1R | 1F+1R (0F+1R with a node-held tape, below) |
| one modified forward, one VJP, same point | `GammaRule`, `WSquareRule`, `FlatRule` (generic layers) | 2F+1R | 1F+1R |
| two points, one shared seed | `ZPlusRule` | 4F+2R | 2F+2R |
| two points × two seeds | `AlphaBetaRule`, `GeneralizedGammaRule` | 8F+4R | 4F+2R (width-2 batch) |
| three points, one seed | `ZBoxRule` | 5F+3R | 3F+3R |

Notes per pattern:

- **`ZeroRule`/`EpsilonRule`**: their VJP is through the unmodified
  affine layer at the unmodified input — the very forward
  `node_forward` already ran in the node's `augmented_primal`.
  If the node built that forward *as* an augmented-forward thunk and
  stored `(tape, pullback!)` alongside `zᵏ`, these rules would need no
  forward at all in the reverse pass (0F+1R).
  The node knows its rule (`LayerWithRule`), so this can be gated on a
  trait (identity `modify_params`/`modify_input`, no fast path).
  Costs tape memory per node — an optimization, not a requirement,
  and unmeasured.
- **`ZPlusRule`**: the seed needs *both* primals (`s` divides by
  `z⁺ + z⁻`), so run both augmented forwards first, then both reverses.
  Split mode expresses this ordering naturally;
  combined mode cannot without recomputation.
- **`AlphaBetaRule`/`GeneralizedGammaRule`**: the code already exploits
  that the α/β (resp. ˡ/ʳ) variants share weights, routing the second
  seed through the first variant's `input_vjp`
  (`src/rules.jl:455-457,472-473,514-515,534-535`).
  On the thunk path that is two seeds through **one tape** —
  exactly `BatchDuplicated` width 2: one augmented forward per point,
  one batched reverse with shadows `(sᵅ, sᵝ)`.
  *Measured:* width-2 batch `==` two separate `seeded_pullback`s.
  This is the only safe way to reuse a tape for a second seed —
  a second `rev` call on the same tape is the documented abort.
  The two crossed bias variants (`zᵝ⁺`, `zᵝ⁻`) still need two plain
  forwards; only two of the four z's arrive as tape primals.
- **`ZBoxRule`**: three *different* primal points `(aᵏ, l, h)` —
  batching does not apply (Enzyme batches shadows at one point,
  never primals). Three independent thunks; `z⁺`/`z⁻` arrive as
  primals, `z` comes from the node cache.
- Thunks are cached per type signature, and the batch width is a type
  parameter — each `(layer type, width)` pair compiles once per
  session. Prefer a small fixed set of widths.

### The `ZBoxRule` activation gap (resolved by the wrap-time split)

To be precise about where activations live in the engine *today*
(the wrap-time split below changes this):
model preparation does *not* strip them — `LayerWithRule` wraps the
original layer, because the node's forward must produce the true
output `y = σ.(z)` for downstream layers.
Stripping happens at the use sites instead:
`node_forward` caches the pre-activation through `remove_activation`,
and the rule bodies call `input_vjp` on `f = rule_layer(layer)`,
the stripped layer.
Every `input_vjp` call in the generic `propagate` and in
`ZPlusRule`/`AlphaBetaRule`/`GeneralizedGammaRule` goes through `f`,
so their fast paths fire regardless of the model's activations.

The one exception is `ZBoxRule`'s `c`-term
(`c = input_vjp(layer, aᵏ, ps, st, s)`, `src/rules.jl:368`),
which deliberately passes the *unstripped* layer;
the fast paths guard on `activation === identity`
(`src/autodiff.jl:245,250,255,269`) and decline.
So `ZBoxRule` on its canonical target — a `Conv(..., relu)` or
`Dense(..., relu)` input layer — falls back to `seeded_pullback`
today, and is therefore dead on Metal/JLArray *even after Change B*.
This is v3-faithful, verified against `main`:
v3's ZBox also ran `pullback(layer, aᵏ)` through the original
activation-bearing layer, while its modified `layer⁺`/`layer⁻`
were stripped (`copy_layer` defaults to `σ=identity`).
In v3 that pullback was Zygote's, which had GPU rules,
so the issue is new to the Enzyme port, not to the rule.

Separately — different mechanism, same destination:
sub-models treated as a single differentiation unit
(`wrap_rules` with one rule on a `Chain`/`Parallel`)
and CRP's container fallback (`src/crp.jl:54`)
reach `seeded_pullback` because *no* fast path exists for
containers, activations aside.

**Resolution (decided):** the wrap-time activation split below.
`ZBoxRule` thereby adopts the affine-only semantics of its docstring
formula — an intentional divergence from v3:
its references are regenerated once,
and the change is called out in the 4.0 changelog.
The alternative that would have preserved v3 numerics exactly —
an activation-aware fast path `affine_vjp(s .* σ′.(zᵏ))` —
was considered and rejected: it patches the gap with a σ′ table
instead of dissolving it,
and keeps the use-site stripping machinery alive.

### Decided: activations are split out at wrap time (`PassRule` nodes)

The engine will no longer wrap full layers and strip σ at each use
site. Activations are split into their own nodes when the model is
wrapped: the affine part carries the rule, and the activation follows
as a separate node whose backward is the explicit pass-through LRP
prescribes. The activation stays in the compute graph —
the forward is unchanged — but the reverse ignores it.

Most of the pieces already exist:

- `ModelSurgeon.activation_fn`/`remove_activation`
  (`src/ModelSurgeon/lux_layers.jl:40-55`) provide the surgery for the
  pair-wrapper realization: strip σ from the leaf, apply it in a second
  node. (`ModelSurgeon.split_activation` is the same idea as a
  `ps`-rebuilding `Chain` (`src/ModelSurgeon/canonize.jl:72-77`) —
  the right tool for the construction-time realization, but its
  `LayerNorm` method also moves the affine part out into a `Scale`,
  which the wrap-time split must *not* do: the `LayerNorm` node keeps
  its affine parameters, only σ is stripped, and `LayerNormRule`'s
  internal `Scale` handling then sees `identity` and simplifies.)
- `PassRule` already *is* the "SkipRule": elementwise activations
  preserve shape, so `reshape_relevance` is a no-op pass-through.
  The activation node is a plain `LayerWithRule(PassRule(), …)` —
  no new rule type, no new `EnzymeRules` function,
  and Enzyme never differentiates σ (the custom rule covers it),
  so the node is GPU-clean.

One structural constraint shapes the implementation:
the wrapped model is applied with the *original* `ps`/`st` trees
(the wrappers are `ps`/`st`-transparent), so activation nodes cannot
simply be appended as chain siblings — extra top-level entries would
need matching `ps`/`st` entries. Either the split runs on the triple
at `LRP` construction (the `split_activation` path, which rebuilds
`ps`/`st`, with `PassRule` entries inserted into the rules tree in
lockstep), or the affine part and the activation live inside one
transparent pair-wrapper whose `ps` routes to the affine child.
The pair-wrapper realization is preferred:
it keeps the original `ps`/`st` trees, the rule keys and
CRP's positional layer indices valid, and it leaves
`length(model.layers)` unchanged — so `insert_taps` indices and the
`layerwise_relevances` contract (one entry per model child) survive
untouched — at the cost of CRP's forward loop learning the
two-stage node.
Sub-models treated as one differentiation unit are exempt:
the split applies to rule-carrying leaves only, and activations
inside an opaque sub-model stay where they are
(that path remains `seeded_pullback`-bound and CPU-only, as before).

What it buys:

- Deletes the use-site machinery — `ActivationSplitLayer`,
  the `node_forward` split, `node_output`, `rule_layer`
  (`src/autodiff.jl:66-109`) — and all of their call sites:
  `rule_layer` in the generic `propagate` and four rule bodies
  (`src/rules.jl:40,357,401,452,511`), `node_output` in
  `ZBoxRule`/`GeneralizedGammaRule` (`src/rules.jl:363,528`), and
  `node_forward` in `LayerNormRule` (`src/rules.jl:586`) and in
  CRP's forward loop (`src/crp.jl:75`).
  `LayerNormRule`'s internal `Scale` then always carries `identity`,
  collapsing its `node_forward`/`rule_layer` indirection.
  Stripping happens once at construction instead of on every call.
- Makes "LRP ignores activations" a structural property instead of a
  per-call-site convention, uniformly extended to layers the current
  split skips: an un-canonized `BatchNorm(…, relu)` becomes
  affine-BN + σ node, composing with the `BatchNorm` fast path
  (task 9).
- Fixes an existing inconsistency: a model whose activations were
  *already* split out (`canonize`) currently gets the true σ-VJP at
  each `WrappedFunction` node via `seeded_pullback` — numerically
  different from the fused handling, and GPU-dead. With `PassRule`
  assigned to split-out activations, fused and pre-split models
  agree, and both are GPU-clean.
  The mechanism differs by origin, and only the first half is
  automatic: wrap-created σ nodes carry `PassRule` structurally,
  while a standalone activation layer already present in the model
  gets whatever rule the user or composite assigns.
  The composite presets already map `typeof(identity)` to `PassRule`
  (`src/composite_presets.jl`); extend that mapping to supported
  activation-only layers (`WrappedFunction`s over
  `LRPSupportedActivation`s), and decide whether the no-rules `LRP`
  constructor (today: `ZeroRule` everywhere) does the same —
  recommended, since a default-constructed pre-split model otherwise
  keeps hitting `seeded_pullback` on its σ nodes, GPU-dead and
  inconsistent with the fused form. No current test model contains a
  standalone activation node, so this changes no existing references
  either way.
- Dissolves the `ZBoxRule` gap architecturally
  (previous subsection).

The accepted consequence: after the split the node has no unstripped
layer, so ZBox's σ-inclusive `z`/`c` terms are no longer expressible —
`ZBoxRule` becomes affine-only, as decided above.
(Exempting ZBox-ruled layers from the split was considered and
rejected: it keeps the machinery alive for one rule and forfeits
most of the simplification.)
Everything else is numerics-neutral: the other rule bodies are
already all-affine, and `GeneralizedGammaRule`'s masks are unchanged
because `leakyrelu` preserves sign (`z .> 0 ⟺ σ(z) .> 0`).
Remaining costs: one extra custom-rule node and shadow allocation
per activation, and the port's acceptance criterion becomes
"bit-identical except the documented ZBox change"
instead of "suite bit-identical".

### Forward-pass equivalence (the split's safety net)

The split rests on one numerical assumption: applying the stripped
layer and then broadcasting σ reproduces the fused layer's output
bit-for-bit. The engine already relies on this *today* —
`node_forward` computes `y = σ.(z)` from the stripped forward and
feeds that recomposed `y` to downstream layers — but nothing tests it
directly, and Lux routes activations through LuxLib's fused kernels
(`fused_dense_bias_activation`, `fused_conv_bias_activation`),
exactly the kind of code that could legally reassociate.

The split task therefore adds an explicit equivalence testset:
assert `==` on CPU between `first(apply(model, x, ps, st))` and the
wrapped model's forward, for every model shape in the test suite —
`Dense`/`Scale`/`Conv` with `relu`/`gelu`/`leakyrelu` activations,
un-canonized `BatchNorm`+σ, both `LayerNorm` variants, nested
`Chain`s, `Parallel`/`SkipConnection`, a container-as-unit sub-model,
a tap-inserted model, and CRP's forward loop (`as[end]`).
`Explanation.output` falls under the same assertion.
If a fused LuxLib path ever breaks bit-equality, this testset is the
tripwire; relaxing it to `isapprox` for a specific layer/activation
pair is a deliberate, documented decision, not a default.
The device legs repeat the assertion in `test_gpu.jl`
(exact on JLArray; Metal compares fused GPU kernels against the split
path, so tight `rtol` rather than guaranteed exactness).

Test adaptation, per the no-test-deletion policy:
the per-layer `propagate` testsets currently pass activation-bearing
layers (`Dense(2 => 2, relu)`, `Scale(2, relu)`, `Conv_relu`, the
`leakyrelu` `GeneralizedGammaRule` case). Under the new contract
`propagate` only ever sees affine layers, so the test harness strips
the activation before calling and *keeps* the reference values —
propagation already went through the stripped layer, so they are
unchanged except `ZBoxRule`'s. `GeneralizedGammaRule`'s masks now
read the affine output instead of `σ(z)`; `leakyrelu` preserves sign,
so its references also hold. Direct tests of removed helpers
(`rule_layer` at `test/test_rules.jl:103`, any
`node_forward`/`node_output` assertions) are adapted to their
structural replacements, not deleted.

### CRP's feature loop is the same-point-many-seeds case

Below the concept layer, CRP re-runs the full `propagate` per feature
(`src/crp.jl:96-113`) although only the seed differs between features —
every modified forward and denominator is recomputed `n_features` times.
Two remedies, both post-4.0 performance work:
for fast-path layers, hoist the forwards/denominators out of the
feature loop and re-run only the cheap hand VJPs;
for thunk-path layers, one tape per node with `BatchDuplicated`
width `n_features` (or fixed chunks to bound compilation).
The future SmoothDiff.jl port has the same shape
(expected VJPs = many seeds at one point) and would reuse the
same machinery.

None of this touches the three `EnzymeRules` custom rules:
everything in this section lives *inside* `propagate`,
which Enzyme treats as opaque. Only end-to-end batching
(several output selections in one reverse pass over the model)
would require `BatchDuplicated` methods on
`lrp_node`/`lrp_connection`/`tap_relevance`.

## What a JLArrays CI job buys

With the tasks below landed, a `test_gpu.jl` running on JLArrays in
ordinary GitHub CI covers, with no GPU hardware:

- the full engine end to end (`analyze` with each rule)
  on `Dense`-family models, with (near-)exact CPU agreement —
  this is what catches any future `make_zero`-style silent aliasing;
- every `Dense`-compatible rule body (`propagate`) against CPU;
- the `Dense`/`Scale` `input_vjp` fast paths;
- indexed seed writes and CRP's concept masking;
- scalar-indexing regressions (this is how `masked_copy` and
  `GeneralizedGammaRule` fail today) —
  `allowscalar(false)` is the default in non-interactive sessions;
- host-array leaks like `zbox_input`
  (fails loudly in the generic matmul, measured).

What it structurally cannot cover:

- **conv, pooling, and anything NNlib**: no JLArray kernels exist,
  the generic fallbacks scalar-index. CNN coverage needs Metal or CUDA.
- **real Enzyme↔backend integration** (CUDA's `EnzymeCoreExt`,
  a future Metal equivalent): JLArray's failure modes mimic the
  *shape* of Metal's but not its mechanisms.
- **performance**: JLArray is a semantics simulator, not a speed one.

### Why this should transfer to CUDA

Everything in the task list is broadcast, `similar`, `copyto!`,
NNlib entry points with cuDNN implementations, or generic Enzyme API —
no Metal- or JLArray-specific code remains in the package.
On the CUDA side, the three risk points are each already covered upstream:
`make_zero` (CUDA's `EnzymeCoreExt` ships the correct methods, and after
task 2 we no longer call it anyway), conv/pooling (cuDNN via NNlibCUDA),
and Enzyme plumbing over device arrays (the same `EnzymeCoreExt`,
the most mature Enzyme-GPU integration there is).
Remaining risk is untested-in-anger thunk plumbing over `CuArray`
inside our custom rules — the follow-up hardware test exists to catch
exactly that, not to drive design changes.

## Corrections to the first-pass GPU notes

Before this investigation, two spots in the package were suspected of
being GPU-unfriendly, a scalar-indexing audit of the `propagate` methods
was outstanding, and one testing recommendation stood.
Both suspicions turned out to be wrong, and the recommendation changed:

- **`relevance_seed` is not a blocker.**
  `seed[idx] .= 1` with `idx::Vector{CartesianIndex{2}}` works on
  Metal and JLArray; GPUArrays handles it without scalar indexing.
  The same applies to CRP's concept masking
  (`R_masked[idx] .= R_original[idx]`, `src/crp.jl:100`) —
  verified working with CPU-derived indices.
- **The `FlatRule`/`Dense` fast path is a performance issue, not a
  correctness one** — it runs on both backends (JLArray dev `0.0`).
- **JLArrays is a real CI target, not just a smoke test** —
  *after* task 2. The earlier caveat
  ("`make_zero` aliases `JLArray`, masking everything else")
  is resolved by switching the package to `Base.zero`;
  with that plus Change A, the whole `Dense`-family engine
  runs on JLArray exactly. The remaining limitation is NNlib
  (no conv/pooling), covered under "What a JLArrays CI job buys".

The unlisted spots this investigation found are `masked_copy`
and `zbox_input`, both above.

## Not yet covered

Ordered by how likely they are to matter:

- ~~`BatchNorm` has no `input_vjp` fast path~~
  **resolved (2026-08-19, task 9)**: testmode-over-running-statistics
  fast path landed; the JLArray end-to-end leg lands with task 10
  (LuxLib's batchnorm on JLArray still unmeasured).
- **CRP is blocked upstream in XAIBase**, not here:
  `TopNFeatures(2)` scalar-indexes in its `top_n` sort —
  measured failing on both `MtlArray` and `JLArray`,
  so the fix is generically testable in CI without GPU hardware.
  The rest of CRP's positional loop is GPU-clean —
  it drives `propagate` directly and never touches the Enzyme
  end-to-end pass, so once feature selection is fixed
  (or run on a host copy of the relevance) CRP should work.
- **End-to-end batched seeds via `BatchDuplicated`
  (future, out of 4.0 scope).**
  Several output selections per input currently mean one full reverse
  pass per seed, and `NOTES.md` records that *reusing* a consumed
  split-mode tape for several seeds aborts Julia.
  Enzyme's documented mechanism is width-N shadows —
  measured working and exact at the rule-internal level
  ("Activity patterns for the rule layer" above),
  where it needs no dispatch changes because it stays inside
  `propagate`. Batching the *end-to-end* pass is the part that is
  out of scope: the custom rules dispatch on
  `::Type{<:Union{Duplicated,DuplicatedNoNeed}}` and `x::Duplicated`
  (`src/autodiff.jl:111-148`, likewise `lrp_connection` and
  `tap_relevance`), so it would need `BatchDuplicated` methods with
  tuple-of-shadows handling throughout —
  the principled replacement for tape reuse
  if multi-seed performance ever matters.
- ~~Layerwise relevance taps under split mode on a GPU array~~
  **resolved (2026-08-18, tasks 2+3 acceptance)**: exact on JLArray
  (`layerwise_relevances` dev `0.0` for every entry);
  keep the assertion in the JLArrays testset (task 10).
- **Rules beyond `ZeroRule` end-to-end** — *narrowed* by the tasks 2+3
  acceptance run: `Epsilon`/`Gamma`/`WSquare`/`Flat` now also pass a
  complete reverse pass exactly on JLArray. Still open end to end:
  `ZPlus`/`AlphaBeta`/`ZBox`/`GeneralizedGamma`/`LayerNorm` rules and
  composites — the JLArrays testset (task 10) closes these for
  `Dense`-family models; Metal end-to-end per rule stays manual.
- **`ConvTranspose`, `Scale`, `LayerNorm` end-to-end**: rule bodies and
  fast paths passed in isolation on Metal, no full-model run.
  `Scale`/`LayerNorm` are `Dense`-family and JLArray-coverable;
  `ConvTranspose` needs hardware.
- **Performance.** Nothing was benchmarked. LuxLib already warns
  `Falling back to slow convolution routine for MetalDevice`
  on the CNN above, so Metal throughput in the Lux stack is immature
  independently of anything here.
- **Float64.** Apple GPUs are Float32-only; CUDA and JLArray are not.
  Treat eltype as a backend property, not a package restriction —
  docs should note the Metal limitation, nothing more.

## Task list

1. ~~Split activations out of rule-carrying nodes at wrap time~~
   **DONE (2026-08-18)** — see the status entry above. Realized as the
   transparent pair-wrapper `SplitActivationNode`; use-site machinery
   deleted; `ZBoxRule` affine-only (references turned out to be
   unchanged, divergence documented in the changelog and docstring);
   forward-equivalence testset in `test/test_forward.jl`;
   un-canonized `BatchNorm(…, relu)` end-to-end coverage in
   `test/test_lrp.jl`.
2. ~~Switch the five `make_zero` call sites to `Base.zero`~~
   **DONE (2026-08-18)** — four sites in `src/autodiff.jl` plus the
   fifth, which task 3's rewrite absorbed; `make_zero` is no longer
   imported. The Metal.jl issue remains to file (task 12).
3. ~~Replace `lrp_loss`/`detached_mask!`/`output_ref` in
   `call_analyzer` with the split-mode `lrp_split` above~~
   **DONE (2026-08-18)** — see the status entry above. CPU suite
   bit-identical with zero reference regeneration; JLArray and Metal
   end-to-end exact via plain `analyze`, no piracy in the setup.
4. ~~Add `input_vjp` fast paths for `MaxPool` and `MeanPool`~~
   **DONE (2026-08-19)** — see the status entry above. Landed on the
   `MaxPoolLayer`/`MeanPoolLayer` unions (all six pooling types),
   cross-checked on CPU; CNN-on-Metal verified end to end.
5. ~~Rewrite the generic VJP fallback from `seeded_pullback`'s
   combined-mode `dot` loss to the two-phase split-mode `prepare_vjp`
   primitive and re-express the rule bodies on it~~
   **Stage 1 DONE (2026-08-19)** — see the status entry above;
   removed the double forward on the fallback path and the last
   `Active` scalar below the engine.
   Stage 2, optional and separable, remains open:
   width-2 `BatchDuplicated` tapes for
   `AlphaBetaRule`/`GeneralizedGammaRule`
   (their second seed currently takes one-shot `input_vjp`s).
   Acceptance unchanged: the CPU test suite is bit-identical
   (the width-2 pattern measured exact on `Dense`, `probe5.jl`).
6. ~~Broadcast `masked_copy` (`src/utils.jl:66`), keeping the size check~~
   **DONE (2026-08-19)** — bit-identical on CPU.
7. ~~Make `zbox_input` allocate through the input
   (`src/rules.jl:374-378`, `fill!(similar(in), …)` / `copyto!`)~~
   **DONE (2026-08-19)**.
8. ~~Broadcast the `FlatRule`/`Dense` fast path (`src/rules.jl:613`)~~
   **DONE (2026-08-19)**.
9. ~~Add a `BatchNorm` `input_vjp` fast path
   (testmode affine map; generic broadcast)~~
   **DONE (2026-08-19)** — see the status entry above; realized as
   `prepare_vjp`/`input_vjp` methods guarded by `batchnorm_is_affine`
   (identity activation, tracked statistics, testmode).
10. Add `test/test_gpu.jl`, parameterized over the device like
   ExplainableAI.jl's (`device = Metal.functional() ? mtl : jl`,
   with `fmap`/`Adapt` for the `ps`/`st` trees).
   The JLArrays leg runs in standard CI on every PR and asserts
   near-exact CPU agreement for: `analyze` end to end per rule on a
   `Dense`-family model (incl. `layerwise_relevances` and a composite),
   the wrapped-forward equivalence assertion from task 1,
   `propagate` per `Dense`-compatible rule, `Dense`/`Scale`/`BatchNorm`
   fast paths, and CRP (`IndexedFeatures` immediately;
   `TopNFeatures` once XAIBase is fixed).
   Conv/pool testsets are gated on a functional hardware backend
   (Metal locally / self-hosted; CUDA via Buildkite in follow-up)
   and compare at `rtol=1e-5` (Float32).
11. Follow-up (separate machine): run the same testset on CUDA,
    which by the argument above should require zero package changes.
12. Upstream reports:
    Metal.jl — `make_zero` aliasing, `EnzymeCoreExt` request (blocker 1);
    Enzyme.jl — `objc_msgSend` on Metal, no `mkcontext` for
    KernelAbstractions' `JLBackend` (blocker 2);
    XAIBase.jl — `TopNFeatures` scalar indexing (CRP item above).
    Reactant.jl — a hook for Julia-level `EnzymeRules` during tracing
    (`PLAN_REACTANT.md`, "Paths to Reactant compatibility");
    optional, and off this package's critical path.
    Blocker 2 is the large one and is not on this package's critical
    path after tasks 3 and 4 — but it does mean any future feature
    that needs Enzyme to differentiate a real array operation
    will be CPU/CUDA-only.

Tasks 1–9 are self-contained, independently testable, and generic
(no backend appears in `src/`).
Task 1 comes first deliberately: it is the largest structural change,
it needs no GPU to test, and every later task then lands on the final
architecture instead of being partially reworked by it — it retires
the activation special cases, including the one rule/layer
combination the fast paths missed.
Tasks 2 and 3 are the ones that make GPU arrays work at all;
task 4 extends that from `Dense`-family models to CNNs on hardware
backends; task 5 removes the last `Active` scalar from
the rule layer (a CPU efficiency win now, the CUDA fallback path
later); tasks 6–9 are correctness/performance fixes that
apply to every backend equally.

## Ranking of the approaches

This settles the ranking of the three candidate routes to GPU support:

- **Generic GPU-array support via raw Enzyme (this document)** is the
  cheapest path to *any* GPU support and is verified working on Metal
  and JLArray. A handful of small generic changes,
  no second implementation of any rule, no semantic differences,
  and a hardware-free CI story via JLArrays.
- **CUDA via raw Enzyme (follow-up)**
  benefits from every task above and additionally has real Enzyme
  integration (CUDA.jl's `EnzymeCoreExt`), so `seeded_pullback` may even
  keep working there. Untested for lack of hardware; expected to be a
  test-only follow-up, not a development effort.
- **Reactant via stop-gradient surrogates (DISCARDED)**
  (`PLAN_REACTANT.md`) stays the most expensive option: a parallel
  implementation of every rule, in a formalism with no documented
  extension point for user-defined rules, carrying known semantic gaps
  for the modified-input rules. Discarded in favor of the generic
  support in this plan; the work here neither depends on it
  nor is made redundant by it.

## Reproducers

Metal v1.10.0 was not the newest release at the time of the check;
re-confirm blockers 1 and 2 against current Metal.jl before filing.

The Metal scripts lived in a session-scoped scratchpad that is likely
already collected (`…/7a11372f-…/scratchpad/metalenv/`);
the two that matter are reproduced inline above.
The JLArray probes (2026-08-18) live in
`…/e6a573d2-…/scratchpad/jlenv/` (same caveat) and should be folded
into `test/test_gpu.jl` rather than restored verbatim:

- `probe1.jl` — no Enzyme: scalar-indexing defaults, `make_zero`
  aliasing on JLArray, host⊙device broadcast, `masked_copy` loop vs
  broadcast, seed writes, NNlib conv/pooling, `Wᵀs`, `dot`.
- `probe2.jl` — Enzyme micro-probes on JLArray: `dot` (works),
  `sum(abs2, ·)`, broadcast, `seeded_pullback` (all fail; messages
  quoted under blocker 2). Patches `EnzymeCore.make_zero` for JLArray
  to emulate task 2.
- `probe3.jl` — the current `dot`-seeded engine end to end on JLArray:
  fails for every rule set (`EnzymeNonScalarReturnException`);
  also `TopNFeatures` scalar indexing.
- `probe4.jl` — `propagate` per rule on JLArray vs CPU (dev `0.0`,
  plus the `ZBoxRule`/`GeneralizedGammaRule` failures),
  and split-mode `lrp_split` on JLArray: dev `0.0` vs CPU `analyze`.
- `probe5.jl` — the rule-layer activity patterns:
  `vjp_with_primal` (split thunk) `==` `apply` + `seeded_pullback`
  on CPU for identity and `relu` `Dense`;
  `BatchDuplicated` width-2 `==` two separate pullbacks (CPU);
  the same nested thunk on JLArray still fails
  (`EnzymeRuntimeActivityError`).
