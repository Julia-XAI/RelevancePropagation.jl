# Plan: Metal (Apple Silicon) GPU support

GPU support is in scope for the upcoming "4.0.0" release of RP.jl.
Since this will be a breaking release anyway, we should make the necessary changes to support GPUs now.

Companion to `PLAN_GPU.md`, which covers the discarded Reactant and "stop-gradient" approaches.;
this document covers the Metal.jl backend and corrects
two claims in `PLAN_GPU.md` that turned out to be wrong.
The hope is that Metal compatibility will also result in JLArrays.jl and CUDA.jl compatibility (the latter can't be tested on this machine).

**Status (investigated 2026-08-17):**
LRP produces correct relevances on Metal after four changes,
three of them in this package and one upstream in Metal.jl.
Verified end-to-end on a `Conv`/`MaxPool`/`FlattenLayer`/`Dense` CNN:
Metal matches the CPU reference to `3.0e-7` relative
(`2.4e-7` max absolute deviation, Float32).
Nothing about the engine's design blocks Metal —
the blockers are a missing upstream extension,
one design decision in `call_analyzer` (the `dot` seed),
and three GPU-unfriendly spots in rule code.

## Measurement setup

Apple M3 Pro (18-core GPU), macOS Darwin 24.6.0, Julia 1.12.6.
Enzyme v0.13.199, EnzymeCore v0.8.21, Metal v1.10.0,
Lux v1.31.4, LuxLib v1.15.9, NNlib v0.9.44,
GPUArrays v11.5.10, JLArrays v0.3.2.
CUDA v5.8.2 was read for reference but not run (no NVIDIA hardware).

Reproduction environment: a scratch project that `Pkg.develop`s this package
and adds `Metal`, `Lux`, `Enzyme`, `EnzymeCore`, `MLDataDevices`, `NNlib`,
`JLArrays`. Probe scripts are listed under "Reproducers" below.
Iterate with a DaemonicCabal session (`--session=RelevancePropagation-metal`)
— Enzyme thunk compilation dominates cold runs
(a single stage cost ~5–6 min cold, seconds warm).
Note that a segfault kills the warm worker;
run crash-prone stages in a throwaway `julia` process
or restart with `juliaclient --project=$PWD --restart`
(the `--project` flag is required — a bare `--restart`
does not target the right worker).

## What already works on Metal

Confirmed by direct measurement, no changes required:

- Every Lux forward pass used by the engine.
- The rule bodies (`propagate`) for `Dense`:
  `ZeroRule`, `EpsilonRule`, `GammaRule`, `ZPlusRule`,
  `AlphaBetaRule`, `WSquareRule`, `FlatRule`, `LayerNormRule`.
- All `input_vjp` fast paths: `Dense` (`Wᵀs`), `Scale`,
  `Conv` (`∇conv_data`), `ConvTranspose` (`conv`).
- `stabilize_denom`, `keep_positive`, `keep_negative`, `ones_like`
  — all broadcast cleanly.
- `NNlib.∇maxpool` / `NNlib.∇meanpool` (needed for the fast paths below).
- **`relevance_seed`**, including `seed[idx] .= 1` and `seed[idx] .= y[idx]`
  with `idx::Vector{CartesianIndex{2}}`.
  GPUArrays routes this through a kernel, not scalar indexing.
- Enzyme's *plumbing*: thunk construction, `Chain` recursion,
  custom-rule dispatch and shadow accumulation all run to completion
  over `MtlArray`s.

## The four blockers

### 1. `Enzyme.make_zero` aliases its argument on every GPU array

The worst of the four: it fails **silently**, returning wrong relevances
with no error.

```julia
y  = MtlArray(Float32[1, 2, 3])
dy = Enzyme.make_zero(y)
dy .= 99
Array(y)   # Float32[99.0, 99.0, 99.0]   <- dy aliases y, and was never zeroed
```

`Enzyme`'s `make_zero` has fast methods only for `Base.Array` of
`AbstractFloat`/`Complex` (`Enzyme/src/typeutils/make_zero.jl:47`).
Anything else falls through to the generic mutable-struct recursion,
which rebuilds the wrapper struct while *sharing* the `data::DataRef` field.
The result is a new `MtlArray` object over the original buffer,
still holding the original values.

**This is not Metal-specific:** `JLArray` aliases identically.
`Base.Array` does not. Plain `Base.zero` is correct on all three.

CUDA.jl is unaffected because it ships `ext/EnzymeCoreExt.jl`,
which defines exactly the missing methods (`CUDA/ext/EnzymeCoreExt.jl:561`):

```julia
@inline EnzymeCore.make_zero(x::DenseCuArray{FT}) where {FT<:AbstractFloat} =
    Base.zero(x)
```

plus the `(::Type, seen::IdDict, prev, ::Val)` form and `make_zero!`.
Metal.jl ships no Enzyme extension at all.

The package calls `make_zero` in five places, all of which are affected:
`src/lrp.jl:217` (the input shadow `dx`),
`src/autodiff.jl:123` (`dy` in `lrp_node`'s augmented primal),
`src/autodiff.jl:180` (`dz` in `lrp_connection`),
`src/autodiff.jl:216` (`dx` in `seeded_pullback`),
`src/autodiff.jl:320` (`dy` in `tap_relevance`).

**Fix:** upstream, a `Metal/ext/EnzymeCoreExt.jl` mirroring CUDA's.
Until that lands, a package extension
(`RelevancePropagationMetalExt`, weakdeps `Metal` + `EnzymeCore`)
can define the same three methods; supplying them was sufficient
to make every result below correct.
Switching the five call sites to `Base.zero` would also work —
all five arguments are plain numeric arrays, never nested structures —
and is backend-agnostic, so it is the better in-package fix
if we do not want a Metal-specific extension.

### 2. Enzyme cannot differentiate *any* Metal array operation

Three different failure modes, all from the same root cause:

| expression | result |
|---|---|
| `autodiff(Reverse, x -> sum(abs2, x), Active, Duplicated(::MtlArray, …))` | **segfault** in LLVM's verifier inside `EnzymeCreateAugmentedPrimal` |
| `autodiff(Reverse, dot, Active, Duplicated(::MtlArray, …), Const(::MtlArray))` | `EnzymeNonScalarReturnException: … found nothing of type Nothing` |
| `autodiff_thunk(…)` over `x -> 2 .* x` | `LLVM error: … Enzyme: Number of arg operands != function parameters` on a call to `objc_msgSend` |

The third message names the cause: Metal.jl's array operations bottom out
in Objective-C message sends to the Metal API (buffer allocation,
command encoding, kernel dispatch), and Enzyme — differentiating at the
LLVM IR level — cannot rewrite `objc_msgSend`.
`set_runtime_activity` does not help (verified for `dot` and
`seeded_apply`; the `sum(abs2, ·)` segfault was not retried with it).
CUDA.jl's 773-line `EnzymeCoreExt` exists precisely to give Enzyme
rules for the equivalent CUDA constructs (`cufunction`, `cudaconvert`,
kernel launch); Metal has no counterpart.

Consequences for the engine:

- **The top-level scalar loss is fatal.** `lrp_loss` (`src/lrp.jl:161`)
  ends in `dot(mask, y)`, and that `dot` is the one array operation
  Enzyme must differentiate itself. It cannot, on Metal.
- **`seeded_pullback` is fatal.** It differentiates
  `dot(first(apply(layer, x, ps, st)), s)` (`src/autodiff.jl:230`),
  so any layer without an `input_vjp` fast path is unusable on Metal.
  Measured: `MaxPool`, `MeanPool` and `BatchNorm` all fail this way
  (`EnzymeRuntimeActivityError`, then the `dot` failure above).

Everything else is already covered by the custom rules,
which Enzyme never differentiates — it only *calls* them.
That is why the engine is so close to working:
its differentiable surface is almost empty by construction.

### 3. `masked_copy` uses scalar indexing

`src/utils.jl:66-74` loops over `CartesianIndices(A)` with scalar
`getindex`/`setindex!`, which `GPUArraysCore` rejects
("Scalar indexing is disallowed").
Only `GeneralizedGammaRule` uses it.

**Fix (verified, bit-identical to the loop on CPU):**

```julia
masked_copy(A::AbstractArray, mask::AbstractArray) =
    ifelse.(mask, A, zero(eltype(A)))
```

The size check should be kept.

### 4. `zbox_input` returns a CPU array

`src/rules.jl:374` builds the `low`/`high` bound arrays with
`fill(convert(T, c), size(in))`, which returns a `Matrix{Float32}`
even when `in` is an `MtlArray`. `ZBoxRule` then fails with
`ArgumentError: Objects are on devices with different types:
MetalDevice and CPUDevice`.
The `AbstractArray` method (`src/rules.jl:375`) has the same problem:
`convert.(T, A)` keeps `A` on whatever device it was already on.

**Fix:** allocate through the input,
`fill!(similar(in), convert(T, c))`, and for the array method
copy onto the input's device rather than converting in place.

## Performance-only issue: `FlatRule` on `Dense`

`src/rules.jl:613-620` fills one view per batch sample and calls
`sum(view(Rᵏ⁺¹, :, i))`, which returns a host scalar —
one GPU→CPU synchronisation per sample.
It *works* on Metal, it is just slow.

**Fix (verified, matches the loop form on CPU):**

```julia
function propagate(_rule::FlatRule, _layer::Dense, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    n = size(aᵏ, 1)
    return similar(aᵏ) .= sum(Rᵏ⁺¹; dims = 1) ./ n
end
```

This is worth doing regardless of GPU support.

## The verified path

Two changes turn the failures above into correct Metal results.
Both were measured; the numbers are from the runs described here.

### Change A: replace the `dot` seed with split-mode seeding

Instead of differentiating the scalar loss `dot(mask, model(x))`,
run Enzyme's augmented forward, write the relevance seed directly into
the returned output shadow, and run the reverse:

```julia
function lrp_split(model, x, ps, st, seed)
    dx = make_zero(x)
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

### Measured result

With Change A, Change B, the `make_zero` methods from blocker 1,
and no other modifications:

| model | Metal vs CPU |
|---|---|
| `Chain(Dense(4=>6, relu), Dense(6=>3))`, `ZeroRule` | max abs dev `6.0e-8` |
| `Chain(Conv((3,3),1=>4,relu;pad=1), MaxPool((2,2)), FlattenLayer(), Dense(64=>3))`, `ZeroRule` | max abs dev `2.4e-7`, relative `3.0e-7` |

Both are Float32 rounding noise.
The CPU split-mode result is *bit-identical* to today's `analyze`.

## Corrections to `PLAN_GPU.md`

Two items in the "Known GPU-unfriendly spots" section are wrong,
and one testing recommendation needs a caveat:

- **`relevance_seed` is not a blocker.**
  `seed[idx] .= 1` with `idx::Vector{CartesianIndex{2}}` works on Metal;
  GPUArrays handles it without scalar indexing.
  The same applies to CRP's concept masking
  (`R_masked[idx] .= R_original[idx]`, `src/crp.jl:100`) —
  verified working with CPU-derived indices.
- **The `FlatRule`/`Dense` fast path is a performance issue, not a
  correctness one** — it runs on Metal, one host sync per batch sample.
- **JLArrays is not a safe stand-in for GPU CI.**
  `Enzyme.make_zero` aliases `JLArray` exactly as it does `MtlArray`,
  so a JLArrays job would reproduce blocker 1 —
  which is useful — but it would do so for *every* Enzyme-based test,
  masking anything else. Either add the `make_zero` methods for
  `JLArray` in the test setup, or treat a JLArrays job as a
  scalar-indexing/device-mismatch check only.

The unlisted spots this investigation found are `masked_copy`
and `zbox_input`, both above.

## Not yet covered

Ordered by how likely they are to matter:

- **`BatchNorm` has no `input_vjp` fast path**, so it takes
  `seeded_pullback` and is dead on Metal. `canonize` fuses BatchNorm into
  the preceding linear layer, which sidesteps it for canonized models,
  but an un-canonized model with BatchNorm will fail.
  A fast path is straightforward (testmode BatchNorm is an affine map).
- **CRP is blocked upstream in XAIBase**, not here:
  `TopNFeatures(2)(::MtlArray)` fails with scalar indexing
  (its `top_n` sort). The rest of CRP's positional loop is GPU-clean —
  it drives `propagate` directly and never touches the Enzyme
  end-to-end pass, so once feature selection is fixed
  (or run on a host copy of the relevance) CRP should work.
- **Layerwise relevance taps** (`TappedLayer`, `src/autodiff.jl:298-338`)
  were not tested on Metal. `store.val[index.val] = copy(dy)` mutates a
  Julia `Vector{Any}` from inside a reverse rule; it should be fine
  (the copy is a device-to-device `copy`) but needs checking.
- **`ConvTranspose`, `Scale`, `LayerNorm` end-to-end**: the rule bodies
  and fast paths were exercised in isolation on Metal and passed,
  but no full-model test was run.
- **Rules beyond `ZeroRule` end-to-end on Metal.** All rule bodies were
  exercised on Metal for a single `Dense` layer, but only `ZeroRule` ran
  through a complete reverse pass.
- **Performance.** Nothing was benchmarked. LuxLib already warns
  `Falling back to slow convolution routine for MetalDevice`
  on the CNN above, so Metal throughput in the Lux stack is immature
  independently of anything here.
- **Float64.** Apple GPUs are Float32-only. Model checks or docs should
  say so; the package has no eltype restriction today.

## Task list

1. Switch the five `make_zero` call sites to `Base.zero`
   (`src/lrp.jl:217`, `src/autodiff.jl:123,180,216,320`),
   **or** add a `RelevancePropagationMetalExt` with the three
   `EnzymeCore.make_zero`/`make_zero!` methods.
   Prefer the former: it is backend-agnostic and fixes JLArrays too.
   File the Metal.jl issue either way, with the CUDA ext as the template.
2. Replace `lrp_loss`/`detached_mask!`/`output_ref` in `call_analyzer`
   with the split-mode `lrp_split` above (`src/lrp.jl:158-241`).
   Acceptance: the full test suite passes unchanged —
   the CPU result is bit-identical, so no references need regenerating.
3. Add `input_vjp` fast paths for `MaxPool` and `MeanPool`
   (`src/autodiff.jl`, next to the `Conv` methods),
   cross-checked against `seeded_pullback` in `test_autodiff.jl`.
4. Broadcast `masked_copy` (`src/utils.jl:66`).
5. Fix `zbox_input` to allocate on the input's device (`src/rules.jl:374-378`).
6. Broadcast the `FlatRule`/`Dense` fast path (`src/rules.jl:613`).
7. Add a `BatchNorm` `input_vjp` fast path.
8. Add a GPU testset. Given the JLArrays caveat above, the honest
   options are a Metal job on self-hosted Apple hardware, or a Buildkite
   CUDA pipeline. Cover, per backend: `input_vjp` per supported layer
   type, one end-to-end `analyze` per composite preset, and a
   CPU-vs-GPU agreement test at `rtol=1e-5` (Float32).
9. Report blocker 1 to Metal.jl and blocker 2 to Enzyme.jl/Metal.jl.
   Blocker 2 is the large one and is not on this package's critical path
   after task 2 and 3 — but it does mean *any* future feature that
   needs Enzyme to differentiate a real array operation
   will be CPU/CUDA-only.

Tasks 1–6 are self-contained and independently testable.
Tasks 2 and 3 are the ones that make Metal work;
tasks 1, 4, 5, 6 are correctness/performance fixes that
apply to CUDA equally.

## Relationship to `PLAN_GPU.md`

This changes the ranking of the options in `PLAN_GPU.md`:

- **Metal via raw Enzyme (this document)** is the cheapest path to
  *any* GPU support and is verified working. Six small changes,
  no second implementation of any rule, no semantic differences.
- **CUDA via raw Enzyme (future work)** (`PLAN_GPU.md` approach 2) benefits from
  every task above and additionally has real Enzyme integration
  (CUDA.jl's `EnzymeCoreExt`), so `seeded_pullback` should keep working
  there. It remains untested for lack of hardware.
  The hope behind this plan is that generic support for Metal.jl also results in CUDA.jl compatibility (this will be tested in follow-up work). 
- **Reactant via stop-gradient surrogates (DISCARDED)** (`PLAN_GPU.md` approach 1,
  item 2) stays the most expensive option: a parallel implementation of
  every rule, in a formalism with no documented extension point for
  user-defined rules, carrying known semantic gaps for the
  modified-input rules. The work here does not depend on it and is not
  made redundant by it.
  This approach has been discarded in favor of more generic GPU support (starting with Metal.jl and JLArrays.jl in this plan)

## Reproducers

Metal v1.10.0 was not the newest release at the time of the check;
re-confirm blockers 1 and 2 against current Metal.jl before filing.

The scripts used live in the (session-scoped, so possibly already
collected) scratchpad directory
`…/7a11372f-5358-421d-82db-72332841e696/scratchpad/metalenv/`.
The two that matter are reproduced inline above;
if this work proceeds they should be rewritten into
`test/test_gpu.jl` rather than restored verbatim:

- `makezero.jl` — blocker 1, three lines, no dependency on this package.
  Compares `Array` / `MtlArray` / `JLArray`.
- `minimal.jl` — blocker 2, the three Enzyme failure modes.
- `probe.jl` / `probe3.jl` — per-function survey of rule bodies,
  `input_vjp` fast paths and the proposed broadcast fixes.
- `warm.jl` + `fix.jl` + `cnn.jl` + `pool.jl` — the split-mode
  seeding path and the end-to-end CPU/Metal comparisons quoted above.
