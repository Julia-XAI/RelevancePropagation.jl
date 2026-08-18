# Reactant.jl: negative result and a verified surrogate design (DISCARDED)

Record of the 2026-08-17 investigation into running this package's LRP engine
through [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl),
the Lux ecosystem's recommended path for Enzyme on GPU.
Superseded as a route to GPU support by generic GPU-array support
via raw Enzyme (`PLAN_GPU.md`),
which is verified working on Metal and JLArray.

**Outcome: discarded**, for two reasons.

- Reactant does not honor Julia-level `EnzymeRules`,
  so the engine's per-layer custom rules are bypassed **silently** —
  the reverse pass compiles and runs,
  and returns plain input gradients instead of relevances.
  The mechanism is upstream and structural, not a knob (below).
- The one workaround that does work —
  rewriting each rule as a stop-gradient surrogate forward pass,
  verified bit-exact under `@jit` —
  is a parallel implementation of every rule,
  in a formalism with no documented extension point for user-defined rules,
  and it carries known semantic gaps for the modified-input rules
  when they are placed mid-network.

Kept rather than deleted because the failure mode is silent,
the reason Reactant cannot honor `EnzymeRules` is not obvious,
and the surrogate derivation would be expensive to reconstruct
should Reactant ever grow a custom-rule hook.
Nothing here is on the 4.0.0 critical path.

## Reactant compatibility check (2026-08-17)

Checked with Reactant v0.2.279, Enzyme v0.13.199, Lux v1.31.4,
Julia 1.12.6, CPU backend on macOS arm64
(all resolve into one environment without conflicts).

Reactant does not run Enzyme.jl's Julia-level differentiation.
Its overlay intercepts `Enzyme.autodiff` during tracing
(`Reactant/src/Overlay.jl` → `overload_autodiff` in `Reactant/src/Enzyme.jl`),
traces only the *primal* into an MLIR function,
and differentiates that at the MLIR level via Enzyme-MLIR.
Julia-level `EnzymeRules` methods therefore never fire —
neither our custom rules on `lrp_node`, `lrp_connection`
and `tap_relevance`,
nor the `EnzymeRules.inactive` mark on `detached_mask!`.
The only Julia-level Enzyme construct Reactant special-cases
is `EnzymeCore.ignore_derivatives`
(lowered to an `enzyme.ignore_derivatives` MLIR op);
there is no trace-level custom-VJP hook to target instead.

Empirical results (small `Dense`/`relu` chain, `EpsilonRule`):

- The wrapped-model reverse pass with a fixed seed **compiles and runs,
  but returns the plain input gradient instead of LRP relevances** —
  the custom rules are bypassed silently. Worst failure mode:
  no error, wrong result.
- The full `analyze` fails at trace time in the seed construction:
  `relevance_seed`'s `seed[idx] .= 1`
  with `idx::Vector{CartesianIndex{2}}`
  has no `TracedRArray` method (`__to_cartesian_index` MethodError),
  and with a max-activation selector the traced integer index array
  cannot be materialized to `CartesianIndex` without scalar indexing.
- `seeded_pullback` standalone compiles and matches the eager result,
  so single-level `autodiff` of `dot(layer(x), s)` traces fine;
  Reactant's docs also claim nested-AD support.

Paths to Reactant compatibility:

1. Upstream support for Julia `EnzymeRules` during tracing.
   Not a hidden knob: Reactant's tracing interpreter *is* Enzyme's
   `EnzymeInterpreter`, constructed with
   `forward_rules=false, reverse_rules=false, inactive_rules=false`
   (`Reactant/src/Interpreter.jl`).
   The flags were `true` originally and were disabled in Dec 2024
   (to work around Enzyme.jl compile-time issues).
   Flipping them back would not suffice:
   the MLIR Enzyme dialect has no op that could reference
   a Julia rule body, so honoring `EnzymeRules` would require
   tracing `augmented_primal`/`reverse` into MLIR functions
   plus a new custom-rule op with Enzyme-MLIR support —
   a substantial upstream feature with no open issue as of the check
   (#2552 is the opposite direction:
   Enzyme.jl differentiating already-compiled Reactant kernels).
   Our rule bodies are pure, non-mutating array code —
   exactly the traceable kind — so a feature request is worth filing.
2. **Verified: keep the rules in Julia via stop-gradient surrogates.**
   Every `propagate` is linear in the incoming relevance,
   so each rule node can be rewritten as a surrogate forward
   `y .+ (z .- stop(z)) .* stop(y ./ modify_denominator(rule, z))`
   (with `stop = EnzymeCore.ignore_derivatives`,
   which Reactant lowers to `enzyme.ignore_derivatives`),
   whose value is exactly `y`
   and whose *mechanical* pullback is the rule's relevance map —
   the "LRP as gradient of a modified forward pass" formulation
   (Montavon et al.).

   Writing $[\,\cdot\,]$ for stop-gradient,
   $\odot$/$\oslash$ for elementwise product and division,
   and eliding denominator stabilization,
   the generic rule (here for unmodified input, $\tilde{a} = a^k$)

   ```math
   R^k
   = a^k \odot \tilde{J}^\top \bigl( R^{k+1} \oslash \tilde{z} \bigr),
   \qquad
   \tilde{z} = \tilde{f}(a^k),
   ```

   becomes the surrogate node

   ```math
   g(x)
   = y + \bigl( \tilde{z}(x) - [\tilde{z}] \bigr)
         \odot \bigl[ y \oslash \tilde{z} \bigr],
   ```

   whose value is exactly $y$.
   The invariant is that every edge value $v$
   carries the cotangent $\bar{v} = R(v) \oslash v$;
   under it, the surrogate's pullback reproduces the rule:

   ```math
   \bar{z}
   = \bar{y} \odot \bigl[ y \oslash \tilde{z} \bigr]
   = R^{k+1} \oslash \tilde{z},
   \qquad
   x \odot \bar{x}
   = a^k \odot \tilde{J}^\top \bar{z}
   = R^k .
   ```

   The pass is seeded with $\bar{y}^N = \mathrm{mask} \oslash [y^N]$
   at the model output
   and read out as $R^0 = x \odot \bar{x}$ at the input.
   Checked empirically (`EpsilonRule`, Dense/relu chain):
   the surrogate matches the `EnzymeRules` engine
   exactly under eager Enzyme
   and **bit-exactly under `@jit`** (max abs deviation 0.0).
   Deployment: a `ReactantExt` package extension defining
   `Reactant.@reactant_overlay` on `lrp_node` —
   under tracing the surrogate body runs,
   under plain Enzyme the `EnzymeRules` engine stays as is.
   Rule coverage (one representative per structural class
   verified bit-exact/1-ulp against the engine under `@jit`,
   reusing the package's `modify_params`/`modify_input`,
   on a `Dense`/`relu` chain with mixed-sign inputs;
   the check scripts should become tests of the eventual `ReactantExt`):
   - single-ratio rules (`Zero`/`Epsilon`/`Gamma`/`NegativeGamma`,
     `Pass` and the Dropout/Reshaping fast paths): direct translation —
     `Epsilon` and `Gamma` verified;
   - sign-split multi-branch rules
     (`ZPlus`/`AlphaBeta`/`GeneralizedGamma`):
     `max.(u, 0)`/`min.(u, 0)` are the *differentiable*
     reparametrization of the `aᵏ⁺`/`aᵏ⁻` splits,
     so the `x ⊙ dx` readout reproduces the `ã ⊙ c` terms exactly;
     `GeneralizedGamma`'s `I(z≷0)` masks fold into the stop-ratios —
     `AlphaBeta(2,1)` verified.
     For `AlphaBetaRule(α, β)`,
     with $x^{\pm}$ the elementwise $\max(x, 0)$ / $\min(x, 0)$
     and branch pre-activations

     ```math
     z^{\alpha}(x) = W^{+} x^{+} + W^{-} x^{-} + b^{+},
     \qquad
     z^{\beta}(x)  = W^{+} x^{-} + W^{-} x^{+} + b^{-},
     ```

     the surrogate node is

     ```math
     g(x)
     = y
     + \alpha \bigl( z^{\alpha}(x) - [z^{\alpha}] \bigr)
              \odot \bigl[ y \oslash z^{\alpha} \bigr]
     - \beta  \bigl( z^{\beta}(x)  - [z^{\beta}]  \bigr)
              \odot \bigl[ y \oslash z^{\beta}  \bigr] ;
     ```

     since $\partial x^{\pm} / \partial x = \mathbb{1}_{x \gtrless 0}$,
     the readout $x \odot \bar{x}$ yields the engine's
     $a^{\pm} \odot c$ terms, including exact zeros;
   - modified-input rules (`Flat`/`WSquare`/`ZBox`):
     on the first layer (their documented placement,
     used by all composite presets)
     the input enters as `stop(ã) .+ ξ`
     and relevance is read as `stop(ã) ⊙ ∇ξ` — exact,
     `Flat` verified:

     ```math
     u = [\tilde{a}] + \xi,
     \qquad
     R^0 = [\tilde{a}] \odot \bar{\xi} \big|_{\xi = 0}
         = \tilde{a} \odot \tilde{J}^\top \bigl( R^1 \oslash \tilde{z} \bigr) ;
     ```

     `ZBox` needs one auxiliary `ξ` per bound term (`x`, `low`, `high`).
     *Caveat*: placed mid-network instead,
     the linearized-input form drops the relevance
     these rules assign to exactly-zero activations
     (a value edge `y = 0` cannot carry multiplicative readout);
   - `LayerNormRule`: mean-subtraction is self-adjoint,
     so `x .- mean(x)` (differentiable) followed by a stop-ratio node
     reproduces `aᵏ ⊙ (s .- mean(s))` mechanically —
     derived, not yet run:
     with the centering map $C(x) = x - \mu_x$, $C^\top = C$,
     and $z = C(a^k)$ the centered input,

     ```math
     \bar{z} = R^{k+1} \oslash z = s,
     \qquad
     R^k = x \odot C(\bar{z}) = a^k \odot ( s - \mu_s ) ;
     ```

   - the branch connection rule (`Parallel`/`SkipConnection`)
     is a single stop-ratio on the summed output $z = \sum_j y_j$,
     with cotangent fan-out to the branches handled by MLIR AD:

     ```math
     \bar{y}_i = \bar{z} = R \oslash z
     \;\Longrightarrow\;
     R_i = y_i \odot \bar{y}_i
         = y_i \odot R \oslash \textstyle\sum_j y_j ;
     ```

   - a single rule on a sub-model differentiates the sub-model
     directly (confirmed compiling, see `seeded_pullback` above).

   Remaining work:
   - `stabilize_denom`'s scalar branch (`iszero(d) && return`)
     does not trace; use the branch-free
     `d + ifelse(signbit(d), -eps, eps)` instead;
   - the relevance seed must be built outside the compiled region
     (or from one-hot ops instead of scalar `setindex!`);
   - the layerwise-relevance taps mutate a Julia store
     and would need to become traced outputs.

The two remaining sections of the original document —
a list of suspected GPU-unfriendly spots in the package
and a testing recommendation — were dropped when this file was trimmed:
both were superseded by measurement,
and two of the three suspected spots turned out to be wrong.
See "Corrections to the first-pass GPU notes" in `PLAN_GPU.md`.
