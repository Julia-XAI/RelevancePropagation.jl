# TODO: relevance-conservation tolerance in `test_cnn.jl` / `test_batches.jl`

During the Lux/Enzyme port, the "Normalized output relevance" checks were
loosened from `isapprox(sum(v1), 1; atol=0.05)` to `atol=0.15`:
with the re-seeded `Lux.setup(StableRNG(123), model)` parameters, `sum(R)` is
0.90 (`test_cnn.jl`) and 1.07 (`test_batches.jl`) instead of within 0.05 of 1.

## Investigation status (2026-08-11)

Hypotheses tested on the `test_cnn.jl` CNN (`LRP` with `ZeroRule`, Julia 1.12,
Enzyme v0.13):

1. **Enzyme mutating stored activations on the backward pass — refuted.**
   `get_activations` output captured before `analyze` is bit-compatible
   (`≈` element-wise) with a fresh forward pass afterwards. The
   `Duplicated(x, dx)` shadow only accumulates into `dx`; the primal `x`
   (the stored `aᵏ`) is untouched.
2. **Bias absorption — confirmed as the cause.** The per-layer relevance sums
   (via `layerwise_relevances=true`) drift by ≤3% per layer and drop
   0.87 ← 1.03 at the first Conv layer; each drop happens at a layer with a
   bias, exactly as LRP theory predicts (LRP-0 assigns relevance to bias
   terms, which is then lost to the input).
3. **Control: a bias-free (`use_bias=false`) copy of the same CNN conserves
   `sum(R) = 1.000001`.** The Enzyme VJP machinery is exactly conservative;
   deviations come only from bias terms and denominator stabilizers.
   This invariant is now pinned by the "Relevance conservation (bias-free)"
   testset in `test/test_cnn.jl` with `atol=1e-4`.

Conclusion: the larger deviation is a property of the new parameter draw
(different bias/activation magnitudes), not of the Zygote → Enzyme switch.

## Remaining work

- [ ] Replace the weak `atol=0.15` band with values pinned to the actual
      draws, e.g. `@test sum(v1) ≈ 0.90 rtol = 0.02` (test_cnn) and
      `@test sum(v1) ≈ 1.07 rtol = 0.02` (test_batches), so a future
      regression in the backward pass cannot hide inside the band.
      Alternatively, pick a seed whose bias absorption lands closer to 1 and
      restore `atol=0.05`.
- [ ] Optional cross-check against v3: port the v3 reference model's Flux
      weights into the Lux `ps` NamedTuple (the layout-compatibility trick
      already used in `test_rules.jl`) and confirm v3's `sum(R)` is
      reproduced under Enzyme.
