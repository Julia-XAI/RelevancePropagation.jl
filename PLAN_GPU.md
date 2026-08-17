# Plan: GPU support

GPU support is out of scope for v4.0.0 and tracked here as follow-up work.

Enzyme itself supports GPUs (it differentiates CUDA kernels, and the Lux
ecosystem's recommended path for Enzyme on GPU is
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)).
What is untested is *this package's* use of Enzyme:
one reverse pass over the wrapped model with `EnzymeRules` custom rules per
layer node, plus the nested combined-mode `autodiff` calls
(`seeded_pullback`) inside rule bodies.

## Candidate approaches

1. **Reactant.jl** (preferred by the Lux ecosystem):
   compile model and analyzer to XLA and run on CPU/GPU/TPU.
   The engine redesign helps here: the backward pass is a single
   `Enzyme.autodiff` over non-mutating rule bodies — no pre-allocated
   relevance buffers, no per-layer thunk construction. The open question
   is whether Reactant traces through `EnzymeRules` custom rules and the
   nested `autodiff` in `seeded_pullback`.
2. **CUDA.jl arrays through raw Enzyme**:
   keep the architecture and validate the custom rules with `CuArray`s.
   The `input_vjp` fast paths (`Wᵀs`, broadcast, `∇conv_data`, `conv`)
   are plain LinearAlgebra/NNlib calls with existing GPU kernels;
   `seeded_pullback` needs Enzyme-on-CUDA validation per remaining layer
   type.

## Known GPU-unfriendly spots in the current code

- `relevance_seed` writes to individual output indices
  (`seed[idx] .= 1` with `idx::Vector{CartesianIndex{2}}`) — scalar
  indexing under `CUDA.allowscalar(false)`. Same pattern in CRP's
  concept masking.
- The `FlatRule` fast path for `Dense` fills views per batch sample.
- An audit of the analytic `propagate` methods for scalar indexing under
  `CUDA.allowscalar(false)` is needed.

## Testing

- Add a GPU test job (JLArrays.jl for CI without GPU hardware, or a
  Buildkite CUDA pipeline) covering `input_vjp`/`seeded_pullback` per
  supported layer type and one end-to-end `analyze` per composite preset.
