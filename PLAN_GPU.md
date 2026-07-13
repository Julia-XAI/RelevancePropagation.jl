# Plan: GPU support

GPU support is out of scope for v4.0.0 and tracked here as follow-up work.

Enzyme itself supports GPUs (it differentiates CUDA kernels, and the Lux
ecosystem's recommended path for Enzyme on GPU is
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)).
What is untested is *this package's* use of Enzyme:
per-layer split-mode thunks over plain Julia arrays.

## Candidate approaches

1. **Reactant.jl** (preferred by the Lux ecosystem):
   compile model and analyzer to XLA and run on CPU/GPU/TPU.
   The LRP backward pass would need to be traceable:
   the current implementation mutates pre-allocated relevance arrays in place
   and constructs Enzyme thunks per layer, both of which likely need rework
   for traced arrays.
2. **CUDA.jl arrays through raw Enzyme**:
   keep the current architecture and verify that `layer_pullback` works with
   `CuArray` inputs for every supported layer type.
   NNlib's GPU kernels and Enzyme's CUDA support need to be validated per
   layer; this mirrors the CPU spike that preceded the v4 port.

## Known GPU-unfriendly spots in the current code

- `mask_output_neuron!` writes to individual output indices.
- The `FlatRule` fast path for `Dense` fills views per batch sample.
- `lrp!` implementations assume cheap scalar indexing in a few analytic
  code paths; an audit for scalar indexing under `CUDA.allowscalar(false)`
  is needed.

## Testing

- Add a GPU test job (JLArrays.jl for CI without GPU hardware, or a
  Buildkite CUDA pipeline) covering `layer_pullback` and one end-to-end
  `analyze` per composite preset.
