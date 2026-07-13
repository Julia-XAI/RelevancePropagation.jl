# `activation_fn` is a ModelSurgeon function; extending it with a method for
# RP's own `FrozenLayer` type is piracy-free.
ModelSurgeon.activation_fn(f::FrozenLayer) = activation_fn(f.layer)

# Parameters live in `ps`, whose entries Lux names uniformly:
# `weight` and `bias` for all layers modified by LRP rules (`Dense`, `Scale`,
# convolutions). Layers constructed with `use_bias=false` have no `bias` key.
has_weight(f::FrozenLayer) = haskey(f.ps, :weight)
has_bias(f::FrozenLayer) = haskey(f.ps, :bias)
