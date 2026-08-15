# Named `StaticLayer` (not `FrozenLayer`) to avoid confusion with the unrelated
# `Lux.Experimental.FrozenLayer`, which freezes *parameters* during training.
"""
    StaticLayer(layer, ps, st)

Immutable bundle of a Lux `layer` with its parameters `ps` and states `st`.
Calling it applies the layer to an input, discarding the updated layer states.

LRP rules and [`modify_layer`](@ref) operate on `StaticLayer`s;
modified parameters are stored in new `StaticLayer` instances.
Since `StaticLayer` is immutable and never differentiated w.r.t. its contents,
it is safe to annotate it as `Enzyme.Const` in [`layer_pullback`](@ref).
"""
struct StaticLayer{L,P,S}
    layer::L
    ps::P
    st::S
end
(f::StaticLayer)(x) = first(apply(f.layer, x, f.ps, f.st))
# ISSUE: I don't yet fully understand why this is needed. Use Lux and Zygote natively and ideomatically instead of trying to force Flux+Zygote like behavior.

Base.show(io::IO, f::StaticLayer) = print(io, "StaticLayer(", f.layer, ")")

# Bundle the children of a `Chain` or `Parallel` with their `ps`/`st` into a
# NamedTuple of `StaticLayer`s mirroring `model.layers`.
function static_children(f::StaticLayer{<:Union{Chain,Parallel}})
    layers = f.layer.layers
    return NamedTuple{keys(layers)}(
        map(StaticLayer, values(layers), values(f.ps), values(f.st))
    )
end

# `SkipConnection` is an `AbstractLuxWrapperLayer`:
# its `ps`/`st` pass through to the wrapped layer directly.
# `<:` because `SkipConnection` is parametric (`SkipConnection{L,C}`), so the
# dispatch must match any concrete instantiation, not the bare `UnionAll` type.
static_inner(f::StaticLayer{<:SkipConnection}) = StaticLayer(f.layer.layers, f.ps, f.st)

# Compute activations of all layers, including the input.
# Returns a tuple `(input, a¹, a², ..., aᴺ)` of length `length(layers) + 1`.
# An execution helper, not a structural rewrite: it runs on NamedTuples of
# callable `StaticLayer`s (or their rule-modified counterparts), sits on the
# hot path of every `analyze` call, and relies on tuple recursion for
# inferrability.
get_activations(layers::NamedTuple, input) = (input, _activations(values(layers), input)...)

function _activations(layers::Tuple, x)
    isempty(layers) && return ()
    y = first(layers)(x)
    return (y, _activations(Base.tail(layers), y)...)
end

"""
    layer_pullback(layer, x)

Compute the primal output `z = layer(x)` of a [`StaticLayer`](@ref) and return
`(z, back)`, where `back(s)` evaluates the VJP of `layer` at `x` with seed `s`
w.r.t. the input `x`.

`back` may be called at most once per `layer_pullback` call:
Enzyme split-mode reverse thunks cannot be re-run on the same tape.
Rules that need VJPs with several seeds through the same layer
construct one pullback per seed.
"""
function layer_pullback(f::F, x::AbstractArray) where {F<:StaticLayer}
    # ISSUE: this function is the biggest code smell in the entire PR.
    # Try to understand how LRP relates to AD and implement this from scratch in ideomatic Lux+Enzyme.
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{F}, Duplicated, Duplicated{typeof(x)}
    )
    dx = make_zero(x)
    tape, z, dz = fwd(Const(f), Duplicated(x, dx))
    # Every activity annotation is load-bearing: `Const(f)` (the layer is not
    # differentiated), the `Duplicated` return (we need both the primal `z` and
    # a seedable cotangent `dz`), and `Duplicated(x, dx)` (the VJP is taken
    # w.r.t. `x`, accumulating into `dx`). Split mode requires the *same*
    # `Duplicated(x, dx)` in `fwd` and `rev`, so it is threaded through both.
    function back(s)
        dz .= s
        rev(Const(f), Duplicated(x, dx), tape)
        return dx
    end
    return z, back
end
# Why a hand-rolled per-layer pullback instead of one Enzyme pass over the model?
# LRP is not plain backprop: each layer is differentiated through its
# *rule-modified* forward pass (`modify_layer`) with a relevance-derived seed,
# and the input relevance is `aᵏ .* vjp` (see `rules.jl`). We therefore need a
# VJP through a *modified* layer with a custom seed, per layer. A single Enzyme
# pass over the unmodified model — or an `EnzymeRule` teaching Enzyme how to
# differentiate a layer — would compute the model's gradient, not the LRP
# relevance redistribution.
