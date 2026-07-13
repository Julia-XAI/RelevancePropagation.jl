#==============================#
# FrozenLayer & Enzyme AD core #
#==============================#

# This file contains `FrozenLayer` — the bundled callable Lux triple all LRP
# rules operate on — its execution helpers, and all Enzyme-specific code in
# the package. The only AD primitive LRP requires is a vector-Jacobian
# product (VJP) w.r.t. a layer's input — never w.r.t. its parameters.

"""
    FrozenLayer(layer, ps, st)

Immutable bundle of a Lux `layer` with its parameters `ps` and states `st`.
Calling it applies the layer to an input, discarding the updated layer states.

LRP rules and [`modify_layer`](@ref) operate on `FrozenLayer`s;
modified parameters are stored in new `FrozenLayer` instances.
Since `FrozenLayer` is immutable and never differentiated w.r.t. its contents,
it is safe to annotate it as `Enzyme.Const` in [`layer_pullback`](@ref).
"""
struct FrozenLayer{L,P,S}
    layer::L
    ps::P
    st::S
end
(f::FrozenLayer)(x) = first(apply(f.layer, x, f.ps, f.st))

Base.show(io::IO, f::FrozenLayer) = print(io, "FrozenLayer(", f.layer, ")")

# Bundle the children of a `Chain` or `Parallel` with their `ps`/`st` into a
# NamedTuple of `FrozenLayer`s mirroring `model.layers`.
function frozen_children(f::FrozenLayer{<:Union{Chain,Parallel}})
    layers = f.layer.layers
    return NamedTuple{keys(layers)}(
        map(FrozenLayer, values(layers), values(f.ps), values(f.st))
    )
end

# `SkipConnection` is an `AbstractLuxWrapperLayer`:
# its `ps`/`st` pass through to the wrapped layer directly.
frozen_inner(f::FrozenLayer{<:SkipConnection}) = FrozenLayer(f.layer.layers, f.ps, f.st)

# Compute activations of all layers, including the input.
# Returns a tuple `(input, a¹, a², ..., aᴺ)` of length `length(layers) + 1`.
# An execution helper, not a structural rewrite: it runs on NamedTuples of
# callable `FrozenLayer`s (or their rule-modified counterparts), sits on the
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

Compute the primal output `z = layer(x)` of a [`FrozenLayer`](@ref) and return
`(z, back)`, where `back(s)` evaluates the VJP of `layer` at `x` with seed `s`
w.r.t. the input `x`.

`back` may be called at most once per `layer_pullback` call:
Enzyme split-mode reverse thunks cannot be re-run on the same tape.
Rules that need VJPs with several seeds through the same layer
construct one pullback per seed.
"""
function layer_pullback(f::F, x::AbstractArray) where {F<:FrozenLayer}
    fwd, rev = autodiff_thunk(
        ReverseSplitWithPrimal, Const{F}, Duplicated, Duplicated{typeof(x)}
    )
    dx = make_zero(x)
    tape, z, dz = fwd(Const(f), Duplicated(x, dx))
    function back(s)
        dz .= s
        rev(Const(f), Duplicated(x, dx), tape)
        return dx
    end
    return z, back
end
