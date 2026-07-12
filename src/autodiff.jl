#================#
# Enzyme AD core #
#================#

# This file contains all Enzyme-specific code in the package.
# The only AD primitive LRP requires is a vector-Jacobian product (VJP)
# w.r.t. a layer's input — never w.r.t. its parameters.

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

"""
    layer_pullback(layer, x)

Compute the primal output `z = layer(x)` of a [`FrozenLayer`](@ref) and return
`(z, back)`, where `back(s)` evaluates the VJP of `layer` at `x` with seed `s`
w.r.t. the input `x`.

`back` may be called at most once per `layer_pullback` call:
Enzyme split-mode reverse thunks cannot be re-run on the same tape.
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

"""
    layer_pullback_2seeds(layer, x)

Compute the primal output `z = layer(x)` of a [`FrozenLayer`](@ref) and return
`(z, back2)`, where `back2(s₁, s₂)` evaluates two VJPs of `layer` at `x` and
returns `(dx₁, dx₂)`. Both seeds share one forward and one reverse pass through
a width-2 `BatchDuplicated` shadow.

`back2` may be called at most once per `layer_pullback_2seeds` call.

!!! warning
    Only use on layers with weight and bias parameters (`Dense`, `Conv`, ...).
    Compiling width-2 thunks through pooling and normalization layers triggers
    Enzyme compiler crashes (see the spike notes in PLAN.md); rules relying on
    two-seed pullbacks are restricted to weight-bias layers anyway.
"""
function layer_pullback_2seeds(f::F, x::AbstractArray) where {F<:FrozenLayer}
    mode = ReverseSplitWidth(ReverseSplitWithPrimal, Val(2))
    fwd, rev = autodiff_thunk(mode, Const{F}, BatchDuplicated, BatchDuplicated{typeof(x),2})
    dx₁, dx₂ = make_zero(x), make_zero(x)
    tape, z, dzs = fwd(Const(f), BatchDuplicated(x, (dx₁, dx₂)))
    function back2(s₁, s₂)
        dzs[1] .= s₁
        dzs[2] .= s₂
        rev(Const(f), BatchDuplicated(x, (dx₁, dx₂)), tape)
        return dx₁, dx₂
    end
    return z, back2
end
