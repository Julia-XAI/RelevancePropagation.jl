# LRP as a hijacked reverse-mode AD pass.
#
# LRP *is* reverse-mode AD in which each layer's true VJP is replaced by the
# rule's relevance-propagation map: the relevance Rᵏ is the cotangent at the
# layer input aᵏ. The generic rule Rᵏ = ãᵏ ⊙ J̃ᵀ(Rᵏ⁺¹ ./ z̃) is a modified VJP:
# massage the incoming cotangent (÷ z̃), pull it back through a ρ-modified
# layer, massage the outgoing cotangent (⊙ ãᵏ).
#
# The engine therefore runs *one* Enzyme reverse pass over the wrapped model
# per `analyze` call. Each rule-carrying node is an `EnzymeRules` custom rule
# whose reverse computes the rule's relevance map (`propagate`, see rules.jl);
# the input shadow `dx` accumulated by the pass *is* the explanation. Lux's
# own `apply` plumbing routes dataflow through `Chain`/`Parallel`/
# `SkipConnection`, and shadow accumulation implements "sum branch relevances"
# without any structural special cases.

#=====================#
# Rule-carrying nodes #
#=====================#

"""
    lrp_node(rule, layer, x, ps, st)

Apply `layer` to the input `x` with parameters `ps` and states `st`,
discarding the updated layer states.

This is the unit of the hijacked reverse pass: an `EnzymeRules` custom rule
replaces its VJP with the LRP rule's relevance propagation ([`propagate`](@ref)),
so differentiating a model in which each layer call is routed through
`lrp_node` computes relevances instead of gradients.
"""
lrp_node(rule, layer, x, ps, st) = first(apply(layer, x, ps, st))

"""
    RuledLayer(rule, layer)

Lux wrapper layer that pairs a `layer` with the LRP `rule` assigned to it.
Applying it routes the layer call through [`lrp_node`](@ref).

The wrapper is parameter- and state-transparent
(`AbstractLuxWrapperLayer{:layer}`), so the `ps`/`st` trees of the unwrapped
model apply to the wrapped model unchanged.
"""
struct RuledLayer{R,L} <: AbstractLuxWrapperLayer{:layer}
    rule::R
    layer::L
end
function LuxCore.apply(rl::RuledLayer, x, ps, st)
    # Return an empty state: LRP is inference-only and discards state updates.
    # Passing `st` through the return value would thread `Const` state arrays
    # (e.g. BatchNorm running statistics) into the active data flow of the
    # outer chain, triggering Enzyme runtime-activity errors. The real `st`
    # enters `lrp_node` only as a `Const` argument.
    return lrp_node(rl.rule, rl.layer, x, ps, st), NamedTuple()
end
Base.show(io::IO, rl::RuledLayer) = print(io, "RuledLayer(", rl.rule, ", ", rl.layer, ")")

# Layers whose forward pass is `σ.(affine(x))`: their forward is split into
# affine part and activation so the pre-activation `z` is cached on the tape.
# Rules on these layers propagate relevance through the affine part only
# (v3 semantics: `modify_layer` stripped activations of layers with weights),
# so rules whose parameter modification is the identity can reuse the cached
# `z` without any additional forward pass.
const ActivationSplitLayer = Union{Dense,Scale,ConvLayer}

"""
    node_forward(layer, x, ps, st)

Compute the forward pass of a layer, returning `(z, y)` where `z` is the
pre-activation and `y` the layer output. For layers that are not split into
affine part and activation, `z == y`.
"""
function node_forward(layer, x, ps, st)
    y = first(apply(layer, x, ps, st))
    return y, y
end
function node_forward(layer::ActivationSplitLayer, x, ps, st)
    z = first(apply(remove_activation(layer), x, ps, st))
    return z, node_output(layer, z)
end

"""
    node_output(layer, z)

Recover the layer output `y` from the cached pre-activation `z`
by applying the layer's activation function.
"""
node_output(layer, z) = z
function node_output(layer::ActivationSplitLayer, z)
    σ = layer.activation
    return σ === identity ? z : σ.(z)
end

"""
    rule_layer(layer)

The layer a rule propagates relevance through: layers with weights have their
activation function stripped, all other layers are returned unchanged.
"""
rule_layer(layer) = layer
rule_layer(layer::ActivationSplitLayer) = remove_activation(layer)

function EnzymeRules.augmented_primal(
    config::RevConfig,
    ::Const{typeof(lrp_node)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
    rule::Const,
    layer::Const,
    x::Duplicated,
    ps::Const,
    st::Const,
)
    z, y = node_forward(layer.val, x.val, ps.val, st.val)
    dy = needs_shadow(config) ? make_zero(y) : nothing
    # `x` is only copied if Enzyme reports it may be overwritten before the
    # reverse pass; the function itself is index 1 of `overwritten`.
    xᵏ = overwritten(config)[4] ? copy(x.val) : x.val
    tape = (; x=xᵏ, z, dy)
    return AugmentedReturn(needs_primal(config) ? y : nothing, dy, tape)
end

function EnzymeRules.reverse(
    ::RevConfig,
    ::Const{typeof(lrp_node)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
    tape,
    rule::Const,
    layer::Const,
    x::Duplicated,
    ps::Const,
    st::Const,
)
    # By the time this node's reverse runs, the return shadow `tape.dy` holds
    # the relevance Rᵏ⁺¹ accumulated by all downstream nodes.
    if !isnothing(tape.dy)
        Rᵏ = propagate(rule.val, layer.val, tape.x, tape.z, ps.val, st.val, tape.dy)
        x.dval .+= Rᵏ
    end
    return (nothing, nothing, nothing, nothing, nothing)
end

#===================================#
# Branch connections (Parallel etc) #
#===================================#

"""
    RuledConnection(connection)

Callable wrapper for the `connection` of a `Parallel` or `SkipConnection`
layer. Its `EnzymeRules` custom rule distributes the incoming relevance to the
branches proportionally to their contribution `yᵢ` to the connection output:
`Rᵢ = yᵢ ⊙ R ./ Σⱼyⱼ`. Shadow accumulation on the branch inputs then sums the
branch relevances without re-running any forward passes.
"""
struct RuledConnection{C}
    connection::C
end
(rc::RuledConnection)(ys...) = lrp_connection(rc.connection, ys...)

lrp_connection(connection, ys...) = connection(ys...)

function EnzymeRules.augmented_primal(
    config::RevConfig,
    ::Const{typeof(lrp_connection)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
    connection::Const,
    ys::Vararg{Duplicated,N},
) where {N}
    yvals = map(y -> y.val, ys)
    z = connection.val(yvals...)
    dz = needs_shadow(config) ? make_zero(z) : nothing
    tape = (; ys=map(copy, yvals), z, dz)
    return AugmentedReturn(needs_primal(config) ? z : nothing, dz, tape)
end

function EnzymeRules.reverse(
    ::RevConfig,
    ::Const{typeof(lrp_connection)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
    tape,
    connection::Const,
    ys::Vararg{Duplicated,N},
) where {N}
    if !isnothing(tape.dz)
        s = tape.dz ./ stabilize_denom(tape.z)
        for (y, yᵏ) in zip(ys, tape.ys)
            y.dval .+= yᵏ .* s
        end
    end
    return (nothing, ntuple(Returns(nothing), N)...)
end

#=====================================#
# VJPs w.r.t. the input of a layer    #
#=====================================#

"""
    seeded_pullback(layer, x, ps, st, s)

Compute the VJP of `layer` at `x` with seed `s` w.r.t. the input `x` using a
nested Enzyme reverse pass over the scalar loss `dot(layer(x), s)`.

This is the generic AD fallback of [`input_vjp`](@ref) and the relevance
propagator for sub-models treated as a single differentiation unit.
"""
function seeded_pullback(layer, x, ps, st, s)
    dx = make_zero(x)
    autodiff(
        Reverse,
        seeded_apply,
        Active,
        Const(layer),
        Duplicated(x, dx),
        Const(ps),
        Const(st),
        Const(s),
    )
    return dx
end

seeded_apply(layer, x, ps, st, s) = dot(first(apply(layer, x, ps, st)), s)

"""
    input_vjp(layer, x, ps, st, s)

Compute the VJP of `layer` at `x` with seed `s` w.r.t. the input `x`.

Activation-free `Dense`, `Scale`, `Conv` and `ConvTranspose` layers use
hand-written fast paths (one transpose op, no nested AD); everything else
falls back to [`seeded_pullback`](@ref). The fast paths are cross-checked
against the fallback in the test suite.
"""
input_vjp(layer, x, ps, st, s) = seeded_pullback(layer, x, ps, st, s)

function input_vjp(layer::Dense, x, ps, st, s)
    layer.activation === identity || return seeded_pullback(layer, x, ps, st, s)
    return ps.weight' * s
end

function input_vjp(layer::Scale, x, ps, st, s)
    layer.activation === identity || return seeded_pullback(layer, x, ps, st, s)
    return s .* ps.weight
end

function input_vjp(layer::Conv, x, ps, st, s)
    layer.activation === identity || return seeded_pullback(layer, x, ps, st, s)
    cdims = DenseConvDims(
        x,
        ps.weight;
        layer.stride,
        padding=layer.pad,
        layer.dilation,
        layer.groups,
        flipkernel=known(layer.cross_correlation),
    )
    return ∇conv_data(s, ps.weight, cdims)
end

function input_vjp(layer::ConvTranspose, x, ps, st, s)
    layer.activation === identity || return seeded_pullback(layer, x, ps, st, s)
    # The forward pass is `∇conv_data(x, weight, cdims)` for a convolution
    # mapping the *output* space back to the input space, so the VJP w.r.t.
    # `x` is that convolution applied to the seed. Since `s` has the shape of
    # the layer output, the `DenseConvDims` of said convolution can be
    # constructed from `s` directly, which also accounts for `outpad`.
    cdims = DenseConvDims(
        s,
        ps.weight;
        layer.stride,
        padding=layer.pad,
        layer.dilation,
        layer.groups,
        flipkernel=known(layer.cross_correlation),
    )
    return conv(s, ps.weight, cdims)
end

#=========================================#
# Relevance taps for layerwise relevances #
#=========================================#

"""
    TappedLayer(layer, store, index)

Lux wrapper layer that records the relevance flowing into `layer` during the
reverse pass in `store[index]`. Used to implement the `layerwise_relevances`
keyword argument of `analyze`.
"""
struct TappedLayer{L,S} <: AbstractLuxWrapperLayer{:layer}
    layer::L
    store::S
    index::Int
end
function LuxCore.apply(t::TappedLayer, x, ps, st)
    return apply(t.layer, tap_relevance(t.store, t.index, x), ps, st)
end

tap_relevance(store, index, x) = x

function EnzymeRules.augmented_primal(
    config::RevConfig,
    ::Const{typeof(tap_relevance)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
    store::Const,
    index::Const,
    x::Duplicated,
)
    # The primal is copied instead of passed through: custom rules should not
    # alias their input into their return value.
    y = copy(x.val)
    dy = needs_shadow(config) ? make_zero(y) : nothing
    return AugmentedReturn(needs_primal(config) ? y : nothing, dy, dy)
end

function EnzymeRules.reverse(
    ::RevConfig,
    ::Const{typeof(tap_relevance)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
    dy,
    store::Const,
    index::Const,
    x::Duplicated,
)
    if !isnothing(dy)
        store.val[index.val] = copy(dy)
        x.dval .+= dy
    end
    return (nothing, nothing, nothing)
end
