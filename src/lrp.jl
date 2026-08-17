#================#
# Model wrapping #
#================#

# Validate the rule NamedTuple against the model structure and check
# rule-layer compatibility. Mirrors the structure of `wrap_rules` below.
function check_rule_compat(rules::NamedTuple, model::Union{Chain,Parallel}, ps)
    layers = model.layers
    if keys(rules) != keys(layers)
        throw(
            ArgumentError(
                "Rule keys $(keys(rules)) don't match layer keys $(keys(layers))."
            ),
        )
    end
    for k in keys(layers)
        check_rule_compat(rules[k], layers[k], ps[k])
    end
end
# `SkipConnection` is an `AbstractLuxWrapperLayer`: its `ps`/`st` pass through
# to the wrapped layer directly, and so do its rules.
check_rule_compat(rules, sc::SkipConnection, ps) = check_rule_compat(rules, sc.layers, ps)
# Disambiguation: a rule assigned to a SkipConnection targets the wrapped layer
function check_rule_compat(rule::AbstractLRPRule, sc::SkipConnection, ps)
    return check_rule_compat(rule, sc.layers, ps)
end
function check_rule_compat(rule::AbstractLRPRule, layer, ps)
    is_compatible(rule, layer, ps) || throw(LRPCompatibilityError(rule, layer))
    return nothing
end
function check_rule_compat(rules, layer, ps)
    throw(ArgumentError("Expected an LRP rule for layer $layer, got $rules."))
end

# Wrap the model in rule-carrying nodes
# by zipping rules and layers along the model structure:
# leaves become `LayerWithRule`s,
# branch connections become `ConnectionWithRule`s.
# Assumes `check_rule_compat` has validated the structure.
function wrap_children(layers::NamedTuple, rules::NamedTuple)
    return NamedTuple{keys(layers)}(map(wrap_rules, values(layers), values(rules)))
end
wrap_rules(model::Chain, rules::NamedTuple) = Chain(; wrap_children(model.layers, rules)...)
function wrap_rules(p::Parallel, rules::NamedTuple)
    return Parallel(ConnectionWithRule(p.connection); wrap_children(p.layers, rules)...)
end
function wrap_rules(sc::SkipConnection, rules)
    return SkipConnection(wrap_rules(sc.layers, rules), ConnectionWithRule(sc.connection))
end
function wrap_rules(sc::SkipConnection, rule::AbstractLRPRule)
    return invoke(wrap_rules, Tuple{SkipConnection,Any}, sc, rule)
end
# A single rule assigned to a container treats the sub-model as one
# differentiation unit; `Chain`/`Parallel` methods disambiguate against the
# container methods above.
wrap_rules(layer, rule::AbstractLRPRule) = LayerWithRule(rule, layer)
wrap_rules(layer::Chain, rule::AbstractLRPRule) = LayerWithRule(rule, layer)
wrap_rules(layer::Parallel, rule::AbstractLRPRule) = LayerWithRule(rule, layer)

#=============================#
# LRP struct and constructors #
#=============================#

"""
    LRP(model, ps, st, rules)
    LRP(model, ps, st, composite)
    LRP(model, ps, st)

Analyze a Lux model by applying Layer-Wise Relevance Propagation.

The analyzer is constructed from the Lux triple of `model`, parameters `ps`
and states `st`, as returned by `Lux.setup`. Since LRP is inference-only,
states are converted once via `Lux.testmode` at construction.

Rules are assigned to layers by passing either
- a `NamedTuple` of LRP rules mirroring the keys of `model.layers`,
- an `AbstractVector` of LRP rules for flat models, matched positionally, or
- a [`Composite`](@ref), which assigns rules based on layer type and position.
If no rules are passed, [`ZeroRule`](@ref) is used on all layers.

Nested `Chain` and `Parallel` layers take a nested `NamedTuple` of rules
mirroring their children; a single rule assigned to such a sub-model instead
treats it as one unit, differentiating through the entire sub-model at once.
A single rule assigned to a `SkipConnection` applies to the wrapped layer,
after relevance has been distributed between the skip and wrapped branches.

# Keyword arguments
- `normalize_output_relevance`: Selects whether output relevance should be set to 1 before applying LRP backward pass.
    Defaults to `true` to match literature. If `false`, values of output activations are used.
- `skip_checks::Bool`: Skip checks whether model is compatible with LRP and contains output softmax. Defaults to `false`.
- `verbose::Bool`: Select whether the model checks should print a summary on failure. Defaults to `true`.

# References
[1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
[2] W. Samek et al., Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications
"""
struct LRP{M<:Chain,P,S,R<:NamedTuple} <: AbstractXAIMethod
    # `model`/`ps`/`st` are the Lux triple the analyzer was built from.
    # `call_analyzer` wraps each rule-carrying leaf of the model
    # in a `LayerWithRule` via `wrap_rules`;
    # since the wrappers are `ps`/`st`-transparent,
    # the original `ps`/`st` trees apply to the wrapped model unchanged.
    # One Enzyme reverse pass over the wrapped model computes the explanation.
    model::M
    ps::P
    st::S
    rules::R
    normalize_output_relevance::Bool

    function LRP(
        model::Chain,
        ps,
        st,
        rules::NamedTuple;
        normalize_output_relevance::Bool=true,
        skip_checks=false,
        verbose=true,
    )
        st = testmode(st)
        if !skip_checks
            check_output_softmax(model)
            check_lrp_compat(model; verbose=verbose)
        end
        check_rule_compat(rules, model, ps)
        return new{typeof(model),typeof(ps),typeof(st),typeof(rules)}(
            model, ps, st, rules, normalize_output_relevance
        )
    end
end

# Rules can be passed as a vector for flat models and are matched positionally
function LRP(model::Chain, ps, st, rules::AbstractVector; kwargs...)
    layer_keys = keys(model.layers)
    if length(rules) != length(layer_keys)
        throw(
            ArgumentError(
                "Got $(length(rules)) rules for a model with $(length(layer_keys)) layers."
            ),
        )
    end
    return LRP(model, ps, st, NamedTuple{layer_keys}(Tuple(rules)); kwargs...)
end

# Construct the NamedTuple of rules by applying a composite
function LRP(model::Chain, ps, st, c::Composite; kwargs...)
    return LRP(model, ps, st, lrp_rules(model, c); kwargs...)
end

# Convenience constructor without rules: use ZeroRule everywhere
function LRP(model::Chain, ps, st; kwargs...)
    return LRP(model, ps, st, map_layers(Returns(ZeroRule()), model); kwargs...)
end

#==========================#
# Call to the LRP analyzer #
#==========================#

# The scalar loss whose Enzyme gradient w.r.t. `x` is the explanation:
# `dot(mask, y)` seeds the reverse pass with the masked output relevance Rᴺ⁺¹,
# and the rule-carrying nodes in `model` replace every VJP on the way down.
function lrp_loss(model, x, ps, st, cap, ns, normalize)
    y = first(apply(model, x, ps, st))
    mask = detached_mask!(cap, y, ns, normalize)
    return dot(mask, y)
end

# Capture the model output and build the output relevance seed.
# Marked `EnzymeRules.inactive`, which detaches the mask from the
# differentiated graph: the seed is a constant w.r.t. the model output.
function detached_mask!(cap, y, ns::AbstractOutputSelector, normalize::Bool)
    cap[] = copy(y)
    return relevance_seed(y, ns(y), normalize)
end
EnzymeRules.inactive(::typeof(detached_mask!), args...) = nothing

function relevance_seed(y, idx, normalize::Bool)
    seed = zero(y)
    if normalize
        seed[idx] .= 1
    else
        seed[idx] .= y[idx]
    end
    return seed
end

model_output(model, x, ps, st) = first(apply(model, x, ps, st))

# Typed capture for the model output. The type is over-approximated by the
# compiler, so the `Ref` stays valid (if less precise) when inference fails.
function output_ref(model, x, ps, st)
    T = Base.promote_op(model_output, typeof(model), typeof(x), typeof(ps), typeof(st))
    T === Union{} && return Ref{Any}()
    return Ref{T}()
end

# Insert relevance taps between the outermost children of the wrapped model.
# The first child needs no tap: the relevance at its input is the input shadow.
function insert_taps(model::Chain, store)
    layers = model.layers
    ks = keys(layers)
    tapped = ntuple(
        i -> i == 1 ? layers[i] : TappedLayer(layers[i], store, i - 1), length(ks)
    )
    return Chain(; NamedTuple{ks}(tapped)...)
end

function call_analyzer(
    input::AbstractArray, lrp::LRP, ns::AbstractOutputSelector; layerwise_relevances=false
)
    (; ps, st, normalize_output_relevance) = lrp
    model = wrap_rules(lrp.model, lrp.rules)
    store = layerwise_relevances ? Vector{Any}(undef, length(model.layers) - 1) : nothing
    if !isnothing(store)
        model = insert_taps(model, store)
    end

    dx = make_zero(input)
    cap = output_ref(model, input, ps, st)
    autodiff(
        Reverse,
        lrp_loss,
        Active,
        Const(model),
        Duplicated(input, dx),
        Const(ps),
        Const(st),
        Const(cap),
        Const(ns),
        Const(normalize_output_relevance),
    )
    output = cap[]
    output_selection = ns(output)

    extras = if isnothing(store)
        nothing
    else
        seed = relevance_seed(output, output_selection, normalize_output_relevance)
        (; layerwise_relevances=(dx, store..., seed))
    end
    return Explanation(dx, input, output, output_selection, :LRP, :attribution, extras)
end
