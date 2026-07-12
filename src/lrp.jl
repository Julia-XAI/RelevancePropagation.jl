#===================#
# Structure helpers #
#===================#

"""
    map_layers(f, model)

Apply `f` to each layer of a Lux model, mirroring the model structure
as nested `NamedTuple`s keyed like the model's `ps` and `st`.
"""
function map_layers(f, model::Union{Chain,Parallel})
    layers = model.layers
    return NamedTuple{keys(layers)}(map(l -> map_layers(f, l), values(layers)))
end
map_layers(f, model::SkipConnection) = map_layers(f, model.layers)
map_layers(f, layer) = f(layer)

# `Chain` and `Parallel` store their children in a `layers` NamedTuple
# that `ps` and `st` mirror.
function frozen_children(f::FrozenLayer{<:Union{Chain,Parallel}})
    layers = f.layer.layers
    return NamedTuple{keys(layers)}(
        map(FrozenLayer, values(layers), values(f.ps), values(f.st))
    )
end

# `SkipConnection` is an `AbstractLuxWrapperLayer`:
# its `ps`/`st` pass through to the wrapped layer directly.
frozen_inner(f::FrozenLayer{<:SkipConnection}) = FrozenLayer(f.layer.layers, f.ps, f.st)

# Construct the NamedTuple of modified layers by zipping rules and layers
# along the model structure.
function get_modified_layers(rules::NamedTuple, frozen::FrozenLayer{<:Union{Chain,Parallel}})
    children = frozen_children(frozen)
    if keys(rules) != keys(children)
        throw(
            ArgumentError(
                "Rule keys $(keys(rules)) don't match layer keys $(keys(children))."
            ),
        )
    end
    return NamedTuple{keys(children)}(
        map(get_modified_layers, values(rules), values(children))
    )
end
function get_modified_layers(rules, frozen::FrozenLayer{<:SkipConnection})
    return get_modified_layers(rules, frozen_inner(frozen))
end
get_modified_layers(rule::AbstractLRPRule, frozen::FrozenLayer) = modify_layer(rule, frozen)
# Disambiguation: a rule assigned to a SkipConnection recurses into the wrapped layer
function get_modified_layers(rule::AbstractLRPRule, frozen::FrozenLayer{<:SkipConnection})
    return get_modified_layers(rule, frozen_inner(frozen))
end
function get_modified_layers(rules, frozen::FrozenLayer)
    throw(ArgumentError("Expected an LRP rule for layer $(frozen.layer), got $rules."))
end

#=============================#
# LRP struct and constructors #
#=============================#

"""
    LRP(model, ps, st, rules)
    LRP(model, ps, st)

Analyze a Lux model by applying Layer-Wise Relevance Propagation.

The analyzer is constructed from the Lux triple of `model`, parameters `ps`
and states `st`, as returned by `Lux.setup`. Since LRP is inference-only,
states are converted once via `Lux.testmode` at construction.

Rules are assigned to layers by passing either
- a `NamedTuple` of LRP rules mirroring the keys of `model.layers`, or
- an `AbstractVector` of LRP rules for flat models, matched positionally.
If no rules are passed, [`ZeroRule`](@ref) is used on all layers.

# Keyword arguments
- `normalize_output_relevance`: Selects whether output relevance should be set to 1 before applying LRP backward pass.
    Defaults to `true` to match literature. If `false`, values of output activations are used.
- `skip_checks::Bool`: Skip checks whether model is compatible with LRP and contains output softmax. Defaults to `false`.
- `verbose::Bool`: Select whether the model checks should print a summary on failure. Defaults to `true`.

# References
[1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
[2] W. Samek et al., Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications
"""
struct LRP{M<:Chain,P,S,R<:NamedTuple,L<:NamedTuple,ML<:NamedTuple} <: AbstractXAIMethod
    model::M
    ps::P
    st::S
    rules::R
    layers::L            # FrozenLayers mirroring model.layers
    modified_layers::ML  # rule-modified FrozenLayers, mirroring model.layers
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
        frozen = FrozenLayer(model, ps, st)
        layers = frozen_children(frozen)
        modified_layers = get_modified_layers(rules, frozen)
        return new{
            typeof(model),
            typeof(ps),
            typeof(st),
            typeof(rules),
            typeof(layers),
            typeof(modified_layers),
        }(
            model, ps, st, rules, layers, modified_layers, normalize_output_relevance
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

# Convenience constructor without rules: use ZeroRule everywhere
function LRP(model::Chain, ps, st; kwargs...)
    return LRP(model, ps, st, map_layers(Returns(ZeroRule()), model); kwargs...)
end

#==========================#
# Call to the LRP analyzer #
#==========================#

function call_analyzer(
    input::AbstractArray, lrp::LRP, ns::AbstractOutputSelector; layerwise_relevances=false
)
    as = get_activations(lrp.layers, input)   # compute activations aᵏ for all layers k
    Rs = similar.(as)
    mask_output_neuron!(Rs[end], as[end], ns, lrp.normalize_output_relevance) # compute relevance Rᴺ of output layer N
    lrp_backward_pass!(Rs, as, lrp.rules, lrp.layers, lrp.modified_layers)
    extras = layerwise_relevances ? (layerwise_relevances=Rs,) : nothing
    return Explanation(first(Rs), input, last(as), ns(last(as)), :LRP, :attribution, extras)
end

# Compute activations of all layers, including the input.
# Returns a tuple `(input, a¹, a², ..., aᴺ)` of length `length(layers) + 1`.
get_activations(layers::NamedTuple, input) = (input, _activations(values(layers), input)...)

function _activations(layers::Tuple, x)
    isempty(layers) && return ()
    y = first(layers)(x)
    return (y, _activations(Base.tail(layers), y)...)
end

function mask_output_neuron!(
    R_out, a_out, ns::AbstractOutputSelector, normalize_output_relevance::Bool
)
    fill!(R_out, 0)
    idx = ns(a_out)
    if normalize_output_relevance
        R_out[idx] .= 1
    else
        R_out[idx] .= a_out[idx]
    end
    return R_out
end

function lrp_backward_pass!(Rs, as, rules, layers::NamedTuple, modified_layers)
    # Apply LRP rules in backward-pass, inplace-updating relevances `Rs[k]` = Rᵏ
    rs, ls, ms = values(rules), values(layers), values(modified_layers)
    for k in length(ls):-1:1
        lrp!(Rs[k], rs[k], ls[k], ms[k], as[k], Rs[k + 1])
    end
    return Rs
end
