#=============================#
# LRP struct and constructors #
#=============================#

"""
    LRP(model, rules)
    LRP(model, composite)

Analyze model by applying Layer-Wise Relevance Propagation.
The analyzer can either be created by passing an array of LRP-rules
or by passing a composite, see [`Composite`](@ref) for an example.

# Keyword arguments
- `skip_checks::Bool`: Skip checks whether model is compatible with LRP and contains output softmax. Default is `false`.
- `verbose::Bool`: Select whether the model checks should print a summary on failure. Default is `true`.

# References
[1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
[2] W. Samek et al., Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications
"""
struct LRP{C<:Chain,R<:ChainTuple,L<:ChainTuple} <: AbstractXAIMethod
    model::C
    rules::R
    modified_layers::L

    # Construct LRP analyzer by assigning a rule to each layer
    function LRP(
        model::Chain, rules::ChainTuple; skip_checks=false, flatten=true, verbose=true
    )
        if flatten
            model = chainflatten(model)
            rules = chainflatten(rules)
        end
        if !skip_checks
            check_output_softmax(model)
            check_lrp_compat(model; verbose=verbose)
        end
        modified_layers = get_modified_layers(rules, model)
        return new{typeof(model),typeof(rules),typeof(modified_layers)}(
            model, rules, modified_layers
        )
    end
end

# Rules can be passed as vector and will be turned to ChainTuple
LRP(model, rules::AbstractVector; kwargs...) = LRP(model, ChainTuple(rules...); kwargs...)

# Convenience constructor without rules: use ZeroRule everywhere
LRP(model::Chain; kwargs...) = LRP(model, Composite(ZeroRule()); kwargs...)

# Construct Chain-/ParallelTuple of rules by applying composite
LRP(model::Chain, c::Composite; kwargs...) = LRP(model, lrp_rules(model, c); kwargs...)

#==========================#
# Call to the LRP analyzer #
#==========================#

function (lrp::LRP)(
    input::AbstractArray,
    ns::AbstractOutputSelector;
    layerwise_relevances=false,
    normalize_output=true,
    R=nothing,
)
    as = get_activations(lrp.model, input)    # compute activations aᵏ for all layers k
    Rs = similar.(as)
    if isnothing(R)                         # allocate relevances Rᵏ for all layers k
        mask_output_neuron!(Rs[end], as[end], ns; normalize_output=normalize_output) # compute relevance Rᴺ of output layer N
    else
        Rs[end] .= R # if there is a user-specified relevance for the last layer, use that instead
    end
    lrp_backward_pass!(Rs, as, lrp.rules, lrp.model, lrp.modified_layers)
    extras = layerwise_relevances ? (layerwise_relevances=Rs,) : nothing
    return Explanation(first(Rs), last(as), ns(last(as)), :LRP, :attribution, extras)
end

get_activations(model, input) = (input, Flux.activations(model, input)...)

function mask_output_neuron!(
    R_out, a_out, ns::AbstractOutputSelector; normalize_output=true
)
    fill!(R_out, 0)
    idx = ns(a_out)
    if normalize_output
        R_out[idx] .= 1
    else
        R_out[idx] .= a_out[idx]
    end
    return R_out
end

function lrp_backward_pass!(Rs, as, rules, layers, modified_layers)
    # Apply LRP rules in backward-pass, inplace-updating relevances `Rs[k]` = Rᵏ
    for k in length(layers):-1:1
        lrp!(Rs[k], rules[k], layers[k], modified_layers[k], as[k], Rs[k + 1])
    end
    return Rs
end

#===========================================#
# Special calls to Flux's "Dataflow layers" #
#===========================================#

function lrp!(Rᵏ, rules::ChainTuple, chain::Chain, modified_chain::ChainTuple, aᵏ, Rᵏ⁺¹)
    as = get_activations(chain, aᵏ)
    Rs = similar.(as)
    last(Rs) .= Rᵏ⁺¹

    lrp_backward_pass!(Rs, as, rules, chain, modified_chain)
    return Rᵏ .= first(Rs)
end

function lrp!(
    Rᵏ, rules::ParallelTuple, parallel::Parallel, modified_parallel::ParallelTuple, aᵏ, Rᵏ⁺¹
)
    # Re-compute contributions of parallel branches to output activation
    aᵏ⁺¹s = [layer(aᵏ) for layer in parallel.layers]

    # Distribute the relevance Rᵏ⁺¹ to the i-th branch of the parallel layer
    # according to the contribution aᵏ⁺¹ᵢ of branch i to the output activation aᵏ⁺¹:
    #   Rᵏ⁺¹s[i] = Rᵏ⁺¹ .* aᵏ⁺¹s[i] ./ aᵏ⁺¹ = c .* aᵏ⁺¹s[i]
    c = Rᵏ⁺¹ ./ stabilize_denom(sum(aᵏ⁺¹s))
    Rᵏ⁺¹s = [c .* aᵏ⁺¹ for aᵏ⁺¹ in aᵏ⁺¹s]

    # Compute individual input relevances Rᵏ for all branches of the parallel layer
    Rᵏs = [similar(aᵏ) for _ in parallel.layers]  # pre-allocate output
    for (Rᵏ, rule, layer, modified_layer, Rᵏ⁺¹) in
        zip(Rᵏs, rules, parallel.layers, modified_parallel, Rᵏ⁺¹s)
        # In-place update Rᵏᵢ and therefore Rᵏs
        lrp!(Rᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
    end
    # Sum up individual input relevances
    return Rᵏ .= sum(Rᵏs)
end

function lrp!(
    Rᵏ,
    rules::SkipConnectionTuple,
    sc::SkipConnection,
    modified_sc::SkipConnectionTuple,
    aᵏ,
    Rᵏ⁺¹,
)
    # Compute contributions of layer and skip connection to output activation.
    # For the skip connection, activations stay constant: aᵏ⁺¹_skip = aᵏ_skip = aᵏ
    aᵏ⁺¹_layers = sc.layers(aᵏ)
    c = Rᵏ⁺¹ ./ stabilize_denom(aᵏ⁺¹_layers + aᵏ) # using aᵏ = aᵏ⁺¹_skip

    # Distribute relevance accoring to contribution to output activation
    # For the skip connection, relevances stay constant: Rᵏ_skip = Rᵏ⁺¹_skip
    Rᵏ⁺¹_layers = c .* aᵏ⁺¹_layers
    Rᵏ_skip = c .* aᵏ  # same as Rᵏ⁺¹_skip = c .* aᵏ⁺¹_skip

    # Compute input relevance Rᵏ of layers
    Rᵏ_layers = similar(Rᵏ_skip) # pre-allocate output
    rules = ChainTuple(rules.vals)
    chain = Chain(sc.layers)
    modified_chain = ChainTuple(modified_sc.vals)
    lrp!(Rᵏ_layers, rules, chain, modified_chain, aᵏ, Rᵏ⁺¹_layers)

    # Sum up input relevances
    return Rᵏ .= Rᵏ_layers .+ Rᵏ_skip
end
