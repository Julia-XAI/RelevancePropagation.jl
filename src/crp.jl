"""
    CRP(lrp_analyzer, layer, features)

Use Concept Relevance Propagation to explain the output of a neural network
with respect to specific features in a given layer.

# Arguments
- `lrp_analyzer::LRP`: LRP analyzer
- `layer::Int`: Index of layer after which the concept is located
- `features`: Concept / feature to explain.

See also [`TopNFeatures`](@ref) and [`IndexedFeatures`](@ref).

# References
[1] R. Achtibat et al., From attribution maps to human-understandable explanations
    through Concept Relevance Propagation
"""
struct CRP{L<:LRP,F<:AbstractFeatureSelector} <: AbstractXAIMethod
    lrp::L
    layer::Int
    features::F

    function CRP(lrp::LRP, layer::Int, features::AbstractFeatureSelector)
        n = length(lrp.model)
        layer ≥ n &&
            throw(ArgumentError("Layer index should be smaller than model length $n"))
        return new{typeof(lrp),typeof(features)}(lrp, layer, features)
    end
end

#======================#
# Call to CRP analyzer #
#======================#

function (crp::CRP)(input::AbstractArray{T,N}, ns::AbstractNeuronSelector) where {T,N}
    rules = crp.lrp.rules
    layers = crp.lrp.model.layers
    modified_layers = crp.lrp.modified_layers

    n_layers = length(layers)
    n_features = number_of_features(crp.features)
    batchsize = size(input, N)

    # Forward pass
    as = get_activations(crp.lrp.model, input) # compute activations aᵏ for all layers k
    Rs = similar.(as)                          # allocate relevances Rᵏ for all layers k
    mask_output_neuron!(Rs[end], as[end], ns)  # compute relevance Rᴺ of output layer N

    # Allocate array for returned relevance, adding features to batch dimension
    R_return = similar(input, size(input)[1:(end - 1)]..., batchsize * n_features)
    colons = ntuple(Returns(:), N - 1)

    # Compute regular LRP backward pass until feature layer
    for k in n_layers:-1:(crp.layer + 1)
        lrp!(Rs[k], rules[k], layers[k], modified_layers[k], as[k], Rs[k + 1])
    end

    # Save full relevance at feature layer before masking
    R_feature = Rs[crp.layer + 1]
    R_original = deepcopy(R_feature)

    # Compute neuron indices based on features
    feature_indices = crp.features(R_original)

    # Mask feature neurons...
    fill!(R_feature, 0)

    for (i, feature) in enumerate(feature_indices)
        # ...keeping original relevance at feature neurons
        for idx in feature
            R_feature[idx] .= R_original[idx]
        end

        # Continue LRP backward pass
        for k in (crp.layer):-1:1
            lrp!(Rs[k], rules[k], layers[k], modified_layers[k], as[k], Rs[k + 1])
        end

        # Write relevance into a slice of R_return
        start = batchsize * (i - 1) + 1
        stop = batchsize * i
        view(R_return, colons..., start:stop) .= first(Rs)

        # Reset feature neurons for masking in next iteration
        if i < n_features
            for idx in feature
                R_feature[idx] .= 0
            end
        end
    end
    return Explanation(R_return, last(as), ns(last(as)), :CRP, :attribution, nothing)
end
