"""
    CRP(lrp_analyzer, layer, features)

Use Concept Relevance Propagation to explain the output of a neural network
with respect to specific features in a given layer.

# Arguments
- `lrp_analyzer::LRP`: LRP analyzer
- `layer::Int`: Index of layer after which the concept is located
- `features`: Concept / feature to explain.

Since layers are indexed positionally, CRP assumes a flat model:
apply [`flatten_model`](@ref) to the model triple before constructing
the [`LRP`](@ref) analyzer.

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

# CRP drives the same pure rule bodies as the Enzyme reverse pass,
# but with an explicit positional loop.
# This preserves its cost profile:
# the backward pass above the concept layer is shared between all features,
# only the pass below it runs once per feature —
# a single end-to-end reverse pass would multiply the above-layer cost
# by the number of features.

# Forward pass through a single wrapped node, returning `(zᵏ, y)`:
# `zᵏ` is the value `propagate` receives in the backward pass
# (the affine child's output for a split node)
# and `y` the node output fed to the next layer.
function node_forward(wrapped::SplitActivationNode, x, ps, st)
    z = first(apply(wrapped.affine, x, ps, st))
    y = first(apply(wrapped.activation, z, NamedTuple(), NamedTuple()))
    return z, y
end
function node_forward(wrapped, x, ps, st)
    y = first(apply(wrapped, x, ps, st))
    return y, y
end

# Propagate relevance through a single wrapped node.
# Leaves are propagated directly through the pure rule body;
# containers are differentiated as one unit through their wrapped form,
# driving the same custom rules as the full reverse pass.
function node_backward(wrapped::LayerWithRule, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    return propagate(wrapped.rule, wrapped.layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
end
# The activation child of a split node carries `PassRule`, which passes
# relevance through unchanged, so the backward pass reduces to the affine child.
function node_backward(wrapped::SplitActivationNode, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    return node_backward(wrapped.affine, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
end
node_backward(wrapped, aᵏ, zᵏ, ps, st, Rᵏ⁺¹) = seeded_pullback(wrapped, aᵏ, ps, st, Rᵏ⁺¹)

function call_analyzer(
    input::AbstractArray{T,N}, crp::CRP, ns::AbstractOutputSelector
) where {T,N}
    # Unpack internal LRP analyzer, matching layers positionally
    (; model, ps, st, normalize_output_relevance) = crp.lrp
    wrapped = values(wrap_rules(model, crp.lrp.rules).layers)
    pss = values(ps)
    sts = values(st)

    n_layers = length(wrapped)
    n_features = number_of_features(crp.features)
    batchsize = size(input, N)

    # Forward pass, collecting layer inputs aᵏ and pre-activations zᵏ
    as = Vector{Any}(undef, n_layers + 1)
    zs = Vector{Any}(undef, n_layers)
    as[1] = input
    for k in 1:n_layers
        z, y = node_forward(wrapped[k], as[k], pss[k], sts[k])
        zs[k] = z
        as[k + 1] = y
    end
    output = as[end]
    output_selection = ns(output)

    # Compute regular LRP backward pass until feature layer
    Rᵏ⁺¹ = relevance_seed(output, output_selection, normalize_output_relevance)
    for k in n_layers:-1:(crp.layer + 1)
        Rᵏ⁺¹ = node_backward(wrapped[k], as[k], zs[k], pss[k], sts[k], Rᵏ⁺¹)
    end
    R_original = Rᵏ⁺¹ # full relevance at the feature layer

    # Compute neuron indices based on features
    feature_indices = crp.features(R_original)

    # Allocate array for returned relevance, adding features to batch dimension
    R_return = similar(input, size(input)[1:(N - 1)]..., batchsize * n_features)
    colons = ntuple(Returns(:), N - 1)

    for (i, feature) in enumerate(feature_indices)
        # Mask feature neurons, keeping original relevance at feature neurons
        R_masked = zero(R_original)
        for idx in feature
            R_masked[idx] .= R_original[idx]
        end

        # Continue LRP backward pass below the feature layer
        Rᵏ = R_masked
        for k in (crp.layer):-1:1
            Rᵏ = node_backward(wrapped[k], as[k], zs[k], pss[k], sts[k], Rᵏ)
        end

        # Write relevance into a slice of R_return
        start = batchsize * (i - 1) + 1
        stop = batchsize * i
        view(R_return, colons..., start:stop) .= Rᵏ
    end
    return Explanation(
        R_return, input, output, output_selection, :CRP, :attribution, nothing
    )
end
