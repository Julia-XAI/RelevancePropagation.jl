module LRP_CONFIG
using RelevancePropagation
using RelevancePropagation: LRPSupportedLayer, LRPSupportedActivation

"""
    LRP_CONFIG.supports_layer(layer)

Check whether LRP can be used on a layer or a Chain.
To extend LRP to your own layers, define:
```julia
LRP_CONFIG.supports_layer(::MyLayer) = true          # for structs
LRP_CONFIG.supports_layer(::typeof(mylayer)) = true  # for functions
```
"""
supports_layer(l) = false
supports_layer(::LRPSupportedLayer) = true
"""
    LRP_CONFIG.supports_activation(σ)

Check whether LRP can be used on a given activation function.
To extend LRP to your own activation functions, define:
```julia
LRP_CONFIG.supports_activation(::typeof(myactivation)) = true  # for functions
LRP_CONFIG.supports_activation(::MyActivation) = true          # for structs
```
"""
supports_activation(fn) = false
supports_activation(::LRPSupportedActivation) = true
end # LRP_CONFIG module

lrp_check_layer(l) = lrp_check_layer_type(l) && lrp_check_activation(l)

lrp_check_layer_type(l) = LRP_CONFIG.supports_layer(l)
# Lux wraps bare functions used as layers in `WrappedFunction`;
# users register the wrapped function itself via `LRP_CONFIG.supports_layer`.
lrp_check_layer_type(l::WrappedFunction) = LRP_CONFIG.supports_layer(l.func)

function lrp_check_activation(layer)
    f = activation_fn(layer)
    !isnothing(f) && return LRP_CONFIG.supports_activation(f)
    return true
end

"""
    check_lrp_compat(model; verbose=true)

Check whether LRP can be used on the model.
"""
function check_lrp_compat(model::Chain; verbose=true)
    passed_checks = chainall(lrp_check_layer, model)
    if !passed_checks
        if verbose
            print_lrp_model_check(stdout, model)
            println()
            display(_MD_CHECK_FAILED)
            println()
        end
        error("Unknown layer or activation function found in model")
    end
    return true
end

function print_lrp_model_check(io::IO, model::DataflowLayer, indent::Int=0)
    println(io, "  "^indent, nameof(typeof(model)), "(")
    for layer in children_layers(model)
        print_lrp_model_check(io, layer, indent + 1)
    end
    println(io, "  "^indent, indent == 0 ? ")" : "),")
end

function print_lrp_model_check(io::IO, layer, indent::Int=0)
    print(io, "  "^indent, layer)
    print(io, " => ")
    print_layer_check(io, layer)
    println(io, ",")
end

function print_layer_check(io, l)
    layer_failed = !lrp_check_layer_type(l)
    activ_failed = !lrp_check_activation(l)
    activ = activation_fn(l)

    if layer_failed && activ_failed
        return printstyled(
            io,
            "unsupported or unknown activation function $activ and layer type";
            color=:red,
        )
    elseif activ_failed
        return printstyled(
            io, "unsupported or unknown activation function $activ"; color=:red
        )
    elseif layer_failed
        return printstyled(io, "unknown layer type"; color=:red)
    end
    return printstyled(io, "supported"; color=:green)
end

_MD_CHECK_FAILED = md"""# LRP model check failed

    Found unknown layer types or activation functions that are not supported
    by RelevancePropagation.jl yet.

    LRP assumes that the model is a deep rectifier network
    that only contains ReLU-like activation functions.

    If you think the missing layer should be supported by default,
    **please [submit an issue](https://github.com/Julia-XAI/RelevancePropagation.jl/issues)**.

    ## Using custom layers

    If you implemented custom layers, register them via
    ```julia
    LRP_CONFIG.supports_layer(::MyLayer) = true          # for structs
    LRP_CONFIG.supports_layer(::typeof(mylayer)) = true  # for functions
    ```
    The default fallback for this layer will use Automatic Differentiation
    according to *"Layer-Wise Relevance Propagation: An Overview"*.

    ## Using custom activation functions

    If you use custom ReLU-like activation functions, register them via
    ```julia
    LRP_CONFIG.supports_activation(::typeof(myfunction)) = true  # for functions
    LRP_CONFIG.supports_activation(::MyActivation) = true        # for structs
    ```

    ## Skip model checks

    Model checks can be skipped at your own risk by setting
    the `LRP` keyword argument `skip_checks=true`.
    """

#=========================#
# Strip output activation #
#=========================#

"""
  check_output_softmax(model)

Check whether model has softmax activation on output.
Return the model if it doesn't, throw error otherwise.
"""
function check_output_softmax(model::Chain)
    if has_output_softmax(model)
        throw(ArgumentError("""Model contains softmax activation function on output.
        Call `strip_softmax` on your model."""))
    end
    return model
end

has_output_softmax(model::Chain) = has_output_softmax(last_element(model))
has_output_softmax(x) = is_softmax(x) || is_softmax(activation_fn(x))

is_softmax(x) = x isa SoftmaxActivation
is_softmax(l::WrappedFunction) = is_softmax(l.func)
