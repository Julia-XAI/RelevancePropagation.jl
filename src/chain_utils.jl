#============================#
# ChainTuple & ParallelTuple #
#============================#

# To support map and zip on Flux Chains containing `Chain`, `Parallel` and `SkipConnection` layers,
# we need a flexible, general purpose container, e.g. a Tuple.
# We opt to introduce `ChainTuple`,` `ParallelTuple` and `SkipConnectionTuple` to avoid type piracy.

"""
    ChainTuple(xs)

Thin wrapper around `Tuple` for use with Flux.jl models.

Combining [`ChainTuple`](@ref), [`ParallelTuple`](@ref) and [`SkipConnectionTuple`](@ref),
data `xs` can be stored while preserving the structure of a Flux model
without risking type piracy.
"""
struct ChainTuple{T<:Tuple}
    vals::T
end

"""
    ParallelTuple(xs)

Thin wrapper around `Tuple` for use with Flux.jl models.

Combining [`ChainTuple`](@ref), [`ParallelTuple`](@ref) and [`SkipConnectionTuple`](@ref),
data `xs` can be stored while preserving the structure of a Flux model
without risking type piracy.
"""
struct ParallelTuple{T<:Tuple}
    vals::T
end

"""
    SkipConnectionTuple(xs)

Thin wrapper around `Tuple` for use with Flux.jl models.

Combining [`ChainTuple`](@ref), [`ParallelTuple`](@ref) and [`SkipConnectionTuple`](@ref),
data `xs` can be stored while preserving the structure of a Flux model
without risking type piracy.
"""
struct SkipConnectionTuple{T}
    vals::T
end
SkipConnectionTuple(xs::Tuple) = SkipConnectionTuple(ChainTuple(xs))

for T in (:ChainTuple, :ParallelTuple, :SkipConnectionTuple)
    name = string(T)

    @eval begin
        ($T)(xs...) = ($T)(xs)

        @forward $T.vals Base.getindex,
        Base.length,
        Base.first,
        Base.last,
        Base.iterate,
        Base.lastindex,
        Base.keys,
        Base.firstindex,
        Base.:(==)
        Base.similar

        # Containers are equivalent if fields are equivalent
        Base.:(==)(a::$T, b::$T) = a.vals == b.vals

        # Print vals
        Base.show(io::IO, m::MIME"text/plain", t::$T) = print_vals(io, t)

        function print_vals(io::IO, t::$T, indent::Int=0)
            println(io, " "^indent, $name, "(")
            for x in t
                print_vals(io, x, indent + 2)
            end
            println(io, " "^indent, ")", ifelse(indent != 0, ",", ""))
        end
    end # eval
end
print_vals(io::IO, x, indent::Int=0) = println(io, " "^indent, x, ",")

#=====================#
# chainmap & chainzip #
#=====================#

# The following implementation of map and zip on Chains and Parallel layers
# are strongly inspired by StructWalk.jl's postwalk function.

isleaf(c::Chain)               = false
isleaf(c::ChainTuple)          = false
isleaf(p::Parallel)            = false
isleaf(p::ParallelTuple)       = false
isleaf(s::SkipConnection)      = false
isleaf(s::SkipConnectionTuple) = false
isleaf(x)                      = true

constructor(::Chain)               = ChainTuple
constructor(::ChainTuple)          = ChainTuple
constructor(::Parallel)            = ParallelTuple
constructor(::ParallelTuple)       = ParallelTuple
constructor(::SkipConnection)      = SkipConnectionTuple
constructor(::SkipConnectionTuple) = SkipConnectionTuple

children(c::Chain)               = c.layers
children(c::ChainTuple)          = c.vals
children(p::Parallel)            = p.layers
children(p::ParallelTuple)       = p.vals
children(s::SkipConnection)      = (s.layers,)
children(s::SkipConnectionTuple) = (s.vals,)

"""
    chainmap(f, x)

`map` for Flux models. Applies the function `f` to nested structures of `Chain`s
and `Parallel` layers.
Returns a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.

Can also be applied to nested structures of `ChainTuple` and `ParallelTuple`.

See also [`chainzip`](@ref).
"""
function chainmap(f, x)
    if isleaf(x)
        return f(x)
    else
        T = constructor(x)
        vals = chainmap.(f, children(x))
        return T(vals...)
    end
end

"""
    chainall(f, model)

Determines whether `f` returns `true` for all elements of a Flux `Chain` `x`.
Can also be applied to nested structures of `ChainTuple` and `ParallelTuple`.
"""
function chainall(f, x)
    isleaf(x) && return f(x)
    return all(chainall.(f, children(x)))
end

"""
    chainzip(f, x, y)
    chainzip(f, xs...)

`zip` for Flux models. Applies the function `f` to nested structures of `Chain`s
and `Parallel` layers. Assumes that arguments have the same structure.
Returns a [`ChainTuple`](@ref) or [`ParallelTuple`](@ref) matching the model structure.

Can also be applied to nested structures of `ChainTuple` and `ParallelTuple`.

See also [`chainmap`](@ref).
"""
function chainzip(f, xs...)
    if all(isleaf, xs)
        return f(xs...)
    else
        Ts = constructor.(xs)
        allequal(Ts) || error("Cannot chainzip arguments of different container types $Ts.")

        cs = children.(xs)
        lens = length.(cs)
        allequal(lens) ||
            error("Cannot chainzip arguments $xs of different lengths: $lens.")

        T = first(Ts)
        vals = chainzip.(f, cs...)
        return T(vals...)
    end
end

#===============#
# Flatten model #
#===============#

"""
    flatten_model(model)

Flatten a Flux `Chain` containing `Chain`s.
"""
flatten_model(x) = chainflatten(x)

"""
    chainflatten(chain)

Flatten a Flux `Chain` containing `Chain`s. Also works with `ChainTuple`s.
"""
chainflatten(c::Chain) = Chain(chainflatten(c.layers)...)
chainflatten(c::ChainTuple) = ChainTuple(chainflatten(c.vals)...)

chainflatten(p::Parallel)      = Parallel(p.connection, chainflatten.(p.layers))
chainflatten(p::ParallelTuple) = ParallelTuple(chainflatten.(p.vals))

chainflatten(s::SkipConnection)      = SkipConnection(chainflatten(s.layers), s.connection)
chainflatten(s::SkipConnectionTuple) = SkipConnectionTuple(chainflatten(s.vals))

chainflatten(x) = x

# Splat Chains, ChainTuples and Tuples using `append!`
function chainflatten(layers::Tuple)
    out = []
    for l in layers
        flat = chainflatten(l)
        if flat isa Chain
            append!(out, flat.layers)
        elseif flat isa ChainTuple
            append!(out, flat.vals)
        elseif flat isa Tuple
            append!(out, flat)
        else
            push!(out, flat)
        end
    end
    return out
end

#=========================#
# Strip output activation #
#=========================#
"""
    first_element(model)

Returns last layer of a Flux `Chain` or `ChainTuple`.
"""
first_element(c::Union{Chain,ChainTuple}) = first_element(c[1])
first_element(layer) = layer

"""
    last_element(model)

Returns last layer of a Flux `Chain` or `ChainTuple`.
"""
last_element(c::Union{Chain,ChainTuple}) = last_element(c[end])
last_element(layer) = layer

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

"""
    strip_softmax(model)
    strip_softmax(layer)

Remove softmax activation on layer or model if it exists.
"""
strip_softmax(l) = copy_layer(l, l.weight, l.bias; σ=identity)
strip_softmax(::SoftmaxActivation) = identity

function strip_softmax(model::Chain)
    output_layer = last_element(model)
    !has_output_softmax(output_layer) && return model

    function _strip_softmax(layer)
        layer != output_layer && return layer
        return strip_softmax(layer)
    end
    _strip_softmax(c::Chain) = Chain(_strip_softmax.(c.layers)...)
    _strip_softmax(p::Parallel) = p # p.connection can't be softmax
    _strip_softmax(p::SkipConnection) = p # p.connection can't be softmax
    return _strip_softmax(model)
end
