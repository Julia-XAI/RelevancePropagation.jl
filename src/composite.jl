#===========#
# Composite #
#===========#

# A Composite is a container of primitives, which are sequentially applied

struct Composite{T<:Union{Tuple,AbstractVector}}
    primitives::T
end
Composite(primitives...) = Composite(primitives)
Composite(rule::AbstractLRPRule, prims...) = Composite((GlobalMap(rule), prims...))

#================#
# Layer indexing #
#================#

# Layers are addressed by `Functors.KeyPath`s mirroring the keys of the
# model's `ps`/`st` NamedTuples, e.g. `KeyPath(:layer_2, :layer_3)`.

"""
    layer_indices(model)

Enumerate all layers in a Lux model, mirroring the model structure as nested
`NamedTuple`s with `Functors.KeyPath` leaves that address each layer like the
model's `ps` and `st`.

# Example:
```julia-repl
julia> d = Dense(2 => 2);

julia> model = Chain(d, Parallel(+, d, d, Chain(d, d)), d);

julia> layer_indices(model)
(layer_1 = KeyPath(:layer_1,),
 layer_2 = (layer_1 = KeyPath(:layer_2, :layer_1),
            layer_2 = KeyPath(:layer_2, :layer_2),
            layer_3 = (layer_1 = KeyPath(:layer_2, :layer_3, :layer_1),
                       layer_2 = KeyPath(:layer_2, :layer_3, :layer_2))),
 layer_3 = KeyPath(:layer_3,))
```
"""
layer_indices(model) = layer_indices(model, KeyPath())
function layer_indices(model::Union{Chain,Parallel}, path::KeyPath)
    layers = model.layers
    ks = keys(layers)
    return NamedTuple{ks}(map(k -> layer_indices(getproperty(layers, k), KeyPath(path, k)), ks))
end
# `SkipConnection` is an `AbstractLuxWrapperLayer`:
# its `ps`/`st` pass through to the wrapped layer directly.
layer_indices(model::SkipConnection, path::KeyPath) = layer_indices(model.layers, path)
layer_indices(layer, path::KeyPath) = path

"""
    show_layer_indices(model)

Print layer indices of Lux models.
This is primarily a utility to help define [`LayerMap`](@ref) primitives.
"""
show_layer_indices(model) = layer_indices(model)

# Prefix matching: `a` is at or nested below `b`.
# keypath_in(KeyPath(:layer_1, :layer_2), KeyPath(:layer_1))            -> true
# keypath_in(KeyPath(:layer_1, :layer_2), KeyPath(:layer_2))            -> false
# keypath_in(KeyPath(:layer_1, :layer_2), KeyPath(:layer_1, :layer_2))  -> true
# keypath_in(KeyPath(:layer_1), KeyPath(:layer_1, :layer_2))            -> false
function keypath_in(a::KeyPath, b::KeyPath)
    length(a) < length(b) && return false
    return all(a[i] == b[i] for i in 1:length(b))
end

first_leaf(nt::NamedTuple) = first_leaf(first(values(nt)))
first_leaf(x) = x
last_leaf(nt::NamedTuple) = last_leaf(last(values(nt)))
last_leaf(x) = x

# Zip a Lux model tree with a mirroring NamedTuple, applying `f` to the leaves.
function zip_layers(f, model::Union{Chain,Parallel}, nt::NamedTuple)
    layers = model.layers
    return NamedTuple{keys(layers)}(
        map((l, x) -> zip_layers(f, l, x), values(layers), values(nt))
    )
end
zip_layers(f, model::SkipConnection, x) = zip_layers(f, model.layers, x)
zip_layers(f, layer, x) = f(layer, x)

#=================#
# Rule primitives #
#=================#

abstract type AbstractCompositePrimitive end
abstract type AbstractCompositeMap <: AbstractCompositePrimitive end

"""
    GlobalMap(rule)

Composite primitive that maps an LRP-rule to all layers in the model.

See [`Composite`](@ref) for an example.
"""
struct GlobalMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    rule::R
end

"""
    LayerMap(index, rule)

Composite primitive that maps an LRP-rule to the layer in the model addressed
by `index`, a `Functors.KeyPath` mirroring the keys of the model's `ps`/`st`
NamedTuples, e.g. `KeyPath(:layer_2, :layer_3)`. All layers nested under the
addressed path are matched as well.

For convenience, an integer or tuple of integers can be passed instead,
addressing Lux's default `layer_i` naming:
`LayerMap(2, rule) == LayerMap(KeyPath(:layer_2), rule)`.

See [`show_layer_indices`](@ref) to print layer indices and [`Composite`](@ref) for an example.
"""
struct LayerMap{K<:KeyPath,R<:AbstractLRPRule} <: AbstractCompositeMap
    index::K
    rule::R
end
LayerMap(index::Union{Integer,Tuple}, rule::AbstractLRPRule) = LayerMap(keypath(index), rule)

keypath(i::Integer) = KeyPath(Symbol(:layer_, i))
keypath(inds::Tuple) = KeyPath(map(i -> Symbol(:layer_, i), inds)...)

"""
    RangeMap(range, rule)

Composite primitive that maps an LRP-rule to the specified positional `range`
of layers in the model.

See [`Composite`](@ref) for an example.
"""
struct RangeMap{T<:AbstractRange,R<:AbstractLRPRule} <: AbstractCompositeMap
    range::T
    rule::R
end

"""
    FirstLayerMap(rule)

Composite primitive that maps an LRP-rule to the first layer in the model.

See [`Composite`](@ref) for an example.
"""
struct FirstLayerMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    rule::R
end

"""
    LastLayerMap(rule)

Composite primitive that maps an LRP-rule to the last layer in the model.

See [`Composite`](@ref) for an example.
"""
struct LastLayerMap{R<:AbstractLRPRule} <: AbstractCompositeMap
    rule::R
end

#=====================#
# TypeMap primitives #
#=====================#

abstract type AbstractCompositeTypeMap <: AbstractCompositePrimitive end
const TypeMapPair = Pair{<:Type,<:AbstractLRPRule}

"""
    GlobalTypeMap(map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct GlobalTypeMap{T<:AbstractVector{<:TypeMapPair}} <: AbstractCompositeTypeMap
    map::T
end

"""
    RangeTypeMap(range, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the specified `range` of layers in the model.

See [`Composite`](@ref) for an example.
"""
struct RangeTypeMap{R<:AbstractRange,T<:AbstractVector{<:TypeMapPair}} <:
       AbstractCompositeTypeMap
    range::R
    map::T
end

"""
    FirstNTypeMap(n, map)

Composite primitive that maps layer types to LRP rules based on a list of
type-rule-pairs `map` within the first `n` layers in the model.

See [`Composite`](@ref) for an example.
"""
struct FirstNTypeMap{T<:AbstractVector{<:TypeMapPair}} <: AbstractCompositeTypeMap
    n::Int
    map::T
end

"""
    FirstLayerTypeMap(map)

Composite primitive that maps the type of the first layer of the model to LRP rules
based on a list of type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct FirstLayerTypeMap{T<:AbstractVector{<:TypeMapPair}} <: AbstractCompositeTypeMap
    map::T
end

"""
    LastLayerTypeMap(map)

Composite primitive that maps the type of the last layer of the model to LRP rules
based on a list of type-rule-pairs `map`.

See [`Composite`](@ref) for an example.
"""
struct LastLayerTypeMap{T<:AbstractVector{<:TypeMapPair}} <: AbstractCompositeTypeMap
    map::T
end

# Convenience constructors
GlobalTypeMap(ps::Vararg{TypeMapPair})     = GlobalTypeMap([ps...])
RangeTypeMap(r, ps::Vararg{TypeMapPair})   = RangeTypeMap(r, [ps...])
FirstNTypeMap(n, ps::Vararg{TypeMapPair})  = FirstNTypeMap(n, [ps...])
FirstLayerTypeMap(ps::Vararg{TypeMapPair}) = FirstLayerTypeMap([ps...])
LastLayerTypeMap(ps::Vararg{TypeMapPair})  = LastLayerTypeMap([ps...])

#=====================#
# LRP-rule assignment #
#=====================#

function get_type_rule(layer, map)
    for (T, rule) in map
        if layer isa T
            return rule
        end
    end
    return nothing
end
# Lux wraps bare functions used as layers in `WrappedFunction`;
# type maps match either the wrapper or the wrapped function itself.
function get_type_rule(layer::WrappedFunction, map)
    for (T, rule) in map
        if layer isa T || layer.func isa T
            return rule
        end
    end
    return nothing
end

"""
    lrp_rules(model, composite)

Apply a composite to obtain LRP-rules for a given Lux model,
returned as a `NamedTuple` mirroring the model's `ps`/`st` structure.
"""
function lrp_rules(model::Chain, c::Composite)
    indices = layer_indices(model)
    idx_first = first_leaf(indices)
    idx_last = last_leaf(indices)
    top_keys = keys(model.layers)
    # Positional primitives (RangeMap, RangeTypeMap, FirstNTypeMap) refer to
    # the top-level position of a layer in the model; nested layers inherit
    # the position of their top-level parent.
    position(idx::KeyPath) = findfirst(==(idx[1]), top_keys)

    get_rule(r::LayerMap, _, idx) = keypath_in(idx, r.index) ? r.rule : nothing
    get_rule(r::GlobalMap, _, _idx) = r.rule
    get_rule(r::RangeMap, _, idx) = position(idx) ∈ r.range ? r.rule : nothing
    get_rule(r::FirstLayerMap, _, idx) = idx == idx_first ? r.rule : nothing
    get_rule(r::LastLayerMap, _, idx) = idx == idx_last ? r.rule : nothing

    function get_rule(r::GlobalTypeMap, layer, _idx)
        return get_type_rule(layer, r.map)
    end
    function get_rule(r::RangeTypeMap, layer, idx)
        return position(idx) ∈ r.range ? get_type_rule(layer, r.map) : nothing
    end
    function get_rule(r::FirstLayerTypeMap, layer, idx)
        return idx == idx_first ? get_type_rule(layer, r.map) : nothing
    end
    function get_rule(r::LastLayerTypeMap, layer, idx)
        return idx == idx_last ? get_type_rule(layer, r.map) : nothing
    end
    function get_rule(r::FirstNTypeMap, layer, idx)
        return position(idx) ∈ 1:(r.n) ? get_type_rule(layer, r.map) : nothing
    end

    # The last rule returned from a composite primitive is assigned to the layer.
    # This is implemented by returning the first rule in reverse order:
    function match_rule(layer, idx)
        for primitive in reverse(c.primitives)
            rule = get_rule(primitive, layer, idx)
            !isnothing(rule) && return rule
        end
        return ZeroRule() # else if no assignment was found, return default rule
    end
    return zip_layers(match_rule, model, indices) # construct NamedTuple of rules
end

"""
    Composite(primitives...)
    Composite(default_rule, primitives...)

Automatically contructs a list of LRP-rules by sequentially applying composite primitives.

# Primitives
To apply a single rule, use:
* [`LayerMap`](@ref) to apply a rule to a layer addressed by its `KeyPath`
* [`GlobalMap`](@ref) to apply a rule to all layers
* [`RangeMap`](@ref) to apply a rule to a positional range of layers
* [`FirstLayerMap`](@ref) to apply a rule to the first layer
* [`LastLayerMap`](@ref) to apply a rule to the last layer

To apply a set of rules to layers based on their type, use:
* [`GlobalTypeMap`](@ref) to apply a dictionary that maps layer types to LRP-rules
* [`RangeTypeMap`](@ref) for a `TypeMap` on generalized ranges
* [`FirstLayerTypeMap`](@ref) for a `TypeMap` on the first layer of a model
* [`LastLayerTypeMap`](@ref) for a `TypeMap` on the last layer
* [`FirstNTypeMap`](@ref) for a `TypeMap` on the first `n` layers

# Example
Using a VGG11 model:
```julia-repl
julia> composite = Composite(
           GlobalTypeMap(
               ConvLayer => AlphaBetaRule(),
               Dense => EpsilonRule(),
               PoolingLayer => EpsilonRule(),
               DropoutLayer => PassRule(),
               ReshapingLayer => PassRule(),
           ),
           FirstNTypeMap(7, Conv => FlatRule()),
       );

julia> analyzer = LRP(model, ps, st, composite)
LRP(
  Conv((3, 3), 3 => 64, relu, pad=1)    => FlatRule(),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  Conv((3, 3), 64 => 128, relu, pad=1)  => FlatRule(),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  Conv((3, 3), 128 => 256, relu, pad=1) => FlatRule(),
  Conv((3, 3), 256 => 256, relu, pad=1) => FlatRule(),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  Conv((3, 3), 256 => 512, relu, pad=1) => AlphaBetaRule{Float32}(2.0f0, 1.0f0),
  Conv((3, 3), 512 => 512, relu, pad=1) => AlphaBetaRule{Float32}(2.0f0, 1.0f0),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  Conv((3, 3), 512 => 512, relu, pad=1) => AlphaBetaRule{Float32}(2.0f0, 1.0f0),
  Conv((3, 3), 512 => 512, relu, pad=1) => AlphaBetaRule{Float32}(2.0f0, 1.0f0),
  MaxPool((2, 2))                       => EpsilonRule{Float32}(1.0f-6),
  FlattenLayer()                        => PassRule(),
  Dense(25088 => 4096, relu)            => EpsilonRule{Float32}(1.0f-6),
  Dropout(0.5)                          => PassRule(),
  Dense(4096 => 4096, relu)             => EpsilonRule{Float32}(1.0f-6),
  Dropout(0.5)                          => PassRule(),
  Dense(4096 => 1000)                   => EpsilonRule{Float32}(1.0f-6),
)
```
"""
Composite
