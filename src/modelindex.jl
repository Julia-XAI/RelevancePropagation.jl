"""
   ModelIndex(inds...)

Index tuple used to enumerate nested layers in a Flux model.
"""
struct ModelIndex{N}
    inds::NTuple{N,Int}
end
ModelIndex(inds...) = ModelIndex(tuple(inds...))
append(idx::ModelIndex, i) = ModelIndex(idx.inds..., i)

Base.show(io::IO, mi::ModelIndex) = Base.show(io, mi.inds)
Base.show(io::IO, mi::ModelIndex{1}) = Base.show(io, only(mi.inds))

Base.:(==)(a::ModelIndex, b::ModelIndex) = a.inds == b.inds
@forward ModelIndex.inds Base.getindex,
Base.length,
Base.first,
Base.last,
Base.iterate,
Base.lastindex,
Base.keys,
Base.firstindex,
Base.similar

# in(ModelIndex(1, 2),    ModelIndex(1))       -> true
# in(ModelIndex(1, 2),    ModelIndex(2))       -> false
# in(ModelIndex(1, 2),    ModelIndex(1, 2))    -> true
# in(ModelIndex(1, 2, 3), ModelIndex(1, 2))    -> true
# in(ModelIndex(1, 2),    ModelIndex(1, 2, 3)) -> false
function Base.in(a::ModelIndex, b::ModelIndex)
    length(a.inds) < length(b.inds) && return false
    for i in eachindex(b.inds)
        a.inds[i] != b.inds[i] && return false
    end
    return true
end

"""
    chainindices(model)

Sequentially enumerate all layers in a Flux model.
Nested `Chain` and `Parallel` layers will result in tuples of indices.

# Example:
```julia-repl
julia> d = Dense(2, 2);

julia> model = Chain(d, Parallel(+, d, d, Chain(d, d)), d);

julia> chainindices(model)
ChainTuple(
  (1,),
  ParallelTuple(
    (2, 1),
    (2, 2),
    ChainTuple(
      (2, 3, 1),
      (2, 3, 2),
    ),
  ),
  (3,),
)
```
"""
chainindices(model) = chainindices(model, ModelIndex())
function chainindices(x, idx::ModelIndex)
    if isleaf(x)
        return idx
    else
        T = constructor(x)
        idxs = append.(Ref(idx), 1:length(children(x)))
        return T(chainindices.(children(x), idxs)...)
    end
end

"""
    show_layer_indices(model)

Print layer indices of Flux models.
This is primarily a utility to help define [`LayerMap`](@ref) primitives.
"""
show_layer_indices(model) = chainindices(model)
