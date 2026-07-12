const COLOR_COMMENT = :light_black
const COLOR_ARROW   = :light_black
const COLOR_RULE    = :yellow
const COLOR_TYPE    = :light_blue

typename(x) = string(nameof(typeof(x)))

#==============#
# LRP analyzer #
#==============#

layer_name(io::IO, l) = string(sprint(show, l; context=io))

# Pad the printed names of leaf layers within a container to align the rules
function get_name_padding(io::IO, layers)
    leaves = [layer_name(io, l) for l in layers if !(l isa DataflowLayer)]
    isempty(leaves) && return 0
    return maximum(length, leaves)
end

function Base.show(io::IO, m::MIME"text/plain", lrp::LRP)
    npad = get_name_padding(io, values(lrp.model.layers))
    println(io, "LRP", "(")
    for (layer, rule) in zip(values(lrp.model.layers), values(lrp.rules))
        print_rule(io, layer, rule, 1, npad)
    end
    print(io, ")")
end

# Rules for `Chain` and `Parallel` are nested NamedTuples mirroring the model
function print_rule(
    io::IO, container::Union{Chain,Parallel}, rules::NamedTuple, indent::Int=0, npad::Int=0
)
    println(io, "  "^indent, typename(container), "(")
    npad = get_name_padding(io, values(container.layers))
    for (layer, rule) in zip(values(container.layers), values(rules))
        print_rule(io, layer, rule, indent + 1, npad)
    end
    println(io, "  "^indent, "),")
end

# `SkipConnection` is transparent in `rules` (like in `ps`/`st`)
function print_rule(io::IO, sc::SkipConnection, rules, indent::Int=0, npad::Int=0)
    println(io, "  "^indent, typename(sc), "(")
    npad = get_name_padding(io, (sc.layers,))
    print_rule(io, sc.layers, rules, indent + 1, npad)
    println(io, "  "^indent, "),")
end

function print_rule(io::IO, layer, rule, indent::Int=0, npad::Int=0)
    print(io, "  "^indent, rpad(layer_name(io, layer), npad))
    printstyled(io, " => "; color=COLOR_ARROW)
    printstyled(io, rule; color=COLOR_RULE)
    println(io, ",")
end

#===========#
# Composite #
#===========#

_range_string(r::LayerMap)         = "layer $(r.index)"
_range_string(::GlobalMap)         = "all layers"
_range_string(r::RangeMap)         = "layers $(r.range)"
_range_string(::FirstLayerMap)     = "first layer"
_range_string(::LastLayerMap)      = "last layer"
_range_string(r::GlobalTypeMap)    = "all layers"
_range_string(r::RangeTypeMap)     = "layers $(r.range)"
_range_string(::FirstLayerTypeMap) = "first layer"
_range_string(::LastLayerTypeMap)  = "last layer"
_range_string(r::FirstNTypeMap)    = "layers $(1:r.n)"

function Base.show(io::IO, m::MIME"text/plain", c::Composite, indent::Int=0)
    println(io, "Composite", "(")
    for p in c.primitives
        _show_primitive(io, p, indent + 1)
    end
    print(io, ")")
end

function _show_primitive(io::IO, r::AbstractCompositeMap, indent::Int=0)
    print(io, "  "^indent, typename(r), "( ")
    printstyled(io, "# ", _range_string(r); color=COLOR_COMMENT)
    println(io)
    printstyled(io, "  "^(indent + 1), r.rule; color=COLOR_RULE)
    println(io)
    println(io, "  "^indent, "),")
end

function _show_primitive(io::IO, r::AbstractCompositeTypeMap, indent::Int=0)
    npad = get_type_padding(io, r.map)
    print(io, "  "^indent, typename(r), "(  ")
    printstyled(io, "# ", _range_string(r); color=COLOR_COMMENT)
    println(io)
    for (type, rule) in r.map
        _print_type_rule(io, type, rule, indent + 1, npad)
    end
    println(io, " "^(indent), "),")
end

function _print_type_rule(io::IO, type::Type, rule, indent::Int=0, npad=0)
    printstyled(io, "  "^indent, rpad(type, npad); color=COLOR_TYPE)
    print(io, " => ")
    printstyled(io, rule; color=COLOR_RULE)
    println(io, ",")
end
function _print_type_rule(io::IO, types::Union, rule, indent::Int=0, npad=0)
    for t in types_in_union(types)
        _print_type_rule(io, t, rule, indent, npad)
    end
end

function get_type_padding(io::IO, map::AbstractVector{<:TypeMapPair})
    isempty(map) && return 0
    types = first.(map)
    return maximum(max_type_name_length.((io,), types))
end

max_type_name_length(io::IO, t::Type) = length(string(sprint(show, t; context=io)))
function max_type_name_length(io::IO, ts::Union)
    maximum(max_type_name_length.((io,), types_in_union(ts)))
end

types_in_union(x) = _types_in_union(x, Any[])
_types_in_union(x::Union, ts) = (_types_in_union(x.a, ts); _types_in_union(x.b, ts); ts)
_types_in_union(x, ts) = (push!(ts, x); ts)
