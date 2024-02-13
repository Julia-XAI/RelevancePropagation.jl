if VERSION < v"1.8.0-DEV.1494" # 98e60ffb11ee431e462b092b48a31a1204bd263d
    export allequal
    allequal(itr) = isempty(itr) ? true : all(isequal(first(itr)), itr)
    allequal(c::Union{AbstractSet,AbstractDict}) = length(c) <= 1
    allequal(r::AbstractRange) = iszero(step(r)) || length(r) <= 1
end
