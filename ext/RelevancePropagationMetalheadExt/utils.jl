function canonize(model::ViT)
    model = model.layers # remove wrapper type
    model = flatten_model(model) # model consists of nested chains
    testmode!(model) # make shure there is no dropout during forward pass
    if !isa(model[end - 2], typeof(seconddimmean))
        model = Chain(model[1:(end - 3)]..., SelectClassToken(), model[(end - 1):end]...) # swap anonymous function to actual layer
    end
    return canonize(model)
end

# these are originally from NNlib.jl, but since they are unexported, we don't want
# to rely on them an re-define them here
split_heads(x, nheads) = reshape(x, size(x, 1) ÷ nheads, nheads, size(x)[2:end]...)
join_heads(x) = reshape(x, :, size(x)[3:end]...)
