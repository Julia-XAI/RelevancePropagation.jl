function prepare_vit(model::ViT)
    model = model.layers # remove wrapper type
    model = flatten_model(model) # model consists of nested chains
    testmode!(model) # make shure there is no dropout during forward pass
    if !isa(model[end - 2], typeof(seconddimmean))
        model = Chain(model[1:(end - 3)]..., SelectClassToken(), model[(end - 1):end]...) # swap anonymous function to actual layer
    end
    return model
end
