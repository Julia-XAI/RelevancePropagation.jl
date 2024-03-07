module VisionTransformerExt
    using RelevancePropagation, Metalhead, Flux

    include("layers.jl")
    include("rules.jl")
    include("utils.jl")

    export prepare_vit
end