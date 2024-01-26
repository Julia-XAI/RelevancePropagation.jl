using Documenter
using Literate
using RelevancePropagation

LITERATE_DIR = joinpath(@__DIR__, "src/literate")
OUT_DIR = joinpath(@__DIR__, "src/generated")

# Use Literate.jl to generate docs and notebooks of examples
function convert_literate(dir_in, dir_out)
    for p in readdir(dir_in)
        path = joinpath(dir_in, p)

        if isdir(path)
            convert_literate(path, joinpath(dir_out, p))
        else # isfile
            Literate.markdown(path, dir_out; documenter=true) # Markdown for Documenter.jl
            Literate.notebook(path, dir_out) # .ipynb notebook
            Literate.script(path, dir_out) # .jl script
        end
    end
end
convert_literate(LITERATE_DIR, OUT_DIR)

makedocs(;
    modules=[RelevancePropagation, XAIBase],
    authors="Adrian Hill",
    sitename="RelevancePropagation.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true", assets=String[]),
    #! format: off
    pages=[
        "Home"            => "index.md",
        "Basic usage"     => "generated/basics.md",
        "Advanced usage"  => Any[
            "Assigning rules to layers"     => "generated/composites.md",
            "Supporting new layer types"    => "generated/custom_layer.md",
            "Custom LRP rules"              => "generated/custom_rules.md",
            "Concept Relevance Propagation" => "generated/crp.md",
            "Developer documentation"       => "developer.md",
        ],
        "API reference"   => "api.md",
    ],
    #! format: on
    linkcheck=true,
    linkcheck_ignore=[
        r"https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10",
        r"https://www.nature.com/articles/s42256-023-00711-8",
    ],
    warnonly=[:missing_docs],
    checkdocs=:exports, # only check docstrings in API reference if they are exported
)

deploydocs(; repo="github.com/Julia-XAI/RelevancePropagation.jl")
