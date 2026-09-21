using RelevancePropagation
using Test

@testset verbose = true "RelevancePropagation.jl" begin
    @testset "Utilities" begin
        @info "Testing utilities..."
        include("test_utils.jl")
    end
    @testset verbose = true "ModelSurgeon" begin
        @info "Testing ModelSurgeon submodule..."
        @testset "Traversal" begin
            include("modelsurgeon/test_traverse.jl")
        end
        @testset "Layer utilities" begin
            include("modelsurgeon/test_layer_utils.jl")
        end
        @testset "Flatten" begin
            include("modelsurgeon/test_flatten.jl")
        end
        @testset "Strip softmax" begin
            include("modelsurgeon/test_strip_softmax.jl")
        end
        @testset "Canonize" begin
            include("modelsurgeon/test_canonize.jl")
        end
    end
    @testset "Layer indexing" begin
        @info "Testing layer indexing..."
        include("test_layer_indices.jl")
    end
    @testset "Enzyme pullbacks" begin
        @info "Testing Enzyme pullbacks against Zygote..."
        include("test_autodiff.jl")
    end
    @testset "Forward-pass equivalence" begin
        @info "Testing forward-pass equivalence of wrapped models..."
        include("test_forward.jl")
    end
    @testset "Model checks" begin
        @info "Testing model checks..."
        include("test_checks.jl")
    end
    @testset "LRP rules" begin
        @info "Testing LRP rules..."
        include("test_rules.jl")
    end
    @testset "LRP analyzer" begin
        @info "Testing LRP analyzer..."
        include("test_lrp.jl")
    end
    @testset "LRP composites" begin
        @info "Testing LRP composites..."
        include("test_composite.jl")
    end
    @testset "CRP" begin
        @info "Testing CRP..."
        include("test_crp.jl")
    end
    @testset "CNN" begin
        @info "Testing analyzers on CNN..."
        include("test_cnn.jl")
    end
    @testset "Batches" begin
        @info "Testing analyzers on batches..."
        include("test_batches.jl")
    end
    @testset "Linting" begin
        @info "Testing linting..."
        include("test_linting.jl")
    end
    @testset "Benchmarks" begin
        @info "Testing benchmarks..."
        include("test_benchmarks.jl")
    end
end
