using RelevancePropagation
using Test

@testset verbose = true "RelevancePropagation.jl" begin
    @testset "Utilities" begin
        @info "Testing utilities..."
        include("test_utils.jl")
    end
    @testset "Model structure" begin
        @info "Testing model structure helpers..."
        include("test_chain_utils.jl")
    end
    @testset "Enzyme pullbacks" begin
        @info "Testing Enzyme pullbacks against Zygote..."
        include("test_autodiff.jl")
    end
    @testset "Model checks" begin
        @info "Testing model checks..."
        include("test_checks.jl")
    end
    @testset "Canonize" begin
        @info "Testing model canonization..."
        include("test_canonize.jl")
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
