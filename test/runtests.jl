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

    # The v3 test files below depend on Flux and on functionality that is not
    # ported yet (v4.0.0 Lux/Enzyme rewrite, see PLAN.md). They cannot be
    # included until their phase lands, so they are marked broken here instead
    # of being removed. Ports replace these markers with `include`s.
    # (Restored v3 tests for unported rules and model utilities live as skipped
    # tests inside test_rules.jl, test_utils.jl and test_chain_utils.jl.)
    @testset verbose = true "Not yet ported" begin
        for file in [
            "test_crp.jl: CRP analyzer (phase 6)",
            "test_cnn.jl: CNN reference tests (phase 7)",
            "test_batches.jl: batch consistency on CNN (phase 7)",
            "test_benchmarks.jl: PkgJogger benchmark suite (phase 7)",
            "test_linting.jl: JuliaFormatter/Aqua/ExplicitImports (phase 7)",
        ]
            @testset "$file" begin
                @test_broken false
            end
        end
    end
end
