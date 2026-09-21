using RelevancePropagation
using Test
using Aqua
using JET
using ExplicitImports
using JuliaFormatter: JuliaFormatter

@testset "Code formatting" begin
    @info "Running JuliaFormatter.jl formatting tests."
    @test JuliaFormatter.format(RelevancePropagation; verbose=false, overwrite=false)
end

@testset "Aqua.jl" begin
    @info "Running Aqua.jl code-quality tests. These might print warnings from dependencies."
    Aqua.test_all(RelevancePropagation; ambiguities=false)
    Aqua.test_ambiguities(RelevancePropagation)
end

@testset "ExplicitImports.jl" begin
    @info "Running ExplicitImports.jl import-hygiene tests."
    @test check_no_implicit_imports(RelevancePropagation) === nothing
    @test check_no_stale_explicit_imports(RelevancePropagation) === nothing
    @test check_all_explicit_imports_via_owners(RelevancePropagation) === nothing
    @test check_all_qualified_accesses_via_owners(RelevancePropagation) === nothing
    @test check_no_self_qualified_accesses(RelevancePropagation) === nothing
end

# JET's v0.11 series supports Julia v1.12 and above only.
if VERSION >= v"1.12"
    @testset "JET.jl" begin
        @info "Running JET.jl type-stability tests."
        JET.test_package(RelevancePropagation; target_modules=(RelevancePropagation,))
    end
end
