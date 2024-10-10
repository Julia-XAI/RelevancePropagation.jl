using RelevancePropagation
using Test

using JuliaFormatter: JuliaFormatter
using Aqua: Aqua
using JET: JET
using ExplicitImports:
    check_no_implicit_imports,
    check_no_stale_explicit_imports,
    check_all_explicit_imports_via_owners,
    check_all_qualified_accesses_via_owners,
    check_no_self_qualified_accesses

@testset "Code formatting" begin
    @info "...with JuliaFormatter.jl"
    @test JuliaFormatter.format(RelevancePropagation; verbose=false, overwrite=false)
end

@testset "Aqua.jl" begin
    @info "...with Aqua.jl"
    Aqua.test_all(RelevancePropagation; ambiguities=false)
end

@testset "JET.jl" begin
    @info "...with JET.jl"
    JET.test_package(RelevancePropagation; target_defined_modules=true)
end

@testset "ExplicitImports tests" begin
    @info "...with ExplicitImports.jl"
    @testset "Improper implicit imports" begin
        @test check_no_implicit_imports(RelevancePropagation) === nothing
    end
    @testset "Improper explicit imports" begin
        @test check_no_stale_explicit_imports(RelevancePropagation;) === nothing
        @test check_all_explicit_imports_via_owners(RelevancePropagation) === nothing
        # TODO: test in the future when `public` is more common
        # @test check_all_explicit_imports_are_public(RelevancePropagation) === nothing
    end
    @testset "Improper qualified accesses" begin
        @test check_all_qualified_accesses_via_owners(RelevancePropagation) === nothing
        @test check_no_self_qualified_accesses(RelevancePropagation) === nothing
        # TODO: test in the future when `public` is more common
        # @test check_all_qualified_accesses_are_public(RelevancePropagation) === nothing
    end
end
