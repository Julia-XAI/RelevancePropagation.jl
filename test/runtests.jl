using LRP
using Test
using Aqua

@testset "LRP.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LRP)
    end
    # Write your tests here.
end
