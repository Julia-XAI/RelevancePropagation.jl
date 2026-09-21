using RelevancePropagation
using Test

using PkgJogger
using BenchmarkTools

# `PkgJogger.@test_benchmarks` calls the internal two-argument `samplefunc`,
# which BenchmarkTools ≥ 1.6 no longer generates (PkgJogger v0.6.0).
# Collect the suite with PkgJogger and run each benchmark once through the
# public API instead, preserving the original "runs without erroring" checks.
PkgJogger.@jog RelevancePropagation
suite = JogRelevancePropagation.suite()

@testset "$(join(key, " / "))" for (key, bench) in BenchmarkTools.leaves(suite)
    run(bench; samples=1, evals=1)
    @test true
end
