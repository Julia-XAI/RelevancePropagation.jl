@testset "CRP analytic" begin
    W1 = [1.0 3.0; 4.0 2.0]
    b1 = [0.0, 1.0]
    d1 = Dense(W1, b1, identity)

    W2 = [2.0 4.0; 3.0 1.0]
    b2 = [1.0, 2.0]
    d2 = Dense(W2, b2, identity)

    model = Chain(d1, d2)
    input = reshape([1.0 2.0], 2, 1)

    layer_index = 1
    features = TopNFeatures(1)
    analyzer = CRP(LRP(model), layer_index, features)

    # Analytic solution:
    # a¹ = input
    # a²[1] = 1*1 + 3*2 + 0 =  1 +  6 + 0 =  7
    # a²[2] = 4*1 + 2*2 + 1 =  4 +  4 + 1 =  9
    # a³[1] = 2*7 + 4*9 + 1 = 14 + 36 + 1 = 51
    # a³[2] = 3*7 + 1*9 + 2 = 21 +  9 + 2 = 32
    # R³ = [1 0], max output neuron selection, masked to 1
    # R²[1] = 14/51 * 1 +  21/32 * 0 = 14/51
    # R²[2] = 36/51 * 1 +   9/32 * 0 = 36/51
    # R² = [0 36/51], CRP Top 1 concept neuron
    # R¹[1] = 1/7 * 0 + 4/9 * 36/51 = 16//51
    # R¹[2] = 6/7 * 0 + 4/9 * 36/51 = 16//51

    expl = analyzer(input)
    @test expl.val ≈ [16 / 51, 16 / 51]
end
