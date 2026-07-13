using Test

using Lux
using Random: AbstractRNG
using StableRNGs: StableRNG
using RelevancePropagation.ModelSurgeon: canonize, canonize_fuse, split_activation

# Also usable as a Lux `init_*` function
pseudorand(rng::AbstractRNG, dims...) = rand(rng, Float32, dims...)

batchsize = 50

# Collect BatchNorm statistics by applying the model in train mode,
# then switch to test mode.
function collect_stats(model, ps, st, x)
    _, st = Lux.apply(model, x, ps, Lux.trainmode(st))
    return Lux.testmode(st)
end

##=====================================#
# Test `canonize_fuse` on Dense layer #
#======================================#

ins = 10
outs = 5
dense = Dense(ins => outs)
bn_dense = BatchNorm(outs, relu; init_bias=pseudorand, init_scale=pseudorand)
model = Chain(dense, bn_dense)
ps, st = Lux.setup(StableRNG(123), model)

x = pseudorand(StableRNG(123), ins, batchsize)
st = collect_stats(model, ps, st, x)

dense_fused, ps_fused = @inferred canonize_fuse(
    dense, ps.layer_1, bn_dense, ps.layer_2, st.layer_2
)
@test first(Lux.apply(dense_fused, x, ps_fused, NamedTuple())) ≈
    first(Lux.apply(model, x, ps, st))

##====================================#
# Test `canonize_fuse` on Conv layer  #
#=====================================#

insize = (10, 10, 3)
conv = Conv((3, 3), 3 => 4)
bn_conv = BatchNorm(4, relu; init_bias=pseudorand, init_scale=pseudorand)
model = Chain(conv, bn_conv)
ps, st = Lux.setup(StableRNG(123), model)

x = pseudorand(StableRNG(123), insize..., batchsize)
st = collect_stats(model, ps, st, x)

conv_fused, ps_fused = @inferred canonize_fuse(
    conv, ps.layer_1, bn_conv, ps.layer_2, st.layer_2
)
@test first(Lux.apply(conv_fused, x, ps_fused, NamedTuple())) ≈
    first(Lux.apply(model, x, ps, st))

##=====================================#
# Test `canonize` on sequential models #
#======================================#

# Sequential BatchNorm layers should be fused until they create a Dense or Conv layer
# with non-linear activation function.
model = Chain(
    Conv((3, 3), 3 => 6),
    BatchNorm(6),
    Conv((3, 3), 6 => 2, identity),
    BatchNorm(2),
    BatchNorm(2, softplus),
    BatchNorm(2),
    FlattenLayer(),
    Dense(72 => 10; use_bias=false),
    BatchNorm(10),
    BatchNorm(10),
    BatchNorm(10, relu),
    BatchNorm(10),
    Dense(10 => 10, gelu),
    BatchNorm(10),
    softmax,
)
ps, st = Lux.setup(StableRNG(123), model)
st = collect_stats(model, ps, st, x)
model_canonized, ps_canonized, st_canonized = canonize(model, ps, st)

# 6 of the BatchNorm layers should be removed and the outputs should match
@test length(model_canonized) == 9 # 15 - 6
@test first(Lux.apply(model_canonized, x, ps_canonized, st_canonized)) ≈
    first(Lux.apply(model, x, ps, st))

##===================================#
# Test `canonize` on nested models   #
#====================================#

model = Chain(
    Conv((3, 3), 3 => 4),
    BatchNorm(4, relu),
    Chain(
        Conv((3, 3), 4 => 5, identity),
        BatchNorm(5),
        Chain(Conv((3, 3), 5 => 6, identity), BatchNorm(6)),
        Chain(Conv((3, 3), 6 => 7, identity), BatchNorm(7)),
    ),
)
ps, st = Lux.setup(StableRNG(123), model)
st = collect_stats(model, ps, st, x)
model_canonized, ps_canonized, st_canonized = canonize(model, ps, st)

# 4 of the BatchNorm layers should be removed and the outputs should match
@test length(model_canonized) == 4
@test first(Lux.apply(model_canonized, x, ps_canonized, st_canonized)) ≈
    first(Lux.apply(model, x, ps, st))

##================================================#
# Test `canonize` on  models w/ Parallel layers   #
#=================================================#

model = Chain(
    Conv((3, 3), 3 => 4), # fuse
    BatchNorm(4, relu),
    Parallel(
        +,
        Conv((3, 3), 4 => 5, identity),
        Chain(
            Conv((3, 3), 4 => 5, identity; use_bias=false), # fuse
            BatchNorm(5),
        ),
        Chain(
            Conv((3, 3), 4 => 5, relu), # don't fuse
            BatchNorm(5),
        ),
    ),
    Conv((3, 3), 5 => 6, identity), # fuse
    BatchNorm(6),
)
ps, st = Lux.setup(StableRNG(123), model)
st = collect_stats(model, ps, st, x)
model_canonized, ps_canonized, st_canonized = canonize(model, ps, st)

@test length(model_canonized) == 3
parallel_canonized = model_canonized[2]
@test length(parallel_canonized.layers.layer_2) == 1
@test length(parallel_canonized.layers.layer_3) == 2
@test first(Lux.apply(model_canonized, x, ps_canonized, st_canonized)) ≈
    first(Lux.apply(model, x, ps, st))

##======================================================#
# Test `canonize` on  models w/ SkipConnection layers   #
# and with layers that have to be split (LayerNorm)     #
#=======================================================#
x = pseudorand(StableRNG(123), 5, batchsize)

model = Chain(
    LayerNorm((5,)), # split
    Parallel(
        +,
        LayerNorm((5,)), # split
        Chain(
            Dense(5 => 5), # fuse
            BatchNorm(5),
        ),
        Chain(
            LayerNorm((5,)), # split
        ),
    ),
    SkipConnection(
        LayerNorm((5,)), # split
        +,
    ),
    SkipConnection(Chain(
        LayerNorm((5,)),
        Dense(5 => 5), # split
    ), +),
    SkipConnection(Chain(
        Dense(5 => 5), # fuse
        BatchNorm(5),
    ), +),
    Dense(5 => 5), # fuse
    BatchNorm(5),
)
ps, st = Lux.setup(StableRNG(123), model)
st = collect_stats(model, ps, st, x)
model_canonized, ps_canonized, st_canonized = canonize(model, ps, st)

@test length(model_canonized) == 7

# Check parallel layer
parallel_canonized = model_canonized[3]
@test length(parallel_canonized.layers.layer_1) == 2
@test length(parallel_canonized.layers.layer_2) == 1
@test length(parallel_canonized.layers.layer_3) == 2

# Check three SkipConnection layers
@test length(model_canonized[4].layers) == 2
@test length(model_canonized[5].layers) == 3
@test length(model_canonized[6].layers) == 1
@test first(Lux.apply(model_canonized, x, ps_canonized, st_canonized)) ≈
    first(Lux.apply(model, x, ps, st))

##==========================#
# Test `split_activation`   #
#===========================#

# Generic split: the layer's activation is moved into a separate
# elementwise layer; layers without one are returned unchanged.
dense = Dense(3 => 4, gelu)
ps, st = Lux.setup(StableRNG(123), dense)
x = pseudorand(StableRNG(123), 3, batchsize)
dense_split, ps_split, st_split = split_activation(dense, ps, st)
@test dense_split isa Chain
@test length(dense_split) == 2
@test dense_split[1].activation == identity
@test first(Lux.apply(dense_split, x, ps_split, st_split)) ≈
    first(Lux.apply(dense, x, ps, st))

conv = Conv((3, 3), 3 => 4, relu)
ps, st = Lux.setup(StableRNG(123), conv)
x = pseudorand(StableRNG(123), 10, 10, 3, batchsize)
conv_split, ps_split, st_split = split_activation(conv, ps, st)
@test conv_split isa Chain
@test first(Lux.apply(conv_split, x, ps_split, st_split)) ≈
    first(Lux.apply(conv, x, ps, st))

# Layers with identity activation or none at all are returned unchanged
dense_id = Dense(3 => 4)
ps, st = Lux.setup(StableRNG(123), dense_id)
@test split_activation(dense_id, ps, st) === (dense_id, ps, st)
pool = MaxPool((2, 2))
ps, st = Lux.setup(StableRNG(123), pool)
@test split_activation(pool, ps, st) === (pool, ps, st)

# LayerNorm splits its affine part and activation into a Scale layer
ln = LayerNorm((5,), relu)
ps, st = Lux.setup(StableRNG(123), ln)
x = pseudorand(StableRNG(123), 5, batchsize)
ln_split, ps_split, st_split = split_activation(ln, ps, st)
@test ln_split isa Chain
@test ln_split[1] isa LayerNorm
@test ln_split[1].activation == identity
@test ln_split[2] isa Scale
@test ln_split[2].activation == relu
@test first(Lux.apply(ln_split, x, ps_split, st_split)) ≈ first(Lux.apply(ln, x, ps, st))
