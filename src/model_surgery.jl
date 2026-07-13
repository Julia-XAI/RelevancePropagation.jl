#======================#
# ModelSurgeon policy  #
#======================#

# RP's zero-config wrappers around `ModelSurgeon.flatten_model` and
# `ModelSurgeon.canonize`. LRP rules and composites dispatch on intact Lux
# pooling wrapper layers, so RP bakes the pooling leaf policy into its
# exported functions: pooling layers stay intact even under a custom
# `unwrap`. This shadows (doesn't extend) the ModelSurgeon functions,
# keeping RP policy out of the submodule and free of type piracy.

# Matches Lux's own pooling leaf table: all nine pooling wrappers,
# including the LP family, which is not part of the exported
# `PoolingLayer` union used for rule dispatch.
keep_pooling(::KeyPath, layer) = layer isa Union{PoolingLayer,LPPoolLayer}

"""
    flatten_model(model, ps, st; unwrap)

Flatten a Lux `Chain` containing nested `Chain`s.
Returns a `(model, ps, st)` triple whose layers are re-keyed to
`layer_1, ..., layer_N`; since this re-keys `ps` and `st` as well,
the transformation is joint over the Lux triple.

`Parallel` and `SkipConnection` layers keep their container,
but their branches are flattened internally.
Lux pooling layers always stay intact, since LRP rules and composites
dispatch on the pooling wrapper types.

Wrapper layers whose application is a plain pass-through to the layer they
wrap (e.g. Boltz.jl model wrappers) can be unwrapped by opting them in via
the `unwrap` keyword argument, e.g. `unwrap=Base.Fix2(isa, MyWrapper)`:
the wrapper is dropped and a wrapped `Chain` is spliced into its parent.
"""
function flatten_model(model, ps, st; unwrap=Returns(false))
    return ModelSurgeon.flatten_model(model, ps, st; exclude=keep_pooling, unwrap)
end

"""
    canonize(model, ps, st; unwrap)

Canonize a model by flattening it and fusing BatchNorm layers into preceding
Dense and Conv layers with linear activation functions.
Returns a `(model, ps, st)` triple, since fusing parameters and flattening
re-key `ps` and `st`.

BatchNorm layers are fused using the running statistics in `st`
(collected by applying the model in train mode).
LayerNorm layers containing an affine transformation or an activation
function are split into a normalization-only LayerNorm followed by a
`Scale` layer carrying both.

The `unwrap` keyword argument opts wrapper layers into unwrapping,
see [`flatten_model`](@ref).
"""
function canonize(model, ps, st; unwrap=Returns(false))
    return ModelSurgeon.canonize(model, ps, st; exclude=keep_pooling, unwrap)
end
