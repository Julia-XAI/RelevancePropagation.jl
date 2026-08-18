# [Developer Documentation](@id developer)
## Generic LRP rule implementation
Before we dive into package-specific implementation details 
in later sections of this developer documentation, 
we first need to cover some fundamentals of LRP, starting with our notation.

The generic LRP rule, of which the ``0``-, ``\epsilon``- and ``\gamma``-rules are special cases, reads[^1][^2]

```math
\begin{equation}
R_j^k = \sum_i \frac{\rho(W_{ij}) \; a_j^k}{\epsilon + \sum_{l} \rho(W_{il}) \; a_l^k + \rho(b_i)} R_i^{k+1}
\end{equation}
```

where 
*  $W$ is the weight matrix of the layer
*  $b$ is the bias vector of the layer
*  $a^k$ is the activation vector at the input of layer $k$
*  $a^{k+1}$ is the activation vector at the output of layer $k$
*  $R^k$ is the relevance vector at the input of layer $k$
*  $R^{k+1}$ is the relevance vector at the output of layer $k$
*  $\rho$ is a function that modifies parameters (what we call [`modify_parameters`](@ref custom-rules))
*  $\epsilon$ is a small positive constant to avoid division by zero


Subscript characters are used to index vectors and matrices 
(e.g. $b_i$ is the $i$-th entry of the bias vector), 
while the superscripts $^k$ and $^{k+1}$ 
indicate the relative positions of activations $a$ and relevances $R$ in the model.
For any $k$, $a^k$ and $R^k$ have the same shape. 

Note that every term in this equation is a scalar value,
which removes the need to differentiate between matrix and element-wise operations.

### Linear layers
LRP was developed for *deep rectifier networks*,
neural networks that are composed of linear layers with ReLU activation functions.
Linear layers are layers that can be represented as affine transformations of the form 

```math
\begin{equation}
f(x) = Wx + b \quad .
\end{equation}
```

This includes most commonly used types of layers, such as fully connected layers, 
convolutional layers, pooling layers, and normalization layers.

We will now describe a generic implementation of equation (1) 
that can be applied to any linear layer.

### [The automatic differentiation fallback](@id fallback)
The computation of the generic LRP rule can be decomposed into four steps[^1]:

```math
\begin{array}{lr}
z_{i} = \sum_{l} \rho(W_{il}) \; a_l^k + \rho(b_i) & \text{(Step 1)} \\[0.5em]
s_{i} = R_{i}^{k+1} / (z_{i} + \epsilon)           & \text{(Step 2)} \\[0.5em]
c_{j} = \sum_i \rho(W_{ij}) \; s_{i}               & \text{(Step 3)} \\[0.5em]
R_{j}^{k} = a_{j}^{k} c_{j}                        & \text{(Step 4)}
\end{array}
```

**To compute step 1**, we first create a modified layer, 
applying $\rho$ to the weights and biases 
and replacing the activation function with the identity function.
The vector $z$ is then computed using a forward pass through the modified layer.
It has the same dimensionality as $R^{k+1}$ and $a^{k+1}$.

**Step 2** is an element-wise division of $R^{k+1}$ by $z$.
To avoid division by zero, a small constant $\epsilon$ is added to $z$ when necessary.

**Step 3** is trivial for fully connected layers, 
as $\rho(W)$ corresponds to the weight matrix of the modified layer.
For other types of linear layers, however, the implementation is more involved:
A naive approach would be to construct a large matrix $W$
that corresponds to the affine transformation $Wx+b$ implemented by the modified layer.
This has multiple drawbacks:
- the implementation is error-prone
- a separate implementation is required for each type of linear layer
- for some layer types, e.g. pooling layers, the matrix $W$ depends on the input
- for many layer types, e.g. convolutional layers, 
  the matrix $W$ is very large and sparse, mostly consisting of zeros,
  leading to a large computational overhead

A better approach can be found by observing that the matrix $W$ is the Jacobian
of the affine transformation $f(x) = Wx + b$.
The vector $c$ computed in step 3 corresponds to $c = s^T W$,
a so-called *Vector-Jacobian-Product* (VJP) of the vector $s$ with the Jacobian $W$. 

VJPs are the fundamental building blocks of reverse-mode automatic differentiation (AD),
and therefore implemented by most AD frameworks in a highly performant, matrix-free, GPU-accelerated manner.
Note that computing the VJP is much more efficient than first computing the full Jacobian
$W$ and later multiplying it with $s$. 
This is due to the fact that computing the full Jacobian of a function 
$f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ requires computing $m$ VJPs.

**Finally, step 4** consists of an element-wise multiplication of the vector $c$ 
with the input activation vector $a^k$, resulting in the relevance vector $R^k$.

This four-step computation is the body of the generic rule implementation,
the function [`propagate`](@ref RelevancePropagation.propagate) described below.
It is used in RelevancePropagation.jl as the default method
for all combinations of rules and layer types
that don't have a more specialized implementation.

For more background information on automatic differentiation, refer to the 
[JuML lecture on AD](https://adrianhill.de/julia-ml-course/L6_Automatic_Differentiation/).

## LRP by redefining Enzyme's VJPs
Notice that the four steps above are a *modified VJP*:
the incoming relevance $R^{k+1}$ plays the role of the output cotangent,
which is massaged (divided by $\tilde z$),
pulled back through a parameter-modified layer,
and massaged again (multiplied by $\tilde a^k$).
In other words, **LRP is reverse-mode AD in which each layer's true VJP is
replaced by the rule's relevance propagation map** —
the relevance $R^k$ *is* the cotangent at the layer input $a^k$.

RelevancePropagation.jl implements LRP exactly this way,
by redefining the VJPs Enzyme uses:
one Enzyme reverse pass over the model computes the entire explanation,
and every rule is an [`EnzymeRules`](https://enzyme.mit.edu/julia/stable/generated/custom_rule/)
custom rule that replaces the layer's VJP.
Enzyme's input shadow `dx` — what would be the input gradient in plain
backpropagation — is the explanation.

This design has several consequences:
- The forward pass, the reverse iteration over layers, and the dataflow
  routing through `Chain`, `Parallel` and `SkipConnection` layers are all
  handled by Lux's own `apply` plumbing, differentiated by Enzyme.
  There is no hand-rolled backward-pass engine.
- Relevance summation at branch points falls out of shadow accumulation:
  when two branches propagate relevance to the same input,
  Enzyme accumulates both contributions into the same shadow.
- The relevance *split* at branch points is one small custom rule on the
  connection function of `Parallel` and `SkipConnection` layers,
  which distributes relevance proportionally to each branch's contribution.

All Enzyme-specific code is contained in the file
[`/src/autodiff.jl`](https://github.com/Julia-XAI/RelevancePropagation.jl/blob/main/src/autodiff.jl).

### Rule-carrying nodes
When called, the [`LRP`](@ref) analyzer first *wraps* the model:
each layer that has a rule assigned to it is wrapped in a `LayerWithRule`,
and each branch connection in a `ConnectionWithRule`.
The wrappers are parameter- and state-transparent,
so the model's original `ps` and `st` trees apply to the wrapped model unchanged.

Layers that carry an activation function are split at wrap time
into a `SplitActivationNode`:
the activation-stripped, affine part carries the rule,
and the activation follows as a separate node carrying the [`PassRule`](@ref).
The activation stays in the compute graph —
the forward pass is unchanged —
but the reverse pass passes relevance through it untouched,
as LRP prescribes for elementwise activations.
"LRP ignores activations" is thereby a structural property of the wrapped model,
and rules only ever propagate through affine (or activation-free) layers.
Sub-models treated as one differentiation unit are exempt from the split.

```@docs
RelevancePropagation.LayerWithRule
RelevancePropagation.SplitActivationNode
RelevancePropagation.lrp_node
RelevancePropagation.ConnectionWithRule
```

Applying a `LayerWithRule` routes the layer call through the function `lrp_node`,
whose Enzyme custom rule does two things:
- The *augmented forward pass* applies the (affine) layer
  and caches the input $a^k$ and its output —
  the pre-activation $z^k$ — on Enzyme's tape.
- The *reverse pass* receives the accumulated output relevance $R^{k+1}$
  in the return shadow and calls the rule's
  [`propagate`](@ref RelevancePropagation.propagate) function,
  accumulating the resulting $R^k$ into the input shadow.

Caching the pre-activation $z^k$ is a key optimization:
for rules that neither modify the input nor the parameters
(like [`ZeroRule`](@ref) and [`EpsilonRule`](@ref), the most common case),
$\tilde z = z^k$ is already on the tape,
so step 1 requires no additional forward pass at all.

### Rule calls
Now that you are familiar with both the API and the four-step computation of
the generic LRP rules, the following implementation,
which is the actual generic rule from `src/rules.jl`,
should be straightforward to understand:

```julia
function propagate(rule::AbstractLRPRule, layer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    ãᵏ = modify_input(rule, aᵏ)
    ρps = modify_params(rule, ps)         # lazily ρ-modified parameters
    z̃ = (ρps === ps && ãᵏ === aᵏ) ? zᵏ : first(apply(layer, ãᵏ, ρps, st))
    s = Rᵏ⁺¹ ./ modify_denominator(rule, z̃)
    c = input_vjp(layer, ãᵏ, ρps, st, s)
    return ãᵏ .* c
end
```

```@docs
RelevancePropagation.propagate
```

`propagate` is a *pure function*: it receives everything it needs as arguments
and returns the input relevance.
Rules only hold their hyperparameters — modified parameters are computed
lazily from the layer's `ps` NamedTuple on each call via
[`modify_params`](@ref RelevancePropagation.modify_params),
which returns `ps` itself (`===`) when nothing changes,
signalling that the cached pre-activation can be reused.

Not only `propagate` dispatches on the rule and layer type, 
but also the internal functions `modify_input` and `modify_denominator`.
Unknown layers that are registered in the `LRP_CONFIG` use this exact function.

All LRP rules are implemented in the file
[`/src/rules.jl`](https://github.com/Julia-XAI/RelevancePropagation.jl/blob/main/src/rules.jl).

### Input VJPs
The VJP in step 3 is computed by `input_vjp`.
For activation-free `Dense`, `Scale`, `Conv` and `ConvTranspose` layers,
hand-written fast paths compute the VJP directly
(e.g. $W^\top s$ for `Dense`, `∇conv_data` for `Conv`) —
one transpose-like operation, with no nested AD involved.
All other layers fall back to `seeded_pullback`,
a nested Enzyme reverse pass over the scalar loss `dot(layer(x), s)`.

```@docs
RelevancePropagation.input_vjp
RelevancePropagation.seeded_pullback
```

Rules that require several VJPs with different seeds through the same layer,
like [`AlphaBetaRule`](@ref), simply call `input_vjp` once per seed.

### Specialized implementations
In other programming languages, LRP is commonly implemented in an object-oriented manner,
providing a single backward pass implementation per rule.
This can be seen as a form of *single dispatch* on the rule type.

Using multiple dispatch, we can implement specialized versions of `propagate`
that not only take into account the rule type, but also the layer type, 
for example for fully connected layers or reshaping layers. 

Reshaping and dropout layers don't affect attributions.
For rules that neither modify the input nor the denominator
([`ZeroRule`](@ref) and [`EpsilonRule`](@ref)),
we can therefore avoid the computational overhead of AD
by writing specialized implementations that simply reshape back:
```julia
function propagate(rule::ZeroRule, layer::ReshapingLayer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    return reshape(Rᵏ⁺¹, size(aᵏ))
end
```
RelevancePropagation.jl provides these specializations for `ZeroRule` and
`EpsilonRule` on both `ReshapingLayer` and `DropoutLayer` types.

Some rule–layer combinations don't require a VJP at all.
The [`FlatRule`](@ref) distributes relevance uniformly over all input neurons
connected to an output neuron, so for `Dense` layers, the input relevance can
be written directly, skipping both the forward pass and AD:

```julia
function propagate(rule::FlatRule, layer::Dense, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    n = size(aᵏ, 1) # number of input neurons connected to each output neuron
    Rᵏ = similar(aᵏ)
    for i in axes(Rᵏ, 2) # samples in batch
        fill!(view(Rᵏ, :, i), sum(view(Rᵏ⁺¹, :, i)) / n)
    end
    return Rᵏ
end
```

For maximum low-level control beyond `modify_input` and `modify_denominator`,
you can also implement your own `propagate` method and dispatch
on individual rule types `MyRule` and layer types `MyLayer`:
```julia
function propagate(rule::MyRule, layer::MyLayer, aᵏ, zᵏ, ps, st, Rᵏ⁺¹)
    return ...
end
```

[^1]: G. Montavon et al., [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
[^2]: W. Samek et al., [Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications](https://ieeexplore.ieee.org/document/9369420)
