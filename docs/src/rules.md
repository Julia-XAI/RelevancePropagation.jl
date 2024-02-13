# [LRP Rule Overview](@id rules)

## Notation
We use the following notation for LRP rules: 

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
while the superscripts $^k$ and $^{k+1}$ indicate the relative positions 
of activations $a$ and relevances $R$ in the model.
For any $k$, $a^k$ and $R^k$ have the same shape. 

Note that all terms in the following equations are scalar value,
which removes the need to differentiate between matrix and element-wise operations.
For more information, refer to the [developer documentation](@ref developer).

## Basic rules
```@docs; canonical=false
ZeroRule
EpsilonRule
PassRule
```

## Lower layer rules
```@docs; canonical=false
GammaRule
AlphaBetaRule
ZPlusRule
```

## Input layer rules
```@docs; canonical=false
FlatRule
WSquareRule
ZBoxRule
```

## Specialized rules
```@docs; canonical=false 
GeneralizedGammaRule
```