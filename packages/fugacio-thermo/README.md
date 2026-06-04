# fugacio-thermo

Differentiable thermodynamics and physical-property engine for the
[Fugacio](https://github.com/owenthcarey/fugacio) stack. JAX-based models for
activity coefficients and phase equilibrium, differentiable with respect to both
composition and model parameters.

```python
from fugacio.thermo import margules_gamma

gamma1, gamma2 = margules_gamma(x1=0.3, a12=0.5, a21=0.8)
```

Part of the `fugacio` namespace; installs independently:
`pip install fugacio-thermo`.
