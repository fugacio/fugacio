# fugacio

Umbrella package for the Fugacio stack. Installing it pulls the three
components together:

- `fugacio-thermo`: differentiable thermodynamics and physical properties
- `fugacio-sim`: differentiable steady-state and dynamic process simulation
- `fugacio-copilot`: an AI design copilot for the stack

```bash
pip install fugacio
```

It ships no code of its own. Everything is importable from the `fugacio`
namespace provided by the component packages, for example `fugacio.thermo`,
`fugacio.sim`, and `fugacio.copilot`.
