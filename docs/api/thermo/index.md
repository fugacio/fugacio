# `fugacio.thermo`

The differentiable thermodynamics and physical-property engine: the foundation
of the stack, with no internal dependencies. Import the public surface straight
from the package root:

```python
from fugacio.thermo import PR, flash_pt, reference_fluid, saturation_state
```

::: fugacio.thermo
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false

## Where to look next

The reference is grouped by topic so each page stays scannable:

| Area | Page | Key symbols |
| --- | --- | --- |
| Components & data | [Components & data](data.md) | `Component`, `DATABASE`, `component_arrays`, `R`, `T_REF` |
| Equations of state | [Equations of state](eos.md) | `PR`, `SRK`, `RK`, `VDW`, `ln_phi_mixture`, `molar_volume` |
| Phase equilibrium | [Phase equilibrium](equilibrium.md) | `flash_pt`, `rachford_rice`, `bubble_pressure_eos`, `EOSModel`, `GammaPhiModel` |
| LLE & VLLE | [Liquid-liquid & VLLE](lle.md) | `flash_lle`, `flash_vlle`, `heterogeneous_azeotrope` |
| Activity models | [Activity & group contribution](activity.md) | `NRTL`, `UNIQUAC`, `Wilson`, `Margules`, `unifac_activity`, `joback_estimate` |
| Energy & properties | [Energy & property models](energy.md) | `flash_ph`, `flash_ps`, `residual_properties`, `cp_ig`, `heat_of_vaporization` |
| Reference fluids | [Reference fluids (Helmholtz)](helmholtz.md) | `reference_fluid`, `saturation_state`, `state_ph`, `water_viscosity` |
| Transport & volumetric | [Transport & volumetric](transport.md) | `gas_viscosities`, `liquid_density`, `surface_tensions`, `gas_diffusivity` |
| Reactions & kinetics | [Reactions & kinetics](reactions.md) | `Reaction`, `equilibrium_constant`, `Arrhenius`, `PowerLaw`, `LHHW` |
| Regression & ThermoML | [Regression & ThermoML](regression.md) | `levenberg_marquardt`, `fit_nrtl_binary`, `ParameterBank`, `read_thermoml` |

See the [non-ideal phase equilibrium guide](../../phase-equilibrium.md) and the
[reference-fluids guide](../../reference-fluids.md) for worked examples.
