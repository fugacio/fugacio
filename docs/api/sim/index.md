# `fugacio.sim`

The differentiable process-simulation layer (depends on `fugacio.thermo`):
streams, energy-balanced unit operations, a recycle/tear solver, columns,
reactors, optimization, economics, and the time-domain dynamics, control, and
heat-integration toolkits. Import the public surface from the package root:

```python
from fugacio.sim import Stream, flash_drum, tear_solve, linear_mpc, pinch_analysis
```

::: fugacio.sim
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false

## Where to look next

| Area | Page | Key symbols |
| --- | --- | --- |
| Streams & properties | [Streams & properties](streams.md) | `Stream`, `enthalpy_flow`, `molar_enthalpy`, `liquid_density` |
| Unit operations | [Unit operations](units.md) | `flash_drum`, `heater`, `pump`, `compressor`, `mix`, `splitter`, `flash_vle` |
| Flowsheet & recycle | [Flowsheet & recycle](flowsheet.md) | `Flowsheet`, `tear_solve` |
| Thermodynamic models | [Thermodynamic models](models.md) | `eos_model_for`, `nrtl_model_for`, `uniquac_model_for`, `unifac_model_for` |
| Distillation & diagrams | [Distillation & diagrams](columns.md) | `shortcut_column`, `solve_column`, `pxy_diagram`, `residue_curve_map` |
| Reactors | [Reactors & reactive separations](reactors.md) | `equilibrium_reactor`, `cstr`, `pfr`, `reactive_flash`, `reactive_distillation` |
| Optimization & design | [Optimization & design](optimization.md) | `minimize`, `argmin`, `meet_spec`, `optimize_flowsheet` |
| Economics & sizing | [Economics & sizing](economics.md) | `heat_exchanger_area`, `bare_module_cost`, `total_annual_cost`, `npv` |
| Dynamics & control | [Dynamics & control](dynamics.md) | `odeint`, `integrate`, `PID`, `DynamicFlowsheet`, `tune_pid` |
| Advanced control (MPC) | [Advanced control (MPC)](mpc.md) | `linear_mpc`, `nonlinear_mpc`, `solve_qp`, `KalmanFilter`, `tune_mpc` |
| Heat integration | [Heat integration](integration.md) | `pinch_analysis`, `composite_curves`, `optimal_dt_min`, `synthesize_network` |
| Steam & cooling utilities | [Steam & cooling utilities](utilities.md) | `steam_heating`, `cooling_water`, `steam_turbine` |

See the [optimization & economics guide](../../optimization.md), the
[dynamics & control guide](../../dynamics.md), the
[advanced-control guide](../../advanced-control.md), and the
[heat-integration guide](../../heat-integration.md) for worked examples.
