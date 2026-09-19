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
| Unit operations | [Unit operations](units.md) | `flash_drum`, `adiabatic_flash`, `heater`, `pump`, `compressor`, `mix`, `decanter`, `unit_limits`, `enable_compilation_cache` |
| Flowsheet & recycle | [Flowsheet & recycle](flowsheet.md) | `Flowsheet`, `FlowsheetResult`, `UnitRecord`, `tear_solve_with_info`, `continuation_solve` |
| Process acceptance | [Process acceptance](acceptance.md) | `audit_stream`, `flash_drum_checked`, `audit_flowsheet`, `BalanceBoundary` |
| Process cases & studies | [Process cases & studies](cases.md) | `ProcessCase`, `CaseRunner`, `SolverOptions`, `sweep`, `optimize`, `sensitivities` |
| Two-sided heat exchanger | [Two-sided heat exchanger](heat_exchanger.md) | `heat_exchanger`, `HeatExchangerResult` |
| Equation-oriented flowsheeting | [Equation-oriented flowsheeting](eo.md) | `EOFlowsheet`, `Flash`, `HeatExchanger`, `Column`, `optimize_flowsheet_eo` |
| Thermodynamic models | [Thermodynamic models](models.md) | `package_for`, `METHODS`, `helmholtz_package_for`, `UnifacModel` |
| Rigorous distillation | [Rigorous distillation](distillation.md) | `rigorous_column`, `ColumnFeed`, `reflux_ratio`, `purity`, `absorber`, `stripper` |
| Distillation & diagrams | [Distillation & diagrams](columns.md) | `shortcut_column`, `relative_volatility`, `pxy_diagram`, `residue_curve_map` |
| Reactors | [Reactors & reactive separations](reactors.md) | `equilibrium_reactor`, `cstr`, `pfr`, `stoichiometric_reactor`, `reactive_flash`, `reactive_column` |
| Optimization & design | [Optimization & design](optimization.md) | `minimize`, `argmin`, `meet_spec`, `optimize_flowsheet` |
| Economics & sizing | [Economics & sizing](economics.md) | `heat_exchanger_area`, `bare_module_cost`, `total_annual_cost`, `npv` |
| Dynamics & control | [Dynamics & control](dynamics.md) | `odeint`, `integrate`, `PID`, `DynamicFlowsheet`, `tune_pid` |
| Advanced control (MPC) | [Advanced control (MPC)](mpc.md) | `linear_mpc`, `nonlinear_mpc`, `solve_qp`, `KalmanFilter`, `tune_mpc` |
| Heat integration | [Heat integration](integration.md) | `pinch_analysis`, `composite_curves`, `optimal_dt_min`, `synthesize_network` |
| Steam & cooling utilities | [Steam & cooling utilities](utilities.md) | `steam_heating`, `cooling_water`, `steam_turbine` |

See the [flowsheeting guide](../../flowsheeting.md), the
[rigorous distillation guide](../../distillation.md), the
[optimization & economics guide](../../optimization.md), the
[equation-oriented flowsheeting guide](../../equation-oriented.md), the
[dynamics & control guide](../../dynamics.md), the
[advanced-control guide](../../advanced-control.md), and the
[heat-integration guide](../../heat-integration.md) for worked examples.
