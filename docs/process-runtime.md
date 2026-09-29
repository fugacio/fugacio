# Shared process runtime

Fugacio represents a process as physical unit kernels connected by named
streams. `Flowsheet`, saved cases, the CLI, and copilot studies use the same
`ProcessGraph`. Sequential and simultaneous execution choose how to find a
solution; they use the same connection equations and implicit derivatives.
Native `EOFlowsheet` declarations also use graph connectivity validation and
local sparse residual assembly. Custom residual units can add their own
scalar equations and auxiliary variables.

## State and equations

Each internal stream stores component molar flows, temperature, pressure, and
component vapor flows. For C components, that is 2C + 2 coordinates. The
vapor inventory carries pure-fluid saturation quality between units. An
unresolved stream retains the ordinary stream sentinel; no extra PH flash is
inserted merely to connect two units.

For a unit with inputs x and parameters p, the connection equations are
`outlet - unit(x, p) = 0`. Unit kernels retain their thermodynamic solves,
reports, and structured column equations. A column's stages aren't expanded
into the process connection matrix.

`ProcessGraph` validates unique stream ownership, undefined inputs, and tear
references. It orders strongly connected components and selects recycle tears.
`CompiledGraph` binds component layouts, scalar incidence, and local derivative
kernels. It contains no numerical operating point. Changed feed values, package
coefficients, and operating parameters remain dynamic inputs.

## Execution strategies

```python
from fugacio.sim import Flowsheet

# Register feeds and physical unit functions on fs.
fs = Flowsheet()
# fs.feed(...)
# fs.unit(...)

# Once the graph has been populated:
# sequential = fs.solve(theta, strategy="sequential")
# simultaneous = fs.solve(theta, strategy="simultaneous")
```

Sequential execution evaluates acyclic units in order and converges each
recycle partition. Broyden remains the default tear method. Simultaneous
execution seeds the connection state with a unit pass and applies sparse
Newton steps with residual-decreasing backtracking. Explicit guesses can seed
either strategy. An unsuccessful solve retains its best state and report.

Concrete process iterations run on the host. Physical unit kernels remain
compiled, so a recycle iteration doesn't compile another copy of an entire
column and exchanger train. An explicit enclosing `jax.jit` still stages the
orchestration. For large saved processes, call `CaseRunner.evaluate` or a study
directly to retain the intended compilation boundaries.

The primal process solve has an explicit derivative boundary before the graph's
implicit derivative is attached. Detached inputs alone don't guarantee kernel
reuse under JAX partial evaluation; the boundary also prevents a study from
recompiling the ordinary primal unit programs when entering autodiff.

Saved columns use one fixed input layout for cold and accepted-profile starts.
A dynamic selector chooses initialization inside the same compiled kernel;
starting a study from its audited run doesn't require a second column solver.
Column and exchanger templates retain their existing property and implicit
solver boundaries. They don't wrap feed preparation, the solver, output
profiles, and reports in an additional fused executable.

Exchanger duty searches also run on the host when concrete. Bracket endpoints,
bisection trials, and the final state share one compiled temperature-curve
kernel. The checked root still supplies implicit derivatives and stages under
an enclosing JIT. Phase-change curves and acceptance tolerances are unchanged.

The saved-case `backend` names remain `sequential` and `eo`. Both now execute
the same graph. `SolverOptions(plant_solver="sparse")` is the default;
`plant_solver="dense"` selects a reference matrix for comparisons. The CLI
exposes `--plant-solver sparse|dense`, and copilot case tools accept the same
option. Native `EOFlowsheet` uses `linear_solver="sparse"` or `"dense"`.

## Local implicit derivatives

The runtime linearizes each unit with respect to its internal input streams.
It assembles those small blocks and the outlet identity into the sparse
process Jacobian A. Parameter and feed directions supply the right-hand side.
The converged state derivative solves `A dx = -F_p dp`; reverse mode solves
its transpose. Neither recycle iterations, Newton steps, nor warm-start
selection carry derivatives.

Each unit is linearized once. Input directions run sequentially through its
retained compiled tangent kernels. Concrete assembly doesn't add a fused JIT
around the linearization callback, whose identity changes with each point.
An explicit enclosing trace stages the directions as a sequential map. Neither
path embeds another nonlinear plant solve in Jacobian assembly.
Increasing the number of process variables doesn't multiply a column's
internal tangent batch. A saved study can retain the assembled linearization
for multiple JVPs or VJPs. The thermo sensitivity API continues to support
forward, reverse, and automatically selected Jacobian directions.

Sparse factorization uses SciPy's pivoted SuperLU on the CPU, including when
called from a JAX computation on an accelerator. It is not an accelerator
sparse solver. JAX differentiates the matrix equation, which supports forward,
reverse, and higher derivatives without differentiating LU pivots.

A bounded numerical factor cache retains at most eight factorizations and
64 MiB of estimated factor storage. Keys include every coefficient, index,
dtype, and matrix dimension. Repeated right-hand sides and transposes reuse
the same factors; changed numerical matrices cannot reuse stale factors.
`fugacio.sim.numerics.clear_factor_cache()` releases them independently of
compiled kernels.

Each sparse solve checks both normwise backward error and residual relative to
each right-hand side. A singular or inaccurate solve returns nonfinite values.
There is no automatic dense process fallback. The separately selected dense
reference solver remains available for small cross-checks. Column internals
retain their existing checked block solver and its documented fallback.

## Coupled design specifications

Saved design specifications add manipulated variables and metric equations to
the connection system. Variables are normalized by their declared bounds.
Newton trials evaluate the connection and metric equations at the current
state; they don't launch an outer sequence of complete process solves.

One ordinary process evaluation supplies the initial state. The bordered
system then converges state and specification variables together. Its implicit
derivative includes the specification equations. A manipulated parameter's
initial guess doesn't become an independent physical degree of freedom.
Bounds restrict trial points; a target that cannot be reached is reported as
a failed specification, with the best state retained for diagnosis.

## Evidence and extensions

Run artifacts record numerical reports and graph incidence under
`checks.numerical.process_graph`. `fugacio diagnose CASE` additionally reports
partitions, unit templates, and structured columns. Reports distinguish actual
stored coefficients from the possible nonzeros in the declared incidence.
Structural matching is an upper bound on rank, not proof of convergence or
physical validity.

After solving, retained unit results provide duties, work, profiles, and
numerical reports. Saved runs independently audit stream equilibrium and
stability, material and energy balances, operating limits, specification
accuracy, and finite metrics. Studies retain their independently audited
finite differences or final candidate. Sparse assembly doesn't relax these
acceptance criteria.

`ResidualGraph` supports custom equation blocks. Each block declares its scalar
input indices, residual count, and local numerical function. The complete
system must be square. A dependency declaration must hold across the model's
domain; it cannot be inferred from zeros at a single operating point. Native
EO custom blocks default to all unknowns unless they declare a narrower
contract. Built-in EO blocks delegate to the common physical unit kernels.

## Breaking changes before 1.0

- `eo_jacobian` and `--eo-jacobian` have been removed. Select `plant_solver` or
  `--plant-solver` for saved cases, and `linear_solver` for native EO builders.
- The separate saved-case EO adapter has been removed.
- Native EO streams now have 2C + 2 coordinates, including vapor inventory.
  Custom equations must constrain those coordinates. Built-in compressors and
  exchangers no longer expose private internal solves as global auxiliaries.
- Native exchanger equations now use the common segmented exchanger model.
- Old run replay defaults and old column-profile reconstruction have been
  removed. Recreate runs from their saved cases with the current engine.
- The unchecked `fugacio.thermo.implicit.newton_system` convenience function
  has been removed. Use `newton_system_with_info` and inspect its report.
