"""Differentiable execution of portable process cases with retained unit results."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from fugacio.sim.cases.backends import CaseFlowsheet, RegisteredBlock
from fugacio.sim.cases.costing import evaluate_economics, parse_economics, validate_economic_values
from fugacio.sim.cases.evidence import build_package
from fugacio.sim.cases.expressions import metric_values
from fugacio.sim.cases.jsonio import canonical_json
from fugacio.sim.cases.quantities import (
    TEMPERATURE,
    CaseValidationError,
    number,
    quantity,
    unit_for,
)
from fugacio.sim.cases.registry import (
    UnitDefinition,
    UnitEvaluation,
    evaluate_unit,
    parse_units,
    validate_unit_values,
)
from fugacio.sim.cases.schema import ProcessCase, parse_feeds, resolve_value, validate_feed_values
from fugacio.sim.eo import EOFlowsheet
from fugacio.sim.stream import Stream
from fugacio.thermo.acceptance import DEFAULT_POLICY, AcceptancePolicy
from fugacio.thermo.diagnostics import SolveReport, residual_report
from fugacio.thermo.implicit import newton_system_with_info


def _unit_template(definition: UnitDefinition) -> tuple[UnitDefinition, dict[str, Any]]:
    """Separate parameter bindings from unit identity for compilation reuse.

    Fixed settings remain part of the template, preserving constant folding.
    Parameter names don't affect its numerical equations or compiled shape.
    """
    bindings: dict[str, Any] = {}

    def convert(value: Any) -> Any:
        # Validated value specifications store parameter references as names.
        if isinstance(value, str):
            name = f"setting_{len(bindings)}"
            bindings[name] = value
            return name
        if isinstance(value, list | tuple):
            return [convert(v) for v in value]
        return value

    structure = copy.deepcopy(definition.structure)
    if definition.kind == "column":
        for group, field in (
            ("specs", "value"),
            ("side_draws", "fraction"),
            ("stage_duties", "duty"),
        ):
            for item in structure[group]:
                item[field] = convert(item[field])
    return replace(
        definition,
        name="unit",
        inlets=tuple(f"inlet_{i}" for i in range(len(definition.inlets))),
        outlets=tuple(f"outlet_{i}" for i in range(len(definition.outlets))),
        settings={k: convert(v) for k, v in definition.settings.items()},
        structure=structure,
    ), bindings


def _typed_arrays(tree: Any) -> Any:
    """Keep dtypes while removing weak scalar types from numerical call boundaries."""
    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.asarray(value).dtype), tree
    )


@dataclass(frozen=True)
class SolverOptions:
    """Reproducible numerical choices; no solver silently falls back to another backend."""

    backend: str = "sequential"
    recycle_method: str = "wegstein"
    tolerance: float = 1e-9
    max_iterations: int = 100
    specification_iterations: int = 40
    column_solver: str = "block"
    eo_jacobian: str = "colored"

    def __post_init__(self) -> None:
        if self.backend not in ("sequential", "eo"):
            raise ValueError("backend must be sequential or eo")
        if self.recycle_method not in ("wegstein", "broyden", "newton"):
            raise ValueError("unknown recycle method")
        if self.column_solver not in ("block", "dense"):
            raise ValueError("column_solver must be block or dense")
        if self.eo_jacobian not in ("colored", "dense"):
            raise ValueError("eo_jacobian must be colored or dense")
        if number(self.tolerance, "tolerance") <= 0:
            raise ValueError("tolerance must be positive")
        for n in (self.max_iterations, self.specification_iterations):
            if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 10000:
                raise ValueError("iteration caps must be integers from zero to 10000")


class CaseEvaluation(NamedTuple):
    """JAX pytree of numerical results in SI, before expensive host acceptance audits."""

    streams: dict[str, Stream]
    units: dict[str, UnitEvaluation]
    parameters: dict[str, Any]
    reports: dict[str, SolveReport]
    plant: dict[str, Any]
    equipment: dict[str, Any]
    metrics: dict[str, Any]


class CaseInitialization(NamedTuple):
    """Detached recycle and column seeds for a numerical study evaluation."""

    streams: dict[str, Stream]
    units: dict[str, dict[str, Any]]


class CaseRunner:
    """Compile a validated case once, then run, differentiate, or study it.

    ``evaluate`` accepts SI parameter values and returns a differentiable pytree.
    It is a numerical kernel, not a physical-acceptance boundary. Use ``run``
    for an audited, serializable result. Host overrides always require explicit
    quantities. Design-spec manipulated parameters are solved within their bounds.
    """

    def __init__(
        self,
        case: ProcessCase,
        *,
        options: SolverOptions | None = None,
        policy: AcceptancePolicy = DEFAULT_POLICY,
    ) -> None:
        self.case = ProcessCase.from_dict(case.to_dict())
        self.document = self.case.to_dict()
        self.options = options or SolverOptions()
        self.policy = policy
        self.parameters = self.case.parameters
        self.defaults = {k: jnp.asarray(p.value) for k, p in self.parameters.items()}
        self.feeds = parse_feeds(self.document["feeds"], self.case.components, self.parameters)
        self.units = parse_units(self.document["units"], self.case.components, self.parameters)
        self.package, self.qualification = build_package(self.document)
        if self.document["property_package"]["method"] == "iapws" and any(
            u.kind == "stoichiometric_reactor" for u in self.units
        ):
            raise CaseValidationError(
                "units",
                "reaction formation enthalpies aren't compatible with the reference-fluid datum",
            )
        self._sequential = CaseFlowsheet()
        self._eo = EOFlowsheet(model=self.package, jacobian_mode=self.options.eo_jacobian)
        self._unit_kernels: dict[str, Any] = {}
        self._unit_templates: dict[str, Any] = {}
        for feed in self.feeds:
            stream = feed.build(self.case.components, self.defaults, self.package)
            self._sequential.feed(feed.name, stream)
            self._eo.feed(feed.name, stream)
        for definition in self.units:

            def make_unit(d: Any) -> Any:
                # Stage the complete registered unit, including feed-property
                # preparation and retained outputs. Keeping every operating
                # value dynamic lets later design points reuse these kernels.
                template, bindings = _unit_template(d)
                key = canonical_json(asdict(template))
                if key not in self._unit_templates:

                    @jax.jit
                    def compiled(inputs: Any, parameters: Any, package: Any, guess: Any) -> Any:
                        return evaluate_unit(
                            template,
                            inputs,
                            parameters,
                            package,
                            guess=guess,
                            column_solver=self.options.column_solver,
                        )

                    self._unit_templates[key] = compiled
                compiled = self._unit_templates[key]

                def kernel(inputs: Any, parameters: Any, package: Any, guess: Any) -> Any:
                    local = {k: resolve_value(v, parameters) for k, v in bindings.items()}
                    return compiled(
                        _typed_arrays(inputs), _typed_arrays(local), package, _typed_arrays(guess)
                    )

                self._unit_kernels[d.name] = kernel

                def unit(*args: Any) -> Any:
                    params, pkg, guesses = args[-1]
                    return kernel(tuple(args[:-1]), params, pkg, guesses.get(d.name)).outlets

                return unit

            self._sequential.unit(
                definition.name,
                make_unit(definition),
                inputs=definition.inlets,
                outputs=definition.outlets,
            )
            self._eo.add(
                RegisteredBlock(
                    definition.inlets,
                    definition.outlets,
                    definition,
                    self._unit_kernels[definition.name],
                )
            )
        self._sequential.freeze_maps()
        consumed = {s for u in self.units for s in u.inlets}
        self.products = tuple(s for u in self.units for s in u.outlets if s not in consumed)
        self._specs = self.document["specifications"]
        self.manipulated = tuple(s["parameter"] for s in self._specs)
        lower = jnp.asarray([self.parameters[k].lower for k in self.manipulated])
        span = jnp.asarray([self.parameters[k].upper for k in self.manipulated]) - lower
        targets, tolerances = [], []
        for spec in self._specs:
            dim = unit_for(self.document["metrics"][spec["metric"]]["unit"]).dimension
            targets.append(
                quantity(
                    spec["target"],
                    dim,
                    "target",
                    difference=self.document["metrics"][spec["metric"]]["unit"].startswith(
                        "delta_"
                    ),
                )
            )
            tolerances.append(
                quantity(spec["tolerance"], dim, "tolerance", difference=dim == TEMPERATURE)
            )
        target, tolerance = jnp.asarray(targets), jnp.asarray(tolerances)

        self._spec_lower, self._spec_span = lower, span
        self._spec_target, self._spec_tolerance = target, tolerance
        self._ready = False

    def diagnose_structure(self) -> dict[str, Any]:
        """Describe process topology and declared numerical structure without solving.

        Registered EO units retain nested column and flash solves. Their global
        incidence report describes stream connections, not an expanded MESH
        system. Structural matching doesn't imply accepted process physics.
        """
        return {
            "case_id": self.case.case_id,
            "solver": asdict(self.options),
            "unit_instances": len(self.units),
            "unit_templates": len(self._unit_templates),
            "partitions": [
                {"units": list(part.units), "cyclic": part.cyclic, "tears": list(part.tears)}
                for part in self._sequential.partition()
            ],
            "equation_oriented": self._eo.diagnose_structure(),
            "columns": {
                unit.name: {
                    "stages": unit.structure["n_stages"],
                    "components": len(self.case.components),
                    "stage_block_size": 2 * len(self.case.components) + 1,
                    "border_size": int(unit.structure["condenser"] is not None)
                    + int(unit.structure["reboiler"] is not None),
                    "linear_solver": self.options.column_solver,
                }
                for unit in self.units
                if unit.kind == "column"
            },
        }

    def parameter_values(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Convert explicit host quantities and validate the entire operating point."""
        values = {k: p.value for k, p in self.parameters.items()}
        for name, raw in (overrides or {}).items():
            if name not in self.parameters:
                raise CaseValidationError("parameters", f"unknown parameter {name!r}")
            p = self.parameters[name]
            values[name] = quantity(raw, p.dimension, "parameters." + name, difference=p.difference)
        self.validate_values(values)
        return {k: jnp.asarray(v) for k, v in values.items()}

    def validate_values(self, values: dict[str, Any]) -> None:
        """Check concrete SI values before a host solve or accepted study point."""
        if set(values) != set(self.parameters):
            raise CaseValidationError("parameters", "SI parameter keys must exactly match the case")
        concrete = {k: number(float(v), "parameters." + k) for k, v in values.items()}
        for k, p in self.parameters.items():
            p.check(concrete[k], "parameters." + k)
        validate_feed_values(self.feeds, concrete)
        validate_unit_values(self.units, concrete)
        validate_economic_values(parse_economics(self.document, self.parameters), concrete)

    def prepare(self) -> None:
        """Build EO plans with concrete defaults before tracing; repeated calls are cheap."""
        if not self._ready:
            if self.options.backend == "eo":
                self._eo._get_plan(
                    self.defaults, 3, self.options.tolerance, self.options.max_iterations
                )
            self._ready = True

    def initialization(self, run: Any) -> CaseInitialization:
        """Recover study seeds from an accepted run of this exact case revision.

        Seeds don't change ``run``: persisted candidates always start cold.
        Studies record the baseline identity and pass these detached values
        explicitly to ``evaluate``. No failed trial updates a shared seed.
        """
        from fugacio.sim.cases.results import CaseRun

        if not isinstance(run, CaseRun) or not run.accepted:
            raise ValueError("study initialization requires an accepted case run")
        data = run.to_dict()
        if data["case_id"] != self.case.case_id:
            raise ValueError("study initialization requires the same case revision")
        streams = {
            name: Stream(
                jnp.asarray(s["component_flow_mol_s"]),
                jnp.asarray(s["temperature_k"]),
                jnp.asarray(s["pressure_pa"]),
                self.case.components,
                None
                if s["vapor_component_flow_mol_s"] is None
                else jnp.asarray(s["vapor_component_flow_mol_s"]),
            )
            for name, s in data["streams"].items()
        }
        values = {k: v["value_si"] for k, v in data["parameters"].items()}
        units = {}
        for d in self.units:
            if d.kind != "column":
                continue
            saved = data["units"][d.name]
            p, q = saved["profiles_si"], saved["quantities_si"]
            traffic = {}
            for phase, composition in (("liquid", "x"), ("vapor", "y")):
                if "stage_" + phase in p:
                    traffic[phase] = jnp.asarray(p["stage_" + phase])
                    continue
                # Older saved runs contain totals including side draws.
                factor = jnp.ones(d.structure["n_stages"])
                for draw in d.structure["side_draws"]:
                    if draw["phase"] == phase:
                        factor = factor.at[draw["stage"] - 1].add(
                            resolve_value(draw["fraction"], values)
                        )
                traffic[phase] = (jnp.asarray(p[phase + "_flow"]) / factor)[:, None] * jnp.asarray(
                    p[composition]
                )
            units[d.name] = {
                **traffic,
                "t": jnp.asarray(p["t"]),
                "condenser_duty": jnp.asarray(q["condenser_duty"]),
                "reboiler_duty": jnp.asarray(q["reboiler_duty"]),
            }
        return jax.lax.stop_gradient(CaseInitialization(streams, units))

    def _process(
        self,
        parameters: dict[str, Any],
        package: Any,
        initialization: CaseInitialization | None,
    ) -> CaseEvaluation:
        feeds = {f.name: f.build(self.case.components, parameters, package) for f in self.feeds}
        guesses = {} if initialization is None else initialization.units
        stream_guess = None if initialization is None else initialization.streams
        if self.options.backend == "sequential":
            flow = copy.copy(self._sequential)
            flow.feeds = feeds
            solution = flow.solve_with_info(
                (parameters, package, guesses),
                guess=stream_guess,
                method=self.options.recycle_method,
                tol=self.options.tolerance,
                max_iter=self.options.max_iterations,
            )
            streams, reports = dict(solution.streams), dict(solution.reports)
        else:
            eo = copy.copy(self._eo)
            eo.feeds, eo.model = feeds, package
            eo_result = eo.solve(
                parameters,
                guess=stream_guess,
                sweeps=3,
                tol=self.options.tolerance,
                max_iter=self.options.max_iterations,
                check=False,
                warm_start=False,
            )
            streams, reports = dict(eo_result.streams), {"eo": eo_result.report}
        units: dict[str, UnitEvaluation] = {}
        for d in self.units:
            result = self._unit_kernels[d.name](
                tuple(streams[k] for k in d.inlets),
                parameters,
                package,
                guesses.get(d.name),
            )
            units[d.name] = result
            reports["unit:" + d.name] = result.report
            terms: list[Any] = []
            for name, expected in zip(d.outlets, result.outlets, strict=True):
                actual = streams[name]
                terms.extend(
                    (
                        (actual.n - expected.n) / jnp.maximum(jnp.sum(jnp.abs(expected.n)), 1),
                        jnp.atleast_1d((actual.t - expected.t) / 100),
                        jnp.atleast_1d((actual.p - expected.p) / jnp.maximum(expected.p, 1e5)),
                    )
                )
                # Restore branch inventories for mixtures after solving EO's PT
                # coordinates. Keep solved n/T/P so replay never conceals a mismatch.
                if self.options.backend == "eo" and len(self.case.components) > 1:
                    fraction = jnp.asarray(expected.vapor_n) / jnp.where(
                        expected.n > 0, expected.n, 1
                    )
                    vapor = jnp.where(
                        expected.phase_known,
                        # Transfer phase fractions onto the solved component
                        # flows. Clipping replayed absolute inventories against
                        # EO flows invents a trace second phase from roundoff.
                        jnp.clip(fraction, 0, 1) * jnp.maximum(actual.n, 0),
                        -jnp.ones_like(actual.n),
                    )
                    streams[name] = Stream(actual.n, actual.t, actual.p, actual.components, vapor)
            reports["closure:" + d.name] = residual_report(
                jnp.concatenate(terms), max(self.options.tolerance * 10, 1e-7)
            )
        plant = {
            k: sum((getattr(u, k) for u in units.values()), jnp.asarray(0.0))
            for k in ("heat", "work", "heating", "cooling")
        }
        plant.update(
            electricity=sum((jnp.maximum(u.work, 0) for u in units.values()), jnp.asarray(0.0)),
            recovered_power=sum(
                (jnp.maximum(-u.work, 0) for u in units.values()), jnp.asarray(0.0)
            ),
        )
        evaluation = CaseEvaluation(streams, units, parameters, reports, plant, {}, {})
        costs, equipment = evaluate_economics(evaluation, self.document, package, self.parameters)
        evaluation = evaluation._replace(plant={**plant, **costs}, equipment=equipment)
        return evaluation._replace(metrics=metric_values(evaluation, self.document, package))

    def evaluate(
        self,
        parameters: dict[str, Any] | None = None,
        *,
        package: Any = None,
        initialization: CaseInitialization | None = None,
    ) -> CaseEvaluation:
        """Evaluate in SI with implicit derivatives through recycles and bounded specs.

        Call ``prepare`` before an enclosing JAX transformation for the EO
        backend. Differentiating with respect to a manipulated parameter's
        initial guess gives zero; the converged specification determines it.
        """
        params = {**self.defaults, **(parameters or {})}
        pkg = self.package if package is None else package
        if not self._specs:
            return self._process(params, pkg, initialization)
        lower, span = self._spec_lower, self._spec_span
        target, tolerance = self._spec_target, self._spec_tolerance

        def residual(x: Any, theta: Any) -> Any:
            p, model, seed = theta
            p = {**p, **dict(zip(self.manipulated, lower + span * x, strict=True))}
            evaluation = self._process(p, model, seed)
            return (
                jnp.asarray([evaluation.metrics[s["metric"]] for s in self._specs]) - target
            ) / tolerance

        start = (jnp.asarray([params[k] for k in self.manipulated]) - lower) / span
        solved = newton_system_with_info(
            residual,
            start,
            (params, pkg, initialization),
            tol=0.1,
            max_iter=self.options.specification_iterations,
            lower=jnp.zeros_like(start),
            upper=jnp.ones_like(start),
        )
        final = {**params, **dict(zip(self.manipulated, lower + span * solved.value, strict=True))}
        evaluation = self._process(final, pkg, initialization)
        return evaluation._replace(reports={**evaluation.reports, "specifications": solved.report})

    def run(self, overrides: dict[str, Any] | None = None, *, check: bool = False) -> Any:
        """Solve and independently audit a case, retaining failures in a run artifact."""
        from fugacio.sim.cases.results import build_run

        values = self.parameter_values(overrides)
        self.prepare()
        evaluation = self.evaluate(values)
        result = build_run(self, evaluation, values, asdict(self.options), asdict(self.policy))
        if check:
            result.check()
        return result
