"""Differentiable execution of portable process cases with retained unit results."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

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
from fugacio.sim.flowsheet import Flowsheet
from fugacio.sim.graph import SpecifiedGraph
from fugacio.sim.numerics import newton_iterations
from fugacio.sim.stream import Stream
from fugacio.thermo.acceptance import DEFAULT_POLICY, AcceptancePolicy
from fugacio.thermo.diagnostics import SolveReport, residual_report
from fugacio.thermo.implicit import gate_tree


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
    if "reactions" in structure:
        reaction = structure["reactions"]
        reaction["reference_concentration"] = convert(reaction["reference_concentration"])
        for item in reaction["reactions"]:
            for field, value in item.get("rate", {}).items():
                if field != "detailed_balance":
                    item["rate"][field] = convert(value)
    return replace(
        definition,
        name="unit",
        inlets=tuple(f"inlet_{i}" for i in range(len(definition.inlets))),
        outlets=tuple(f"outlet_{i}" for i in range(len(definition.outlets))),
        settings={k: convert(v) for k, v in definition.settings.items()},
        structure=structure,
    ), bindings


#: Compiled unit templates shared by every runner in this process, keyed by the
#: canonical template document and the column linear solver. Literal settings
#: are part of the template document, so templates that differ in any fixed
#: value never share a kernel. The package enters each kernel as a dynamic
#: argument; its structure is part of JAX's own compilation key.
_TEMPLATE_CACHE: OrderedDict[tuple[str, str], Any] = OrderedDict()
_TEMPLATE_CACHE_SIZE = 256


def _compiled_template(template: UnitDefinition, column_solver: str) -> Any:
    """The process-wide compiled kernel of one unit template (bounded LRU)."""
    key = (canonical_json(asdict(template)), column_solver)
    if key in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE.move_to_end(key)
        return _TEMPLATE_CACHE[key]

    def compiled(inputs: Any, parameters: Any, package: Any, guess: Any) -> Any:
        return evaluate_unit(
            template, inputs, parameters, package, guess=guess, column_solver=column_solver
        )

    # Columns and exchangers already compile their implicit and property kernels.
    # Wrapping their preparation and retained profiles in another JIT fuses
    # those boundaries back together and duplicates their nonlinear program
    # when JAX prepares the unit's implicit derivative.
    kernel = compiled if template.kind in ("column", "heat_exchanger") else jax.jit(compiled)
    _TEMPLATE_CACHE[key] = kernel
    while len(_TEMPLATE_CACHE) > _TEMPLATE_CACHE_SIZE:
        _TEMPLATE_CACHE.popitem(last=False)
    return kernel


def _typed_arrays(tree: Any) -> Any:
    """Keep dtypes while removing weak scalar types from numerical call boundaries."""
    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.asarray(value).dtype), tree
    )


@dataclass(frozen=True)
class SolverOptions:
    """Reproducible numerical choices; no solver silently falls back to another backend."""

    backend: str = "sequential"
    recycle_method: str = "broyden"
    tolerance: float = 1e-9
    max_iterations: int = 100
    specification_iterations: int = 40
    column_solver: str = "block"
    plant_solver: str = "sparse"

    def __post_init__(self) -> None:
        if self.backend not in ("sequential", "eo"):
            raise ValueError("backend must be sequential or eo")
        if self.recycle_method not in ("wegstein", "broyden", "newton"):
            raise ValueError("unknown recycle method")
        if self.column_solver not in ("block", "dense"):
            raise ValueError("column_solver must be block or dense")
        if self.plant_solver not in ("sparse", "dense"):
            raise ValueError("plant_solver must be sparse or dense")
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
        self.units = parse_units(
            self.document["units"],
            self.case.components,
            self.parameters,
            self.document.get("reaction_sets", {}),
        )
        self.package, self.qualification = build_package(self.document)
        if self.document["property_package"]["method"] == "iapws" and any(
            u.kind == "stoichiometric_reactor" or "reactions" in u.structure for u in self.units
        ):
            raise CaseValidationError(
                "units",
                "reaction formation enthalpies aren't compatible with the reference-fluid datum",
            )
        self._flow = Flowsheet()
        self._unit_kernels: dict[str, Any] = {}
        self._unit_templates: dict[str, Any] = {}
        for feed in self.feeds:
            stream = feed.build(self.case.components, self.defaults, self.package)
            self._flow.feed(feed.name, stream)
        for definition in self.units:

            def make_unit(d: Any) -> Any:
                # Stage the complete registered unit, including feed-property
                # preparation and retained outputs. Keeping every operating
                # value dynamic lets later design points reuse these kernels.
                template, bindings = _unit_template(d)
                compiled = _compiled_template(template, self.options.column_solver)
                self._unit_templates[canonical_json(asdict(template))] = compiled

                def kernel(inputs: Any, parameters: Any, package: Any, guess: Any) -> Any:
                    local = {k: resolve_value(v, parameters) for k, v in bindings.items()}
                    if d.kind == "column":
                        # Cold and accepted-profile starts share one input tree.
                        # A dynamic selector avoids retaining a second compiled
                        # MESH solver when a study starts from its audited run.
                        n, c = d.structure["n_stages"], len(inputs[0].components)
                        guess = {
                            "liquid": jnp.zeros((n, c)),
                            "vapor": jnp.zeros((n, c)),
                            "t": jnp.zeros(n),
                            "condenser_duty": jnp.asarray(0.0),
                            "reboiler_duty": jnp.asarray(0.0),
                            **(guess or {}),
                            "_use_profile": jnp.asarray(guess is not None),
                        }
                    return compiled(
                        _typed_arrays(inputs), _typed_arrays(local), package, _typed_arrays(guess)
                    )

                self._unit_kernels[d.name] = kernel

                def unit(*args: Any) -> Any:
                    params, pkg, guesses = args[-1]
                    return kernel(tuple(args[:-1]), params, pkg, guesses.get(d.name))

                return unit

            self._flow.unit(
                definition.name,
                make_unit(definition),
                inputs=definition.inlets,
                outputs=definition.outlets,
            )
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
        self._specified_graph: SpecifiedGraph | None = None

    def diagnose_structure(self) -> dict[str, Any]:
        """Describe process topology and declared numerical structure without solving.

        Incidence describes stream connections and specification borders.
        Columns retain their structured internal MESH solves. Structural
        matching doesn't imply accepted process physics.
        """
        return {
            "case_id": self.case.case_id,
            "solver": asdict(self.options),
            "unit_instances": len(self.units),
            "unit_templates": len(self._unit_templates),
            "partitions": [
                {"units": list(part.units), "cyclic": part.cyclic, "tears": list(part.tears)}
                for part in self._flow.partition()
            ],
            "process_graph": self._graph_diagnostics(),
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

    def _graph_diagnostics(self) -> dict[str, Any]:
        graph = self._compiled_graph()
        if not self._specs:
            return graph.diagnose()
        # Inspection only needs incidence, not numerical metric callbacks.
        system = SpecifiedGraph(
            graph, len(self._specs), lambda *args: None, lambda *args: jnp.empty(0)
        )
        return system.diagnose(self.manipulated)

    def _compiled_graph(self) -> Any:
        template = next(iter(self._flow.feeds.values()))
        streams = {name: template for unit in self.units for name in unit.outlets}
        return self._flow.compile(streams, linear_solver=self.options.plant_solver)

    def prepare(self) -> None:
        """Bind the shared graph layout without solving or capturing operating values."""
        self._compiled_graph()

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
        units = {}
        for d in self.units:
            if d.kind != "column":
                continue
            saved = data["units"][d.name]
            p, q = saved["profiles_si"], saved["quantities_si"]
            traffic = {phase: jnp.asarray(p["stage_" + phase]) for phase in ("liquid", "vapor")}
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
        flow = copy.copy(self._flow)
        flow.feeds = feeds
        solution = flow.solve_with_info(
            (parameters, package, guesses),
            guess=stream_guess,
            audit=False,
            retain_results=True,
            strategy="sequential" if self.options.backend == "sequential" else "simultaneous",
            linear_solver=self.options.plant_solver,
            method=self.options.recycle_method,
            tol=self.options.tolerance,
            max_iter=self.options.max_iterations,
        )
        return self._evaluate_streams(
            solution.streams,
            parameters,
            package,
            guesses,
            dict(solution.reports),
            solution.unit_results,
        )

    def _evaluate_streams(
        self,
        streams: dict[str, Stream],
        parameters: dict[str, Any],
        package: Any,
        guesses: Any,
        reports: dict[str, SolveReport],
        retained: dict[str, UnitEvaluation] | None = None,
    ) -> CaseEvaluation:
        """Retain unit and metric evidence at a supplied shared process state."""
        units: dict[str, UnitEvaluation] = {}
        for d in self.units:
            result = (
                retained[d.name]
                if retained is not None
                else self._unit_kernels[d.name](
                    tuple(streams[k] for k in d.inlets), parameters, package, guesses.get(d.name)
                )
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
                        (jnp.asarray(actual.vapor_n) - jnp.asarray(expected.vapor_n))
                        / jnp.maximum(jnp.sum(jnp.abs(expected.n)), 1),
                    )
                )
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

    def _specification_system(self) -> SpecifiedGraph:
        """Reuse specification callbacks without capturing an operating point."""
        if self._specified_graph is not None:
            return self._specified_graph
        lower, span = self._spec_lower, self._spec_span
        target, tolerance = self._spec_target, self._spec_tolerance

        def bind(auxiliary: Any, external: Any) -> Any:
            p, model, seeds = external
            p = {**p, **dict(zip(self.manipulated, lower + span * auxiliary, strict=True))}
            feeds = {f.name: f.build(self.case.components, p, model) for f in self.feeds}
            return (p, model, seeds), feeds

        def equations(streams: Any, bound: Any) -> Any:
            p, model, seeds = bound[0]
            evaluation = self._evaluate_streams(streams, p, model, seeds, {})
            return (
                (jnp.asarray([evaluation.metrics[s["metric"]] for s in self._specs]) - target)
                / tolerance
                * (self.options.tolerance / 0.1)
            )

        self._specified_graph = SpecifiedGraph(
            self._compiled_graph(), len(self._specs), bind, equations
        )
        return self._specified_graph

    def evaluate(
        self,
        parameters: dict[str, Any] | None = None,
        *,
        package: Any = None,
        initialization: CaseInitialization | None = None,
    ) -> CaseEvaluation:
        """Evaluate in SI with implicit derivatives through recycles and bounded specs.

        Both strategies use the same graph and local implicit derivatives.
        Differentiating with respect to a manipulated parameter's
        initial guess gives zero; the converged specification determines it.
        """
        params = {**self.defaults, **(parameters or {})}
        pkg = self.package if package is None else package
        if not self._specs:
            return self._process(params, pkg, initialization)
        lower, span = self._spec_lower, self._spec_span
        system = self._specification_system()
        graph = system.graph
        guesses = {} if initialization is None else initialization.units
        external = (params, pkg, guesses)
        # One ordinary process solution supplies an initialization. Subsequent
        # Newton trials evaluate local equations, never a nested plant solve.
        seed = self._process(*jax.lax.stop_gradient((params, pkg, initialization)))
        start = jnp.concatenate(
            (
                graph.pack(seed.streams),
                (jnp.asarray([params[k] for k in self.manipulated]) - lower) / span,
            )
        )
        scale = jnp.concatenate(
            (jnp.maximum(jnp.abs(start[: graph.size]), 1.0), jnp.ones(len(self._specs)))
        )
        solved = newton_iterations(
            system.residual,
            lambda x, p: system.linearize(x, p)[0],
            start,
            external,
            scale=scale,
            tolerance=self.options.tolerance,
            max_iterations=self.options.specification_iterations,
            lower=jnp.concatenate((jnp.full(graph.size, -jnp.inf), jnp.zeros(len(self._specs)))),
            upper=jnp.concatenate((jnp.full(graph.size, jnp.inf), jnp.ones(len(self._specs)))),
        )
        value = system.attach(solved.value, external, solved.report.converged)
        bound = system.bind(value[graph.size :], external)
        streams = graph.unpack(value[: graph.size], bound[1])
        final = bound[0][0]
        evaluation = self._evaluate_streams(
            streams, final, pkg, guesses, {"specifications": solved.report}
        )
        valid = jnp.all(jnp.asarray([r.converged for r in evaluation.reports.values()]))
        valid &= jnp.all(jnp.asarray([s.report.converged for s in streams.values()]))
        return gate_tree(evaluation, valid)

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
