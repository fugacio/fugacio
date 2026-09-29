"""Canonical process connectivity and independently linearized unit equations.

A graph contains unit kernels and their ports. Both sequential and simultaneous
execution use this graph. Its implicit derivative assembles local unit blocks;
it never differentiates a complete recycle iteration or a nested plant solve.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from fugacio.sim.numerics import SparseJacobian, is_traced
from fugacio.sim.stream import Stream
from fugacio.thermo.linear import DenseJacobian
from fugacio.thermo.sparsity import SparsityPattern


@dataclass(frozen=True)
class ProcessUnit:
    """One physical kernel and its named material ports."""

    name: str
    fn: Callable[..., Any]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class Partition:
    """One block of the flowsheet's sequential-modular calculation order.

    Attributes:
        units: Unit names in evaluation order (a valid order once ``tears`` are
            treated as known).
        tears: Stream names torn to break this block's recycle loops (empty for an
            acyclic block).
    """

    units: tuple[str, ...]
    tears: tuple[str, ...]

    @property
    def cyclic(self) -> bool:
        """Whether the block contains a recycle."""
        return bool(self.tears)


# --------------------------------------------------------------------------- #
# Graph analysis
# --------------------------------------------------------------------------- #

Edge = tuple[int, int, str]  # (producer unit, consumer unit, stream name)


def connection_edges(
    feeds: Sequence[str],
    connections: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]],
    preferred_tears: Sequence[str] = (),
) -> list[Edge]:
    """Validate unbound port declarations using the canonical graph contract."""
    if len(set(feeds)) != len(feeds):
        raise ValueError("duplicate feed name")
    if len({name for name, _, _ in connections}) != len(connections):
        raise ValueError("duplicate unit name")
    producer: dict[str, int] = {}
    for i, (unit, _, outputs) in enumerate(connections):
        if not outputs:
            raise ValueError(f"unit {unit!r} needs at least one outlet")
        for name in outputs:
            if name in producer:
                raise ValueError(
                    f"duplicate producer: stream {name!r} is produced by both units "
                    f"{connections[producer[name]][0]!r} and {unit!r}"
                )
            if name in feeds:
                raise ValueError(
                    f"duplicate producer: stream {name!r} is both a feed and an output"
                )
            producer[name] = i
    edges = []
    for i, (unit, inputs, _) in enumerate(connections):
        for name in inputs:
            if name in producer:
                edges.append((producer[name], i, name))
            elif name not in feeds:
                raise ValueError(
                    f"unit {unit!r} consumes undefined stream {name!r}, "
                    "neither a feed nor an output"
                )
    if set(preferred_tears) - set(producer):
        raise ValueError("a designated tear must be a produced stream")
    return edges


def _strongly_connected(n: int, edges: Sequence[Edge]) -> list[list[int]]:
    """Tarjan's algorithm; components come out in reverse topological order."""
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for a, b, _ in edges:
        adjacency[a].append(b)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    counter = 0
    result: list[list[int]] = []

    def visit(v: int) -> None:
        nonlocal counter
        index[v] = low[v] = counter
        counter += 1
        stack.append(v)
        on_stack[v] = True
        for w in adjacency[v]:
            if index[w] < 0:
                visit(w)
                low[v] = min(low[v], low[w])
            elif on_stack[w]:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            result.append(comp)

    for v in range(n):
        if index[v] < 0:
            visit(v)
    return result


def _cyclic(nodes: Sequence[int], edges: Sequence[Edge]) -> bool:
    """Whether the subgraph on ``nodes`` with ``edges`` contains a cycle."""
    node_set = set(nodes)
    inner = [e for e in edges if e[0] in node_set and e[1] in node_set]
    if any(a == b for a, b, _ in inner):
        return True
    return any(len(c) > 1 for c in _strongly_connected(max(nodes) + 1, inner))


def _coupling(nodes: Sequence[int], edges: Sequence[Edge]) -> int:
    """Number of edges that still lie inside a recycle (score for tear selection)."""
    node_set = set(nodes)
    inner = [e for e in edges if e[0] in node_set and e[1] in node_set]
    comps = _strongly_connected(max(nodes) + 1, inner)
    comp_of = {v: i for i, c in enumerate(comps) for v in c}
    return sum(
        1
        for a, b, _ in inner
        if comp_of[a] == comp_of[b] and (a == b or len(comps[comp_of[a]]) > 1)
    )


def _topological(nodes: Sequence[int], edges: Sequence[Edge]) -> list[int]:
    """Kahn's algorithm on the acyclic subgraph; ties keep registration order."""
    node_set = set(nodes)
    indeg = dict.fromkeys(nodes, 0)
    succ: dict[int, list[int]] = {v: [] for v in nodes}
    for a, b, _ in edges:
        if a in node_set and b in node_set:
            indeg[b] += 1
            succ[a].append(b)
    ready = sorted(v for v in nodes if indeg[v] == 0)
    order: list[int] = []
    while ready:
        v = ready.pop(0)
        order.append(v)
        for w in succ[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
        ready.sort()
    if len(order) != len(nodes):  # pragma: no cover - guarded by the tear selection
        raise ValueError("internal error: graph still cyclic after tearing")
    return order


def _select_tears(
    nodes: Sequence[int], edges: Sequence[Edge], preferred: Sequence[str]
) -> tuple[str, ...]:
    """Greedy feedback-arc tear selection for one strongly connected block.

    User-designated streams (``preferred``) that lie inside the block are torn
    first. While the block is still cyclic, the candidate stream whose removal
    leaves the fewest edges inside recycle loops is torn, the classic "break the
    most loops per tear" heuristic. Ties go to a stream that flows *backwards*
    in registration order (units are usually registered in process order, so
    the back edge is the recycle the author has in mind), then to the stream
    carried by the most edges, then alphabetically.
    """
    node_set = set(nodes)
    inner = [e for e in edges if e[0] in node_set and e[1] in node_set]
    streams_inside = {name for _, _, name in inner}
    tears: list[str] = [s for s in preferred if s in streams_inside]

    def remaining(torn: Sequence[str]) -> list[Edge]:
        return [e for e in inner if e[2] not in torn]

    while _cyclic(nodes, remaining(tears)):
        candidates = sorted(streams_inside - set(tears))
        best: tuple[int, int, int, str] | None = None
        for s in candidates:
            after = remaining([*tears, s])
            score = _coupling(nodes, after)
            forward = 0 if any(a > b for a, b, name in inner if name == s) else 1
            multiplicity = -sum(1 for e in inner if e[2] == s)
            key = (score, forward, multiplicity, s)
            if best is None or key < best:
                best = key
        assert best is not None
        tears.append(best[3])
    return tuple(tears)


@dataclass(frozen=True)
class ProcessGraph:
    """Immutable connectivity shared by process construction and execution.

    Kernels receive input streams followed by a dynamic parameter pytree.
    Numerical values and property-package coefficients aren't graph metadata.
    """

    feeds: tuple[str, ...]
    units: tuple[ProcessUnit, ...]
    preferred_tears: tuple[str, ...] = ()

    def edges(self) -> list[Edge]:
        """Validate ownership and return declared stream dependencies."""
        return connection_edges(
            self.feeds,
            tuple((u.name, u.inputs, u.outputs) for u in self.units),
            self.preferred_tears,
        )

    def partition(self) -> list[Partition]:
        """Order strongly connected components and select their tear streams."""
        edges = self.edges()
        result = []
        for comp in reversed(_strongly_connected(len(self.units), edges)):
            nodes = sorted(comp)
            tears = (
                _select_tears(nodes, edges, self.preferred_tears) if _cyclic(nodes, edges) else ()
            )
            order = _topological(nodes, [e for e in edges if e[2] not in tears])
            result.append(Partition(tuple(self.units[i].name for i in order), tears))
        return result

    def compile(
        self, streams: Mapping[str, Stream], *, linear_solver: str = "sparse"
    ) -> CompiledGraph:
        """Bind component layouts without capturing operating values."""
        return CompiledGraph(self, streams, linear_solver=linear_solver)


def stream_vector(stream: Stream) -> Array:
    """Pack material, temperature, pressure, and resolved vapor inventory.

    Keeping both inventories preserves pure-fluid saturation quality without
    embedding an extra PH flash in every process connection.
    """
    return jnp.concatenate(
        (stream.n, jnp.atleast_1d(stream.t), jnp.atleast_1d(stream.p), jnp.asarray(stream.vapor_n))
    )


def vector_stream(vector: Array, components: tuple[str, ...]) -> Stream:
    """Reconstruct a stream from its material and phase coordinates."""
    n = len(components)
    return Stream(vector[:n], vector[n], vector[n + 1], components, vector[n + 2 :])


def unit_outlets(result: Any) -> tuple[Stream, ...]:
    """Read the shared unit-result contract, including plain stream functions."""
    if hasattr(result, "outlets"):
        return tuple(result.outlets)
    return tuple(result) if isinstance(result, tuple | list) else (result,)


class CompiledGraph:
    """Fixed process layout with local residual and derivative assembly.

    Each unit linearizes once at the current point. Input-coordinate directions
    are applied sequentially so a column's stage tangent isn't batched by the
    number of plant variables. Sparse storage follows the declared connections.
    Instances contain structure only and never cache numerical operating points.
    """

    def __init__(
        self, graph: ProcessGraph, streams: Mapping[str, Stream], *, linear_solver: str = "sparse"
    ) -> None:
        graph.edges()
        if linear_solver not in ("sparse", "dense"):
            raise ValueError("plant linear solver must be sparse or dense")
        self.graph, self.linear_solver = graph, linear_solver
        self.names = tuple(name for unit in graph.units for name in unit.outputs)
        self.components = {name: streams[name].components for name in self.names}
        self.slices: dict[str, slice] = {}
        offset = 0
        for name in self.names:
            width = 2 * len(self.components[name]) + 2
            self.slices[name] = slice(offset, offset + width)
            offset += width
        self.size = offset
        self.inputs = tuple(
            tuple(dict.fromkeys(name for name in unit.inputs if name in self.slices))
            for unit in graph.units
        )
        self.input_indices = tuple(self.indices(names) for names in self.inputs)
        self.output_indices = tuple(self.indices(unit.outputs) for unit in graph.units)
        self.rows = tuple(range(offset)) + tuple(
            row
            for inputs, outputs in zip(self.input_indices, self.output_indices, strict=True)
            for row in outputs
            for _ in inputs
        )
        self.columns = tuple(range(offset)) + tuple(
            column
            for inputs, outputs in zip(self.input_indices, self.output_indices, strict=True)
            for _ in outputs
            for column in inputs
        )

        self._linearizers: dict[tuple[int, ...], tuple[Any, ...]] = {}

    def indices(self, names: Sequence[str]) -> tuple[int, ...]:
        """Scalar coordinates of a sequence of named internal streams."""
        return tuple(
            i for name in names for i in range(self.slices[name].start, self.slices[name].stop)
        )

    def pack(self, streams: Mapping[str, Stream]) -> Array:
        """Collect internal stream coordinates in deterministic unit order."""
        return jnp.concatenate([stream_vector(streams[name]) for name in self.names])

    def unpack(self, value: Array, feeds: Mapping[str, Stream]) -> dict[str, Stream]:
        """Restore all streams, retaining the supplied differentiable feeds."""
        return {
            **feeds,
            **{
                name: vector_stream(value[self.slices[name]], self.components[name])
                for name in self.names
            },
        }

    def _local(self, index: int, local: Array, parameters: Any) -> Array:
        theta, feeds = parameters
        inputs = dict(feeds)
        start = 0
        for name in self.inputs[index]:
            width = self.slices[name].stop - self.slices[name].start
            inputs[name] = vector_stream(local[start : start + width], self.components[name])
            start += width
        unit = self.graph.units[index]
        outputs = unit_outlets(unit.fn(*(inputs[name] for name in unit.inputs), theta))
        if len(outputs) != len(unit.outputs):
            raise ValueError(f"unit {unit.name!r} returned an incorrect outlet count")
        return jnp.concatenate([stream_vector(stream) for stream in outputs])

    def residual(self, value: Array, parameters: Any) -> Array:
        """The connection equations, evaluated through independent unit kernels."""
        predicted = [
            self._local(i, value[jnp.asarray(indices, dtype=int)], parameters)
            for i, indices in enumerate(self.input_indices)
        ]
        return value - jnp.concatenate(predicted)

    def local_linearization(
        self, value: Array, parameters: Any, direction: Any = None
    ) -> tuple[Any, Array]:
        """Assemble unit Jacobians and a parameter direction from shared local data."""
        active, dots = _active_direction(direction)
        if active not in self._linearizers:
            self._linearizers[active] = tuple(
                _local_linearizer(partial(self._local, i), active)
                for i in range(len(self.graph.units))
            )
        blocks, rhs = [jnp.ones(self.size, dtype=value.dtype)], []
        for indices, kernel in zip(self.input_indices, self._linearizers[active], strict=True):
            local = value[jnp.asarray(indices, dtype=int)]
            block, parameter_rhs = kernel(local, parameters, dots)
            if indices:
                blocks.append(-block.ravel())
            rhs.append(-parameter_rhs)
        sparse = SparseJacobian(jnp.concatenate(blocks), self.rows, self.columns, self.size)
        return sparse, jnp.concatenate(rhs)

    def linearize(self, value: Array, parameters: Any, direction: Any = None) -> tuple[Any, Array]:
        """Select sparse or explicit dense algebra over the same local assembly."""
        sparse, rhs = self.local_linearization(value, parameters, direction)
        matrix = sparse if self.linear_solver == "sparse" else DenseJacobian(sparse.to_dense())
        return matrix, rhs

    def attach(self, value: Array, parameters: Any, valid: Array) -> Array:
        """Attach assembled implicit derivatives to a detached converged state."""
        return _graph_solution(self, value, parameters, valid)

    def diagnose(self) -> dict[str, Any]:
        """Describe actual storage, scalar labels, and graph incidence."""
        incidence: list[set[int]] = [set() for _ in range(self.size)]
        for row, column in zip(self.rows, self.columns, strict=True):
            incidence[row].add(column)
        rows = tuple(tuple(sorted(columns)) for columns in incidence)
        labels = []
        for name in self.names:
            c = self.components[name]
            labels.extend(
                [
                    *(f"{name}:n[{x}]" for x in c),
                    f"{name}:temperature",
                    f"{name}:pressure",
                    *(f"{name}:vapor_n[{x}]" for x in c),
                ]
            )
        return {
            **SparsityPattern(self.size, rows).diagnose(equations=labels, variables=labels),
            "linear_solver": self.linear_solver,
            "stored_coefficients": len(self.rows),
            "local_input_directions": sum(map(len, self.input_indices)),
            "largest_unit_input": max(map(len, self.input_indices), default=0),
            "equation_labels": labels,
        }


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _graph_solution(graph: Any, value: Array, parameters: Any, valid: Array) -> Array:
    return value


@partial(_graph_solution.defjvp, symbolic_zeros=True)
def _graph_jvp(graph: Any, primals: Any, tangents: Any) -> tuple[Array, Array]:
    value, parameters, valid = primals
    _, direction, _ = tangents
    root = _graph_solution(graph, value, parameters, valid)
    matrix, rhs = graph.linearize(root, parameters, direction)
    tangent = matrix.solve(-rhs)
    return root, tangent * jnp.where(valid, 1.0, jnp.nan)


class SpecifiedGraph:
    """Connection equations with a small border of bounded design variables.

    ``bind(aux, parameters)`` supplies the graph's dynamic parameters and feeds.
    ``equations(streams, bound)`` evaluates specification residuals at that
    state. It mustn't solve the process. Unit physics remains in the graph;
    design metrics add rows and manipulated variables add columns.
    """

    def __init__(
        self,
        graph: CompiledGraph,
        count: int,
        bind: Callable[..., Any],
        equations: Callable[..., Array],
    ) -> None:
        self.graph, self.count = graph, count
        self.bind, self.equations = bind, equations
        self.size = graph.size + count
        self.rows = (
            graph.rows
            + tuple(i for i in range(graph.size) for _ in range(count))
            + tuple(i for i in range(graph.size, self.size) for _ in range(self.size))
        )
        self.columns = (
            graph.columns
            + tuple(j for _ in range(graph.size) for j in range(graph.size, self.size))
            + tuple(j for _ in range(count) for j in range(self.size))
        )

    def specification_residual(self, value: Array, parameters: Any) -> Array:
        """Evaluate the border equations using the current connection state."""
        bound = self.bind(value[self.graph.size :], parameters)
        streams = self.graph.unpack(value[: self.graph.size], bound[1])
        return self.equations(streams, bound)

    def residual(self, value: Array, parameters: Any) -> Array:
        """Evaluate one square process and design-specification system."""
        bound = self.bind(value[self.graph.size :], parameters)
        return jnp.concatenate(
            (
                self.graph.residual(value[: self.graph.size], bound),
                self.specification_residual(value, parameters),
            )
        )

    def linearize(self, value: Array, parameters: Any, direction: Any = None) -> tuple[Any, Array]:
        """Assemble local unit blocks and the small specification border."""
        core, auxiliary = value[: self.graph.size], value[self.graph.size :]
        bound = self.bind(auxiliary, parameters)
        matrix, _ = self.graph.local_linearization(core, bound)
        columns = jax.jacfwd(lambda aux: self.graph.residual(core, self.bind(aux, parameters)))(
            auxiliary
        )
        rows = jax.jacrev(self.specification_residual)(value, parameters)
        sparse = SparseJacobian(
            jnp.concatenate((matrix.values, columns.ravel(), rows.ravel())),
            self.rows,
            self.columns,
            self.size,
        )
        rhs = jnp.zeros_like(value)
        if direction is not None:
            leaves, tree = jax.tree_util.tree_flatten(parameters)
            active, dots = _active_direction(direction)

            def evaluate(selected: Any) -> Array:
                values = list(leaves)
                for index, item in zip(active, selected, strict=True):
                    values[index] = item
                return self.residual(value, jax.tree_util.tree_unflatten(tree, values))

            if active:
                _, rhs = jax.jvp(evaluate, (tuple(leaves[i] for i in active),), (dots,))
        result = (
            sparse if self.graph.linear_solver == "sparse" else DenseJacobian(sparse.to_dense())
        )
        return result, rhs

    def diagnose(self, specifications: Sequence[str]) -> dict[str, Any]:
        """Describe the augmented state, including every specification border."""
        if len(specifications) != self.count:
            raise ValueError("specification labels don't match the border size")
        labels = self.graph.diagnose()["equation_labels"]
        equations = [*labels, *("specification:" + name for name in specifications)]
        variables = [*labels, *("manipulated:" + name for name in specifications)]
        incidence: list[set[int]] = [set() for _ in range(self.size)]
        for row, column in zip(self.rows, self.columns, strict=True):
            incidence[row].add(column)
        rows = tuple(tuple(sorted(columns)) for columns in incidence)
        return {
            **SparsityPattern(self.size, rows).diagnose(equations=equations, variables=variables),
            "linear_solver": self.graph.linear_solver,
            "stored_coefficients": len(self.rows),
            "specification_variables": self.count,
            "equation_labels": equations,
        }

    def attach(self, value: Array, parameters: Any, valid: Array) -> Array:
        """Attach the full bounded specification system's implicit derivative."""
        return _graph_solution(self, value, parameters, valid)


class ResidualGraph:
    """Locally assembled equations for custom residual units and specifications.

    Each block receives only its declared scalar inputs and the dynamic process
    parameters. Incidence is a structural contract, never inferred from values.
    Compiled unit boundaries are retained in primal and derivative evaluation.
    """

    def __init__(
        self,
        size: int,
        blocks: Sequence[tuple[tuple[int, ...], int, Callable[..., Array]]],
        *,
        linear_solver: str = "sparse",
    ) -> None:
        if linear_solver not in ("sparse", "dense"):
            raise ValueError("linear_solver must be sparse or dense")
        self.size, self.linear_solver = size, linear_solver
        self.blocks = tuple((indices, count, jax.jit(fn)) for indices, count, fn in blocks)
        self.rows: tuple[int, ...] = ()
        self.columns: tuple[int, ...] = ()
        offset = 0
        for indices, count, _ in self.blocks:
            self.rows += tuple(i for i in range(offset, offset + count) for _ in indices)
            self.columns += indices * count
            offset += count
        if offset != size:
            raise ValueError(f"residual graph isn't square: {size} unknowns and {offset} equations")
        self._linearizers: dict[tuple[int, ...], tuple[Any, ...]] = {}

    def residual(self, value: Array, parameters: Any) -> Array:
        """Evaluate independently compiled residual blocks."""
        return jnp.concatenate(
            [
                fn(value[jnp.asarray(indices, dtype=int)], parameters)
                for indices, _, fn in self.blocks
            ]
        )

    def linearize(self, value: Array, parameters: Any, direction: Any = None) -> tuple[Any, Array]:
        """Assemble sparse local Jacobians and one parameter right-hand side."""
        active, dots = _active_direction(direction)
        if active not in self._linearizers:
            self._linearizers[active] = tuple(
                _local_linearizer(fn, active) for _, _, fn in self.blocks
            )
        values, rhs = [], []
        for (indices, _, _), kernel in zip(self.blocks, self._linearizers[active], strict=True):
            block, parameter_rhs = kernel(value[jnp.asarray(indices, dtype=int)], parameters, dots)
            values.append(block.ravel())
            rhs.append(parameter_rhs)
        sparse = SparseJacobian(jnp.concatenate(values), self.rows, self.columns, self.size)
        matrix = sparse if self.linear_solver == "sparse" else DenseJacobian(sparse.to_dense())
        return matrix, jnp.concatenate(rhs)

    def attach(self, value: Array, parameters: Any, valid: Array) -> Array:
        """Attach implicit derivatives assembled from the same residual blocks."""
        return _graph_solution(self, value, parameters, valid)


def _active_direction(direction: Any) -> tuple[tuple[int, ...], tuple[Any, ...]]:
    if direction is None:
        return (), ()
    dots = jax.tree_util.tree_leaves(direction)
    active = tuple(
        i
        for i, dot in enumerate(dots)
        if not isinstance(dot, jax.custom_derivatives.SymbolicZero)
        and getattr(dot, "dtype", None) != jax.dtypes.float0
    )
    return active, tuple(dots[i] for i in active)


def _local_linearizer(function: Any, active: tuple[int, ...]) -> Any:
    """Build one unit's state matrix and parameter RHS from one linearization.

    Partial evaluation retains primal residual data outside the parameter
    direction. Repeated JVPs and VJPs reuse those data, including nested unit
    factors. No direction is batched through a nested thermodynamic solver.
    """

    def linearize(local: Array, parameters: Any, direction: Any) -> tuple[Array, Array]:
        leaves, tree = jax.tree_util.tree_flatten(parameters)
        selected = tuple(leaves[i] for i in active)

        def evaluate(x: Array, selected: Any) -> Array:
            values = list(leaves)
            for index, item in zip(active, selected, strict=True):
                values[index] = item
            return function(x, jax.tree_util.tree_unflatten(tree, values))

        primal, push = jax.linearize(evaluate, local, selected)
        zeros = tuple(jnp.zeros_like(leaf) for leaf in selected)
        matrix = (
            _state_matrix(push, jnp.zeros_like(local), zeros)
            if local.size
            else jnp.empty((primal.size, 0), dtype=primal.dtype)
        )
        rhs = push(jnp.zeros_like(local), direction) if active else jnp.zeros_like(primal)
        return matrix, rhs

    return linearize


def _state_matrix(push: Any, local: Array, selected: Any) -> Array:
    """Apply directions through the retained unit kernels without an extra JIT.

    A linearization's callback identity changes with the point, even when its
    program is identical. Jitting that callback as static metadata recompiles
    the complete tangent wrapper at every point. Concrete assembly instead
    reuses the unit's compiled tangent kernels. An enclosing trace stages a
    sequential map, retaining bounded direction storage and higher derivatives.
    """
    basis = jnp.eye(local.size, dtype=local.dtype)
    if is_traced((push, local, selected)):
        return jax.lax.map(lambda v: push(v, selected), basis).T
    return jnp.stack([push(v, selected) for v in basis]).T
