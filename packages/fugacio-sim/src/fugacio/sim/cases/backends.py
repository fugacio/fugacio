"""Compile portable registered units into the existing flowsheet engines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import Array

from fugacio.sim.cases.registry import UnitDefinition, evaluate_unit
from fugacio.sim.eo.blocks import Block, Context
from fugacio.sim.flowsheet import Flowsheet, Partition
from fugacio.sim.properties import molar_enthalpy
from fugacio.sim.stream import Stream


class CaseFlowsheet(Flowsheet):
    """Fixed case topology with reusable maps and explicit dynamic feed inputs."""

    def freeze_maps(self) -> None:
        """Build callbacks before tracing; case definitions don't mutate afterward."""
        self._case_maps = {
            block: super(CaseFlowsheet, self)._block_map(block)
            for block in self.partition()
            if block.cyclic
        }

    def _block_map(self, block: Partition) -> Any:
        return self._case_maps[block]


@dataclass(frozen=True)
class RegisteredBlock(Block):
    """A procedural EO block delegating unit physics to the registered kernel.

    Outlet material, thermal, and pressure coordinates are global unknowns.
    Internal column/flash equations remain nested implicit solves. This adapter
    is deliberately distinct from a fully expanded simultaneous MESH system.
    Pure fluids use enthalpy coordinates to retain saturation-line quality.
    """

    definition: UnitDefinition
    kernel: Any = None

    def residual_dependencies(self, ctx: Context) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Declare the ports read by this registered procedural unit."""
        return self.inlets + self.outlets, ()

    def n_residuals(self, ctx: Context) -> int:
        """Return one material vector, thermal equation, and pressure equation per outlet."""
        return len(self.outlets) * (ctx.n_components + 2)

    def forward(
        self, streams: Mapping[str, Stream], params: Mapping[str, Any], ctx: Context
    ) -> dict[str, Stream]:
        """Seed outlets using the same unit kernel as the sequential engine."""
        inputs = tuple(streams[k] for k in self.inlets)
        result = (
            evaluate_unit(self.definition, inputs, dict(params), ctx.package)
            if self.kernel is None
            else self.kernel(inputs, dict(params), ctx.package, None)
        )
        return dict(zip(self.outlets, result.outlets, strict=True))

    def residuals(
        self,
        streams: Mapping[str, Stream],
        aux: Mapping[str, Array],
        params: Mapping[str, Any],
        ctx: Context,
    ) -> Array:
        """Compare global outlet unknowns with independently evaluated unit outputs."""
        predicted = self.forward(streams, params, ctx)
        terms: list[Any] = []
        for name, expected in predicted.items():
            actual = streams[name]
            if ctx.n_components == 1:
                h = ctx.enthalpy_coordinates.get(id(actual))
                if h is None:
                    h = molar_enthalpy(actual, model=ctx.package)
                thermal = (
                    h - molar_enthalpy(expected, model=ctx.package)
                ) / ctx.scales.enthalpy_molar
            else:
                thermal = (actual.t - expected.t) / ctx.scales.temperature
            terms.extend(
                (
                    (actual.n - expected.n) / ctx.scales.flow,
                    jnp.atleast_1d(thermal),
                    jnp.atleast_1d((actual.p - expected.p) / ctx.scales.pressure),
                )
            )
        return jnp.concatenate(terms)
