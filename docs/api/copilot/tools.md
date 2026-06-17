# Tool registry

The JSON-schema tool registry that exposes the differentiable engine to a
language model: each `ToolSpec` wraps an engine call with a validated schema, so
the model can size, cost, flash, and optimize without touching Python. The tools
are plain Python (floats and lists, not JAX arrays), so they serialize cleanly
into a function-calling loop.

::: fugacio.copilot.tools

## Domain tool packs

The default registry is assembled from the core tools plus three domain packs.
Each factory returns the `ToolSpec` list for its area.

::: fugacio.copilot.dynamics_tools

::: fugacio.copilot.integration_tools

::: fugacio.copilot.mpc_tools
