"""Keep concrete solver orchestration outside compiled physical kernels."""

from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


def is_traced(tree: Any) -> bool:
    """Whether numerical inputs are being staged by a JAX transformation."""
    return any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(tree))


def while_loop(condition: Callable[..., Any], body: Callable[..., Any], state: Any) -> Any:
    """Use the same iteration eagerly and under an enclosing JAX trace.

    Test the predicate too: a condition may close over traced parameters even
    when its initial state is concrete. Implicit solvers detach their primal
    inputs before entering this loop, so derivatives never unroll iterations.
    """
    predicate = condition(state)
    if is_traced((state, predicate)):
        return jax.lax.while_loop(condition, body, state)
    while bool(predicate):
        state = body(state)
        predicate = condition(state)
    return state


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def primal_call(function: Callable[..., Any], arguments: Any) -> Any:
    """Execute a detached solver below the ambient derivative trace.

    A stop-gradient alone can still transform and recompile nested JIT calls
    under partial evaluation. This boundary reuses the ordinary primal kernels.
    The caller must attach the converged equation's derivative separately.
    """
    return function(*jax.lax.stop_gradient(arguments))


@primal_call.defjvp
def _primal_jvp(function: Callable[..., Any], primals: Any, tangents: Any) -> Any:
    value = primal_call(function, primals[0])

    def zero(x: Any) -> Any:
        dtype = jnp.asarray(x).dtype
        return jnp.zeros_like(
            x, dtype=dtype if jnp.issubdtype(dtype, jnp.inexact) else jax.dtypes.float0
        )

    zeros = jax.tree_util.tree_map(zero, value)
    return value, zeros
