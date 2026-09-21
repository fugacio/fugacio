"""Opt-in persistent compilation cache for repeated process studies.

Every Fugacio kernel compiles once per process (see `fugacio.sim.units`), but a
fresh process compiles again. `enable_compilation_cache` points JAX's
persistent compilation cache at a directory so a second process (a CLI run, a
worker in a sweep, a notebook restart) loads the compiled executables instead.

The cache is keyed by the JAX version, the backend, and the exact program, so
a stale entry is never reused for a different computation. It's opt-in because
the directory grows with every distinct program compiled.
"""

from __future__ import annotations

from pathlib import Path

import jax


def enable_compilation_cache(directory: str | Path, *, min_compile_time_s: float = 0.0) -> Path:
    """Persist compiled JAX programs in ``directory`` for reuse across processes.

    Call it before the first computation when possible. Calling it later also
    works: the cache is re-initialized so the next compilation uses it.

    Args:
        directory: Cache directory. It's created if missing.
        min_compile_time_s: Only programs that took at least this long to
            compile are stored. The default, zero, stores every program, since
            process kernels are small individually but numerous.

    Returns:
        The resolved cache directory.
    """
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_time_s))
    # JAX decides once per process whether the cache is in use, at the first
    # compilation. Reset that decision so a late call still takes effect.
    try:
        from jax._src import compilation_cache
    except ImportError:  # pragma: no cover - private module moved
        return path
    reset = getattr(compilation_cache, "reset_cache", None)
    if callable(reset):
        reset()
    return path


__all__ = ["enable_compilation_cache"]
