"""JAX implementation of the neural bipartite matching algorithm."""

from __future__ import annotations

from typing import Any

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:  # pragma: no cover - exercised via import guard
    raise ImportError(
        "The jax backend requires JAX. Install with "
        "`pip install neural-bipartite-matching[jax]`."
    ) from e

from ._core import MatchingConfig, MatchingResult, RellocType


def _row_normalize(A: jnp.ndarray, budget: jnp.ndarray) -> jnp.ndarray:
    row_sum = A.sum(axis=1, keepdims=True)
    safe = jnp.where(row_sum > 0, row_sum, 1.0)
    scaled = A * (budget / safe)
    return jnp.where(row_sum > 0, scaled, A)


def _competition(A: jnp.ndarray, f: jnp.ndarray, alpha: float) -> jnp.ndarray:
    total = (f[:, None] * A).sum(axis=0, keepdims=True)  # (1, M)
    competitors = total - f[:, None] * A
    return jnp.maximum(A - alpha * competitors, 0.0)


def _reallocate(
    A: jnp.ndarray,
    f: jnp.ndarray,
    R: float,
    beta: float,
    realloc: RellocType,
) -> jnp.ndarray:
    budget = (R / f)[:, None]
    row_sum = A.sum(axis=1, keepdims=True)
    retracted = jnp.maximum(budget - row_sum, 0.0)
    if realloc == "multiplicative":
        safe = jnp.where(row_sum > 0, row_sum, 1.0)
        frac = jnp.where(row_sum > 0, A / safe, 0.0)
        return A + beta * retracted * frac
    if realloc == "constant":
        active = A > 0
        degree = jnp.maximum(active.sum(axis=1, keepdims=True), 1).astype(A.dtype)
        inc = beta * retracted / degree
        return jnp.where(active, A + inc, A)
    raise ValueError(f"Unknown realloc type: {realloc!r}")


def _resolve_triangles(A: jnp.ndarray) -> jnp.ndarray:
    active = A > 0
    masked = jnp.where(active, A, -jnp.inf)
    winners = jnp.argmax(masked, axis=0)  # (M,)
    any_active = active.any(axis=0)
    keep = jax.nn.one_hot(winners, A.shape[0], axis=0, dtype=jnp.bool_)
    keep = keep & any_active[None, :]
    return jnp.where(keep, A, 0.0)


def _extract_matching(A: jnp.ndarray) -> jnp.ndarray:
    active = A > 0
    masked = jnp.where(active, A, -jnp.inf)
    idx = jnp.argmax(masked, axis=0)
    any_active = active.any(axis=0)
    return jnp.where(any_active, idx, -1).astype(jnp.int32)


def _step(A, f, alpha, R, beta, realloc):
    A_prime = _competition(A, f, alpha)
    return _reallocate(A_prime, f, R, beta, realloc)


def neural_match(
    A: Any,
    f: Any | None = None,
    config: MatchingConfig | None = None,
    **kwargs: Any,
) -> MatchingResult:
    """Run the neural bipartite matching algorithm (JAX backend).

    See :func:`neural_bipartite_matching.torch_backend.neural_match` for
    the full parameter documentation. This backend otherwise mirrors it.
    """
    if config is None:
        config = MatchingConfig(**kwargs)
    elif kwargs:
        config = MatchingConfig(**{**config.__dict__, **kwargs})

    A_j = jnp.asarray(A, dtype=jnp.float32 if jnp.asarray(A).dtype.kind != "f" else None)
    A_j = jnp.asarray(A_j, dtype=jnp.result_type(A_j, jnp.float32))
    if (A_j < 0).any():
        raise ValueError("Input weight matrix must be non-negative.")
    N, M = A_j.shape
    if N > M:
        raise ValueError(f"Expected N <= M, got N={N}, M={M}.")

    if f is None:
        f_j = jnp.ones(N, dtype=A_j.dtype)
    else:
        f_j = jnp.asarray(f, dtype=A_j.dtype).reshape(-1)
        if f_j.shape[0] != N:
            raise ValueError(f"f must have length N={N}, got {f_j.shape[0]}.")
        if (f_j <= 0).any():
            raise ValueError("Firing rates f must be strictly positive.")

    budget = (config.R / f_j)[:, None]
    if config.normalize_input:
        A_j = _row_normalize(A_j, budget)

    # JIT a single step; python-level loop for convergence check.
    step = jax.jit(
        lambda X: _step(X, f_j, config.alpha, config.R, config.beta, config.realloc)
    )

    converged = False
    iterations = 0
    for t in range(1, config.max_iter + 1):
        iterations = t
        prev = A_j
        A_j = step(prev)
        diff = float(jnp.max(jnp.abs(A_j - prev)))
        support_change = bool(jnp.any((A_j > 0) != (prev > 0)))
        if diff < config.tol and not support_change:
            converged = True
            break

    if config.resolve:
        A_j = _resolve_triangles(A_j)

    matching = _extract_matching(A_j)
    return MatchingResult(A=A_j, matching=matching, iterations=iterations, converged=converged)
