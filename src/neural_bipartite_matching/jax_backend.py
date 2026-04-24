"""JAX implementation of the neural bipartite matching algorithm.

The entire iteration loop — including the convergence check, optional
resolution of multi-matched fibers, and extraction of the final
``matching`` vector — is compiled inside a single :func:`jax.jit`'ed
function driven by :func:`jax.lax.while_loop`. This avoids per-iteration
host/device synchronization and lets the whole call be composed with
further ``jit`` / ``vmap`` / ``pmap`` transforms from the caller.
"""

from __future__ import annotations

from functools import partial
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


def _step(A, f, alpha, R, beta, realloc):
    return _reallocate(_competition(A, f, alpha), f, R, beta, realloc)


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


@partial(
    jax.jit,
    static_argnames=("realloc", "normalize_input", "resolve"),
)
def _run(
    A: jnp.ndarray,
    f: jnp.ndarray,
    alpha: float,
    beta: float,
    R: float,
    tol: float,
    max_iter: int,
    realloc: RellocType,
    normalize_input: bool,
    resolve: bool,
):
    """Fully compiled matching loop. All arguments except the three static
    ones may be traced; ``max_iter`` is passed as a traced scalar, so
    changing its value does not trigger recompilation."""
    if normalize_input:
        A = _row_normalize(A, (R / f)[:, None])

    def body(carry):
        A_prev, _prev_prev, i, _done = carry
        A_new = _step(A_prev, f, alpha, R, beta, realloc)
        diff = jnp.max(jnp.abs(A_new - A_prev))
        support_eq = jnp.all((A_new > 0) == (A_prev > 0))
        done_new = (diff < tol) & support_eq
        return (A_new, A_prev, i + 1, done_new)

    def cond(carry):
        _A, _prev, i, done = carry
        return (~done) & (i < max_iter)

    init = (A, A, jnp.int32(0), jnp.bool_(False))
    A_final, _prev, iters, converged = jax.lax.while_loop(cond, body, init)

    if resolve:
        A_final = _resolve_triangles(A_final)
    matching = _extract_matching(A_final)
    return A_final, matching, iters, converged


def neural_match(
    A: Any,
    f: Any | None = None,
    config: MatchingConfig | None = None,
    **kwargs: Any,
) -> MatchingResult:
    """Run the neural bipartite matching algorithm (JAX backend).

    The entire iteration — competition, reallocation, convergence check,
    optional triangle resolution, and matching extraction — runs inside a
    single JIT-compiled function.
    """
    if config is None:
        config = MatchingConfig(**kwargs)
    elif kwargs:
        config = MatchingConfig(**{**config.__dict__, **kwargs})

    A_j = jnp.asarray(A)
    if not jnp.issubdtype(A_j.dtype, jnp.floating):
        A_j = A_j.astype(jnp.float32)
    # Validation runs eagerly so errors surface before compilation.
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

    A_final, matching, iters, converged = _run(
        A_j,
        f_j,
        jnp.asarray(config.alpha, dtype=A_j.dtype),
        jnp.asarray(config.beta, dtype=A_j.dtype),
        jnp.asarray(config.R, dtype=A_j.dtype),
        jnp.asarray(config.tol, dtype=A_j.dtype),
        jnp.int32(config.max_iter),
        config.realloc,
        config.normalize_input,
        config.resolve,
    )

    return MatchingResult(
        A=A_final,
        matching=matching,
        iterations=int(iters),
        converged=bool(converged),
    )
