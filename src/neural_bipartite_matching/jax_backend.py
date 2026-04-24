"""JAX implementation of the neural bipartite matching algorithm.

Implements the competition / reallocation update described in
:mod:`neural_bipartite_matching._core`. The entire iteration loop —
including the convergence check, optional triangle resolution, and
extraction of the final ``matching`` vector — is compiled inside a
single :func:`jax.jit`'ed function driven by :func:`jax.lax.while_loop`.
This avoids per-iteration host/device synchronization and lets the
whole call be composed with further ``jit`` / ``vmap`` / ``pmap``
transforms from the caller.
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

from ._core import ConvergenceType, MatchingConfig, MatchingResult, RellocType


def _row_normalize(A: jnp.ndarray, budget: jnp.ndarray) -> jnp.ndarray:
    """Rescale rows so that ``sum_j A_ij = budget_i = R / f_i``."""
    row_sum = A.sum(axis=1, keepdims=True)
    safe = jnp.where(row_sum > 0, row_sum, 1.0)
    scaled = A * (budget / safe)
    return jnp.where(row_sum > 0, scaled, A)


def _competition(A: jnp.ndarray, f: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Competition step (paper Eq. 1)::

        A'_ij = max( A_ij - alpha * ( T_j - f_i * A_ij ) , 0 )
        T_j   = sum_k f_k * A_kj
    """
    total = (f[:, None] * A).sum(axis=0, keepdims=True)  # T_j, shape (1, M)
    competitors = total - f[:, None] * A
    return jnp.maximum(A - alpha * competitors, 0.0)


def _reallocate(
    A: jnp.ndarray,
    f: jnp.ndarray,
    R: float,
    beta: float,
    realloc: RellocType,
) -> jnp.ndarray:
    """Reallocation step, applied to the post-competition matrix ``A'``.

    With ``S_i = sum_{j'} A'_ij'`` and ``retract_i = R/f_i - S_i``:

    Multiplicative (paper Eq. 2)::

        A_new_ij = A'_ij + beta * retract_i * A'_ij / S_i

    Constant (paper Eq. 3), ``d_i`` = active degree of neuron ``i``::

        A_new_ij = A'_ij + beta * retract_i / d_i   if A'_ij > 0
                 = 0                                if A'_ij = 0
    """
    budget = (R / f)[:, None]                          # R / f_i
    row_sum = A.sum(axis=1, keepdims=True)             # S_i
    retracted = jnp.maximum(budget - row_sum, 0.0)     # retract_i
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


def _resolve_triangles(A: jnp.ndarray, one_to_one: bool) -> jnp.ndarray:
    """Prune so each fiber has at most one incident edge (and each neuron
    too, when ``one_to_one``). Ties broken by keeping the max-weight edge.
    """
    if one_to_one:
        active = A > 0
        masked = jnp.where(active, A, -jnp.inf)
        row_winners = jnp.argmax(masked, axis=1)  # (N,)
        keep = jax.nn.one_hot(row_winners, A.shape[1], axis=1, dtype=jnp.bool_)
        any_row = active.any(axis=1, keepdims=True)
        keep = keep & any_row
        A = jnp.where(keep, A, 0.0)

    active = A > 0
    masked = jnp.where(active, A, -jnp.inf)
    col_winners = jnp.argmax(masked, axis=0)  # (M,)
    keep = jax.nn.one_hot(col_winners, A.shape[0], axis=0, dtype=jnp.bool_)
    any_col = active.any(axis=0, keepdims=True)
    keep = keep & any_col
    return jnp.where(keep, A, 0.0)


def _extract_matching(A: jnp.ndarray) -> jnp.ndarray:
    active = A > 0
    masked = jnp.where(active, A, -jnp.inf)
    idx = jnp.argmax(masked, axis=0)
    any_active = active.any(axis=0)
    return jnp.where(any_active, idx, -1).astype(jnp.int32)


def _structural_converged(A: jnp.ndarray, one_to_one: bool) -> jnp.ndarray:
    """True when every fiber has exactly one incident edge (and, if
    ``one_to_one``, every neuron does too).
    """
    col_ok = jnp.all((A > 0).sum(axis=0) == 1)
    if one_to_one:
        row_ok = jnp.all((A > 0).sum(axis=1) == 1)
        return col_ok & row_ok
    return col_ok


@partial(
    jax.jit,
    static_argnames=("realloc", "normalize_input", "resolve", "convergence", "one_to_one"),
)
def _run(
    A: jnp.ndarray,
    f: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    R: jnp.ndarray,
    tol: jnp.ndarray,
    max_iter: jnp.ndarray,
    realloc: RellocType,
    convergence: ConvergenceType,
    normalize_input: bool,
    resolve: bool,
    one_to_one: bool,
):
    """Fully compiled matching loop."""
    if normalize_input:
        A = _row_normalize(A, (R / f)[:, None])

    def body(carry):
        A_prev, i, _done = carry
        A_new = _step(A_prev, f, alpha, R, beta, realloc)
        unchanged = jnp.all(A_new == A_prev)
        if convergence == "structural":
            done_new = unchanged | _structural_converged(A_new, one_to_one)
        else:  # "weights"
            diff = jnp.max(jnp.abs(A_new - A_prev))
            support_eq = jnp.all((A_new > 0) == (A_prev > 0))
            done_new = unchanged | ((diff < tol) & support_eq)
        return (A_new, i + 1, done_new)

    def cond(carry):
        _A, i, done = carry
        return (~done) & (i < max_iter)

    init = (A, jnp.int32(0), jnp.bool_(False))
    A_final, iters, converged = jax.lax.while_loop(cond, body, init)

    if resolve:
        A_final = _resolve_triangles(A_final, one_to_one)
    matching = _extract_matching(A_final)
    return A_final, matching, iters, converged


def _apply_noise(A: jnp.ndarray, seed: int | None) -> jnp.ndarray:
    """Symmetry-breaking perturbation::

        A_ij <- A_ij * (1 + u_ij),    u_ij ~ Uniform(0.01, 0.05)
    """
    key = jax.random.PRNGKey(0 if seed is None else int(seed))
    u = jax.random.uniform(key, A.shape, A.dtype, minval=0.01, maxval=0.05)
    return A + A * u


def neural_match(
    A: Any,
    f: Any | None = None,
    config: MatchingConfig | None = None,
    **kwargs: Any,
) -> MatchingResult:
    """Run the neural bipartite matching algorithm (JAX backend)."""
    if config is None:
        config = MatchingConfig(**kwargs)
    elif kwargs:
        config = MatchingConfig(**{**config.__dict__, **kwargs})

    A_j = jnp.asarray(A)
    if not jnp.issubdtype(A_j.dtype, jnp.floating):
        A_j = A_j.astype(jnp.float32)
    if (A_j < 0).any():
        raise ValueError("Input weight matrix must be non-negative.")
    N, M = A_j.shape
    if N > M:
        raise ValueError(f"Expected N <= M, got N={N}, M={M}.")
    one_to_one = N == M

    if f is None:
        f_j = jnp.ones(N, dtype=A_j.dtype)
    else:
        f_j = jnp.asarray(f, dtype=A_j.dtype).reshape(-1)
        if f_j.shape[0] != N:
            raise ValueError(f"f must have length N={N}, got {f_j.shape[0]}.")
        if (f_j <= 0).any():
            raise ValueError("Firing rates f must be strictly positive.")

    if config.add_noise:
        A_j = _apply_noise(A_j, config.seed)

    max_iter = config.resolved_max_iter()

    A_final, matching, iters, converged = _run(
        A_j,
        f_j,
        jnp.asarray(config.alpha, dtype=A_j.dtype),
        jnp.asarray(config.beta, dtype=A_j.dtype),
        jnp.asarray(config.R, dtype=A_j.dtype),
        jnp.asarray(config.tol, dtype=A_j.dtype),
        jnp.int32(max_iter),
        config.realloc,
        config.convergence,
        config.normalize_input,
        config.resolve,
        one_to_one,
    )

    return MatchingResult(
        A=A_final,
        matching=matching,
        iterations=int(iters),
        converged=bool(converged),
    )
