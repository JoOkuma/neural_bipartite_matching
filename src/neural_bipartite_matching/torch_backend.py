"""PyTorch implementation of the neural bipartite matching algorithm.

Implements the competition / reallocation update described in
:mod:`neural_bipartite_matching._core`. All operations are vectorized
over the ``N x M`` weight matrix; there are no Python inner loops.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError as e:  # pragma: no cover - exercised via import guard
    raise ImportError(
        "The torch backend requires PyTorch. Install with "
        "`pip install neural-bipartite-matching[torch]`."
    ) from e

from ._core import ConvergenceType, MatchingConfig, MatchingResult, RellocType


def _row_normalize(A: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    """Rescale rows so that ``sum_j A_ij = budget_i = R / f_i``.

    Empty rows (all zeros) are left untouched.
    """
    row_sum = A.sum(dim=1, keepdim=True)
    safe = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    return torch.where(row_sum > 0, A * (budget / safe), A)


def _competition(A: torch.Tensor, f: torch.Tensor, alpha: float) -> torch.Tensor:
    """Competition step (paper Eq. 1)::

        A'_ij = max( A_ij - alpha * sum_{k != i} f_k * A_kj , 0 )
              = max( A_ij - alpha * ( T_j - f_i * A_ij ) ,  0 )

    where ``T_j = sum_k f_k * A_kj`` is the total input to fiber ``j``.
    """
    total = (f.unsqueeze(1) * A).sum(dim=0, keepdim=True)  # (1, M) = T_j
    competitors = total - f.unsqueeze(1) * A               # T_j - f_i * A_ij
    return torch.clamp(A - alpha * competitors, min=0.0)


def _reallocate(
    A: torch.Tensor,
    f: torch.Tensor,
    R: float,
    beta: float,
    realloc: RellocType,
) -> torch.Tensor:
    """Reallocation step, applied to the post-competition matrix ``A'``.

    Let ``S_i = sum_{j'} A'_ij'`` and ``retract_i = R/f_i - S_i``.

    Multiplicative (paper Eq. 2)::

        A_new_ij = A'_ij + beta * retract_i * A'_ij / S_i

    Constant (paper Eq. 3), with ``d_i`` = active degree of neuron ``i``::

        A_new_ij = A'_ij + beta * retract_i / d_i   if A'_ij > 0
                 = 0                                if A'_ij = 0

    Zero entries stay zero in both rules, so pruned edges never reappear.
    """
    budget = (R / f).unsqueeze(1)  # (N, 1) = R / f_i
    row_sum = A.sum(dim=1, keepdim=True)              # S_i
    retracted = torch.clamp(budget - row_sum, min=0.0)  # retract_i
    if realloc == "multiplicative":
        safe = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
        frac = torch.where(row_sum > 0, A / safe, torch.zeros_like(A))
        return A + beta * retracted * frac
    if realloc == "constant":
        active = A > 0
        degree = active.sum(dim=1, keepdim=True).to(A.dtype).clamp(min=1.0)
        inc = beta * retracted / degree
        return torch.where(active, A + inc, A)
    raise ValueError(f"Unknown realloc type: {realloc!r}")


def _apply_noise(A: torch.Tensor, seed: int | None) -> torch.Tensor:
    """Symmetry-breaking perturbation used by the reference implementation::

        A_ij <- A_ij * (1 + u_ij),    u_ij ~ Uniform(0.01, 0.05)

    This is applied to the raw input *before* row-normalization so that
    neurons with identical rows no longer have perfectly tied dynamics.
    """
    gen = torch.Generator(device=A.device)
    if seed is not None:
        gen.manual_seed(int(seed))
    u = torch.empty_like(A).uniform_(0.01, 0.05, generator=gen)
    return A + A * u


def _structural_converged(A: torch.Tensor, one_to_one: bool) -> bool:
    """True when the current matrix encodes a valid matching::

        (A > 0).sum(axis=0) == 1  for every fiber j,

    and, if ``one_to_one`` (i.e. ``N == M``), also::

        (A > 0).sum(axis=1) == 1  for every neuron i.
    """
    col_ok = ((A > 0).sum(dim=0) == 1).all().item()
    if not col_ok:
        return False
    if one_to_one:
        return ((A > 0).sum(dim=1) == 1).all().item()
    return True


def _resolve_triangles(A: torch.Tensor, one_to_one: bool) -> torch.Tensor:
    """Prune so each fiber has at most one incident edge (and each neuron
    too, when ``one_to_one``). Ties broken by keeping the max-weight edge.
    """
    if A.numel() == 0 or not (A > 0).any():
        return A

    if one_to_one:
        # First pass: keep only the max-weight fiber per neuron.
        active = A > 0
        masked = torch.where(active, A, torch.full_like(A, float("-inf")))
        row_winners = masked.argmax(dim=1)  # (N,)
        keep = torch.zeros_like(A, dtype=torch.bool)
        rows = torch.arange(A.shape[0], device=A.device)
        keep[rows, row_winners] = True
        keep &= active
        A = torch.where(keep, A, torch.zeros_like(A))

    # Second pass: keep only the max-weight neuron per fiber.
    active = A > 0
    if not active.any():
        return A
    masked = torch.where(active, A, torch.full_like(A, float("-inf")))
    col_winners = masked.argmax(dim=0)  # (M,)
    keep = torch.zeros_like(A, dtype=torch.bool)
    cols = torch.arange(A.shape[1], device=A.device)
    keep[col_winners, cols] = True
    keep &= active
    return torch.where(keep, A, torch.zeros_like(A))


def _extract_matching(A: torch.Tensor) -> torch.Tensor:
    """(M,) int64 tensor: matching[j] = i, or -1 if fiber j is unmatched."""
    active = A > 0
    matching = torch.full((A.shape[1],), -1, dtype=torch.long, device=A.device)
    any_active = active.any(dim=0)
    if any_active.any():
        masked = torch.where(active, A, torch.full_like(A, float("-inf")))
        idx = masked.argmax(dim=0)
        matching = torch.where(any_active, idx.to(torch.long), matching)
    return matching


def _check_convergence(
    A: torch.Tensor,
    prev: torch.Tensor,
    mode: ConvergenceType,
    tol: float,
    one_to_one: bool,
) -> bool:
    # Safety net common to both modes: weights did not change at all.
    if torch.equal(A, prev):
        return True
    if mode == "structural":
        return _structural_converged(A, one_to_one)
    # "weights"
    diff = (A - prev).abs().max().item()
    if diff >= tol:
        return False
    return not ((A > 0) != (prev > 0)).any().item()


def neural_match(
    A: Any,
    f: Any | None = None,
    config: MatchingConfig | None = None,
    **kwargs: Any,
) -> MatchingResult:
    """Run the neural bipartite matching algorithm on a weight matrix.

    Parameters
    ----------
    A
        (N, M) non-negative initial weight matrix. Converted to a float
        tensor on the default device if not already a tensor.
    f
        Optional (N,) firing-rate vector. Defaults to all-ones.
    config
        Optional :class:`MatchingConfig`. Any keyword argument overrides
        the corresponding field.
    """
    if config is None:
        config = MatchingConfig(**kwargs)
    elif kwargs:
        config = MatchingConfig(**{**config.__dict__, **kwargs})

    # Preserve the input's floating dtype; only cast integer/bool inputs
    # to the default dtype. This avoids silently truncating a float64
    # tensor when ``torch.get_default_dtype()`` is float32.
    A_t = torch.as_tensor(A)
    if not A_t.is_floating_point():
        A_t = A_t.to(torch.get_default_dtype())
    if (A_t < 0).any():
        raise ValueError("Input weight matrix must be non-negative.")
    N, M = A_t.shape
    if N > M:
        raise ValueError(f"Expected N <= M, got N={N}, M={M}.")
    one_to_one = N == M

    if f is None:
        f_t = torch.ones(N, dtype=A_t.dtype, device=A_t.device)
    else:
        f_t = torch.as_tensor(f, dtype=A_t.dtype, device=A_t.device).reshape(-1)
        if f_t.shape[0] != N:
            raise ValueError(f"f must have length N={N}, got {f_t.shape[0]}.")
        if (f_t <= 0).any():
            raise ValueError("Firing rates f must be strictly positive.")

    if config.add_noise:
        A_t = _apply_noise(A_t, config.seed)

    budget = (config.R / f_t).unsqueeze(1)
    if config.normalize_input:
        A_t = _row_normalize(A_t, budget)

    max_iter = config.resolved_max_iter()
    converged = False
    iterations = 0
    for t in range(1, max_iter + 1):
        iterations = t
        prev = A_t
        A_prime = _competition(A_t, f_t, config.alpha)
        A_t = _reallocate(A_prime, f_t, config.R, config.beta, config.realloc)
        if _check_convergence(A_t, prev, config.convergence, config.tol, one_to_one):
            converged = True
            break

    if config.resolve:
        A_t = _resolve_triangles(A_t, one_to_one)

    matching = _extract_matching(A_t)
    return MatchingResult(A=A_t, matching=matching, iterations=iterations, converged=converged)
