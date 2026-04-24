"""PyTorch implementation of the neural bipartite matching algorithm."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError as e:  # pragma: no cover - exercised via import guard
    raise ImportError(
        "The torch backend requires PyTorch. Install with "
        "`pip install neural-bipartite-matching[torch]`."
    ) from e

from ._core import MatchingConfig, MatchingResult, RellocType


def _row_normalize(A: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    row_sum = A.sum(dim=1, keepdim=True)
    # leave empty rows untouched
    safe = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    return torch.where(row_sum > 0, A * (budget / safe), A)


def _competition(A: torch.Tensor, f: torch.Tensor, alpha: float) -> torch.Tensor:
    # total input into each fiber j: sum_k f_k A_kj
    total = (f.unsqueeze(1) * A).sum(dim=0, keepdim=True)  # (1, M)
    # competitors for (i, j): total_j - f_i * A_ij
    competitors = total - f.unsqueeze(1) * A
    return torch.clamp(A - alpha * competitors, min=0.0)


def _reallocate(
    A: torch.Tensor,
    f: torch.Tensor,
    R: float,
    beta: float,
    realloc: RellocType,
) -> torch.Tensor:
    budget = (R / f).unsqueeze(1)  # (N, 1)
    row_sum = A.sum(dim=1, keepdim=True)
    retracted = torch.clamp(budget - row_sum, min=0.0)
    if realloc == "multiplicative":
        # proportional redistribution; zero rows are left untouched
        safe = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
        frac = torch.where(row_sum > 0, A / safe, torch.zeros_like(A))
        return A + beta * retracted * frac
    if realloc == "constant":
        active = A > 0
        degree = active.sum(dim=1, keepdim=True).to(A.dtype).clamp(min=1.0)
        inc = beta * retracted / degree
        return torch.where(active, A + inc, A)
    raise ValueError(f"Unknown realloc type: {realloc!r}")


def _resolve_triangles(A: torch.Tensor) -> torch.Tensor:
    """Keep at most one incident edge per fiber (the largest)."""
    if A.numel() == 0:
        return A
    # per-column argmax among strictly positive entries
    active = A > 0
    if not active.any():
        return A
    # mask non-positive entries with -inf so they never win
    masked = torch.where(active, A, torch.full_like(A, float("-inf")))
    winners = masked.argmax(dim=0)  # (M,)
    keep = torch.zeros_like(A, dtype=torch.bool)
    cols = torch.arange(A.shape[1], device=A.device)
    keep[winners, cols] = True
    keep &= active  # columns with no active entries stay unmatched
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

    Returns
    -------
    MatchingResult
    """
    if config is None:
        config = MatchingConfig(**kwargs)
    elif kwargs:
        config = MatchingConfig(**{**config.__dict__, **kwargs})

    A_t = torch.as_tensor(A, dtype=torch.get_default_dtype())
    if (A_t < 0).any():
        raise ValueError("Input weight matrix must be non-negative.")
    N, M = A_t.shape
    if N > M:
        raise ValueError(f"Expected N <= M, got N={N}, M={M}.")

    if f is None:
        f_t = torch.ones(N, dtype=A_t.dtype, device=A_t.device)
    else:
        f_t = torch.as_tensor(f, dtype=A_t.dtype, device=A_t.device).reshape(-1)
        if f_t.shape[0] != N:
            raise ValueError(f"f must have length N={N}, got {f_t.shape[0]}.")
        if (f_t <= 0).any():
            raise ValueError("Firing rates f must be strictly positive.")

    budget = (config.R / f_t).unsqueeze(1)
    if config.normalize_input:
        A_t = _row_normalize(A_t, budget)

    converged = False
    iterations = 0
    for t in range(1, config.max_iter + 1):
        iterations = t
        prev = A_t
        A_prime = _competition(A_t, f_t, config.alpha)
        A_t = _reallocate(A_prime, f_t, config.R, config.beta, config.realloc)
        diff = (A_t - prev).abs().max().item()
        support_change = ((A_t > 0) != (prev > 0)).any().item()
        if diff < config.tol and not support_change:
            converged = True
            break

    if config.resolve:
        A_t = _resolve_triangles(A_t)

    matching = _extract_matching(A_t)
    return MatchingResult(A=A_t, matching=matching, iterations=iterations, converged=converged)
