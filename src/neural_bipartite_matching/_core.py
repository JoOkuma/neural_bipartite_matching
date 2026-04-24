"""Backend-agnostic definitions for the neural bipartite matching algorithm.

Reference
---------
Dasgupta, Meirovitch, Zheng, Bush, Lichtman, Navlakha (2024).
"A neural algorithm for computing bipartite matchings." PNAS 121(37).

The algorithm iterates two steps on an N x M non-negative weight matrix A:

    competition:
        A'_ij = max( A_ij - alpha * sum_{k != i} f_k * A_kj , 0 )

    reallocation (multiplicative):
        A_ij = A'_ij + beta * (R/f_i - sum_j' A'_ij') * A'_ij / sum_j' A'_ij'

    reallocation (constant):
        A_ij = A'_ij + beta * (R/f_i - sum_j' A'_ij') * (1 / degree_i)  if A'_ij > 0

Each row is pre-normalized so that sum_j A_ij = R/f_i. Once an entry
reaches 0 it is considered pruned and stays at 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RellocType = Literal["multiplicative", "constant"]


@dataclass
class MatchingConfig:
    """Configuration for :func:`neural_match`.

    Parameters
    ----------
    alpha
        Competition coefficient. Smaller values converge more slowly but
        reduce the risk of pruning every incident edge at a fiber in one
        step. The paper uses ``0.001``.
    beta
        Reallocation coefficient in (0, 1]. The paper fixes ``1.0`` so that
        neurons always reallocate their full retracted budget.
    R
        Per-neuron resource budget. Each row is normalized to sum to ``R/f_i``.
    realloc
        ``"multiplicative"`` (eq. 2) or ``"constant"`` (eq. 3).
    max_iter
        Maximum number of iterations.
    tol
        Convergence tolerance: iteration stops once the max absolute change
        in ``A`` and in its support drops below ``tol``.
    normalize_input
        If True, row-normalize the input so that each row sums to ``R/f_i``.
    resolve
        If True, break ties after convergence so that each fiber has at
        most one incident edge (keeping the largest-weight edge).
    """

    alpha: float = 0.001
    beta: float = 1.0
    R: float = 1.0
    realloc: RellocType = "multiplicative"
    max_iter: int = 1000
    tol: float = 1e-8
    normalize_input: bool = True
    resolve: bool = True


@dataclass
class MatchingResult:
    """Output of :func:`neural_match`.

    Attributes
    ----------
    A
        Final (N, M) weight matrix (backend-native array).
    matching
        (M,) integer array; ``matching[j]`` is the index of the neuron
        matched to fiber ``j``, or ``-1`` if the fiber is unmatched.
    iterations
        Number of iterations performed.
    converged
        Whether convergence was reached before ``max_iter``.
    """

    A: Any
    matching: Any
    iterations: int
    converged: bool


def matching_weight(A0, matching) -> float:
    """Total weight of a matching, summed over the original weights ``A0``.

    ``matching[j] = i`` pairs fiber ``j`` with neuron ``i``; ``-1`` means
    unmatched. Works with any array type supporting numpy-style indexing.
    """
    import numpy as np

    A0 = np.asarray(A0)
    matching = np.asarray(matching)
    js = np.where(matching >= 0)[0]
    if js.size == 0:
        return 0.0
    return float(A0[matching[js], js].sum())
