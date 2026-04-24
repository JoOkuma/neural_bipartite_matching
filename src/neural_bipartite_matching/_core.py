"""Backend-agnostic definitions for the neural bipartite matching algorithm.

Reference
---------
Dasgupta, Meirovitch, Zheng, Bush, Lichtman, Navlakha (2024).
"A neural algorithm for computing bipartite matchings." PNAS 121(37).

Problem
-------
Given an ``N x M`` non-negative weight matrix ``A`` with ``N <= M``,
find an assignment of fibers ``j = 0..M-1`` to neurons ``i = 0..N-1``
such that each fiber is matched to at most one neuron. In the balanced
case ``N == M`` this is a classical 1-to-1 matching; when ``M > N`` each
neuron may claim multiple fibers (a 1-to-many matching).

Notation
--------
- ``A_ij >= 0`` : synaptic area between neuron ``i`` and fiber ``j``.
- ``f_i > 0``   : firing rate (activity level) of neuron ``i``.
- ``R > 0``     : per-neuron resource budget.
- ``alpha > 0`` : competition coefficient.
- ``0 < beta <= 1`` : reallocation coefficient.

Invariants
----------
The algorithm starts from a row-normalized matrix and preserves the
per-neuron resource constraint throughout iteration::

    sum_j A_ij = R / f_i                                             (0)

Once an entry ``A_ij`` reaches zero it is treated as pruned and never
resurrected.

Update rule
-----------
Each iteration performs a synchronous two-phase update. Write ``A`` for
the current matrix, ``A'`` for the post-competition matrix, and
``A_new`` for the post-reallocation matrix (the next iterate).

**1. Competition** (paper Eq. 1)::

    A'_ij = max( A_ij  -  alpha * sum_{k != i} f_k * A_kj ,  0 )     (1)

Equivalently, letting ``T_j = sum_k f_k * A_kj`` be the total input to
fiber ``j``,::

    A'_ij = max( A_ij  -  alpha * ( T_j  -  f_i * A_ij ) ,  0 )

This is how both backends implement it: ``T_j`` is computed once per
step and broadcast, so each update is a pure tensor operation.

**2a. Multiplicative reallocation** (paper Eq. 2). Each neuron's
retracted resources are redistributed over its surviving connections in
proportion to their current weights::

    S_i       = sum_{j'} A'_ij'                                       (2)
    retract_i = R / f_i  -  S_i
    A_new_ij  = A'_ij  +  beta * retract_i * A'_ij / S_i
              = (1 - beta) * A'_ij  +  beta * (R / f_i) * A'_ij / S_i

(Zero entries contribute zero, so pruned edges stay pruned.)

**2b. Constant reallocation** (paper Eq. 3). Retracted resources are
split evenly across the neuron's *currently active* edges::

    d_i       = |{ j' : A'_ij' > 0 }|    (active degree of neuron i)  (3)
    A_new_ij  = A'_ij  +  beta * retract_i / d_i    if A'_ij > 0
              = 0                                   if A'_ij = 0

With ``beta = 1`` both rules restore the invariant
``sum_j A_new_ij = R / f_i``.

Convergence
-----------
Iteration stops once either:

- **Structural** (default): every fiber has exactly one incident edge,
  i.e. ``(A_new > 0).sum(axis=0) == 1`` for all fibers; and additionally
  every neuron has exactly one incident edge when ``N == M``.
- **Weights**: ``max_ij | A_new_ij - A_ij | < tol`` and the support of
  ``A`` is unchanged between the two iterates.

As a safety net, iteration also halts when ``A_new == A`` exactly (no
progress is possible).

Triangle resolution
-------------------
After iteration, if any fiber still has multiple incident edges the
``resolve_triangles`` post-processing keeps only the one with the
largest weight. When ``N == M`` the same pruning is applied per neuron
first, so the output is a true 1-to-1 matching.

Matching weight
---------------
The efficiency of a matching on the original input ``A0`` is::

    W = sum_{j : matching[j] >= 0}  A0[ matching[j] , j ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RellocType = Literal["multiplicative", "constant"]
ConvergenceType = Literal["structural", "weights"]


@dataclass
class MatchingConfig:
    """Configuration for :func:`neural_match`.

    See the module docstring of ``neural_bipartite_matching._core`` for
    the full set of equations; the summary below names the symbol each
    field controls.

    Parameters
    ----------
    alpha
        Competition coefficient ``alpha > 0`` in::

            A'_ij = max( A_ij - alpha * sum_{k != i} f_k * A_kj , 0 )

        Smaller values converge more slowly but reduce the risk of
        pruning every incident edge at a fiber in one step. ``0.001`` is
        the value used in the paper's experiments.
    beta
        Reallocation coefficient ``beta`` in (0, 1]::

            A_new_ij = (1 - beta) * A'_ij + beta * (R/f_i) * A'_ij / S_i

        (multiplicative form). ``beta = 1`` means neurons always
        reallocate their full retracted budget.
    R
        Per-neuron resource budget ``R``. The initial matrix is
        row-normalized so that ``sum_j A_ij = R / f_i`` for every neuron.
        Its absolute value is a global scale and does not affect the
        matching.
    realloc
        Which reallocation rule to use after competition:

        ``"multiplicative"`` -- distribute retracted resources in
        proportion to surviving edge weights::

            A_new_ij = A'_ij + beta * (R/f_i - S_i) * A'_ij / S_i
            S_i      = sum_{j'} A'_ij'

        ``"constant"`` -- split retracted resources evenly across the
        neuron's currently active edges::

            A_new_ij = A'_ij + beta * (R/f_i - S_i) / d_i   if A'_ij > 0
                     = 0                                    if A'_ij = 0
            d_i      = |{ j' : A'_ij' > 0 }|
    max_iter
        Maximum number of iterations. If ``None`` (default), it is set
        to ``int(10 / alpha)``, matching the authors' reference script.
    tol
        Weight-convergence tolerance ``tol`` used only when
        ``convergence="weights"``; iteration stops when
        ``max_ij | A_new_ij - A_ij | < tol`` (and the support is stable).
    convergence
        Stopping criterion:

        ``"structural"`` (default) -- stop as soon as the matching is
        structurally valid, i.e. ``(A > 0).sum(axis=0) == 1`` for every
        fiber, plus ``(A > 0).sum(axis=1) == 1`` for every neuron when
        ``N == M``. This matches the authors' reference script.

        ``"weights"`` -- stop when the weights themselves have settled
        (see ``tol`` above).

        In both modes iteration also halts if ``A_new == A`` exactly
        (no further progress is possible).
    normalize_input
        If True, rescale each row of the input so that ``sum_j A_ij
        = R / f_i`` before the first competition step.
    resolve
        If True, apply triangle resolution after convergence: for each
        fiber keep only its largest-weight incident edge, and (when
        ``N == M``) likewise per neuron. This guarantees a true
        matching on output.
    add_noise
        If True, multiply every input entry by ``1 + u_ij`` with
        ``u_ij ~ Uniform(0.01, 0.05)`` before normalization. This breaks
        symmetry on inputs with tied rows (otherwise the algorithm
        cannot separate identical neurons) and matches the reference
        implementation's behaviour on real-world datasets.
    seed
        Optional seed for the ``add_noise`` RNG.
    """

    alpha: float = 0.001
    beta: float = 1.0
    R: float = 1.0
    realloc: RellocType = "multiplicative"
    max_iter: int | None = None
    tol: float = 1e-8
    convergence: ConvergenceType = "structural"
    normalize_input: bool = True
    resolve: bool = True
    add_noise: bool = False
    seed: int | None = None

    def resolved_max_iter(self) -> int:
        """Return ``max_iter``, defaulting to ``int(10 / alpha)``."""
        return self.max_iter if self.max_iter is not None else int(10 / self.alpha)


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

    Computes::

        W = sum over fibers j with matching[j] >= 0  of  A0[ matching[j] , j ]

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
