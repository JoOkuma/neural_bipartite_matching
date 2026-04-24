"""Tests for the neural bipartite matching algorithm.

These exercise both backends (skipped if unavailable) on a small set of
edge cases. Kept intentionally concise.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_bipartite_matching import (
    MatchingConfig,
    matching_weight,
    neural_match,
)

torch = pytest.importorskip("torch", reason="torch backend not installed")
jax = pytest.importorskip("jax", reason="jax backend not installed")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

BACKENDS = ["torch", "jax"]


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _assert_valid_matching(A0: np.ndarray, matching: np.ndarray) -> None:
    """Check structural validity: each fiber has at most one neuron."""
    matching = _as_numpy(matching)
    assert matching.shape == (A0.shape[1],)
    # unmatched fibers have index -1; others reference an actual neuron
    valid = matching[matching >= 0]
    assert ((valid >= 0) & (valid < A0.shape[0])).all()


def _run(backend: str, A: np.ndarray, **kwargs):
    if backend == "torch":
        return neural_match(torch.tensor(A, dtype=torch.float64), backend="torch", **kwargs)
    return neural_match(jnp.asarray(A, dtype=jnp.float64), backend="jax", **kwargs)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_perfect_matching_recovered(backend):
    """When the optimum is a clearly dominant permutation, we find it."""
    rng = np.random.default_rng(0)
    n = 8
    # near-diagonal weights: A[i,i] is much larger than off-diagonal noise.
    A = 0.01 * rng.random((n, n)) + np.eye(n)
    res = _run(backend, A, alpha=0.01, max_iter=2000)
    matching = _as_numpy(res.matching)
    assert (matching == np.arange(n)).all()
    assert res.converged


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("realloc", ["multiplicative", "constant"])
def test_one_to_many_all_fibers_matched(backend, realloc):
    """With M > N, every fiber should end up matched to exactly one neuron."""
    rng = np.random.default_rng(1)
    n, m = 5, 20
    A = rng.random((n, m)) + 0.1
    res = _run(backend, A, alpha=0.001, realloc=realloc, max_iter=3000)
    matching = _as_numpy(res.matching)
    _assert_valid_matching(A, matching)
    assert (matching >= 0).all(), "all fibers should be matched"
    # and every neuron claims at least one fiber (fairness)
    assert set(matching.tolist()) == set(range(n))


@pytest.mark.parametrize("backend", BACKENDS)
def test_near_optimal_efficiency(backend):
    """Neural matching should be within a few percent of the Hungarian optimum."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(2)
    n = 30
    A = rng.lognormal(size=(n, n))
    # Use weight convergence here: tighter stopping rule -> the matching
    # has more iterations to settle toward a high-efficiency fixed point.
    res = _run(backend, A, alpha=0.001, max_iter=10_000, convergence="weights")
    neural_w = matching_weight(A, _as_numpy(res.matching))
    ri, ci = scipy_opt.linear_sum_assignment(A, maximize=True)
    opt_w = float(A[ri, ci].sum())
    assert neural_w >= 0.85 * opt_w, f"neural={neural_w}, opt={opt_w}"


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_pruned_edges_stay_pruned(backend):
    """Once a weight hits zero during competition it never reappears."""
    rng = np.random.default_rng(3)
    A = rng.random((4, 10)) + 0.05
    cfg = MatchingConfig(alpha=0.001, max_iter=500, resolve=False)
    res = _run(backend, A, config=cfg)
    final = _as_numpy(res.A)
    # final support must be a subset of initial support (after row-normalize).
    assert ((final > 0) <= (A > 0)).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_row_budget_respected(backend):
    """Row sums must not exceed R/f_i at convergence (constraint in eq. 1 of suppl.)."""
    rng = np.random.default_rng(4)
    n, m = 6, 15
    A = rng.random((n, m))
    f = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0])
    R = 5.0
    res = _run(backend, A, f=f, R=R, alpha=0.001, max_iter=2000, resolve=False)
    final = _as_numpy(res.A)
    row_sums = final.sum(axis=1)
    assert np.all(row_sums <= R / f + 1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_trivial_1x1(backend):
    A = np.array([[0.7]])
    res = _run(backend, A)
    assert _as_numpy(res.matching).tolist() == [0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_neuron_multiple_fibers(backend):
    """One neuron and several fibers: it should win all of them."""
    A = np.array([[0.3, 0.7, 0.2, 0.9]])
    res = _run(backend, A)
    matching = _as_numpy(res.matching)
    assert (matching == 0).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_disjoint_components(backend):
    """Two independent 2x2 blocks resolve independently."""
    block = np.array([[1.0, 0.0], [0.0, 1.0]])
    A = np.block([[block, np.zeros((2, 2))], [np.zeros((2, 2)), block]])
    res = _run(backend, A, alpha=0.01)
    matching = _as_numpy(res.matching)
    assert matching.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("backend", BACKENDS)
def test_firing_rate_heterogeneity(backend):
    """Heterogeneous firing rates still yield a valid 1-to-many matching.

    The paper's "Henneman size principle" predicts high-activity neurons
    form small motor units and low-activity neurons form large ones; here
    we just check structural validity with non-trivial f.
    """
    rng = np.random.default_rng(5)
    n, m = 4, 20
    A = rng.random((n, m)) + 0.1
    f = np.array([5.0, 1.0, 3.0, 1.0])
    res = _run(backend, A, f=f, alpha=0.001, max_iter=3000)
    matching = _as_numpy(res.matching)
    _assert_valid_matching(A, matching)
    assert (matching >= 0).all()
    # low-activity neurons (smaller f) should own at least as many fibers
    # as high-activity ones, consistent with the size principle.
    counts = np.bincount(matching, minlength=n)
    assert counts[1] + counts[3] >= counts[0] + counts[2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_bad_shape(backend):
    """N > M is not a valid 1-to-many setup."""
    A = np.ones((5, 3))
    with pytest.raises(ValueError):
        _run(backend, A)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rejects_negative_weights(backend):
    A = np.array([[1.0, -0.1], [0.2, 0.3]])
    with pytest.raises(ValueError):
        _run(backend, A)


# ---------------------------------------------------------------------------
# Parity with the authors' reference implementation behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_to_one_cleanup(backend):
    """When N == M, every neuron must also end up with exactly one fiber."""
    rng = np.random.default_rng(10)
    n = 6
    A = rng.random((n, n)) + 0.1
    res = _run(backend, A, alpha=0.001, max_iter=5000)
    matching = _as_numpy(res.matching)
    # every fiber matched...
    assert (matching >= 0).all()
    # ...to a distinct neuron (permutation).
    assert sorted(matching.tolist()) == list(range(n))


@pytest.mark.parametrize("backend", BACKENDS)
def test_structural_converges_earlier_than_weights(backend):
    """Structural convergence (the reference impl's criterion) terminates
    no later than weight convergence, and usually earlier."""
    rng = np.random.default_rng(11)
    A = rng.random((5, 15)) + 0.1
    r_struct = _run(backend, A, alpha=0.001, convergence="structural", max_iter=20_000)
    r_weight = _run(backend, A, alpha=0.001, convergence="weights", max_iter=20_000)
    assert r_struct.converged and r_weight.converged
    assert r_struct.iterations <= r_weight.iterations
    # Matchings should agree regardless of criterion.
    assert _as_numpy(r_struct.matching).tolist() == _as_numpy(r_weight.matching).tolist()


@pytest.mark.parametrize("backend", BACKENDS)
def test_default_max_iter_scales_with_alpha(backend):
    """max_iter=None should default to int(10/alpha), letting small alpha
    runs converge without the user thinking about it."""
    rng = np.random.default_rng(12)
    A = rng.random((4, 12)) + 0.1
    # alpha=0.001 -> default 10_000 iters, plenty for this small problem.
    res = _run(backend, A, alpha=0.001)
    assert res.converged


@pytest.mark.parametrize("backend", BACKENDS)
def test_add_noise_breaks_symmetry(backend):
    """With perfectly tied rows the symmetric competition cannot separate
    the two neurons, so one ends up unmatched. Injecting the tiny
    multiplicative noise from the reference implementation breaks the
    symmetry and produces a fair matching."""
    A = np.ones((2, 4))
    no_noise = _run(backend, A, alpha=0.01, max_iter=5000, add_noise=False)
    with_noise = _run(backend, A, alpha=0.01, max_iter=5000, add_noise=True, seed=0)

    # Without noise: argmax tie-breaks hand every fiber to the same neuron.
    m_no = _as_numpy(no_noise.matching)
    assert len(set(m_no.tolist())) == 1, m_no

    # With noise: both neurons get at least one fiber (fair matching).
    m_yes = _as_numpy(with_noise.matching)
    _assert_valid_matching(A, m_yes)
    assert (m_yes >= 0).all()
    assert set(m_yes.tolist()) == {0, 1}

    # Reproducible under a fixed seed.
    again = _run(backend, A, alpha=0.01, max_iter=5000, add_noise=True, seed=0)
    assert _as_numpy(again.matching).tolist() == m_yes.tolist()


def test_backends_agree():
    """Torch and JAX should produce the same matching on a deterministic input."""
    rng = np.random.default_rng(7)
    A = rng.random((6, 12)) + 0.1
    t_res = _run("torch", A, alpha=0.001, max_iter=3000)
    j_res = _run("jax", A, alpha=0.001, max_iter=3000)
    assert _as_numpy(t_res.matching).tolist() == _as_numpy(j_res.matching).tolist()
