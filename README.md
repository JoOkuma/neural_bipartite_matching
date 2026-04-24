# neural-bipartite-matching

Python implementation of the distributed bipartite matching algorithm from

> Dasgupta, Meirovitch, Zheng, Bush, Lichtman, Navlakha.
> *A neural algorithm for computing bipartite matchings.*
> PNAS 121(37), 2024.
> [doi:10.1073/pnas.2321032121](https://doi.org/10.1073/pnas.2321032121)

The authors' reference implementation (NumPy, single-file benchmark
script) lives at <https://github.com/metalloids/nmj_matching>. This
package reproduces that algorithm faithfully — matchings and iteration
counts agree exactly on identical inputs — and additionally provides:

- a vectorized **PyTorch** backend,
- a fully `jit`-compiled **JAX** backend (one `lax.while_loop` covers
  the entire iteration, the convergence check, triangle resolution, and
  matching extraction, so the call composes with outer `jit` / `vmap`),
- a clean library API (`neural_match` + `MatchingConfig` dataclass)
  instead of module-level globals.

The algorithm itself is inspired by competitive synaptic pruning in the
neuromuscular circuit. Given an `N × M` non-negative weight matrix
(`N ≤ M`), it iterates a local **competition + reallocation** rule
until each fiber (column) is matched to at most one neuron (row). It
is fully distributed, privacy-preserving, and in practice reaches
near-optimal total weight.

Both **PyTorch** and **JAX** backends are provided as optional dependencies.

## Algorithm

### Notation

- `A_ij ≥ 0` — synaptic area (weight) between neuron `i` and fiber `j`.
- `f_i > 0` — firing rate (activity level) of neuron `i`.
- `R > 0` — per-neuron resource budget.
- `α > 0` — competition coefficient.
- `0 < β ≤ 1` — reallocation coefficient.

### Invariant

The input is row-normalized and the update rule preserves, for every neuron:

```
sum_j A_ij  =  R / f_i                                            (0)
```

Once `A_ij` reaches `0` it is treated as pruned and never resurrected.

### One iteration — two synchronous phases

Let `A` be the current matrix, `A'` the matrix after competition, and `A_new`
the next iterate.

**1. Competition** (paper Eq. 1).

```
A'_ij  =  max( A_ij  −  α · sum_{k ≠ i} f_k · A_kj ,  0 )         (1)
```

Equivalently, with `T_j = sum_k f_k · A_kj` (total input to fiber `j`):

```
A'_ij  =  max( A_ij  −  α · ( T_j − f_i · A_ij ) ,  0 )
```

Both backends compute `T_j` once per step and broadcast, so the update is
a pure vectorized tensor operation.

**2a. Multiplicative reallocation** (paper Eq. 2). Retracted resources are
redistributed over the neuron's surviving edges in proportion to their
weights. With `S_i = sum_{j'} A'_ij'` and `retract_i = R/f_i − S_i`:

```
A_new_ij  =  A'_ij  +  β · retract_i · A'_ij / S_i                (2)
          =  (1 − β) · A'_ij  +  β · (R / f_i) · A'_ij / S_i
```

**2b. Constant reallocation** (paper Eq. 3). Retracted resources are split
evenly across the neuron's currently active edges. Let
`d_i = |{ j' : A'_ij' > 0 }|`:

```
A_new_ij  =  A'_ij  +  β · retract_i / d_i      if A'_ij > 0      (3)
          =  0                                  if A'_ij = 0
```

With `β = 1`, both rules restore the row-sum invariant (0).

### Convergence

Iteration stops once either:

- **Structural** (default) — `(A > 0).sum(axis=0) == 1` for every fiber
  `j`; and additionally `(A > 0).sum(axis=1) == 1` for every neuron `i`
  when `N == M`. Matches the authors' reference script.
- **Weights** — `max_ij |A_new_ij − A_ij| < tol` and the support is
  unchanged.

As a safety net, iteration also halts when `A_new == A` exactly.

### Triangle resolution

After iteration, if any fiber still has multiple incident edges, only the
highest-weight one is kept. When `N == M` the same pruning is also applied
per neuron first, so the output is a true 1-to-1 matching.

### Matching weight

Given the final `matching` vector (`matching[j] = i`, or `−1` if fiber `j`
is unmatched), the total efficiency with respect to the original input
`A0` is:

```
W  =  sum_{j : matching[j] ≥ 0}  A0[ matching[j] , j ]
```

## Installation

```bash
pip install neural-bipartite-matching[torch]          # torch backend
pip install neural-bipartite-matching[jax]            # jax backend
pip install neural-bipartite-matching[torch,jax]
```

## Usage

```python
import numpy as np
from neural_bipartite_matching import neural_match, matching_weight

rng = np.random.default_rng(0)
A = rng.random((10, 30)) + 0.1            # 10 neurons, 30 fibers

res = neural_match(A, alpha=0.001)         # auto-selects a backend
print(res.matching)                        # (M,) array: fiber -> neuron, -1 if unmatched
print(matching_weight(A, res.matching))    # total weight of matched edges
```

You can also call a specific backend directly:

```python
import torch
from neural_bipartite_matching.torch_backend import neural_match

A = torch.rand(10, 30) + 0.1
res = neural_match(A, alpha=0.001, realloc="multiplicative")
```

### Key parameters (see `MatchingConfig`)

| param               | default            | meaning                                                                         |
|---------------------|--------------------|---------------------------------------------------------------------------------|
| `alpha`             | `0.001`            | competition coefficient `α` in Eq. (1)                                          |
| `beta`              | `1.0`              | reallocation coefficient `β` in Eqs. (2) / (3)                                  |
| `R`                 | `1.0`              | per-neuron resource budget (row sums to `R/f_i`)                                |
| `realloc`           | `"multiplicative"` | `"multiplicative"` (Eq. 2) or `"constant"` (Eq. 3)                              |
| `f`                 | `ones(N)`          | firing-rate vector `f`                                                          |
| `max_iter`          | `int(10 / alpha)`  | iteration cap (matches [the reference script](https://github.com/metalloids/nmj_matching) — `alpha=0.001 → 10_000`) |
| `convergence`       | `"structural"`     | `"structural"`: stop on valid matching (reference-script behaviour); `"weights"`: stop when `‖ΔA‖∞ < tol` |
| `tol`               | `1e-8`             | only used when `convergence="weights"`                                          |
| `add_noise`, `seed` | `False`, `None`    | inject `Uniform(0.01, 0.05)` multiplicative noise to break tied-row symmetry ([same trick as the reference implementation](https://github.com/metalloids/nmj_matching/blob/main/src/neural_matching.py)) |

When `N == M`, both the convergence check and triangle resolution also
require each neuron to end up with exactly one fiber, so the output is a
true 1-to-1 matching.

## Testing

```bash
uv sync --extra test
uv run pytest
```

## See also

- Paper: Dasgupta *et al.*, PNAS 2024 — [doi:10.1073/pnas.2321032121](https://doi.org/10.1073/pnas.2321032121).
- Authors' reference implementation (NumPy, benchmarks, datasets, figure-generation scripts): <https://github.com/metalloids/nmj_matching>.
