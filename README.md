# neural-bipartite-matching

Python implementation of the distributed bipartite matching algorithm from

> Dasgupta, Meirovitch, Zheng, Bush, Lichtman, Navlakha.
> *A neural algorithm for computing bipartite matchings.*
> PNAS 121(37), 2024.

The algorithm is inspired by competitive synaptic pruning in the neuromuscular
circuit. Given an `N × M` non-negative weight matrix (`N ≤ M`), it iterates a
local **competition + reallocation** rule until each fiber (column) is matched
to at most one neuron (row). It is fully distributed, privacy-preserving, and
in practice reaches near-optimal total weight.

Both **PyTorch** and **JAX** backends are provided as optional dependencies.

## Installation

```bash
pip install neural-bipartite-matching[torch]   # torch backend
pip install neural-bipartite-matching[jax]     # jax backend
pip install neural-bipartite-matching[torch,jax]
```

## Usage

```python
import numpy as np
from neural_bipartite_matching import neural_match, matching_weight

rng = np.random.default_rng(0)
A = rng.random((10, 30)) + 0.1            # 10 neurons, 30 fibers

res = neural_match(A, alpha=0.001, max_iter=5000)  # auto-selects a backend
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

| param             | default           | meaning                                               |
|-------------------|-------------------|-------------------------------------------------------|
| `alpha`           | `0.001`           | competition coefficient                               |
| `beta`            | `1.0`             | reallocation coefficient                              |
| `R`               | `1.0`             | per-neuron resource budget (row sums to `R/f_i`)      |
| `realloc`         | `"multiplicative"`| `"multiplicative"` (eq. 2) or `"constant"` (eq. 3)    |
| `f`               | `ones(N)`         | per-neuron firing rates                               |
| `max_iter`, `tol` | `1000`, `1e-8`    | stopping criteria                                     |

## Testing

```bash
uv sync --extra test
uv run pytest
```
