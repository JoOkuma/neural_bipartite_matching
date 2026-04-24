"""Neural algorithm for computing bipartite matchings.

Implementation of the distributed matching algorithm from

    Dasgupta, Meirovitch, Zheng, Bush, Lichtman, Navlakha (2024).
    "A neural algorithm for computing bipartite matchings." PNAS 121(37).

Both PyTorch and JAX are optional dependencies. Import the backend you
want explicitly, or use :func:`neural_match`, which dispatches based on
the type of the input array.
"""

from __future__ import annotations

from typing import Any

from ._core import (
    MatchingConfig,
    MatchingResult,
    matched_pairs,
    matching_weight,
    to_permutation,
)

__all__ = [
    "MatchingConfig",
    "MatchingResult",
    "available_backends",
    "matched_pairs",
    "matching_weight",
    "neural_match",
    "to_permutation",
]


def available_backends() -> list[str]:
    """Return the list of backends whose dependency is installed."""
    backends: list[str] = []
    try:  # noqa: SIM105
        import torch  # noqa: F401

        backends.append("torch")
    except ImportError:
        pass
    try:  # noqa: SIM105
        import jax  # noqa: F401

        backends.append("jax")
    except ImportError:
        pass
    return backends


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def _is_jax_array(x: Any) -> bool:
    try:
        import jax
    except ImportError:
        return False
    return isinstance(x, jax.Array)


def neural_match(
    A: Any,
    f: Any | None = None,
    backend: str | None = None,
    config: MatchingConfig | None = None,
    **kwargs: Any,
) -> MatchingResult:
    """Dispatch to the torch or jax backend.

    Parameters
    ----------
    A
        (N, M) non-negative initial weight matrix.
    f
        Optional (N,) firing-rate vector (defaults to ones).
    backend
        ``"torch"``, ``"jax"``, or ``None`` to auto-select from ``A``'s type
        (falling back to the first installed backend).
    config
        Optional :class:`MatchingConfig`; kwargs override its fields.
    """
    if backend is None:
        if _is_torch_tensor(A):
            backend = "torch"
        elif _is_jax_array(A):
            backend = "jax"
        else:
            avail = available_backends()
            if not avail:
                raise ImportError(
                    "Neither torch nor jax is installed. Install one of "
                    "`neural-bipartite-matching[torch]` or "
                    "`neural-bipartite-matching[jax]`."
                )
            backend = avail[0]

    if backend == "torch":
        from . import torch_backend

        return torch_backend.neural_match(A, f=f, config=config, **kwargs)
    if backend == "jax":
        from . import jax_backend

        return jax_backend.neural_match(A, f=f, config=config, **kwargs)
    raise ValueError(f"Unknown backend: {backend!r}")
