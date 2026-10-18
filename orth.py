"""spanqr entry point.

    import sys; sys.path.insert(0, "/path/to/spanqr")
    from orth import orth
    Q = orth(X)          # X: fp32 CUDA [K, r] -> Q: fp32 CUDA [K, r], orthonormal columns, span(Q) = span(X)
"""
import torch

from candidates.kcholqr import kcholqr_f


def orth(X: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the column span of X (fp32 CUDA, K >= r). Where X is rank-deficient or zero, the missing
    directions are completed with arbitrary orthonormal ones (never zero or NaN columns). Deterministic, no host sync.
    r must be a multiple of 32; otherwise this falls back to torch.linalg.qr."""
    return kcholqr_f(X)
