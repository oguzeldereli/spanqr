"""spanqr entry point.

    import sys; sys.path.insert(0, "/path/to/spanqr")
    from orth import orth
    Q = orth(X)          # X: fp32 CUDA [K, r] -> Q: fp32 CUDA [K, r], orthonormal columns, span(Q) = span(X)
"""
import torch

from candidates.kbig import borth
from candidates.kcholqr import kcholqr_f


def orth(X: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of the column span of X (fp32 CUDA, K >= r). Where X is rank-deficient or zero, the missing
    directions are completed with arbitrary orthonormal ones (never zero or NaN columns). Deterministic, no host sync.
    r must be a multiple of 32; otherwise this falls back to torch.linalg.qr. Above r 512 (r a multiple of 256) the
    columns go through block Gram-Schmidt over 256-column blocks (borth), so every Cholesky stays 256 wide."""
    if X.shape[1] > 512 and X.shape[1] % 256 == 0:
        return borth(X, 256)
    return kcholqr_f(X)
