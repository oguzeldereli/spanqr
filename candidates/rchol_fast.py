"""rcholqr with the triangular solve replaced by the Triton row-parallel solve (tri.trsm2, tf32x3 dots: fp32-accurate
on tensor cores). Cholesky stays cuSOLVER (shifted, cannot break down); the inverses of its 64 x 64 diagonal blocks
(needed by the solve) come from one batched triangular solve."""
import torch

from qrbench.methods import register
from candidates.rcholqr import _prep
from candidates import tri

BS = 64


def _diag_inv(L):
    n = L.shape[0]
    nb = n // BS
    D = L.view(nb, BS, nb, BS).diagonal(dim1=0, dim2=2).permute(2, 0, 1)        # [nb, BS, BS] diagonal blocks
    I = torch.eye(BS, device=L.device, dtype=L.dtype).expand(nb, BS, BS)
    return torch.linalg.solve_triangular(D, I, upper=False).reshape(n, BS).contiguous()


def _pass(Y, alpha):
    G = Y.T @ Y
    d = torch.diagonal(G)
    G = G + torch.diag_embed((alpha * torch.linalg.matrix_norm(G) + 1e-30).expand_as(d))
    R, _ = torch.linalg.cholesky_ex(G, upper=True)    # upper: cuSOLVER's lower path runs an LU kernel, 0.5 ms at 512
    L = R.mT.contiguous()                               # G = L L^T, row-major for the kernel
    return tri.trsm2(Y, L, tri.blockinv(L, BS), bm=16, bs=BS, prec="tf32x3")


def _ns_mm(Y, n):
    """Newton-Schulz Y <- Y (1.5 I - 0.5 Y^T Y) as one K x r x r GEMM per step (no K x r elementwise passes)"""
    for _ in range(n):
        G = Y.T @ Y
        M = torch.add(torch.eye(G.shape[0], device=G.device), G, alpha=-1.0 / 3.0).mul_(1.5)   # 1.5 I - 0.5 G
        Y = Y @ M
    return Y


def rchol_fast(X, eps=2e-7, alphas=(1e-6, 1e-6, 1e-7), ns=3):
    if X.shape[1] % BS:
        return torch.linalg.qr(X)[0]
    Y = _prep(X, eps)
    for a in alphas:
        Y = _pass(Y, a)
    return _ns_mm(Y, ns)


register("rchol3f_ns3")(lambda X: rchol_fast(X))
register("rchol3f_ns2")(lambda X: rchol_fast(X, ns=2))
register("rchol4f")(lambda X: rchol_fast(X, alphas=(1e-6, 1e-6, 1e-6, 1e-7), ns=1))
