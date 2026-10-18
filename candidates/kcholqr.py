"""Robust Cholesky-QR on hand-written kernels (option 1): the rcholqr recipe (exact power-of-2 rescale, seeded random
dust for the completion, shifted/clamped passes, triangular solves, Newton-Schulz polish) with
  - Cholesky: cuk/chol.cu, one cooperative kernel, pivots clamped at tau (never breaks down, no host sync),
  - solve: Triton row-parallel triangular solve with tf32x3 dots (candidates/tri.py),
  - Gram and polish: cuBLAS GEMMs.
"""
import torch

from qrbench.methods import register
from candidates.rcholqr import _prep
from candidates import kchol, tri

TB = kchol.TB


def _ns_mm(Y, n, accurate_last=True):
    """Newton-Schulz Y <- Y (1.5 I - 0.5 Y^T Y), written as the correction Y + Y D, D = 0.5 (I - Y^T Y): the GEMM
    rounds only the small term, so orthogonality reaches ~fp32 rounding (Y @ (1.5 I - 0.5 G) floored at ~1e-6)"""
    for i in range(n):
        G = tri.gram_split(Y, bt=64) if (accurate_last and i == n - 1) else Y.T @ Y
        D = torch.eye(Y.shape[1], device=Y.device).sub_(G).mul_(0.5)
        Y = torch.addmm(Y, Y, D)
    return Y


_Z = {}


def _pow2_scale(m):
    return torch.where(m > 0, torch.exp2(-torch.round(torch.log2(m.clamp_min(1e-38)))), torch.ones_like(m))


def prep_fused(X, eps):
    """same as rcholqr._prep (largest entry -> ~1, then largest column norm -> ~1, both exact powers of 2; plus eps x
    seeded unit-column random Z), in 3 sweeps over X instead of ~8"""
    K, r = X.shape
    key = (K, r, str(X.device), eps)
    if key not in _Z:
        g = torch.Generator(device=X.device).manual_seed(20261018)
        Z = torch.randn(K, r, device=X.device, generator=g)
        _Z[key] = (Z / Z.norm(dim=0, keepdim=True)) * eps
    s1 = _pow2_scale(torch.linalg.vector_norm(X, ord=float("inf")))
    s2 = _pow2_scale(tri.colsq(X, s1.reshape(1)).amax().sqrt())
    return torch.addcmul(_Z[key], X, s1 * s2)


def kpass(Y, alpha, tau_rel, bm=16, accurate=False):
    G = tri.gram_split(Y, bt=64) if accurate else Y.T @ Y
    nrm = torch.linalg.matrix_norm(G)                     # ||G||_F >= largest eigenvalue (device scalar)
    if alpha:
        G.diagonal().add_(alpha * nrm)
    L, Li = kchol.chol32(G, (tau_rel * nrm + 1e-30).reshape(1))
    return tri.trsm2(Y, L, Li, bm=bm, bs=TB, prec="tf32x3")


def kcholqr(X, eps=2e-7, passes=((1e-6, 1e-7), (1e-6, 1e-7), (1e-7, 1e-8)), ns=2):
    if X.shape[1] % TB:
        return torch.linalg.qr(X)[0]
    Y = prep_fused(X, eps)
    for a, t in passes:
        Y = kpass(Y, a, t)
    return _ns_mm(Y, ns)


register("kchol3_ns2")(lambda X: kcholqr(X))
register("kchol3_ns1")(lambda X: kcholqr(X, ns=1))
register("kchol3c_ns2")(lambda X: kcholqr(X, passes=((0.0, 1e-6), (0.0, 1e-6), (0.0, 1e-7))))   # clamp only
register("kchol2c_ns3")(lambda X: kcholqr(X, passes=((0.0, 1e-7), (0.0, 1e-7)), ns=3))
register("kchol3_ns3")(lambda X: kcholqr(X, ns=3))
A7 = ((1e-7, 1e-8),) * 3
register("kchol3a7_ns2")(lambda X: kcholqr(X, passes=A7, ns=2))


# ---- fused pipeline: no scaled copy of X, no separate dust pass ----

_TRSM_CFG = {}   # (K, r) -> (bm, warps, stages); defaults from the 4080 Laptop sweep


def _trsm_cfg(K, r):
    return _TRSM_CFG.get((K, r), (16, 2, 2) if r <= 256 and K <= 8192 else (16, 4, 3))


def _zdust(K, r, dev):
    key = (K, r, str(dev), 1.0)
    if key not in _Z:
        g = torch.Generator(device=dev).manual_seed(20261018)
        Z = torch.randn(K, r, device=dev, generator=g)
        _Z[key] = Z / Z.norm(dim=0, keepdim=True)
    return _Z[key]


def kcholqr_f(X, eps=5e-7, alpha=1e-7, tau=1e-8, ns=2):
    """pass 1: the Gram reads s * X on the fly (s: power of 2 from max |x|, a device scalar: overflow-safe, no copy);
    the solve reads s * X + eps * Z on the fly. The dust is left out of the pass-1 Gram: its contribution there is far
    below the shift, and any invertible factor keeps the span; in the solve it fills exactly dependent / zero columns
    (lifted by all three passes). Passes 2, 3: plain. Then 2 polish steps."""
    K, r = X.shape
    if r % TB:
        return torch.linalg.qr(X)[0]
    X = X.contiguous()
    Z = _zdust(K, r, X.device)
    s = _pow2_scale(torch.linalg.vector_norm(X, ord=float("inf"))).reshape(1)
    bm, w, st = _trsm_cfg(K, r)
    Y = X
    for p in range(3):
        G = tri.gram_f(Y, s=s) if p == 0 else Y.T @ Y
        nrm = torch.linalg.matrix_norm(G)
        G.diagonal().add_(alpha * nrm)
        L, Li = kchol.chol32(G, (tau * nrm + 1e-30).reshape(1))
        Y = tri.trsm_f(Y, L, Li, s=s if p == 0 else None, Z=Z if p == 0 else None, eps=eps if p == 0 else 0.0,
                       bm=bm, warps=w, stages=st)
    return _ns_mm(Y, ns)


register("kcholf")(lambda X: kcholqr_f(X))
