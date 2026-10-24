"""kcholf at large r (TTF 27B: r 1024-2048, K 5120-17408). At 17408 x 1024 on the 4080 Laptop kcholf takes ~31 ms:
3 passes of Gram + Cholesky + solve (~6.6 ms each) and 2 Newton-Schulz polish steps (10.7 ms, a third of the call).
Variants: fewer polish steps, the split (tf32x3) Gram in the later passes, two passes instead of three."""
import torch

from qrbench.methods import register
from candidates import kchol, tri
from candidates.kcholqr import TB, _ns_mm, _zdust, _pow2_scale, _trsm_cfg


def kbig(X, eps=5e-7, alpha=1e-7, tau=1e-8, ns=1, passes=3, split_later=False):
    K, r = X.shape
    if r % TB:
        return torch.linalg.qr(X)[0]
    X = X.contiguous()
    Z = _zdust(K, r, X.device)
    s = _pow2_scale(torch.linalg.vector_norm(X, ord=float("inf"))).reshape(1)
    bm, w, st = _trsm_cfg(K, r)
    Y = X
    for p in range(passes):
        if p == 0:
            G = tri.gram_f(Y, s=s)
        else:
            G = tri.gram_split(Y, bt=64) if split_later else Y.T @ Y
        nrm = torch.linalg.matrix_norm(G)
        G.diagonal().add_(alpha * nrm)
        L, Li = kchol.chol32(G, (tau * nrm + 1e-30).reshape(1))
        Y = tri.trsm_f(Y, L, Li, s=s if p == 0 else None, Z=Z if p == 0 else None, eps=eps if p == 0 else 0.0,
                       bm=bm, warps=w, stages=st)
    return (_ns_split(Y, ns) if split_later else _ns_mm(Y, ns)) if ns else Y


def _ns_split(Y, n):
    """_ns_mm with every Gram from the split (tf32x3) kernel"""
    for i in range(n):
        G = tri.gram_split(Y, bt=64)
        D = torch.eye(Y.shape[1], device=Y.device).sub_(G).mul_(0.5)
        Y = torch.addmm(Y, Y, D)
    return Y


register("kbig_ns1")(lambda X: kbig(X, ns=1))
register("kbig_ns0")(lambda X: kbig(X, ns=0))
register("kbig_ns1s")(lambda X: kbig(X, ns=1, split_later=True))
register("kbig_p2ns1")(lambda X: kbig(X, ns=1, passes=2))
register("kbig_p2ns2")(lambda X: kbig(X, ns=2, passes=2))
register("kbig_s")(lambda X: kbig(X, ns=2, split_later=True))


def _kf(X, Z, eps=5e-7, alpha=1e-7, tau=1e-8, ns=2):
    """kcholqr_f with the dust Z given (borth: each block its own slice, so dust completions differ between blocks)"""
    K, r = X.shape
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


_ZB = {}


def borth(X, b=512, reorth=True):
    """block Gram-Schmidt over b-column blocks (BCGS2): each block has the earlier blocks' span removed (twice) and is
    orthonormalised at width b (kcholqr_f's recipe with the block's own slice of a K x r dust), so every Cholesky is
    b x b. With reorth the block is projected and orthonormalised a second time (its dust completions are then
    orthogonal to the earlier blocks too)."""
    from candidates.kcholqr import kcholqr_f
    K, r = X.shape
    if r <= b or r % b:
        return kcholqr_f(X)
    X = X.contiguous()
    key = (K, r, b, str(X.device))
    if key not in _ZB:
        Zf = _zdust(K, r, X.device)
        _ZB[key] = [Zf[:, j:j + b].contiguous() for j in range(0, r, b)]
    Zs = _ZB[key]
    Q = torch.empty_like(X)
    s = _pow2_scale(torch.linalg.vector_norm(X, ord=float("inf")))
    for jb, j in enumerate(range(0, r, b)):
        W = X[:, j:j + b] * s
        if j:
            P = Q[:, :j]
            W = W - P @ (P.T @ W); W = W - P @ (P.T @ W)
        W = _kf(W, Zs[jb])
        if j and reorth:
            W = W - P @ (P.T @ W); W = _kf(W, Zs[jb])
        Q[:, j:j + b] = W
    return Q


register("borth512")(lambda X: borth(X, 512))
register("borth256")(lambda X: borth(X, 256))
