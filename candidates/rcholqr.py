"""Robust Cholesky-QR family: fixes the two ways plain Cholesky-QR dies, while keeping its GEMM-shaped work.

1. Scale: X is scaled by a power of 2 (exact) so its largest column has norm ~1 (no Gram under/overflow; zero X stays 0).
2. Completion: a fixed, seeded random matrix Z (unit columns) times eps is added. Exactly dependent or zero columns
   then get a random component, which becomes the "arbitrary orthonormal completion" the spec allows. The span tilt
   this adds is ~eps / sigma_i, i.e. span_n ~ eps / u32 (eps = 2e-7 -> ~3 units, below Householder's own 4 - 9).
3. Shift: each pass factors G + s I with s = alpha * ||G||_F (>= the largest eigenvalue), so the Cholesky cannot break down. A shift does
   not change the span (Y = X R^-1 for any invertible R). It only leaves weak directions shorter than 1, so the
   next pass has to finish them: each pass with alpha ~ 1e-6 stretches the weak directions by ~1e3.
4. Apply with a triangular solve (backward stable: the span stays as accurate as Householder's), or with an explicit
   inverse + GEMM (faster, but amplifies rounding by cond(R): only safe once Y is nearly orthonormal).
5. Cleanup: Newton-Schulz steps Y <- Y (1.5 I - 0.5 Y^T Y) (GEMMs only) once Y is close to orthonormal.
"""
import torch

from qrbench.methods import register

_Z = {}


def _z(K, r, dev):
    """fixed random completion directions: deterministic, cached per shape and device"""
    k = (K, r, str(dev))
    if k not in _Z:
        g = torch.Generator(device=dev).manual_seed(20261018)
        Z = torch.randn(K, r, device=dev, generator=g)
        _Z[k] = Z / Z.norm(dim=0, keepdim=True)
    return _Z[k]


def _pow2_scale(m):
    """2^-round(log2 m) (exact scaling), 1 where m is 0"""
    return torch.where(m > 0, torch.exp2(-torch.round(torch.log2(m.clamp_min(1e-38)))), torch.ones_like(m))


def _prep(X, eps):
    Y = X * _pow2_scale(X.abs().amax())          # largest entry ~1 first: column norms cannot over/underflow
    m = Y.norm(dim=0).amax()
    Y = Y * _pow2_scale(m)                         # then largest column norm ~1
    if eps > 0:
        Y = torch.addcmul(Y, _z(*X.shape, X.device), torch.full_like(m, eps))
    return Y


def _chol_pass(Y, alpha, apply="trsm", g64=False):
    G = (Y.double().T @ Y.double()) if g64 else (Y.T @ Y)
    d = torch.diagonal(G)
    G = G + torch.diag_embed((alpha * torch.linalg.matrix_norm(G) + 1e-30).expand_as(d))  # ||G||_F >= ||G||_2
    R, _ = torch.linalg.cholesky_ex(G, upper=True)
    R = R.float()
    if apply == "trsm":
        return torch.linalg.solve_triangular(R, Y, upper=True, left=False)
    Ri = torch.linalg.solve_triangular(R, torch.eye(R.shape[0], device=R.device), upper=True)
    return Y @ Ri


def _ns(Y, n):
    for _ in range(n):
        Y = 1.5 * Y - 0.5 * (Y @ (Y.T @ Y))
    return Y


def rcholqr(X, eps=2e-7, alphas=(1e-6, 1e-6, 1e-7), applies=("trsm", "trsm", "trsm"), ns=1, g64_first=False):
    Y = _prep(X, eps)
    for i, (a, ap) in enumerate(zip(alphas, applies)):
        Y = _chol_pass(Y, a, ap, g64=(g64_first and i == 0))
    return _ns(Y, ns).contiguous()


register("rchol3")(lambda X: rcholqr(X))
register("rchol4")(lambda X: rcholqr(X, alphas=(1e-6, 1e-6, 1e-6, 1e-7), applies=("trsm",) * 4))
register("rchol3_inv_last")(lambda X: rcholqr(X, applies=("trsm", "trsm", "inv")))
register("rchol3_inv")(lambda X: rcholqr(X, applies=("inv", "inv", "inv")))
register("rchol_g64_2")(lambda X: rcholqr(X, alphas=(1e-14, 1e-7), applies=("trsm", "trsm"), g64_first=True))

# speed variants: fewer Cholesky passes, more GEMM-only cleanup, explicit inverse once Y is tamed
register("rchol3_ns3")(lambda X: rcholqr(X, ns=3))
register("rchol2_ns6")(lambda X: rcholqr(X, alphas=(1e-6, 1e-6), applies=("trsm", "trsm"), ns=6))
register("rchol_t3inv")(lambda X: rcholqr(X, alphas=(1e-6, 1e-6, 1e-6, 1e-7), applies=("trsm", "inv", "inv", "inv")))
register("rchol_t2inv_ns2")(lambda X: rcholqr(X, alphas=(1e-6, 1e-6, 1e-6), applies=("trsm", "inv", "inv"), ns=2))
