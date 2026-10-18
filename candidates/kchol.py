"""Python side of cuk/chol.cu: one-kernel cooperative Cholesky with pivot clamping (float or double)."""
from pathlib import Path

import torch

from candidates import cudak as k

_SRC = str(Path(__file__).resolve().parent / "cuk" / "chol.cu")
TB = 32
_f32 = k.Kernel(_SRC, "chol_f32", (k.P, k.P, k.P, k.P, k.I32, k.P))
_f64 = k.Kernel(_SRC, "chol_f64", (k.P, k.P, k.P, k.P, k.P, k.P, k.I32, k.P))
_bar = {}


def _barrier(dev):
    if dev not in _bar:
        _bar[dev] = torch.zeros(2, dtype=torch.int32, device=dev)   # [count, generation]; count returns to 0
    return _bar[dev]


GRID_MULT = 1


def _tau(tau, G):
    """pivot clamp as a 1-element device tensor of G's dtype (a float is copied without a sync)"""
    if torch.is_tensor(tau):
        return tau.reshape(1).to(G.dtype)
    return torch.full((1,), float(tau), device=G.device, dtype=G.dtype)


def _grid(n):
    return max(1, min(GRID_MULT * k.num_sms(), (n // TB) * (n // TB + 1) // 2))


def chol32(G, tau):
    """G [n, n] fp32 (lower triangle read). Returns (L, Li): L lower-triangular (zeros above), Li [n, 32] the inverses
    of L's 32 x 32 diagonal tiles. tau: pivot clamp (float, or a device tensor: no host sync)."""
    n = G.shape[0]
    assert n % TB == 0 and G.dtype == torch.float32
    A = G.contiguous().clone()
    L = torch.zeros_like(A)
    Li = torch.empty(n, TB, device=G.device, dtype=torch.float32)
    _f32((_grid(n),), (256,), (A, L, Li, _barrier(G.device), n, _tau(tau, G)), cooperative=True)
    return L, Li


def chol64(G, tau):
    """G [n, n] fp64. Returns (L64, Li64, Lf, Lif) with fp32 copies of the lower tiles and the tile inverses."""
    n = G.shape[0]
    assert n % TB == 0 and G.dtype == torch.float64
    A = G.contiguous().clone()
    L = torch.zeros_like(A)
    Li = torch.empty(n, TB, device=G.device, dtype=torch.float64)
    Lf = torch.zeros(n, n, device=G.device, dtype=torch.float32)
    Lif = torch.empty(n, TB, device=G.device, dtype=torch.float32)
    _f64((_grid(n),), (256,), (A, L, Li, Lf, Lif, _barrier(G.device), n, _tau(tau, G)), cooperative=True)
    return L, Li, Lf, Lif
