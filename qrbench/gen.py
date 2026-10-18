"""Test matrices for orth(X): the acceptance cases (README: Requirements) plus extra robustness cases.

Every case is built in fp64 on the CPU from a seeded generator (deterministic), then rounded to fp32 and moved to the
GPU. The reference span is computed later from the fp32 matrix itself, so the rounding to fp32 is part of the input.

Conditioning is always mixed on both sides (X = Q1 diag(s) Q2^T) unless the case says otherwise: column-scaled matrices
are benign for Cholesky methods and gave misleading "fast and accurate" results in earlier attempts.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

# (K, r) per model (the TTF layers that call orth)
SHAPES = {
    "110m": [(768, 512), (2048, 512)],
    "0.8b": [(1024, 256), (2048, 256), (3584, 256)],
    "27b": [(5120, 256), (6144, 256), (17408, 256)],
}
BENCH_SHAPES = [(768, 512), (2048, 512), (5120, 256), (17408, 256)]   # acceptance shapes
ALL_SHAPES = sorted({s for v in SHAPES.values() for s in v})


@dataclass
class Case:
    name: str
    X: torch.Tensor                       # fp32 CUDA [K, r], contiguous
    ctx: dict = field(default_factory=dict)   # optional structure (V: previous basis, ...) for methods that use it
    hard: bool = True                      # part of the acceptance set (False: extra robustness, reported only)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _orth64(K, r, g):
    """random K x r orthonormal (fp64, CPU)"""
    Q, R = torch.linalg.qr(torch.randn(K, r, generator=g, dtype=torch.float64))
    return Q * torch.sign(torch.diagonal(R))


def _mixed(K, r, s, g):
    return _orth64(K, r, g) @ torch.diag(s) @ _orth64(r, r, g).T


def _finish(X64):
    return X64.to(torch.float32).cuda().contiguous()


# ---- acceptance cases ----

def mixed(K, r, g, c):
    """singular values logspace(0, -c), mixed on both sides"""
    return _mixed(K, r, torch.logspace(0, -c, r, dtype=torch.float64), g)


def cluster(K, r, g, c):
    """half the singular values at 1, half at 10^-c (a gap instead of a spread)"""
    s = torch.ones(r, dtype=torch.float64)
    s[r // 2:] = 10.0 ** -c
    return _mixed(K, r, s[torch.randperm(r, generator=g)], g)


def zero(K, r, g):
    return torch.zeros(K, r, dtype=torch.float64)


def rank(K, r, g, k):
    """exact rank k (up to the fp32 rounding of the product)"""
    return torch.randn(K, k, generator=g, dtype=torch.float64) @ torch.randn(k, r, generator=g, dtype=torch.float64)


def dup_cols(K, r, g):
    """second half of the columns duplicates the first half (bitwise identical after rounding)"""
    X = torch.randn(K, r, generator=g, dtype=torch.float64)
    X[:, r // 2: 2 * (r // 2)] = X[:, : r // 2]
    return X


def zero_block(K, r, g):
    """a block of zero columns in the middle of an otherwise random X"""
    X = torch.randn(K, r, generator=g, dtype=torch.float64)
    X[:, r // 4: r // 2] = 0
    return X


def graded(K, r, g, c):
    """column-scaled (benign for Cholesky methods): kept to show the difference to mixed"""
    return torch.randn(K, r, generator=g, dtype=torch.float64) * torch.logspace(0, -c, r, dtype=torch.float64)


# ---- TTF-structured cases: X = V B + P, B = U^T U, P = u^T U ----

def _ttf_parts(K, r, g, cu, tilt, sig_rank=32, noise=0.3):
    """V: previous orthonormal basis (K x r). U = Qn diag(su) Qr^T: the accumulator's left factor with spectrum
    logspace(0, -cu) (so cond(B) = 10^(2 cu)). u = (low-rank signal + iid noise): u^T U is formed without the N x K u,
    using u = L R^T + noise and noise^T U ~ Z diag(su) Qr^T (same distribution). P is scaled so that
    ||P|| = tilt * ||V B||."""
    su = torch.logspace(0, -cu, r, dtype=torch.float64)
    Qr = _orth64(r, r, g)
    N = max(K, r)
    Qn = _orth64(N, r, g)
    U_rs = torch.diag(su) @ Qr.T                       # U = Qn @ U_rs
    B = U_rs.T @ U_rs                                  # U^T U
    V = _orth64(K, r, g)
    L = torch.randn(N, sig_rank, generator=g, dtype=torch.float64) / math.sqrt(N)
    R = torch.randn(K, sig_rank, generator=g, dtype=torch.float64)
    P = R @ ((L.T @ Qn) @ U_rs) + noise * torch.randn(K, r, generator=g, dtype=torch.float64) @ U_rs
    VB = V @ B
    if tilt > 0:
        P = P * (tilt * VB.norm() / P.norm())
    else:
        P = torch.zeros_like(P)
    return V, B, P, VB


def ttf_dense(K, r, g, cu, tilt):
    V, B, P, VB = _ttf_parts(K, r, g, cu, tilt)
    return VB + P, {"V": V}


def ttf_first(K, r, g, noise):
    """first step of a layer: X = u^T G, G random N x r, u = rank-32 signal + iid noise (noise^T G is iid gaussian;
    noise is relative to the signal's entry scale). noise = 0: exactly rank 32 (rank-deficient X)."""
    N = max(K, r)
    L = torch.randn(N, 32, generator=g, dtype=torch.float64) / math.sqrt(N)
    R = torch.randn(K, 32, generator=g, dtype=torch.float64)
    G = torch.randn(N, r, generator=g, dtype=torch.float64)
    X = R @ (L.T @ G)                                   # entries ~ sqrt(32)
    if noise > 0:
        X = X + noise * math.sqrt(32) * torch.randn(K, r, generator=g, dtype=torch.float64)
    return X


# ---- registry ----

def _cases_for(K, r):
    """(name, builder(g) -> X64 or (X64, ctx), hard)"""
    out = []
    for c in (2, 4, 6, 8):
        out.append((f"mixed_c{c}", lambda g, c=c: mixed(K, r, g, c), True))
    out += [
        ("zero", lambda g: zero(K, r, g), True),
        ("rank_half", lambda g: rank(K, r, g, r // 2), True),
        ("dup_cols", lambda g: dup_cols(K, r, g), True),
        ("ttf_cu2_t1e-1", lambda g: ttf_dense(K, r, g, 2, 1e-1), True),
        ("ttf_cu4_t1e-2", lambda g: ttf_dense(K, r, g, 4, 1e-2), True),
        ("ttf_cu4_t1", lambda g: ttf_dense(K, r, g, 4, 1.0), True),
        ("ttf_zero_grad_cu4.5", lambda g: ttf_dense(K, r, g, 4.5, 0.0), True),   # u = 0: X = V B, cond ~1e9
        ("ttf_first", lambda g: ttf_first(K, r, g, 0.1), True),
        ("ttf_first_rank32", lambda g: ttf_first(K, r, g, 0.0), True),
        # extra robustness (reported, not part of the acceptance set)
        ("mixed_c10", lambda g: mixed(K, r, g, 10), False),
        ("mixed_c14", lambda g: mixed(K, r, g, 14), False),
        ("cluster_c5", lambda g: cluster(K, r, g, 5), False),
        ("cluster_c9", lambda g: cluster(K, r, g, 9), False),
        ("zero_block", lambda g: zero_block(K, r, g), False),
        ("rank_1", lambda g: rank(K, r, g, 1), False),
        ("graded_c8", lambda g: graded(K, r, g, 8), False),
        ("tiny_1e-25", lambda g: 1e-25 * mixed(K, r, g, 4), False),    # Gram underflows in fp32
        ("huge_1e25", lambda g: 1e25 * mixed(K, r, g, 4), False),      # Gram overflows in fp32
    ]
    return out


def case_names(K=768, r=512):
    return [n for n, _, _ in _cases_for(K, r)]


def make_cases(K, r, names=None, seeds=(0,), include_extra=True):
    """yields Case objects; names: list of case names or prefixes (None: all)"""
    for i, (name, build, hard) in enumerate(_cases_for(K, r)):
        if names and not any(name == n or name.startswith(n) for n in names):
            continue
        if not hard and not include_extra and not names:
            continue
        for s in seeds:
            out = build(_gen(1_000_003 * s + 7919 * i + K * 31 + r))
            X64, ctx = out if isinstance(out, tuple) else (out, {})
            ctx = {k: _finish(v) for k, v in ctx.items()}
            yield Case(f"{name}/s{s}", _finish(X64), ctx, hard)


def real_cases(dump_dir="dumps", shape=None):
    """matrices dumped from a training run by qrbench.dump (files *.pt with key 'X', optional 'V')"""
    root = Path(dump_dir)
    for p in sorted(root.rglob("*.pt")):
        d = torch.load(p, map_location="cpu", weights_only=True)
        X = d["X"] if isinstance(d, dict) else d
        if shape and tuple(X.shape) != tuple(shape):
            continue
        ctx = {"V": d["V"].float().cuda().contiguous()} if isinstance(d, dict) and d.get("V") is not None else {}
        yield Case(f"real/{p.relative_to(root).with_suffix('')}", X.float().cuda().contiguous(), ctx, True)
