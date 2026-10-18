"""Registry of orth(X) implementations: baselines here, candidates in new_qr/candidates/*.py.

A method is fn(X) -> Q (fp32 CUDA [K, r] -> fp32 CUDA [K, r]). register(name, uses_ctx=True) marks a method that also
takes the case's structure as keyword arguments (V: the previous basis), for warm-started / structured methods.
"""
import importlib
import pkgutil
from pathlib import Path

import torch

METHODS = {}


def register(name, uses_ctx=False):
    def deco(fn):
        fn.uses_ctx = uses_ctx
        METHODS[name] = fn
        return fn
    return deco


def get(name):
    load_candidates()
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; known: {', '.join(sorted(METHODS))}")
    return METHODS[name]


_loaded = [False]


def load_candidates():
    if _loaded[0]:
        return
    _loaded[0] = True
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for m in pkgutil.iter_modules([str(root / "candidates")]):
        importlib.import_module(f"candidates.{m.name}")


# ---- baselines ----

@register("householder")
def householder(X):
    """the current TTF orth (ttf-core/ttf/update.py _orth_): cuSOLVER geqrf + orgqr"""
    return torch.linalg.qr(X)[0]


@register("cholqr2_fp32")
def cholqr2_fp32(X):
    """Cholesky-QR twice in fp32 (breaks down from condition ~1e4). cholesky_ex: no host sync"""
    for _ in range(2):
        R, _ = torch.linalg.cholesky_ex(X.T @ X, upper=True)
        X = torch.linalg.solve_triangular(R, X, upper=True, left=False)
    return X


@register("cholqr2_fp64")
def cholqr2_fp64(X):
    """Cholesky-QR twice in fp64 (robust to ~1e8 unless rank-deficient; slower than Householder on consumer GPUs)"""
    Y = X.double()
    for _ in range(2):
        R, _ = torch.linalg.cholesky_ex(Y.T @ Y, upper=True)
        Y = torch.linalg.solve_triangular(R, Y, upper=True, left=False)
    return Y.float()


@register("householder_fp64")
def householder_fp64(X):
    return torch.linalg.qr(X.double())[0].float()
