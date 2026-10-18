"""Accuracy and robustness checks for orth(X) (README: Requirements).

Span error: for every left singular direction u_i of X with sigma_i > SPAN_THRESH * sigma_1 (fp64 reference from the
fp32 input), s_i = ||(I - Q Q^T) u_i||. A backward-stable method has s_i ~ c * u32 * sigma_1 / sigma_i (u32 = 2^-24),
so the size-independent number is span_n = max_i s_i * (sigma_i / sigma_1) / u32 ("units of fp32 rounding"). Methods
are judged against fp32 Householder on the same matrix: span_n <= max(SPAN_RATIO * householder's, SPAN_FLOOR).
"""
import warnings
from dataclasses import dataclass, asdict

import torch

warnings.filterwarnings("ignore", message="Synchronization debug mode is a prototype")

U32 = 2.0 ** -24
ORTH_TOL = 1e-6        # max |Q^T Q - I|
SPAN_THRESH = 1e-6     # directions with sigma_i above this x sigma_1 must be in span(Q)
SPAN_RATIO = 10.0      # allowed factor over Householder's span_n
SPAN_FLOOR = 100.0     # ... but never fail below this many units of rounding (Householder itself varies)
BAD_COL = 0.5          # a column with norm below this (or non-finite) is degenerate


def reference(X):
    """fp64 SVD of the fp32 matrix X (via QR, on the CPU): singular values S and left vectors Ud [K, r]"""
    Xd = X.detach().double().cpu()
    Qd, Rd = torch.linalg.qr(Xd)
    Ur, S, _ = torch.linalg.svd(Rd)
    return S, Qd @ Ur


@dataclass
class Result:
    ok_shape: bool = True
    contiguous: bool = True
    finite: bool = True
    bad_cols: int = 0
    orth: float = float("nan")
    span_raw: float = 0.0       # max_i s_i over the checked directions
    span_n: float = 0.0         # max_i s_i sigma_i / sigma_1 / u32
    n_dirs: int = 0             # directions checked
    deterministic: bool = True
    sync_free: bool = True
    input_intact: bool = True
    error: str = ""

    def failures(self, span_ref=None):
        """list of hard-requirement failures (empty: pass); span_ref: Householder's span_n on the same matrix"""
        f = []
        if self.error:
            return [f"error: {self.error}"]
        if not self.ok_shape:
            f.append("shape/dtype/device")
        if not self.finite:
            f.append("non-finite")
        if self.bad_cols:
            f.append(f"{self.bad_cols} degenerate cols")
        if not (self.orth <= ORTH_TOL):
            f.append(f"orth {self.orth:.1e}")
        lim = max(SPAN_RATIO * span_ref, SPAN_FLOOR) if span_ref is not None else None
        if lim is not None and not (self.span_n <= lim):
            f.append(f"span {self.span_n:.3g} > {lim:.3g}")
        if not self.deterministic:
            f.append("non-deterministic")
        if not self.sync_free:
            f.append("host sync")
        if not self.input_intact:
            f.append("modified input")
        return f

    def as_dict(self):
        return asdict(self)


def evaluate(Q, X, ref):
    """checks one output Q of orth(X); ref = reference(X)"""
    r_ = Result()
    K, r = X.shape
    r_.ok_shape = tuple(Q.shape) == (K, r) and Q.dtype == torch.float32 and Q.device == X.device
    r_.contiguous = Q.is_contiguous()    # info only: torch.linalg.qr itself returns a column-major Q
    if tuple(Q.shape) != (K, r):
        r_.finite, r_.bad_cols = False, r
        return r_
    Qd = Q.detach().double().cpu()
    fin_col = torch.isfinite(Qd).all(dim=0)
    r_.finite = bool(fin_col.all())
    norms = torch.where(fin_col, Qd.norm(dim=0), torch.zeros(r, dtype=torch.float64))
    r_.bad_cols = int(((~fin_col) | (norms < BAD_COL)).sum())
    if not r_.finite:
        return r_
    r_.orth = float((Qd.T @ Qd - torch.eye(r, dtype=torch.float64)).abs().max())
    S, Ud = ref
    if S[0] > 0:
        keep = S > SPAN_THRESH * S[0]
        Uk = Ud[:, keep]
        s = (Uk - Qd @ (Qd.T @ Uk)).norm(dim=0)
        r_.n_dirs = int(keep.sum())
        if r_.n_dirs:
            r_.span_raw = float(s.max())
            r_.span_n = float((s * (S[keep] / S[0]) / U32).max())
    return r_


def _bits_equal(A, B):
    """bit-for-bit (NaN == NaN, -0 != +0)"""
    return A.shape == B.shape and A.dtype == B.dtype and bool(torch.equal(
        A.contiguous().view(torch.int32), B.contiguous().view(torch.int32)))


def run_checked(fn, case, ref, check_sync=True):
    """calls fn on the case twice (determinism), under CUDA sync-debug 'error' mode (host syncs raise), and evaluates"""
    X = case.X
    X0 = X.clone()
    call = (lambda: fn(X, **case.ctx)) if getattr(fn, "uses_ctx", False) else (lambda: fn(X))
    res = Result()
    try:
        call(); torch.cuda.synchronize()            # warm-up outside sync checking (lazy init, autotuning)
        if check_sync:
            prev = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                Q1 = call()
            except RuntimeError as e:
                if "synchroniz" not in str(e).lower():
                    raise
                res.sync_free = False
                torch.cuda.set_sync_debug_mode(prev)
                Q1 = call()
            finally:
                torch.cuda.set_sync_debug_mode(prev)
        else:
            Q1 = call()
        Q2 = call()
        torch.cuda.synchronize()
    except Exception as e:                          # noqa: BLE001  (a crash is a reported failure, not a harness crash)
        torch.cuda.set_sync_debug_mode(0)
        res.error = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
        return res, None
    ev = evaluate(Q1, X, ref)
    ev.sync_free = res.sync_free
    ev.deterministic = _bits_equal(Q1, Q2)
    ev.input_intact = _bits_equal(X, X0)
    return ev, Q1
