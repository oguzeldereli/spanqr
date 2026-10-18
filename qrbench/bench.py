"""Speed of orth methods at the TTF shapes.

    python -m qrbench.bench -m householder cholqr2_fp32 mymethod
    python -m qrbench.bench -m mymethod --shapes all --flush

Two numbers per method and shape, both GPU time from CUDA events (median over repeats):
  lat: one call, synchronised before and after (what a single call costs when the GPU is otherwise idle)
  thr: N calls back to back (the CPU runs ahead, launch overhead hidden: the per-call cost inside a training step)
--flush evicts L2 before each latency sample (the 4080 Laptop's 64 MB L2 holds every input; the 3090 Ti's is 6 MB).
"""
import argparse
import statistics
import sys

import torch

from . import gen, methods
from .check import parse_shapes

CALLS_110M = {(768, 512): 72, (2048, 512): 12}   # 12 layers x (q, k, v, o, gate, up at K=768; down at K=2048)


def _input(K, r, case):
    c = next(gen.make_cases(K, r, [case], (0,)))
    return c.X, c.ctx


def time_method(fn, X, ctx, reps=30, warmup=5, burst=20, flush=False):
    call = (lambda: fn(X, **ctx)) if getattr(fn, "uses_ctx", False) else (lambda: fn(X))
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    buf = torch.empty(96 * 2 ** 20, dtype=torch.uint8, device="cuda") if flush else None
    lat = []
    for _ in range(reps):
        if buf is not None:
            buf.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record(); call(); e.record()
        torch.cuda.synchronize()
        lat.append(s.elapsed_time(e))
    thr = []
    for _ in range(max(3, reps // 5)):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(burst):
            call()
        e.record()
        torch.cuda.synchronize()
        thr.append(s.elapsed_time(e) / burst)
    return statistics.median(lat), statistics.median(thr)


def gemm_ref(X):
    """one K x r x r GEMM (the spec's yardstick)"""
    W = torch.ones(X.shape[1], X.shape[1], device=X.device) / X.shape[1]
    return lambda X_: X_ @ W


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--methods", nargs="+", default=["householder"])
    p.add_argument("--shapes", default="bench")
    p.add_argument("--case", default="ttf_cu4_t1e-2", help="input matrix (speed may depend on content)")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--flush", action="store_true")
    a = p.parse_args(argv)
    names = list(dict.fromkeys(["householder"] + a.methods))
    fns = {n: methods.get(n) for n in names}
    print(f"{torch.cuda.get_device_name()}  (ms, median; lat = isolated call, thr = back-to-back)")
    print(f"{'shape':<11} {'method':<20} {'lat':>8} {'thr':>8} {'x hh(thr)':>10} {'x gemm':>8}")
    step = {n: 0.0 for n in names}
    for K, r in parse_shapes(a.shapes):
        X, ctx = _input(K, r, a.case)
        g_lat, g_thr = time_method(gemm_ref(X), X, {}, a.reps, flush=a.flush)
        print(f"{K}x{r:<6} {'(one GEMM)':<20} {g_lat:8.3f} {g_thr:8.3f}")
        hh_thr = None
        for n, fn in fns.items():
            try:
                lat, thr = time_method(fn, X, ctx, a.reps, flush=a.flush)
            except Exception as e:     # noqa: BLE001
                print(f"{'':<11} {n:<20} error: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
                continue
            hh_thr = thr if n == "householder" else hh_thr
            step[n] += CALLS_110M.get((K, r), 0) * thr
            print(f"{'':<11} {n:<20} {lat:8.3f} {thr:8.3f} {hh_thr / thr:10.2f} {thr / g_thr:8.1f}")
    if all(s in parse_shapes(a.shapes) for s in CALLS_110M):
        print("\n110M step estimate (84 calls):", "  ".join(f"{n} {v:.0f} ms" for n, v in step.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
