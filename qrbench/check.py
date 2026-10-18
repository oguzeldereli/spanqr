"""Accuracy / robustness check of orth methods against the requirements (README: Requirements).

    python -m qrbench.check -m householder cholqr2_fp32                 # all cases, acceptance shapes
    python -m qrbench.check -m mymethod --shapes 768x512 --cases mixed ttf
    python -m qrbench.check -m mymethod --real dumps/                    # matrices dumped from training

Every case is also run through Householder (the span reference: methods must be within SPAN_RATIO of it).
"""
import argparse
import json
import sys
import time

import torch

from . import gen, metrics, methods


def parse_shapes(spec):
    if spec == "bench":
        return gen.BENCH_SHAPES
    if spec == "all":
        return gen.ALL_SHAPES
    if spec in gen.SHAPES:
        return gen.SHAPES[spec]
    return [tuple(int(v) for v in s.split("x")) for s in spec.split(",")]


def run(method_names, shapes, case_names=None, seeds=(0,), real=None, extra=True, check_sync=True, verbose=True):
    fns = {n: methods.get(n) for n in method_names}
    hh = methods.get("householder")
    rows = []
    for K, r in shapes:
        cases = list(gen.make_cases(K, r, case_names, seeds, include_extra=extra))
        if real:
            cases += list(gen.real_cases(real, (K, r)))
        for case in cases:
            ref = metrics.reference(case.X)
            hres, _ = metrics.run_checked(hh, case, ref, check_sync=False)
            cond = float(ref[0][0] / ref[0][-1]) if ref[0][-1] > 0 else float("inf")
            for n, fn in fns.items():
                res, _ = metrics.run_checked(fn, case, ref, check_sync=check_sync)
                fails = res.failures(span_ref=hres.span_n)
                row = dict(method=n, K=K, r=r, case=case.name, hard=case.hard, cond=cond,
                           hh_span_n=hres.span_n, fails=fails, **res.as_dict())
                rows.append(row)
                if verbose:
                    _print_row(row)
    return rows


def _fmt(v, f="{:.1e}"):
    return "-" if v is None or v != v else f.format(v)


def _print_row(w):
    status = "ok  " if not w["fails"] else ("FAIL" if w["hard"] else "fail")
    print(f"{status} {w['method']:<18} {w['K']:>5}x{w['r']:<4} {w['case']:<24} cond {_fmt(w['cond'], '{:.0e}'):>6}"
          f"  orth {_fmt(w['orth']):>7}  span_n {_fmt(w['span_n'], '{:.3g}'):>7} (hh {_fmt(w['hh_span_n'], '{:.3g}'):>6})"
          f"  {'; '.join(w['fails'])}", flush=True)


def summarize(rows):
    print("\nsummary (hard = acceptance cases, extra = robustness cases):")
    for n in dict.fromkeys(w["method"] for w in rows):
        mine = [w for w in rows if w["method"] == n]
        hard = [w for w in mine if w["hard"]]
        extra = [w for w in mine if not w["hard"]]
        worst_orth = max((w["orth"] for w in mine if w["orth"] == w["orth"]), default=float("nan"))
        ratios = [w["span_n"] / max(w["hh_span_n"], 1.0) for w in mine if w["finite"] and not w["error"]]
        print(f"  {n:<18} hard {sum(not w['fails'] for w in hard)}/{len(hard)}  "
              f"extra {sum(not w['fails'] for w in extra)}/{len(extra)}  worst orth {_fmt(worst_orth)}  "
              f"worst span/hh {_fmt(max(ratios, default=float('nan')), '{:.3g}')}")
        for w in hard:
            if w["fails"]:
                print(f"      FAIL {w['K']}x{w['r']} {w['case']}: {'; '.join(w['fails'])}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--methods", nargs="+", default=["householder"])
    p.add_argument("--shapes", default="bench", help="bench | all | 110m | 0.8b | 27b | KxR[,KxR...]")
    p.add_argument("--cases", nargs="*", help="case names or prefixes (default: all)")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--real", help="directory of dumped X matrices (qrbench.dump)")
    p.add_argument("--real-only", action="store_true", help="only the dumped matrices (with --real)")
    p.add_argument("--no-extra", action="store_true", help="acceptance cases only")
    p.add_argument("--no-sync-check", action="store_true")
    p.add_argument("--json", help="write all rows to this file")
    p.add_argument("--list", action="store_true", help="list methods and cases")
    a = p.parse_args(argv)
    if a.list:
        methods.load_candidates()
        print("methods:", ", ".join(sorted(methods.METHODS)))
        print("cases:  ", ", ".join(gen.case_names()))
        return 0
    t0 = time.time()
    if a.real_only:
        a.cases = ["__none__"]
    rows = run(a.methods, parse_shapes(a.shapes), a.cases, tuple(range(a.seeds)), a.real,
               extra=not a.no_extra, check_sync=not a.no_sync_check)
    summarize(rows)
    print(f"({time.time() - t0:.0f} s, {torch.cuda.get_device_name()})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(device=torch.cuda.get_device_name(), rows=rows), f, indent=1)
    return 0 if all(not w["fails"] for w in rows if w["hard"]) else 1


if __name__ == "__main__":
    sys.exit(main())
