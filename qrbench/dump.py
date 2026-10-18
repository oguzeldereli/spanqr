"""Dump real X matrices from a TTF training run, as test inputs (python -m qrbench.check --real dumps/).

No TTF file is changed: install() patches ttf.update._orth_ at runtime, in the training process. Call it once after
importing ttf, before training starts (e.g. from a small launcher script or an interactive session):

    import sys; sys.path.insert(0, "/path/to/spanqr")
    from qrbench import dump
    dump.install("/path/to/spanqr/dumps", calls_per_step=84, steps=(0, 1, 50, 500, 2000))

Each dumped call writes <dir>/s<step>_c<call>_<K>x<r>.pt = {X, V (previous basis, when the layer has one), Q_hh
(the Householder result, as returned to training), step, call, layer}. The dump steps synchronise with the host (the
.cpu() copies); the other steps are untouched. The training itself is unchanged: the original _orth_ result is returned.
"""
import inspect
import os

import torch


def install(out_dir, calls_per_step, steps=(0, 1, 50, 500, 2000), max_per_step=None, with_v=True):
    """calls_per_step: orth calls per optimizer step (110M: 84; 0.8B: 150; 27B: 400; grad_accum micro: per step, as
    the basis turns once per step). steps: 0-based optimizer steps to dump. max_per_step: limit calls dumped per step."""
    import ttf.update as tu
    os.makedirs(out_dir, exist_ok=True)
    orig = tu._orth_
    steps = set(steps)
    n = [0]

    def _orth_dump(X):
        Q = orig(X)
        i = n[0]; n[0] += 1
        step, call = divmod(i, calls_per_step)
        if step in steps and (max_per_step is None or call < max_per_step):
            V, layer = None, None
            f = inspect.currentframe().f_back
            while f is not None and f.f_code.co_name not in ("ts_layer", "_ts_layer_factored"):
                f = f.f_back
            if f is not None:
                loc = f.f_locals
                v = loc.get("V")
                if with_v and torch.is_tensor(v) and v.shape == X.shape:
                    V = v.detach().float().cpu()
                l_ = loc.get("l")
                layer = getattr(l_, "_ts_idx", None) if l_ is not None else None
            K, r = X.shape
            torch.save(dict(X=X.detach().float().cpu(), V=V, Q_hh=Q.detach().float().cpu(), step=step, call=call,
                            layer=layer), os.path.join(out_dir, f"s{step}_c{call:03d}_{K}x{r}.pt"))
        return Q

    tu._orth_ = _orth_dump
    return lambda: setattr(tu, "_orth_", orig)     # uninstall
