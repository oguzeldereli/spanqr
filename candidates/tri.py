"""Triton kernels for the small, latency-bound parts of Cholesky-QR (cuSOLVER potrf at r = 512 is ~0.44 ms of tiny
kernels; the arithmetic is ~0.05 ms).

chol(G, tau): lower Cholesky G = L L^T in ONE program (one CTA), right-looking, blocks of BS. A pivot below tau is
    clamped to tau instead of breaking down ("modified Cholesky"): for Cholesky-QR any invertible factor keeps the span,
    so the clamp only leaves those (weak) directions short, for the next pass to finish. Also returns the inverses of
    the diagonal blocks (for the solve).
trsm(X, L, Li): Y = X L^-T (i.e. X R^-1 with R = L^T), rows in parallel (each row of Y is an independent solve),
    column blocks in order; the diagonal blocks are applied with their explicit inverses.
All fp32, IEEE dots (no TF32), deterministic (fixed reduction order), no host sync.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _chol_kernel(W, L, Li, n, tau, BS: tl.constexpr):
    offs = tl.arange(0, BS)
    rr = offs[:, None]
    cc = offs[None, :]
    nb = n // BS
    for jb in range(0, nb):
        j = jb * BS
        A = tl.load(W + (j + rr) * n + (j + cc))
        Lb = tl.zeros([BS, BS], tl.float32)
        for c in tl.static_range(BS):
            col = tl.sum(tl.where(cc == c, A, 0.0), axis=1)                 # A[:, c]
            piv = tl.sum(tl.where(offs == c, col, 0.0), axis=0)
            d = tl.sqrt(tl.maximum(piv, tau))
            l = tl.where(offs > c, col / d, 0.0)
            Lb = tl.where(cc == c, tl.where(offs == c, d, l)[:, None], Lb)
            A = A - l[:, None] * l[None, :]
        # inverse of the lower-triangular diagonal block: row c of Lb^-1 by forward substitution
        Xi = tl.zeros([BS, BS], tl.float32)
        for c in tl.static_range(BS):
            row = tl.sum(tl.where(rr == c, Lb, 0.0), axis=0)               # Lb[c, :]
            dc = tl.sum(tl.where(offs == c, row, 0.0), axis=0)
            row = tl.where(offs < c, row, 0.0)
            s = tl.sum(row[:, None] * Xi, axis=0)                           # sum_k<c Lb[c,k] Xi[k,:]
            xc = (tl.where(offs == c, 1.0, 0.0) - s) / dc
            Xi = tl.where(rr == c, xc[None, :], Xi)
        tl.store(L + (j + rr) * n + (j + cc), Lb)
        tl.store(Li + (j + rr) * BS + cc, Xi)                               # Li: [n, BS], block jb at rows j..j+BS
        # panel: L[i, j] = W[i, j] Lb^-T for the blocks below
        for ib in range(jb + 1, nb):
            i = ib * BS
            P = tl.load(W + (i + rr) * n + (j + cc))
            P = tl.dot(P, tl.trans(Xi), input_precision="ieee")
            tl.store(L + (i + rr) * n + (j + cc), P)
        tl.debug_barrier()
        # trailing update of the lower triangle: W[i, k] -= L[i, j] L[k, j]^T
        for ib in range(jb + 1, nb):
            i = ib * BS
            Pi = tl.load(L + (i + rr) * n + (j + cc))
            for kb in range(jb + 1, ib + 1):
                k = kb * BS
                Pk = tl.load(L + (k + rr) * n + (j + cc))
                Wt = tl.load(W + (i + rr) * n + (k + cc))
                Wt = Wt - tl.dot(Pi, tl.trans(Pk), input_precision="ieee")
                tl.store(W + (i + rr) * n + (k + cc), Wt)
        tl.debug_barrier()


@triton.jit
def _trsm_kernel(X, L, Li, Y, K, n, BM: tl.constexpr, BS: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows[:, None] < K
    offs = tl.arange(0, BS)
    rr = offs[:, None]
    cc = offs[None, :]
    nb = n // BS
    for jb in range(0, nb):
        j = jb * BS
        acc = tl.load(X + rows[:, None] * n + (j + cc), mask=rmask, other=0.0)
        for kb in range(0, jb):
            k = kb * BS
            Yk = tl.load(Y + rows[:, None] * n + (k + cc), mask=rmask, other=0.0)
            Ljk = tl.load(L + (j + rr) * n + (k + cc))                      # L[j-block, k-block]
            acc = acc - tl.dot(Yk, tl.trans(Ljk), input_precision="ieee")
        Xi = tl.load(Li + (j + rr) * BS + cc)
        Yj = tl.dot(acc, tl.trans(Xi), input_precision="ieee")
        tl.store(Y + rows[:, None] * n + (j + cc), Yj, mask=rmask)
        tl.debug_barrier()


BS = 32


def chol(G, tau, bs=BS):
    """G: [n, n] fp32 symmetric (only the lower triangle is read), n % bs == 0; tau: 0-dim tensor or float"""
    n = G.shape[0]
    assert n % bs == 0
    W = G.contiguous().clone()
    L = torch.zeros_like(W)
    Li = torch.empty(n, bs, device=G.device, dtype=G.dtype)
    _chol_kernel[(1,)](W, L, Li, n, tau, BS=bs, num_warps=4)
    return L, Li


def trsm(X, L, Li, bm=16, bs=BS):
    K, n = X.shape
    X = X.contiguous()
    Y = torch.empty_like(X)
    _trsm_kernel[(triton.cdiv(K, bm),)](X, L, Li, Y, K, n, BM=bm, BS=bs, num_warps=4)
    return Y


# ---- blocked version: one small kernel per diagonal block (factor + inverse + panel below), cuBLAS trailing update ----

@triton.jit
def _diag_panel_kernel(W, L, Li, n, j, tau, BS: tl.constexpr, PREC: tl.constexpr, PANEL: tl.constexpr = True):
    offs = tl.arange(0, BS)
    rr = offs[:, None]
    cc = offs[None, :]
    A = tl.load(W + (j + rr) * n + (j + cc))
    Lb = tl.zeros([BS, BS], tl.float32)
    for c in tl.static_range(BS):
        col = tl.sum(tl.where(cc == c, A, 0.0), axis=1)
        piv = tl.sum(tl.where(offs == c, col, 0.0), axis=0)
        d = tl.sqrt(tl.maximum(piv, tau))
        l = tl.where(offs > c, col / d, 0.0)
        Lb = tl.where(cc == c, tl.where(offs == c, d, l)[:, None], Lb)
        A = A - l[:, None] * l[None, :]
    Xi = tl.zeros([BS, BS], tl.float32)
    for c in tl.static_range(BS):
        row = tl.sum(tl.where(rr == c, Lb, 0.0), axis=0)
        dc = tl.sum(tl.where(offs == c, row, 0.0), axis=0)
        row = tl.where(offs < c, row, 0.0)
        s = tl.sum(row[:, None] * Xi, axis=0)
        xc = (tl.where(offs == c, 1.0, 0.0) - s) / dc
        Xi = tl.where(rr == c, xc[None, :], Xi)
    tl.store(L + (j + rr) * n + (j + cc), Lb)
    tl.store(Li + (j + rr) * BS + cc, Xi)
    if PANEL:
        for i in range(j + BS, n, BS):
            P = tl.load(W + (i + rr) * n + (j + cc))
            P = tl.dot(P, tl.trans(Xi), input_precision=PREC)
            tl.store(L + (i + rr) * n + (j + cc), P)


def chol_blocked(G, tau, bs=64, prec="ieee", warps=4, panel_in_kernel=True):
    n = G.shape[0]
    assert n % bs == 0
    W = G.contiguous().clone()
    L = torch.zeros_like(W)
    Li = torch.empty(n, bs, device=G.device, dtype=G.dtype)
    for j in range(0, n, bs):
        _diag_panel_kernel[(1,)](W, L, Li, n, j, tau, BS=bs, PREC=prec, PANEL=panel_in_kernel, num_warps=warps)
        if j + bs < n:
            if not panel_in_kernel:
                torch.mm(W[j + bs:, j:j + bs], Li[j:j + bs].T, out=L[j + bs:, j:j + bs]) if False else \
                    L[j + bs:, j:j + bs].copy_(W[j + bs:, j:j + bs] @ Li[j:j + bs].T)
            P = L[j + bs:, j:j + bs]
            W[j + bs:, j + bs:].addmm_(P, P.T, alpha=-1.0)
    return L, Li


@triton.jit
def _trsm2_kernel(X, L, Li, Y, K, n, BM: tl.constexpr, BS: tl.constexpr, PREC: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows[:, None] < K
    offs = tl.arange(0, BS)
    rr = offs[:, None]
    cc = offs[None, :]
    for j in range(0, n, BS):
        acc = tl.load(X + rows[:, None] * n + (j + cc), mask=rmask, other=0.0)
        for k in range(0, j, BS):
            Yk = tl.load(Y + rows[:, None] * n + (k + cc), mask=rmask, other=0.0)
            Ljk = tl.load(L + (j + rr) * n + (k + cc))
            acc = acc - tl.dot(Yk, tl.trans(Ljk), input_precision=PREC)
        Xi = tl.load(Li + (j + rr) * BS + cc)
        Yj = tl.dot(acc, tl.trans(Xi), input_precision=PREC)
        tl.store(Y + rows[:, None] * n + (j + cc), Yj, mask=rmask)
        tl.debug_barrier()


def trsm2(X, L, Li, bm=32, bs=64, prec="ieee", warps=4):
    K, n = X.shape
    X, L, Li = X.contiguous(), L.contiguous(), Li.contiguous()
    Y = torch.empty_like(X)
    _trsm2_kernel[(triton.cdiv(K, bm),)](X, L, Li, Y, K, n, BM=bm, BS=bs, PREC=prec, num_warps=warps)
    return Y


@triton.jit
def _blockinv_kernel(L, Li, n, BS: tl.constexpr):
    """one program per diagonal block: Li[block b] = inverse of the lower-triangular block L[b, b]"""
    j = tl.program_id(0) * BS
    offs = tl.arange(0, BS)
    rr = offs[:, None]
    cc = offs[None, :]
    Lb = tl.load(L + (j + rr) * n + (j + cc))
    Xi = tl.zeros([BS, BS], tl.float32)
    for c in tl.static_range(BS):
        row = tl.sum(tl.where(rr == c, Lb, 0.0), axis=0)
        dc = tl.sum(tl.where(offs == c, row, 0.0), axis=0)
        row = tl.where(offs < c, row, 0.0)
        s = tl.sum(row[:, None] * Xi, axis=0)
        xc = (tl.where(offs == c, 1.0, 0.0) - s) / dc
        Xi = tl.where(rr == c, xc[None, :], Xi)
    tl.store(Li + (j + rr) * BS + cc, Xi)


def blockinv(L, bs=64):
    n = L.shape[0]
    L = L.contiguous()
    Li = torch.empty(n, bs, device=L.device, dtype=L.dtype)
    _blockinv_kernel[(n // bs,)](L, Li, n, BS=bs, num_warps=4)
    return Li


# ---- Gram G = Y^T Y: upper tiles only (mirrored), Kahan-compensated accumulation over K chunks ----

@triton.jit
def _gram_kernel(Y, G, K, n, BT: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr, KAHAN: tl.constexpr):
    ti = tl.program_id(0)
    tj = tl.program_id(1)
    if ti > tj:
        return
    oi = ti * BT + tl.arange(0, BT)
    oj = tj * BT + tl.arange(0, BT)
    ok = tl.arange(0, BK)
    acc = tl.zeros([BT, BT], tl.float32)
    comp = tl.zeros([BT, BT], tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + ok
        km = kk[:, None] < K
        A = tl.load(Y + kk[:, None] * n + oi[None, :], mask=km, other=0.0)     # [BK, BT] = Y[k, i]
        B = tl.load(Y + kk[:, None] * n + oj[None, :], mask=km, other=0.0)     # [BK, BT] = Y[k, j]
        part = tl.dot(tl.trans(A), B, input_precision=PREC)
        if KAHAN:
            yv = part - comp
            t = acc + yv
            comp = (t - acc) - yv
            acc = t
        else:
            acc += part
    tl.store(G + oi[:, None] * n + oj[None, :], acc)
    if ti != tj:
        tl.store(G + oj[:, None] * n + oi[None, :], tl.trans(acc))


def gram(Y, bt=64, bk=32, prec="tf32x3", kahan=True, warps=4, stages=3):
    K, n = Y.shape
    Y = Y.contiguous()
    G = torch.empty(n, n, device=Y.device, dtype=Y.dtype)
    nt = triton.cdiv(n, bt)
    _gram_kernel[(nt, nt)](Y, G, K, n, BT=bt, BK=bk, PREC=prec, KAHAN=kahan, num_warps=warps, num_stages=stages)
    return G


@triton.jit
def _gram_split_kernel(Y, P, K, n, KS, BT: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr):
    """partial Gram of rows [s KS, (s+1) KS) for upper tile (ti, tj) -> P[s] (Kahan over BK chunks)"""
    ti = tl.program_id(0)
    tj = tl.program_id(1)
    s = tl.program_id(2)
    if ti > tj:
        return
    oi = ti * BT + tl.arange(0, BT)
    oj = tj * BT + tl.arange(0, BT)
    ok = tl.arange(0, BK)
    acc = tl.zeros([BT, BT], tl.float32)
    comp = tl.zeros([BT, BT], tl.float32)
    k_end = tl.minimum(s * KS + KS, K)
    for k0 in range(s * KS, k_end, BK):
        kk = k0 + ok
        km = kk[:, None] < k_end
        A = tl.load(Y + kk[:, None] * n + oi[None, :], mask=km, other=0.0)
        B = tl.load(Y + kk[:, None] * n + oj[None, :], mask=km, other=0.0)
        part = tl.dot(tl.trans(A), B, input_precision=PREC)
        yv = part - comp
        t = acc + yv
        comp = (t - acc) - yv
        acc = t
    base = P + s * n * n
    tl.store(base + oi[:, None] * n + oj[None, :], acc)
    if ti != tj:
        tl.store(base + oj[:, None] * n + oi[None, :], tl.trans(acc))


def gram_split(Y, bt=32, bk=32, prec="ieee", target_programs=None, warps=4, stages=3):
    """G = Y^T Y, accurate (Kahan within a split, fp64 sum across splits), deterministic, half the tile work"""
    K, n = Y.shape
    Y = Y.contiguous()
    nt = triton.cdiv(n, bt)
    up = nt * (nt + 1) // 2
    tp = target_programs or 3 * torch.cuda.get_device_properties(Y.device).multi_processor_count
    S = max(1, min(triton.cdiv(tp, up), triton.cdiv(K, 4 * bk)))
    KS = triton.cdiv(triton.cdiv(K, S), bk) * bk
    S = triton.cdiv(K, KS)
    P = torch.empty(S, n, n, device=Y.device, dtype=torch.float32)
    _gram_split_kernel[(nt, nt, S)](Y, P, K, n, KS, BT=bt, BK=bk, PREC=prec, num_warps=warps, num_stages=stages)
    return P.sum(0, dtype=torch.float64).float() if S > 1 else P[0]


# ---- prep: largest column norm of X * s1 without forming it (s1 = power of 2 from max |x|) ----

@triton.jit
def _colsq_kernel(X, s1p, P, K, n, BR: tl.constexpr, BC: tl.constexpr):
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    rows = pr * BR + tl.arange(0, BR)
    cols = pc * BC + tl.arange(0, BC)
    s1 = tl.load(s1p)
    x = tl.load(X + rows[:, None] * n + cols[None, :], mask=(rows[:, None] < K) & (cols[None, :] < n), other=0.0) * s1
    tl.store(P + pr * n + cols, tl.sum(x * x, axis=0), mask=cols < n)


def colsq(X, s1, br=256, bc=64):
    """sum over rows of (X * s1)^2 per column (deterministic: per-block partials, then a fixed-order sum)"""
    K, n = X.shape
    nb = triton.cdiv(K, br)
    P = torch.empty(nb, n, device=X.device, dtype=torch.float32)
    _colsq_kernel[(nb, triton.cdiv(n, bc))](X, s1, P, K, n, BR=br, BC=bc, num_warps=4)
    return P.sum(0)


# ---- fused-input variants: the kernels read  s * A + eps * Z  (s: device scalar, Z: fixed dust) on the fly ----

@triton.jit
def _gram_split_f_kernel(Y, Z, sp, eps, P, K, n, KS, BT: tl.constexpr, BK: tl.constexpr, PREC: tl.constexpr,
                         HAS_S: tl.constexpr, HAS_Z: tl.constexpr):
    ti = tl.program_id(0)
    tj = tl.program_id(1)
    s = tl.program_id(2)
    if ti > tj:
        return
    oi = ti * BT + tl.arange(0, BT)
    oj = tj * BT + tl.arange(0, BT)
    ok = tl.arange(0, BK)
    if HAS_S:
        sc = tl.load(sp)
    acc = tl.zeros([BT, BT], tl.float32)
    comp = tl.zeros([BT, BT], tl.float32)
    k_end = tl.minimum(s * KS + KS, K)
    for k0 in range(s * KS, k_end, BK):
        kk = k0 + ok
        km = kk[:, None] < k_end
        A = tl.load(Y + kk[:, None] * n + oi[None, :], mask=km, other=0.0)
        B = tl.load(Y + kk[:, None] * n + oj[None, :], mask=km, other=0.0)
        if HAS_S:
            A = A * sc
            B = B * sc
        if HAS_Z:
            A += eps * tl.load(Z + kk[:, None] * n + oi[None, :], mask=km, other=0.0)
            B += eps * tl.load(Z + kk[:, None] * n + oj[None, :], mask=km, other=0.0)
        part = tl.dot(tl.trans(A), B, input_precision=PREC)
        yv = part - comp
        t = acc + yv
        comp = (t - acc) - yv
        acc = t
    base = P + s * n * n
    tl.store(base + oi[:, None] * n + oj[None, :], acc)
    if ti != tj:
        tl.store(base + oj[:, None] * n + oi[None, :], tl.trans(acc))


def gram_f(Y, s=None, Z=None, eps=0.0, bt=64, bk=32, prec="ieee", target_programs=None, warps=4, stages=3):
    """G = A^T A for A = s * Y + eps * Z, without forming A (Kahan within a split, fp64 across splits)"""
    K, n = Y.shape
    nt = triton.cdiv(n, bt)
    up = nt * (nt + 1) // 2
    tp = target_programs or 3 * torch.cuda.get_device_properties(Y.device).multi_processor_count
    S = max(1, min(triton.cdiv(tp, up), triton.cdiv(K, 4 * bk)))
    KS = triton.cdiv(triton.cdiv(K, S), bk) * bk
    S = triton.cdiv(K, KS)
    P = torch.empty(S, n, n, device=Y.device, dtype=torch.float32)
    _gram_split_f_kernel[(nt, nt, S)](Y, Z if Z is not None else Y, s if s is not None else Y, float(eps), P, K, n, KS,
                                      BT=bt, BK=bk, PREC=prec, HAS_S=s is not None, HAS_Z=Z is not None,
                                      num_warps=warps, num_stages=stages)
    return P.sum(0, dtype=torch.float64).float() if S > 1 else P[0]


@triton.jit
def _trsm_f_kernel(X, Z, sp, eps, L, Li, Y, K, n, BM: tl.constexpr, BS: tl.constexpr, PREC: tl.constexpr,
                   HAS_S: tl.constexpr, HAS_Z: tl.constexpr):
    """Y = A L^-T for A = s * X + eps * Z (formed on the fly)"""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows[:, None] < K
    offs = tl.arange(0, BS)
    rr = offs[:, None]
    cc = offs[None, :]
    if HAS_S:
        sc = tl.load(sp)
    for j in range(0, n, BS):
        acc = tl.load(X + rows[:, None] * n + (j + cc), mask=rmask, other=0.0)
        if HAS_S:
            acc = acc * sc
        if HAS_Z:
            acc += eps * tl.load(Z + rows[:, None] * n + (j + cc), mask=rmask, other=0.0)
        for k in range(0, j, BS):
            Yk = tl.load(Y + rows[:, None] * n + (k + cc), mask=rmask, other=0.0)
            Ljk = tl.load(L + (j + rr) * n + (k + cc))
            acc = acc - tl.dot(Yk, tl.trans(Ljk), input_precision=PREC)
        Xi = tl.load(Li + (j + rr) * BS + cc)
        Yj = tl.dot(acc, tl.trans(Xi), input_precision=PREC)
        tl.store(Y + rows[:, None] * n + (j + cc), Yj, mask=rmask)
        tl.debug_barrier()


def trsm_f(X, L, Li, s=None, Z=None, eps=0.0, bm=16, bs=32, prec="tf32x3", warps=4, stages=2):
    K, n = X.shape
    Y = torch.empty_like(X)
    _trsm_f_kernel[(triton.cdiv(K, bm),)](X, Z if Z is not None else X, s if s is not None else X, float(eps),
                                          L, Li, Y, K, n, BM=bm, BS=bs, PREC=prec, HAS_S=s is not None,
                                          HAS_Z=Z is not None, num_warps=warps, num_stages=stages)
    return Y
