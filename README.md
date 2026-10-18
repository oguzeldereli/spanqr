# spanqr

A fast orthonormal basis for the column span of a tall fp32 matrix on consumer GPUs that never degenerates. It is a
robust Cholesky-QR with hand-written CUDA and Triton kernels, built as a drop-in replacement for `torch.linalg.qr(X)[0]`
in the [TTF](https://github.com/oguzeldereli/ttf-ternary) optimizer.

```python
import sys; sys.path.insert(0, "/path/to/spanqr")
from orth import orth

Q = orth(X)   # X: fp32 CUDA [K, r]  ->  Q: fp32 CUDA [K, r], Q^T Q = I, span(Q) = span(X)
```

## Why this exists

[TTF (two-timescale threshold flips)](https://github.com/oguzeldereli/ttf-ternary) trains ternary ({-1, 0, 1})
language models without full-precision master weights. Each ternary layer (weight N x K) keeps a rank-r accumulator
of Adam-normalised gradients as two factors, U (N x r) and V (K x r, orthonormal). Every training step, inside every
layer's backward pass, the update forms a tall matrix

    X = V (U^T U) + u^T U          (u: the layer's normalised gradient)

and needs **an orthonormal basis of its column span**. The accumulator is then re-expressed in that basis.

Until now that was `torch.linalg.qr(X)[0]` (cuSOLVER Householder). That is robust but slow at TTF's shapes. At 110M
the QR was ~80% of the whole TTF update and ~17% of the GPU step, because cuSOLVER's panel kernels are latency-bound:
a 768 x 512 QR takes ~2.5 ms where one GEMM of the same size takes 0.05 ms.

The obvious fast replacement, Cholesky-QR, failed in training. X is often badly conditioned (condition numbers up to
5.6e8 were observed, because U^T U squares U's conditioning), and sometimes exactly rank-deficient or zero. fp32
Cholesky-QR then returned degenerate columns. Those subspaces stayed at zero for the rest of training: 88 of 168 locked,
against 0 with Householder. fp64 Cholesky-QR survives conditioning to ~1e8 but still breaks on rank-deficient input,
and consumer GPUs run fp64 at 1/64 speed.

So the requirement was **Householder's robustness at Cholesky-QR's GEMM-shaped cost**, with no host syncs (they cost
~30% GPU idle per layer before) and bit-for-bit determinism.

## How it works

The plain-words version, including why every earlier method failed, is in [METHODS.md](METHODS.md). In short:

1. **Scale on the fly.** X is multiplied by an exact power of 2 as the kernels load it, so the Gram matrix can never
   overflow or underflow.
2. **Dust.** A fixed, seeded random speck (5e-7 of the scale) is added in the first solve. Where columns are exactly
   dependent or zero it becomes the arbitrary orthonormal completion the update allows. Elsewhere it is below fp32
   rounding.
3. **Three shifted Cholesky-QR passes.** Each pass computes the Gram matrix, adds a shift of 1e-7 ||G||_F (the
   factorization cannot break down), factors it with a hand-written CUDA Cholesky that also clamps tiny pivots, and
   applies the factor with a *triangular solve*, not an inverse multiply. The solve keeps the span as accurate as
   Householder's. Each pass stretches the weak directions by ~1000x.
4. **Two Newton-Schulz polish steps,** written as a small correction, the last one with a compensated (Kahan)
   split-K Gram. Orthogonality reaches ~1e-7, below fp32 Householder's.

## Results (RTX 4080 Laptop, fp32)

**Accuracy:**
- **Synthetic:** all 208 acceptance cases and all 144 extra robustness cases at the 8 TTF shapes (2 seeds), with
  condition numbers up to 1e14, rank 1, zero blocks, duplicated columns, and scales of 1e-25 and 1e25.
- **Real:** all 528 X matrices dumped from TTF training runs (110M x 2 runs and 27B).
- **Precision:** worst orthogonality max |Q^T Q - I| = 1.4e-7. Span error at most 0.8x Householder's on synthetic
  data and 2.1x on real data (limit: 10x).
- **Hygiene:** deterministic, no host sync, input untouched.

**Speed** (median, back-to-back calls; the laptop's boost clock moves numbers by ~20% between runs):

| K x r (TTF layer) | Householder | spanqr | speedup |
|---|---|---|---|
| 768 x 512 (110M) | 2.08 ms | 1.10 ms | 1.9x |
| 2048 x 512 (110M down_proj) | 2.69 ms | 1.43-1.89 ms | 1.4-1.9x |
| 1024 x 256 (0.8B) | 1.08 ms | 0.47 ms | 2.3x |
| 2048 x 256 (0.8B) | 1.32 ms | 0.57 ms | 2.3x |
| 3584 x 256 (0.8B) | 2.88 ms | 0.79 ms | 3.7x |
| 5120 x 256 (27B) | 2.32 ms | 0.83 ms | 2.8x |
| 6144 x 256 (27B) | 2.73 ms | 0.93-1.10 ms | 2.5-2.9x |
| 17408 x 256 (27B down_proj) | 5.04 ms | 1.88-2.71 ms | 1.9-2.7x |

At 110M the QR time per training step drops from ~180 ms to ~100 ms.

**Target vs reached.** The goal was 5x. What limits it:
- **r = 512 (110M):** three sequential 512 x 512 Choleskys cost ~0.2 ms each even hand-written. The critical path is
  16 dependent 32-column steps.
- **Large K (27B):** each of the ~10 remaining full passes over X costs about one fp32 GEMM (~0.13 ms at
  17408 x 256).
- **Fewer passes:** shifts below 1e-7 make fp32 Cholesky blow up even with an exact Gram matrix, and an fp64 Cholesky
  is too slow on consumer GPUs (0.4-1 ms). Batching TTF's same-shape layers, or keeping U orthonormal so U^T U stops
  squaring the conditioning, are the routes past this; both are changes on the TTF side.

## Kernels

| file | kernel | vs library |
|---|---|---|
| `candidates/cuk/chol.cu` | Lower Cholesky with pivot clamping in **one cooperative kernel** (all SMs, one grid barrier per 32-column step, look-ahead factor of the next diagonal tile in one warp's registers, diagonal-tile inverses as a by-product). float and double. | r=256: 0.07 ms (cuSOLVER 0.20); r=512: 0.21 ms (cuSOLVER 0.44) |
| `candidates/tri.py: trsm_f` | Triangular solve Y = (s X + eps Z) L^-T, rows in parallel, tf32x3 tensor-core dots (fp32-accurate), scale and dust fused into the load | 3-4x faster than cuBLAS trsm |
| `candidates/tri.py: gram_f, gram_split` | Gram A^T A: upper tiles only, Kahan-compensated within a K split, fp64 sum across splits, optional fused scale/dust | 5-10x more accurate than cuBLAS sgemm |
| `candidates/cudak.py` | Runtime build/launch: **NVRTC** (the wheel that ships with PyTorch) and the CUDA driver API via `cuda-python`, on PyTorch's current stream | no CUDA toolkit or host compiler needed |

The CUDA kernels compile on first use (a few seconds) and are cached in `candidates/cuk/build/`.

**Requirements:** PyTorch with CUDA, Triton, and `cuda-python`. The NVRTC 12.6 that ships with cu126 PyTorch builds for
up to sm_90. Blackwell (sm_120) needs NVRTC >= 12.8, e.g. from a cu128 PyTorch.

## Requirements it was built against

- **Output:** Q only (R unused), K x r with orthonormal columns; any basis of span(X) is fine (column order and
  signs free).
- **Conditioning:** X conditioned up to ~1e9 must work. Directions above 1e-6 of the largest singular value must lie
  in span(Q) about as accurately as fp32 Householder (pass: within 10x). Weaker directions may be any orthonormal
  completion.
- **Rank deficiency:** rank-deficient, duplicated or all-zero X must give a full orthonormal Q, never zero or NaN
  columns.
- **Orthogonality:** max |Q^T Q - I| <= 1e-6.
- **Determinism:** bit-for-bit across runs on the same GPU (TTF tests compare a resumed run's losses exactly).
- **No host synchronisation:** no `.item()`, no data-dependent CPU branching.
- **Interface:** inputs and outputs fp32 CUDA [K, r]. Calls are sequential, one per layer inside its backward pass.
- **Shapes:** 110M: 768/2048 x 512; Qwen3.5-0.8B: 1024/2048/3584 x 256; Qwen3.8-27B: 5120/6144/17408 x 256.

## Test and benchmark harness

    python -m qrbench.check --list                                      # methods and test cases
    python -m qrbench.check -m householder kcholf                       # acceptance shapes, all cases
    python -m qrbench.check -m kcholf --shapes all --seeds 2 --json out.json
    python -m qrbench.check -m kcholf --real dumps/ --real-only         # matrices dumped from training
    python -m qrbench.bench -m kcholf --shapes all                      # speed vs Householder and one GEMM
    python -m pytest -q tests/                                          # harness self-tests

- **`qrbench/gen.py`:** test matrices.
  - Conditioning is mixed on both sides: X = Q1 diag(s) Q2^T. Column-scaled test matrices are benign for Cholesky
    methods and gave misleading results in earlier attempts.
  - Cases include rank-deficient, duplicated and zero inputs, and TTF-structured X = V B + P.
- **`qrbench/metrics.py`:** the pass criteria.
  - Span is measured in units of fp32 rounding against an fp64 reference, and compared to Householder on the same
    matrix.
  - It also checks orthogonality, degenerate or NaN columns, bitwise determinism, host syncs (CUDA sync-debug mode)
    and that the input is untouched.
- **`qrbench/dump.py`:** captures real X (and the previous basis V) from a TTF training run by patching
  `ttf.update._orth_` at runtime, with no TTF file changed.
- **`candidates/`:** every method tried.
  - `rcholqr.py`: the robust recipe on cuSOLVER/cuBLAS.
  - `rchol_fast.py`, `kcholqr.py`: kernel versions.
  - `kcholf` is the one `orth.py` uses.
- **`results/`:** saved runs.
