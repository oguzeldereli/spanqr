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

## Results

**Accuracy** (RTX 4080 Laptop, Ada; the same synthetic suite and a real subset also pass on an RTX 5060 Ti, Blackwell):
- **Synthetic:** all 208 acceptance cases and all 144 extra robustness cases at the 8 TTF shapes (2 seeds), with
  condition numbers up to 1e14, rank 1, zero blocks, duplicated columns, and scales of 1e-25 and 1e25.
- **Real:** all 528 X matrices dumped from TTF training runs (110M x 2 runs and 27B).
- **Precision:** worst orthogonality max |Q^T Q - I| = 1.4e-7. Span error at most 0.8x Householder's on synthetic
  data and 2.1x on real data (limit: 10x).
- **Hygiene:** deterministic, no host sync, input untouched.

**Speed** (fp32, median of back-to-back calls; the laptop's boost clock moves numbers by ~20% between runs):

| K x r (TTF layer) | Householder, 4080 Laptop | spanqr, 4080 Laptop | Householder, 5060 Ti | spanqr, 5060 Ti |
|---|---|---|---|---|
| 768 x 512 (110M) | 1.9-2.1 ms | 1.1 ms (1.6-1.9x) | 2.02 ms | 1.09 ms (1.9x) |
| 2048 x 512 (110M down_proj) | 2.7 ms | 1.4-1.9 ms (1.4-1.9x) | 2.89 ms | 1.54 ms (1.9x) |
| 1024 x 256 (0.8B) | 1.09 ms | 0.47-0.53 ms (2.1-2.3x) | 1.06 ms | 0.87 ms (1.2x)* |
| 2048 x 256 (0.8B) | 1.32 ms | 0.57-0.64 ms (2.1-2.3x) | 1.31 ms | 0.87 ms (1.5x)* |
| 3584 x 256 (0.8B) | 2.9 ms | 0.79-0.82 ms (3.5-3.7x) | 2.04 ms | 0.88 ms (2.3x) |
| 5120 x 256 (27B) | 2.3 ms | 0.83-1.04 ms (2.2-2.8x) | 2.31 ms | 0.94 ms (2.5x) |
| 6144 x 256 (27B) | 2.7 ms | 0.93-1.13 ms (2.5-2.9x) | 2.66 ms | 1.04 ms (2.6x) |
| 17408 x 256 (27B down_proj) | 5.1 ms | 1.9-2.8 ms (1.8-2.7x) | 5.77 ms | 2.34 ms (2.5x) |

\* CPU-bound on that machine: about 0.9 ms of Python launch work per call on its i5-11400F (0.5 ms on an i9-14900HX).
A CUDA graph per shape would remove it.

At 110M the QR time per training step drops from ~170-180 ms to ~100 ms.

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
| `candidates/cuk/chol.cu` | Lower Cholesky with pivot clamping in **one cooperative kernel** (all SMs, one grid barrier per 32-column step, look-ahead factor of the next diagonal tile in one warp's registers, diagonal-tile inverses as a by-product). float and double. | 4080 Laptop r=256: 0.08-0.09 ms (cuSOLVER 0.20), r=512: 0.22-0.24 ms (cuSOLVER 0.44); 5060 Ti r=512: 0.16 ms (cuSOLVER 0.32) |
| `candidates/tri.py: trsm_f` | Triangular solve Y = (s X + eps Z) L^-T, rows in parallel, tf32x3 tensor-core dots (fp32-accurate), scale and dust fused into the load | 3-4x faster than cuBLAS trsm |
| `candidates/tri.py: gram_f, gram_split` | Gram A^T A: upper tiles only, Kahan-compensated within a K split, fp64 sum across splits, optional fused scale/dust | 5-10x more accurate than cuBLAS sgemm |
| `candidates/cudak.py` | Runtime build/launch: **NVRTC** (the wheel that ships with PyTorch) and the CUDA driver API via `cuda-python`, on PyTorch's current stream | no CUDA toolkit or host compiler needed |

The CUDA kernels compile on first use (a few seconds) and are cached in `candidates/cuk/build/`.

A Blackwell lesson baked into `chol.cu`: on sm_120 the compiler wraps a warp shuffle in `WARPSYNC`/convergence
barriers when it sits under a condition like `if (tid < 32)` or next to a branch (724 `WARPSYNC`s vs 5 on sm_89). That
made the in-register 32 x 32 tile factor 36x slower (Cholesky 1.30 ms instead of 0.16 ms), while the same source built as
compute_90 PTX ran fine on the same card. The fp32 tile factor is now branch-free and run by every warp (only warp 0
stores).

**Requirements:** PyTorch with CUDA, Triton (which needs a host C compiler such as `gcc` for its launch stubs), and
`cuda-python`. NVRTC builds a native cubin for every architecture it
knows (NVRTC 12.6 from cu126 PyTorch: up to sm_90; NVRTC >= 12.8 from cu128 PyTorch: also Blackwell sm_120). For a GPU
newer than the local NVRTC it falls back to PTX for the newest known architecture, which the driver JIT-compiles.

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

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free for noncommercial use (research, personal, education, nonprofits);
commercial use needs a separate license.

Required Notice: Copyright (c) 2026 Ozgur Guzeldereli (https://github.com/oguzeldereli/spanqr)
