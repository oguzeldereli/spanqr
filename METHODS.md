# How the orth methods work, in plain words

The job: X is r arrows in a K-dimensional room. We need r new arrows that are each length 1, all at right angles to
each other, and that reach exactly the same flat region ("span") as the old ones. Which particular arrows we return
does not matter (TTF re-expresses its accumulator in whatever basis it gets).

Two things make it hard:
- **Weak directions.** Some of X's arrows nearly lie on top of each other. What separates them is a sliver many
  millions of times smaller than the arrows themselves. fp32 keeps ~7 digits, so slivers below ~1e-7 of the biggest
  arrow are pure rounding noise. TTF lets those come out as any right-angled filler, but they must come out as
  proper length-1 arrows: never zero, never NaN.
- **The GPU wants big uniform jobs.** It is fast at "multiply two big blocks" (a GEMM: 0.05 ms here) and slow at
  long chains of small steps where each step waits for the previous one.

## Householder (what TTF uses now)

Picture a mirror. For the first column you place a mirror that reflects that arrow onto the first axis. You apply
the same mirror to every other arrow. Then you do the same for the second column, using only the leftover axes. And
so on, r times.

- **Why it is safe:** a mirror never stretches or squashes anything, so rounding errors never get amplified. Weak
  directions come out as accurate as fp32 allows, and exactly dependent columns still produce proper arrows.
- **Why it is slow:** it is r steps in a row, and each step depends on the one before. Every step is a skinny,
  memory-bound job, so the GPU waits most of the time. The real arithmetic is ~0.02 ms; the call takes ~2.5 ms.

## Cholesky-QR (fast, but breaks)

Instead of fixing arrows one at a time, measure all the overlaps at once. That gives the "Gram" table: how much
every arrow overlaps every other, one big GEMM. A small r x r factorization of that table (Cholesky) gives a recipe
for undoing the overlaps. Applying the recipe to X gives the clean arrows.

- **Why it is fast:** almost everything is big GEMMs.
- **Why it breaks:** the overlap table measures everything *squared*. A direction 10,000x weaker than the strongest
  shows up in the table 100,000,000x weaker. That is past fp32's 7 digits, so the table says that direction does not
  exist. The factorization then tries to take the square root of a negative number and returns NaN. This is exactly
  why the earlier fp32 version failed from condition ~1e4: 1e4 squared is 1e8, more than 7 digits.
- **"Do it twice" (Cholesky-QR2):** the first pass leaves the arrows nearly clean, the second pass polishes them. It
  only helps if the first pass did not already break.
- **fp64 versions:** 16 digits instead of 7, so it survives up to ~1e8 (1e8 squared is 1e16). But it still breaks on
  exactly dependent or zero columns, and consumer GPUs run fp64 at 1/64 speed.

## Shifted Cholesky-QR (the cushion)

Before factoring the overlap table, add a small cushion to every arrow's self-overlap. The factorization can then
never break. The price is that weak directions come out short instead of length 1, so you need another pass to
finish stretching them. Each pass can only stretch weak directions by about 1000x in fp32, so a direction 1e8x
weaker needs about three passes.

## Newton-Schulz (the nudge)

A GEMM-only recipe that nudges every arrow's length toward 1 and their angles toward 90 degrees. Near the answer it
converges very fast: the error is squared every step. Far from the answer it crawls: an arrow 1e6x too short grows
only ~1.5x per step. So it is great for final polishing and useless for the heavy lifting.

## TSQR (split the rows)

Cut X into horizontal slabs, clean each slab separately (in parallel), then merge the small results. It works well
when X is very tall and thin. TTF's shapes are either not thin enough (768 x 512), or the merge steps are again small
sequential jobs.

## What spanqr does: robust Cholesky-QR (candidates/kcholqr.py, `kcholf`)

Five tricks on top of Cholesky-QR:

1. **Rescale on the fly.** The kernels multiply X by an exact power of 2 as they read it, so its biggest entry is ~1.
   Huge values (TTF's 27B X reaches 1e9) or tiny ones can then never overflow or underflow the overlap table, and no
   scaled copy of X is written.
2. **Sprinkle dust.** A fixed, seeded random speck (5e-7 of the scale) is added to every column in the first solve.
   Where columns are exactly duplicated or zero, the speck becomes a random direction, which is precisely the "any
   filler" TTF allows. Elsewhere it is about as small as fp32 rounding, so it costs no accuracy. This is the fix for the
   failure that killed the earlier fast version (degenerate columns that locked subspaces for good).
3. **Cushion every pass, then three passes.** A small cushion (1e-7 of the table's size) means the factorization can
   never break, and the hand-written Cholesky also clamps any pivot that still comes out tiny. Each pass stretches
   the weak directions about 1000x more, so three passes cover everything fp32 can represent. With less cushion the
   fp32 factorization blows up, even when the overlap table is computed exactly.
4. **Apply the recipe by solving, not by multiplying with an inverse.** Solving a triangular system is careful
   arithmetic: the span stays as accurate as Householder's. Multiplying by an explicit inverse is sloppy when the
   recipe contains huge numbers; it gave span errors 25-60x Householder's on clustered spectra. Only the polish steps
   use plain multiplies.
5. **Polish** with two Newton-Schulz nudges, written as "arrows plus a small correction" so the multiply only rounds
   the correction. The last one measures the overlaps with compensated (Kahan) sums, so orthogonality lands at ~1e-7,
   below what fp32 Householder reaches.

**Accuracy:** passes all 208 acceptance cases and all 144 extra cases (condition up to 1e14, rank 1, zero blocks,
scales 1e-25 and 1e25) at the 8 TTF shapes. It passes all 528 real matrices dumped from TTF training. Plain fp32
Cholesky-QR2 fails 79 of those real ones.

## Where the time goes, and why it is not 5x

The small r x r factorization is a chain of dependent steps. cuSOLVER takes 0.44 ms for 512 x 512 (hundreds of tiny
kernels for ~0.02 ms of arithmetic); the hand-written cooperative kernel takes 0.21 ms, and its critical path is
16 dependent 32-column steps of ~13 us. At r = 512 (110M), three of them are most of the call. The same
factorization on a batch of 72 tables costs 0.043 ms per table, so batching layers would remove this wall.

For tall X (27B), every full pass over X costs about one fp32 GEMM (~0.13 ms at 17408 x 256). Three passes plus the
polish read and write X about 10 times. Fusing the next overlap table into the solve does not pay: each solve block
owns only 16-64 rows but would produce a full r x r partial table, and summing those costs more than the pass it saves
(atomic sums would break determinism).

Routes past this, all outside a pure orth replacement:

1. **Batch TTF's same-shape layers** (defer the 72 calls of a 110M step and run them together). The small
   factorizations become ~10x cheaper.
2. **Remove the squaring at the source.** X = V (U^T U) + u^T U. The U^T U term squares U's own conditioning. The
   same span is reached by X' = V (U^T Q_U) + u^T Q_U, where Q_U is an orthonormal basis of U's columns. X' has
   roughly the square root of X's condition number (5.6e8 becomes ~2e4). Keeping the accumulator in three factors
   (U = Q_U S, incremental-SVD style) would give that for free, and plain fp32 Cholesky-QR2 would then be safe.
3. **Warm start from V at later steps.** If late-training X is a small tilt of V, a cheap correction of V could
   replace a full factorization. The current dumps (steps 0, 1, 3) are too early to judge.
4. **One high-precision pass.** Not worth it on consumer GPUs: fp64 Cholesky there costs 0.4-1 ms, and the fp32 work
   around it saves only ~2 of the ~10 passes over X. On A100-class GPUs (fast fp64) it may be.
