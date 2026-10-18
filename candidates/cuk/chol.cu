// Lower Cholesky A = L L^T of a small dense matrix (n % 32 == 0, n <= ~1024) in ONE cooperative kernel, one grid
// barrier per 32-column block step.
//
// Step jb (j = 32 jb), on entry: columns >= j of A are updated through step jb-1, and L[j, j], Linv[j] are known.
// Every trailing tile (i, k), jb < k <= i, is updated A[i, k] -= P_i P_k^T with the panel P_i = A[i, j] Linv[j]^T
// recomputed on the fly by the tile's owner (cheap: removes a barrier and a round trip). The owner of tile (i, jb+1)
// also stores L[i, j] = P_i. CTA 0 owns tile (jb+1, jb+1): it updates it, then factors and inverts it at once
// (look-ahead), so the next step can start right after the barrier.
// A pivot below tau (a device scalar: no host sync) is clamped to tau ("modified Cholesky"); a NaN pivot also becomes tau (fmax ignores NaN). For
// Cholesky-QR any invertible factor keeps the span: the clamp only leaves weak directions short.
// Outputs: L [n, n] (lower triangle; the caller zero-fills the upper), Li [n, 32] inverses of the diagonal tiles.
// The double version also writes fp32 copies Lf, Lif.
// Deterministic (fixed per-element order); loads of data written by other CTAs bypass L1 (__ldcg).

#define TB 32
#define LD (TB + 1)
#ifdef CHOL_DEBUG
__device__ unsigned long long g_dbg[8192];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define STAMP(idx) do { if (threadIdx.x == 0 && blockIdx.x < 4) g_dbg[blockIdx.x * 2048 + (idx)] = gtime(); } while (0)
#else
#define STAMP(idx) do { } while (0)
#endif

__device__ __forceinline__ void grid_sync(unsigned* bar) {
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile unsigned* vgen = bar + 1;
        unsigned g = *vgen;
        __threadfence();
        if (atomicAdd(bar, 1u) == gridDim.x - 1) {
            atomicExch(bar, 0u);
            __threadfence();
            atomicAdd(bar + 1, 1u);
        } else {
            while (*vgen == g) { __nanosleep(64); }
        }
        __threadfence();
    }
    __syncthreads();
}

__device__ __forceinline__ float ldg_(const float* p) { return __ldcg(p); }
__device__ __forceinline__ double ldg_(const double* p) { return __ldcg(p); }
__device__ __forceinline__ float rsqrt_(float x) { return rsqrtf(x); }
__device__ __forceinline__ double rsqrt_(double x) { return rsqrt(x); }

template <typename T>
struct Smem {
    T inv[TB][LD];
    T a[TB][LD];
    T b[TB][LD];
    T p[TB][LD];
    T q[TB][LD];
    T rd[TB];
};

// 2 x 2 register-blocked tile product: acc[u][v] += sum_m X[r0+u][m] Y[c0+v][m]; thread -> (r0, c0) below
__device__ __forceinline__ int r0_() { return 2 * (threadIdx.x / 16); }
__device__ __forceinline__ int c0_() { return 2 * (threadIdx.x % 16); }

template <typename T>
__device__ __forceinline__ void mm22(T (*X)[LD], T (*Y)[LD], T acc[2][2]) {
    const int r0 = r0_(), c0 = c0_();
    #pragma unroll 8
    for (int m = 0; m < TB; ++m) {
        T x0 = X[r0][m], x1 = X[r0 + 1][m], y0 = Y[c0][m], y1 = Y[c0 + 1][m];
        acc[0][0] += x0 * y0; acc[0][1] += x0 * y1;
        acc[1][0] += x1 * y0; acc[1][1] += x1 * y1;
    }
}

// tile loads: all global loads into registers first, then the shared stores (the compiler does not move global loads
// past shared stores on its own, which serialized the round trips). Assumes blockDim.x == 256 (4 elements/thread).
template <typename T>
__device__ __forceinline__ void load_tile(T (*dst)[LD], const T* src, int ld) {
    T v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) { int e = threadIdx.x + q * 256; v[q] = ldg_(src + (size_t)(e / TB) * ld + e % TB); }
    #pragma unroll
    for (int q = 0; q < 4; ++q) { int e = threadIdx.x + q * 256; dst[e / TB][e % TB] = v[q]; }
}

template <typename T>
__device__ __forceinline__ void load_tiles3(T (*d0)[LD], const T* s0, int ld0, T (*d1)[LD], const T* s1, int ld1,
                                            T (*d2)[LD], const T* s2, int ld2, bool do0) {
    T v0[4], v1[4], v2[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int e = threadIdx.x + q * 256, r = e / TB, c = e % TB;
        if (do0) v0[q] = ldg_(s0 + (size_t)r * ld0 + c);
        v1[q] = ldg_(s1 + (size_t)r * ld1 + c);
        v2[q] = ldg_(s2 + (size_t)r * ld2 + c);
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int e = threadIdx.x + q * 256, r = e / TB, c = e % TB;
        if (do0) d0[r][c] = v0[q];
        d1[r][c] = v1[q];
        d2[r][c] = v2[q];
    }
}

// dst = src Linv^T (dst[r][c] = sum_m src[r][m] inv[c][m])
template <typename T>
__device__ __forceinline__ void panel(T (*dst)[LD], T (*src)[LD], T (*inv)[LD]) {
    T acc[2][2] = {{T(0), T(0)}, {T(0), T(0)}};
    mm22(src, inv, acc);
    const int r0 = r0_(), c0 = c0_();
    dst[r0][c0] = acc[0][0]; dst[r0][c0 + 1] = acc[0][1];
    dst[r0 + 1][c0] = acc[1][0]; dst[r0 + 1][c0 + 1] = acc[1][1];
}

// factor the (updated) tile in S.a, write L[j, j] and its inverse; leaves the inverse in S.inv
template <typename T, bool F32COPY>
__device__ void factor_diag(T* L, T* Li, float* Lf, float* Lif, int n, int j, T tau, Smem<T>& S) {
    const int tid = threadIdx.x;
    __syncthreads();
    // fp32: every warp factors the tile redundantly in registers (lane i holds row i) and only warp 0 stores. No branch
    // may surround the shuffles: on sm_120 a shuffle under `if (tid < 32)` or next to a branch gets wrapped in
    // WARPSYNC / convergence barriers, which made this loop 36x slower than on sm_89 (compute_90 code was unaffected).
    // fp64: warp 0 only (8 warps of redundant fp64 work would fight over the few fp64 units of consumer GPUs).
    constexpr bool RED = sizeof(T) == 4;
    const int i = tid & 31;
    T a[TB];
    T myrd = T(0);
    if (RED || tid < 32) {
        #pragma unroll
        for (int k = 0; k < TB; ++k) a[k] = S.a[i][k];
        #pragma unroll
        for (int c = 0; c < TB; ++c) {
            T piv = fmax(__shfl_sync(0xffffffffu, a[c], c), tau);
            T rd = rsqrt_(piv);
            T l = (i > c) ? a[c] * rd : ((i == c) ? piv * rd : T(0));
            myrd = (i == c) ? rd : myrd;
            a[c] = l;
            #pragma unroll
            for (int k = c + 1; k < TB; ++k) {
                T lk = __shfl_sync(0xffffffffu, l, k);
                a[k] -= ((i >= k) ? l : T(0)) * lk;
            }
        }
    }
    if (RED) __syncthreads();             // all warps have read S.a before warp 0 overwrites it
    if (tid < 32) {
        #pragma unroll
        for (int k = 0; k < TB; ++k) S.a[i][k] = (k <= i) ? a[k] : T(0);
        S.rd[i] = myrd;
    }
    __syncthreads();
    if (tid < 32) {                       // inverse: lane kk computes column kk by forward substitution
        const int kk = tid;
        T x[TB];
        #pragma unroll
        for (int i = 0; i < TB; ++i) {
            T acc0 = (i == kk) ? T(1) : T(0), acc1 = T(0), acc2 = T(0), acc3 = T(0);   // 4 chains: latency
            #pragma unroll
            for (int m = 0; m + 3 < i; m += 4) {
                acc0 -= S.a[i][m] * x[m]; acc1 -= S.a[i][m + 1] * x[m + 1];
                acc2 -= S.a[i][m + 2] * x[m + 2]; acc3 -= S.a[i][m + 3] * x[m + 3];
            }
            #pragma unroll
            for (int m = (i / 4) * 4; m < i; ++m) acc0 -= S.a[i][m] * x[m];
            x[i] = (i < kk) ? T(0) : ((acc0 + acc1) + (acc2 + acc3)) * S.rd[i];
        }
        #pragma unroll
        for (int i = 0; i < TB; ++i) S.inv[i][kk] = x[i];
    }
    __syncthreads();
    for (int e = tid; e < TB * TB; e += blockDim.x) {
        int r = e / TB, c = e % TB;
        L[(size_t)(j + r) * n + j + c] = S.a[r][c];
        Li[(size_t)(j + r) * TB + c] = S.inv[r][c];
        if (F32COPY) {
            Lf[(size_t)(j + r) * n + j + c] = (float)S.a[r][c];
            Lif[(size_t)(j + r) * TB + c] = (float)S.inv[r][c];
        }
    }
    __syncthreads();
}

template <typename T, bool F32COPY>
__device__ void chol_body(T* A, T* L, T* Li, float* Lf, float* Lif, unsigned* bar, int n, const T* taup) {
    __shared__ Smem<T> S;
    const T tau = *taup;
    const int tid = threadIdx.x, nt = blockDim.x;
    const int G = gridDim.x, b = blockIdx.x;
    const int nb = n / TB;
    if (b == 0) {
        load_tile(S.a, A, n);
        factor_diag<T, F32COPY>(L, Li, Lf, Lif, n, 0, tau, S);
    }
    STAMP(0);
    grid_sync(bar);
    for (int jb = 0; jb + 1 < nb; ++jb) {
        const int j = jb * TB, j1 = j + TB;
        STAMP(8 + jb * 8 + 0);
        if (b == 0) {
            // look-ahead: P = A[j1, j] Linv[j]^T, store L[j1, j], update A[j1, j1], factor it (S.inv -> Linv[j1])
            load_tiles3(S.p, (const T*)nullptr, 0, S.b, A + (size_t)j1 * n + j, n, S.a, A + (size_t)j1 * n + j1, n, false);
            __syncthreads();
            panel(S.p, S.b, S.inv);
            __syncthreads();
            {
                T acc[2][2] = {{T(0), T(0)}, {T(0), T(0)}};
                mm22(S.p, S.p, acc);
                const int r0 = r0_(), c0 = c0_();
                #pragma unroll
                for (int u = 0; u < 2; ++u)
                    #pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const int r = r0 + u, c = c0 + v;
                        S.a[r][c] -= acc[u][v];
                        L[(size_t)(j1 + r) * n + j + c] = S.p[r][c];
                        if (F32COPY) Lf[(size_t)(j1 + r) * n + j + c] = (float)S.p[r][c];
                    }
            }
            STAMP(8 + jb * 8 + 1);
            factor_diag<T, F32COPY>(L, Li, Lf, Lif, n, j1, tau, S);
            STAMP(8 + jb * 8 + 2);
        }
        if (G == 1 || b > 0) {
            const int W = (G > 1) ? G - 1 : 1, me = (G > 1) ? b - 1 : 0;
            bool have_inv = false;
            int t = 0;
            for (int ib = jb + 1; ib < nb; ++ib) {
                for (int kb = jb + 1; kb <= ib; ++kb) {
                    if (ib == jb + 1 && kb == jb + 1) continue;
                    if (t++ % W != me) continue;
                    const int i = ib * TB, kc = kb * TB;
                    const int r0 = r0_(), c0 = c0_();
                    T cur[2][2];                      // issue every load of the tile at once (one L2 round trip)
                    #pragma unroll
                    for (int u = 0; u < 2; ++u)
                        #pragma unroll
                        for (int v = 0; v < 2; ++v) cur[u][v] = ldg_(A + (size_t)(i + r0 + u) * n + kc + c0 + v);
                    load_tiles3(S.inv, Li + (size_t)j * TB, TB, S.a, A + (size_t)i * n + j, n,
                                S.b, A + (size_t)kc * n + j, n, !have_inv);
                    have_inv = true;
                    __syncthreads();
                    panel(S.p, S.a, S.inv);
                    panel(S.q, S.b, S.inv);
                    __syncthreads();
                    {
                        T acc[2][2] = {{T(0), T(0)}, {T(0), T(0)}};
                        mm22(S.p, S.q, acc);
                        #pragma unroll
                        for (int u = 0; u < 2; ++u)
                            #pragma unroll
                            for (int v = 0; v < 2; ++v) {
                                const int r = r0 + u, c = c0 + v;
                                A[(size_t)(i + r) * n + kc + c] = cur[u][v] - acc[u][v];
                                if (kb == jb + 1) {
                                    L[(size_t)(i + r) * n + j + c] = S.p[r][c];
                                    if (F32COPY) Lf[(size_t)(i + r) * n + j + c] = (float)S.p[r][c];
                                }
                            }
                    }
                    __syncthreads();
                }
            }
            if (G == 1) {                     // single CTA: restore Linv[j1] for the next look-ahead
                load_tile(S.inv, Li + (size_t)j1 * TB, TB);
                __syncthreads();
            }
        }
        STAMP(8 + jb * 8 + 3);
        grid_sync(bar);
        STAMP(8 + jb * 8 + 4);
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) chol_f32(float* A, float* L, float* Li, unsigned* bar, int n,
                                                              const float* tau) {
    chol_body<float, false>(A, L, Li, nullptr, nullptr, bar, n, tau);
}

extern "C" __global__ void __launch_bounds__(256, 2) chol_f64(double* A, double* L, double* Li, float* Lf, float* Lif,
                                                           unsigned* bar, int n, const double* tau) {
    chol_body<double, true>(A, L, Li, Lf, Lif, bar, n, tau);
}
