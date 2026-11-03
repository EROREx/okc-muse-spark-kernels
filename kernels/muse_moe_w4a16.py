# okc: problem=muse_moe_w4a16
# Derived from okc-2026 (https://github.com/vjeux/okc-2026), licensed under the Apache License 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Modifications by EROREx (K-permuted WGS packing; WGS final waits per token on per-expert phase-B counters instead of a grid barrier; lossless 5-bit block-scale compression of the streamed weight copies (SC5); WGS main build from the decode layer's MoE source: two-copy WGS flags with epilogue sums of squares and a one-pass final (v150), weights-warp generations and warp-scan stage ranges plus a coalesced tile gather (A300), per-token-count cost tables; four helper warps, one per SM sub-partition, each summing and storing a quarter of every unit's epilogue with lean per-unit bookkeeping, in the TINY / TINY2 / MID builds, and in the MID build the ring refills issued by the helper warps instead of thread 0); this file is distributed under Apache-2.0.
# Further modifications by EROREx, 2026-11-01 (v085, safety only): every lane acquires the phase-A flags it reads
# behind; fence.proxy.async.global between the flag acquires and the bulk copies of xA / xB; cooperative launch;
# a correct (slow) PyTorch path for any other local expert range and a prepare-time self-test of every build;
# (v090) the builds compile in parallel threads.
"""Routed experts (NVFP4 W4A16 MoE) as one persistent CUDA kernel.

Algorithm
---------
For each token, up to ``TOP_K`` of the 256 experts are active, and this rank
owns ``LOCAL_EXPERTS`` of them. The work is a gated MLP per expert:

1.  Routing (``route`` in the CUDA source): every CTA redundantly builds the
    same per-expert token lists from ``topk_ids`` -- counts, offsets, and a
    sorted ``(token, slot)`` entry list -- plus a job list. An expert with
    more than ``SKINNY_MAX_TOKENS`` tokens becomes one or more *fat* jobs
    (chunks of ``FAT_MAX_TOKENS`` tokens); the rest become *skinny* jobs.
    Fat jobs are listed first, largest first.
2.  Phase A (gate + up, ``GEMM1 + SwiGLU``): each job's tokens are
    multiplied by the expert's packed NVFP4 gate/up weights. A *skinny* unit
    is 16 rows x full K with warps splitting K and ``mma.sync``; a *fat*
    unit is 256 rows x full K streamed through shared memory in K-stages.
    The SwiGLU output lands in ``inter`` (bf16).
3.  Phase B (down, ``GEMM2``): the same jobs run again with the expert's
    down weights, reading ``inter``. The epilogue multiplies by
    ``1 + post_norm_w`` (the post-expert RMSNorm weight, kept bf16 as given)
    and writes fp32 rows to ``ybuf`` plus each route's sum of squares to
    ``ss`` (double-buffered per launch generation).
4.  Final pass: ``out[t] = sum_s topk_w[t, s] * rsqrt(ss[pos] / HIDDEN + eps)
    * ybuf[pos]``. A grid barrier between the phases and the final pass
    makes every CTA's phase-B writes visible to every CTA.

Scheduling: one persistent grid (one CTA per SM) runs both phases. Fat
units go first (round-robin over CTAs); skinny units are split by cost so
every CTA carries the same total cost, with a small run-time pool of tail
units claimed atomically (stream-K). With no fat jobs, phase A's last ring
refills already prefetch this CTA's first phase-B units (cross-phase
prefill).

Which kernel runs depends on the batch: ``kernel_fn`` picks one of four
compile-time variants of the same source (``MAIN`` / ``TINY`` / ``TINY2`` /
``MID``), selected by token count; the large-batch variant additionally
switches to an RS-wgmma implementation (``OKC_WGS``) for the main band.
Weights are only repacked (bit/byte permutations) in ``prepare`` -- never
requantized -- into the four layouts the kernel reads: packed
mma.sync order, fat order, unit-contiguous order, and natural-k / WGS
order.

Constraints the code preserves (OKC rules)
------------------------------------------
* One Python file; native CUDA only, via ``torch.utils.cpp_extension.
  load_inline``.
* CUDA-graph capturable: no host synchronisation in the timed path.
  ``kernel_fn`` allocates a fresh ``out`` every call, reads no tensor
  back to host, and launches only ``okc_run``. Host syncs exist only in
  ``prepare`` (one-time setup) and in the ``DIAG`` diagnostics, both
  outside the timed path.
* Fresh inputs each call, no caching across calls: the only
  module-level cache is ``_EXTS``, which maps a compile-time define
  tuple to its compiled extension module. It caches compiled code,
  never activations, weights, or outputs.
"""

import concurrent.futures  # okc-allow: optics: the threads only parallelize load_inline (prepare's compiles)
import hashlib
import time
from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


# ---------------------------------------------------------------- model shape
# These are the task's model dimensions, not tunables. The CUDA source has
# its own copies (HD / ID) because it is compiled standalone; keep them in
# sync by hand -- there is no mechanism to share a constant across the
# Python/CUDA boundary in a load_inline submission.
HIDDEN = 4096            # d_hidden and d_expert (the MoE is square)
TOP_K = 8                # routes per token
N_EXPERTS_TOTAL = 256    # experts across all ranks
LOCAL_START = 0          # first expert owned by this rank
LOCAL_EXPERTS = 32       # experts owned by this rank (ep8 sharding)
SHARDING = "ep8"

# ------------------------------------------------------- per-call chunking
# kernel_fn runs the kernel in slices of at most this many tokens: the
# kernel's scratch (inter / ybuf / ss) is sized for this bound.
CHUNK_TOKENS = 384
EMAX = CHUNK_TOKENS * TOP_K          # max (token, route) entries per chunk
JMAX = 32 + EMAX // 64 + 8           # max jobs per chunk (upper bound)

# ------------------------------------------------------------- WGS scratch
# Stream-K fixup area: mirror of the CUDA-side WGS_FIXF / WGS_XROWS.
WGS_SLOTS = 160                      # fixup slots per phase (>= grid + 1)
WGS_FIXW = 4096                      # floats per (slot, warpgroup) fixup
WGS_FIXF = 2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16 * 2  # (v150: two doneAs copies)
# WGS main build ("@wgh"): runtime mode bits in okc_run's grid argument (bit 0: the final waits per token on its
# experts' phase-B counters, no grid barrier) and the cost table per token count (shape only): (M hi, A, B) packed
# 8 x 8 bits, tiles 1 in the low byte (decode layer gen_v390: Modal A/B steep d32, half v16, fit v32)
WGH_MODE = 1
WGS_TABLES = ((64, '0xfef4deccb59a9270', '0xd3b8a694867a6a64'), (191, '0xcac0b2a699888370', '0xbdac9f93877b6f64'), (100000, '0xc5b8aa9f94837e70', '0xccbcada193827564'))
WGS_XROWS = EMAX + 8 * (32 + EMAX // 32 + 8)

# ------------------------------------------- compile-time variant selection
# The four variants are the same CUDA source compiled with different
# -D flags -- i.e. different tile / scheduling choices, not different code.
# Flag tuples are part of the submission's tuned behaviour; do not edit
# them without re-gating on GPU. kernel_fn picks by token count M.
MAIN_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1", "-DOKC_A300=3", "@wgh")
# M <= TINY_MAX_M: skinny-only build; a job never has more than M tokens,
# so there are no fat units.
TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2",
                "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
                "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1")
TINY_MAX_M = 4
# TINY_MAX_M < M <= TINY2_MAX_M: a second skinny-only build with a shorter
# helper-warp tail.
TINY2_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_CPF=0", "-DOKC_EPI4=0",
                 "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1",
                 "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=1", "-DOKC_FINPF=1")
TINY2_MAX_M = 8
# TINY2_MAX_M < M <= MID_MAX_M: tiny-style build with experts of more than
# SKINNY_MAX_TOKENS tokens split into <= SKINNY_MAX_TOKENS-token jobs.
MID_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_SPLIT=1", "-DOKC_CPF=0",
               "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0",
               "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2",
               "-DOKC_FINPF=1")
MID_MAX_M = 16
# M 17..63 runs on the WGS MAIN build; M >= 64 is MAIN by default below.
# The skinny ring's slot count when the caller does not pin -DOKC_NS.
MAIN_NS = 3
if any(d.startswith('@') for d in TINY_DEFINES + TINY2_DEFINES + MID_DEFINES):
    raise RuntimeError("TINY / TINY2 / MID defines: no alternate source expected before the HSP suffix")
TINY_DEFINES = TINY_DEFINES + ('-DOKC_HSP=2', '@hsp')  # gen_v070 hsp2tm
TINY2_DEFINES = TINY2_DEFINES + ('-DOKC_HSP=2', '@hsp')  # gen_v070 hsp2tm
MID_DEFINES = MID_DEFINES + ('-DOKC_HSP=2', '-DOKC_HTMA=1', '@hsp')  # gen_v070 hsp2tm

# The M >= 64 build used when the scale bytes rule out the PRMT decode
# (the WGS kernel needs it): the mma.sync kernel with the same shapes.
MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7",
                         "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0")

# ------------------------------------------------------------- feature flags
# Programmatic dependent launch for embedders whose graph runs this kernel
# right after other kernels (the kernel waits on griddepcontrol.wait before
# touching their outputs). Off: one launch per call gains nothing.
PDL = False
# Scale decode with PRMT + IMAD instead of the smem LUT; exact only when
# every block-scale byte of the local experts is a positive normal e4m3
# value (0x08..0x7E). prepare() checks the actual weights and keeps the
# LUT build otherwise.
PRMT_SCALES = True
# SC5: store the streamed copies' block scales as 5-bit offsets from per-group bases (lossless; see the CUDA SC5
# comment and _sc5_*). prepare() verifies every group of the actual weights fits 32 codes, else it keeps the
# 8-bit layouts and the builds without -DOKC_SC5 (any input stays exact).
SC5 = True
SC5_BYTES = 35552   # bytes per unit / stage bulk copy (mirror of the CUDA SC5_BYTES)
SC5_STRIDE = 35584  # their pitch in global memory (mirror of SC5_STRIDE)
# DQ7: the bit layout whose decode is 7 ALU ops + 2 IMAD.SHL per 8 weights
# (the kernel gets -DOKC_DQ7=1 via _ext).
DQ7 = True

# ---------------------------------------------------------------- diagnostics
# All off in a submission run. DIAG makes prepare() print an H200 timing
# table (untimed); DIAG_BW additionally runs the BW_SRC microbenchmarks.
DIAG = False
DIAG_VARIANTS = {}         # name -> defines, timed in diag mode
DIAG_BW = False
DIAG_BW_MODES = (5, 8, 12, 16) + tuple(range(17, 25))

# Extension-point for out-of-tree experiments: "@name" in a defines tuple
# swaps the compiled source to ALT_SRCS[name]. Empty in a submission.
ALT_SRCS = {}

# Bit layout of a 32-bit word holding 8 E2M1 codes for one MMA k-step of
# one lane: register R (0..3) is (h, j0) = (0,0) (1,0) (0,2) (1,2): rows
# g+8h, k = base + 4s + j0 (+1 for the upper bf16). Code bits (m, e0, e1, s)
# land at output bits (6, 7, 8, 15) (+16 for the upper half) after the
# per-register rotation used by dq4().
_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


CUDA_SRC = r'''
// ---------------------------------------------------------------- roadmap
// Compile-time knobs (the OKC_* values the host actually passes, plus
// defaults for the rest) -> Params / Smem -> small helpers (dequant,
// mma, mbarrier, L2 hints) -> route() -> skinny ring (two variants:
// dynamic-pool SKDYN / static, see below) -> fat stream (cp.async +
// mma.sync, wgmma variant) -> WS helper warp -> WGS (RS-wgmma,
// producer/consumer) -> run_phase -> moe_fused -> C++ launcher okc_run.
// v2: dead #if branches pruned for the only builds the host compiles
// (MAIN / TINY / TINY2 / MID, each with and without -DOKC_PRMTSC=1, plus
// the MAIN fallback; BW with no -D), sm_90a only; dead functions,
// defines and BW switch cases removed. Taken-path code is unchanged.
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#ifndef OKC_NACC
#define OKC_NACC 2
#endif
#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores
                     // OKC_FTSC2; the TINY builds measured -0.5%, so off by default)
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#define OKC_TOKQ 1  // wewillcontrol w1xx: WGS final waits per token on per-expert phase-B counters (no grid barrier)
#define OKC_WGSFINSS 1  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
#endif
#define OKC_WGMIN 9
#define OKC_XPRE 1
#define FAST1_ARCH 1
#ifndef OKC_FAST1
#define OKC_FAST1 3
#endif
#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)
#ifndef OKC_CPFF
#define OKC_CPFF 1
#endif
#ifndef OKC_FCOSTBB
#define OKC_FCOSTBB OKC_FCOSTB
#endif
#define OKC_FCOSTSB OKC_FCOST
#define OKC_DYN 1
#ifndef OKC_DYNK
#define OKC_DYNK 4
#endif
#define OKC_TDYN 0
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#define OKC_FT_T 64
#ifndef OKC_SPLIT
#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)
#endif
#ifndef OKC_TINY
#define OKC_TINY 0
#endif
#ifndef OKC_NW
#define OKC_NW 8
#endif
#ifndef OKC_EARLYN
#define OKC_EARLYN 1
#endif
#define OKC_BIGM 64
#define OKC_XPF 0
#ifndef OKC_CPF
#define OKC_CPF 1
#endif
#define OKC_FATSTATIC 1
#define OKC_FCOST 40
#ifndef OKC_FCOSTB
#define OKC_FCOSTB 40
#endif
#ifndef OKC_WS
#define OKC_WS 0  // tiny build: one extra helper warp issues the ring's TMA copies and runs every skinny epilogue
#endif
#define OKC_WGSX 0  // dev: 1 = WGS without the wgmma, 2 = WGS phases skipped
#ifndef OKC_KPERM
#define OKC_KPERM 1  // WGS: K-permuted packing (one scale decode per 4 k16 steps); must match WGS_KPERM in Python
#endif
#ifndef OKC_WGS
#define OKC_WGS 0  // tiny build on RS-wgmma: 4 consumer warpgroups split K of 64-row units, plus a TMA producer warp
#endif
#define SILU_DIV(a, b) __fdividef((a), (b))
#ifndef OKC_UWARP
#define OKC_UWARP 1  // warp index through a shuffle, so ptxas can keep warp-derived values in uniform registers
#endif
#if OKC_UWARP
#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)
#else
#define WARP_IDX(t) ((t) >> 5)
#endif
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#ifndef OKC_FINPF
#define OKC_FINPF 0   // skinny builds: final item's topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)
#endif
#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp
#define OKC_NOLOAD 0
#ifndef OKC_SC5
#define OKC_SC5 0  // lossless 5-bit block scales in the streamed copies (skinny WS units, WGS stages); prepare() sets it
#endif

namespace okc {

#ifndef OKC_L2H
#define OKC_L2H 1
#endif
#if OKC_L2H
// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in
// the same call (evict last). Hints only: the math and the memory traffic's results are unchanged.
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}
#else
#define L2H_SUFFIX ""
#define L2H_OPERAND ""
#define L2H_ARG
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }
__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
#endif

constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;   // cross-warp partial-sum buffers
#if OKC_WS || OKC_WGS
// CTA barrier over the NT compute threads only (the helper warp syncs with them through mbarriers and barrier 2)
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)
#define OKC_SKT 8
constexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A
// dynamic tail: the last pool units of a phase's skinny work are claimed at run time (OKC_DYNK per CTA)
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int KS = 256;         // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here
  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)
  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)
  const unsigned char* wAn; const unsigned char* wBn;  // natural-k fat copies (wgmma path)
  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]
  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]
  int* xready;                           // WGS: per job, rows of xA written
  int* doneAs;                           // WGS: per (job, phase-B stage), phase-A warpgroup epilogues done (8 = ready)
  uint32_t zero;  // always 0 (opaque to the compiler)
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr;        // [8]: 0 ctrA, 1 ctrB, 2 barrier, 3 exit
  int* doneA;      // [JMAX]
  __nv_bfloat16* inter;  // [EMAX, ID]
  float* ybuf;     // [EMAX, HD]
  float* ss;       // [EMAX] copy 0 of the per-token sums of squares (copy 1: ss1); a launch uses copy gen & 1
  float* ss1;      // [EMAX]
  __nv_bfloat16* out;
  unsigned long long* dbg;
};

struct Smem {
  uint32_t lut[256];
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];  // per-unit cross-warp partial sums
#else
  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];  // WS: ring slot released by all compute warps / partials ready / buffer free
  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#endif
#if OKC_WGS
  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty
  int wglast[4];      // WGS: this warpgroup completed a split unit
  int wcp[JMAX + 1];  // WGS: phase-A cost prefix per job
  int wcpB[JMAX + 1]; // WGS: phase-B cost prefix per job
  int jrow[JMAX + 1]; // WGS: first padded row of each job in xA / xB (rows padded to 8 per job)
  uint32_t wtot;      // WGS: phase-A total cost
  uint32_t wtotB;     // WGS: phase-B total cost
#else
  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)
#endif
#if OKC_WGS
  int wgr[4];         // WGS: this CTA's phase A and phase B unit ranges
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue's job -> expert map)
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
};

// floor(a * c / G) without an integer divide (per-launch code runs from a cold instruction cache, where the divide
// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p) --q;
  else if ((q + 1) * G <= p) ++q;
  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
#define EXPB_PTR (p.ctr + 648)  // wewillcontrol tokq: per local expert, phase-B warpgroup-unit epilogues done
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).
// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,
// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:
  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
  // bits kept by 0x81808180 never wrap, so the left rotate is a plain shift: as a multiply it runs on the FMA pipe
  // (IMAD.SHL) instead of the ALU pipe the rest of the decode saturates
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}


// ---------------------------------------------------------------- routing prologue (every CTA, identical result)
__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_RLOAD && OKC_WGS  // (the small builds lose 1-2% with it: their E is at most 256)
  {  // 16 B loads (E = 8 M; the launcher checks the alignment): one or two per thread, all issued before the stores
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
  }
#else
#pragma unroll 8
  for (int i = tid; i < E; i += NT) {
    int e = __ldg(p.topk_ids + i) - p.lo;
    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
    int off = incl - tot;
    S.cnt[lane] = tot;
    S.off[lane] = off;
    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
#endif
    int fi = nfj, si = nsj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
#if OKC_SPLIT
    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }
#else
    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }
#endif
    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }
    __syncwarp();
#if !OKC_WGS  // (the WGS build has no fat queue)
    // fat queue order: largest n-tile count first (LPT), stable by job index
    int pos = 0;
    for (int key = FT_T / 8; key >= 1; --key) {
      for (int f0 = 0; f0 < nfTot; f0 += 32) {
        const int f = f0 + lane;
        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;
        pos += __popc(bal);
      }
    }
#endif
  }
  CTA_SYNC();
  early_issue(p, S);
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else if (i < i1) {
      S.pos[i] = -1;
    }
    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}

// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)
// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile
__device__ __forceinline__ float probe_q(float v) {
  return v;
}
__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    if (tk < ntok) {
      float gt = r[h] * gsc, up = r[2 + h] * gsc;
      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));
    }
  }
}
// returns per-token partial sum of squares (2 values) for this lane's two tokens
__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,
                                      float* sq) {
  const float w0 = 1.f + __bfloat162float(p.normw[n0]), w1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    float v0 = r[h] * gsc * w0, v1 = r[2 + h] * gsc * w1;
    sq[h] = v0 * v0 + v1 * v1;
    if (tk < ntok) {
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);
    }
  }
}
__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);
  }
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);
  }
}



// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through OKC_NS smem slots (TMA bulk copies on sm_90).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
// SC5 (lossless scale compression, prepare's _sc5_* packers): every e4m3 block-scale byte is stored as a 5-bit offset
// from its group's base (group = 8 rows (h) x one warp's K slice; prepare checks every group spans < 32 codes). A
// pair = (row g, row g + 8) of one lane: row g in bits s..s+4, row g + 8 in bits s+16..s+20 of a word, so
// ((word >> s) & 0x001F001F) * 16 + K2 with K2 = (base_g | base_g8 << 16) * 16 + 0x5C005C00 rebuilds exactly the bf16x2
// (scale * 2^64) the 8-bit PRMT decode gives, one IMAD per pair; the multiplies take one half each (HMUL2 .H0_H0 /
// .H1_H1). A unit / stage in global memory is SC5_BYTES (one bulk copy) at a SC5_STRIDE pitch (128 B aligned); the
// smem slots keep UNIT_BYTES.
constexpr int SC5_BYTES = 35552, SC5_STRIDE = 35584;
constexpr int UNIT_GB = OKC_SC5 ? SC5_BYTES : UNIT_BYTES;    // bytes per weight bulk copy (skinny unit / WGS stage)
constexpr int UNIT_GS = OKC_SC5 ? SC5_STRIDE : UNIT_BYTES;   // their pitch in global memory
// skinny unit (SC5): [32 KB codes][W: warp (8) x lane (32) x 2 words (pairs c = 0..2 | 3..5 at s = 0, 5, 10)]
// [X: warp (8) x (11 words chunk 6 | 11 words chunk 7), lane l in word l / 3 at s = 5 (l % 3)][bases: byte 2 warp + h][16 B 0]
constexpr int SC5_SW = UNIT_CODES, SC5_SX = SC5_SW + 2048, SC5_SH = SC5_SX + 704;
static_assert(SC5_SH + 32 == SC5_BYTES && SC5_STRIDE % 128 == 0 && SC5_BYTES % 16 == 0 && SC5_BYTES <= UNIT_BYTES, "SC5 unit");
static_assert(!OKC_SC5 || OKC_PRMTSC, "SC5 rebuilds the PRMT decode's scale values (positive normal e4m3 bytes only)");
static_assert(!OKC_SC5 || OKC_WGS || (OKC_WS && OKC_TINY && NW == 8), "SC5 skinny decode: the WS tiny builds (8 warps) only");


__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
// e4m3 scale byte b of `word` -> bf16x2(scale * 2^64) in both halves with one PRMT and one IMAD: (byte << 4) puts the
// e4m3 exponent and mantissa into the bf16 fields, + 184 << 7 rebiases the exponent. Exact for positive normal bytes
// 0x08..0x7E only; prepare() checks every scale byte and uses the LUT build otherwise.
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
// same, with the multiplier opaque to ptxas (k16 = 16 + p.zero): an IMAD on the FMA pipe instead of a LEA on the
// ALU pipe, which the FP4 decode saturates
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_GS;  // [32 KB codes | scales (8-bit 4 KB / SC5)]
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_GB) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_GB), "r"(smem_u32(bar)) L2H_ARG : "memory");
}

template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  if (threadIdx.x == 0) ring_issue<PB>(p, S, v, slot, bar);
}

template <bool PB>
__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,
                                            uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  {
    const int r = g;
    if (r < cnt) {
      const __nv_bfloat16* row;
      if (PB) row = p.inter + (size_t)(st + r) * ID;
      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);
        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
    }
  }
}

constexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4
  const int cnt = S.jC[j], st = S.jS[j];
  for (int i = lane; i < 8 * 64; i += 32) {
    const int r = i >> 6, c = i & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (8 + r < cnt) {
      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;
      v = PB ? __ldcg(src) : __ldg(src);
    }
    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;
  }
  __syncwarp();
}

// skinny_ring exists twice, deliberately: the SKDYN variant claims tail units
// from a run-time atomic pool, the #else variant issues a fixed static
// range. The bodies share most of their epilogue code, but the pool-claim
// logic interleaves with the ring refill in ways a merge would risk
// changing; keep them in sync by hand.
#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  // thread 0 owns the unit source: static units first, then pool units claimed with one atomic in flight
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);  // the next claim; its value is first needed at the next refill
    if (!PB) {
      const int jj = njF + (v >> rsh);
      S.dync[jj] += 1;
      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);
    }
    return v;
  };
  auto refill = [&](int sl) {
    const int v = take();
    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }
    else if (cpf && preB < nB) {
      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);
      S.uslot[sl] = BFLAG | (v0B + preB);
      ++preB;
    } else {
      S.uslot[sl] = -1;
    }
  };
  if (tid == 0)
    for (int i = pre; i < OKC_NS; ++i) {
      const int sl = (seq0 + i) % OKC_NS;
      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
  if (nU < OKC_NS) CTA_SYNC();  // pool slots filled by the prologue: their unit ids must be visible
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) {
    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);
    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);
    nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
  }
  int nproc = 0;
  for (int i = 0;; ++i) {
    int v;
    if (i < nU) {
      v = v0 + i;  // static units: no slot lookup
    } else {
      v = S.uslot[(seq0 + i) % OKC_NS];
      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B's prefetched units follow)
      v &= ~BFLAG;
    }
    const int j = njF + (v >> rsh), rb = v & (rbs - 1);
    float nwn0 = 0.f, nwn1 = 0.f;
    int vn = -1;
    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];
    if (PB && tid < 128 && vn >= 0) {
      const int rbq = (vn & ~BFLAG) & (rbs - 1);
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    // a warp covers half of an 8-warp K slice: its 4 chunks' scale bytes are 8 of the slice lane's 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (tid == 0) {
      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
    ++nproc;
  }
  seq = seq0 + nproc;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      if (nU > 0) {
        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);
        for (int jj = jf; jj <= jl; ++jj) {
          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
          atomicAdd(p.doneA + jj, b - a);
        }
      }
      for (int jj = jdmin; jj <= jdmax; ++jj) {  // pool units
        const int k = S.dync[jj];
        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else  // static skinny_ring: fixed range, no run-time pool
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  // cross-phase prefill: with no fat jobs, phase A's last ring refills already fetch this CTA's first B units
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  for (int i = pre; i < OKC_NS; ++i) {
    const int sl = (seq0 + i) % OKC_NS;
    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }
  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit's rows one unit ahead
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    float nwn0 = 0.f, nwn1 = 0.f;
    if (PB && tid < 128 && i + 1 < nU) {
      const int rbq = (v + 1) % rbs;
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();  // slot sl fully consumed; partials visible
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i + OKC_NS - nU < nB) {
      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);
      ++preB;
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
  }
  seq = seq0 + nU;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;
      for (int jj = jf; jj <= jl; ++jj) {
        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
        atomicAdd(p.doneA + jj, b - a);
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#endif

// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens
// ---------------------------------------------------------------- fat unit through the smem ring
// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x
// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile
// for the stage is staged by cp.async into two buffers that live in slots 3..4.
constexpr int NSF = OKC_NSF;
template <bool PB>
__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {
  const int nub = PB ? UB_B : UB_A;
  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
}

// ---------------------------------------------------------------- fat units: one continuous stream per CTA
// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps
// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit's weights).
// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int NTT> struct WG;
template <> struct WG<1> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<2> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<3> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<4> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<5> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<6> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<7> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};

// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from
// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)
template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
  // x stage: [kb 4][64 rows][128 B], 128 B swizzle (chunk c of row n at c ^ (n & 7)); k-step st of chunk cc reads
  // k block cc at byte offset 32 st
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];  // [chunk parity][k-step x rb][fragment]; a set stays untouched until its group retired
  uint32_t keep = 0;    // folds every fragment after its group retired, so ptxas keeps the registers live (no reuse)
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      // e4m3 -> bf16 without the LUT: bf16 bits (b << 4) hold value * 2^-120 (subnormal codes included), two exact
      // multiplies by 2^92 give value * 2^64 like the LUT; rows g | g+8 travel as one bf16x2
      const uint32_t K92 = 0x6D806D80u;  // bf16x2(2^92, 2^92)
      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);
      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;
      dq4((&w0.x)[st], r0, r1, r2, r3);
      dq4((&w1.x)[st], q0, q1, q2, q3);
      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);
      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);
      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);
      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);
#undef FMG0
#undef FMG80
#undef FMG1
#undef FMG81
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);
      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);
      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);
    }
    wg_commit();
    if (cc >= 1) {
      wg_wait<1>();  // the previous chunk's group retired: its register set may be rewritten
#pragma unroll
      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];
  acc0[0][0] += __uint_as_float(keep & zero);  // adds +0.0f
}

// one 36 KB k-stage of a fat unit: 2 row blocks per warp x NTT token tiles (exact tile count, no predicated MMAs)
template <int NTT>
__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);
#if OKC_NW == 8
  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);
#endif
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
#if OKC_NW == 8
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
#endif
#if OKC_PRMTSC
    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);
#if OKC_NW == 8
    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);
#endif
#else
    const int sh = (cc & 1) * 16;
    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;
    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];
#endif
    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;
#pragma unroll
    for (int sp = 0; sp < 2; ++sp) {
      uint32_t a0[2][4], a1[2][4];
      (void)a1;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s2 = 2 * sp + h;
        uint32_t r0, r1, r2, r3;
        dq4((&w0.x)[s2], r0, r1, r2, r3);
        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);
#if OKC_NW == 8
        dq4((&w1.x)[s2], r0, r1, r2, r3);
        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);
#endif
      }
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);
        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);
#endif
        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);
#endif
      }
    }
  }
}

template <bool PB, bool BIG>
__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;
  int* ctr = p.ctr + (PB ? 1 : 0);
  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
  uint64_t* fb = S.bars + 5;
  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);
  constexpr int NKS = HD / KS;  // 16
  int u = blockIdx.x;
  (void)ctr;
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;  // == ubs, as a constant (shift instead of a divide)
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {  // phase B: inter rows of job j complete
    if (PB) {
      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }
      CTA_SYNC();
    }
  };
  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };
  auto stage_x = [&](int j, int ks, int xsl) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int ntl = (cnt + 7) >> 3;
    const int nrow = ntl * 8;
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
    const int per = nrow * (KS / 8);
    if (is_nat(j)) {
      // 128 B swizzled K-major layout [kb 4][64 rows][128 B]: a warp copies one 512 B row (conflict-free)
      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);
      for (int i = tid; i < per; i += NT) {
        const int r = i >> 5, ch = i & 31;
        const bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;
        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);
      }
    } else {
      for (int i = tid; i < per; i += NT) {
        int r = i / (KS / 8), c = i - r * (KS / 8);
        bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;
        cp_async16(xs + r * XP_FT + c * 8, src, v);
      }
    }
    cp_commit();
  };
  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)
  const __nv_bfloat16* xsrc[8];
  uint32_t xdst[8];
  uint32_t xval = 0;
  int xn = 0;
  auto prep_x = [&](int j) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);
    xval = 0;
    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = tid + k * NT;
      const int r = i >> 5, c = i & 31;
      const bool v = i < per && r < cnt;
      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
      xdst[k] = r * XP_FT + c * 8;
      if (v) xval |= 1u << k;
    }
  };
  auto stage_xp = [&](int ks, int xsl) {
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);
    cp_commit();
  };
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int f = S.fseq;  // ring position of the current unit's k-stage 0
  int xq = 0;      // x stage parity counter
  int j, ub;
  job_of(u, j, ub);
#pragma unroll
  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));
  wait_A(j);
  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);
  while (true) {
    const int un = u + gridDim.x;
    int jn = 0, ubn = 0;
    if (un < nF) job_of(un, jn, ubn);
    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];
    const int ntile = (cnt + 7) >> 3;
    const int ntc = ntile;  // exact token-tile count (rows past cnt are zero-filled)
    const bool nat = is_nat(j);
    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;
    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];
#pragma unroll
    for (int i = 0; i < FT_T / 8; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }
    for (int ks = 0; ks < NKS; ++ks) {
      if (OKC_XPRE) {
        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }
        else cp_commit();
      } else {
        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }
        else cp_commit();
      }
      cp_wait<1>();
      const int q = f + ks, sl = q % NSF;
      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));
      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // cp.async x writes -> wgmma reads
      CTA_SYNC();  // x stage visible to every warp
      const unsigned char* slot = ring + sl * UNIT_BYTES;
      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);
      if (nat) {
        switch (ntc) {
          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
        }
      } else {
        switch (ntc) {
          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;
          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;
        }
      }
      CTA_SYNC();  // slot sl and the x stage consumed
      ++xq;
      // refill: the stage NSF ahead, in this unit or the next one
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
    // epilogue of unit u (registers -> global)
    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;
    if (!PB) {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);
#if OKC_NW == 8
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);
#endif
        }
      __threadfence();
      CTA_SYNC();
      if (tid == 0) atomicAdd(p.doneA + j, 1);
    } else {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          float sq0[2], sq1[2];
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);
#if OKC_NW == 8
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);
          sq0[0] += sq1[0]; sq0[1] += sq1[1];
#else
          (void)sq1;
#endif
          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);
        }
    }
    f += NKS;
    if (un >= nF) break;
    u = un; j = jn; ub = ubn;
  }
  cp_wait<0>();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if (tid == 0) S.fseq = f;
  CTA_SYNC();
}

template <bool PB>
__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  fat_stream<PB, false>(p, S, xbuf, nF, ubs);
}



// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)
__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;  // the first units were issued at kernel start (FAST1), or the WGS producer warp owns the ring
#endif
  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region
    if (threadIdx.x == 0) S.preissued = 0;
    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);  // static part (the pool tail is claimed at run time)
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in
  // skinny_ring issues the rest after routing, so every CTA's first unit is served ahead of the second ones
  const int n = min(OKC_EARLYN, v1 - v0);
  if (threadIdx.x == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
}

// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost
#if OKC_WS
// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)
// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper
// warp (threads NT..NT+31) issues every TMA copy as soon as all compute warps released a slot and runs every unit's
// epilogue, so no CTA-wide barrier sits in the per-unit loop.
// the last WSBLK units keep their partial buffers until the compute warps' own epilogues: WSBLK > WSNR deadlocks the
// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;  // opaque 16 (scale decode IMAD)
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)
#if OKC_SC5
  const uint32_t sxo = (uint32_t)(lane / 3) * 4u, sxs = 5u * (uint32_t)(lane % 3);  // SC5: this lane's X word / slot
#endif
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
    if (j != curj) {
      curj = j;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_SC5
    // SC5: this lane's 8 scale pairs (chunks 0..5 in its 2 words, 6 and 7 in the warp's shared X words) + the bases
    const uint2 scw = *reinterpret_cast<const uint2*>(slot + SC5_SW + warp * 256 + lane * 8);
    const uint32_t scxa = *reinterpret_cast<const uint32_t*>(slot + SC5_SX + warp * 88 + sxo);
    const uint32_t scxb = *reinterpret_cast<const uint32_t*>(slot + SC5_SX + warp * 88 + 44 + sxo);
    uint32_t sck = __byte_perm((uint32_t)*reinterpret_cast<const unsigned short*>(slot + SC5_SH + warp * 2), 0u, 0x4140u) * k16 + 0x5C005C00u;
    asm("" : "+r"(sck));  // opaque: else NVVM folds it into (offsets + bases) * 16 + bias, one IADD more per pair
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_SC5
      // pair c: rows g / g + 8 in the low / high half
      const uint32_t sgp = (((c < 3 ? scw.x : c < 6 ? scw.y : c == 6 ? scxa : scxb) >> (c < 6 ? 5u * (uint32_t)(c % 3) : sxs)) & 0x001F001Fu) * k16 + sck;
#elif OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
#if OKC_SC5
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2l(r0, sgp), bmul2h(r1, sgp), bmul2l(r2, sgp), bmul2h(r3, sgp), b[2 * st], b[2 * st + 1]);
#else
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
#endif
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    // WSB: the compute warps keep the base loop's CTA barrier per unit (slot consumed + partials visible), thread 0
    // refills the slot and hands the partial buffer to the helper warp, which runs every epilogue off this loop
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1
    }
  }
  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA's last units: epilogues here, not behind the helper's queue
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    const int n0 = rb * 16 + gg;
    const float gsc = lgs * TWO62;
    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;
      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
      float sq = v0f * v0f + v1f * v1f;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
  seq = seq0 + nU;
}

__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q's slot once every compute warp released it
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif

template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B's skinny units are slower, so its ratio is lower)
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);  // total cost = G x level
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
    // split point of CTA x: floor(nS * W(x) / WG) in float (one formula for both neighbours, so the shares tile [0, nS))
    const float sc = __fdividef((float)nS, (float)WG);
    auto split = [&](int x) -> int {
      if (x >= G) return nS;
      const int v = __float2int_rz((float)W(x) * sc);
      return v < nS ? v : nS;
    };
    v0 = split(c); v1 = split(c + 1);
  } else {
    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);
  }
}

#if OKC_WGS
// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)
// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,
// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales
// [grp][warp][g][h][j]; the stage's x slice (the job's token rows, rounded up to 8, x 256 K as wgmma B core matrices
// [n8][32 k8][8 rows][16 B]) sits in its own ring. Warpgroup q accumulates its 64 rows over K with wgmma m64nNk16
// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).
// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a
// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.
#define OKC_WGSNX 3                             // x ring slots
constexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job
constexpr int WGS_PIECE = OKC_SC5 ? 8880 : 9216;  // one warpgroup's weights in a stage
// WGS stage (SC5): 4 pieces of [8 KB codes][W: wp (4) x lane (32) words: pairs lg = 0..2 at s = 0, 5, 10][X: 44 words:
// pair lg = 3 of warpgroup thread wt in word wt / 3 at s = 5 (wt % 3)], then 32 B of bases: byte 2 (wg * 4 + wp) + h
constexpr int SC5_PW = 8192, SC5_PX = 8704, SC5_PH = 4 * 8880;
static_assert(!OKC_SC5 || (SC5_PX + 176 == WGS_PIECE && SC5_PH + 32 == SC5_BYTES && OKC_KPERM), "SC5 WGS stage");
constexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;  // barrier indices
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;                  // fixup slots per phase (>= grid + 1)
constexpr int WGS_FIXW = WGS_NT * 512;          // floats per (slot, warpgroup): [n-tile][128 threads][4]
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
// per-stage cost by n-tiles (1..8), phase A and phase B (relative units, <= 255; fit on per-CTA phase times at
// bs 96..256, t_fit2.py): tuner knobs OKC_WGSA / OKC_WGSB = 8 comma-separated values each
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
constexpr unsigned long long wgs_pack8(const int (&c)[8]) {
  unsigned long long r = 0;
  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);
  return r;
}
constexpr int WGS_CA[8] = {OKC_WGSA};
constexpr int WGS_CB[8] = {OKC_WGSB};
constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);
static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");
__device__ __forceinline__ int wgs_cost(int pb, int cnt) {
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
}
#define OKC_WGSREORD 1  // WGS: phase A starts with the trailing partial unit (see wgs_split)
#define OKC_GATHB 8  // WGS: x rows gathered per batch (loads first, then stores); 0 = one row at a time
#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather
#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B
#define OKC_WGSFCI 3
#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
  (void)tag;
  mbar_wait(bar, parity);
}
template <int NTT> struct WGSMMA;
template <> struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
// two producer warps: the weights warp streams every stage's 36 KB through the weight ring (TMA, lane 0), the x warp
// gathers every stage's x slice into the x ring (cp.async) and, in phase B, first waits for the job's phase-A units;
// so the weights run ahead by the whole ring even while x waits
// Phase A runs a CTA's trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit's
// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase
// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the
// order (its first stages wait for phase A's units in order).
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range's last stage
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;  // log2 of 256-row units per job
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_GS;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_GB) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_GB), "r"(smem_u32(bar)) L2H_ARG : "memory");
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
// x producer: one bulk copy per stage (lane 0): the job's x slice for the stage is contiguous in xA (phase A, written
// by every CTA's gather pass at the start) or xB (phase B, written by the phase-A epilogues), N_pad x 512 B
constexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the
  // stage ranges while the consumers gather phase A's x
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;  // [sp, g1) done: then [g0, sp)
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered
      }
      // phase B stage st reads inter columns 256 st .. +255: the 2 phase-A units (x 4 warpgroups) that wrote them
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
// copied into xA's per-stage wgmma layout, then counted into xready (padding rows are counted, not written)
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = threadIdx.x, nj = S.njF + S.njS;
  const int R = S.jrow[nj];  // R * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
  // all of a batch's row loads first, then the stores: one x row at a time was one dependent HBM round trip per row
  // (the next row's load could not move above the previous row's store: they may alias), 1.3-7 us at bs256
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  constexpr int GB = OKC_GATHB;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
    uint4* d[GB];
#pragma unroll
    for (int k = 0; k < GB; ++k) {
      const int q = qb + k;
      d[k] = nullptr;
      if (q < q1) {
        while (S.jrow[j + 1] <= q) ++j;
        const int r = q - S.jrow[j], cnt = S.jC[j];
        if (r < cnt) {
          const int ntt = (cnt + 7) >> 3;
          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));
#if OKC_KPERM
          {  // output chunk P = tid & 7 of its 64-group: word t = k 16t + 4(P >> 1) + 2(P & 1) + {0, 1}
            const uint32_t* xr = reinterpret_cast<const uint32_t*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + (tid & ~7) * 8
                                                                   + 4 * ((tid & 7) >> 1) + 2 * (tid & 1));
            v[k] = make_uint4(__ldg(xr), __ldg(xr + 8), __ldg(xr + 16), __ldg(xr + 24));
          }
#else
          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);
#endif
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  CTA_SYNC();
  if (tid == 0) {  // count this CTA's rows per job (release: the CTA's stores are ordered before by the barrier)
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
// Work left from the previous segment, done after the next segment's first wgmma group (so the stores and atomics it
// waits for have long completed): kind 1 = release doneA for job j (phase A), kind 2 = add the stage count of a split
// unit's share and, if that completes the unit, read the summed partials back and run its epilogue.
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    // SwiGLU straight into xB (phase B's per-stage wgmma B layout) instead of row-major inter
#if OKC_KPERM
    const int col0 = r16 * 8 + g;
    const int col = (col0 & ~63) | (((col0 >> 2) & 3) << 4) | (((col0 >> 1) & 1) << 3) | (((col0 >> 4) & 3) << 1) | (col0 & 1);
#else
    const int col = r16 * 8 + g;
#endif
    const int ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
  } else {
    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
      }
    }
  }
}
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
#if OKC_TOKQ && OKC_WGSFINSS
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(PB ? EXPB_PTR + S.jE[j] : p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
#else
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
#endif
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
  // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {
  if (wt == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
    S.wglast[wg] = old + pd.nst == 16;
  }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  const bool last = S.wglast[wg];
  pd.kind = 0;
  if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  // two n-tiles' loads in flight before their zeroing stores and epilogues (the scout): one at a time, each load
  // waits behind the previous tile's stores (they may alias), ntt serialized L2 round trips under full HBM load
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
#if OKC_TOKQ && OKC_WGSFINSS
  else {  // tokq: this warpgroup's 64 rows of the unit are in ybuf
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(EXPB_PTR + S.jE[j]), "r"(1) : "memory");
  }
#endif
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode's multiply stays an IMAD (FMA pipe)
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];  // written first by a scale-d = 0 wgmma (no zeroing: ptxas would serialize the wgmmas, C7513)
  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas
  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
  int prev = -1;  // x slot the last committed group may still read
  // slot-0 descriptor and code pointer, once per segment: a stage adds its slot's offset (the address field cannot
  // carry: shared addresses >> 4 stay below 2^14) instead of rebuilding the descriptor (~8 uniform ops per stage)
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
#if OKC_SC5
  // SC5: this thread's X word / slot, and its warp's base pair relative to its piece (the bases follow the 4 pieces)
  const uint32_t sxo = (uint32_t)(wt / 3) * 4u, sxs = 5u * (uint32_t)(wt % 3);
  const int sho = SC5_PH - wg * WGS_PIECE + (wg * 4 + wp) * 2;
#endif
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    // codes and scales one 4-step group at a time (fewer live registers)
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
#if OKC_SC5
    // SC5: pairs lg = 0..2 from this lane's word, lg = 3 from the shared X word; the warp's bases (rows g / g + 8)
    const uint32_t scw = *reinterpret_cast<const uint32_t*>(pc + SC5_PW + (wp * 32 + lane) * 4);
    const uint32_t scx = *reinterpret_cast<const uint32_t*>(pc + SC5_PX + sxo);
    uint32_t sck = __byte_perm((uint32_t)*reinterpret_cast<const unsigned short*>(pc + sho), 0u, 0x4140u) * k16c + 0x5C005C00u;
    asm("" : "+r"(sck));  // opaque: one IMAD per pair (else (offsets + bases) * 16 + bias)
#elif OKC_KPERM
    const int t4 = lane & 3;
    uint32_t scn = *reinterpret_cast<const unsigned short*>(pc + 8192 + (wp * 8 + g) * 8 + t4 * 2);
#else
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#endif
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
#if OKC_SC5
      const uint32_t sgk = (((lg < 3 ? scw : scx) >> (lg < 3 ? 5u * (uint32_t)lg : sxs)) & 0x001F001Fu) * k16c + sck;
#elif OKC_KPERM
      const uint32_t sgk = __byte_perm(scn, 0u, 0x4140u) * k16c + 0x5C005C00u;  // (row g, row g+8) scales of block t
#else
      const uint2 scg = scn;
#endif
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
#if OKC_SC5
#elif OKC_KPERM
        scn = *reinterpret_cast<const unsigned short*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8 + t4 * 2);
#else
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
#endif
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;  // commit group index in the stage
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;      // k16 step within the 4-step load group
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          // scales stored [j][h] (rows g, g+8 of step j side by side): one PRMT + one IMAD build both rows' scales in
          // one bf16x2, and the multiplies broadcast its halves (HMUL2 .H0_H0 / .H1_H1): 2 instructions per k16 fewer
#if OKC_KPERM
          const uint32_t sgp = sgk;
#else
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
#endif
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
        }
        wg_commit();
        wg_wait<1>();
        if (ci == 0) {
          if (prev >= 0) {  // the previous stage's last group just retired: its x slot is free
            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
            prev = -1;
          }
        }
        // the previous segment's deferred work, behind group OKC_WGSFCI of the first stage (scout: CI 3 lets this WG's
        // own red.adds / epilogue stores drain a little longer before the count / release atomics: +0.5-1%)
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
    prev = xs;
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
#if OKC_TOKQ && OKC_WGSFINSS
  pd = WgsPend{1, u, -1, 0};  // phase A: doneAs release; phase B (tokq): expB release; deferred behind the stores
#else
  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred
#endif
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {  // [sp, g1) done: then [g0, sp); the bounds come back from smem (no extra live state)
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    // a unit split between CTAs: its fixup slot is the CTA holding its first stage
    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
    switch (ntt) {
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
    }
    gs += s1 - s0;
  }
  wgs_flush<PB>(p, S, pd);
}
// cost-weighted stage ranges: per-stage cost by the job's n-tiles, a prefix per job and phase (S.wcp, S.wcpB)
// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run
// from a cold instruction cache at the start of every launch
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;
    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
// stage at cost position x (phase pb), the stage whose cost interval holds x
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;  // job: last j with prefix(j) <= x
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);
  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);
}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
}
// this CTA's stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);
  // largest c with floor(tot * c / G) < xe
  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);
  if (c > G - 1) c = G - 1;
  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;
  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;
  return c;
}
#endif

template <bool PB, bool BIG>
__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
  const int njF = S.njF, njS = S.njS;
  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;
  const int nS = njS * rbs, nF = njF * ubs;
  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost
#if OKC_TINY
  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A's last ring
  // refills already fetch those units (same split function as phase B uses)
  int cB0 = 0, cBn = 0;
  bool cpf = false;
  if (!PB) {
    const int nFB = njF * UB_B;
    if (c >= nFB) {
      int b1;
      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);
      cBn = b1 - cB0;
      cpf = OKC_CPFF || njF == 0;
    }
  }
#if OKC_WS
  (void)cpf; (void)cBn;
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
#else
  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,
                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);
#endif
}

// a volatile shared load at every use: hoisted, the pointer would stay live across the accumulate loops (96-register
// cap: spills)
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist
#endif
}
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
  const int tid = threadIdx.x;
  DBG(0);
  // programmatic dependent launch: everything below reads the previous kernel's outputs (routing, x) or writes
  // scratch, so wait for it here; without the launch attribute this returns at once
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif
#if FAST1_ON
  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA's first units follow
  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
    // per-expert entry counts -> jobs per expert (ceil(c / 8)) -> inclusive prefix over experts (lane = expert)
    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));
    }
    __syncwarp();
    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }
    const int njob = __shfl_sync(0xffffffffu, jincl, 31);
    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);
    const int n1 = min(OKC_FAST1, v11 - v01);
    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];
#pragma unroll
    for (int i = 0; i < OKC_FAST1; ++i) {
      const int jj = (v01 + i) >> RB_A_LOG;   // (n1 is warp-uniform: every lane runs the same ballots)
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
    // the topk loads go out first (E <= 64 in the tiny builds: two per lane at most) and lane 0 initializes the
    // barriers while they are in flight (the inits used to wait behind the cold loads)
    uint32_t m = 0;
    const int E8 = p.M * 8;
    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;
    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    for (int i = ln + 64; i < E8; i += 32) {  // (never in the tiny builds: M <= 8)
      const int e = __ldg(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
#endif
    if (ln == 0) {
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
#if OKC_SPLIT
      const int v0 = v01, n = n1;
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);  // static part (pool tail claimed at run time)
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#if OKC_SPLIT
#pragma unroll
      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = v0 + i;
        const int e = fex[i];
#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_GS;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_GB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_GB), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {  // fp8 e4m3 code -> bf16x2(value * 2^64), computed (no global load)
    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) {
      b = 0xFFFFu;
    } else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sgn ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
#if !FAST1_ON
#if OKC_WGS
    // full slots (TMA tx), empty slots (one arrival per consumer warp), partials ready (every consumer thread),
    // partial buffer free (the epilogue warpgroup)
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
#endif
  route(p, S);
#if OKC_WGS && OKC_WGSPR
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges
#endif
  DBG(1);
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#if OKC_WS
  if (tid == 0) {  // this CTA's unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)
    const int G = gridDim.x, c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {  // padded-row base of every job in xA / xB (warp scan)
    const int nj = S.njF + S.njS;
    int carry = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
      const int j = j0 + tid;
      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
      int incl = v;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }
      if (j < nj) S.jrow[j] = carry + incl - v;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (tid == 0) S.jrow[nj] = carry;
  }
  CTA_SYNC();
  // phase A's x into xA (this CTA's share of the rows), before the producers start: its x reads do not queue behind
  // every CTA's first weight stages
  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int seq = 0;
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    DBG(2);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
    DBG(3);
  }
#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
#if OKC_WS
  WS_JOIN();  // the helper's last epilogue stores are visible
#endif
  // programmatic dependent launch: the next kernel may start its own prologue now (its griddepcontrol.wait still
  // waits for this whole grid to finish and its writes to be visible)
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  // grid barrier on this launch's counter (two, by generation: the other one was zeroed by this launch for the next);
  // the last arriver bumps the generation, which no CTA of this launch reads again
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
  // the CTA's first final item: its routing (pos) and route weights are inputs, so they load before the barrier
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
  }
#endif
#if !(OKC_WGS && OKC_TOKQ && OKC_WGSFINSS)
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
  // shuffles, then the 16 warps' partials through smem: the ring is free now), pass 2 re-reads the rows (L2 hits)
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_TOKQ
      if (tid < 8) {  // tokq: wait until each local expert of this token finished phase B (64 per job)
        const int e = S.eid[tk * 8 + tid];
        if (e >= 0) {
          const int c = S.cnt[e];
          const int need = 64 * (c > SK_T ? (c + FT_T - 1) / FT_T : 1);
          while (ld_acquire(EXPB_PTR + e) < need) __nanosleep(32);
        }
      }
      CTA_SYNC();
#endif
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {  // uniform over the CTA (one token)
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();  // red is reused by the next token
    }
  }
#if OKC_TOKQ
  // arrival (the old barrier counter, no wait): the last CTA zeroes the per-launch flags and bumps the generation
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    S.wglast[0] = old == (int)gridDim.x - 1;
  }
  CTA_SYNC();
  DBG(4);
  if (S.wglast[0]) {
    __threadfence();
    if (tid < 2) p.ctr[tid] = 0;
    if (tid < 32) EXPB_PTR[tid] = 0;
    for (int i = tid; i < JMAX; i += NT) { p.doneA[i] = 0; p.xready[i] = 0; }
    for (int i = tid; i < JMAX * 16; i += NT) p.doneAs[i] = 0;
    CTA_SYNC();
    if (tid == 0) { __threadfence(); atomicAdd(p.ctr + 3, 1); }
  }
#endif
#elif OKC_FINCOL
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
    float4 yv[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;
      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (qvp[s] >= 0) {
        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  }
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float4 v = __ldcg(src + i);
          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;
        }
      }
    }
    uint4 o[2];
    uint32_t* ow = reinterpret_cast<uint32_t*>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      ow[i] = *reinterpret_cast<uint32_t*>(&v);
    }
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);
    dst[0] = o[0]; dst[1] = o[1];
  }
#endif
  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered
}

}  // namespace okc

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

static size_t okc_smem_bytes() {
  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
#if OKC_WGS
  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;
#endif
  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;
  return base + (xa > xb ? xa : xb);
}

static int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)

void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }

void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }  // no natural copies: the wgmma path must stay off (OKC_WGMIN 9)
#if OKC_WGS
  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");
  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,
              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");
  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();
#else
  p.wAs = nullptr; p.wBs = nullptr;
#endif
  p.zero = 0;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
#if OKC_WGS && OKC_RLOAD
  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");
#endif
  int* ib = (int*)scratch[0].data_ptr();
  p.ctr = ib; p.doneA = ib + 8;
  p.ss = (float*)scratch[1].data_ptr();
  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
  p.ybuf = p.ss + okc::EMAX;
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#else
  p.xA = p.xB = nullptr;
  p.xready = nullptr;
  p.doneAs = nullptr;
#endif
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  static bool attr = false;
  size_t sm = okc_smem_bytes();
  if (!attr) {
    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0; (void)e1;
    attr = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which
  // leaves the skinny loop's code and registers as tuned
  if (g_okc_pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3(okc::NTL);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
    return;
  }
  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
}


'''

# ---------------------------------------------------------------- WGS main build source ("@wgh")
# The decode layer's in-layer MoE WGS source (BABAGAZ/muse_decode_layer/v394_et.py: MOE_CUDA_SRC + its v150, A300
# and WFIT patch sets), rebuilt from this file's CUDA_SRC at import: _WGH_BASE turns CUDA_SRC into the layer's
# MOE_CUDA_SRC (it drops this file's own tokq final, which v150 replaces, and some comments / dead fields), then the
# layer's patch sets apply verbatim. Every patch must match exactly once. Generator: work/muse_moe_w4a16/gen_v020.py.
# CUDA_SRC -> the layer's MOE_CUDA_SRC
_WGH_BASE = (
    ('base0',
r'''
// ---------------------------------------------------------------- roadmap
// Compile-time knobs (the OKC_* values the host actually passes, plus
// defaults for the rest) -> Params / Smem -> small helpers (dequant,
// mma, mbarrier, L2 hints) -> route() -> skinny ring (two variants:
// dynamic-pool SKDYN / static, see below) -> fat stream (cp.async +
// mma.sync, wgmma variant) -> WS helper warp -> WGS (RS-wgmma,
// producer/consumer) -> run_phase -> moe_fused -> C++ launcher okc_run.
// v2: dead #if branches pruned for the only builds the host compiles
// (MAIN / TINY / TINY2 / MID, each with and without -DOKC_PRMTSC=1, plus
// the MAIN fallback; BW with no -D), sm_90a only; dead functions,
// defines and BW switch cases removed. Taken-path code is unchanged.
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#ifndef OKC_NACC
#define OKC_NACC 2
#endif
#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores
                     // OKC_FTSC2; the TINY builds measured -0.5%, so off by default)
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#define OKC_TOKQ 1  // wewillcontrol w1xx: WGS final waits per token on per-expert phase-B counters (no grid barrier)
#define OKC_WGSFINSS 1  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
''',
r'''
// muse_moe_w4a16 v2: one persistent kernel: routing prologue, GEMM1+SwiGLU, GEMM2+norm stats, final norm/sum.
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1  // routing: all topk_ids loads issued before the smem stores
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#define OKC_WGSFINSS 1  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
'''),
    ('base1',
r'''static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int KS = 256;         // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
''',
r'''static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)
constexpr int KS = 256;         // fat: K per x stage (4 chunks)
constexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)
constexpr int NST = 3;          // fat x stages
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
'''),
    ('base2',
r'''  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
''',
r'''  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)
  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];   // sorted entries (token*8+slot) by expert
  short pos[EMAX];   // entry -> pair position, -1 if not local
'''),
    ('base3',
r'''#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
#define EXPB_PTR (p.ctr + 648)  // wewillcontrol tokq: per local expert, phase-B warpgroup-unit epilogues done
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
''',
r'''#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();  // this launch's ss copy (defined after Smem users)
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
'''),
    ('base4',
r'''  CTA_SYNC();
}

// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)
// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile
__device__ __forceinline__ float probe_q(float v) {
''',
r'''  CTA_SYNC();
}



// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)
// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile
__device__ __forceinline__ float probe_q(float v) {
'''),
    ('base5',
r'''  }
}



// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through OKC_NS smem slots (TMA bulk copies on sm_90).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
// SC5 (lossless scale compression, prepare's _sc5_* packers): every e4m3 block-scale byte is stored as a 5-bit offset
''',
r'''  }
}

// ---------------------------------------------------------------- the persistent kernel


// ---------------------------------------------------------------- skinny path: shared-memory weight ring
// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.
// Units of the CTA's range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).
// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
// SC5 (lossless scale compression, prepare's _sc5_* packers): every e4m3 block-scale byte is stored as a 5-bit offset
'''),
    ('base6',
r'''  __syncwarp();
}

// skinny_ring exists twice, deliberately: the SKDYN variant claims tail units
// from a run-time atomic pool, the #else variant issues a fixed static
// range. The bodies share most of their epilogue code, but the pool-claim
// logic interleaves with the ring refill in ways a merge would risk
// changing; keep them in sync by hand.
#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
''',
r'''  __syncwarp();
}

#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
'''),
    ('base7',
r'''  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else  // static skinny_ring: fixed range, no run-time pool
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
''',
r'''  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
'''),
    ('base8',
r'''  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
#if OKC_TOKQ && OKC_WGSFINSS
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(PB ? EXPB_PTR + S.jE[j] : p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
#else
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
#endif
    pd.kind = 0;
    return;
  }
''',
r'''  if (pd.kind == 1) {
    // release (cumulative over the warpgroup's epilogue stores ordered by the barrier); pairs with the producer's
    // acquire. Unit ub wrote inter columns ub*128 .. +127: half of phase B stage ub / 2
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
'''),
    ('base9',
r'''    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
#if OKC_TOKQ && OKC_WGSFINSS
  else {  // tokq: this warpgroup's 64 rows of the unit are in ybuf
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(EXPB_PTR + S.jE[j]), "r"(1) : "memory");
  }
#endif
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
''',
r'''    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)
template <bool PB, int NTT>
'''),
    ('base10',
r'''  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
#if OKC_TOKQ && OKC_WGSFINSS
  pd = WgsPend{1, u, -1, 0};  // phase A: doneAs release; phase B (tokq): expB release; deferred behind the stores
#else
  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred
#endif
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
''',
r'''  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
'''),
    ('base11',
r'''    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
  }
#endif
#if !(OKC_WGS && OKC_TOKQ && OKC_WGSFINSS)
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
''',
r'''    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
  }
#endif
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
'''),
    ('base12',
r'''    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
''',
r'''    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1 sums each local route's y^2 over the row (warp
'''),
    ('base13',
r'''    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_TOKQ
      if (tid < 8) {  // tokq: wait until each local expert of this token finished phase B (64 per job)
        const int e = S.eid[tk * 8 + tid];
        if (e >= 0) {
          const int c = S.cnt[e];
          const int need = 64 * (c > SK_T ? (c + FT_T - 1) / FT_T : 1);
          while (ld_acquire(EXPB_PTR + e) < need) __nanosleep(32);
        }
      }
      CTA_SYNC();
#endif
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
''',
r'''    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
'''),
    ('base14',
r'''      CTA_SYNC();  // red is reused by the next token
    }
  }
#if OKC_TOKQ
  // arrival (the old barrier counter, no wait): the last CTA zeroes the per-launch flags and bumps the generation
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    S.wglast[0] = old == (int)gridDim.x - 1;
  }
  CTA_SYNC();
  DBG(4);
  if (S.wglast[0]) {
    __threadfence();
    if (tid < 2) p.ctr[tid] = 0;
    if (tid < 32) EXPB_PTR[tid] = 0;
    for (int i = tid; i < JMAX; i += NT) { p.doneA[i] = 0; p.xready[i] = 0; }
    for (int i = tid; i < JMAX * 16; i += NT) p.doneAs[i] = 0;
    CTA_SYNC();
    if (tid == 0) { __threadfence(); atomicAdd(p.ctr + 3, 1); }
  }
#endif
#elif OKC_FINCOL
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
''',
r'''      CTA_SYNC();  // red is reused by the next token
    }
  }
#elif OKC_FINCOL
  // column-parallel: item = (token, 1024-column segment); thread = 4 columns
  constexpr int FSEG = HD / (NT * 4);  // column segments per token (thread = 4 columns)
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
'''),
    ('base15',
r'''    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
''',
r'''    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
#endif
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
'''),
)

# v150 (decode agent, 2026-10-30): the WGS main build's tail (Params.mode bit 0: per-token expert counters)
_WGH_V150 = (
    ('define',
r'''#define OKC_WGSFINSS 1  //''',
r'''#define OKC_V150 1  // decode agent v150: WGS tail (2-copy flags, epilogue sums of squares, one-pass final, Params.mode)
#define OKC_WGSFINSS 1  //'''),
    ('params',
r'''  __nv_bfloat16* out;
  unsigned long long* dbg;
};''',
r'''  __nv_bfloat16* out;
  unsigned long long* dbg;
  int mode;        // v150 (WGS main build): bit 0 per-token expert counters instead of the grid barrier, bit 1 warming pass
};'''),
    ('smem',
r'''  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy''',
r'''  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  int wpar;         // v150: this launch's copy of the WGS flags (WGS generation & 1)'''),
    ('cta_and',
r'''__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}''',
r'''__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
// v150: AND of a predicate over the NT consumer threads (the CTA_SYNC barrier: bar.red on named barrier 1)
__device__ __forceinline__ bool cta_and(bool v) {
  uint32_t r;
  asm volatile("{\n\t.reg .pred pi, po;\n\tsetp.ne.u32 pi, %1, 0;\n\tbar.red.and.pred po, 1, %2, pi;\n\tselp.u32 %0, 1, 0, po;\n}\n"
               : "=r"(r) : "r"((uint32_t)v), "n"(NT) : "memory");
  return r != 0;
}'''),
    ('fixf',
r'''constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs''',
r'''constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16 * 2;  // + 2 doneAs copies (v150)
// v150: the per-launch WGS flags have two copies, picked by the WGS generation (S.wpar): scratch[0] ints xready 520 /
// 616, per-expert phase-B counters 720 / 752, start-arrival counters 784 / 785, the WGS generation 786; doneAs copies
// 136 x 16 ints apart. A launch zeroes the other copy at its start; the last CTA to read the generations bumps them.
constexpr int V150_XRS = 96, V150_DAS = 136 * 16, V150_EXPB = 720, V150_SA = 784, V150_WGEN = 786;
static_assert(JMAX <= V150_XRS && JMAX * 16 <= V150_DAS && 520 + 2 * V150_XRS <= V150_EXPB, "v150 flag copies");
// the parity: a volatile shared load at every use (as ss_cur): a plain S.wpar read inside wgs_flush (inlined into
// every wgs_seg's first stage) costs the 96-register kernel spills in the stage loops (+50 KB SASS, 385-403 instead of
// 305-317 instructions per stage, local loads inside)
__device__ __forceinline__ int v150_par(const int* w) {
  int v;
  asm volatile("ld.volatile.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(w)));
  return v;
}
#define WGS_XREADY(p, S) ((p).xready + V150_XRS * v150_par(&(S).wpar))
#define WGS_DONEAS(p, S) ((p).doneAs + V150_DAS * v150_par(&(S).wpar))
#define WGS_EXPB(p, S) ((p).ctr + V150_EXPB + 32 * v150_par(&(S).wpar))'''),
    ('xp_xready',
r'''        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }  // every row gathered''',
r'''        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(WGS_XREADY(p, S) + j) < npad) __nanosleep(32); }  // every row gathered'''),
    ('xp_doneas',
r'''      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }''',
r'''      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(WGS_DONEAS(p, S) + j * 16 + st) < 8) __nanosleep(32); }'''),
    ('gather_release',
r'''      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");''',
r'''      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(WGS_XREADY(p, S) + jj), "r"(e - q) : "memory");'''),
    ('flush_kind1',
r'''    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;''',
r'''    // phase A: half of phase-B stage ub / 2 is in xB; phase B (v150): one of the expert's 64 warpgroup unit epilogues
    // (released in every mode: gating it on the mode costs the phase-A stage loops 8-13 rematerialized instructions)
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(PB ? WGS_EXPB(p, S) + S.jE[j] : WGS_DONEAS(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;'''),
    ('finisher',
r'''  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}''',
r'''  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(WGS_DONEAS(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
  } else {  // v150: this warpgroup's rows of the split unit (and their sums of squares) are out
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(WGS_EXPB(p, S) + S.jE[j]), "r"(1) : "memory");
  }
}'''),
    ('seg_end',
r'''  if (!PB) pd = WgsPend{1, u, -1, 0};  // the doneA release waits until the stores are done: deferred''',
r'''  pd = WgsPend{1, u, -1, 0};  // phase A: doneAs release; phase B (v150): expert counter; deferred behind the stores'''),
    ('epi_b',
r'''  } else {
    // y rows only: the per-route sum of squares is formed by the final pass from these rows (no shuffles / atomics)
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
      }
    }
  }
}''',
r'''  } else {
    // v150: y rows, and the route's sum of squares over this warp's 16 rows (8 lanes g per token: one atomic each)
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
    float sq[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      const float v0 = d4[h] * w0, v1 = d4[2 + h] * w1;
      sq[h] = v0 * v0 + v1 * v1;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), v0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), v1);
      }
    }
    // sum over the 8 lanes g of each token, both tokens in one 3-step butterfly: the first step trades halves (lane g
    // keeps token g & 1), then lanes g = 0 / 1 add token tok0 / tok0 + 1 (a fire-and-forget global red: atomicAdd on
    // the generic ss pointer would add the address-space dispatch code at every tile)
    const bool odd = (g & 1) != 0;
    float ks = odd ? sq[1] : sq[0];
    ks += __shfl_xor_sync(0xffffffffu, odd ? sq[0] : sq[1], 4);
    ks += __shfl_xor_sync(0xffffffffu, ks, 8);
    ks += __shfl_xor_sync(0xffffffffu, ks, 16);
    if (g < 2 && tok0 + g < cnt)
      asm volatile("red.global.add.f32 [%0], %1;\n" ::"l"(ss_cur() + jst + tok0 + g), "f"(ks) : "memory");
  }
}'''),
    ('sa_decl',
r'''  if (tid == 0) {
#if !FAST1_ON
#if OKC_WGS
    // full slots (TMA tx)''',
r'''  int v150_sa = 0;  // v150: this CTA's start-arrival rank (thread 0)
  if (tid == 0) {
#if !FAST1_ON
#if OKC_WGS
    // full slots (TMA tx)'''),
    ('gen_read',
r'''    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;''',
r'''#if OKC_WGS
    // v150: the launch generation and the WGS generation; the last CTA to read them bumps both (start arrival, after
    // the routing below), so neither changes before every CTA of this launch has read it
    const int gen = ld_acquire(p.ctr + 3), wgen = ld_acquire(p.ctr + V150_WGEN);
    S.wpar = wgen & 1;
    v150_sa = atomicAdd(p.ctr + V150_SA + (wgen & 1), 1);
#else
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
#endif
    S.gen = gen;'''),
    ('start_zero',
r'''    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }''',
r'''    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
#if OKC_WGS
    // v150: the next WGS launch's flag copy (the previous WGS launch used it and is complete), then the generations
    const int op = S.wpar ^ 1;
    int* xo = p.xready + op * V150_XRS;
    int* dso = p.doneAs + op * V150_DAS;
    int* eo = p.ctr + V150_EXPB + op * 32;
    for (int i = blockIdx.x * NT + tid; i < V150_DAS; i += gridDim.x * NT) {
      dso[i] = 0;
      if (i < V150_XRS) xo[i] = 0;
      if (i < 32) eo[i] = 0;
    }
    if (blockIdx.x == 0 && tid == 0) p.ctr[V150_SA + op] = 0;
    if (tid == 0 && v150_sa == (int)gridDim.x - 1) {  // every CTA has read both generations
      atomicAdd(p.ctr + 3, 1);
      atomicAdd(p.ctr + V150_WGEN, 1);
    }
#endif
  }'''),
    ('barrier',
r'''  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }''',
r'''#if OKC_WGS
  // v150 (WGS only): the flags are per-launch copies zeroed at the start and the generations were bumped at the start:
  // nothing to reset here. Mode bit 0 clear: the grid barrier's arrival here; only a CTA with a token waits for the
  // others, in the final pass (after its warming pass with mode bit 1). Bit 0 set (tokq): no grid barrier at all
  if (!(p.mode & 1)) {
    CTA_SYNC();
    if (tid == 0) {
      __threadfence();
      atomicAdd(p.ctr + 4 + (S.gen & 1), 1);
    }
  }
  DBG(4);
#else
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  // phases A and B are over everywhere: zero the per-launch flags for the next launch (this CTA's share)
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;  // dynamic pool counters
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
#endif'''),
    ('final',
r'''#if OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1''',
r'''#if OKC_WGS && OKC_V150
  // v150: one token's whole row per pass (thread = 8 columns) in one L2 round trip: route weights, the routes' sums of
  // squares (phase-B epilogue atomics) and their y rows all load before the first use; no CTA syncs inside the pass.
  // A token is ready when (mode bit 0 clear) every CTA arrived at the grid barrier, or (bit 0 set, tokq) its local
  // experts' phase-B counters are complete (64 warpgroup unit epilogues per job). Bit 1: if the CTA's first token is
  // not ready yet, its pass runs once right away (the store is overwritten by the real pass), so the pass's
  // instructions are fetched while the CTA waits instead of after
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    const bool tq = (p.mode & 1) != 0;
    bool warm = (p.mode & 2) != 0;
    const float* ssb = ss_cur();
    int dry = 0;
#pragma unroll 1
    for (int tk = blockIdx.x; tk < p.M; tk += dry ? 0 : (int)gridDim.x) {
      dry = 0;
      {
        const int* f = nullptr;  // the flag this thread checks (thread 0: the barrier; tokq: threads 0..7, one slot each)
        int need = 0;
        if (tq) {
          if (tid < 8) {
            const int e = S.eid[tk * 8 + tid];
            if (e >= 0) {
              const int c = S.cnt[e];
              need = 64 * (c > SK_T ? (c + FT_T - 1) / FT_T : 1);
              f = WGS_EXPB(p, S) + e;
            }
          }
        } else if (tid == 0) {
          f = p.ctr + 4 + (S.gen & 1);
          need = (int)gridDim.x;
        }
        if (!cta_and(f == nullptr || ld_acquire(f) >= need)) {
          if (warm) {
            dry = 1;
            warm = false;
          } else {
            if (f != nullptr) { while (ld_acquire(f) < need) __nanosleep(32); }
            CTA_SYNC();
          }
        }
      }
      // every load first, nothing that uses a loaded value in between (a use inside a slot's branch region waits
      // there: one L2 round trip per slot); a non-local slot loads nothing and contributes 0 x rsqrt(eps) x 0
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];  // uniform over the CTA (one token)
      float sv[8], wv[8];
      float4 ya[8], yb[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        sv[sl] = 0.f;
        wv[sl] = 0.f;
        ya[sl] = make_float4(0.f, 0.f, 0.f, 0.f);
        yb[sl] = ya[sl];
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          ya[sl] = __ldcg(src);
          yb[sl] = __ldcg(src + 1);
          sv[sl] = __ldcg(ssb + qv[sl]);
          wv[sl] = __ldg(p.topk_w + tk * 8 + sl);
        }
      }
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const float rw = wv[sl] * rsqrtf(sv[sl] * (1.f / HD) + 1e-8f);
        acc[0] += rw * ya[sl].x; acc[1] += rw * ya[sl].y; acc[2] += rw * ya[sl].z; acc[3] += rw * ya[sl].w;
        acc[4] += rw * yb[sl].x; acc[5] += rw * yb[sl].y; acc[6] += rw * yb[sl].z; acc[7] += rw * yb[sl].w;
      }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
    }
  }
#elif OKC_WGS && OKC_WGSFINSS
  // item = one token's whole row (thread = 8 columns): pass 1'''),
    ('okc_run_mode',
r'''  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;''',
r'''  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  p.mode = (int)(grid >> 16);  // v150: the mode bits ride in the grid argument (only the WGS main build is passed any)
  grid &= 0xFFFF;'''),
)

# A300 (agent A, 2026-10-30): the WGS head (bit 0 generations / warp ranges, bit 1 tile gather)
_WGH_A300 = (
    ('wproducer',
r'''__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
''',
r'''#ifndef OKC_A300
#define OKC_A300 0  // agent A: bit 0 generations + zeroing on the weights warp, warp ranges; bit 1 tile gather; bit 2 early weights
#endif
#define A300_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NT + 32) : "memory")  // (bit 2: the join without the weights warp)
#if OKC_A300 & 1
// A300: the launch generation and the WGS generation (the last CTA to read them bumps both at its start arrival, so
// neither changes before every CTA of this launch has read it), then the other copies' zeroing (the previous launch's
// ss copy and barrier counter, the next WGS launch's flags: nothing in this launch reads them), on the weights producer
// warp while the consumers route (thread 0's two acquire loads and atomic sat before routing's first barrier, the
// zeroing after routing; probe_moe2). Consumer warp 0 syncs on barrier 8 before its first S.wpar use (the gather's
// release); every other reader is behind the join. (Inlined: a noinline callee taking the Params reference makes
// the kernel copy Params to the stack and read every field from local memory: 293 LDL/STL instead of 103.)
__device__ __forceinline__ void a300_gen_zero(const Params& p, Smem& S) {
  const int lane = threadIdx.x & 31;
  int gen = 0, wgen = 0, sa = 0;
  if (lane == 0) {
    gen = ld_acquire(p.ctr + 3);
    wgen = ld_acquire(p.ctr + V150_WGEN);
    sa = atomicAdd(p.ctr + V150_SA + (wgen & 1), 1);
    S.gen = gen;
    S.wpar = wgen & 1;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
  gen = __shfl_sync(0xffffffffu, gen, 0);
  wgen = __shfl_sync(0xffffffffu, wgen, 0);
  float* other = (gen & 1) ? p.ss : p.ss1;
  for (int i = blockIdx.x * 32 + lane; i < EMAX; i += gridDim.x * 32) other[i] = 0.f;
  if (blockIdx.x == 0 && lane == 0) p.ctr[4 + ((gen + 1) & 1)] = 0;
  const int op = (wgen & 1) ^ 1;
  int* xo = p.xready + op * V150_XRS;
  int* dso = p.doneAs + op * V150_DAS;
  int* eo = p.ctr + V150_EXPB + op * 32;
  for (int i = blockIdx.x * 32 + lane; i < V150_DAS; i += gridDim.x * 32) {
    dso[i] = 0;
    if (i < V150_XRS) xo[i] = 0;
    if (i < 32) eo[i] = 0;
  }
  if (blockIdx.x == 0 && lane == 0) p.ctr[V150_SA + op] = 0;
  if (lane == 0 && sa == (int)gridDim.x - 1) {  // every CTA has read both generations
    atomicAdd(p.ctr + 3, 1);
    atomicAdd(p.ctr + V150_WGEN, 1);
  }
  __syncwarp();
  asm volatile("bar.arrive 8, 64;\n" ::: "memory");
}
__device__ __noinline__ void a300_ranges(Smem& S);  // below (the cost tables)
#endif
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_A300 & 1
  a300_gen_zero(p, S);
#endif
#if OKC_A300 & 4
  asm volatile("bar.sync 9, 64;\n" ::: "memory");  // A300: this CTA's stage ranges (the x producer warp)
#else
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
#endif
'''),
    ('xproducer',
r'''  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
''',
r'''  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
#if OKC_A300 & 1
  a300_ranges(S);  // (the whole warp)
#else
  if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
#endif
#if OKC_A300 & 4
  asm volatile("bar.arrive 9, 64;\n" ::: "memory");  // A300: the weights producer starts on the ranges
  A300_JOIN();  // routing tables and stage ranges are ready, phase A's x is gathered
#else
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
#endif
'''),
    ('xready',
r'''        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(WGS_XREADY(p, S) + j) < npad) __nanosleep(32); }  // every row gathered
''',
r'''#if OKC_A300 & 2
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(WGS_XREADY(p, S) + j) < 2 * npad) __nanosleep(32); }  // A300: every (8-row tile, stage) unit
#else
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(WGS_XREADY(p, S) + j) < npad) __nanosleep(32); }  // every row gathered
#endif
'''),
    ('gather_a300',
r'''// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
''',
r'''#if OKC_A300 & 2
// A300: phase A's x gather in (8-row tile, stage) units of 4 KB, split evenly over the CTAs and, in a CTA, over its
// warps: a warp loads the tile's 8 rows x 256 K (lane = one 16 B chunk of KPERM words), transposes them through its
// 4 KB of the x ring (idle until the x producer starts; xor-swizzled 16 B slots, conflict-free both ways) and stores
// the unit as 8 fully coalesced 512 B pieces (the row gather stored 16 B per lane into 32 lines: 1.1-3.6 us + a
// 0.8-1.5 us release at bs 32..256, probe_moe2). Padding rows are stored as zeros; xready counts units (2 npad per job).
static_assert(HD == 16 * 256 && NW * 4096 <= WGS_NX * WGS_XS, "A300 gather: 16 stages of 256 K, staging in the x ring");
__device__ void wgs_gather(const Params& p, Smem& S, unsigned char* stg_base) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nj = S.njF + S.njS;
  const int U = (S.jrow[nj] >> 3) << 4;  // U * G < 2^24: exact float muldiv, no divide routine
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int u0 = muldiv_g(U, blockIdx.x, gridDim.x, rG), u1 = muldiv_g(U, blockIdx.x + 1, gridDim.x, rG);
  uint4* stg = reinterpret_cast<uint4*>(stg_base) + warp * 256;
  const uint32_t* xl = reinterpret_cast<const uint32_t*>(p.x) + (lane >> 3) * 32 + 2 * ((lane & 7) >> 1) + (lane & 1);
#pragma unroll 1
  for (int u = u0 + warp; u < u1; u += NW) {
    const int t8 = (u >> 4) << 3, st = u & 15;  // the tile's first padded row, its stage (256 K)
    int j = 0;  // the tile's job: the number of jobs that end at or before it
    for (int b = 0; b < nj; b += 32)
      j += __popc(__ballot_sync(0xffffffffu, b + lane < nj && S.jrow[b + lane + 1] <= t8));
    const int r0 = t8 - S.jrow[j], cnt = S.jC[j], es = S.jS[j];
    const uint32_t* xs = xl + st * 128;
    uint4 v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[i] = make_uint4(0u, 0u, 0u, 0u);
      if (r0 + i < cnt) {  // chunk P = lane & 7 of 64-group lane >> 3: words 16t + 4(P >> 1) + 2(P & 1) + {0, 1}
        const uint32_t* xr = xs + (size_t)(S.ent[es + r0 + i] >> 3) * (HD / 2);
        v[i] = make_uint4(__ldg(xr), __ldg(xr + 8), __ldg(xr + 16), __ldg(xr + 24));
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) stg[lane * 8 + (i ^ (lane & 7))] = v[i];
    __syncwarp();
    // xA: 16 B index ((st * ntt + r / 8) * 32 + chunk) * 8 + (r & 7) from the job's base, so the unit is the 4 KB at
    // (st * ntt + r0 / 8) * 256; output 16 B m = 32 k + lane is chunk 4 k + (lane >> 3), row lane & 7
    uint4* dst = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((st * ((cnt + 7) >> 3) + (r0 >> 3)) << 8);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int L = 4 * k + (lane >> 3);
      dst[32 * k + lane] = stg[L * 8 + ((lane & 7) ^ (L & 7))];
    }
    __syncwarp();  // (the warp's next unit rewrites its slots)
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // the staging slots are x ring slots (TMA-written next)
  CTA_SYNC();
#if OKC_A300 & 1
  if (tid < 32) asm volatile("bar.sync 8, 64;\n" ::: "memory");  // S.wpar (a300_gen_zero, the weights producer warp)
#endif
  if (tid == 0) {  // count this CTA's units per job (release: the CTA's stores are ordered before by the barrier)
    int jj = 0;
    for (int u = u0; u < u1;) {
      while (S.jrow[jj + 1] <= ((u >> 4) << 3)) ++jj;
      const int e = min(u1, (S.jrow[jj + 1] >> 3) << 4);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(WGS_XREADY(p, S) + jj), "r"(e - u) : "memory");
      u = e;
    }
  }
}
#else
// phase A's x gather (all consumer threads of every CTA, once, at the start): this CTA's share of the jobs' padded rows
'''),
    ('gather_sig',
r'''__device__ void wgs_gather(const Params& p, Smem& S) {
''',
r'''__device__ void wgs_gather(const Params& p, Smem& S, unsigned char* stg_base) {
  (void)stg_base;
'''),
    ('gather_release',
r'''  if (tid == 0) {  // count this CTA's rows per job''',
r'''#if OKC_A300 & 1
  if (tid < 32) asm volatile("bar.sync 8, 64;\n" ::: "memory");  // A300: S.wpar (a300_gen_zero)
#endif
  if (tid == 0) {  // count this CTA's rows per job'''),
    ('gather_end',
r'''// Work left from the previous segment''',
r'''#endif
// Work left from the previous segment'''),
    ('ranges',
r'''__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
''',
r'''__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
#if OKC_A300 & 1
// A300: wgs_ranges on the whole x producer warp: the cost prefixes as warp scans (lane = job), the four range ends on
// lanes 0..3 by wgs_stage_at (the same 32-bit arithmetic: identical ranges; one thread took 2.8-5.3 us here at
// bs 32..256 and finished after the gather at bs 32, probe_moe2)
__device__ __noinline__ void a300_ranges(Smem& S) {
  const int lane = threadIdx.x & 31;
  const int nj = S.njF + S.njS;
  uint32_t ca = 0u, cb = 0u;
  for (int j0 = 0; j0 < nj; j0 += 32) {
    const int j = j0 + lane;
    const bool in = j < nj;
    const int cnt = in ? S.jC[j] : 1;
    const uint32_t a = in ? (uint32_t)wgs_cost(0, cnt) << 9 : 0u, b = in ? (uint32_t)wgs_cost(1, cnt) << 8 : 0u;
    uint32_t ia = a, ib = b;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t ta = __shfl_up_sync(0xffffffffu, ia, d), tb = __shfl_up_sync(0xffffffffu, ib, d);
      if (lane >= d) { ia += ta; ib += tb; }
    }
    if (in) { S.wcp[j] = (int)(ca + ia - a); S.wcpB[j] = (int)(cb + ib - b); }
    ca += __shfl_sync(0xffffffffu, ia, 31);
    cb += __shfl_sync(0xffffffffu, ib, 31);
  }
  if (lane == 0) { S.wcp[nj] = (int)ca; S.wcpB[nj] = (int)cb; S.wtot = ca; S.wtotB = cb; }
  __syncwarp();
  if (lane < 4) {  // S.wgr[2 pb + e]: e = 0 the range start (CTA c), e = 1 its end (CTA c + 1's start, or every stage)
    const int pb = lane >> 1, c = (int)blockIdx.x + (lane & 1), G = (int)gridDim.x;
    S.wgr[lane] = c == G ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, (pb ? cb : ca) * (uint32_t)c / (uint32_t)G);
  }
  __syncwarp();
}
#endif
'''),
    ('sa_decl',
r'''  int v150_sa = 0;  // v150: this CTA's start-arrival rank (thread 0)
''',
r'''#if !(OKC_A300 & 1)
  int v150_sa = 0;  // v150: this CTA's start-arrival rank (thread 0)
#endif
'''),
    ('gen',
r'''    const int gen = ld_acquire(p.ctr + 3), wgen = ld_acquire(p.ctr + V150_WGEN);
    S.wpar = wgen & 1;
    v150_sa = atomicAdd(p.ctr + V150_SA + (wgen & 1), 1);
#else
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
#endif
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
''',
r'''#if !(OKC_A300 & 1)  // (A300: a300_gen_zero, on the weights producer warp)
    const int gen = ld_acquire(p.ctr + 3), wgen = ld_acquire(p.ctr + V150_WGEN);
    S.wpar = wgen & 1;
    v150_sa = atomicAdd(p.ctr + V150_SA + (wgen & 1), 1);
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
#endif
#else
    // the generation can only change at this launch's grid barrier, which every CTA reaches after this point
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);  // launch generation (bumped at the barrier)
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
#endif
  }
'''),
    ('zero_start',
r'''  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
''',
r'''#if !(OKC_WGS && (OKC_A300 & 1))  // (A300: a300_gen_zero, on the weights producer warp)
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
'''),
    ('zero_end',
r'''    if (tid == 0 && v150_sa == (int)gridDim.x - 1) {  // every CTA has read both generations
      atomicAdd(p.ctr + 3, 1);
      atomicAdd(p.ctr + V150_WGEN, 1);
    }
#endif
  }
''',
r'''    if (tid == 0 && v150_sa == (int)gridDim.x - 1) {  // every CTA has read both generations
      atomicAdd(p.ctr + 3, 1);
      atomicAdd(p.ctr + V150_WGEN, 1);
    }
#endif
  }
#endif
'''),
    ('call',
r'''  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);
''',
r'''  wgs_gather(p, S, reinterpret_cast<unsigned char*>(xbuf) + WGS_NS * UNIT_BYTES);  // (A300: staging in the x ring)
  DBG(7);
#if OKC_A300 & 4
  A300_JOIN();
#else
  WS_JOIN();
#endif
  DBG(6);
'''),
)

# WFIT (agent A, 2026-10-30): packed per-build cost tables -DOKC_WGSAH / -DOKC_WGSBH
_WGH_WFIT = (
    ('wfit_tables',
r'''#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
''',
r'''#ifdef OKC_WGSAH  // agent A (gen_v320): the table packed in 8 bytes, tiles 1 in the low byte
#define OKC_WGSA (int)(((OKC_WGSAH) >> 0) & 255ull), (int)(((OKC_WGSAH) >> 8) & 255ull), (int)(((OKC_WGSAH) >> 16) & 255ull), (int)(((OKC_WGSAH) >> 24) & 255ull), (int)(((OKC_WGSAH) >> 32) & 255ull), (int)(((OKC_WGSAH) >> 40) & 255ull), (int)(((OKC_WGSAH) >> 48) & 255ull), (int)(((OKC_WGSAH) >> 56) & 255ull)
#else
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207  // tuner combo_m798 on v182x (Oct 26: bs32 +0.45, bs128 +0.94, bs256 +0.17%)
#endif
#ifdef OKC_WGSBH
#define OKC_WGSB (int)(((OKC_WGSBH) >> 0) & 255ull), (int)(((OKC_WGSBH) >> 8) & 255ull), (int)(((OKC_WGSBH) >> 16) & 255ull), (int)(((OKC_WGSBH) >> 24) & 255ull), (int)(((OKC_WGSBH) >> 32) & 255ull), (int)(((OKC_WGSBH) >> 40) & 255ull), (int)(((OKC_WGSBH) >> 48) & 255ull), (int)(((OKC_WGSBH) >> 56) & 255ull)
#else
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
#endif
'''),
)



def _src_patch(src, patches, what):
    for nm, old, new in patches:
        if src.count(old) != 1:
            raise RuntimeError(f"{what} patch {nm}: {src.count(old)} matches")
        src = src.replace(old, new)
    return src


_WGH_SRC = _src_patch(_src_patch(_src_patch(_src_patch(CUDA_SRC, _WGH_BASE, "wgh base"), _WGH_V150, "v150"), _WGH_A300, "A300"),
                      _WGH_WFIT, "WFIT")

ALT_SRCS["wgh"] = _WGH_SRC  # the fence.acq_rel.gpu before the WGS split-unit read-back is required (without it: a race)
# (gen_v070) helper-warp variants for the TINY / TINY2 / MID builds (work/muse_moe_w4a16/hsp_patches.py)
_HSP_BASE = CUDA_SRC.replace(r'''mbar_init(S.wsre + k, 1); }''', r'''mbar_init(S.wsre + k, WS_NHW); }''')
if _HSP_BASE.count(r'''mbar_init(S.wsre + k, WS_NHW); }''') != 2:
    raise RuntimeError("HSP base: wsre init sites")
_HSP_PATCHES = (
    ('hsp_ntl',
r'''constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);''',
r'''#ifndef OKC_HSP
#define OKC_HSP 0  // HSP (gen_v070): four helper warps, one per SMSP, share every unit's epilogue
#endif
#ifndef OKC_HL2
#define OKC_HL2 0  // HL2 (gen_v070): the trimmed single helper warp
#endif
#define WS_NHW (OKC_HSP ? 4 : 1)  // helper warps (WS)
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 * WS_NHW : 0);'''),
    ('hsp_helper',
r'''__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
''',
r'''#if OKC_HSP || OKC_HL2
// HSP / HL2 (gen_v070): the helper's per-unit instructions compete with compute warps 0 and 4 on SMSP 0 (210 extra
// helper instructions per unit cost 6.7 % at d16, w10 v054_hfat), so the helper is trimmed: unit -> (job, row block)
// by shifts, the job's count / start / global scale loaded once per job, phase B's sums of squares kept per lane
// through a job (one shuffle reduction and one atomic per job and token), SiLU's exp as ex2.approx.ftz (1 + e is the
// same float). HSP: four helper warps (one per SMSP); helper warp hw sums and stores the epilogue values of compute
// lanes 8 hw .. 8 hw + 7 (one partial float per lane), so no SMSP carries a whole epilogue.
static_assert(OKC_WSB && !OKC_WSRF && RED_W == 4, "HSP / HL2: thread 0 issues the ring, 4-float partials");
__device__ __forceinline__ float hsp_silu_mul(float gt, float up) {
  float e;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(gt * -1.44269504088896341f));  // __expf(-gt) without the denormal fix-up
  return SILU_DIV(gt, 1.f + e) * up;
}

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
#if OKC_HSP
  const int hw = (threadIdx.x - NT) >> 5;  // helper warp 0..3 (SMSP hw)
#else
  constexpr int hw = 0;
#endif
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (hw == 0 && lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int Nh = OKC_WSBL ? N - min(OKC_WSBLK, b1 - b0) : N;
#if OKC_HSP
  // this lane's partial: compute lane cl (row gg, tokens 2 tt, 2 tt + 1), float k = h + 2 hi: token 2 tt + h, row gg + 8 hi
  const int cl = 8 * hw + (lane >> 2), k = lane & 3;
  const int gg = cl >> 2, tt = cl & 3, h = k & 1, hi = k >> 1, tk = 2 * tt + h;
  float ssq = 0.f;
#else
  const int gg = lane >> 2, tt = lane & 3;
  float ssq0 = 0.f, ssq1 = 0.f;  // tokens 2 tt, 2 tt + 1 (rows gg and gg + 8)
#endif
  int cj = -1, cnt = 0, st = 0;
  float gsc = 0.f;
  auto flush = [&]() {  // phase B: the job's sums of squares, one atomic per token (warp-uniform call)
#if OKC_HSP
    float s = ssq + __shfl_xor_sync(0xffffffffu, ssq, 2);  // rows gg, gg + 8
    s += __shfl_xor_sync(0xffffffffu, s, 16);                // compute lanes cl, cl ^ 4 (rows gg ^ 1)
    if ((lane & 0x12) == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], s);
    ssq = 0.f;
#else
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      float s = hh ? ssq1 : ssq0;
      s += __shfl_xor_sync(0xffffffffu, s, 4);
      s += __shfl_xor_sync(0xffffffffu, s, 8);
      s += __shfl_xor_sync(0xffffffffu, s, 16);
      if (gg == 0 && 2 * tt + hh < cnt) atomicAdd(&ss_cur()[st + 2 * tt + hh], s);
    }
    ssq0 = 0.f; ssq1 = 0.f;
#endif
  };
  for (int q = 0; q < Nh; ++q) {
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rsh = PB ? RB_B_LOG : RB_A_LOG;
    const int j = njF + (v >> rsh), rb = v & ((1 << rsh) - 1);
    if (j != cj || q == nA) {  // a new job, or phase B's first unit
      if (q > nA) flush();
      cj = j; cnt = S.jC[j]; st = S.jS[j];
      gsc = (PB ? p.gsB : p.gsA)[S.jE[j]] * TWO62;
    }
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
#if OKC_HSP
    float s = S.red[bb][0][cl][k];
#pragma unroll
    for (int w = 1; w < NW; ++w) s += S.red[bb][w][cl][k];
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // this warp's quarter is read (the buffer is free after all four)
    if (!PB) {
      const float up = __shfl_down_sync(0xffffffffu, s, 2);  // float k + 2 of the same compute lane (row gg + 8)
      if (hi == 0 && tk < cnt)
        st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(hsp_silu_mul(s * gsc, up * gsc))));
    } else {
      const int n0 = rb * 16 + gg + 8 * hi;
      const float y = s * gsc * (1.f + __bfloat162float(p.normw[n0]));
      ssq += y * y;
      if (tk < cnt) st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), y);
    }
#else
    float4 sm = *reinterpret_cast<const float4*>(&S.red[bb][0][lane][0]);
#pragma unroll
    for (int w = 1; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // buffer free again
    // sm: (row gg, token 2tt), (row gg, token 2tt+1), (row gg+8, token 2tt), (row gg+8, token 2tt+1)
    if (!PB) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt)
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg),
                       __float2bfloat16(probe_q(hsp_silu_mul((h ? sm.y : sm.x) * gsc, (h ? sm.w : sm.z) * gsc))));
      }
    } else {
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        if (h) ssq1 += v0f * v0f + v1f * v1f; else ssq0 += v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
      }
    }
#endif
    if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
      __threadfence();
#if OKC_HSP
      asm volatile("bar.sync 3, 128;\n" ::: "memory");  // every helper warp's phase-A stores are fenced
#else
      __syncwarp();
#endif
      if (hw == 0 && lane == 0) {
        __threadfence();
        const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
        for (int jj = jf; jj <= jl; ++jj) {
          const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
          atomicAdd(p.doneA + jj, b - a);
        }
      }
    }
  }
  if (Nh > nA) flush();
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#else
__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
'''),
    ('hsp_tail',
r'''  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif
''',
r'''  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif  // OKC_HSP || OKC_HL2
#endif
'''),
    ('hsp2_helper',
r'''__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
#if OKC_HSP
''',
r'''#if OKC_HSP >= 2
__device__ __forceinline__ void mbar_wait_a(uint32_t a, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void mbar_arrive_a(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory");
}
__device__ __forceinline__ float hsp_lds(uint32_t a) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(v) : "r"(a));
  return v;
}
// one unit's quarter of the partial sums (after the wsrf wait); releases the buffer for this warp
#define HSP2_SUM(q)                                                                     \
  const uint32_t bb = (uint32_t)(q) & (NRB - 1);                                        \
  mbar_wait_a(rf0 + bb * 8u, ((uint32_t)(q) / NRB) & 1u);                               \
  const uint32_t ra = rd0 + bb * RBUF;                                                  \
  float s = hsp_lds(ra);                                                                \
  _Pragma("unroll") for (int w = 1; w < NW; ++w) s += hsp_lds(ra + (uint32_t)w * RWARP); \
  __syncwarp();                                                                         \
  if (lane == 0) mbar_arrive_a(re0 + bb * 8u);

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31, hw = (threadIdx.x - NT) >> 5;  // helper warp 0..3 (SMSP hw)
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (hw == 0 && lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int Nh = OKC_WSBL ? N - min(OKC_WSBLK, b1 - b0) : N;
  const int cl = 8 * hw + (lane >> 2), k = lane & 3;
  const int gg = cl >> 2, tt = cl & 3, h = k & 1, hi = k >> 1, tk = 2 * tt + h;
  constexpr uint32_t RWARP = 32u * RED_W * 4u, RBUF = (uint32_t)NW * RWARP;  // bytes per compute warp / partial buffer
  const uint32_t rf0 = smem_u32(S.wsrf), re0 = smem_u32(S.wsre), rd0 = smem_u32(&S.red[0][0][cl][k]);
  if (nA > 0) {  // phase A
    int j = njF + (a0 >> RB_A_LOG), rb = a0 & (RB_A - 1);
    int cnt = S.jC[j], st = S.jS[j];
    float gsc = p.gsA[S.jE[j]] * TWO62;
    for (int q = 0; q < nA; ++q) {
      HSP2_SUM(q)
      const float up = __shfl_down_sync(0xffffffffu, s, 2);  // float k + 2 of the same compute lane (row gg + 8)
      if (hi == 0 && tk < cnt)
        st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(hsp_silu_mul(s * gsc, up * gsc))));
      if (++rb == RB_A && q + 1 < nA) { rb = 0; ++j; cnt = S.jC[j]; st = S.jS[j]; gsc = p.gsA[S.jE[j]] * TWO62; }
    }
    // this CTA's phase A is done: credit its units to the jobs' counters
    __threadfence();
    asm volatile("bar.sync 3, 128;\n" ::: "memory");  // every helper warp's phase-A stores are fenced
    if (hw == 0 && lane == 0) {
      __threadfence();
      const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
      for (int jj = jf; jj <= jl; ++jj) {
        const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
        atomicAdd(p.doneA + jj, b - a);
      }
    }
  }
  if (Nh > nA) {  // phase B
    int j = njF + (b0 >> RB_B_LOG), rb = b0 & (RB_B - 1);
    int cnt = S.jC[j], st = S.jS[j];
    float gsc = p.gsB[S.jE[j]] * TWO62, ssq = 0.f;
    for (int q = nA; q < Nh; ++q) {
      HSP2_SUM(q)
      const int n0 = rb * 16 + gg + 8 * hi;
      const float y = s * gsc * (1.f + __bfloat162float(p.normw[n0]));
      ssq += y * y;
      if (tk < cnt) st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), y);
      if (++rb == RB_B && q + 1 < Nh) {  // the next job: this job's sums of squares, one atomic per token
        float sj = ssq + __shfl_xor_sync(0xffffffffu, ssq, 2);
        sj += __shfl_xor_sync(0xffffffffu, sj, 16);
        if ((lane & 0x12) == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sj);
        ssq = 0.f;
        rb = 0; ++j; cnt = S.jC[j]; st = S.jS[j]; gsc = p.gsB[S.jE[j]] * TWO62;
      }
    }
    float sj = ssq + __shfl_xor_sync(0xffffffffu, ssq, 2);
    sj += __shfl_xor_sync(0xffffffffu, sj, 16);
    if ((lane & 0x12) == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sj);
  }
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#undef HSP2_SUM
#else
__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = threadIdx.x & 31;
#if OKC_HSP
'''),
    ('hsp2_tail',
r'''  if (Nh > nA) flush();
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#else
''',
r'''  if (Nh > nA) flush();
  __threadfence();
  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier
}
#endif  // OKC_HSP >= 2
#else
'''),
    ('ht_def',
r'''#define WS_NHW (OKC_HSP ? 4 : 1)  // helper warps (WS)
''',
r'''#ifndef OKC_HTMA
#define OKC_HTMA 0  // HT (gen_v070): the helper warps issue the ring refills, thread 0 only hands over
#endif
#define WS_NHW (OKC_HSP ? 4 : 1)  // helper warps (WS)
'''),
    ('ht_loop',
r'''    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
''',
r'''    if (tid == 0) {
#if !OKC_HTMA  // HT: helper warp q % 4 refills this slot after the wsrf hand-over below
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
#endif
'''),
    ('ht_h1',
r'''    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
#if OKC_HSP
''',
r'''    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
#if OKC_HTMA  // unit q passed its CTA barrier: its ring slot takes unit q + NS
    static_assert(!OKC_WSBL || OKC_WSBLK < OKC_NS, "HT: every refilled unit's predecessor goes through the helper");
    if (lane == 0 && hw == (q & (WS_NHW - 1)) && q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
#endif
#if OKC_HSP
'''),
    ('ht_h2',
r'''  mbar_wait_a(rf0 + bb * 8u, ((uint32_t)(q) / NRB) & 1u);                               \
''',
r'''  mbar_wait_a(rf0 + bb * 8u, ((uint32_t)(q) / NRB) & 1u);                               \
  if (OKC_HTMA && lane == 0 && hw == (int)((q) & 3) && (q) + OKC_NS < N)                \
    ws_issue(p, S, (q) + OKC_NS, nA, a0, b0, ring);                                     \
'''),
)
ALT_SRCS["hsp"] = _src_patch(_HSP_BASE, _HSP_PATCHES, "hsp")


# ------------------------------------------------------------ v085 safety patches
# Applied to every source a build compiles (CUDA_SRC and each ALT_SRCS entry), after all the variant patches above.
_SAFE_PATCHES = (
    # phase B's skinny loops read other CTAs' inter rows behind doneA: every lane takes the acquire itself (one
    # broadcast load per poll), instead of lane 0 alone followed by __syncwarp
    ('warp-acquire', 3,
r'''        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
''',
r'''        { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }  // every lane acquires: each reads other CTAs' rows
        __syncwarp();
'''),
    # WGS x producer: xA / xB are written by other CTAs with generic stores and read here by bulk copies (async
    # proxy): a proxy fence after the flag acquires orders them
    ('tma-proxy', 1,
r'''      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
''',
r'''      asm volatile("fence.proxy.async.global;\n" ::: "memory");  // generic writes seen by the acquires -> bulk copy
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
'''),
    # the PDL launch (off by default) also asks for co-residency
    ('coop-pdl', 1,
r'''    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
''',
r'''    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    at[1].id = cudaLaunchAttributeCooperative;
    at[1].val.cooperative = 1;
    cfg.attrs = at;
    cfg.numAttrs = 2;
'''),
    # cooperative launch: the grid barrier and the cross-CTA flag waits need every CTA resident at once; the driver
    # guarantees it or refuses the launch (an error, never a hang)
    ('coop-launch', 1,
r'''  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
}
''',
r'''  {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3(okc::NTL);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeCooperative;
    at[0].val.cooperative = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const cudaError_t le = (p.M >= OKC_BIGM) ? cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p)
                                             : cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
    if (le != cudaSuccess) {
      (void)cudaGetLastError();
      TORCH_CHECK(false, "okc_run: cooperative launch failed: ", cudaGetErrorString(le));
    }
  }
}
'''),
)


def _safe_src(src, what):
    for nm, n, old, new in _SAFE_PATCHES:
        if src.count(old) != n:
            raise RuntimeError(f"{what} safety patch {nm}: {src.count(old)} matches, expected {n}")
        src = src.replace(old, new)
    return src


CUDA_SRC = _safe_src(CUDA_SRC, "base")
for _k in list(ALT_SRCS):
    ALT_SRCS[_k] = _safe_src(ALT_SRCS[_k], _k)


BW_SRC = r'''
// H200 memory-pipeline microbenchmarks (diagnostics only).
#include <cuda_runtime.h>
#include <stdint.h>

namespace bw {






// TMA bulk (cp.async.bulk) ring: thread 0 issues 16 KB bulk copies into NS stages, full/empty mbarriers.
template <int NS>
__global__ void __launch_bounds__(256, 1) tma_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = 1024;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(b));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(16384));
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(d), "l"(src + s * ST16), "r"(16384), "r"(b) : "memory");
    }
  };
  if (tid == 0)
    for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
#pragma unroll
    for (int k = 0; k < 4; ++k) { uint4 v = st[k * 256 + tid]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    if (tid == 0) issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
}


template <int SB, int NS, int P, bool MIX>
__global__ void __launch_bounds__(256) tma2_kernel(const uint4* __restrict__ src, size_t n16, unsigned* out) {
  extern __shared__ __align__(128) uint4 sm[];
  __shared__ __align__(8) uint64_t full[NS];
  const int tid = threadIdx.x;
  constexpr int ST16 = SB / 16;
  const size_t stages = n16 / ST16;
  const size_t per = (stages + gridDim.x - 1) / gridDim.x;
  const size_t s0 = blockIdx.x * per, s1 = s0 + per < stages ? s0 + per : stages;
  if (tid == 0) {
    for (int k = 0; k < NS; ++k) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[k]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(b), "r"(MIX ? 257 : 1));
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::);
  }
  __syncthreads();
  auto issue = [&](size_t s, int slot) {
    if (s < s1) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      constexpr int TB = MIX ? SB / 2 : SB;  // bytes via TMA
      if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(b), "r"(TB));
        for (int k = 0; k < P; ++k) {
          uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + k * (TB / 16 / P));
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                       ::"r"(d), "l"(src + s * ST16 + k * (TB / 16 / P)), "r"(TB / P), "r"(b) : "memory");
        }
      }
      if (MIX) {
        for (int i = tid; i < (SB - TB) / 16; i += 256) {
          uint32_t d = (uint32_t)__cvta_generic_to_shared(sm + slot * ST16 + TB / 16 + i);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src + s * ST16 + TB / 16 + i));
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(b) : "memory");
      }
    } else if (MIX) {
      uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
      (void)b;
    }
  };
  for (int k = 0; k < NS; ++k) issue(s0 + k, k);
  unsigned acc = 0;
  for (size_t s = s0; s < s1; ++s) {
    const int slot = (int)((s - s0) % NS);
    const uint32_t phase = (uint32_t)(((s - s0) / NS) & 1);
    uint32_t b = (uint32_t)__cvta_generic_to_shared(&full[slot]);
    asm volatile(
        "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
        ::"r"(b), "r"(phase));
    const uint4* st = sm + slot * ST16;
    for (int k = tid; k < ST16; k += 256) { uint4 v = st[k]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncthreads();
    issue(s + NS, slot);
  }
  if (acc == 0x12345678u) out[0] = acc;
}



}  // namespace bw

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid) {
  const uint4* p = (const uint4*)src.data_ptr();
  size_t n16 = src.numel() * src.element_size() / 16;
  unsigned* o = (unsigned*)out.data_ptr();
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  switch (mode) {
    case 5: {
      size_t sm = 6 * 16384;
      cudaFuncSetAttribute(bw::tma_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma_kernel<6><<<(int)grid, 256, sm, st>>>(p, n16, o); break;
    }
    case 8: { constexpr int SB = 32768, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 12: { constexpr int SB = 16384, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid * 2, 256, sm, st>>>(p, n16, o); break; }
    case 16: { constexpr int SB = 36864, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 17: { constexpr int SB = 36864, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 18: { constexpr int SB = 36864, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 2, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 2, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 19: { constexpr int SB = 36864, NS = 3; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 20: { constexpr int SB = 32768, NS = 5; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 21: { constexpr int SB = 16384, NS = 8; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 22: { constexpr int SB = 16384, NS = 4; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 23: { constexpr int SB = 16384, NS = 6; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid, 256, sm, st>>>(p, n16, o); break; }
    case 24: { constexpr int SB = 36864, NS = 2; size_t sm = (size_t)SB * NS;
      cudaFuncSetAttribute(bw::tma2_kernel<SB, NS, 1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      bw::tma2_kernel<SB, NS, 1, false><<<(int)grid * 2, 256, sm, st>>>(p, n16, o); break; }
  }
}

'''

CPP_SRC = r'''
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
'''


# ------------------------------------------------------------------ extension
# Compiled extensions, keyed by their define tuple. This caches compiled
# code only -- never weights, activations, or outputs.
_EXTS = {}


def _ext(defines=None):
    """Compile (once per define tuple) and return the extension module."""
    defines = tuple(MAIN_DEFINES if defines is None else defines)
    if DQ7 and "-DOKC_DQ7=1" not in defines:
        defines = defines + ("-DOKC_DQ7=1",)
    if defines not in _EXTS:
        src = CUDA_SRC
        defs = [d for d in defines if not d.startswith("@")]
        if torch.cuda.get_device_capability()[0] < 9:
            # Small-smem dev GPUs: keep the ring at 1 slot.
            defs = [d for d in defs if not d.startswith("-DOKC_NS=") and not d.startswith("-DOKC_NSF=")]
        for d in defines:
            if d.startswith("@"):
                src = ALT_SRCS[d[1:]]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
        name = f"okc_moe_w4a16_b085_{tag}_sm{arch}"
        if len(name) > 200:  # CPython looks up PyInit_ + the first 200 characters of a module name only
            name = f"okc_moe_w4a16_b085_{tag[:100]}_h{hashlib.sha1(tag.encode()).hexdigest()[:12]}_sm{arch}"
        _EXTS[defines] = load_inline(
            name=name,
            cpp_sources=CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={MAIN_NS if major >= 9 else 1}"]),
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )
        _EXTS[defines].okc_set_pdl(1 if PDL else 0)
    return _EXTS[defines]


# ------------------------------------------------------------- weight packing
def _scales_prmt_exact(*sfs):
    """True if every scale byte is a positive normal e4m3 value (0x08..0x7E),
    which is what makes the PRMT scale decode exact."""
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True


def _stored_pos():
    """(register, upper-half, code-bit) -> source bit of the packed word.

    The packers scatter each 4-bit code's bits into a 32-bit word at these
    positions; the kernel's dq4() rotates them back. The positions must be
    a permutation of 0..31, which the assert checks.
    """
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  # code bit -> output bit
            for b, o in outb.items():
                if DQ7:
                    # register R reads source bit (o + d) mod 32: r0 as stored, r1 = word << 4, r2 = rotl 10,
                    # r3 = word << 6 except the e1 bits (output 8 / 24), which come from rotl 14
                    d = (0, 28, 22, 18 if b == 2 else 26)[R]
                elif R == 0:
                    d = 0
                elif R == 1:
                    d = 6
                elif R == 2:
                    d = 10
                else:
                    d = 4 if b == 0 else 28   # m via rotr 4, (e0, e1, s) via rotl 4 == rotr 28
                pos[(R, up, b)] = (o + d) % 32
    if sorted(pos.values()) != list(range(32)):
        raise RuntimeError("packed code bit positions are not a permutation of 0..31")
    return pos


_POS = _stored_pos()


def _split_nibbles(codes, rows):
    """Packed uint8 codes (2 per byte) for the given rows -> (nibbles, R).

    nibbles is [R, K]: one 4-bit E2M1 code per entry, low nibble first.
    """
    c = codes.index_select(0, rows)
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(c.shape[0], -1)
    return nib, c.shape[0]


def _scatter_code_bits(word, code, reg, up):
    """OR one code's 4 bits into `word` at the positions for (reg, up)."""
    for b in range(4):
        word |= ((code >> b) & 1) << _POS[(reg, up, b)]
    return word


def _pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk)."""
    nib, R = _split_nibbles(codes, rows)
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  # rb, h, g, w, c, t, s, j
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  # rb, w, c, g, t, s, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            word = _scatter_code_bits(word, code, Rg, up)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _pack_scales(sf, rows):
    """sf fp8 [R0, K/16] -> bytes [rb][w][g][t][c][h]."""
    s = sf.view(torch.uint8).index_select(0, rows)
    R = s.shape[0]
    nb = R // 16
    s = s.view(nb, 2, 8, 8, 8, 4)  # rb, h, g, w, c, t
    s = s.permute(0, 3, 2, 5, 4, 1)  # rb, w, g, t, c, h
    return s.contiguous().view(-1)


def _fat_order(codes, scales):
    """Packed per-expert codes (int32 [rb][w][c][lane][4]) and scales (uint8 [rb][w][lane][c][h]) -> fat order:
    [ub (256-row unit)][ks (16 k-stages of 256)][32 KB codes [rb16][cc][lane][16 B] + 4 KB scales [rb16][lane][cc][h]]."""
    nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
    c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
    c = c.view(torch.uint8).view(nub, 16, 32768)
    s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
    return torch.cat([c, s], dim=2).contiguous().view(-1)


def _unit_order(codes, scales):
    """Per 16-row block: its 32 KB of codes followed by its 4 KB of scales (one TMA copy per skinny unit)."""
    nblk = codes.numel() // (8 * 8 * 32 * 4)
    c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
    s = scales.contiguous().view(nblk, 4096)
    return torch.cat([c, s], dim=1).contiguous().view(-1)


def _pack_codes_nat(codes, rows):
    """Like _pack_codes but with the natural mma k order inside each 16-k step (for the wgmma fat path):
    word (rb, w, c, g, t, s) holds k = 64c + 16s + 8q + 2t + e of rows g (h=0) / g+8 (h=1)."""
    nib, R = _split_nibbles(codes, rows)
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  # rb, h, g, w, c, s, q, t, e
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  # rb, w, c, g, t, s, h, q, e
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            word = _scatter_code_bits(word, code, Rg, up)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


# K permutation inside each 64-wide K group for the WGS path: wgmma k position p = 16s + 8hi + 2t + lo holds
# k = 16t + 4s + 2hi + lo, so lane t's 4 k16 steps of a group all read the same scale block (one decode per 4 steps).
WGS_KPERM = True


def _wgs_codes(codes, rows):
    """The code bytes of the WGS copy: [256-row unit][stage (16)][warpgroup q (4)][8192 B] (see _wgs_copy)."""
    wn = _pack_codes(codes, rows) if WGS_KPERM else _pack_codes_nat(codes, rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    # rb16, w8, c8, g, t, s
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   # rb16, k16 step (w8, c8, s), lane (g, t)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               # ub, q, warp, st, grp, j, lane
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   # ub, st, q, grp, warp, lane, j
    return w.view(torch.uint8).view(nub, 16, 4, 8192)


def _wgs_copy(codes, sf, rows):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    R = rows.numel()
    nub = R // 256
    cb = _wgs_codes(codes, rows)
    s = sf.view(torch.uint8).index_select(0, rows)                    # [R, 256]: one byte per row per 16 K
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             # ub, q, warp, h, g, st, grp, j
    s = s.permute(0, 5, 1, 6, 2, 4, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, grp, warp, g, j, h (OKC_WGSSC2)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)


# ------------------------------------------------------------- SC5: lossless 5-bit block scales
# A group = 8 rows (one row half h of a 16-row block) x one warp's K slice (skinny: 512 K, WGS: 256 K of a stage).
# Each e4m3 scale byte b is stored as b - min(group) (5 bits; prepare checks max - min < 32 for every group of the
# actual weights). Two offsets per pair (row g / row g + 8 of one lane and K block), 16 bits apart, three pairs per
# 32-bit word at bits 5p / 5p + 16. The kernel rebuilds ((word >> 5p) & 0x001F001F) * 16 + (base_g | base_g8 << 16) * 16
# + 0x5C005C00 = the bf16x2 its 8-bit PRMT decode makes from the raw bytes.
def _sc5_pairs(f):
    """f int64 [..., n, 2] 5-bit offsets (pair p, row half h), n <= 3 -> int64 words: pair p's row-g offset at bit 5p,
    its row-(g + 8) offset at bit 5p + 16."""
    w = torch.zeros(f.shape[:-2], dtype=torch.int64, device=f.device)
    for q in range(f.shape[-2]):
        w = w | (f[..., q, 0] << (5 * q)) | (f[..., q, 1] << (5 * q + 16))
    return w


def _sc5_share(pr, nw):
    """pr int64 [..., L, 2] (one pair per thread) -> int64 [..., nw] words: thread l in word l // 3, pair slot l % 3."""
    L = pr.shape[-2]
    pad = torch.zeros(*pr.shape[:-2], 3 * nw - L, 2, dtype=pr.dtype, device=pr.device)
    return _sc5_pairs(torch.cat([pr, pad], dim=-2).view(*pr.shape[:-2], nw, 3, 2))


def _sc5_bytes(w):
    """int64 words (< 2^31) [..., n] -> their little-endian bytes, uint8 [..., 4 n]."""
    return w.to(torch.int32).contiguous().view(torch.uint8)


def _sc5_unit_tail(sf, rows):
    """SC5 scales of every skinny unit (16-row block x full K) of one expert: uint8 [R / 16, SC5_BYTES - 32768] =
    W (2048 B) + X (704 B) + 16 bases + 16 zero bytes, or None if a group (8 rows x a warp's 512 K) spans > 32 codes."""
    s = sf.view(torch.uint8).index_select(0, rows).to(torch.int64)
    nb = s.shape[0] // 16
    s = s.view(nb, 2, 8, 8, 8, 4)                       # rb, h, g, w, c, t: row 16 rb + 8 h + g, K block 32 w + 4 c + t
    base = s.amin(dim=(2, 4, 5))                         # rb, h, w
    f = s - base[:, :, None, :, None, None]
    if int(f.max()) > 31:
        return None
    f = f.permute(0, 3, 2, 5, 4, 1).reshape(nb, 8, 32, 8, 2)   # rb, w, lane (g, t), chunk c, h
    W = torch.stack((_sc5_pairs(f[..., 0:3, :]), _sc5_pairs(f[..., 3:6, :])), dim=-1)   # rb, w, lane, 2
    X = torch.cat((_sc5_share(f[..., 6, :], 11), _sc5_share(f[..., 7, :], 11)), dim=-1)   # rb, w, 22
    hdr = torch.cat((base.permute(0, 2, 1).reshape(nb, 16), torch.zeros_like(base.reshape(nb, 16))), dim=1)
    return torch.cat((_sc5_bytes(W.reshape(nb, 512)), _sc5_bytes(X.reshape(nb, 176)), hdr.to(torch.uint8)), dim=1)


def _sc5_unit_order(codes, tail):
    """Per 16-row block: its 32 KB of codes + its SC5 scales (SC5_BYTES, one bulk copy), at a SC5_STRIDE pitch."""
    nb = tail.shape[0]
    c = codes.contiguous().view(torch.uint8).view(nb, 32768)
    pad = torch.zeros(nb, SC5_STRIDE - SC5_BYTES, dtype=torch.uint8, device=c.device)
    return torch.cat([c, tail, pad], dim=1).contiguous().view(-1)


def _sc5_wgs_copy(codes, sf, rows):
    """SC5 WGS copy: [256-row unit][stage (16)][SC5_STRIDE B]: 4 warpgroup pieces of 8880 B (the 8 KB of codes of
    _wgs_copy, W 512 B, X 176 B) + 32 B of bases (+ pad); None if a group (8 rows x a warp's 256 K) spans > 32 codes."""
    s = sf.view(torch.uint8).index_select(0, rows).to(torch.int64)
    nub = rows.numel() // 256
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)               # ub, q, wp, h, g, st, lg, t: row 256 ub + 64 q + 16 wp + 8 h + g,
    base = s.amin(dim=(4, 6, 7))                         # K block 16 st + 4 lg + t;  base: ub, q, wp, h, st
    f = s - base[:, :, :, :, None, :, None, None]
    if int(f.max()) > 31:
        return None
    f = f.permute(0, 5, 1, 2, 4, 7, 6, 3).reshape(nub, 16, 4, 128, 4, 2)   # ub, st, q, wt (wp, g, t), lg, h
    W = _sc5_pairs(f[..., 0:3, :])                       # ub, st, q, 128
    X = _sc5_share(f[..., 3, :], 44)                     # ub, st, q, 44 (43 used)
    piece = torch.cat((_wgs_codes(codes, rows), _sc5_bytes(W), _sc5_bytes(X)), dim=3)          # ub, st, q, 8880
    hdr = base.permute(0, 4, 1, 2, 3).reshape(nub, 16, 32).to(torch.uint8)                    # byte 2 (q * 4 + wp) + h
    pad = torch.zeros(nub, 16, SC5_STRIDE - SC5_BYTES, dtype=torch.uint8, device=hdr.device)
    return torch.cat((piece.reshape(nub, 16, 4 * 8880), hdr, pad), dim=2).contiguous().view(-1)


def _sc5_skinny_builds_ok():
    """SC5 replaces the unit-order copies read by every non-WGS build: only valid when all of them are the WS
    builds that carry the SC5 decode."""
    return all("-DOKC_WS=1" in tuple(d) for d in (MAIN_DEFINES, TINY_DEFINES, TINY2_DEFINES, MID_DEFINES)
               if d and "-DOKC_WGS=1" not in tuple(d))


def _bdefs(d, state):
    """Build d's full define tuple for this weight set: + the PRMT scale decode flag + -DOKC_SC5=1 when the copy it
    streams (WGS stages / skinny units) is in the SC5 layout."""
    d = tuple(d) + tuple(state.get("sfx", ()))
    sc5s, sc5w = state.get("sc5", (False, False))
    if sc5w if "-DOKC_WGS=1" in d else sc5s:
        d = d + ("-DOKC_SC5=1",)
    return d


def _main_defines(sfx, M=None):
    """The MAIN variant's defines for this weight set: the WGS build needs
    the PRMT scale decode, so without it fall back to the mma.sync build.
    The "@wgh" WGS build takes the cost table for M (token count: shape only)."""
    if not sfx and "-DOKC_WGS=1" in tuple(MAIN_DEFINES):
        return MAIN_FALLBACK_DEFINES
    d = tuple(MAIN_DEFINES)
    if "@wgh" in d and WGS_TABLES:
        for hi, ta, tb in WGS_TABLES:
            if M is None or M <= hi:
                return tuple(x for x in d if x != "@wgh") + (f"-DOKC_WGSAH={ta}", f"-DOKC_WGSBH={tb}", "@wgh")
    return d


def _main_variants(sfx):
    """Every MAIN build kernel_fn may pick (one per cost table)."""
    if WGS_TABLES and "@wgh" in tuple(MAIN_DEFINES) and _main_defines(sfx) != MAIN_FALLBACK_DEFINES:
        return [_main_defines(sfx, hi) for hi, _a, _b in WGS_TABLES]
    return [_main_defines(sfx)]


def _wgs_on():
    """True if any compiled variant uses the WGS (RS-wgmma) kernel, which
    needs the extra WGS weight copies in state["w"]."""
    return any("-DOKC_WGS=1" in tuple(d) for d in (MAIN_DEFINES, TINY_DEFINES, TINY2_DEFINES, MID_DEFINES))


# ---------------------------------------------------------------------- setup
def _prepare_fast(weights):
    """Repack weights (bit/byte permutations only, never requantized) into
    the layouts the kernel reads, allocate scratch, and compile the
    variants kernel_fn may pick. Called once; not on the timed path.
    Raises when the weights do not fit the kernel (prepare then falls back)."""
    t0 = time.time()
    dev = weights["w13"].device
    lo, cnt = LOCAL_START, LOCAL_EXPERTS
    I = weights["w13"].shape[1] // 2   # gate+up rows: w13 stacks [gate; up]
    H = weights["w2"].shape[1]         # down rows
    if not (I == HIDDEN and H == HIDDEN and weights["w13"].shape[2] * 2 == HIDDEN and weights["w2"].shape[2] * 2 == HIDDEN):
        raise RuntimeError(f"weight shapes {tuple(weights['w13'].shape)} / {tuple(weights['w2'].shape)} are not the kernel's")
    if weights["w13"].shape[0] < lo + cnt or weights["w2"].shape[0] < lo + cnt:
        raise RuntimeError("fewer experts than the kernel's local range")
    if weights["post_norm_w"].dtype != torch.bfloat16:  # kept as given (TASK.md: BF16 weights stay BF16)
        raise RuntimeError("post_norm_w is not bf16")
    if weights["w13_gs"].dtype != torch.float32 or weights["w2_gs"].dtype != torch.float32:  # stored as given
        raise RuntimeError("global scales are not fp32")
    if weights["w13"].dtype != torch.uint8 or weights["w2"].dtype != torch.uint8:
        raise RuntimeError("packed codes are not uint8")
    # rowsA: gate rows and up rows interleaved in 8-row groups (the mma
    # row-group size); rowsB: down rows in order.
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    wA, sA, wB, sB = [], [], [], []
    for e in range(lo, lo + cnt):
        wA.append(_pack_codes(weights["w13"][e], rowsA))
        sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(_pack_codes(weights["w2"][e], rowsB))
        sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
    # LUT: fp8 e4m3 byte -> bf16x2(value * 2^64), packed into one int32.
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    prmt = PRMT_SCALES and _scales_prmt_exact(weights["w13_sf"][lo:lo + cnt], weights["w2_sf"][lo:lo + cnt])
    # SC5 skinny units: the scales of every unit as 5-bit offsets (None anywhere: the 8-bit layout for all)
    tA = tB = None
    if SC5 and prmt and _sc5_skinny_builds_ok():
        tA = [_sc5_unit_tail(weights["w13_sf"][e], rowsA) for e in range(lo, lo + cnt)]
        tB = [_sc5_unit_tail(weights["w2_sf"][e], rowsB) for e in range(lo, lo + cnt)]
        if any(t is None for t in tA + tB):
            tA = tB = None
    if tA is not None:
        uA = torch.cat([_sc5_unit_order(a, t) for a, t in zip(wA, tA)])
        uB = torch.cat([_sc5_unit_order(b, t) for b, t in zip(wB, tB)])
    else:
        uA = torch.cat([_unit_order(a, b) for a, b in zip(wA, sA)])
        uB = torch.cat([_unit_order(a, b) for a, b in zip(wB, sB)])
    state = {
        # w: [packed gate/up codes, scales, group scales,
        #     packed down codes, scales, group scales,
        #     e4m3 LUT, post_norm_w (bf16 as given; kernel forms 1 + w in FP32),
        #     fat-order gate/up, fat-order down,
        #     unit-order gate/up, unit-order down,
        #     (+ WGS copies appended below, w[12] / w[13], when any variant uses WGS)]
        "w": [torch.cat(wA), torch.cat(sA), weights["w13_gs"][lo:lo + cnt].contiguous(),
              torch.cat(wB), torch.cat(sB), weights["w2_gs"][lo:lo + cnt].contiguous(),
              lut.contiguous(), weights["post_norm_w"].contiguous(),
              torch.cat([_fat_order(a, b) for a, b in zip(wA, sA)]), torch.cat([_fat_order(a, b) for a, b in zip(wB, sB)]),
              uA, uB],
        "lo": lo, "cnt": cnt, "scratch": _make_scratch(dev),
        "grid": torch.cuda.get_device_properties(dev).multi_processor_count,
    }
    cA = cB = None
    if _wgs_on():  # w[12], w[13]: the WGS copies (one more copy of the weights, laid out per 64-row unit)
        if SC5 and prmt:
            cA = [_sc5_wgs_copy(weights["w13"][e], weights["w13_sf"][e], rowsA) for e in range(lo, lo + cnt)]
            cB = [_sc5_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB) for e in range(lo, lo + cnt)]
            if any(c is None for c in cA + cB):
                cA = cB = None
        if cA is not None:
            state["w"].append(torch.cat(cA))
            state["w"].append(torch.cat(cB))
        else:
            state["w"].append(torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], rowsA) for e in range(lo, lo + cnt)]))
            state["w"].append(torch.cat([_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB) for e in range(lo, lo + cnt)]))
    state["sfx"] = ("-DOKC_PRMTSC=1",) if prmt else ()
    state["sc5"] = (tA is not None, cA is not None)
    print(f"[sc5] prmt={prmt} skinny={tA is not None} wgs={cA is not None} unit/stage bytes "
          f"{SC5_BYTES if tA is not None else 36864}/{SC5_BYTES if cA is not None else 36864}", flush=True)
    mods = [_bdefs(d, state) for d in _main_variants(state["sfx"])]
    for d in (TINY_DEFINES, TINY2_DEFINES, MID_DEFINES):
        if d and (state["sfx"] or "-DOKC_WGS=1" not in tuple(d)):  # the WGS build needs the PRMT scale decode
            mods.append(_bdefs(d, state))
    print(f"[v090] t={time.time() - t0:.1f}s repacked; compiling {len(mods)} builds in parallel", flush=True)
    # v090: the builds are independent (distinct extension names, so distinct build dirs): compile them at once.
    # okc-allow: optics: the threads only parallelize load_inline (nvcc runs outside the GIL), joined right here
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(mods)) as pool:
        list(pool.map(_ext, mods))  # re-raises any build error
    torch.cuda.synchronize()
    print(f"[v090] t={time.time() - t0:.1f}s compiled", flush=True)
    if DIAG:
        _diag(state, weights)
    return state


# ------------------------------------------------------------- slow fallback
# Used for any configuration the fast kernel does not cover (another local expert range, e.g. sharding "full", or
# weights it cannot take). Every local expert runs on every token, kept by the token's routing weight (zero where it
# was not routed there), as in the task's starter. The NVFP4 weights are expanded to FP32 inside the call (e2m1 *
# block scale * global scale) and never stored. Graph-capturable: it branches on shapes and config only and reads
# nothing back to the host.
POST_NORM_EPS = 1e-8
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fb_lut(dev):
    """uint8 code byte -> its two E2M1 values (low nibble first), fp32 [256, 2]."""
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32)
    c = torch.arange(16)
    v16 = torch.where(c >= 8, -grid[c & 7], grid[c & 7])
    b = torch.arange(256)
    return torch.stack((v16[b & 15], v16[b >> 4]), -1).contiguous().to(dev)


def _fb_dequant(codes, sf, gs, lut):
    """codes uint8 [R, K/2], sf e4m3 [R, K/16], gs fp32 [] -> fp32 [R, K]."""
    R = codes.shape[0]
    v = lut[codes.long()].reshape(R, -1, 16)
    return (v * sf.float().unsqueeze(-1)).reshape(R, -1) * gs


def _fb_moe(x, ids, tw, w, lut, lo, cnt):
    M = x.shape[0]
    D = w["w13"].shape[1] // 2
    xf = x.float()
    ids = ids.long()
    tw = tw.float()
    out = torch.zeros((M, w["w2"].shape[1]), dtype=torch.float32, device=x.device)
    nw = 1.0 + w["post_norm_w"].float()
    for e in range(max(0, lo), min(lo + cnt, w["w13"].shape[0])):
        g = (tw * (ids == e)).sum(-1)
        h = xf @ _fb_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e], lut).T
        inter = (torch.nn.functional.silu(h[:, :D]) * h[:, D:]).to(torch.bfloat16).float()  # bf16 as SGLang keeps it
        y = (inter @ _fb_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e], lut).T) * nw
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
        out += g[:, None] * y
    return out.to(torch.bfloat16)


def _selftest(state, weights):
    """Run every compiled build (twice) on synthetic routings made with a local generator (the global RNG is
    untouched) and compare with the fallback path. Raises on a mismatch, NaN / Inf or a launch error."""
    dev = weights["w13"].device
    gen = torch.Generator(device=dev).manual_seed(8501)
    cases = []
    for M, hot in ((1, False), (4, False), (8, False), (16, False), (32, False), (128, False), (256, True), (257, False)):
        x = torch.randn((M, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
        sc = torch.rand((M, N_EXPERTS_TOTAL), generator=gen, device=dev)
        sc[:, LOCAL_START:LOCAL_START + LOCAL_EXPERTS] += 0.15  # about half the routes local
        if hot:
            sc[:, LOCAL_START] += 2.0  # one expert takes every token: the fat path
        r = sc.topk(TOP_K, dim=-1)
        cases.append((x, r.indices.to(torch.int32), (r.values / r.values.sum(-1, keepdim=True)).float()))
    ref = _fb_moe(torch.cat([c[0] for c in cases]), torch.cat([c[1] for c in cases]), torch.cat([c[2] for c in cases]),
                  weights, state["fb_lut"], LOCAL_START, LOCAL_EXPERTS).float()
    m0 = 0
    for x, ids, tw in cases:
        M = x.shape[0]
        rr = ref[m0:m0 + M]
        rn = rr.norm(dim=-1)
        for rep in range(2):
            o = _fast_call(state, x, ids, tw).float()
            if not bool(torch.isfinite(o).all()):
                raise RuntimeError(f"self-test M={M}: NaN / Inf")
            err = float((o - rr).norm() / rr.norm().clamp(min=1e-12))
            row = float(((o - rr).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
            if not (err < 1e-2 and row < 5e-2):
                raise RuntimeError(f"self-test M={M} rep {rep}: rel L2 {err:.2e}, worst row {row:.2e}")
        m0 += M
    torch.cuda.synchronize()
    print(f"[v085] self-test OK ({len(cases)} cases x 2)", flush=True)


def prepare(weights):
    """The fast kernel's state (repacked weights, scratch, compiled builds) after a self-test of every build, or,
    if anything fails, the fallback path's state. Called once; not on the timed path. Keeps references to the
    given weights (no copies) for the fallback path."""
    dev = weights["w13"].device
    state = {"fast": False, "src": weights, "fb_lut": _fb_lut(dev)}
    try:
        t0 = time.time()
        state.update(_prepare_fast(weights))
        _selftest(state, weights)
        print(f"[v090] prepare fast path ready in {time.time() - t0:.1f}s", flush=True)
        state["fast"] = True
    except Exception as ex:
        state["fast"] = False
        print(f"[v085] fast path off ({type(ex).__name__}: {ex}); kernel_fn runs the fallback path", flush=True)
    return state


def _make_scratch(dev):
    """Per-call scratch, sized for CHUNK_TOKENS tokens. Layout is fixed by
    the CUDA launcher (okc_run), which indexes scratch[0]/[1]/[2] as:
    [0] int32: 8 counters + 1024 doneA slots + the second ss copy
    (ss1, EMAX floats viewed as int32 -- the launcher takes its address
    as ib + 8 + 1024);
    [1] fp32: ss (EMAX) + ybuf (EMAX x HIDDEN) + WGS stream-K fixups;
    [2] bf16: inter (EMAX x HIDDEN) + WGS xA / xB (2 x WGS_XROWS x HIDDEN).
    Zeroed here so the first call starts from a clean grid barrier state;
    the kernel re-zeroes what it needs between launches.
    """
    return [torch.zeros(8 + 1024 + EMAX, dtype=torch.int32, device=dev),
            torch.zeros(EMAX + EMAX * HIDDEN + WGS_FIXF, dtype=torch.float32, device=dev),
            torch.zeros((EMAX + 2 * WGS_XROWS) * HIDDEN, dtype=torch.bfloat16, device=dev)]


# ------------------------------------------------------------------ timed path
def kernel_fn(data, state):
    """Run one MoE layer. Fresh output every call; no host syncs; the only
    per-call work is picking the compiled variant by token count and
    launching it in CHUNK_TOKENS-token slices. A configuration the kernel
    does not cover (a branch on config and shapes only) runs the fallback."""
    cfg = data["config"]
    x = data["hidden_states"]
    ids, tw = data["topk_ids"], data["topk_weights"]
    if not (state["fast"] and cfg["local_start"] == state["lo"] and cfg["local_count"] == state["cnt"]
            and cfg.get("top_k", TOP_K) == TOP_K and x.dim() == 2 and x.shape[1] == HIDDEN
            and x.dtype == torch.bfloat16 and tuple(ids.shape) == (x.shape[0], TOP_K)):
        return _fb_moe(x, ids, tw, state["src"], state["fb_lut"], cfg["local_start"], cfg["local_count"])
    return _fast_call(state, x, ids, tw)


def _fast_call(state, x, ids, tw):
    x = x.contiguous()
    M = x.shape[0]
    ids = ids.to(torch.int32).contiguous()
    if ids.data_ptr() & 15:  # the WGS routing loads topk_ids 16 B at a time (an offset view gets an aligned copy)
        ids = ids.clone()
    tw = tw.to(torch.float32).contiguous()
    out = torch.empty((M, HIDDEN), dtype=torch.bfloat16, device=x.device)
    sfx = state.get("sfx", ())
    gm = 0
    if TINY_DEFINES and M <= TINY_MAX_M:
        ext = _ext(_bdefs(TINY_DEFINES, state))
    elif TINY2_DEFINES and M <= TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(TINY2_DEFINES)):
        ext = _ext(_bdefs(TINY2_DEFINES, state))  # (the WGS build needs the PRMT scale decode: else MID/MAIN)
    elif MID_DEFINES and M <= MID_MAX_M:
        ext = _ext(_bdefs(MID_DEFINES, state))
    else:
        dm = _main_defines(sfx, M)
        ext = _ext(_bdefs(dm, state))
        if "@wgh" in dm:
            gm = WGH_MODE << 16  # (the "@wgh" source reads the mode bits from the grid argument)
    for m0 in range(0, M, CHUNK_TOKENS):
        m1 = min(M, m0 + CHUNK_TOKENS)
        ext.okc_run(state["w"], state["scratch"], x[m0:m1], ids[m0:m1], tw[m0:m1], out[m0:m1],
                    state["lo"], state["cnt"], state["grid"] | gm)
    return out


# ---------------------------------------------------------------------------------------------- diagnostics
# Everything below runs only when DIAG is set (prepare) or when called
# directly. It is development tooling, not part of the submission's timed
# path, and it is the only place host synchronisation appears besides
# prepare()'s one-time compile wait.
def _gen_input(bs, seed, dev):
    """Same distribution as the task's generate_input (normal dist, busiest-GPU relabel), written here."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS_TOTAL), generator=gen, device=dev)
    z = torch.randn((N_EXPERTS_TOTAL,), generator=gen, device=dev)
    u = torch.rand((N_EXPERTS_TOTAL,), generator=gen, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=TOP_K)
    ids = (ids - LOCAL_EXPERTS * pairs.argmax()) % N_EXPERTS_TOTAL
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": HIDDEN, "d_expert": HIDDEN, "n_experts": N_EXPERTS_TOTAL, "local_start": LOCAL_START,
                       "local_count": LOCAL_EXPERTS, "top_k": TOP_K, "sharding": SHARDING}}


def _time_graph(fn, reps=40):
    """Median / min CUDA-graph replay time of fn in microseconds."""
    dev = torch.device("cuda")
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.add_(1)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    del g
    return ts[len(ts) // 2], ts[0]


def _diag_bw(state):
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    ext = load_inline(name=f"okc_bw_v216x_sm{arch}", cpp_sources="#include <torch/extension.h>\nvoid bw_run(torch::Tensor src, torch::Tensor out, int64_t mode, int64_t grid);",
                      cuda_sources=BW_SRC, functions=["bw_run"],
                      extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"], verbose=False)
    src = state["w"][0]
    nbytes = src.numel() * src.element_size()
    out = torch.zeros(4, dtype=torch.int32, device=src.device)
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=src.device)
    names = {5: "tma 6x16KB", 8: "tma 4x32KB", 12: "tma 4x16KB 2cta/sm", 16: "tma 5x36KB",
             17: "tma 4x36KB", 18: "tma 4x36KB as 2x18KB", 19: "tma 3x36KB", 20: "tma 5x32KB", 21: "tma2 8x16KB",
             22: "tma2 4x16KB", 23: "tma2 6x16KB", 24: "tma 2x36KB 2cta/sm"}
    for mode in DIAG_BW_MODES:
        if mode >= 5 and mode != 7 and major < 9:
            continue
        ts = []
        for _ in range(5):
            flush.add_(1)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            ext.bw_run(src, out, mode, state["grid"])
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000.0)
        ts.sort()
        print(f"[diag-bw] {names[mode]:18s}: {ts[2]:8.1f} us  {nbytes / ts[2] / 1e6:6.2f} TB/s", flush=True)


def _diag(state, weights):
    import time
    dev = weights["w13"].device
    t0 = time.time()
    if DIAG_BW:
        try:
            _diag_bw(state)
        except Exception as ex:  # diagnostics must never break the run
            print(f"[diag-bw] failed: {ex}", flush=True)
    shapes = [1, 16, 32, 8, 128, 256]
    variants = {"main": tuple(MAIN_DEFINES)}
    variants.update(DIAG_VARIANTS)
    # compile the variants concurrently (separate extension dirs; nvcc runs in subprocesses)
    import concurrent.futures  # okc-allow: optics: the threads only parallelize load_inline (diagnostics only)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_ext, [d for d in variants.values() if tuple(d) not in _EXTS]))
    print(f"[diag] compiled {len(variants)} variants in {time.time() - t0:.0f}s", flush=True)
    lines = []
    for bs in shapes:
        d = _gen_input(bs, 1234 + bs, dev)
        ids = d["topk_ids"]
        touched = int(torch.unique(ids[(ids >= 0) & (ids < LOCAL_EXPERTS)]).numel())
        row = [f"bs={bs:3d} touched={touched:2d}"]
        for name, defs in variants.items():
            ext = _ext(defs)
            x = d["hidden_states"]
            idt = ids.to(torch.int32).contiguous()
            tw = d["topk_weights"].float().contiguous()
            out = torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev)

            def fn():
                ext.okc_run(state["w"], state["scratch"], x, idt, tw, out, LOCAL_START, LOCAL_EXPERTS, state["grid"])

            med, mn = _time_graph(fn)
            # 28.31 MB: approx. weight bytes streamed per touched expert
            # (NVFP4 codes + scales for gate/up + down), diagnostic only.
            row.append(f"{name}: {med:7.1f}us ({touched * 28.31 / med:4.2f} TB/s)")
        lines.append("  ".join(row))
        print("[diag] " + lines[-1], flush=True)
    # timeline of the main variant with debug timestamps
    for dbgname in [k for k, v in variants.items() if "-DOKC_DEBUG=1" in v]:
        ext = _ext(variants[dbgname])
        G = state["grid"]
        dbg = torch.zeros(G * 8 + G * 8 * 4, dtype=torch.int64, device=dev)
        sc = list(state["scratch"]) + [dbg]
        names = ["start", "routed", "A done", "B done", "barrier", "final"]
        for bs in shapes:
            d = _gen_input(bs, 1234 + bs, dev)
            x = d["hidden_states"]
            idt = d["topk_ids"].to(torch.int32).contiguous()
            tw = d["topk_weights"].float().contiguous()
            out = torch.empty((bs, HIDDEN), dtype=torch.bfloat16, device=dev)
            flush = torch.empty(64 * 1024 * 1024, dtype=torch.int32, device=dev)
            for _ in range(3):
                flush.add_(1)
                dbg.zero_()
                torch.cuda.synchronize()
                ext.okc_run(state["w"], sc, x, idt, tw, out, LOCAL_START, LOCAL_EXPERTS, G)
                torch.cuda.synchronize()
            tt = dbg[:G * 8].view(G, 8).cpu().double()
            t00 = tt[:, 0][tt[:, 0] > 0].min()
            parts = []
            for k, n in enumerate(names):
                col = tt[:, k][tt[:, k] > 0]
                if col.numel():
                    parts.append(f"{n} {(col.min() - t00) / 1000:.1f}-{(col.max() - t00) / 1000:.1f}")
            print(f"[diag] timeline {dbgname} bs={bs}: " + "  ".join(parts), flush=True)
    print(f"[diag] total {time.time() - t0:.0f}s", flush=True)
