# okc: problem=muse_decode_layer
# s303f_rep (EROREx, 2026-11-03): byte-distinct replicate of s303_att_exp.py.
# s302_c001_pin_att (EROREx, 2026-11-03): s301 + attention = board t853_both build (early PDL trigger, layer pooler L2 prefetch kept) + pr T=128 cuBLASLt N-batch-3 pick; KEELL/muse_decode_layer/stack3/build.py.
# s303_att_exp (EROREx, 2026-11-03): s302 + attention P by the bit-trick exp2 (f16 fma -> e4m3 code, SCH_K 6, row-max offset test = KEELL/attn_exp x012; organizer OK 11-02, attention P only, every other exp exact); KEELL/muse_decode_layer/stack3/build.py.
# --- attention kernel (s302) = EROREx t853_both build (KEELL/attn_gap/c000_t853.py); its attribution lines, verbatim:
# + keell0renz stack [bulk]: TMA bulk-reduction merge (cp.reduce.async.bulk) + 72-float workspace rows for the q_len-1 per-request kernels that are not one-tile builds (Madhavan113/ParadigmKernelChallenge c0f3460 dab77b3 012d92b, Apache-2.0)
# Source: team keell0renz, muse-open-kernel-challenge kernels/muse_attention_decode.py (Apache-2.0; Copyright 2026 team
# keell0renz (Bohdan Agarkov); their NOTICE lists vjeux/okc-2026, ZhimaoL/sglang and Madhavan113/ParadigmKernelChallenge,
# all Apache-2.0, credited above). This file is distributed under Apache-2.0.
# c001_t213_clean (EROREx, 2026-11-02): t213_a207_g121 minus provably unused code: the rs128 rq bake (RQ_USE is empty: pw57_flush build, flush validate, fixed cost, rq extension thread and bake), _make_chen, 3 unreachable defs, the two never-run `if _P9_D4:` blocks; kernels, races and numerics unchanged; KEELL/muse_decode_layer/clean/gen_clean.py.
# t213_a207_g121 (EROREx, 2026-11-02): f005_norq + a207 (a202 + pre-wait L2 prefetch of the stream row in _graft_row_boundary_pre / _graft_row_boundary only) + g121 route race T256; KEELL/muse_decode_layer/pf_audit/build_t213.py.
# f004_all (EROREx, 2026-11-02): n904_slim + SC5W: lossless 5-bit WGS block scales (group minimum + 5-bit offsets, 35552 B per stage; SC6 / 8-bit fallback when a group of the actual weights spans 32+ codes); KEELL/muse_decode_layer/p9/gen_p9.py.
# n904_dnstack (EROREx, 2026-11-02): s802_safe_hsp8.py + dense set (T = 1 qkv / unpooler slice offered bf16_proj's cgemv (PDL); o at T = 128 / 256 offered ol_projection (NathenFernandes, MIT) and the bf16_proj ag cuBLASLt pin; T = 128 pooler + router offered rq pooler + FP32 cuBLASLt router over [hi; lo]); each raced in prepare, kept only on a win (KEELL/muse_decode_layer/dense/gen_dn.py).
# s802_safe_hsp8 (EROREx, 2026-11-02): m801_stack + in-layer MoE safety set: every lane acquires the phase-B doneA flags, unbounded MFIN waits, MID A->B by the doneA counters (no data-flag DF), fence.proxy.async.global before the WGS xA / xB bulk copies, a once-per-build co-residency check in okc_run; HSP2 at T 5..8 (TINY2 build): four helper warps share every unit's epilogue (port of EROREx's own BABAGAZ/muse_moe_w4a16 gen_v070 hsp2 helper); KEELL/muse_decode_layer/safe/gen_safe.py.
# m801_stack (EROREx, 2026-11-02): m800_aw_tmp + T >= 17 row boundaries: T 33..192 in the T 17..32 kernel's structure (norm weights / gates before the wait, stream + branch right after it); L2 prefetch of the stream row right after the wait (T 17..32, T > 192) and of the weight vectors before it (T > 192); bit-identical outputs (KEELL/muse_decode_layer/weak/gen_w7.py).
# m800_aw_tmp (EROREx, 2026-11-02): p602_w.py + q_len 1 attention split merge by one bulk async reduction per CTA, all q_len 1 kernels (port of v450_bred); B == 1 attention grids prefetch their page-table lines toward L2 at entry (port of attention v031 PFIDX); KEELL/muse_decode_layer/attn/gen_a7.py.
# p602_w (EROREx, 2026-11-02): h512 with its MoE changes baked into the sources + WGS MoE prologue edits (P6_PRE, P6_WARM=1, P6_DX=0: mbarrier setup and a warm run of the stage-range code before griddepcontrol.wait, the first phase-A x stages straight from x); KEELL/muse_decode_layer/prolog/gen_p6cand.py.
# h512_hfm501 (EROREx, 2026-11-02): h501_ex27 + in-layer MoE changes at T >= 9 (hsp+fin1+mpdl; KEELL/muse_decode_layer/moe/gen_h511.py, see the h51x sections at the end).
# h501 (EROREx, 2026-11-01, EX also at T = 2..7 (bf16_proj sliced unpooler called directly; each T kept only if it times no slower than keell's chain), per-token chain self-test (FP64 front-reference margin for near-tie expert flips)): base = team keell0renz's kernels/muse_decode_layer.py (https://github.com/keell0renz/muse-open-kernel-challenge, commit 8e30e1b, Apache-2.0; Copyright 2026 team keell0renz (Bohdan Agarkov)); every attribution line of that file follows unchanged. Modified by EROREx: T = 2..8 take EROREx's v480_cpbx GLE chain (gos o projection + in-loop-router pooler/router + deferred-norm route + sliced boundary 2; section 'EX' at the end); MFAB in-prepare A/B diagnostics removed; SK2_DELAY_ESC env read -> 0; // comment lines of the CUDA sources dropped for the 1 MB file limit (code unchanged).
# stack_all1: stack_mf1 (MFIN, no MoE grid barrier) + SC6 (6-bit block scales, WGS MoE builds) + PRF32 + PDL at T=128 (moe-fin bundle; L2 hints at T=1 left out: judge d1 25.7-26.6 us with them vs 24.5-25.8 without).
# st2l: st1l with OKC_WGSFCI 4 (E374).
# st1l: ybfl.py + OKC_WGSFCI 5 + weight-ring pacing at M <= 64 (E372).
# ybfl: cepil_u160.py + WGS ybuf in bf16 (epilogue store + final-pass reads; harness/make_ybf.py).
# cepil_u160: pcx_h1 + in-layer MoE per-unit stream-K cost U = 160 (A) / 140 (B), zhebrak SK_CEPI (harness/make_cepi.py --layer).
# pcx_h1: pre-wait control of the sk2 kernels: hoist (harness/make_pcx.py; team keell0renz; base stack_all1_snap1)
# + alygg77/okc-muse-spark-kernels kernels/muse_decode_layer.py (Apache-2.0, commit 6cae49a); header below is theirs. Patched by team keell0renz: trig, rq, atrig0, rtrig8, pf4.
# bt_sc6: lossless 6-bit block scales in the layer's WGS MoE builds (T >= 17), moe-scale SC6 via harness/make_bt_sc6.py.
# stack_df1: data-flag A->B hand-off in the MoE MID build (T = 9..16), harness/make_dl_df.py (moe-small E173).
# + chenwainuo/okc muse_bf16_proj rs128 GEMVs (Apache-2.0) at T = 128 / 256 for qkv and T = 128 for the unpooler (team keell0renz).
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# attribution is retained here.
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
# --- EROREx v480_cpbx header (source of the EX section), kept verbatim:
# Derived from okc-2026 (https://github.com/vjeux/okc-2026), licensed under the Apache License 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Modifications by EROREx (cooperative launch for spin finalize, no speculative length; K-permuted WGS expert packing; attention early PDL trigger + o-proj weights fully prefetched before the wait; route_split 2 warps; verify-16 sliding attention tune; dense picks pinned for T <= 32; MoE expert block scales stored losslessly as 5-bit group offsets in the streamed weight copies (SC5); WGS MoE tail: per-launch flags in two copies picked by a WGS generation, per-route sums of squares from the phase-B epilogue, one-pass final pass, optional per-token expert counters instead of the grid barrier and a warming pass (v150); T <= 8 boundary 1 by deferred normalization (o epilogue partial sums, 2T-row pooler, sliced boundary 2) (v161); T = 128 / 256 cuBLASLt plan pins, T > 32 row boundaries and two-program route (v201, stacked as v210); 9 <= T <= 16 cluster boundaries (v211); GLE route top 8 by tl.topk on packed keys, bf16_proj picks for every T <= 8 (v164); fused pooler + router: the router inside the pooler's stages, 2 full-K hi + lo rows per CTA, no cross-CTA reduction, also at T 17..32 (v360, v361); MoE head: generation reads and flag-copy zeroing on the weights producer warp, warp-parallel stage ranges, coalesced 8-row-tile x gather, mid-build FAST1 counts with the topk loads first (v300..v302); WGS MoE cost tables A [112, 131, 136, 153, 166, 178, 192, 202] B [100, 111, 123, 135, 147, 159, 172, 189] (v324_half), MoE extension names over 200 chars hashed; per-T WGS cost tables [(64, 'steep'), (191, 'half'), (100000, 'fit')] + v368 with the trigger in a separate -DSK2_ET build used by qkv T 9..32 only (v394) (v394_et); attention q_len 1 split merge by one bulk async reduction (v450_bred); MoE TINY / TINY2 / MID builds: four helper warps, one per SM sub-partition, each summing and storing a quarter of every unit's epilogue with lean per-unit bookkeeping; in the MID build the ring refills issued by the helper warps instead of thread 0 (v444_mhsp2tm); the GLE chain's o projection (T <= 8) as a one-shot register GEMV: weights decoded and escape-patched in registers before griddepcontrol.wait, x fragments from global after it (v460_gos); the cluster boundaries' branch x read by a coherent asm load after griddepcontrol.wait (ptxas had hoisted its ld.global.nc above the wait) (v480_cpbx)); portions from alygg77/okc-muse-spark-kernels (Apache-2.0), whose cuBLASLt plan pins and per-thread CUDA-context bind helper follow code posted by Matsvei (kernarc user 'matvey') on agents.kernarc.com (Apache-2.0); this file is distributed under Apache-2.0.
# Portions (the small-T deferred boundary-1 normalization design and its o-projection epilogue structure) follow
# alygg77/okc-muse-spark-kernels (https://github.com/alygg77/okc-muse-spark-kernels), licensed under the Apache
# License 2.0; that attribution is retained here.
"""Muse Spark decode layer: one CUDA-graph-capturable kernel_fn.

Pipeline (one decode step, T = batch * q_len tokens)
-----------------------------------------------------
1.  qkv projection (cuBLAS, the bf16_proj kernel, or a Triton split-K GEMM --
    prepare() times the three and pins the winner per (T, matrix)).
2.  Attention: the muse_attention_decode entry's fused kernel (QK-norm,
    RoPE, FP8 cache write, split-KV FP8 attention, merge, per-head
    RMSNorm * sigmoid(gate)) for the shapes it supports; otherwise the glue's
    own QK-prep Triton kernel plus SGLang's FA3 (or a Triton split-KV
    attention fallback when SGLang is unavailable).
3.  o projection, then the first residual boundary (residual add, residual
    gate, pre-FFN norm) -- one-program-per-token kernels or parallel
    atomic-sum kernels, per T, timed in prepare().
4.  Pooler + router (bf16_proj's fused launch, its two separate launches, or
    the Triton split-K GEMM), routing (top-8 of biased sigmoid scores), and
    the pre-expert norm.
5.  Experts: the muse_moe_w4a16 entry's NVFP4 kernel (three compile-time
    builds by token count), or SGLang's Marlin NVFP4 path, whichever
    prepare() timed faster per T.
6.  Unpooler, then the second residual boundary (post-FFN norm, next layer's
    pre-attention norm).

Provenance: the attention, MoE and dense kernels are our own team's entries
on this challenge's component boards, copied (not imported). The CUDA sources
(ATT7_CUDA_SRC, MOE_CUDA_SRC, the bfp_* template strings) and the bfp_*
generators are kept verbatim -- they are the tuned, proven kernels. The glue
(boundaries, routing, tuning, launchers) is this file's own code.

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding
with exact escape correction); NVFP4 experts stay NVFP4; norms apply
(1 + w) formed in FP32 at use; FP8 queries / FP8 KV cache as SGLang;
softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too;
FP32 accumulation throughout; activations BF16 wherever SGLang keeps BF16.

Selection: prepare() times each backend per (token count, matrix) and the
expert kernels per token count, on the judge's own GPU with synthetic
activations (it reads only weights, never data), then fixes the choices;
untuned token counts take the next tuned bucket's.

OKC rules: one Python file; native CUDA only via
torch.utils.cpp_extension.load_inline; CUDA-graph capturable (no host syncs
in timed paths -- prepare() and the tuners sync, kernel_fn does not); fresh
inputs each call, no caching across calls (module-level caches hold only
compiled extensions and per-device scratch, never activations).
Safety: every scratch/workspace buffer is followed by sentinel words;
prepare() checks them after the tuner and the attention self-test.
"""

import os
import threading
import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
QKV_ROWS = 2304
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
Q_DIM = NUM_Q_HEADS * HEAD_DIM  # 1024: q / o-projection width
ROPE_THETA = 500000.0
ROUTER_MULT = 0.1767766952966369  # router logit scale (1 / sqrt(32))
LOG2E = 1.4426950408889634

_TUNE_LOG = False
FORCE_PBMAX = None  # override the tuner's pbmax (no consumer; key kept for state shape)
PDL = True
PDL_MAX_T = 128  # a launch-latency win at decode sizes, a ~1% loss at T = 256


def _pdl(T):
    return PDL and T <= PDL_MAX_T


DEFAULT_MOE = "agent"
TUNE_MARGIN = 0.95
PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp", (32, "pr"): "bfp",
            (128, "qkv"): True, (256, "qkv"): True}
FORCE_GLUEV2 = None  # None: tuner's choice; True/False: force the parallel boundary/router kernels


ATT7_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdint.h>

namespace {

__device__ __noinline__ void spin_fail() { __trap(); }   // (out of line: keeps the trap off the polling loop's code)
#define SPIN_T0 unsigned _spin_n = 0   /* (an iteration count, not a clock: a timer read per poll slowed the coop barriers ~1%) */
#define SPIN_CHECK do { if (++_spin_n > (1u << 24)) spin_fail(); } while (0)   /* (>= 2^24 global polls: seconds) */
template <int L, bool F>
struct Cfg {
  static constexpr int NT = (L == 1) ? 128 : 256;   // consumer threads per CTA
  // flat kernel on sm_90: one extra producer warp issues every K/V copy (warp specialization), so the
  // consumers never stall in cp.async issue; the stages are handed over with mbarriers
  static constexpr bool WS = F;
  static constexpr int NP = WS ? 128 : 0;   // producer threads
  static constexpr int NTX = NT + NP;                     // threads per CTA
  static constexpr int MINB = 2;   // CTAs per SM the registers must allow
  // setmaxnreg split of the CTA's registers (launch: 65536 / (MINB * NTX), rounded down to 8)
  static constexpr int BASEREGS = (65536 / (2 * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (2 * NTX)) / 8 * 8;
  static constexpr int PREGS = (L == 1) ? 56 : 32;
  static constexpr int CREGS = ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8 > 248
                                   ? 248 : ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8;
  // cp.async pipeline depth (64-key tiles): deep for the long flat runs, shallow (smaller shared memory,
  // more CTAs per SM, all clusters resident) for the per-request grids of sliding windows / one request
  static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;
  // producer passes: NP/8 rows each must tile the 64-row tile exactly (else rows go missing and the consumers wait
  // forever on the stage barrier)
  static_assert(!WS || (NP % 8 == 0 && NP >= 8 && 64 % (NP >= 8 ? NP / 8 : 1) == 0),
                "producer threads must cover a tile in whole passes");
};
constexpr int TILE = 64;
constexpr int STAGES = 4;
constexpr int ROWB = 128;                    // bytes per tile row: 2 KV heads x 64
constexpr int STAGE_BYTES = TILE * ROWB;     // 8 KB per stage for K (same for V)
constexpr int IDX_SLOTS = 2 * STAGES;
constexpr int MAXB = 512;               // flat partition: requests per call
constexpr int MAXSEG = 64;              // flat partition: segments per CTA

struct Params {
  const __nv_bfloat16* qkv;
  const int* seq_lens;
  const int64_t* cache_loc;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const int* page_table;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;
  __nv_bfloat16* out;
  float* ws_acc;
  float* ws_l;
  int* ws_cnt;
  int B, Hq, Hkv, P, max_pages, window, rope, n_splits, flat, nc;
  long long* dbg;      // optional debug progress mark; null on every kernel_fn call
  int cs, ncl;          // cluster mode: cs CTAs per cluster (DSMEM reduction), ncl clusters per request
  int ld_qkv, row_bytes;
  float scale_log2;   // sm_scale * qk_scale_factor * log2(e)
  int coop;             // per-request grids that are all resident: every CTA finalizes a slice of the rows
  int lp;               // log2(page size)
  int lrb;              // log2(row bytes of a cache row: Hkv * 64)
  int off32;            // both caches below 4 GB (32-bit cache offsets); else the 64-bit path
  unsigned ns_magic;    // ceil(2^32 / n_splits) (0: n_splits == 1)
  const uint8_t* pf;    // (layer) L2 prefetch source (the pooler's weight stream), null = off
  unsigned pf_chunk;    // (layer) bytes per CTA, 0 = off
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async16_pred(uint32_t dst, const void* src, int src_bytes, bool pred) {
  // predicated copy (no branch): pred = 0 issues nothing, src_bytes = 0 zero-fills
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %3, 0;\n @q cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n"
               ::"r"(dst), "l"(src), "r"(src_bytes), "r"((int)pred));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---- sm_90 TMA row loads tracked by one mbarrier per pipeline stage ----
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_init_n(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// unbounded wait for a phase (producer / consumer hand-off)
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
  asm volatile("{\n.reg .pred q;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra WAIT_%=;\n}\n"
               ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
// the mbarrier receives one arrival when all of this thread's earlier cp.async copies have landed
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
// plain bulk copy global -> shared (16-byte aligned, multiple of 16 bytes), completion on an mbarrier
__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// TMA bulk reduction: global[dst .. dst + bytes) += shared[src ..] (fp32 add), tracked by the thread's bulk async-group
    __device__ __forceinline__ void bulk_red_add_f32(float* dst, const void* src, int bytes) {
      asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                   ::"l"(dst), "r"(smem_u32(src)), "r"(bytes) : "memory");
    }
    __device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
    __device__ __forceinline__ void bulk_wait_all() {   // every bulk op of this thread complete (writes performed)
      asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
    }
    __device__ __forceinline__ void fence_proxy_async_global() {   // async-proxy global writes ordered before generic ones
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
    }
    // 16 zero bytes into shared memory through the same cp.async path (no global read)
    __device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(dst), "l"(any_src));
}
// barrier over the NT consumer threads only (named barrier 1); __syncthreads when there is no producer
// barrier over one warpgroup (named barriers 3 and 4)
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wgsync(int wg) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
}
template <bool WS, int NT>
__device__ __forceinline__ void csync() {
  if constexpr (WS) asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
  else __syncthreads();
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// two e4m3 (HIGH 16 bits of x) -> f16x2: ptxas folds the half select into F2FP's operand (.H1), no SHF
__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {_, t}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
// two e4m3 (low 16 bits of x) -> f16x2 (low byte -> low half)
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {t, _}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
// 2^lo, 2^hi as a packed f16 pair: the inputs rounded to f16 (|x| <= 8 here: error <= 2^-8), one MUFU op
__device__ __forceinline__ uint32_t ex2h2(float lo, float hi) {
  return pack_h2(ex2(lo), ex2(hi));
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}


__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }

// V B-fragments for 16 keys starting at kb (rows kb+2tig, +1, +8, +9), head hk, dims 8g..8g+7.
__device__ __forceinline__ void load_v_frags(const uint8_t* V, int kb, int hk, int g, int tig, uint32_t* b0,
                                             uint32_t* b1) {
  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;
  const int k0 = kb + 2 * tig;
  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);
  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);
  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);
  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);
  uint32_t x;
  x = prmt(v0.x, v1.x, 0x5140); b0[0] = cvt2(x); b0[1] = cvt2h(x);
  x = prmt(v0.x, v1.x, 0x7362); b0[2] = cvt2(x); b0[3] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x5140); b0[4] = cvt2(x); b0[5] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x7362); b0[6] = cvt2(x); b0[7] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x5140); b1[0] = cvt2(x); b1[1] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x7362); b1[2] = cvt2(x); b1[3] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x5140); b1[4] = cvt2(x); b1[5] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x7362); b1[6] = cvt2(x); b1[7] = cvt2h(x);
}

// the merge / barrier fences as fence.acq_rel.gpu (MEMBAR.ALL.GPU) instead of __threadfence's
// fence.sc.gpu (MEMBAR.SC.GPU + ERRBAR + CCTL.IVALL): the message-passing pattern needs release / acquire only
__device__ __forceinline__ void gpu_fence() {
  __threadfence();
}
__device__ __forceinline__ void red_f32(float* a, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory");
}
__device__ __forceinline__ void red_v4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}


__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// four 8x8 b16 matrices from registers to shared memory (each lane gives one row address: lanes 8m..8m+7 the
// rows of matrix m; thread t holds row t/4, bytes 4(t%4)..4(t%4)+3 of each matrix)
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}


// ---- per-warp phase profile of the main loops (diagnostic builds only: -DOKC_PROF=1) ----
#define PROF_INIT
#define PROF_MARK(k)
#define PROF_TILE
#define PROF_DUMP

// ---- sm_90 warpgroup MMA helpers ----
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
// K-major, 128-byte swizzled shared-memory matrix descriptor (8-row groups 1024 bytes apart)
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
// D[64x64] (+)= A[64x32] (e4m3, registers) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));
}
// D[64x64] (+)= A[64x32] (e4m3, shared memory descriptor) * B[64x32]^T (e4m3, shared memory descriptor)
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
// the same with scale-d = 0 and write-only accumulators: the previous tile's (masked) scores do not feed this
// wgmma, so ptxas sees no non-wgmma definition of its accumulators inside the pipeline (warning C7515, which
// serializes every wgmma of the function)
// K-major operand with 64-byte rows, 64B swizzle (8-row atoms of 512 bytes)
__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
// D[64x72] += A[64x32] (e4m3, registers) * B[72x32]^T (e4m3, K-major shared memory): d = 32 + 4 accumulators
__device__ __forceinline__ void wgmma_m64n72k32_e4m3(float* d, float* e, const uint32_t* a, uint64_t desc) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
      "{%36, %37, %38, %39}, %40, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
// floor(x / d) for 0 <= x < 2^24 and 1 <= d < 2^24 through the float reciprocal (one correction step):
// ~8 instructions instead of the ~20 of an integer division, which matters in once-per-launch (cold) code
__device__ __forceinline__ int fdiv_i(int x, int d) {
  // approximate reciprocal (MUFU.RCP, no IEEE slow-path call: the correction below absorbs its error)
  const float rd = __fdividef(1.f, __int2float_rn(d));
  int q = __float2int_rz(__int2float_rn(x) * rd);
  const int r = x - q * d;
  q += (r >= d) - (r < 0);
  return q;
}
// sigmoid for the output gate: 0.5 + 0.5 tanh(x / 2) with the hardware tanh (3 instructions; abs error ~2^-12,
// below the BF16 output's rounding) or exp + fast reciprocal
__device__ __forceinline__ float sigm(float x) {
  return __fdividef(1.f, 1.f + __expf(-x));
}
// Programmatic dependent launch (sm_90): wait for the preceding grid (the one writing qkv) before the first qkv read;
// let the next grid start its own prologue once the main loop is done. Both are no-ops for a plain launch.
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_launch_dependents() {
  // (no early trigger at the end of the tile loop -- the dependent kernel launches when this grid
  // exits; for decode_layer's chain A/B, no effect on standalone launches)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
// 32-bit offsets for qkv, the workspace and the caches (the host checks the sizes): no 64-bit multiplies
// in the once-per-launch address math
typedef int ix_t;
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);
  return base + row * p.row_bytes + col_bytes;
}
// `for (e = tid; e < N; e += NTH)` with a compile-time trip count: fully unrolled, the tail predicated (the
// once-executed prologue loops otherwise keep 3-4 branches each; a taken branch costs 60-100 cycles cold)
template <int N, int NTH, class F>
__device__ __forceinline__ void tfor(int tid, F&& f) {
#pragma unroll
  for (int e0 = 0; e0 < N; e0 += NTH) {
    const int e = e0 + tid;
    if ((N % NTH) == 0 || e < N) f(e);
  }
}
// four floats -> four e4m3 bytes (x0 in the low byte), saturating
__device__ __forceinline__ uint32_t e4m3x4f(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(x0), "f"(x1));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(x2), "f"(x3));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
#define SCH_MODE 1
#define SCH_K 6.0f

// EROREx attn_exp (2026-11-03): attention P by the bit trick (the e4m3 code of 2^(x - d) from one fma), see gen_exp.py.
#ifndef SCH_K
#define SCH_K 7.5f
#endif
// row offset: a multiple of 1/8 (the bias is exact), SCH_K octaves below the row max (P_max = 2^SCH_K)
__device__ __forceinline__ float sch_q(float m) { return ceilf(m * 8.f) * 0.125f - SCH_K; }
#if SCH_MODE == 1
typedef uint32_t sch_t;
__device__ __forceinline__ uint32_t sch_c8(float ssc) {
  uint32_t h;
  asm("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(h) : "f"(8.f * ssc));
  return h;
}
__device__ __forceinline__ uint32_t sch_b(float d) {   // 1024 + 56 - 8d (an integer: exact in f16 below 2048)
  uint32_t h;
  asm("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(h) : "f"(fmaf(-8.f, d, 1080.f)));
  return h;
}
// two scores -> f16x2 of 1024 + code (code = round(8 (x - d) + 56) >= 0): the low byte of each half is the e4m3 code
__device__ __forceinline__ uint32_t sch_h2(float a, float b, uint32_t c8, uint32_t bias) {
  uint32_t h, r;
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(h) : "f"(a), "f"(b));
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(h), "r"(c8), "r"(bias));
  asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(r), "r"(0x64006400u));
  return r;
}
__device__ __forceinline__ uint32_t sch_max(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
constexpr uint32_t SCH_ZERO = 0x64006400u;
// a code above 120 (P > 256; +inf / NaN halves too): the offset must move
__device__ __forceinline__ bool sch_over(uint32_t mx) { return ((mx & 0xffffu) > 0x6478u) || ((mx >> 16) > 0x6478u); }
#else
typedef float sch_t;
__device__ __forceinline__ float sch_c8(float ssc) { return 8.f * ssc; }
__device__ __forceinline__ float sch_b(float d) { return fmaf(-8.f, d, 12582968.f); }   // 1.5 * 2^23 + 56 - 8d (exact)
// two scores -> two int16 codes (low half first), negative codes -> 0
__device__ __forceinline__ uint32_t sch_h2(float a, float b, float c8, float bias) {
  return __vmaxs2(prmt(__float_as_uint(fmaf(a, c8, bias)), __float_as_uint(fmaf(b, c8, bias)), 0x5410u), 0u);
}
__device__ __forceinline__ uint32_t sch_max(uint32_t a, uint32_t b) { return __vmaxs2(a, b); }
constexpr uint32_t SCH_ZERO = 0u;
__device__ __forceinline__ bool sch_over(uint32_t mx) { return ((mx + 0x00070007u) & 0xff80ff80u) != 0u; }
#endif
// four f16 (two f16x2 words, low first) -> four e4m3 bytes (low byte first)
__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t h01, uint32_t h23) {
  __half2_raw a, b;
  a.x = (unsigned short)(h01 & 0xffffu); a.y = (unsigned short)(h01 >> 16);
  b.x = (unsigned short)(h23 & 0xffffu); b.y = (unsigned short)(h23 >> 16);
  const uint32_t lo = __nv_cvt_halfraw2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_halfraw2_to_fp8x2(b, __NV_SATFINITE, __NV_E4M3);
  return (lo & 0xffffu) | (hi << 16);
}


__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float oct_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}

// ---- thread-block clusters (sm_90: DSMEM; elsewhere emulated through global memory for testing) ----
__device__ __forceinline__ int cl_rank(const Params& p) {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return (int)r;
}
__device__ __forceinline__ int cl_gid(const Params& p) {
  return (blockIdx.z * gridDim.y + blockIdx.y) * (gridDim.x / p.cs) + blockIdx.x / p.cs;
}
__device__ __forceinline__ void cl_sync(const Params& p, int phase) {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// float4 at byte offset `off` of peer `rank`'s partial buffer (`local` = this CTA's copy)
__device__ __forceinline__ float4 cl_ld4(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(r));
  return v;
}
__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r));
  return v;
}

template <bool PW>
    struct WSR {   // workspace row layout: PW = padded rows (the q_len 1 per-request kernels that are not one-tile-per-split builds)
      static constexpr int STR = PW ? 72 : 64;   // row stride (floats); padded: the row sum in column 64 (else the separate ws_l vector)
    };
    constexpr int WSL = 64;                      // padded rows: the row sum's column
    // the workspace row sum of row `row` (padded rows: column 64 of the row; else the separate ws_l vector)
    #define WSL_AT(PW, row) ((PW) ? (p.ws_acc + (long long)(row) * 72 + WSL) : (p.ws_l + row))

    template <int L, bool F>
    struct Smem {
      uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint64_t mbar[8];   // TMA mode: one per stage (64 bytes keeps the members after it 16-byte aligned)
  uint64_t pbar[4];   // producer mode: page-index slots landed (TMA)
  int pgc;            // page-index fills issued in this CTA's earlier segments
  int dbgn;           // debug builds: progress-mark count
  int vidx;           // flat weighted partition: this CTA's slot (2 * SM + arrival rank), -1 = uniform
  alignas(128) int pidx[4][80];   // producer mode: page indices of upcoming tiles (16-byte aligned window)
  uint64_t full[8];   // producer mode: stage landed (32 producer lanes arrive as their copies complete)
  uint64_t empty[8];  // producer mode: stage consumed (one arrival per consumer warp)
  uint64_t vtf[2];    // V^T slot written by the producers (one arrival per producer thread)
  uint64_t vte[2];    // V^T slot read by both heads' PV wgmmas (one arrival per consumer warp)
  uint64_t qdone;     // consumers have read the q staging (S.kvq) and staged S.kn / S.vn
  int claim[2];       // Cfg::F2: next unclaimed tile per KV head
  int nxt[4][2];      // Cfg::F2: each warpgroup's next claimed tile (double buffered)
  int idx[2 * Cfg<L, F>::STAGES * TILE];
  // raw q rows of the block (bf16, prologue only) | L==8: the current tile's K and V in f16 (2 x 2 x 64 x 64)
  // L == 8: q staging (producer mode) before the loop, then the per-tile f16 V copy -- or, FP8 PV, per tile
  // (double buffer) and KV head, V^T as 72 rows (64 dims + 8 rows of ones for the row sums) x 64 keys (permuted to
  // the P register order), K-major with 64B swizzle: 4 x 4608 bytes
  alignas(1024) uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : 18432];
  // L == 8 with FP8 PV and the block's e4m3 queries per KV head (64 rows x 64 dims, K-major, 64B swizzle)
  alignas(1024) uint8_t q8s[(L == 8) ? 8192 : 16];
  __device__ __forceinline__ __nv_bfloat16* qst() { return reinterpret_cast<__nv_bfloat16*>(kvq); }
  __nv_bfloat16 kvst[L * 4 * 64];   // raw k (2 heads) then v (2 heads) rows per token
  __nv_bfloat16 gst[L * 4 * 64];    // cluster mode: gate rows of this CTA's output slice (<= L*16/4 rows)
  uint8_t kn[L * ROWB];        // new keys of the block, [j][2 heads x 64] e4m3
  uint8_t vn[L * ROWB];
  float cs[L * 32];
  float sn[L * 32];
  float ifreq[32];          // RoPE inverse frequencies (staged with cp.async)
  float scl[4];             // k_scale, v_scale
  long long cloc[L];        // cache rows of the block's tokens
  int flag;
  int sense;          // the cooperative barrier's flag, read early (prologue) with the constants
  int cum[F ? MAXB + 1 : 1];
  int seg[F ? MAXSEG : 1][3];
  int nseg;
  int fcs, fce;             // this CTA's key range (warp 0 -> everyone)
  // flat dynamic chunks : the producer's segment list of its current chunk, and a queue that hands the
  // segments (request, first key, end key, copy; request -1 = done) to the consumers
  int pseg[F ? MAXSEG : 1][3];
  int pnseg, pchunk;
  int dq[F ? 16 : 1][4];
  uint64_t dqf[F ? 16 : 1];
  uint64_t dqe[F ? 16 : 1];
};

// Cluster mode, several clusters per request (Params::ncl > 1): add this CTA's reduced slice into the global
// workspace; the last cluster to arrive sums the copies back into RB/RL and finalizes. Out of line: no scored shape
// uses it, and keeping it out of the one-cluster epilogue keeps that code contiguous (cold instruction fetch).
template <int L, typename SM>
__device__ __noinline__ bool cluster_multi(const Params& p, SM& S, const int CS, const int rank, const int RPC,
                                           const int row0, const int bp, float* RB, float* RL) {
  constexpr int NT = Cfg<L, false>::NT;
  const int tid = threadIdx.x;
  bool fin = true;
    const int clu = fdiv_i(blockIdx.x, CS);
    const int copy = clu - fdiv_i(clu, p.nc) * p.nc;
    for (int e = tid; e < RPC * 16; e += NT) {
      const int row = row0 + (e >> 4);
      float* acc = p.ws_acc + ((ix_t)(bp * p.nc + copy) * L * 16 + row) * WSR<false>::STR + (e & 15) * 4;
      const float4 a = *reinterpret_cast<const float4*>(&RB[e * 4]);
      red_v4(acc, a.x, a.y, a.z, a.w);
    }
    for (int e = tid; e < RPC; e += NT) red_f32(WSL_AT(false, (bp * p.nc + copy) * L * 16 + row0 + e), RL[e]);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      // every cluster of the request arrives (empty ones with zero partials)
      const int old = atomicAdd(p.ws_cnt + bp * CS + rank, 1);
      S.flag = (old == p.ncl - 1);
    }
    __syncthreads();
    fin = S.flag;
    if (fin) {
      __threadfence();
      for (int e = tid; e < RPC * 16; e += NT) {
        const int row = row0 + (e >> 4);
        constexpr int NCM = 4;
        float4 v[NCM];
#pragma unroll
        for (int c = 0; c < NCM; ++c)
          if (c < p.nc) v[c] = __ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * WSR<false>::STR + (e & 15) * 4));
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int c = 0; c < NCM; ++c) {
          if (c < p.nc) {
            a.x += v[c].x; a.y += v[c].y; a.z += v[c].z; a.w += v[c].w;
            *reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * WSR<false>::STR + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        *reinterpret_cast<float4*>(&RB[e * 4]) = a;
      }
      for (int e = tid; e < RPC; e += NT) {
        float a = 0.f;
        for (int c = 0; c < p.nc; ++c) {
          float* lw = WSL_AT(false, (bp * p.nc + c) * L * 16 + row0 + e);
          a += __ldcg(lw);
          *lw = 0.f;
        }
        RL[e] = a;
      }
      if (tid == 0) p.ws_cnt[bp * CS + rank] = 0;
      __syncthreads();
    }
  return fin;
}

// Cluster mode epilogue. Every CTA of the cluster has written its rescaled partial (PB: [L*16][64] o * 2^m,
// PL: [L*16] l * 2^m) into its own shared memory. CTA `rank` sums rows [rank*RPC, (rank+1)*RPC) over the
// cluster through DSMEM, then either finalizes them (one cluster per request) or accumulates them into the
// global workspace, where the last cluster to arrive for this slice finalizes.
template <int L, typename SM>
__device__ __forceinline__ void cluster_finish(const Params& p, SM& S, const int b, const int pair,
                                               const int n_keys, const int chunk) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int tid = threadIdx.x;
  const int CS = p.cs, rank = cl_rank(p);
  const int RPC = fdiv_i(L * 16, CS);
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  // S.k and S.v are adjacent: PB [L*16*64] | PL [L*16, padded to 1024] | RB [RPC*64] | RL [RPC]
  float* PB = reinterpret_cast<float*>(S.k);
  float* PL = PB + L * 16 * 64;
  float* RB = PL + 1024;
  float* RL = RB + RPC * 64;
  constexpr int REGION = L * 16 * 64 + 1024;   // emulation: per-CTA scratch floats (PB, then PL at +L*16*64)
  cl_sync(p, 1);
  
  const int row0 = rank * RPC;
  if constexpr (L == 1) {   // compact reduction (this code runs once per launch from a cold instruction cache, so it is kept short):
      // (row, slot) items -- 16 float4 slots of o and one for l -- times the CS peers; CS consecutive lanes load
      // one item from every peer and sum it with shuffles. Two items per thread per pass keep loads in flight.
    const int TOT0 = RPC * 17 * CS;
    const int lcs = 31 - __clz(CS);
#pragma unroll 1
    int TOT = TOT0;
       // dry pass: same code, no work
    
    for (int k0 = 0; k0 < TOT; k0 += 2 * NT) {   // warp-uniform trip count
      float4 v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k < TOT) {
          if (slot < 16) v[u] = cl_ld4(p, PB, q, (row0 + rl) * 64 + slot * 4, REGION);
          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);
        }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        for (int off = 1; off < CS; off <<= 1) {
          v[u].x += __shfl_xor_sync(0xffffffffu, v[u].x, off);
          v[u].y += __shfl_xor_sync(0xffffffffu, v[u].y, off);
          v[u].z += __shfl_xor_sync(0xffffffffu, v[u].z, off);
          v[u].w += __shfl_xor_sync(0xffffffffu, v[u].w, off);
        }
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        if (k < TOT && q == 0) {
          if (slot < 16) *reinterpret_cast<float4*>(&RB[(rl * 16 + slot) * 4]) = v[u];
          else RL[rl] = v[u].x;
        }
      }
    }
      } else
  {
  // compact: peers in groups of 4 (one loop body, warm after its first pass: this code runs once per launch)
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll 1
    for (int q0 = 0; q0 < CS; q0 += 4) {
      float4 v[4];
      float lv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        v[u] = cl_ld4(p, PB, q0 + u, off, REGION);
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) { a.x += v[u].x; a.y += v[u].y; a.z += v[u].z; a.w += v[u].w; la += lv[u]; }
    }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  
  }
  
  cl_sync(p, 2);   // peers are done reading this CTA's partial

  bool fin = true;
  if (p.ncl > 1) {
    fin = cluster_multi<L>(p, S, CS, rank, RPC, row0, bp, RB, RL);
  }
  if (fin) {
    // rmsnorm * sigmoid(gate) for the slice: 16 threads per row, 4 dims each
    
    const float vscale = S.scl[1];
    for (int eb = tid - (tid & 31); eb < RPC * 16; eb += NT) {   // warp-uniform trip count (shuffles below)
      const int e = eb + (tid & 31);
      const bool act = e < RPC * 16;
      const int rl = act ? (e >> 4) : 0, c4 = e & 15;
      const int row = row0 + rl;
      const int j = row >> 4, hh = row & 15;
      const float lsum = act ? RL[rl] : 0.f;
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      const float4 a = act ? *reinterpret_cast<const float4*>(&RB[e * 4]) : make_float4(0.f, 0.f, 0.f, 0.f);
      float y[4] = {(a.x * inv), (a.y * inv), (a.z * inv), (a.w * inv)};
      float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 8);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const uint2 graw = *reinterpret_cast<const uint2*>(&S.gst[rl * 64 + c4 * 4]);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint2 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float2 gt = __bfloat1622float2(g2[k]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * k] * rn * s0, y[2 * k + 1] * rn * s1);
        ow[k] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint2*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + c4 * 4) = ov;
    }
    
  }
}

// Gate rows of this CTA's cluster slice -> S.gst (cp.async, in the caller's commit group)
template <int L, typename SM>
__device__ __forceinline__ void cluster_stage_gate(const Params& p, SM& S, const int b, const int pair) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int RPC = fdiv_i(L * 16, p.cs), row0 = cl_rank(p) * RPC;
  for (int e = threadIdx.x; e < RPC * 8; e += NT) {
    const int rl = e >> 3, ch = e & 7;
    const int row = row0 + rl, j = row >> 4, hh = row & 15;
    cp_async16(smem_u32(&S.gst[rl * 64 + ch * 8]),
               p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + ch * 8, 16);
  }
}

// One (request b, KV pair) segment [start, end) of the request's key positions.
// Sum the nc workspace copies of one (row, 8-dim part) of request-pair bp, token j, head hh, and zero them. Copies in
// groups of 4 (loads of a group in flight together): one compact loop body, since this runs once per launch from a
// cold instruction cache (an 8-way unrolled version cost ~2x the instructions).
template <int L, bool PW>
__device__ __forceinline__ void sum_copies(const Params& p, const int bp, const int j, const int hh, const int part,
                                           const bool act, float (&y)[8], float& lsum) {
  if (p.nc == 1) {   // one copy (the common case): no copy loop, no per-copy branches (scout's census)
    if (act) {
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));
      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      const float lc = __ldcg(WSL_AT(PW, (bp * L + j) * 16 + hh));
      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;
      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;
      lsum += lc;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0) *WSL_AT(PW, (bp * L + j) * 16 + hh) = 0.f;
    }
    return;
  }
  if (p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy's loads in one round trip
    if (act) {                   // (the rolled 4-copy loop below took two dependent L2 round trips)
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(WSL_AT(PW, ((bp * 8 + c) * L + j) * 16 + hh));
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) *WSL_AT(PW, ((bp * 8 + c) * L + j) * 16 + hh) = 0.f;
      }
    }
    return;
  }
#pragma unroll 1
  for (int c0 = 0; c0 < p.nc; c0 += 4) {
    float4 xa[4], xb[4];
    float lc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      xa[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      xb[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      lc[u] = 0.f;
      if (act && c < p.nc) {
        const float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
        xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[u] = __ldcg(WSL_AT(PW, ((bp * p.nc + c) * L + j) * 16 + hh));
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      if (act && c < p.nc) {
        y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;
        y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;
        lsum += lc[u];
        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) *WSL_AT(PW, ((bp * p.nc + c) * L + j) * 16 + hh) = 0.f;
      }
    }
  }
}

// Last-CTA finalize of a request with one workspace copy: sum, normalize, rmsnorm * sigmoid(gate), store,
// reset the workspace. Not inlined: this runs once per launch and is otherwise fetched cold.
template <int L, bool PW, int NT>
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
  // loads of a group of passes first (one L2 round trip per group), then math and stores; groups of 2 passes in a
  // rolled loop (this runs once per launch from a cold instruction cache: q_len 8 has 4 passes)
  constexpr int PASSES = L * 16 / (NT / 8);
  constexpr int GP = PASSES >= 2 ? 2 : 1;
  const int part = tid & 7;
#pragma unroll 1
  for (int p0 = 0; p0 < PASSES; p0 += GP) {
    float4 xa[GP], xb[GP];
    float lc[GP];
    uint4 graw[GP];
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      lc[u] = __ldcg(WSL_AT(PW, (bp * L + j) * 16 + hh));
      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
    }
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * WSR<PW>::STR + part * 8;
      if (!dry) {
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) *WSL_AT(PW, (bp * L + j) * 16 + hh) = 0.f;
      }
      const float inv = lc[u] > 0.f ? __fdividef(vscale, lc[u]) : 0.f;
      float y[8] = {(xa[u].x * inv), (xa[u].y * inv), (xa[u].z * inv), (xa[u].w * inv),
                    (xb[u].x * inv), (xb[u].y * inv), (xb[u].z * inv), (xb[u].w * inv)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (!dry) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
}

// the q_len 8 flat loop's rare new-row patch as a call, so ptxas cannot hoist its bounds math into the
// common path of every tile
template <int L, int NT, bool PTR, class SM>
__device__ __noinline__ bool ws_patch_ni(SM* Sp, int i, int st, int start, int seq, int end, int tid) {
  SM& S = *Sp;
  const int t0 = start + i * TILE;
  const int lo = max(t0, seq), hi = min(t0 + TILE, end);
  if (lo < hi) {
    for (int e = tid; e < (hi - lo) * 8; e += NT) {
      const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
      const int sw = ((c ^ (r & 7)) << 4);
      *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
      *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
    }
    if (L == 8) fence_proxy_async();
  }
  return lo < hi;
}

template <int L, bool FLAT, int ROLE, int MODE = 0>
__device__ __forceinline__ bool segment(const Params& p, Smem<L, FLAT>& S, const int b_, const int pair_, const bool first,
                                        const int start_, const int end_, const int seq_, const int n_keys_,
                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,
                                        const bool skip_init_ = false) {
  // (returns false only in the builds, when the speculated length missed: nothing global was written yet and
  // the caller re-runs the segment with the real length)
  // per-segment values as lane-0 broadcasts : provably warp-uniform, so ptxas keeps the address math
  // derived from them in uniform registers (they come from shared memory or per-CTA math, uniform but unproven)
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)
  const int b = UNI(b_), pair = UNI(pair_), start = UNI(start_), end = UNI(end_), seq = UNI(seq_);
  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);
#undef UNI
  // tb: tiles this CTA's pipeline has already run (flat producer mode: the stage ring and its mbarrier phases
  // continue across segments, so the producer runs ahead into the next segment instead of draining)
  // MODE: the per-request (non-flat) kernel is built once per launch mode, so each build carries only its own
  // epilogue (once-per-launch code runs from a cold instruction cache): 1 plain split-KV, 2 clusters,
  // 3 cooperative; 0 = decided at run time (flat kernels)
  // MODE | 4 = a build for grids whose every split holds at most one tile (the host checks): the tile loop,
  // the in-loop copies and the first-tile tests fold at compile time (scout's once-path census, s1 / vs1)
  constexpr bool ONE = !FLAT && (MODE & 4) != 0;
  constexpr bool PADWS = L == 1 && !FLAT;   // padded workspace rows + bulk-reduction merge (EROREx a8: + one-tile builds)
  const bool M_CL = !FLAT && ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2);
  const bool M_COOP = (MODE & 3) == 0 ? p.coop != 0 : (MODE & 3) == 3;
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  // flat q_len 8 with FP8 PV: the producer warps (idle most of the time) build each tile's V^T for the PV wgmma
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool WS = Cfg<L, FLAT>::WS;
  // (the warp index as a lane-0 broadcast, so ptxas can prove it warp-uniform and keep the addresses
  // derived from it (KV head, descriptors, buffers) in uniform registers; CUTLASS's canonical_warp_idx_sync idiom; okc-license: own)
  const int tid = threadIdx.x, lane = tid & 31, warp = __shfl_sync(0xffffffffu, tid >> 5, 0);
  const int g = lane >> 2, tig = lane & 3;
  const int split = first ? 0 : 1;
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;   // (ONE: a CTA without keys returned earlier)

  if constexpr (WS) {
    constexpr int NP = Cfg<L, FLAT>::NP;
    constexpr int PR = NP / 8;            // rows per producer pass
    constexpr int PK = TILE / PR;         // passes per tile
    if (!skip_init_) {   // (the first segment's barriers were set up at kernel start)
      if (ROLE == 2 && tid == 0) {   // fresh phases for this segment (the previous one ended with every stage consumed)
        for (int st = 0; st < STAGES; ++st) {
          mbar_init_n(&S.full[st], NP);
          mbar_init_n(&S.empty[st], NT);   // F2: one warpgroup per head takes each tile
        }
        
        mbar_fence_init();
      }
      __syncthreads();
    }
    if constexpr (ROLE == 1) {
      // ---- producer threads: 16-byte chunk (q & 7) of rows (q >> 3) + PR*k; the cache rows of every tile,
      // zeros past the range; the block's own new keys are patched in by the consumers
      const int q = tid - NT;
      const int ch = q & 7, r0 = q >> 3;
      const int* ptr = p.page_table + (ix_t)b * p.max_pages;
      const int cache_end = min(end, seq);
      int pg[PK];
      static_assert(PR % 8 == 0, "PFAST: the swizzle chunk is the same for all of a thread's rows");
      const bool pfast = p.off32 && p.lp == 0;
      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);
      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
      // page indices by TMA (q_len 1 flat only: f16 -2.4%, f32 -1.7%; the q_len 8 flat kernel lost ~0.7%)
      constexpr bool TMAPG_ON = (L == 1);
      // page indices by TMA: one 64-entry box per tile, PD tiles ahead, PS slots, producer barrier per tile
      constexpr int PS = 4, PD = 3;
      int pgb = 0;   // fills issued by earlier segments (slot / phase base)
      if constexpr (TMAPG_ON) {
          
          if (q == 0) {
            for (int st = 0; st < PS; ++st) mbar_init_n(&S.pbar[st], 1);
            mbar_fence_init();
          }
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
        
      }
      // the 16-byte aligned window holding the tile's page indices; reads stay inside the table's allocation
      // (CUDA allocations are 256-byte aligned and sized in larger multiples than 16 bytes)
      const long long pt_total = (long long)p.B * p.max_pages;
      const long long pt_lim = (pt_total + 3) & -4;
      auto fetch_pages = [&](int x) {
        if (q == 0 && x < ntiles) {
          const int slot = (pgb + x) % PS;
          const long long e0 = (long long)b * p.max_pages + ((start + x * TILE) >> p.lp);
          const long long w0 = e0 & -4;
          const long long w1 = min(pt_lim, (e0 + 64 + 3) & -4);
          const int bytes = (int)(w1 - w0) * 4;
          mbar_expect_tx(&S.pbar[slot], bytes);
          bulk_load(&S.pidx[slot][0], p.page_table + w0, bytes, &S.pbar[slot]);
        }
      };
      if constexpr (TMAPG_ON) {
        for (int x = 0; x < PD; ++x) fetch_pages(x);
      }
      auto load_pages_t = [&](int x, int* dst) {
        const int slot = (pgb + x) % PS;
        mbar_wait_full(&S.pbar[slot], ((pgb + x) / PS) & 1);
        const int fp = (start + x * TILE) >> p.lp;
        const int dl = (int)(((long long)b * p.max_pages + fp) & 3);   // window offset of the first page
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (pos < cache_end) ? S.pidx[slot][(pos >> p.lp) - fp + dl] : 0;
        }
      };
      auto load_pages = [&](int x, int* dst) {
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (x < ntiles && pos < cache_end) ? __ldg(ptr + (pos >> p.lp)) : 0;
        }
      };
      if constexpr (!TMAPG_ON) load_pages(0, pg);
      
      // tile y's V (both KV heads) -> V^T slot y & 1 (72 x 64 per head: 64 dims + 8 rows of e4m3 ones for the
      // row sums, keys permuted to the P register order, K-major, 64B swizzle), the layout the PV wgmma reads
      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];
        if constexpr (TMAPG_ON) {
          fetch_pages(x + PD);
          load_pages_t(x, pg);
        } else {
          load_pages(x + 1, pgn);   // in flight while this tile's copies issue
        }
        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
        const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
        // common tiles (every row cached, page size 1, caches below 4 GB): one shift-add per row, no per-row
        // branches (the producer's instructions take issue slots from the consumers on the same SMSPs)
        if (pfast && start + x * TILE + TILE <= cache_end) {
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const uint32_t off = ((uint32_t)pg[k] << p.lrb) + pcol;
            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
          }
        } else
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int r = r0 + PR * k;
          const int pos = start + x * TILE + r;
          const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
          if (pos < cache_end) {
            if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)
              const uint32_t off = ((((uint32_t)pg[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            } else
            {
              const int64_t off = (((int64_t)pg[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }
          } else if (pos >= end) {
            cp_async16_zero(kb + so, p.k_cache);
            cp_async16_zero(vb + so, p.v_cache);
          }
        }
        cp_async_mbar_arrive(&S.full[st]);
        
        PROF_TILE;
        if constexpr (TMAPG_ON) {
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");   // slot x % PS is free for tile x + PS
        } else {
#pragma unroll
          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];
        }
      }
      
      
      PROF_DUMP;
      
        cp_wait<0>();
        __syncthreads();   // matches the consumers' end-of-segment barrier
      
      return true;   // okc-pdl: in the layer this grid is a PDL launch (T <= 128); this exit reads no qkv, and the CTAs that read qkv / write the output wait first
    }
  }
  if constexpr (ROLE != 1) {

  const int prow = tid >> 1, phalf = tid & 1;
  const int* pt_row = p.page_table + (ix_t)b * p.max_pages;
  const int part = tid & 7;

  // ---- producer helpers ----
  // K/V: thread -> 16-byte chunk (tid & 7) of rows (tid >> 3) + 16*k: a warp copies 4 whole 128-byte rows.
  // New-token rows come from S.kn / S.vn; rows past the range are zeroed.
  const int cchunk = tid & 7, crow0 = tid >> 3;
  constexpr int RPP = NT / 8;   // rows per pass
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % STAGES;
    const int* idxs = &S.idx[(x % IDX_SLOTS) * TILE];
    bool generic = false;   // st.shared into a K stage: the wgmma (async proxy) needs a proxy fence
    // every row of the tile cached (pos < min(end, seq)), page size 1, caches below 4 GB: one shift-add
    // per row, no per-row branches (the per-request issue runs from a cold instruction cache, mostly once)
    if (with_new && p.off32 && p.lp == 0 && start + x * TILE + TILE <= min(end, seq)) {   // (in-loop issues only: the
      // prologue's cold copy of the tile-0 issue stays the plain one, s1 measured +1.5% with it)
      const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES), vst = smem_u32(S.v + st * STAGE_BYTES);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        const uint32_t off = ((uint32_t)idxs[r] << p.lrb) + col;
        cp_async16(kst + so, p.k_cache + off, 16);
        cp_async16(vst + so, p.v_cache + off, 16);
      }
      return;
    }
#pragma unroll
    for (int k = 0; k < TILE / RPP; ++k) {
      const int r = crow0 + RPP * k;
      const int pos = start + x * TILE + r;
      const int sw = ((cchunk ^ (r & 7)) << 4);
      uint8_t* dk = S.k + st * STAGE_BYTES + r * ROWB + sw;
      uint8_t* dv = S.v + st * STAGE_BYTES + r * ROWB + sw;
      if (pos < end && pos < seq) {
        if (p.off32) {   // caches below 4 GB: 32-bit offsets (uniform branch)
          const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + cchunk * 16);
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        } else
        {
          const int64_t row = ((int64_t)idxs[r] << p.lp) + (pos & (p.P - 1));
          const int64_t off = row * p.row_bytes + pair * 128 + cchunk * 16;
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        }
      } else if (pos < end) {
        if (with_new) {
          *reinterpret_cast<uint4*>(dk) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(dv) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
          generic = true;
        }
      } else {   // past the range: masked out of S; V must be finite (P = 0 times V)
        *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dv) = make_uint4(0, 0, 0, 0);
      }
    }
    if (L == 8 && generic) fence_proxy_async();
  };
  auto issue_idx = [&](int y) {
    if (phalf == 0 && prow < TILE && y < ntiles) {
      const int pos = start + y * TILE + prow;
      if (pos < end && pos < seq) cp_async4(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), pt_row + (pos >> p.lp));
    }
  };
  // producer mode: wait for stage `st` (tile i) and write the block's own new keys / values into it
  auto ws_patch = [&](int i, int st) {   // the block's own new rows (positions >= seq) into tile i's stage
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {   // same for every consumer thread
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };
  auto ws_acquire = [&](int i, int st) {
    mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {   // same for every consumer thread
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };

  // ---- 1. page indices of the first STAGES-1 tiles; raw q / k / v rows into registers ----
  int tidL = tid, partL = part;   // (re-read after the page-index copies are issued)
  constexpr int QPASS = L * 16 / RPP;            // L*16 rows, RPP rows per pass
  const bool has_new = end > seq;
  constexpr int KPASS = (L * 4 + RPP - 1) / RPP; // L*4 rows (k, v of 2 heads), RPP rows per pass
  constexpr bool QDIR = L == 8 && !WS;
  constexpr bool IDXF = !WS;
  {
  if constexpr (!WS) {
#pragma unroll
    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);
  }
  cp_commit();
  if (!WS && (MODE & 8)) {   // (speculative builds only) a late copy of tid: the q / new-row / table math below cannot be hoisted above the page-index
    // copies (or out of the speculative retry loop to the kernel top: scout's v200 census, segment head 21 -> 62/97)
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tidL));
    partL = tidL & 7;
  }

  pdl_wait();   // qkv (q, new k / v, gate) is the preceding kernel's output; nothing above reads or writes it
  pdl_launch_dependents();   // (layer v480) early trigger: the o projection launches and prefetches its weights
  // raw q rows: L*16 rows x 128 B = 8 chunks of 16 B per row
  tfor<L * 16 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
    const int rr = e >> 3, ch = e & 7;
    const int j = rr >> 4, hh = rr & 15;
    cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);
  });
  if (has_new) {
    tfor<L * 4 * 8, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const int rr = e >> 3, ch = e & 7;
      const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + ch * 8;
      cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);
    });
  }
  if (M_CL) cluster_stage_gate<L>(p, S, b, pair);
  // small per-call constants: staged with the same group instead of cold loads in the middle of the prologue
  if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);
  else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);
  else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
  else if (!FLAT && M_COOP && tidL == 10)   // the coop barrier's flag / epoch, early
    cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);
  else if (tidL >= 16 && tidL < 16 + L) {
    const int jj = tidL - 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));
  }
  cp_commit();
  if (split == 0 && (FLAT || !M_CL)) {  // warm L2 with the gate rows for whichever CTA finishes last
    tfor<L * 16, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (e & 15)) * 64;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
    });
  }

  // ---- 2. K/V of the first STAGES-1 tiles in flight as soon as their page indices land ----
  // (per-request kernels): wait for the page-index group only (committed first, 256 B); the q / new
  // k,v / constants group (up to 16 KB) may still be in flight while the K/V copies go out
  cp_wait<IDXF ? 1 : 0>();
  csync<WS, NT>();
  if constexpr (!FLAT && (MODE & 8) != 0) {
    // this CTA's ranges and page indices came from the speculated length (see the kernel); the real
    // cache_seqlens value has landed by now (it was loaded alongside the indices). On a miss, nothing global has been
    // written: drain the copies and let the caller redo the segment with the real length (today's path)
    if (seq_real_ != seq) {
      cp_wait<0>();
      __syncthreads();
      return false;
    }
  }
  if constexpr (!WS) {
    constexpr int PRE = STAGES - 1;   // two streams: one cp.async group per tile, the loop issues pairs
    bool one = false;
    // a CTA with a single fully cached tile (page size 1, caches < 4 GB): its copies straight-line (one
    // shift-add per row), none of the loop / per-row tests / issue_idx past the last tile, then the commits the
    // pipeline's group count expects (scout's census: ~168 instr, 11 taken branches here on s1, cold)
    if (ONE && p.off32 && p.lp == 0) {
      one = true;
      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
      const int cend = min(end, seq);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        
          // branch-free (scout's v200 census: the per-row branches cost 16 instr + 4 taken branches per warp, cold):
          // cached rows copy, rows past the range zero-fill (masked out of S; V must be finite), the block's new
          // rows are skipped (written by the patch pass after the new-k/v pass)
          const int pr = start + r;
          const bool cached = pr < cend;
          const bool doit = cached || pr >= end;
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;   // (zero-filled index past the cached rows)
          cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);
          cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);
            // (the block's new rows: written by the patch pass after the new-k/v pass, as with the general issue)
      }
#pragma unroll
      for (int k = 0; k < STAGES - 1; ++k) cp_commit();
    }
    if (!one) {
#pragma unroll
    for (int k = 0; k < STAGES - 1; ++k) {   // (unrolled: no loop branches; rolled: one copy of the issue code)
      if (k < PRE && k < ntiles) issue_kv(k, false);
      issue_idx(k + STAGES - 1);
      if (k < PRE - 1 || k == STAGES - 2) cp_commit();
    }
    }
  }
  }

  if constexpr (IDXF) {   // the q group (all but the STAGES-1 K/V groups just committed), visible to every thread
    cp_wait<STAGES - 1>();
    csync<WS, NT>();
  }
  // ---- 3. while they load: RoPE table, queries, the block's new keys / values ----
  if (p.rope) {
    tfor<L * 32, NT>(tidL, [&](int e) {   // (straight-line: no loop branches on the once-path)
      const int j = e >> 5, i = e & 31;
      const float ang = (float)(seq + j) * S.ifreq[i];   // fp32 angle, as the reference builds it
      float sv, cv;
      // Cody-Waite reduction mod 2 pi with a 3-part constant (6.25 has 5 significant bits: exact products for
      // |k| < 2^19, angles up to ~3e6), then the hardware sin / cos on [-pi, pi]: ~5e-7 absolute, against
      // ~30 cold instructions (and a Payne-Hanek call above |x| ~ 1e5) for the accurate sincosf
      const float kq = rintf(ang * 0.15915493667125702f);
      float rr = fmaf(-kq, 6.25f, ang);
      rr = fmaf(-kq, 0.03318530693650246f, rr);
      rr = fmaf(-kq, 2.430837753308879e-10f, rr);
      __sincosf(rr, &sv, &cv);
      S.cs[e] = cv;
      S.sn[e] = sv;
    });
  }
  csync<WS, NT>();

  const float kscale = S.scl[0], vscale = S.scl[1];
  const float sl2 = p.scale_log2 * kscale;

  if constexpr (QDIR) {
    // (q_len 8, QDIRECT): four threads per query row, 16 dims each (half the passes of the 8-thread
    // layout, two shuffle levels instead of three, one 16-byte e4m3 store); same math per element
    constexpr int RP4 = NT / 4;                    // rows per pass
    constexpr int QP4 = L * 16 / RP4;
    const int q4 = tidL & 3;
#pragma unroll
    for (int ps = 0; ps < QP4; ++ps) {
      const int rr = ps * RP4 + (tidL >> 2);
      const int j = rr >> 4, hh = rr & 15;
      const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);
      const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);
      const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);
      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);
      float x[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 fa = __bfloat1622float2(ha[e]);
        const float2 fb = __bfloat1622float2(hb[e]);
        x[2 * e] = fa.x; x[2 * e + 1] = fa.y;
        x[8 + 2 * e] = fb.x; x[8 + 2 * e + 1] = fb.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += x[e] * x[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 16; ++e) x[e] *= r;
      if (p.rope) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = q4 * 8 + e;
          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
          const float a0 = x[2 * e], a1 = x[2 * e + 1];
          x[2 * e] = a0 * c - a1 * sn;
          x[2 * e + 1] = a0 * sn + a1 * c;
        }
      }
      const int hk4 = hh >> 3, gg4 = hh & 7;
      const int r4 = j * 8 + gg4, kb = q4 * 16;
      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                     e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
    }
  } else
#pragma unroll
  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    float x[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(h2[e]);
      x[2 * e] = f.x;
      x[2 * e + 1] = f.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
    ss = oct_sum(ss);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] *= r;
    if (p.rope) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = partL * 4 + e;
        const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * sn;
        x[2 * e + 1] = a0 * sn + a1 * c;
      }
    }
    // f16 query pre-scaled to log2 units, staged in the (still free) last pipeline stage
    uint4 qw;
    uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);
    // (q_len 8 on sm_90 rounds the query to e4m3: keep it unit-RMS like SGLang does -- pre-scaling it by the
    // kv scale would push it into e4m3's subnormals -- and scale the scores instead)
    const float qsc = (L == 8) ? 1.f : sl2;
    const int hk = hh >> 3, gg = hh & 7;
    if constexpr (QDIR) {
      // straight to the e4m3 A tile of the SS QK^T wgmma (64-byte rows, 16-byte chunks swizzled by
      // (row >> 1) & 3), one fp32 -> e4m3 rounding, no f16 staging / re-read (scout's census: ~90 instr / warp)
      const int r = j * 8 + gg, kb = partL * 8;
      *reinterpret_cast<uint2*>(S.q8s + hk * 4096 + r * 64 + (((kb >> 4) ^ ((r >> 1) & 3)) << 4) + (kb & 15)) =
          make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));
      continue;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
    if constexpr (L == 1) { const uint32_t t1 = qwp[1]; qwp[1] = qwp[2]; qwp[2] = t1; }   // (scout: quad order 0,2,1,3)
    if constexpr (WS) {   // in place over the raw row (same thread, same 16 bytes)
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }
  }
  if constexpr (QDIR) fence_proxy_async();   // the e4m3 q tile (st.shared) is read by the wgmma after the loop's first barrier

  if (has_new) {
#pragma unroll
    for (int ps = 0; ps < KPASS; ++ps) {
      const int rr = ps * RPP + (tidL >> 3);
      const bool live = rr < L * 4;
      const int kind = rr / (2 * L);
      const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      float x[8];
      const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&S.kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        x[2 * e] = f.x;
        x[2 * e + 1] = f.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
      ss = oct_sum(ss);   // every lane of the warp (rows of both kinds share warps)
      float sc;
      if (kind == 0) {
        const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] *= r;
        if (p.rope && live) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = partL * 4 + e;
            const float c = S.cs[jj * 32 + i], sn = S.sn[jj * 32 + i];
            const float a0 = x[2 * e], a1 = x[2 * e + 1];
            x[2 * e] = a0 * c - a1 * sn;
            x[2 * e + 1] = a0 * sn + a1 * c;
          }
        }
        sc = __fdividef(1.f, kscale);   // (MUFU.RCP: no IEEE slow-path call and its branches on the once-path;
      } else {                          // exact for the power-of-two scales)
        sc = __fdividef(1.f, vscale);
      }
      if (live) {
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
          w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
        }
        uint8_t* dst = (kind == 0 ? S.kn : S.vn) + jj * ROWB + hh * 64 + partL * 8;
        *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);
        const int pos = seq + jj;
        if (pos >= start && pos < end) {
          const int64_t row = S.cloc[jj];
          uint8_t* g8 = const_cast<uint8_t*>(cache_row(kind == 0 ? p.k_cache : p.v_cache, p, row, (2 * pair + hh) * 64 + partL * 8));
          *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);
        }
      }
    }
  }
  csync<WS, NT>();
  if (!WS && has_new) {
    // new-token rows of the tiles already issued (their cached rows are in flight with cp.async)
    for (int x = 0; x < (STAGES - 1) && x < ntiles; ++x) {
      const int st = x % STAGES;
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const int pos = start + x * TILE + r;
        if (pos >= seq && pos < end) {
          const int sw = ((cchunk ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
        }
      }
    }
    if (L == 8) fence_proxy_async();   // new keys written with st.shared, read by the QK wgmma
  }

  
  // ---- consumer state ----
  constexpr int MT = 1;                    // m-tiles per warp
  constexpr int NROW = (L == 1) ? 1 : 2;   // rows per thread
  // (lane-0 broadcasts, so ptxas keeps them in uniform registers instead of re-deriving them every tile)
  const int hk = __shfl_sync(0xffffffffu, (L == 1) ? (warp & 1) : ((warp & 7) >> 2), 0);      // L==8: warpgroup = KV head
  const int wsub = __shfl_sync(0xffffffffu, (L == 1) ? (warp >> 1) : (warp & 3), 0);    // L==1: key half; L==8: m-tile (tokens 2*wsub, 2*wsub+1)
  const int wgi = hk;          // named barrier of this warpgroup
  float o[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[mt][t][e] = 0.f;
  float m_r[NROW], l_r[NROW];
#pragma unroll
  for (int r = 0; r < NROW; ++r) { m_r[r] = -INFINITY; l_r[r] = 0.f; }
  // tiles with no window / causal / range edge skip the per-element mask
  const int lo_all = (L == 1 || !p.window) ? start : max(start, seq + (L - 1) - p.window + 1);
  const int hi_all = __shfl_sync(0xffffffffu, min(end - 1, seq), 0);

  if constexpr (L == 1) {
  // Q A-fragments (f16, pre-scaled to log2 units) for 4 k-steps of 16 dims: thread holds dims
  // 16*tig + 4*ks + {0,1} (a0) and + {2,3} (a2); the K B-fragments use the same dims.
  uint32_t qa[4][2];
  {
    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
    const uint4 l0 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig);
    const uint4 l1 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig + 16);
    // (scout: chunk words stored (0, 2, 1, 3): l0 = {ks0 a0, ks1 a0, ks0 a2, ks1 a2}, l1 the same for ks 2, 3)
    qa[0][0] = l0.x; qa[0][1] = l0.y; qa[1][0] = l0.z; qa[1][1] = l0.w;
    qa[2][0] = l1.x; qa[2][1] = l1.y; qa[3][0] = l1.z; qa[3][1] = l1.w;
  }

  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
    }

    const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;
    const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;
    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
    // warp: kv head hk, keys [32*wsub, 32*wsub+32) of the tile
    float s[4][4];
    uint32_t kw[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {   // (scout: unpadded quads; even k-steps -> rows g of se, odd -> rows g+8 of so)
      float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]), cvt2h(kw[nt][2 * h]));
        mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]), cvt2h(kw[nt][2 * h + 1]));
      }
      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) s[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));
    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      if constexpr (ONE) {   // (scout: the only tile: o = 0 and l = 0, alpha would only multiply zeros)
        l_r[0] = ls;
      } else {
        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
      }
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t b0[8], b1[8];
        load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
        const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
        const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
      }
    }
    if constexpr (WS) {   // this warp is done with the stage
      __syncwarp();
      mbar_arrive(&S.empty[(tb + i) % STAGES]);
    }
    PROF_TILE;
  }
  PROF_DUMP;
  } else {
  // ---- L == 8 on sm_90: QK^T with FP8 wgmma straight from the FP8 K tile in shared memory (warpgroup =
  // KV head, 64 rows = 8 tokens x 8 heads, K-major 128B-swizzled B operand), PV with f16 mma.sync from a
  // per-tile f16 copy of V. Softmax with a lazily updated per-row offset; row sums from a ones column.
  uint8_t* V16 = S.kvq;
  uint32_t qa8[2][4];   // e4m3 A fragments, k-steps of 32 dims
  if constexpr (!QDIR) {
    const uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;   // f16 staging, row = j*8 + g
    const int r0 = 16 * wsub + g, r1 = r0 + 8;
    // producer mode: in place in S.kvq, row j*16 + hk*8 + g
    const uint8_t* q0 = WS ? S.kvq + ((r0 >> 3) * 16 + hk * 8 + (r0 & 7)) * 128 : qs + r0 * 128;
    const uint8_t* q1 = WS ? S.kvq + ((r1 >> 3) * 16 + hk * 8 + (r1 & 7)) * 128 : qs + r1 * 128;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {        // dims 32*ks + 16*hf + 4*tig .. +3
        const int dofs = (32 * ks + 16 * hf + 4 * tig) * 2;
        const uint2 lo = *reinterpret_cast<const uint2*>(q0 + dofs);
        const uint2 hi = *reinterpret_cast<const uint2*>(q1 + dofs);
        qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);
        qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);
      }
    }
       // the A tile for the SS QK^T wgmma
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int kb = 32 * ks + 16 * hf + 4 * tig;
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf];
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf + 1];
        }
      fence_proxy_async();   // read by the wgmma after the loop's first barrier
    
    
  }
  float cr[2] = {0.f, 0.f};   // per-row offsets (log2 units), rows g and g+8
  const float ssc = sl2;   // raw score -> log2 units
  const sch_t c8s = sch_c8(ssc);   // (attn_exp: 8 * ssc for the codes)
  // the lazy-max test on raw scores against (offset + 8) / ssc (ssc > 0), updated with the offset
  float thr[2] = {0.f, 0.f};
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  bool first = true;
     // the zero init stays before the loop: ptxas had sunk these moves into the first QK^T
    float* oz = &o[0][0][0];   // wgmma's pipeline stage (warning C7515: every wgmma of the kernel serialized)
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(oz[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  
  const int ckey = tid >> 2, cq = tid & 3;   // V conversion: key, two 16-byte chunks
  // FP8 PV: the per-CTA kernels (the flat kernel's register budget serializes its wgmmas: measured slower)
  // transpose roles (warpgroup = head): warp wq takes keys 16wq..16wq+15, lane = (q, w) of the ldmatrix fragments
  const int vwq = warp & 3, vq = lane & 3, vw = lane >> 2;   // (warp: uniform)
  const uint32_t vsel0 = (vw >= 4) ? 0x7531u : 0x6420u, vsel1 = (vw >= 4) ? 0x6420u : 0x7531u;
  const int vr0 = (vw >= 4) ? 1 : 0;   // rows d + vr0 then d + 1 - vr0 (bank spread)
  // hoisted addressing (registers are tight in the flat kernel): V^T row n = 16*dgrp + 2*vw + r has swizzle chunk
  // vwq ^ (vw & 3) for every dgrp, so the 8 stores per tile are one per-thread base plus immediates; the two
  // ldmatrix rows are key = 16*vwq + 8*(mat & 1) + (lane & 7) with chunk (4*hk + 2u + (mat >> 1)) ^ (lane & 7)
  const uint32_t vt_off0 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;
  const uint32_t vt_off1 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);
  const int vkey_t = 16 * vwq + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld_off0 = vkey_t * ROWB + (((4 * hk + 0 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  // the V^T stores as two stmatrix.x4 per thread (instead of eight 4-byte stores). For ldmatrix group u,
  // matrix m = 2*g2 + ab holds, in row j, the 16 key bytes of V^T row 16(2u+g2) + 2j + (ab ^ (j >= 4)) (the same
  // rows and the same half swap for bank spread as the per-thread stores: thread (q, w) supplies row j = w)
  const int stj = lane & 7, stm = lane >> 3;
  const uint32_t vst_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((vwq ^ (stj & 3)) << 4);
  // (per-request kernels: stmatrix measured vs16 -2.5%, vs1/vs32 -1.5%, vf1 -1%; the flat kernel: +0..1.5%)
  constexpr bool STSM_ON = !FLAT;
  // P fragments of the in-flight PV wgmma: kept live (fenced) until the next wgmma wait, so the compiler cannot
  // hand their registers to anything the next tile writes before that
  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
  // per-tile addressing as loop-carried values instead of re-derived from (tb + i) every tile: stage and
  // full-barrier phase counters, wgmma descriptors as base + stage offset (the 14-bit address field takes it without
  // a carry: shared memory < 256 KB), the V^T buffer and its descriptor toggled, new-row tiles found once
  // per-request (non-producer) q8 kernels with the split PV and the consumers' own V^T
  constexpr bool LSYNCN = !WS && !FLAT;
  constexpr bool LSYNC = WS;
  int st_c = tb % STAGES;
  uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;
  // descriptor low words broadcast from lane 0: a shuffle result is uniform and is not rematerialized, so the
  // bases stay in uniform registers (the high words are constants); +st*512 / +2 never carry out of the low word
  const uint32_t qlo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.q8s) + hk * 4096) >> 4) | 0x10000u, 0);
  const uint32_t klo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.k) + hk * 64) >> 4) | 0x10000u, 0);
  const uint64_t qdesc0 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | qlo0;
  const uint32_t vsrc0 = smem_u32(S.v);
  uint8_t* vtb_c = S.kvq + hk * 4608;
  const uintptr_t vtb_x = uintptr_t(S.kvq + hk * 4608) ^ uintptr_t(S.kvq + (2 + hk) * 4608);
  uint64_t vdesc_c = desc_sw64(smem_u32(S.kvq + hk * 4608));
  const uint64_t vdesc_x = vdesc_c ^ desc_sw64(smem_u32(S.kvq + (2 + hk) * 4608));
  const int i_new = seq > start ? (seq - start) / TILE : 0;   // first tile that can hold the block's new rows
  // interior tiles (no mask) are i in [i_lo, i_lo + i_span], found once (shuffled: kept uniform)
  uint32_t i_lo = 0, i_span = 0;
  

  // (producer mode): the common tiles -- no mask, no new rows (every key before the block's first new
  // position seq) -- are i in [r_lo, r_lo + r_span], found once; the per-tile test is then one compare, and t0, the
  // exact mask test and the new-row patch run only on the rare tiles
  uint32_t r_lo = 0x7fffffffu, r_span = 0u;
  if constexpr (WS) {
    const int hi2 = min(end - 1, seq - 1);
    const int a = (lo_all - start + TILE - 1) / TILE;
    const int z = hi2 - (TILE - 1) - start;
    const int bz = z >= 0 ? z / TILE : -1;
    r_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    r_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }
  // (flat producer mode): the two KV heads' warpgroups take turns in the exponential phase (named barriers
  // 5 / 6), so the 4 consumer warps of an SMSP do not all queue on the MUFU unit at once (scout: the exps segment
  // takes ~650-820 cycles for ~110 instructions; both warpgroups start each tile together on the same stage)
  
  int i_cur = 0, cnt2 = 0;   // F2: this warpgroup's claimed tile, and how many it has taken
  
  const int nit = ntiles;
  constexpr int UNR8 = WS ? 2 : 1;   // (pragma arguments are not macro-expanded;
  // plain split-KV with many tiles per CTA (b32 q8 sliding: 9) gains from the unroll, 1-tile coop grids lose)
#pragma unroll (UNR8)
  for (int jj = 0; jj < nit; ++jj) {
    const int i = jj;   // S2: tiles 2jj and 2jj + 1; F2: claimed
    
      if constexpr (WS) {
      // tile i landed (+ new rows); previous tile's PV of this warpgroup done (its V16 half is free). The two
      // warpgroups (one per KV head) only meet when both must: tile 0 (q staging) and tiles with new rows.
      bool patched;
      
        mbar_wait_full(&S.full[st_c], ph_c);
        patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni<L, NT, false>(&S, i, st_c, start, seq, end, tid);
        
       
      if (patched || i == 0) csync<WS, NT>();
      // (FP8 PV: V^T is double-buffered and each warp's wgmma wait already covers the PV that read this tile's
      // buffer, so the tile needs no warpgroup barrier before its transpose)
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)
    }
      
    

    const int st = st_c;
    
    float s[32];
    wgmma_fence();
    
      const uint32_t kl = klo0 + (uint32_t)(st * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      
       wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
     
    wgmma_commit();
    if constexpr (!WS) {   // the copies for tile i + STAGES - 1
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
    }
    if constexpr (FLAT) {
      // the previous tile's PV (committed before this QK^T) is done: its P registers are free for the transpose
      // below (the flat kernel's 104-register consumers spilled there) and o / lacc are settled
      wgmma_wait<1>();
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      float* od0 = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    }
    uint8_t* V16t = V16 + 0;   // PV reads it asynchronously: double buffer
    // this head's V^T for this tile (PV reads it asynchronously: double buffer per stream)
    uint8_t* vtb = vtb_c;
    
      if (jj == 0) {   // rows 64..71 of this head's two V^T buffers: e4m3 ones (0x38), so the PV
                                     // wgmma also produces the row sums (after the first barrier: the q staging shares this space)
        const int wt = tid & 127;
        *reinterpret_cast<uint2*>(S.kvq + ((0 + (wt >> 6)) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =
            make_uint2(0x38383838u, 0x38383838u);
      }
    
         // meanwhile: this head's V tile -> V^T (FP8, 72 x 64, keys in P-register order: k = 4q + 2h + b for
        // key = 8h + 2q + b within each 16-key block), two ldmatrix.trans (b16 pairs) + byte permutes per thread
      const uint32_t vsrc = vsrc0 + st * STAGE_BYTES;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vsrc + (u ? (vld_off0 ^ 32u) : vld_off0), r0, r1, r2, r3);   // (scout: off1 == off0 ^ 32)
        // r0/r1: dims group 2u (h = 0 / 1), r2/r3: dims group 2u+1
        if constexpr (STSM_ON) {
          stsm_x4(smem_u32(vtb) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),
                  prmt(r2, r3, vsel1));
        } else {
#pragma unroll
        for (int g2 = 0; g2 < 2; ++g2) {
          const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
          const uint32_t w0 = prmt(R0, R1, vsel0), w1 = prmt(R0, R1, vsel1);
          *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;   // row 16(2u+g2) + 2vw + vr0
          *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) = w1;   // row 16(2u+g2) + 2vw + 1 - vr0
        }
        }
      }
      if (!LSYNC && !LSYNCN) fence_proxy_async();   // V^T written with st.shared, read by the PV wgmma
     
      if (LSYNC || LSYNCN) {   // (/ N: fence + barrier just before the PV)
    } else wgsync(wgi);   // this head's V16 / V^T ready
                              // V16 ready
    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
    
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    
    if constexpr (WS) {   // K (wgmma) and V (conversion) of the stage are consumed
      mbar_arrive(&S.empty[st]);
    }
    

    const bool common_t = WS && ((uint32_t)i - r_lo <= r_span);
    const int t0 = start + i * TILE;
    const bool interior = common_t || ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all));
    if (!interior) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 8 * nt + 2 * tig + e;
            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
          }
      }
    }
    // after a segment's first tile the offset test runs on the packed P instead of on 32 row-max steps: an
    // e4m3 byte above 0x78 (P > 256, i.e. a score past the offset + 8) sets bit 7 when 0x07 is added to every byte
    // (P >= 0: no carry between bytes); the (rare) hit moves the offset and repacks that half (see the PV loop)
    constexpr bool PCHK = false;   // (attn_exp rowmax: per-tile row max test on the flat path too)
    const bool chk = PCHK && !first;
    float mlo = -INFINITY, mhi = -INFINITY;
    if (!chk) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
      mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
    }
    }
    // the threshold check needs no row reduction: any element past the offset + 8 shows in its own thread's
    // partial max; only the (rare) offset move and the first tile reduce over the row's quad
    if (!chk) {
      mlo *= ssc;   // scores -> log2 units (ssc > 0; ssc = 1 when the query was pre-scaled)
      mhi *= ssc;
    }
    float dlo, dhi;
    if (first) {
       mlo = quad_max(mlo); mhi = quad_max(mhi); 
      
      dlo = (mlo == -INFINITY) ? 0.f : sch_q(mlo);   // (attn_exp: 1/8 grid, SCH_K below the max)
      dhi = (mhi == -INFINITY) ? 0.f : sch_q(mhi);
      cr[0] = dlo; cr[1] = dhi;
      
      first = false;
    } else {
      dlo = cr[0]; dhi = cr[1];
      if (!PCHK && __any_sync(0xffffffffu, (mlo > cr[0] + 8.f || mhi > cr[1] + 8.f))) {
        // rare: a score grew past 2^8 relative to the offset -> move the offset up, rescale the state
         mlo = quad_max(mlo); mhi = quad_max(mhi);    // (warp-uniform branch: full-mask shuffles)
        
        const float nlo = fmaxf(sch_q(mlo), cr[0]), nhi = fmaxf(sch_q(mhi), cr[1]);
        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
        // (the previous tile's PV wgmma completed at this tile's wgmma wait)
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;   // (FP8 PV: the row sums live here too)
        cr[0] = nlo; cr[1] = nhi;
        dlo = nlo; dhi = nhi;
        
      }
    }
       // my turn for the exponentials
    
       // P (e4m3, registers, k-steps of 32 permuted keys) x V^T (e4m3, K-major, 72 columns: 64 dims + 8 copies of
        // the row sum) as an asynchronous wgmma; the next tile's wgmma wait covers it before P, o or V^T are reused
      constexpr bool PK2 = WS;
      if constexpr (PK2) {
        auto pack2 = [&]() -> bool {   // both halves of P (bit-trick codes); true when a code passed 120 (P > 256)
          const sch_t blo = sch_b(dlo), bhi = sch_b(dhi);
          uint32_t mx = SCH_ZERO;
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            const uint32_t r0 = sch_h2(sv[0], sv[1], c8s, blo), r1 = sch_h2(sv[4], sv[5], c8s, blo);
            const uint32_t r2 = sch_h2(sv[2], sv[3], c8s, bhi), r3 = sch_h2(sv[6], sv[7], c8s, bhi);
            const uint32_t r4 = sch_h2(sv[8], sv[9], c8s, blo), r5 = sch_h2(sv[12], sv[13], c8s, blo);
            const uint32_t r6 = sch_h2(sv[10], sv[11], c8s, bhi), r7 = sch_h2(sv[14], sv[15], c8s, bhi);
            pa[kk][0] = prmt(r0, r1, 0x6420u);
            pa[kk][1] = prmt(r2, r3, 0x6420u);
            pa[kk][2] = prmt(r4, r5, 0x6420u);
            pa[kk][3] = prmt(r6, r7, 0x6420u);
            mx = sch_max(mx, sch_max(sch_max(sch_max(r0, r1), sch_max(r2, r3)), sch_max(sch_max(r4, r5), sch_max(r6, r7))));
          }
          return sch_over(mx);
        };
        const bool povr = pack2();
        if (chk) {
          if (__any_sync(0xffffffffu, povr)) {
            // rare: offsets up to the tile's row maxima, state rescaled (no PV of this tile issued yet; the previous
            // tile's finished at its end), both halves repacked (P <= 1 afterwards: one pass)
            float hlo = -INFINITY, hhi = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
              hlo = fmaxf(hlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
              hhi = fmaxf(hhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
            }
            hlo = quad_max(hlo) * ssc;
            hhi = quad_max(hhi) * ssc;
            const float nlo = fmaxf(sch_q(hlo), cr[0]), nhi = fmaxf(sch_q(hhi), cr[1]);
            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
            for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
            lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
            cr[0] = nlo; cr[1] = nhi;
            dlo = nlo; dhi = nhi;
            pack2();
          }
        }
        float* odk = &o[0][0][0];
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(odk[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        if constexpr (LSYNC) {   // every warp's V^T for this tile written (and visible to the async proxy)
          fence_proxy_async();
          wgsync(wgi);
        }
        wgmma_fence();
        
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);
         
        wgmma_commit();
      }
#pragma unroll
      for (int kk = 0; kk < (PK2 ? 0 : 2); ++kk) {
        const float* sv = s + 16 * kk;
#define EXQ(x, q) ex2(x)
        // P rounded to e4m3 for the FP8 PV wgmma: the same precision as SGLang's FA3 FP8-KV path, which also feeds
        // P to its PV GEMM as e4m3 (flash-attention hopper, Is_FP8).
        bool rchk = chk;   // one retry at most: after the offset move every finite P is <= 1 (NaN / garbage scores
                           // would otherwise retrigger the check forever)
#pragma unroll 1
        for (;;) {   // (repeated once after a rare offset move; otherwise a single pass)
        {   // bit-trick codes (this path moves the offset at the row max test, so every code is <= 120 + rounding)
          const sch_t blo = sch_b(dlo), bhi = sch_b(dhi);
          pa[kk][0] = prmt(sch_h2(sv[0], sv[1], c8s, blo), sch_h2(sv[4], sv[5], c8s, blo), 0x6420u);
          pa[kk][1] = prmt(sch_h2(sv[2], sv[3], c8s, bhi), sch_h2(sv[6], sv[7], c8s, bhi), 0x6420u);
          pa[kk][2] = prmt(sch_h2(sv[8], sv[9], c8s, blo), sch_h2(sv[12], sv[13], c8s, blo), 0x6420u);
          pa[kk][3] = prmt(sch_h2(sv[10], sv[11], c8s, bhi), sch_h2(sv[14], sv[15], c8s, bhi), 0x6420u);
        }
        if constexpr (!PCHK) break;
        else {
          if (!rchk) break;
          {
            // (test hook, a hit at P > 1 instead of P > 256: exercises the offset move on most tiles)
            constexpr uint32_t PB = 0x07070707u;
            const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
            if (!__any_sync(0xffffffffu, ov != 0u)) break;
            {
              // rare: a score of this half passed its row's offset + 8: offsets up to the half's row maxima,
              // state rescaled (half 0's PV wgmma finished first), this half repacked
              float hlo = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));
              float hhi = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])), fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));
              hlo = quad_max(hlo) * ssc;
              hhi = quad_max(hhi) * ssc;
              const float nlo = fmaxf(sch_q(hlo), cr[0]), nhi = fmaxf(sch_q(hhi), cr[1]);
              const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
              if ((kk == 1)) {   // (a PV wgmma may still accumulate into o)
                wgmma_wait<0>();
                float* odw = &o[0][0][0];
#pragma unroll
                for (int e = 0; e < 32; ++e) reg_fence(odw[e]);
#pragma unroll
                for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
              }
#pragma unroll
              for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
              lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
              cr[0] = nlo; cr[1] = nhi;
              dlo = nlo; dhi = nhi;
              rchk = false;
            }
          }
        }
        }   // (pack / check loop)
#undef EXQ
           // each half of P goes to the tensor cores as soon as it is packed
          float* od = &o[0][0][0];
          if constexpr (LSYNCN) {   // every warp's V^T for this tile written (and visible to the async proxy)
            if (kk == 0) {
              fence_proxy_async();
              wgsync(wgi);
            }
          }
          if (kk == 0) {
#pragma unroll
            for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
            for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[kk][e]));
          wgmma_fence();
          wgmma_m64n72k32_e4m3(od, lacc, pa[kk], vdesc_c + 2 * kk);
          
          wgmma_commit();
        
      }
      
      
      float* od = &o[0][0][0];
      
           // done before the next tile (no wgmma in flight across the loop edge)
          wgmma_wait<0>();
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        
          // (!PVSPLIT)
     
    PROF_TILE;
    
    
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
    
  }
  PROF_DUMP;
  
  wgmma_wait<0>();
  
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
  
  {
    float* od = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  
  m_r[0] = first ? -INFINITY : cr[0];
  m_r[1] = first ? -INFINITY : cr[1];
  l_r[0] = 0.25f * lacc[0];   // every lane of the quad holds the full row sum; the epilogue quad-sums
  l_r[1] = 0.25f * lacc[2];
  
  }

  // ---- merge: rescale to the fixed reference 2^0 and accumulate with FP32 atomics ----
  bool do_red = true;
     // stream 1's state was merged into stream 0
  if (L == 1) {
    // warps 2,3 (second key half) hand their state to warps 0,1 (same heads) through shared memory
    // (outside cluster mode both halves add their own rescaled state to the workspace instead)
    cp_wait<0>();
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;
    if (wsub == 1) {
      xs[0] = m_r[0];
      xs[1] = l_r[0];
#pragma unroll
      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }
    }
    csync<WS, NT>();
    if (wsub == 0) {
      const float m1 = xs[0], l1 = xs[1];
      const float mm = fmaxf(m_r[0], m1);
      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
      const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
      m_r[0] = mm;
      l_r[0] = l_r[0] * f0 + l1 * f1;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;
        o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;
      }
    } else {
      do_red = false;
    }
  }
  if (M_CL) {
    cp_wait<0>();
    csync<WS, NT>();
    float* PB = reinterpret_cast<float*>(S.k);
    float* PL = PB + L * 16 * 64;
    // two tile streams (Cfg::S2): stream 0 writes its rows, then stream 1 (same rows) adds its own
#pragma unroll 1
    for (int ph = 0; ph < 1; ++ph) {
    if (ph == 1) csync<WS, NT>();
    const bool mine = true;
#pragma unroll
    for (int r = 0; r < NROW; ++r) {
      const int mt = r >> 1, hf = r & 1;
      const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
      const int row = j * 16 + hk * 8 + g;
      const float lsum = quad_sum(l_r[r]);
      if (do_red && mine) {
        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
        if constexpr (L == 1) {   // permuted dims: n-tile t, column c <-> dim 16*tig + 8*c + t
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(&PB[row * 64 + 16 * tig + 8 * c + t4]) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
        } else {                  // natural dims: n-tile t, column c <-> dim 8*t + 2*tig + c
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            float2* d2 = reinterpret_cast<float2*>(&PB[row * 64 + 8 * t + 2 * tig]);
            const float2 pv = ph ? *d2 : make_float2(0.f, 0.f);
            *d2 = make_float2(pv.x + o[mt][t][2 * hf] * f, pv.y + o[mt][t][2 * hf + 1] * f);
          }
        }
        if (tig == 0) PL[row] = (ph ? PL[row] : 0.f) + lsum * f;
      }
    }
    }
    csync<WS, NT>();
    cluster_finish<L>(p, S, b, pair, n_keys, chunk);
    return true;
  }
  // BULKRED (q_len 1 per-request kernels, not the one-tile-per-split builds): the partial goes to shared memory (the V
      // stages; the K stages hold the exchange) in the workspace's own row layout ([16][72]: dims, the row sum in column 64,
      // pad) and one TMA bulk reduction adds it into the workspace below
      constexpr bool BULKRED = PADWS;
      constexpr int PSTR = WSR<PADWS>::STR;                        // smem partial row stride (floats)
      float* pbuf = reinterpret_cast<float*>(S.v);
      if (do_red) {   // (one branch, not one per row: the post-loop merge runs straight-line from a cold cache)
    #pragma unroll
      for (int r = 0; r < NROW; ++r) {
    const int mt = r >> 1, hf = r & 1;
    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
    const int hh = hk * 8 + g;
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * WSR<PADWS>::STR;
        if constexpr (BULKRED) {   // permuted dims (see below): float4 of n-tiles t4..t4+3 at dim 16*tig + 8*c + t4
    #pragma unroll
          for (int c = 0; c < 2; ++c)
    #pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(pbuf + hh * PSTR + 16 * tig + 8 * c + t4) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
          if (tig == 0) {   // the row sum in column 64, the pad zeroed (the bulk op adds whole rows)
            *reinterpret_cast<float4*>(pbuf + hh * PSTR + WSL) = make_float4(lsum * f, 0.f, 0.f, 0.f);
            *reinterpret_cast<float4*>(pbuf + hh * PSTR + WSL + 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
          continue;
        } else if constexpr (L == 1) {
    #pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
    } else {
      // 16-byte atomics (half the L2 atomic operations of 8-byte ones): lane pairs (tig, tig ^ 1) swap halves so
      // each lane holds 4 consecutive dims: even tig keeps t even, odd tig keeps t odd (dims 8t + 4(tig >> 1) + 0..3)
      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
        // mine: (t, 2tig..2tig+1) and (t+1, same); partner's: same t's at 2(tig^1)..
        const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;
        const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;
        // even lane sends its t+1 pair, odd lane sends its t pair
        const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
        const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        // even lane: t (dims 2tig, 2tig+1, then the partner's); odd lane: t+1 (the partner's, then mine) -- selects,
        // one atomic, no branch
        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);
      }
    }
    if (tig == 0) red_f32(WSL_AT(PADWS, ((bp * p.nc + copy) * L + j) * 16 + hh), lsum * f);
      }
      }
      if constexpr (BULKRED) {
        fence_proxy_async();   // the partial's st.shared, visible to the async proxy
        csync<WS, NT>();
        if (tid == 0) {
          bulk_red_add_f32(p.ws_acc + ((ix_t)((bp * p.nc + copy) * L) * 16) * WSR<PADWS>::STR, pbuf,
                           L * 16 * WSR<PADWS>::STR * 4);
          bulk_commit();
          bulk_wait_all();
          fence_proxy_async_global();   // ... performed, and ordered (through the CTA barrier) before the ticket's release
        }
      }
      // (last-CTA finalizes): no per-thread fences; one acq_rel atomic by thread 0 after the CTA barrier
  // releases every thread's reds (cumulativity through bar.sync, as CUTLASS's semaphore release; okc-license: own) and acquires the
  // others' for the last CTA's reads
  const bool mode1 = !((FLAT && (L == 8)) && M_COOP) && !(!FLAT && M_COOP);
  // the same for the cooperative grids (the barrier's arrive is an acq_rel atomic after a CTA barrier)
  if (!(mode1 || !mode1)) gpu_fence();
  csync<WS, NT>();
  if ((FLAT && (L == 8)) && M_COOP) {
    // flat cooperative mode: every request is finalized after all CTAs have merged all their segments
  } else if (!FLAT && M_COOP) {
    // ---- all splits of this (request, kv pair) are resident: wait for every split's merge, then each
    // CTA finalizes its own slice of the rows (instead of one CTA doing all of them)
    // (the slice arithmetic first: it does not depend on the merge, so its divisions leave the barrier's chain)
    const int split_idx = (int)blockIdx.x;
    int nact = fdiv_i(n_keys + chunk - 1, chunk);    // splits with keys
    int R = fdiv_i(L * 16 + nact - 1, nact);          // rows per CTA
    // keep both divisions above the barrier wait (ptxas otherwise sinks them below the spin, where
    // they are first used: scout's v200 census found them after the poll); volatile asm keeps program order
    asm volatile("" : "+r"(nact), "+r"(R));
      if (tid == 0) {
      // two key counters alternating per launch (parity = this slot's epoch, read in the prologue): the last split
      // to arrive resets the OTHER counter (unused until the next launch) and bumps the epoch, so nobody needs a
      // departure counter (no atomic round trip at the very end of the kernel)
      const int ep = S.sense;
      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
      const int mine = end - start;
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
      
      if (old + mine == n_keys) {
        p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;
        p.ws_cnt[24576 + bp] = ep + 1;
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
          if (v >= n_keys) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0) {
      atomicAdd(p.ws_cnt + bp, end - start);
      int v;
      SPIN_T0;
      for (int spin = 0;; ++spin) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + bp) : "memory");
        if (v >= n_keys) break;
        SPIN_CHECK;
        
        __nanosleep(64);
      }
    }
    csync<WS, NT>();
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);
    const int part = tid & 7;
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
      const int rr = rbase + (tid >> 3);
      const bool act = rr < r_hi;
      const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                             : make_uint4(0, 0, 0, 0);
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      sum_copies<L, PADWS>(p, bp, j, hh, part, act, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = (y[e] * inv); ss += y[e] * y[e]; }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
    
  } else {
  if (tid == 0) {
    const int mine = end - start;
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    
    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
  if (S.flag) {
  

  // ---- last CTA: sum the copies, normalize, rmsnorm * sigmoid(gate), store; reset the workspace ----
  if (p.nc == 1) {
    fin_nc1<L, PADWS, NT>(p, b, pair, bp, vscale, 0);
  } else
  {
    const int part = tid & 7;
    for (int rr = tid >> 3; rr < L * 16; rr += NT / 8) {
      const int j = rr >> 4, hh = rr & 15;
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      // gate row in flight with the copies (one L2 round trip per row instead of two)
      const uint4 graw = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
      sum_copies<L, PADWS>(p, bp, j, hh, part, true, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = (y[e] * inv); ss += y[e] * y[e]; }
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
  if (tid == 0) p.ws_cnt[bp] = 0;
  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 16 + 6] = 1;
  }
  }
     // consumers' shared memory is reused by the next segment
   __syncthreads();                   // shared memory is reused by the next segment
  }
  return true;   // okc-pdl: in the layer this grid is a PDL launch (T <= 128); this exit reads no qkv, and the CTAs that read qkv / write the output wait first
}

// Flat grids whose CTAs are all resident: once every CTA has merged all its segments (grid-wide count),
// each CTA finalizes its own slice of all requests' rows (sum the copies, normalize, rmsnorm * gate, store,
// zero the workspace). Runs on the NT consumer threads; the last CTA out resets the two counters.
template <int L, bool WS, int NT>
__device__ __forceinline__ void flat_finalize(const Params& p, const int* sense_s) {
  const int tid = threadIdx.x;
  pdl_wait();   // a CTA without segments has not waited yet (the gate rows are qkv)
  const int G = gridDim.x * gridDim.y * gridDim.z;
  const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int* cnt = p.ws_cnt + 30000;
     // (thread 0's acq_rel arrive releases the CTA's reds)
  csync<WS, NT>();
  if (tid == 0) {
    // two grid counters alternating per launch (parity = the epoch thread 0 read at kernel start): the last CTA
    // to arrive resets the other counter and bumps the epoch; no departure counter at the end of the kernel
    cp_wait<0>();
    const int ep = *reinterpret_cast<const volatile int*>(sense_s);
    int* ac = p.ws_cnt + 30020 + (ep & 1);
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(ac) : "memory");
    
    if (old + 1 == G) {
      p.ws_cnt[30020 + ((ep & 1) ^ 1)] = 0;
      p.ws_cnt[30022] = ep + 1;
      
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(ac) : "memory");
        if (v >= G) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else   if (tid == 0) {
    atomicAdd(cnt, 1);
    int v;
    SPIN_T0;
    for (;;) {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt) : "memory");
      if (v >= G) break;
      SPIN_CHECK;
      __nanosleep(64);
    }
  }
  csync<WS, NT>();
  const int pairs = p.Hkv >> 1;
  const int rows = p.B * pairs * L * 16;
  const int R = fdiv_i(rows + G - 1, G);
  const int r_lo = c * R, r_hi = min(rows, r_lo + R);
  const float vscale = __ldg(p.v_scale);
  const int part = tid & 7;
  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {   // warp-uniform trip count (shuffles below)
    const int r = rbase + (tid >> 3);
    const bool act = r < r_hi;
    const int bp = act ? r / (L * 16) : 0, rr = act ? r % (L * 16) : 0;
    const int j = rr >> 4, hh = rr & 15;
    const int b = (pairs == 1) ? bp : fdiv_i(bp, pairs), pair = bp - b * pairs;
    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                           : make_uint4(0, 0, 0, 0);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    sum_copies<L, false>(p, bp, j, hh, part, act, y, lsum);
    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = (y[e] * inv); ss += y[e] * y[e]; }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
  
}

// (EROREx a8, OKC_PFIDX) B == 1: the page-table lines of the range this CTA owns if seq = width - L, toward L2 (a hint:
// the page table is a pure input, read before griddepcontrol.wait like the index copies; the real length drives everything)
template <int L>
__device__ __forceinline__ void pf_idx(const Params& p, const int split) {
  const int seq = (p.max_pages << p.lp) - L;
  if (seq <= 0) return;
  const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
  const int n_keys = seq + L - lo;
  const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;
  const int chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;
  const int start = lo + split * chunk;
  const int end = min(seq + L, start + chunk);
  if (start >= end) return;
  const int l0 = (start >> p.lp) >> 5, l1 = ((end - 1) >> p.lp) >> 5;   // 128-byte lines of row 0 (32 entries)
  for (int l = l0 + (int)threadIdx.x; l <= l1; l += (int)blockDim.x)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p.page_table + ((ix_t)l << 5)));
}

template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  // 1024-byte aligned base: the 128B-swizzled tiles must start on swizzle-pattern boundaries (wgmma)
  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  const int pair = blockIdx.z;
  if (p.pf_chunk && threadIdx.x == 0) {   // (layer) the pooler's weight stream toward L2, one slice per CTA
    const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;
    for (unsigned o = 0; o < p.pf_chunk; o += 32768u) {
      const unsigned n = min(32768u, p.pf_chunk - o);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + o), "r"(n) : "memory");
    }
  }
  if (!FLAT) {
    // grid (n_splits, B): request b, split s of its window / context
    const int split = blockIdx.x, b = blockIdx.y;
    // (MODE | 8 builds, chosen by the host for one request with page size 1): SGLang's FA3 backend builds
    // page_table = req_to_token[req_pool_indices, :max_seq_len_k] (sglang/srt/layers/attention/
    // flashattention_backend.py, init_forward_metadata: decode / target-verify paths), so with one request its row
    // is exactly seq_len + q_len entries wide. The page table's width is a shape: speculate seq = width - q_len and
    // issue the window's page indices in the same trip as cache_seqlens; the real length is checked (segment) before
    // the indices are used for any K/V address or any softmax work, and a miss re-runs everything with the real one
    const int seq_ld = __ldg(p.seq_lens + b);
    if (p.B == 1 && ((MODE & 3) == 0 ? p.cs <= 1 : (MODE & 3) != 2)) pf_idx<L>(p, split);   // (EROREx a8 OKC_PFIDX)
    constexpr bool SPEC = (MODE & 8) != 0;
    int seq = SPEC ? p.max_pages - L : seq_ld;
    for (;;) {
    const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
    const int n_keys = seq + L - lo;
    int chunk = fdiv_i(n_keys + p.n_splits - 1, p.n_splits);
    chunk = (chunk + 15) & -16;
    // plain split-KV: whole tiles per split (the per-CTA tile count is what costs); trailing splits may get no keys:
    // they skip the workspace merge (the merge counts keys, the cooperative finalize counts the splits with keys).
    // Not for clusters (an empty cluster CTA still runs the cluster epilogue: measured slower)
    if ((MODE & 3) == 0 ? p.cs <= 1 : (MODE & 3) != 2) {
      const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;
      // ceil(2^32 / n_splits) from the host (exact for ntl < 2^20, n_splits <= 1024): one IMAD.HI instead
      // of the ~20-instruction fdiv_i on the once-path (scout's census)
      chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;
      
    }
    int start = lo + split * chunk;
    int end = min(seq + L, start + chunk);
    // splits beyond the SM count share an SM with an earlier split, which gets the warp schedulers'
    // priority: the first-wave splits take w0/1024 of a second-wave split's tiles (only when every split still gets
    // at least one tile, so all splits have keys and the cooperative finalize counts all of them)
    
    if (start >= end) {
      if (SPEC && seq != seq_ld) { seq = seq_ld; continue; }   // (an empty split only on the real length)
      if ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2) {   // still part of its cluster: contributes an empty partial, finalizes its slice
        pdl_wait();     // (the gate rows are qkv)
        cluster_stage_gate<L>(p, S, b, pair);
        if (threadIdx.x == 0) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
        cp_commit();
        float* PB = reinterpret_cast<float*>(S.k);
        float* PL = PB + L * 16 * 64;
        for (int e = threadIdx.x; e < L * 16 * 64; e += NT) PB[e] = 0.f;
        for (int e = threadIdx.x; e < L * 16; e += NT) PL[e] = 0.f;
        cp_wait<0>();
        __syncthreads();
        cluster_finish<L>(p, S, b, pair, n_keys, chunk);
      }
      return;   // okc-pdl: in the layer this grid is a PDL launch (T <= 128); this exit reads no qkv, and the CTAs that read qkv / write the output wait first
    }
    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys, split - fdiv_i(split, p.nc) * p.nc,
                                  chunk, 0, SPEC ? seq_ld : seq))
      return;   // okc-pdl: in the layer this grid is a PDL launch (T <= 128); this exit reads no qkv, and the CTAs that read qkv / write the output wait first
    seq = seq_ld;   // (miss: today's path with the real length)
    }
  }
  if constexpr (FLAT && (L == 8)) {
    if (p.coop && threadIdx.x == 0) {   // the grid barrier's epoch, long before it is needed
      cp_async4(smem_u32(&S.sense), p.ws_cnt + 30022);
      cp_commit();
    }
  }
  // the per-SM registration's atomics go out from warp 1 while warp 0 builds the prefix sums
  // the same uneven per-SM shares for the non-cooperative (q_len 1) flat grids. No grid barrier ends those,
  // so the registration state is generation-tagged: a monotonic launch counter picks one of two state copies, and
  // the launch's last registrant clears the other copy (the previous launch's) for the next launch
  int w0g_rank = 2, w0g_par = 0, w0g_gen = 0;
  unsigned w0_smid = 0;
  int w0_rank = 0;
  
  
  // grid (G): CTA c takes keys [c*T/G, (c+1)*T/G) of all requests' keys laid end to end (equal work)
  // warp 0 computes the key range and the segment list right after its prefix sums (one request per lane)
  constexpr bool fpart = L == 1;   // (compile time: no run-time branch in the main flow)
  if constexpr (L == 1 && Cfg<L, FLAT>::WS) {
    if (threadIdx.x == 32) {   // the first segment's stage barriers (segment skips its init for that segment)
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::NT);
      }
      mbar_fence_init();
    }
  }
  if (threadIdx.x < 32) {
    int run = 0;
    int incl_l = 0, n_l = 0, seq_l = 0;
    for (int b0 = 0; b0 < p.B; b0 += 32) {
      const int b = b0 + (int)threadIdx.x;
      int n = 0;
      if (b < p.B) {
        const int seq = __ldg(p.seq_lens + b);
        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
        n = seq + L - lo;
        seq_l = seq;
      }
      int incl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, o);
        if ((int)threadIdx.x >= o) incl += t;
      }
      if (b < p.B) S.cum[b + 1] = run + incl;
      run += __shfl_sync(0xffffffffu, incl, 31);
      incl_l = incl;
      n_l = n;
    }
    if (threadIdx.x == 0) S.cum[0] = 0;
    if (fpart && p.B > 32) {   // many requests: the generic partition by lane 0 (after the whole prefix sum)
      __syncwarp();
      if (threadIdx.x == 0) {
        constexpr int SEGKF = (L == 8) ? 600 : 400;
        const int64_t Tg = (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
        int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);
        if (SEGKF > 0) {
          auto vmapg = [&](int v) -> int {
            int lo_b = 0, hi_b = p.B - 1;
            while (lo_b < hi_b) {
              const int mid = (lo_b + hi_b + 1) >> 1;
              if (S.cum[mid] + mid * SEGKF <= v) lo_b = mid; else hi_b = mid - 1;
            }
            return min(v - lo_b * SEGKF, S.cum[lo_b + 1]);
          };
          csg = vmapg(csg);
          ceg = vmapg(ceg);
        }
        int b = 0, n = 0;
        while (S.cum[b + 1] <= csg) ++b;
        for (; b < p.B && S.cum[b] < ceg && n < MAXSEG; ++b) {
          S.seg[n][0] = b;
          S.seg[n][1] = max(csg, S.cum[b]) - S.cum[b];
          S.seg[n][2] = min(ceg, S.cum[b + 1]) - S.cum[b];
          ++n;
        }
        S.nseg = n;
        S.fcs = csg;
        S.fce = ceg;
      }
    } else if (fpart) {   // (one iteration above: lane b holds cum[b + 1] = incl_l, cum[b] = incl_l - n_l; lanes >= B: n = 0)
      const unsigned FULL = 0xffffffffu;
      const int lane_f = (int)threadIdx.x;
      constexpr int SEGKF = (L == 8) ? 600 : 400;
      const int excl = incl_l - n_l;
      const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
      const int cf = blockIdx.x, Gf = gridDim.x;
      int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);
      if (SEGKF > 0) {   // virtual -> real: the last request whose virtual start (cum[b] + b * SEGK) is <= v
        const int vstart = excl + lane_f * SEGKF;
        const unsigned ms = __ballot_sync(FULL, lane_f < p.B && vstart <= csf);
        const unsigned me = __ballot_sync(FULL, lane_f < p.B && vstart <= cef);
        const int bs = 31 - __clz(ms), be = 31 - __clz(me);   // (lane 0 always qualifies: virtual start 0)
        const int is = __shfl_sync(FULL, incl_l, bs), ie = __shfl_sync(FULL, incl_l, be);
        csf = min(csf - bs * SEGKF, is);
        cef = min(cef - be * SEGKF, ie);
      }
      // segment list: the requests with cum[b + 1] > cs and cum[b] < ce (a contiguous run of lanes)
      const bool in = lane_f < p.B && incl_l > csf && excl < cef;
      const unsigned mi = __ballot_sync(FULL, in);
      const int rank = __popc(mi & ((1u << lane_f) - 1u));
      if (in && rank < MAXSEG) {
        S.seg[rank][0] = lane_f;
        S.seg[rank][1] = max(csf, excl) - excl;
        S.seg[rank][2] = min(cef, incl_l) - excl;
      }
      if (lane_f == 0) {
        S.nseg = min(__popc(mi), MAXSEG);
        S.fcs = csf;
        S.fce = cef;
      }
      if (mi != 0u && csf < cef) {   // the first segment's first STAGES tiles of page indices toward L2
        const int b0 = __ffs(mi) - 1;
        const int seqb = __shfl_sync(FULL, seq_l, b0), exb = __shfl_sync(FULL, excl, b0), inb = __shfl_sync(FULL, incl_l, b0);
        const int lob = p.window ? max(0, seqb + 1 - p.window) : 0;
        const int s0 = max(csf, exb) - exb + lob, s1 = min(cef, inb) - exb + lob;
        const int cend = min(min(s1, seqb), s0 + Cfg<L, FLAT>::STAGES * TILE);
        if (cend > s0) {
          const long long e0 = (long long)b0 * p.max_pages + (s0 >> p.lp);
          const long long e1 = (long long)b0 * p.max_pages + ((cend - 1) >> p.lp);
          const long long l0 = e0 >> 5, l1 = e1 >> 5;   // 128-byte lines of int32 page indices
          if (l0 + lane_f <= l1) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.page_table + ((l0 + lane_f) << 5)));
        }
      }
    }
  }
  __syncthreads();
  const int64_t T0 = S.cum[p.B];
  // every request boundary costs a CTA that crosses it a whole extra segment (q staging, pipeline refill,
  // merge: ~7 us at q_len 8), so the shares are cut in a virtual key space with SEGK keys inserted before each
  // request after the first, then mapped back (a CTA crossing a boundary gets SEGK fewer real keys)
  constexpr int SEGK = (L == 8) ? 600 : 400;
  const int64_t T = T0 + (int64_t)(p.B - 1) * SEGK;
  const int c = blockIdx.x, G = gridDim.x;
  // (experiment: the first-dispatched half of the grid takes the second half of the keys)
  const int cpart = c;
  int cs, ce;
  if constexpr (fpart) {   // (computed by warp 0)
    cs = S.fcs;
    ce = S.fce;
  } else {
    cs = (int)((int64_t)cpart * T / G);
    ce = (int)((int64_t)(cpart + 1) * T / G);
  }   // (a float-estimate + correction
  // version without the 64-bit division call measured slower: q_len 1 b16 full +3 us, likely code layout)
  
  int vc = c;
  
  
  if (!fpart) {   // virtual -> real key positions
    auto vmap = [&](int v) -> int {
      int lo_b = 0, hi_b = p.B - 1;   // the last request whose virtual start is <= v
      while (lo_b < hi_b) {
        const int mid = (lo_b + hi_b + 1) >> 1;
        if (S.cum[mid] + mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;
      }
      return min(v - lo_b * SEGK, S.cum[lo_b + 1]);
    };
    cs = vmap(cs);
    ce = vmap(ce);
  }
  if (cs >= ce) {
    if constexpr (L == 8) {
      if (p.coop) {   // still part of the grid-wide finalize
        if constexpr (Cfg<L, FLAT>::WS) {
          if (threadIdx.x < Cfg<L, FLAT>::NT) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
        } else {
          flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
        }
      }
    }
    return;   // okc-pdl: in the layer this grid is a PDL launch (T <= 128); this exit reads no qkv, and the CTAs that read qkv / write the output wait first
  }
  
  
  // segment list in shared memory (keeps the per-segment loop state out of registers)
  if (threadIdx.x == 0 && !fpart) {
    int b = 0, n = 0;
    while (S.cum[b + 1] <= cs) ++b;
    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {
      S.seg[n][0] = b;
      S.seg[n][1] = max(cs, S.cum[b]) - S.cum[b];
      S.seg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];
      ++n;
    }
    S.nseg = n;
  }
  if (!fpart) __syncthreads();   // (warp 0 wrote the list before the barrier above)
  const int nseg = S.nseg;
  auto run_segments = [&](auto role) {
    constexpr int R = decltype(role)::value;
    int tb = 0;
#pragma unroll 1
    for (int k = 0; k < nseg; ++k) {
      const int b = S.seg[k][0];
      const int seq = __ldg(p.seq_lens + b);
      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
      const int n_keys = seq + L - lo;
      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;
      
      if (s0 < s1) {
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            0, 0,
                            L == 1 && Cfg<L, FLAT>::WS && k == 0);
        tb += (s1 - s0 + TILE - 1) / TILE;
      }
      
    }
       // the producer leaves with its copies landed
  };
  if constexpr (Cfg<L, FLAT>::WS) {
    // separate code for the two roles, so the producer warpgroup can hand its registers to the consumers
    if (threadIdx.x >= Cfg<L, FLAT>::NT) {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::PREGS));
      run_segments(std::integral_constant<int, 1>());
    } else {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::CREGS));
      run_segments(std::integral_constant<int, 2>());
      if constexpr (L == 8) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }
    }
  } else {
    run_segments(std::integral_constant<int, 0>());
    if constexpr (FLAT && (L == 8)) {
      if (p.coop) flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
    }
  }
}

}  // namespace

template <int L>
void launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool pdl) {
  Params p = p0;
  static bool init = false;
  const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;
  const int sm = p.flat ? sm_f : sm_n;
  static int occ_c = 0, occ_f = 0, nsm = 0;
  static int occ_c1 = 0, occ_sp = 0, occ_sp1 = 0;   // the coop ONE / SPEC / SPEC-ONE instances
  typedef void (*KernFn)(const Params);
  // per-request kernels, one build per launch mode (1 plain split-KV, 2 clusters, 3 cooperative)
  static const KernFn kn[4] = {attn_kernel<L, false, 2>, attn_kernel<L, false, 1>,
                               attn_kernel<L, false, 2>, attn_kernel<L, false, 3>};
  static const KernFn kn1[4] = {0, attn_kernel<L, false, 5>, 0, attn_kernel<L, false, 7>};
  static const KernFn ksp[4] = {0, attn_kernel<L, false, 9>, 0, attn_kernel<L, false, 11>};
  static const KernFn ksp1[4] = {0, attn_kernel<L, false, 13>, 0, attn_kernel<L, false, 15>};
  if (!init) {
    for (int m = 1; m < 4; ++m)
      cudaFuncSetAttribute(kn[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kn1[3], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    for (int m = 1; m < 4; m += 2) {
      cudaFuncSetAttribute(ksp[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
      cudaFuncSetAttribute(ksp1[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    }
    cudaFuncSetAttribute(attn_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);
    cudaFuncSetAttribute(kn[2], cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, kn[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp, ksp[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel<L, true>, Cfg<L, true>::NTX, sm_f);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    init = true;
  }
  // the cooperative finalize waits for every split of a request: only when the whole grid is resident
  Params q = p;
  if (q.coop && (long long)grid.x * grid.y * grid.z > (long long)(p.flat ? occ_f : occ_c) * nsm) q.coop = 0;
  const bool cl = cs > 1;
  KernFn kern = p.flat ? (KernFn)attn_kernel<L, true> : kn[cl ? 2 : (q.coop ? 3 : 1)];
  if (!p.flat && !cl) {
    // every split holds at most one tile: whole tiles per split (TCHUNK) and at least as many splits as the longest
    // request's tiles (sliding windows: window + L - 1 keys)
    const long long kmax = p.window ? std::min<long long>((long long)p.window + L - 1, (long long)p.max_pages * p.P)
                                    : (long long)p.max_pages * p.P;
    if ((long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];
  }
  // one request, page size 1: the speculative-length builds (; the kernel verifies and falls back)
  if (!p.flat && !cl && p.B == 1 && p.P == 1 && p.max_pages >= L && (int)grid.y == 1) {
    const bool one = kern != kn[q.coop ? 3 : 1];
    kern = one ? ksp1[q.coop ? 3 : 1] : ksp[q.coop ? 3 : 1];
  }
  if (q.coop && !p.flat && !cl) {
    // the instance actually launched must hold the whole grid, else its grid barrier would wait forever
    int occ_k = occ_c;
    KernFn plain = kn[1];
    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }
    if (kern == ksp[3]) { occ_k = occ_sp; plain = ksp[1]; }
    if (kern == ksp1[3]) { occ_k = occ_sp1; plain = ksp1[1]; }
    if ((long long)grid.x * grid.y * grid.z > (long long)occ_k * nsm) {
      q.coop = 0;
      kern = plain;
    }
  }
  if (cl || pdl) {
    // cluster dims and / or programmatic stream serialization (PDL: this grid may start while the preceding
    // kernel finishes; the kernel waits with griddepcontrol.wait before reading qkv)
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, 1, 1);
    cfg.dynamicSmemBytes = sm;   // (one cluster CTA per SM)
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (cl) {
      attr[na].id = cudaLaunchAttributeClusterDimension;
      attr[na].val.clusterDim.x = cs;
      attr[na].val.clusterDim.y = 1;
      attr[na].val.clusterDim.z = 1;
      ++na;
    }
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    cudaLaunchKernelEx(&cfg, kern, q);
  } else {
    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);
  }
}

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
// [rows, row_bytes] byte view of a cache, boxes of one 128-byte row (one KV pair), 128-byte swizzle

void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,
         int64_t tma, int64_t pf_ptr, int64_t pf_bytes) {
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = page_table.data_ptr<int>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = ws_acc.data_ptr<float>();
  p.ws_l = ws_l.data_ptr<float>();
  p.ws_cnt = ws_cnt.data_ptr<int>();
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.P = (int)P;
  TORCH_CHECK(P > 0 && (P & (P - 1)) == 0, "page size must be a power of two");
  TORCH_CHECK(k_cache.numel() / (Hkv * 64) < (1LL << 31), "cache rows must fit in 31 bits");
  p.lp = 0;
  while ((1 << p.lp) < P) ++p.lp;
  p.max_pages = (int)page_table.size(1);
  p.window = (int)window; p.rope = (int)rope; p.n_splits = (int)n_splits; p.flat = (int)flat; p.nc = (int)nc;
  p.ns_magic = (n_splits >= 2 && n_splits <= 1024) ? (unsigned)(((1ull << 32) + (uint64_t)n_splits - 1) / (uint64_t)n_splits) : 0u;
  TORCH_CHECK(n_splits <= 1024, "n_splits <= 1024");
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.cs = (int)cs; p.ncl = (int)ncl;
  p.ld_qkv = (int)qkv.size(-1);
  p.row_bytes = (int)(Hkv * 64);
  TORCH_CHECK((p.row_bytes & (p.row_bytes - 1)) == 0, "Hkv * 64 must be a power of two");
  p.lrb = 0;
  while ((1 << p.lrb) < p.row_bytes) ++p.lrb;
  p.off32 = (k_cache.numel() < (1LL << 32) && v_cache.numel() < (1LL << 32)) ? 1 : 0;   // larger caches: 64-bit offsets
  TORCH_CHECK(ws_acc.numel() < (1LL << 31), "workspace too large");
  TORCH_CHECK(B * (Hkv / 2) <= 4096, "at most 4096 (request, kv pair) slots in the counter workspace");
  TORCH_CHECK(qkv.numel() < (1LL << 31), "qkv too large");
  TORCH_CHECK((out.numel() < (1LL << 31) && page_table.numel() < (1LL << 31)), "out / page table too large");
  p.scale_log2 = (float)scale_log2;
  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));
  p.pf = reinterpret_cast<const uint8_t*>(pf_ptr);
  p.pf_chunk = pf_ptr ? (unsigned)((pf_bytes / ((int64_t)grid.x * grid.y * grid.z)) & ~15ll) : 0u;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  p.coop = (int)((tma >> 1) & 1);
  const bool pdl = ((tma >> 16) & 1);   // programmatic dependent launch (off by default)
  if (L == 1) launch<1>(p, grid, stream, (int)cs, pdl);
  else launch<8>(p, grid, stream, (int)cs, pdl);
}
"""
ATT7_CPP_SRC = r"""
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl,
         int64_t tma, int64_t pf_ptr, int64_t pf_bytes);
"""
_ATT7 = None


def _att7():
    global _ATT7
    if _ATT7 is None:
        import os
        from torch.utils.cpp_extension import load_inline
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major >= 9 else "")
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
        # s302: the board file t853_both's build and flags (sm_90a only; the env write above stays for the
        # other extensions built in parallel threads)
        _ATT7 = load_inline(name=f"muse_att_t853x_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,
                            functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr"],
                            verbose=False)
    return _ATT7


_WS7 = {}
_PF7 = None   # (pooler weight stream pointer, bytes) for the next attention launch
_NUM_SMS7 = {}


def _workspace7(n_rows, device):
    """Persistent FP32 accumulators (zero between calls: the last CTA resets
    them), each followed by sentinel words -- see _guarded / _check_guards."""
    key = str(device)
    ws = _WS7.get(key)
    if ws is None or ws[0].numel() < n_rows * 72:
        n = max(n_rows, 1 << 15)
        ws = tuple(_guarded(t) for t in (
            torch.zeros(n * 72, dtype=torch.float32, device=device),   # (EROREx a7: 72-float rows, OKC_BRED1)
            torch.zeros(n, dtype=torch.float32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device)))
        _WS7[key] = ws
    return ws


def _num_sms7(device):
    key = str(device)
    if key not in _NUM_SMS7:
        _NUM_SMS7[key] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS7[key]


_TUNED7 = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),   # (experiment: 252 instead of 264 splits)
}


OKC_FLATCOOP = 1
OKC_PDL = 0      # launch with programmatic dependent launch (for a fused layer: overlaps the qkv GEMM's tail)
OKC_FLAT1B = 0   # q_len 1, one full-layer request: flat kernel (Oct 26 00:14: the per-request coop plan, 264 splits, 4 copies, is now faster: f1 13.2 -> 12.6 us)
OKC_PLAIN8 = 1   # q_len 8 sliding, >= 16 requests: plain split-KV, ~1 CTA per SM
OKC_F2 = 0   # must match the CUDA macro OKC_F2 (q_len 8 flat: 1 CTA per SM, two streams)
OKC_S2 = 0   # must match the CUDA macro OKC_S2 (q_len 8 per-request kernels: 2 streams, 1 CTA per SM)


def _plan7(B, pairs, L, window, max_pages, P, device):
    """Launch plan for a shape: n_splits CTAs per (request, KV pair) or, flat, per KV pair over all
    requests; clusters of cs CTAs reduce through DSMEM (ncl clusters per request); nc workspace copies."""
    sms = _num_sms7(device)
    target = 2 * sms
    if not window and 1 < B <= 512:
        ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target
        return dict(n_splits=max(1, ft // pairs), flat=1, cs=0, ncl=1, nc=1)
    keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
    tiles = max(1, (keys_max + 63) // 64)
    reqs = B * pairs
    if L == 8 and OKC_S2 and torch.cuda.get_device_capability(device)[0] >= 9:
        target = sms   # two tile streams per CTA, one CTA per SM (the kernel's OKC_S2)
    if not window:
        if L == 1 and B == 1 and OKC_FLAT1B:
            return dict(n_splits=max(1, ((target * 21) // 22) // pairs), flat=1, cs=0, ncl=1, nc=4)
        if L == 8:
            n = max(1, min(tiles, ((target * 21) // 22) // reqs))
            return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 8)))
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)), coop=0)
    if L == 1 and reqs < 24:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    if L == 8 and reqs < 16 and OKC_PLAIN8:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)   # (one copy since v101: b1 9.8 -> 9.6 us)
    if L == 8 and reqs >= 16 and OKC_PLAIN8:
        n = max(1, min(tiles, (target if reqs >= 24 else sms) // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
    if reqs >= 24:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
    cs = 16
    limit = 2 * sms if L == 1 else sms
    if L == 8 and target == sms:
        limit = (sms * 15) // 16   # one CTA per SM: clusters of 8 fit 15 per 132 SMs (GPC sizes), so stay below
    while cs > 4 and reqs * cs > limit:
        cs //= 2
    ncl = 1
    while cs * ncl > tiles and ncl > 1:
        ncl -= 1
    while cs * ncl > tiles and cs > 4:
        cs //= 2
    if cs * ncl > tiles or cs < 4:
        n = max(1, min(tiles, target // reqs))
        return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)))
    return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=max(1, min(ncl, 4)))



MOE_CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1
#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0's barrier inits
#define OKC_WGSFINSS 1
#ifndef OKC_MFIN
#define OKC_MFIN 1
#endif
#ifndef OKC_B4L
#define OKC_B4L 0
#endif
#ifndef OKC_MFP
#define OKC_MFP 15
#endif
#ifndef OKC_MFS
#define OKC_MFS 0
#endif
#ifndef OKC_MFQ
#define OKC_MFQ 47
#endif
#if !OKC_WGS
#undef OKC_MFIN
#define OKC_MFIN 0
#undef OKC_B4L
#define OKC_B4L 0
#endif
#ifndef OKC_B4FIN
#define OKC_B4FIN 0
#endif
#ifndef B4_EARLY_Y
#define B4_EARLY_Y 1
#endif  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
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
#define OKC_SPLIT 0
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
#define OKC_WGSX 0
#ifndef OKC_WGS
#define OKC_WGS 0
#endif
#define SILU_DIV(a, b) __fdividef((a), (b))
#ifndef OKC_UWARP
#define OKC_UWARP 1
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
#define OKC_WSNR 4
#define OKC_NOLOAD 0

namespace okc {

#ifndef OKC_L2H
#define OKC_L2H 1
#endif
#if OKC_L2H
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
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;
#if OKC_WS || OKC_WGS
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;
#define OKC_SKT 8
constexpr int SK_T = OKC_SKT;
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;
constexpr int KS = 256;
constexpr int XP_FT = KS + 8;
constexpr int NST = 3;
constexpr float TWO62 = 4.611686018427388e18f;

struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;
  const unsigned char* wAf; const unsigned char* wBf;
  const unsigned char* wAu; const unsigned char* wBu;
  const unsigned char* wAn; const unsigned char* wBn;
  const unsigned char* wAs; const unsigned char* wBs;
  __nv_bfloat16* xA; __nv_bfloat16* xB;
  int* xready;
  int* doneAs;
  int* mfl;
  uint32_t zero;
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr;
  int* doneA;
  __nv_bfloat16* inter;
  float* ybuf;
  float* ss;
  float* ss1;
  __nv_bfloat16* out;
  unsigned long long* dbg;
  float* sspart;
};

struct Smem {
  uint32_t lut[256];
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];
#else
  float red[1][1][1][RED_W];
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];
  int wsr[4];                          // WS: this CTA's phase A and phase B unit ranges
#endif
#if OKC_WGS
  uint64_t bars[24];
  int wglast[4];
  int wcp[JMAX + 1];
  int wcpB[JMAX + 1];
  int jrow[JMAX + 1];
  uint32_t wtot;
  uint32_t wtotB;
#else
  uint64_t bars[12];
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
  int njF, njS, unit;
  int gen;          // the grid barrier's generation at launch start (+1 per launch): picks this launch's ss copy
  float* ssp;       // this launch's ss copy
  int uslot[8];
  int dync[JMAX];
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];
  short pos[EMAX];
};

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
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}


__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_RLOAD && OKC_WGS
  {
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
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;
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
#if !OKC_WGS
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



#ifndef OKC_SC6S
#define OKC_SC6S 0
#endif
#ifndef OKC_SC6W
#define OKC_SC6W 0
#endif
constexpr int UNIT_CODES = 32768, UNIT_SC = OKC_SC6S ? 3072 : 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;


__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#if OKC_SC6S
#undef SCDK
#define SCDK(word, b, k4) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k4) + 0x60006000u)
#endif
#endif
#define SCK ((OKC_SC6S ? 4u : 16u))
__device__ __forceinline__ uint4 sc_load(const unsigned char* slot, int warp, int lane) {
#if OKC_SC6S
  const unsigned char* b = slot + UNIT_CODES + warp * 128 + lane * 4;
  const uint32_t w0 = *reinterpret_cast<const uint32_t*>(b), w1 = *reinterpret_cast<const uint32_t*>(b + 1024),
                 w2 = *reinterpret_cast<const uint32_t*>(b + 2048);
  uint4 r;
  r.x = w0 & 0xFCFCFCFCu; r.y = w1 & 0xFCFCFCFCu; r.z = w2 & 0xFCFCFCFCu;
  r.w = ((w0 << 2) & 0x0C0C0C0Cu) | ((w1 << 4) & 0x30303030u) | ((w2 << 6) & 0xC0C0C0C0u);
  return r;
#else
  return *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
}
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
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
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

constexpr int X1P = 512 + 8;
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
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

#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);
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
  if (nU < OKC_NS) CTA_SYNC();
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
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
      v = v0 + i;
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
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = sc_load(slot, warp, lane);
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
    CTA_SYNC();
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
      for (int jj = jdmin; jj <= jdmax; ++jj) {
        const int k = S.dync[jj];
        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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
    const uint4 sc = sc_load(slot, warp, lane);
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
    CTA_SYNC();
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

__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {
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

template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];
  uint32_t keep = 0;
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint32_t K92 = 0x6D806D80u;
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
  acc0[0][0] += __uint_as_float(keep & zero);
}

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
  constexpr int NKS = HD / KS;
  int u = blockIdx.x;
  (void)ctr;
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {
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
  int xq = 0;
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
    const int ntc = ntile;
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
      CTA_SYNC();
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
      CTA_SYNC();
      ++xq;
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
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



__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;
#endif
  if (OKC_FATSTATIC && S.njF > 0) {
    if (threadIdx.x == 0) S.preissued = 0;
    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
  const int n = min(OKC_EARLYN, v1 - v0);
  if (threadIdx.x == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
}

#if OKC_WS
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;                             // the last phase-B unit's epilogue constants (warp 0)
  uint32_t lnw0 = 0u, lnw1 = 0u;
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
    const uint4 sc = sc_load(slot, warp, lane);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
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
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer's previous unit
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);
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
    if (lane == 0) mbar_arrive1(S.wsre + bb);
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
  WS_JOIN();
}
#endif

template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
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
#define OKC_WGSNX 3
constexpr int WGS_NS = 3;
constexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage's last wgmma retired)
constexpr int WGS_NT = FT_T / 8;
constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;  // one warpgroup's weights in a stage (SC6: 8 KB codes + 768 B scales)
constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;
constexpr int WGS_XS = WGS_NT * 4096;
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;
constexpr int WGS_FIXW = WGS_NT * 512;
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
constexpr int MF_CE = 0, MF_EXIT = 64, MF_XR = 128, MF_XRS = 96, MF_DA = 384, MF_INTS = 4096;
static_assert(JMAX <= MF_XRS && MF_DA + 2 * JMAX * 16 <= MF_INTS, "MFIN flag region");
__device__ __forceinline__ int mf_gen(const Smem& S) {
#if OKC_MFQ & 32
  int v;
  asm volatile("ld.volatile.shared.s32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(&S.gen)));
  return v;
#else
  return S.gen;
#endif
}
__device__ __forceinline__ int* mf_xr(const Params& p, const Smem& S) { return p.xready + ((OKC_MFIN && (OKC_MFQ & 2)) ? mf_gen(S) * MF_XRS : 0); }
__device__ __forceinline__ int* mf_da(const Params& p, const Smem& S) { return p.doneAs + ((OKC_MFIN && (OKC_MFQ & 1)) ? mf_gen(S) * (JMAX * 16) : 0); }
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207
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
#define OKC_WGSREORD 1
#define OKC_GATHB 8
#define OKC_WGSPR 1
#define OKC_WGSDFIN 2
#define OKC_WGSFCI 4
#define OKC_WGSGS 2
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
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range's last stage
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
#ifndef P6_WARM
#define P6_WARM 1
#endif
#ifndef P6_DX
#define P6_DX 0
#endif
#ifndef P6_DXN
#define P6_DXN WGS_NX
#endif
#define P6_PRE (OKC_WGS && !FAST1_ON)
__device__ void p6_ranges(const int* jC, int nj, int* wcp, int* wcpB, uint32_t* wt, int* wgr);
static_assert(__builtin_offsetof(Smem, wtotB) == __builtin_offsetof(Smem, wtot) + 4, "p6_ranges: wt[0..1] = wtot, wtotB");
__device__ __forceinline__ void wgs_ranges(const Params& p, Smem& S) {
  p6_ranges(S.jC, S.njF + S.njS, S.wcp, S.wcpB, &S.wtot, S.wgr);
}
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
  if ((threadIdx.x & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      if (n >= 2 && p.M <= 64) { const int pn = n - 2; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
constexpr int WGS_XROWS = EMAX + 8 * JMAX;
#if P6_DX
// 32 lanes x 8 rows x 16 B), as keell's gather does into xA; pad rows are zeroed. One cp.async group.
__device__ __forceinline__ int p6_dx_issue(const Params& p, const Smem& S, unsigned char* ring) {
  const int g0 = S.wgr[0], g1 = S.wgr[1], sp = wgs_split(0, g0, g1);
  const int nd = min(P6_DXN, g1 - g0);
  const int G = threadIdx.x & 31, o = 8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1);
  const uint32_t x0 = smem_u32(ring + WGS_NS * UNIT_BYTES) + (uint32_t)G * 128u;
#pragma unroll 1
  for (int i = 0, gs = sp; i < nd; ++i, ++gs) {
    if (gs == g1) gs = g0;
    const int st = gs & 15, j = gs >> 9;
    const int cnt = S.jC[j], npad = ((cnt + 7) >> 3) << 3, js = S.jS[j];
    const uint32_t xb = x0 + (uint32_t)i * (uint32_t)WGS_XS;
#pragma unroll 1
    for (int r = 0; r < npad; ++r) {
      const uint32_t d = xb + (uint32_t)(r >> 3) * 4096u + (uint32_t)(r & 7) * 16u;
      if (r < cnt) {
        const unsigned* s = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[js + r] >> 3) * HD + st * 256) + o;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(s) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d + 4), "l"(s + 32) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d + 8), "l"(s + 64) : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d + 12), "l"(s + 96) : "memory");
      } else {
        asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};\n" ::"r"(d), "r"(0) : "memory");
      }
    }
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  return nd;
}
__device__ __forceinline__ void p6_dx_done(Smem& S, int nd) {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncwarp();
  if ((threadIdx.x & 31) == 0)
    for (int i = 0; i < nd; ++i) mbar_arrive1(S.bars + BX_F + i);
}
#endif
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  wgs_ranges(p, S);
  __syncwarp();
#if P6_DX
  const int nd = p6_dx_issue(p, S, ring);
#else
  constexpr int nd = 0;
#endif
  WS_JOIN();  // routing tables and this CTA's stage ranges are ready
#if P6_DX
  if (nd > 0) p6_dx_done(S, nd);
#endif
  if ((threadIdx.x & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, xrj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
      if (n < nd) { if (++xs == WGS_NX) { xs = 0; xph ^= 1; } continue; }
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
      }
      if (!pb && j != xrj && !(OKC_WGSX & 8)) { xrj = j; { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); } }
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = threadIdx.x, nj = S.njF + S.njS;
  const int R = S.jrow[nj];
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
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
          {
            const int G = tid & 31;
            const unsigned* xs32 = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + (tid >> 5) * 256)
                                   + (8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1));
            v[k] = make_uint4(__ldg(xs32), __ldg(xs32 + 32), __ldg(xs32 + 64), __ldg(xs32 + 96));
          }
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
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
#if !OKC_B4FIN
#define OKC_YBF 1
#else
#define OKC_YBF 0
#endif
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
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
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
#if OKC_YBF
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0), __float2bfloat16(d4[h] * w0));
        st_keep_bf16(reinterpret_cast<__nv_bfloat16*>(p.ybuf) + ((size_t)(jst + tk) * HD + n0 + 8), __float2bfloat16(d4[2 + h] * w1));
#else
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
#endif
      }
    }
#if OKC_B4FIN
    {  // bet4_claude: this tile's 16-column sum of squares per token, reduced over the 8 column lanes (fixed tree)
      float q0 = 0.f, q1 = 0.f;
      if (tok0 < cnt) { const float a = d4[0] * w0, b = d4[2] * w1; q0 = a * a + b * b; }
      if (tok0 + 1 < cnt) { const float a = d4[1] * w0, b = d4[3] * w1; q1 = a * a + b * b; }
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        q0 += __shfl_xor_sync(0xffffffffu, q0, o);
        q1 += __shfl_xor_sync(0xffffffffu, q1, o);
      }
      if (g == 0) {
        if (tok0 < cnt) p.sspart[(size_t)(jst + tok0) * 256 + r16] = q0;
        if (tok0 + 1 < cnt) p.sspart[(size_t)(jst + tok0 + 1) * 256 + r16] = q1;
      }
    }
#endif
  }
}
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
#if OKC_MFIN && (OKC_MFP & 1)
  if (pd.kind == 4) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + ((OKC_MFP & 8) ? pd.slot : (int)(mf_gen(S) * 32 + S.jE[j]))), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
#endif
  if (pd.kind == 1) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;
  if constexpr (DF) {
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
#if OKC_MFIN && (OKC_MFP & 2)
  if (PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), "r"(1) : "memory");
  }
#endif
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = (OKC_SC6W ? 4u : 16u) + p.zero;  // opaque 16 (4 with SC6W): the scale decode's multiply stays an IMAD (FMA pipe)
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
  int prev = -1;
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
#if OKC_SC6W
    uint32_t sgs[4];
    {
      const int tt = wp * 32 + lane;
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + tt * 4);
      const uint32_t H = static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(pc + 8192 + 512 + tt * 2));
      const uint32_t X = W & 0xFCFCFCFCu, Hm = H & 0xFCFCu;
      const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;
      const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;
      sgs[0] = __byte_perm(X, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[1] = __byte_perm(X, 0u, 0x4342u) * k16c + 0x60006000u;
      sgs[2] = __byte_perm(Hm, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[3] = __byte_perm(p6, p7, 0x0703u) * k16c + 0x60006000u;
    }
#else
    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);
#endif
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
#if !OKC_SC6W
      const uint2 scg = scs;
#endif
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
#if OKC_SC6W
          const uint32_t sgp = sgs[js];
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
  if (slot >= 0) {
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};
#if OKC_MFIN && (OKC_MFP & 1)
  else pd = WgsPend{4, u, (OKC_MFP & 8) ? (int)(mf_gen(S) * 32 + S.jE[j]) : -1, 0};
#endif
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
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
#define OKC_WGSU 160
#define OKC_WGSUB 140
__device__ __forceinline__ int wgs_unitc(int pb) { return pb ? OKC_WGSUB : OKC_WGSU; }
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;
    accB += ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = (uint32_t)wgs_unitc(pb), cu = 16u * cs + U;
  const uint32_t r = x - (uint32_t)pre[lo], un = r / cu, rem = r - un * cu;
  const uint32_t st = rem < U ? 0u : min(15u, (rem - U) / cs);
  return (lo << (9 - pb)) + (int)(un * 16u + st);
}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
}
__device__ __noinline__ void p6_ranges(const int* jC, int nj, int* wcp, int* wcpB, uint32_t* wt, int* wgr) {
  const int lane = threadIdx.x & 31;
  const int G = gridDim.x, c = blockIdx.x;
  uint32_t carryA = 0, carryB = 0;
  for (int j0 = 0; j0 < nj; j0 += 32) {
    const int j = j0 + lane;
    uint32_t ia = 0, ib = 0;
    if (j < nj) {
      ia = ((uint32_t)wgs_cost(0, jC[j]) * 16u + OKC_WGSU) << 5;
      ib = ((uint32_t)wgs_cost(1, jC[j]) * 16u + OKC_WGSUB) << 4;
    }
    uint32_t sa = ia, sb = ib;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t va = __shfl_up_sync(0xffffffffu, sa, d), vb = __shfl_up_sync(0xffffffffu, sb, d);
      if (lane >= d) { sa += va; sb += vb; }
    }
    if (j < nj) { wcp[j] = (int)(carryA + sa - ia); wcpB[j] = (int)(carryB + sb - ib); }
    carryA += __shfl_sync(0xffffffffu, sa, 31);
    carryB += __shfl_sync(0xffffffffu, sb, 31);
  }
  if (lane == 0) { wcp[nj] = (int)carryA; wcpB[nj] = (int)carryB; wt[0] = carryA; wt[1] = carryB; }
  __syncwarp();
  const int l4 = lane & 3, pbl = l4 >> 1, kk = l4 & 1;
  const uint32_t tot = pbl ? carryB : carryA;
  int res;
  if (kk == 1 && c == G - 1) {
    res = nj << (9 - pbl);
  } else {
    const uint32_t x = tot * (uint32_t)(c + kk) / (uint32_t)G;
    const int* pre = pbl ? wcpB : wcp;
    int lo = 0, hi = nj;
    while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
    const uint32_t cs = (uint32_t)wgs_cost(pbl, jC[lo]), U = (uint32_t)wgs_unitc(pbl), cu = 16u * cs + U;
    const uint32_t r = x - (uint32_t)pre[lo], un = r / cu, rem = r - un * cu;
    const uint32_t st = rem < U ? 0u : min(15u, (rem - U) / cs);
    res = (lo << (9 - pbl)) + (int)(un * 16u + st);
  }
  if (lane < 4) wgr[lane] = res;
  __syncwarp();
}
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t sj = (uint32_t)(s & ((1 << (9 - pb)) - 1)), cs = (uint32_t)wgs_cost(pb, S.jC[j]), U = (uint32_t)wgs_unitc(pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (sj >> 4) * (16u * cs + U) + U + ((sj & 15u) + 1u) * cs;
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
#if OKC_TINY
  if (nF > 0) __trap();
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
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

__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;
#endif
}
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
  const int tid = threadIdx.x;
  DBG(0);
#if P6_PRE
  if (tid == 0) {
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.preissued = 0;
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
  }
#if P6_WARM
  if (tid >= NT + 32) {
    if ((tid & 31) == 0) { S.fcnt[0] = 8; S.fcnt[1] = 8; }
    __syncwarp();
    p6_ranges(S.fcnt, 2, S.fcnt + 2, S.fcnt + 5, reinterpret_cast<uint32_t*>(S.fcnt + 8), S.fcnt + 12);
  }
#endif
#endif
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
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
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
      const int jj = (v01 + i) >> RB_A_LOG;
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
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
    for (int i = ln + 64; i < E8; i += 32) {
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
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);
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
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {
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
#if !FAST1_ON && !P6_PRE
#if OKC_WGS
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.preissued = 0;
#endif
#if !P6_PRE
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
#endif
#if OKC_MFIN && (OKC_MFQ & 8)
    const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);
#endif
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
#if OKC_MFIN && OKC_WGS && (OKC_MFQ & 8)
  {  // the other parity's counters and hand-off flags belong to the next launch (the previous launch is over): zero them
    const int o = 1 - S.gen;
    for (int i = blockIdx.x * NT + tid; i < 32 + MF_XRS + JMAX * 16; i += gridDim.x * NT) {
      if (i < 32) p.mfl[MF_CE + o * 32 + i] = 0;
      else if (i < 32 + MF_XRS) p.mfl[MF_XR + o * MF_XRS + (i - 32)] = 0;
      else p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
    }
  }
#else
  {  // the previous launch's ss copy and barrier counter are the next launch's: zero them now (this CTA's share)
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#endif
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
  if (tid < 32) {
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
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
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
#if !(OKC_MFIN && OKC_WGS && (OKC_MFQ & 4))
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
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_B4FIN
  {
    static_assert(NT == 512 && NW == 16, "B4FIN: 16 warps = 8 routes x 2 halves, 512 threads x 2 columns");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    constexpr int SEG = 1024, NSEG = HD / SEG;
    for (int it = blockIdx.x; it < p.M * NSEG; it += gridDim.x) {
      const int tk = it / NSEG, col = (it % NSEG) * SEG + tid * 2;
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {  // the token's local experts are complete when all their jobs' units were counted by all 4 warpgroups
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
#endif
#if B4_EARLY_Y
      float2 yv[8];  // A2: the 8 routes' y loads go out before the ss partial loads and the reduction
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        yv[sl] = q >= 0 ? __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col)) : make_float2(0.f, 0.f);
      }
#endif
      {
        const int sl = warp >> 1, hh = warp & 1;
        const int q = S.pos[tk * 8 + sl];
        float v = 0.f;
        if (q >= 0) {
          const float4 a = __ldcg(reinterpret_cast<const float4*>(p.sspart + (size_t)q * 256 + hh * 128 + lane * 4));
          v = (a.x + a.y) + (a.z + a.w);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp] = v;
      }
      CTA_SYNC();
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        if (q >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);
#if B4_EARLY_Y
          const float2 v = yv[sl];
#else
          const float2 v = __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col));
#endif
          acc.x += r * v.x; acc.y += r * v.y;
        }
      }
      *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);
      CTA_SYNC();
    }
  }
#elif OKC_WGS && OKC_WGSFINSS
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {  // the token's local experts are complete when all their jobs' units were counted by all 4 warpgroups
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
      if (tk == blockIdx.x) DBG(4);
#endif
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
#if OKC_YBF
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
#else
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
#endif
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
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
#if OKC_YBF
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
#else
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
#endif
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
      CTA_SYNC();
    }
  }
#elif OKC_FINCOL
  constexpr int FSEG = HD / (NT * 4);
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {
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
#endif
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
  DBG(5);
}

}

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

static int g_okc_pdl = 0;

void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }

std::vector<int64_t> okc_attrs() {
  cudaFuncAttributes a0, a1;
  cudaFuncGetAttributes(&a0, okc::moe_fused<false>);
  cudaFuncGetAttributes(&a1, okc::moe_fused<true>);
  return {a0.numRegs, (int64_t)a0.localSizeBytes, (int64_t)a0.sharedSizeBytes, a1.numRegs, (int64_t)a1.localSizeBytes, (int64_t)a1.sharedSizeBytes};
}

void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }
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
#if OKC_WGS
  p.mfl = ib + 8 + 1024 + okc::EMAX;
#else
  p.mfl = nullptr;
#endif
  p.ybuf = p.ss + okc::EMAX;
#if OKC_WGS && OKC_B4FIN
  TORCH_CHECK(x.size(0) <= 32 && x.size(0) * 8 <= okc::EMAX - 64, "B4FIN build is for M <= 32");
  p.sspart = p.ybuf + (size_t)(okc::EMAX - 64) * okc::HD;
#else
  p.sspart = nullptr;
#endif
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
#if OKC_MFIN
  p.xready = p.mfl + okc::MF_XR;
#endif
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#if OKC_MFIN
  p.doneAs = p.mfl + okc::MF_DA;
#endif
  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + okc::MF_INTS), "MFIN needs the flag region in scratch[0]");
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


"""
MOE_CPP_SRC = r"""
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
std::vector<int64_t> okc_attrs();
"""
_MOE = None
_MOE_B4 = None


_MOE_TINY = None
_MOE_TINY2 = None
_MOE_MID = None
MOE_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")              # the moe agent's MAIN_DEFINES
MOE_TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2")    # their small-M build ( () = absent )
MOE_TINY_MAX_M = 4
MOE_TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')  # their second small-M build ( () = absent )
MOE_TINY2_MAX_M = 8
MOE_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')      # their mid-M build ( () = absent )
MOE_MID_MAX_M = 16


_MA_PRMT_SCALES = True
_MA_SFX_IF_PRMT = ("-DOKC_PRMTSC=1",)


def _ma_scales_prmt_exact(*sfs):
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x08) | (b > 0x7E)).any()):
            return False
    return True


MOE_HAS_PDL = True   # their extension exports okc_set_pdl (programmatic stream serialization)


SC6_WGS = True      # lossless 6-bit block scales in the WGS stages (moe-scale SC6, harness/make_sc6.py)
_MA_SC6W = False    # set in prepare(): scale bytes all in 0x40..0x7F and the PRMT decode and a WGS build
_MA_SFX = ()   # set in prepare() from the weights' block-scale bytes, before any moe build starts


_MA_EXTS = {}
_MA_ALT_SRCS = {}
_MA_MAIN_NS = 3
_MA_PDL = False   # set per call (okc_set_pdl)


def _ma_ext(defines=None):
    defines = tuple(MOE_DEFINES if defines is None else defines)
    if _MA_SC6W and "-DOKC_WGS=1" in defines and "-DOKC_SC6W=1" not in defines and "-DOKC_SC6OFF=1" not in defines:
        defines = defines + ("-DOKC_SC6W=1",)
    if DQ7 and "-DOKC_DQ7=1" not in defines:
        defines = defines + ("-DOKC_DQ7=1",)
    if defines not in _MA_EXTS:
        src = MOE_CUDA_SRC
        defs = [d for d in defines if not d.startswith("@")]
        for d in defines:
            if d.startswith("@"):
                src = _MA_ALT_SRCS[d[1:]]
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}" + ("a" if major == 9 else "")
        tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
        import hashlib as _native_build_hash
        source_tag = _native_build_hash.md5((MOE_CPP_SRC + src + repr((defines, _MA_MAIN_NS))).encode()).hexdigest()[:16]
        _MA_EXTS[defines] = load_inline(
            name=f"glue_moe_{source_tag}_sm{arch}",
            cpp_sources=MOE_CPP_SRC,
            cuda_sources=src,
            functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={_MA_MAIN_NS if major >= 9 else 1}"]),
                               f"-DOKC_FNST={4 if major >= 9 else 3}",
                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
            verbose=False,
        )
        _MA_EXTS[defines].okc_set_pdl(1 if _MA_PDL else 0)
    return _MA_EXTS[defines]


def _moe_build(defines):
    """Their own extension builder (lifted verbatim as _ma_ext: flags, DQ7 define, module name), called with the
    build's defines + the weights' suffix, exactly as their kernel_fn calls _ext."""
    ext = _ma_ext(tuple(defines) + tuple(_MA_SFX))
    if MOE_HAS_PDL:
        ext.okc_set_pdl(0)
    return ext


def _moe(M=None):
    """Their kernel, in the build their kernel_fn would pick for this token count."""
    global _MOE, _MOE_TINY, _MOE_TINY2, _MOE_MID, _MOE_B4
    if M is not None and MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        if _MOE_TINY is None:
            _MOE_TINY = _moe_build(MOE_TINY_DEFINES)
        return _MOE_TINY
    if M is not None and _moe_tiny2_ok() and M <= MOE_TINY2_MAX_M:
        if _MOE_TINY2 is None:
            _MOE_TINY2 = _moe_build(MOE_TINY2_DEFINES)
        return _MOE_TINY2
    if M is not None and MOE_MID_DEFINES and M <= MOE_MID_MAX_M:
        if _MOE_MID is None:
            _MOE_MID = _moe_build(MOE_MID_DEFINES)
        return _MOE_MID
    if (M is not None and 16 < M <= 32 and "_main_defines" in globals()
            and tuple(_main_defines(_MA_SFX)) == tuple(MOE_DEFINES)):  # bet4_claude: decode 32 gets the B4FIN WGS build
        if _MOE_B4 is None:
            _MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",))
        return _MOE_B4
    if _MOE is None:   # their M >= 64 rule: _main_defines(sfx) (the WGS build needs the PRMT decode, else fallback)
        _MOE = _moe_build(_main_defines(_MA_SFX) if "_main_defines" in globals() else MOE_DEFINES)
    return _MOE


def _moe_mid():
    global _MOE_MID
    if MOE_MID_DEFINES and _MOE_MID is None:
        _MOE_MID = _moe_build(MOE_MID_DEFINES)
    return _MOE_MID


def _moe_tiny2_ok():
    """Their kernel_fn's rule: the TINY2 build only with the PRMT scale decode when it is a WGS build."""
    return bool(MOE_TINY2_DEFINES) and (bool(_MA_SFX) or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES))


def _moe_tiny2():
    global _MOE_TINY2
    if _moe_tiny2_ok() and _MOE_TINY2 is None:
        _MOE_TINY2 = _moe_build(MOE_TINY2_DEFINES)
    return _MOE_TINY2


def _moe_tiny():
    global _MOE_TINY
    if MOE_TINY_DEFINES and _MOE_TINY is None:
        _MOE_TINY = _moe_build(MOE_TINY_DEFINES)
    return _MOE_TINY


_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


DQ7 = True


def _stored_pos():
    pos = {}
    for R in range(4):
        for up in range(2):
            outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}  # code bit -> output bit
            for b, o in outb.items():
                if DQ7:
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
    assert sorted(pos.values()) == list(range(32))
    return pos


_POS = _stored_pos()


def _pack_codes(codes, rows):
    """codes uint8 [R0, K/2] (one expert), rows: LongTensor row order (R). Returns int32 words
    laid out [rb][w][c][g][t][s] (uint4 per lane per chunk)."""
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)  # rb, h, g, w, c, t, s, j
    n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)  # rb, w, c, g, t, s, h, j
    word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
    for Rg, (h, j0) in enumerate(_REG_HJ):
        for up in range(2):
            code = n[..., h, j0 + up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
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


def _sc6_ok(*sfs):
    """SC6: every scale byte is in 0x40..0x7F (bits 7:6 == 01), so byte - 0x40 is a lossless 6-bit value."""
    for sf in sfs:
        b = sf.contiguous().view(torch.uint8)
        if bool(((b < 0x40) | (b > 0x7F)).any()):
            return False
    return True




def _sc6_wgs(s):
    """SC6 WGS piece scales: [..., 1024] bytes (thread t = 8 bytes s0..s7 = [j][h]) -> [..., 768]: plane W (t * 4 B: bytes 4 v_n | piece_n,
    n = 0..3) then plane H (t * 2 B: bytes 4 v_(4+n) | piece_(4+n)); v6 = pieces 0..2, v7 = pieces 3..5 (see harness/make_sc6.py)."""
    lead = s.shape[:-1]
    v = s.view(*lead, 128, 8).to(torch.int32) - 0x40
    assert int(v.min()) >= 0 and int(v.max()) <= 63
    v6, v7 = v[..., 6], v[..., 7]
    piece = [(v6 >> 0) & 3, (v6 >> 2) & 3, (v6 >> 4) & 3, v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3]
    wb = torch.stack([(v[..., n] << 2) | piece[n] for n in range(4)], -1).to(torch.uint8)
    hb = torch.stack([(v[..., 4 + n] << 2) | piece[4 + n] for n in range(2)], -1).to(torch.uint8)
    return torch.cat([wb.reshape(*lead, 512), hb.reshape(*lead, 256)], -1).contiguous()



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
    c = codes.index_select(0, rows)
    R = c.shape[0]
    nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)  # [R, K]
    nb = R // 16
    n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)  # rb, h, g, w, c, s, q, t, e
    n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)  # rb, w, c, g, t, s, h, q, e
    word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
    for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for up in range(2):
            code = n[..., h, q, up]
            for b in range(4):
                word |= ((code >> b) & 1) << _POS[(Rg, up, b)]
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.contiguous().view(-1)


def _kperm_src():
    """KPERM3: stage position p -> stage column k. p = 16 s + c (k16 step s = 4 lg + j, fragment column c of lane
    t = (c >> 1) & 3, i = (c & 1) | ((c >> 3) << 1)); k = 16 (4 t + j) + 4 lg + i, so lane t's columns of step j in all
    four load groups are exactly scale block 4 t + j."""
    p = torch.arange(256)
    s, c = p >> 4, p & 15
    lg, j, t, i = s >> 2, s & 3, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
    k = 16 * (4 * t + j) + 4 * lg + i
    assert sorted(k.tolist()) == list(range(256))
    return k


def _kperm_codes(codes):
    """codes uint8 [R, K/2] (low nibble first) -> the same codes with K permuted inside every 256-K stage."""
    R, K = codes.shape[0], codes.shape[1] * 2
    nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 256, 256)
    nib = nib.index_select(2, _kperm_src().to(codes.device)).reshape(R, K // 2, 2)
    return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()


def _kperm_rows(rowsA):
    """Gate/up row order whose 16-row block r16, lane g holds inter column colmap(P), P = 8 r16 + g, where
    colmap(P) = 256 (P >> 8) + _kperm_src()[P & 255] is the column phase B reads at K position P (the phase-A
    epilogue then stores at its usual position)."""
    r = rowsA.view(-1, 2, 8)
    gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
    P = torch.arange(gate.numel(), device=rowsA.device)
    cm = (P >> 8) * 256 + _kperm_src().to(rowsA.device)[P & 255]
    return torch.stack((gate.index_select(0, cm).view(-1, 8), up.index_select(0, cm).view(-1, 8)), 1).reshape(-1)


def _wgs_copy(codes, sf, rows, sc6=False):
    """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout)."""
    wn = _pack_codes_nat(_kperm_codes(codes), rows)
    R = rows.numel()
    nb, nub = R // 16, R // 256
    w = wn.view(nb, 8, 8, 8, 4, 4)                                    # rb16, w8, c8, g, t, s
    w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)   # rb16, k16 step (w8, c8, s), lane (g, t)
    w = w.view(nub, 4, 4, 16, 4, 4, 32)                               # ub, q, warp, st, grp, j, lane
    w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()                   # ub, st, q, grp, warp, lane, j
    cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
    s = sf.view(torch.uint8).index_select(0, rows)                    # [R, 256]: one byte per row per 16 K
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)                             # ub, q, warp, h, g, st, t, j (block 16 st + 4 t + j)
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, warp, g, t, j, h (KPERM3)
    if sc6:
        s = _sc6_wgs(s)
    return torch.cat([cb, s], dim=3).contiguous().view(-1)


MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7", "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0", "-DOKC_FPF=5")


def _main_defines(sfx):
    if not sfx and "-DOKC_WGS=1" in tuple(MOE_DEFINES):
        return MAIN_FALLBACK_DEFINES
    return MOE_DEFINES


def _wgs_on():
    return any("-DOKC_WGS=1" in tuple(d) for d in (MOE_DEFINES, MOE_TINY_DEFINES, MOE_TINY2_DEFINES, MOE_MID_DEFINES))


bfp__SK2_DELAY = 0  # debug only: delay (ns) of escape warp 0 + ring spin-limit trap


bfp_CUDA_SRC = r"""
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>


__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

@@MBAR@@
__device__ __noinline__ void okc_spin_trap() { asm volatile("trap;"); }
__device__ __forceinline__ void okc_spin_ge(const int* p, int target, int ns) {
  for (unsigned it = 0;; ++it) {
    if (*((volatile const int*)p) - target >= 0) return;
    if (ns > 0) __nanosleep(ns);
    if (it >= (1u << 24)) okc_spin_trap();
  }
}
@@BODY@@
"""

bfp_MBAR_HELPERS = r"""
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(l2_pol_ef()) : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}

"""

bfp_WG_HELPERS = r"""
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }


__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}


"""


def bfp_wgmma_defs(bns):
    """Generate wgmma_call overloads for m64nBNk16 bf16 with A from registers."""
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        a0 = nacc
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        pi = a0 + 5
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{pi}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}""")
    return "\n".join(out)


bfp_TMA_SRC = r"""
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);
  const int nst = NS / KS;
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;

  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;
  __syncthreads();

  if (warp == NW) {
    const int st0 = NSTAGE < nst ? NSTAGE : nst;
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
    }
    return;
  }

  const int rg = rb * RG + warp % RG;
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];

  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;
  }

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;  // x part holding this stage's last k
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
        if constexpr (WGM) fence_async_smem();
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    constexpr int NL = CHUNK / 512;
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (dg == 3) {
      if (wr[0].x == 0x9e3779b9u && wr[NL - 1].w == 0x7f4a7c15u) acc[0][0] += 1.f;
      continue;
    }
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;   // local k of this unit's first k-step (multiple of 64)
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }

  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  if (dg >= 1) {
    if (acc[0][0] == 1.2345e-30f) out[threadIdx.x] = cvt_out<OutT>(acc[0][1]);
    return;
  }
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + rb, target, 20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }
}

template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""

bfp_MAP_SRC = r"""
#include <cudaTypedefs.h>
#include <cstring>
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(l2_pol_el()) : "memory");
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

"""

bfp_GEMM_SRC = r"""

@@WGMMA_SS_DEFS@@

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();

  if (warp >= NWG * 4) {
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
        if constexpr (AS)
          bulk_g2s(sbase + slot * ST_BYTES, Wst + ((size_t)rb * nkb_all + kb0 + i) * A_BYTES, A_BYTES, full0 + 8 * slot);
        else
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }

  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;

  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    okc_spin_ge(counters + rb, target, 32);
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}

template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


bfp_SK2_SRC = r"""

#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
static int g_sk2_pdl = 0;
void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }
#define SK2_DIAG 0
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK, bool FUSE>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,
           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {
  const int dgv = SK2_DIAG ? dg : 0;
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;
  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);
  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;
  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;
  if (trace && threadIdx.x == 0) tr[0] = t_start;
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
  constexpr int NSTR = FUSE ? 4 : 0;
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");
  const int rt = cta >> 2, rq = cta & 3;
  const uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;
  const int nstt = nst + NSTR;
  const long long pcx_d0 = (long long)((pdl >> 4) & 31) * 448, pcx_g = (long long)((pdl >> 9) & 31) * 448;
  const int pcx_e = (pdl >> 14) & 7;
  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL == 1)   // PDL: the first P stages' weights go out before the wait; their barriers first
      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    const long long pcx_t0 = clock64();
    for (int st = 0; st < P; ++st) {
      const long long pcx_tg = st < pcx_e ? 0 : pcx_d0 + (long long)(st - pcx_e) * pcx_g;
      if (pcx_tg > 0) { while (clock64() - pcx_t0 < pcx_tg) { } }
      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
    }
    if constexpr (CL == 1) {
      if (COMP) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) {
        if (i >= P) mbar_init(full0 + 8 * i, 1);
        mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));
      }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    } else {
      if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
      if (!(dgv & 4))
        for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
      if (COMP) {
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here
  if constexpr (CL == 1) {
    if (tid == 0 && !(dgv & 4))
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
  }
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    if (lane == 0) {
      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
        } else {   // router piece: global router stage rq * 4 + j of tile rt
          const int gsr = rq * NSTR + (st - nst);
          mbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);
          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);
        }
      }
      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // last weight stage issued
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      int pe = 1;
      for (int st = 0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (FUSE && st >= nst) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();
    const uint32_t s0 = sbase + slot * SST;
    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      if (dgv & 8) {
        const int nh = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
          }
        continue;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();
  }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue
  if constexpr (FUSE) {
    // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators
    float accr[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
    for (int st = nst; st < nstt; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const uint32_t s0 = sbase + slot * SST;
      const int ui = wk;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;
      uint32_t bh[8], bo[8];
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3]) : "r"(a0));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7]) : "r"(a0 + 16));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3]) : "r"(a0 + 1024));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7]) : "r"(a0 + 1040));
      uint32_t a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4((PK && trow[mt] >= M) ? zrow
                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
        }
    }
    if constexpr (COMP) mbar_wait(cbar, 0);
    named_bar(1, NC);
    float* redr = reinterpret_cast<float*>(gbase);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = 2 * t4, tok = mt * 16 + g;
      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
    }
    named_bar(1, NC);
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
    for (int i = tid; i < TOKS * 8; i += NC) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
      __stcg(pp + i, v);
    }
    named_bar(1, NC);
    __shared__ int s_rlast;
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");
      s_rlast = (old & 3) == 3;
      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (s_rlast) {
      const float* pr = part + (size_t)rt * 4 * 32 * 8;
      for (int i = tid; i < M * 8; i += NC) {
        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);
        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();

  if (dgv & 1) {
    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);
    return;
  }
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    if (trace && tid == 0) tr[15] = gtime();
    return;
  }
  named_bar(1, NC);
  if (trace && tid == 0) tr[9] = gtime();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      if (trace && mt == 0 && nt == 0) {
        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
        if (tid == 0) tr[12] = gtime();
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  if (trace && tid == 0) tr[11] = gtime();
  named_bar(1, NC);
  if (trace && tid == 0) tr[10] = gtime();
  if (trace && tid == 0) tr[13] = gtime();
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  if (trace && tid == 0) tr[14] = gtime();
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
  if (trace && tid == 0) tr[15] = gtime();
  if ((dgv & 32) && dbg != nullptr && tid == 0) {
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();
  }
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg,
                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    const int pdl = g_sk2_pdl;
    if (CL > 1 || pdl > 0) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[2];
      int na = 0;
      if (CL > 1) {
        at_[na].id = cudaLaunchAttributeClusterDimension;
        at_[na].val.clusterDim.x = CL;
        at_[na].val.clusterDim.y = 1;
        at_[na].val.clusterDim.z = 1;
        ++na;
      }
      if (pdl > 0) {
        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at_[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
      }
      lc.attrs = at_;
      lc.numAttrs = na;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg,
                                        wsr, outr, part, rcnt, pdl));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg,
                                                                               wsr, outr, part, rcnt, 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2f(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
          int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr) {
  @@SK2F_DISPATCH@@
  TORCH_CHECK(false, "sk2f: no config");
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""


bfp_WIDE_SRC = r"""
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;

  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }

  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
  };
  const bool producer = tid == 0;
  if (producer) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }

  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t* af = afrag[h];
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                       : "r"(wa + h * 2048 + j * 512));
#pragma unroll
          for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
    }
    }
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }

  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    const int old = atomicAdd(counters + zh * nrb + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + zh * nrb + rb, target, 32);
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  __threadfence();
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}

template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""


def bfp_wgmma_ss_defs(bns):
    out = []
    for bn in bns:
        nacc = bn // 2
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
        a, b, p = nacc, nacc + 1, nacc + 2
        out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{p}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}""")
    return "\n".join(out)


bfp_CPP_SRC = """
void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void sk2f(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,
          int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr);
void sk2_set_pdl(int64_t n);
"""

bfp_w_rows = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
bfp_COMPRESS = {"qkv_gate", "o", "pooler", "unpooler"}

bfp__ext = None


bfp_TMA_CONFIGS = {  # (layer, mt) -> (splits, RG, WS, KS); x chunk M*Kc*2 small, ~128 CTAs; mt 0 = m == 1
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
bfp_USE_WGM = True   # m >= 9 compressed configs with RG % 4 == 0 use RS-wgmma (sm_90)
bfp_WGM_SM80_TEST = False  # local check of the wgmma path's data layout (mma.sync emulation on sm_80)


def bfp__tma_nstage(nw, comp, sm90, ks, hl=False):
    chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
    budget = (112 if sm90 else 40) * 1024
    return max(2, min(8, budget // (nw * chunk)))


bfp_TMA_TU = {}  # (layer, mtk) -> 0 / 1 (which translation unit instantiates it)


bfp_TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def bfp__tma_cfg_items():
    items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(bfp_TMA_CONFIGS.items())]
    items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(bfp_TMA_ALT.items())]
    return items


def bfp__tma_srcs(sm90):
    lines = {}
    for key, (splits, rg, wsn, ks) in bfp__tma_cfg_items():
        if key[0] == "alt":
            layer, mtk = key[1], bfp__mt_key(key[2])
        else:
            layer, mtk = key
        mt = max(mtk, 1)
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        comp = 0 if f32 else 1
        hl = f32
        ns = bfp__tma_nstage(rg * wsn, comp, sm90, ks, hl)
        if f32 and mt >= 4:
            ns = min(ns, 3)  # x chunk (32 x 1024) + ring must fit in shared memory
        n_cfg = 1024 if (layer == "unpooler" and mtk <= 1) else bfp_w_rows[layer]
        wgm = bool((sm90 or bfp_WGM_SM80_TEST) and comp and bfp_USE_WGM and rg % 4 == 0 and mtk >= 2)
        line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
                f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
                f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
        lines.setdefault(line, []).append(key)
    keys = list(lines)
    for ln in keys:
        for cfg in lines[ln]:
            bfp_TMA_TU[cfg] = 0
    return bfp_TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(keys)).replace("@@TMA_NAME@@", "skinny_tma_a")


bfp_GEMM_CONFIGS = {  # (layer, bn) -> (nwg, splits, stages); grid = (N / (64 nwg), splits) <= 132 (co-resident)
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


bfp_DIAG_LAYERS = ()
bfp_GEMM_DIAG = False   # diagnostic: also time gemm90 variants (TMA-2D A vs contiguous pre-swizzled A) on m 65..256


def bfp_swizzle_tiles(w, bm):
    """W [N, K] -> [N/bm][K/64][bm rows x 128 B] with the 128B swizzle (16-byte chunk c of row r at c ^ (r % 8))."""
    n, k = w.shape
    t = w.view(n // bm, bm, k // 64, 8, 8).permute(0, 2, 1, 3, 4)  # [rb, kb, r, c, 8]
    r = torch.arange(bm, device=w.device).view(bm, 1)
    p = torch.arange(8, device=w.device).view(1, 8)
    src_c = (p ^ (r % 8)).view(1, 1, bm, 8, 1).expand(t.shape[0], t.shape[1], bm, 8, 8)
    return torch.gather(t, 3, src_c).contiguous()


def bfp__gemm_src(sm90):
    lines = []
    for (layer, bn), (nwg, splits, stages) in sorted(bfp_GEMM_CONFIGS.items()):
        f32 = layer == "router"
        ot = "float" if f32 else "__nv_bfloat16"
        for a_s in ((0, 1) if (not f32 and bfp_GEMM_DIAG) else (0,)):
            lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                         f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
    bns = sorted({bn for (_, bn) in bfp_GEMM_CONFIGS})
    return (bfp_GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", bfp_wgmma_ss_defs(bns))
            .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


bfp_GQ_SRC = r"""
template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);

  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);   // split epilogue: the peer's half of our rows
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();   // the previous stage's group is done: hand its slot back
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");

  if constexpr (CL <= 2) {
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // the peer's qbar is initialized
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;   // this warp's first kept row
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
  if (CL > 1 && crank > 0) {
    float* mine = reinterpret_cast<float*>(gbase);
    if constexpr (NWG > 1) named_bar(1, NWG * 128);
#pragma unroll
    for (int q = 0; q < NACC; ++q) mine[(wg * NACC + q) * 128 + t] = acc[q];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar(1, NWG * 128);
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    if (tid == 0) {
      const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * NWG * NACC * 128);
      uint32_t peer_r, qbar_r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(peer_r), "r"((uint32_t)__cvta_generic_to_shared(mine)), "r"(NWG * NACC * 128 * 4), "r"(qbar_r)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    return;
  }
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
#pragma unroll
    for (int r = 0; r < CL - 1; ++r)
#pragma unroll
      for (int q = 0; q < NACC; ++q) acc[q] += peer[((r * NWG + wg) * NACC + q) * 128 + t];
  }
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + wg * 64 + (warp & 3) * 16;
#pragma unroll
  for (int j = 0; j < BN / 8; ++j) {
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(acc[4 * j]);
      out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(acc[4 * j + 1]);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
}

template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
    kern<<<N / BM, NWG * 128 + 32, smem, at::cuda::getCurrentCUDAStream()>>>(mW, mX, op, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""

bfp_GQ_CONFIGS = {  # (layer, bn) -> (nwg, cl, nstage): BM = 64*nwg rows per CTA, grid = N/BM * cl CTAs (cl-CTA clusters split K)
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 7),
}
bfp_USE_GQ = True


def bfp__gq_src(sm90):
    lines = []
    for (layer, bn), (nwg, cl, ns) in sorted(bfp_GQ_CONFIGS.items()):
        lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                     f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
    return bfp_GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def bfp__gq_bn(m):
    return 64 if m <= 64 else 128 if m <= 128 else 256


def bfp__run_gq(x, layer, m, n, k, state):
    nwg, cl, ns = bfp_GQ_CONFIGS[(layer, bfp__gq_bn(m))]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    bfp__get_ext().gq(state["raw"][layer], x, out, m, n, k, bfp__gq_bn(m), nwg, cl, ns)
    return out


bfp_WIDE_CONFIGS = {  # (layer, tok, nz) -> (nwg, splits, stages); grid = (N / (64 nwg), splits, nz) <= 132 (co-resident)
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def bfp__wide_key(m):
    return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def bfp__wide_src(sm90):
    lines = []
    for (layer, tok, nz), (nwg, splits, stages) in sorted(bfp_WIDE_CONFIGS.items()):
        hl = layer == "router"
        ot = "float" if hl else "__nv_bfloat16"
        lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                     f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                     f"M, N, K, splits); return; }}")
    return bfp_WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def bfp__get_ext():
    """Compile everything as one CUDA translation unit with load_inline.  no_implicit_headers: the CUDA file includes
    only the ATen headers it needs (torch/types.h alone costs ~45 s of nvcc time)."""
    global bfp__ext
    if bfp__ext is None:
        major, minor = torch.cuda.get_device_capability()
        sm90 = major >= 9
        arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
        common = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", bfp_WG_HELPERS if sm90 else "")
        defs = bfp_wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
        cuda_src = (common + defs + bfp__tma_srcs(sm90) + bfp_MAP_SRC + bfp__gemm_src(sm90) + bfp__wide_src(sm90) + bfp__sk2_src(sm90)
                    + bfp__gq_src(sm90))
        bfp__ext = load_inline(
            name=f"dl_okc_bf16proj_v86y4_{arch}" + (f"_dly{bfp__SK2_DELAY}" if bfp__SK2_DELAY else ""),
            cpp_sources="#include <torch/extension.h>\n" + bfp_CPP_SRC,
            cuda_sources=cuda_src,
            functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={bfp__SK2_DELAY}"] if bfp__SK2_DELAY else []),
            no_implicit_headers=True,
            verbose=False,
        )
    return bfp__ext


def bfp_pack_fragments(w: torch.Tensor) -> torch.Tensor:
    """W [N, K] bf16 -> [N/16, K/16, 32 lanes, 8] in mma.m16n8k16 A-fragment order."""
    n, k = w.shape
    t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)  # [rg, hr, g, s, hk, t, e]: row = hr*8+g, col = hk*8+2t+e
    t = t.permute(0, 3, 2, 5, 4, 1, 6)  # [rg, s, g, t, hk, hr, e]
    return t.contiguous().view(n // 16, k // 16, 32, 8)


def bfp_pack_compressed(w: torch.Tensor):
    """Lossless 12-bit packing of BF16 W [N, K].  Returns None if the matrix doesn't suit it."""
    n, k = w.shape
    if n % 16 or k % 64:
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8                  # sign | exponent bits 7..1
    e71 = hi & 0x7F
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((hi >> 7) << 3) | o
    dec_hi = ((hi >> 7) << 7) | (base + o)
    dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
    esc = (e71 < base) | (e71 > base + 7)
    n_esc = int(esc.sum())
    if n_esc > n * k // 200:
        return None
    rows, cols = esc.nonzero(as_tuple=True)
    wtrue = w[rows, cols].float()
    wdec = dec[rows, cols].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lo_f = bfp_pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
    cd_f = bfp_pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
    nrg, S = n // 16, k // 16
    lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)   # [rg, u, ks, lane, 8]
    cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
    lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
    cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))           # [rg, u, ks, lane, 4]
    cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
    packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()  # [u, rg, 3, lane, 16]
    order = torch.argsort(rows * k + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    rg_idx = rows // 16
    poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
    poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
    return {
        "packed": packed,
        "base4": base * 0x01010101,
        "poff": poff,
        "pk": cols.to(torch.int32).contiguous(),
        "pr": (rows % 16).to(torch.int32).contiguous(),
        "pd": delta.to(torch.float32).contiguous(),
        "n_esc": n_esc,
    }


def bfp_build_patch_table(c, n_cfg, k, splits, rg):
    """Per-CTA exact-correction table for a (splits, RG) decomposition over the first n_cfg rows:
    int2 [num_ctas][pmax]; entry 0 = (count, 0); entries = ((row_local << 16) | k_local, delta bits)."""
    dev = c["pk"].device
    rows_cta = rg * 16
    nrb = n_cfg // rows_cta
    nunits = k // 64
    poff = c["poff"].long()
    nrg_all = poff.numel() - 1
    rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
    rows = rgi * 16 + c["pr"].long()
    cols = c["pk"].long()
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
    unit = cols // 64
    spl = ((unit + 1) * splits + nunits - 1) // nunits - 1          # split owning this 64-k unit
    k_start = (spl * nunits // splits) * 64
    cta = spl * nrb + rows // rows_cta
    ncta = nrb * splits
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
    tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
    return tab.contiguous(), pmax


def bfp_build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
    """Per-CTA contiguous weight stream for a (splits, RG, WS, KS) decomposition of the first n_cfg rows.
    WS > 1: [cta = sp * nrb + rb][stage][warp][chunk].  WS == 1: [rb][stage over all K][warp][chunk] (independent of
    splits; a CTA reads its contiguous stage range).  src: compressed [nrg, K/64, 3, 32, 16] uint8 or uncompressed
    fragments [nrg, K/16, 32, 8] bf16."""
    dev = src.device
    nw = rg * wsn
    nrb = (n_cfg // 16) // rg
    ar = lambda n: torch.arange(n, device=dev)
    if wsn == 1:
        s_all = (k // 16) // ks
        rb = ar(nrb).view(-1, 1, 1, 1)
        st = ar(s_all).view(1, -1, 1, 1)
        w = ar(nw).view(1, 1, -1, 1)
        rg_idx = rb * rg + w
        if comp:
            u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
            return src[rg_idx, u_idx].contiguous()
        s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
        return src[rg_idx, s_idx].contiguous()
    kc = k // splits
    ns = kc // (16 * wsn)
    nst = ns // ks
    sp = ar(splits).view(-1, 1, 1, 1, 1)
    rb = ar(nrb).view(1, -1, 1, 1, 1)
    st = ar(nst).view(1, 1, -1, 1, 1)
    w = ar(nw).view(1, 1, 1, -1, 1)
    rg_idx = rb * rg + w % rg
    if comp:
        ku = ar(ks // 4).view(1, 1, 1, 1, -1)
        u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
        out = src[rg_idx, u_idx]
    else:
        j = ar(ks).view(1, 1, 1, 1, -1)
        s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
        out = src[rg_idx, s_idx]
    return out.contiguous()


bfp_SK2_CONFIGS = [  # (name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL); a name ending in _rNNN caps the ring at NNN KB
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv16_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv8_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler16h_r100', 'unpooler', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
]
bfp_SK2_PICK = {  # (layer, tokr) -> config name
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o8h_r88',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8_r176',
    ('qkv_gate', 16): 'qkv16_r176',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16h_r100',
    ('unpooler', 32): 'unpooler32_r144',
}
bfp_SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
bfp_USE_SK2 = True
bfp_WIDE_BF16 = False  # the wide (RS-wgmma) kernel never beats cuBLAS for BF16 m 33..256


def bfp__sk2_name(layer, m):
    return bfp_SK2_PICK[(layer, bfp__sk2_tokr(m))]


def bfp__sk2_tokr(m):
    return 8 if m <= 8 else 16 if m <= 16 else 32


def bfp__sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(bfp_SK2_K[layer] // (ups * 64)) // CL)   # (the largest rank's stages)
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1   # (3 when the ring has 3 stages; sized for 4 here)
    rp = R + ((40 - (R % 32)) % 32)
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap = 200
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap = int(tail[1])
    ring = min(cap * 1024, 232448 - extra)
    nstage = max(2, min(16, nst, ring // sst))
    return hl, nstage


def bfp__sk2_src(sm90):
    lines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        for tokr in (8, 16, 32):
            if bfp_SK2_PICK.get((layer, tokr)) != name:
                continue
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            ot = "float" if hl else "__nv_bfloat16"
            lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                         f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
    flines = []
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        if name != "pooler":
            continue
        for tokr in (8, 16, 32):
            hl, nstage = bfp__sk2_geom(cfg, tokr)
            mt = 2 if tokr == 32 else 1
            nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
            flines.append(f"if (tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
                          f"{nacc}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                          f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                          f"rcnt.data_ptr<int>()); return; }}")
    return (bfp_SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))
            .replace("@@SK2F_DISPATCH@@", "\n  ".join(flines)))


def bfp_sk2_tileunits(w):
    """W [N, K] -> [N/8, K/64, 32 lanes, 16] in mma.m16n8k16 B-fragment order: lane (g, t) holds row g and, for each
    k-step j of the 64-k unit, k = 16j + 2t, 2t+1, 2t+8, 2t+9."""
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)          # [tile, g, unit, j, h, t, p]: k = 64u + 16j + 8h + 2t + p
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def bfp_sk2_compress(w):
    """Lossless 12-bit packing of BF16 W [N, K] into B-fragment tile-units [N/8, K/64, 768] (512 B low bytes, lane-major,
    then 256 B of 4-bit codes) + escapes (row, k, delta) that the kernel corrects exactly.  Returns None if unsuitable."""
    n, k = w.shape
    tu = bfp_sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(win.argmax())
    o = (e71 - base).clamp(0, 7)
    code = (sign << 3) | o
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)                  # [T, U, 32, 8]
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()                                   # [E, 4]: tile, unit, lane, v
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "delta": delta.contiguous()}


def bfp_sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    """Stage-major stream [stage][cta][warp][unit][tile][bytes] for the first n_cfg rows."""
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


def bfp_sk2_table(comp, n_cfg, R):
    """Per-CTA escape table int2 [ncta][pmax]: entry 0 = (count, 0); entries ((row_local << 16) | k, delta bits)."""
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["delta"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())   # per CTA in k order (the kernel walks K once)
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.view(torch.int32)
    return tab.contiguous(), pmax


def bfp_sk2_prepare(weights, state):
    state["sk2"] = {}
    if not (bfp_USE_SK2 and state["sm90"]):
        return
    dev = weights["o"].device
    shared = {}
    for ci, cfg in enumerate(bfp_SK2_CONFIGS):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer]
        if layer == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            T, U = w.shape[0] // 8, w.shape[1] // 64
            tub = torch.cat([bfp_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                             bfp_sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
            stream = bfp_sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
            tab, pmax, b4 = state["dummy"], 0, 0
        else:
            comp = state.setdefault("sk2_comp", {}).get(layer)
            if comp is None:
                comp = bfp_sk2_compress(w.contiguous())
                state["sk2_comp"][layer] = comp
            if comp is None:
                continue
            lkey = (layer, n_rows, R, NT, NWR, WSK, UPW)   # configs differing only in NACC / ring depth share a stream
            if lkey not in shared:
                shared[lkey] = (bfp_sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW),) + tuple(bfp_sk2_table(comp, n_rows, R))
            stream, tab, pmax = shared[lkey]
            b4 = comp["base4"]
        state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
    state.pop("sk2_comp", None)


def bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
    name = bfp__sk2_name(layer, m)
    ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
    return out


def bfp_fused_pooler_router(x, state):
    """Pooler and router of one decode step in ONE launch (m <= 32, sm_90): x [m, 8192] bf16 ->
    (pooler [m, 4096] bf16, router [m, 256] fp32). Same numerics as the separate sk2 kernels: 12-bit lossless pooler
    weights with exact escape corrections, router as BF16 hi + lo with FP32 accumulation (quarter-K partials added in
    a fixed order)."""
    m = x.shape[0]
    assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]
    ci, stream, tab, pmax, b4, n_rows = state["sk2"]["pooler"]
    rstream = state["sk2"]["router"][1]
    if "fused_part" not in state:
        state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
        state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    ext = bfp__get_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, bfp__sk2_tokr(m))
    return out, outr


bfp_SK2_PDL = 0   # decode_layer: set to N > 0 for programmatic dependent launches (N weight stages before the wait)


def bfp__mt_key(m):
    return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def bfp_prepare(weights):
    ext = bfp__get_ext()
    dev = weights["qkv_gate"].device
    sm90_ = torch.cuda.get_device_capability()[0] >= 9
    old_paths = not (sm90_ and bfp_USE_SK2)
    state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
    state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
    ws_floats = 0
    for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
        mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
        ws_floats = max(ws_floats, splits * mmax * bfp_w_rows[lname])
    for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
        ws_floats = max(ws_floats, splits * mm * bfp_w_rows[lname])
    state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
    gemm_floats = 0
    for (lname, bn), (nwg, splits, stages) in bfp_GEMM_CONFIGS.items():
        gemm_floats = max(gemm_floats, (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
    for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
        gemm_floats = max(gemm_floats, nz * (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
    state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
    if bfp_GEMM_DIAG:
        state["ws_shared_diag"] = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=dev)
    for name, w in weights.items():
        if name == "router":
            hi = w.to(torch.bfloat16)
            lo = (w - hi.float()).to(torch.bfloat16)
            state["router_hi_t"] = hi.t().contiguous()
            state["router_lo_t"] = lo.t().contiguous()
            state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()   # gemm90 fallback (m 33..256)
            src = torch.stack([bfp_pack_fragments(hi), bfp_pack_fragments(lo)], dim=2).contiguous()
            comp, kk = False, w.shape[1]
            c = None
            nrg_r = w.shape[0] // 16
            u_all = w.shape[1] // 64
            t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()   # [rg, u, hl, 4, 32, 8]
            nwg_r = bfp_WIDE_CONFIGS[("router", 128, 1)][0]
            t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
            state["streams"][("router", 0, 4 * nwg_r, 1, 4, w.shape[0])] = t
            for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
                if lname == "router":
                    state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
        else:
            w = w.contiguous()
            state["raw"][name] = w
            c = bfp_pack_compressed(w) if name in bfp_COMPRESS else None
            if c is not None:
                src, comp = c["packed"].transpose(0, 1).contiguous(), True  # [rg, u, 3, 32, 16]
                state["base4"][name] = c["base4"]
                del c["packed"]
            else:
                src, comp = bfp_pack_fragments(w), False
            kk = w.shape[1]
        n = w.shape[0]
        state["comp"][name] = comp
        for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
            if lname != name or not old_paths:
                continue
            n_cfg = 1024 if (name == "unpooler" and mtk <= 1) else n
            key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
            if key not in state["streams"]:
                state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
            ckey = (name, splits, rgc, wsn, ks, n_cfg)
            if ckey not in state["cnt"]:
                state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)  # per-config counters
            if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
        if comp and old_paths:
            for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
                if lname != name:
                    continue
                n_cfg = 1024 if (name == "unpooler" and mm <= 8) else n
                key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
                if key not in state["streams"]:
                    state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
                ckey = (name, splits, rgc, wsn, ks, n_cfg)
                if ckey not in state["cnt"]:
                    state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
                if (name, splits, rgc, n_cfg) not in state["tabs"]:
                    state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
        for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
            if lname != name or not comp:
                continue
            rgc = 4 * nwg
            key = (name, 0, rgc, 1, 4, n)
            if key not in state["streams"]:
                state["streams"][key] = bfp_build_stream(src, comp, n, kk, splits, rgc, 1, 4)
            if (name, splits, rgc, n) not in state["tabs"]:
                state["tabs"][(name, splits, rgc, n)] = bfp_build_patch_table(c, n, kk, splits, rgc)
            state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        del src
        state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    state["sm90"] = sm90_
    state["old_paths"] = old_paths
    bfp_sk2_prepare(weights, state)
    if bfp_GEMM_DIAG and state["sm90"]:
        state["wtile"], state["diag_cnt"] = {}, {}
        for layer in bfp_DIAG_LAYERS:
            state["wtile"][layer] = bfp_swizzle_tiles(state["raw"][layer], 128)
            for a_s in (0, 1):
                for sp_ in sorted({bfp_GEMM_CONFIGS[(layer, 128)][1], bfp_GEMM_CONFIGS[(layer, 256)][1], 1}):
                    state["diag_cnt"][(layer, a_s, sp_)] = torch.zeros(bfp_w_rows[layer] // 16, dtype=torch.int32, device=dev)
    bfp__selftest(weights, state)
    bfp__autotune(weights, state)
    torch.cuda.empty_cache()
    return state


bfp_ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
bfp_WIDE_SM80_TEST = False  # local check of the wide kernel's sm_80 path
bfp_ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def bfp__route(layer, m, sm90):
    for mmax, r in (bfp_ROUTE_ROUTER if layer == "router" else bfp_ROUTE_BF16):
        if m <= mmax:
            if r == "gemm" and not sm90:
                return "torch"
            if r == "wide" and not (sm90 or bfp_WIDE_SM80_TEST):
                return "torch"
            return r
    return "torch"


def bfp__route_key(layer, m, route):
    if route == "sk2":
        return (layer, route, bfp__sk2_name(layer, m), bfp__sk2_tokr(m))
    if route == "skinny":
        return (layer, route, bfp__mt_key(m))
    if route == "wide":
        return (layer, route) + bfp__wide_key(m)
    return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


bfp_SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def bfp__selftest(weights, state):
    """Run every custom path once on a random input against an FP64 reference; disable (-> cuBLAS) any path that
    does not match.  Guards against a kernel that misbehaves on a GPU this file was not tested on."""
    state["disabled"] = set()
    g = torch.Generator(device=weights["o"].device)
    g.manual_seed(1234)
    for layer, w in weights.items():
        n, k = w.shape
        for m in bfp_SELFTEST_M:
            nn = 1024 if (layer == "unpooler" and m <= 8) else n
            x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
            ref = x.double() @ w[:nn].double().T
            if layer != "router":
                ref = ref.to(torch.bfloat16).double()
            tol = 2e-5 if layer == "router" else 5e-3
            for ch in bfp__candidates(layer, m, state):
                if ch[0] == "torch":
                    continue
                key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
                       else bfp__route_key(layer, m, ch[0]))
                try:
                    out = bfp__run(ch, x, layer, m, nn, k, state).double()
                    torch.cuda.synchronize()
                    err = ((out - ref).norm() / ref.norm()).item()
                except RuntimeError:  # launch failure -> fall back
                    err = float("inf")
                if not err < tol:
                    state["disabled"].add(key)


def bfp__run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
    splits, rg, wsn, ks = cfg
    mtk = bfp__mt_key(m)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
    cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
    fn = bfp__get_ext().skinny_tma_a
    if state["comp"][layer]:
        tab, pmax = state["tabs"][(layer, splits, rg, n)]
        fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 1, bfp_w_rows[layer] // 16, ks, dg)
    else:
        fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
           max(mtk, 1), rg, wsn, 0, bfp_w_rows[layer] // 16, ks, dg)
    return out


def bfp__run_wide(x, layer, m, n, k, state):
    tok, nz = bfp__wide_key(m)
    nwg, splits, _ = bfp_WIDE_CONFIGS[(layer, tok, nz)]
    rgc = 4 * nwg
    if layer == "router":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        tab, pmax, b4 = state["dummy"], 0, 0
    else:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
    bfp__get_ext().wide(state["streams"][(layer, 0, rgc, 1, 4, n)], x, out, state["mid_ws_shared"],
                    state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
    return out


def bfp__run_gemm(x, layer, m, n, k, state):
    bn = 64 if m <= 64 else 128 if m <= 128 else 256
    nwg, splits, _ = bfp_GEMM_CONFIGS[(layer, bn)]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    w = state["router_cat"] if layer == "router" else state["raw"][layer]
    bfp__get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                      splits, nwg, bn, 0)
    return out


def bfp__run_torch(x, layer, m, n, k, state):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return F.linear(x, state["raw"][layer][:n])   # what SGLang's bf16 dispatch runs on H200


def bfp__run(choice, x, layer, m, n, k, state):
    kind = choice[0]
    if kind == "skinny":
        return bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
    if kind == "wide":
        return bfp__run_wide(x, layer, m, n, k, state)
    if kind == "sk2":
        return bfp__run_sk2(x, layer, m, n, k, state)
    if kind == "gemm":
        return bfp__run_gemm(x, layer, m, n, k, state)
    if kind == "gq":
        return bfp__run_gq(x, layer, m, n, k, state)
    return bfp__run_torch(x, layer, m, n, k, state)


def bfp__static_choice(layer, m, state):
    route = bfp__route(layer, m, state["sm90"])
    if not state.get("old_paths", True):
        if route == "skinny":
            route = "sk2" if (bfp__sk2_name(layer, m) in state.get("sk2", {})
                              and bfp__route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
        elif route == "wide" and layer != "router" and not bfp_WIDE_BF16:
            route = "torch"
    if route != "torch" and bfp__route_key(layer, m, route) in state.get("disabled", ()):
        route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                           and bfp__route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
    if route == "skinny":
        mtk = bfp__mt_key(m)
        return ("skinny", bfp_TMA_CONFIGS[(layer, mtk)], (layer, mtk))
    return (route,)


def bfp__candidates(layer, m, state):
    """Implementations worth timing for (layer, m): the static choice, alternative skinny configs, cuBLAS."""
    c = [bfp__static_choice(layer, m, state)]
    if c[0][0] == "skinny" and (layer, m) in bfp_TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
        c.append(("skinny", bfp_TMA_ALT[(layer, m)], ("alt", layer, m)))
    if (c[0][0] == "wide" and layer == "router" and state["sm90"]
            and bfp__route_key(layer, m, "gemm") not in state["disabled"]):
        c.append(("gemm",))
    if (m <= 32 and c[0][0] != "sk2" and bfp__sk2_name(layer, m) in state.get("sk2", {})
            and bfp__route_key(layer, m, "sk2") not in state["disabled"]):
        c.insert(0, ("sk2",))
    if (bfp_USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, bfp__gq_bn(m)) in bfp_GQ_CONFIGS
            and bfp__route_key(layer, m, "gq") not in state["disabled"]):
        c.insert(0, ("gq",))
    if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
        c.append(("torch",))
    return c


bfp_AUTOTUNE = True
bfp_TUNE_M = (1, 16, 32, 8, 128, 256)   # the scored token counts


def bfp__autotune(weights, state):
    """Time each candidate as a CUDA graph replay from a cold L2 (a read sweep larger than L2 before every replay,
    per-replay events, candidates interleaved), on synthetic activations (reads only the weights), and keep the
    fastest per (layer, m)."""
    state["choice"] = {}
    if not bfp_AUTOTUNE:
        return
    dev = weights["o"].device
    flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)   # 256 MB, read-only sweep
    sink = torch.empty(1024, dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(4321)
    reps = 24
    for m in bfp_TUNE_M:
        for layer, w in weights.items():
            n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
            k = w.shape[1]
            cands = bfp__candidates(layer, m, state)
            if len(cands) < 2:
                continue
            x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
            graphs = []
            for ch in cands:
                bfp__run(ch, x, layer, m, n, k, state)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    bfp__run(ch, x, layer, m, n, k, state)
                graphs.append(gr)
            evs = []
            for r in range(reps):
                order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
                for i in order:
                    torch.sum(flush, dim=0, out=sink)
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    graphs[i].replay()
                    e1.record()
                    evs.append((i, e0, e1))
            torch.cuda.synchronize()
            per = [[] for _ in cands]
            for i, e0, e1 in evs:
                per[i].append(e0.elapsed_time(e1))
            med = [sorted(v)[len(v) // 2] for v in per]
            best = min(range(len(cands)), key=lambda i: med[i])
            state["choice"][(layer, m)] = cands[best]
            state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
            del graphs
    del flush
    torch.cuda.synchronize()


def bfp__nearby_choice(layer, m, state):
    """An untuned m takes the choice tuned for the scored m of its bucket (1 | 2-8 | 9-16 | 17-32 | 33-128 |
    129-256): every implementation handles any m of its bucket."""
    tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
    if not tm:
        return None
    ch = state.get("choice", {}).get((layer, tm))
    if ch is None:
        return None
    if ch[0] == "gq" and (layer, bfp__gq_bn(m)) not in bfp_GQ_CONFIGS:
        return None
    return ch


def bfp_kernel_fn(data, state):
    x = data["x"]
    cfg = data["config"]
    layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
    choice = state.get("choice", {}).get((layer, m)) or bfp__nearby_choice(layer, m, state) or bfp__static_choice(layer, m, state)
    out = bfp__run(choice, x, layer, m, n, k, state)
    if bfp_GEMM_DIAG:
        bfp__diag(x, layer, m, n, k, state, choice)
    return out


def bfp__diag(x, layer, m, n, k, state, choice):
    """Extra kernels for the judge's per-kernel profile (outputs discarded)."""
    if choice[0] == "skinny" and m == 1 and layer in ("o", "unpooler") and state["comp"][layer]:
        for dg in (1, 2, 3):   # same kernel without epilogue / x loads / compute
            bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2], dg)


def _bfp_prepare(weights, st):
    """Their prepare() on the five dense matrices, guarded: any failure just disables the backend."""
    st["bfp"] = None
    try:
        w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
        st["bfp"] = bfp_prepare(w5)
    except Exception as ex:  # noqa: BLE001
        print("bfp disabled:", repr(ex)[:160], flush=True)


_BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}


BFP_PDL_N = 2   # bf16_proj's sk2 / fused kernels: weight stages issued before griddepcontrol.wait (0 = off)


PCX_CFG = {}   # (role, T or 0) -> (P, d0, g[, e]); see harness/make_pcx.py


def _pcx_val(role, T):
    P, d0, g, e = (tuple(PCX_CFG.get((role, T), PCX_CFG.get((role, 0), (BFP_PDL_N, 0, 0, 0)))) + (0,))[:4]
    return (1 << 30) | P | (d0 << 4) | (g << 9) | (e << 14)


def _bfp_pdl(T, role=None):
    """Their launchers read SK2_PDL (bfp_SK2_PDL once inlined) at every launch: same T rule as this file's glue."""
    global bfp_SK2_PDL
    bfp_SK2_PDL = _pcx_val(role, T) if _pdl(T) else 0


def _bfp_gemm(st, name, x, m, n, k):
    _bfp_pdl(m, name)
    return bfp_kernel_fn({"x": x, "config": {"layer": _BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


_MA_MMAX = 384
_MA_EMAX = _MA_MMAX * 8


WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16   # moe WGS_FIXF: stream-K fixups + doneAs after ybuf (zero between calls)
WGS_XROWS = _MA_EMAX + 8 * (32 + _MA_EMAX // 32 + 8)  # >= moe WGS_XROWS for FT_T >= 32


def _df_inter(inter):
    """DF: inter rows [128, 256) start (and, after every MID launch, end) as bf16 0xFFFF = not yet written."""
    inter[128 * 4096:256 * 4096].view(torch.int16).fill_(-1)
    return inter


def _ma_make_scratch(dev):
    return [torch.zeros(8 + 1024 + _MA_EMAX + 4096, dtype=torch.int32, device=dev),  # counters + doneA + ss copy 1 + MFIN flags
            torch.zeros(_MA_EMAX + _MA_EMAX * 4096 + WGS_FIXF, dtype=torch.float32, device=dev),
            _df_inter(torch.zeros((_MA_EMAX + 2 * WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev))]  # inter, then WGS xA / xB

SCRATCH_GUARD = 1 << 16   # sentinel words after each buffer my glue allocates: wide enough for far overruns (v154m-style state ~4K words out)
SCRATCH_SENTINEL = 0x5EADBEEF


def _ma_prepare(weights, st):
    dev = weights["w13"].device
    I = H = 4096
    rb = torch.arange(I // 8, device=dev)
    g = torch.arange(8, device=dev)
    rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
    rowsB = torch.arange(H, device=dev)
    wA, sA, wB, sB = [], [], [], []
    for e in range(32):
        wA.append(_pack_codes(weights["w13"][e].view(torch.uint8), rowsA))
        sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
        wB.append(_pack_codes(weights["w2"][e].view(torch.uint8), rowsB))
        sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
    lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
           * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    lut = (lut | (lut << 16)).to(torch.int32)
    st["ma_w"] = [torch.cat(wA), torch.cat(sA), weights["w13_gs"].float().contiguous(),
                  torch.cat(wB), torch.cat(sB), weights["w2_gs"].float().contiguous(),
                  lut.contiguous(), weights["post_expert_norm"].contiguous(),  # bf16; kernel forms 1 + w
                  torch.cat([_fat_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_fat_order(x, y) for x, y in zip(wB, sB)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wA, sA)]),
                  torch.cat([_unit_order(x, y) for x, y in zip(wB, sB)])]
    if "_wgs_on" in globals() and _wgs_on():   # their prepare(): w[12], w[13] = the WGS copies (per 256-row unit)
        st["ma_w"].append(torch.cat([_wgs_copy(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], _kperm_rows(rowsA), _MA_SC6W)
                                     for e in range(32)]))
        st["ma_w"].append(torch.cat([_wgs_copy(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB, _MA_SC6W)
                                     for e in range(32)]))
    st["ma_scratch"] = [_guarded(t) for t in _ma_make_scratch(dev)]


def _to_i32(x64):
    x64 = x64 & 0xFFFFFFFF
    return (x64 - (x64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)


def _repack_q(wp, interleave):
    """wp uint8 [E, R, 2048] (low nibble = even k) -> int32 [E, R/16, 64, 32, 4] mma-fragment words."""
    E, R, _ = wp.shape
    if interleave:
        wp = wp.view(E, 2, R // 32, 16, 2048).transpose(1, 2).reshape(E, R, 2048)
    x = wp.view(E, R // 16, 16, 256, 8).permute(0, 1, 3, 2, 4).to(torch.int64)  # [E, RB, kb, row, j]
    B3 = x[..., :8, :4]
    B2 = x[..., 8:, :4]
    B1 = x[..., :8, 4:]
    B0 = x[..., 8:, 4:]
    w = ((B0 & 15) | ((B1 & 15) << 4) | ((B2 & 15) << 8) | ((B3 & 15) << 12)
         | ((B0 >> 4) << 16) | ((B1 >> 4) << 20) | ((B2 >> 4) << 24) | ((B3 >> 4) << 28))
    w = _to_i32(w).reshape(E, R // 16, 64, 4, 32).permute(0, 1, 2, 4, 3).contiguous()
    return w


def _repack_s(sf, interleave):
    """sf uint8 [E, R, 256] -> uint8 [E, R/16, 64, 8, 8]: (g, [rows g kb0..3, rows g+8 kb0..3])."""
    E, R, _ = sf.shape
    if interleave:
        sf = sf.view(E, 2, R // 32, 16, 256).transpose(1, 2).reshape(E, R, 256)
    x = sf.view(E, R // 16, 16, 64, 4).permute(0, 1, 3, 2, 4)  # [E, RB, ch, row, kk]
    return torch.cat([x[:, :, :, :8, :], x[:, :, :, 8:, :]], dim=-1).contiguous()


@triton.jit
def _gemm_splitk(X, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                 BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # X is the producer's output
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _gemm_splitk_ng(X, G, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                    BK: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Like _gemm_splitk with x = rmsnorm_per_64(X) * sigmoid(G) applied on the fly (X bf16, G fp32)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    pt = tl.program_id(2)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    kbase = ps * KSPLIT
    w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
    x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
    g_ptr = G + rt[None, :] * K + (kbase + rk)[:, None]
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for _k in range(0, KSPLIT, BK):
        w = tl.load(w_ptr)
        x = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
        gt = tl.load(g_ptr, mask=tmask, other=0.0)
        x3 = tl.reshape(x, [BK // 64, 64, BT])
        r = tl.rsqrt(tl.sum(x3 * x3, 1) / 64.0 + 1e-5)
        xn = tl.reshape(x3 * r[:, None, :], [BK, BT]) * tl.sigmoid(gt)
        acc = tl.dot(w, xn.to(tl.bfloat16), acc)
        w_ptr += BK
        x_ptr += BK
        g_ptr += BK
    tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _normgate(X, G, OUT, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    h = tl.arange(0, 16)[:, None]
    d = tl.arange(0, 64)[None, :]
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + t * 1024 + h * 64 + d).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 1) / 64.0 + 1e-5)[:, None]
    x = x * tl.sigmoid(tl.load(G + t * 1024 + h * 64 + d))
    tl.store(OUT + t * 1024 + h * 64 + d, x.to(tl.bfloat16))


def dense_ng(x, gate, w, T, state=None):
    N, K = w.shape
    if T > 32 or (state is not None and state.get("mm_pick", {}).get((T, "o"), False)):
        xn = torch.empty((T, K), device=x.device, dtype=torch.bfloat16)
        _normgate[(T,)](x, gate, xn, num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
        return dense(xn, w, T, state, "o")
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    S = 1
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk_ng[grid](x, gate, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


TUNED_T = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _bucket(T):
    """The tuned token count whose choices apply to T (the next one at or above it)."""
    for t in TUNED_T:
        if T <= t:
            return t
    return TUNED_T[-1]


BFP_MAX_T = 32   # bf16_proj's qkv / o / unpooler kernels where they win inside the chain (T=1..32: +4-9% on those


def dense(x, w, T, state=None, name=None):
    """Returns (partials [S, T, N] fp32, S). The tuner picks cuBLAS or the split-K Triton kernel per
    (token count, weight matrix): cuBLAS wins on the o projection, the Triton one on the pooler+router."""
    N, K = w.shape
    pick = state.get("mm_pick", {}).get((T, name), False) if state is not None else False
    if pick == "bfp" and T <= BFP_MAX_T and name in _BFP_LAYER and state.get("bfp") is not None:
        return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
    if pick is True or pick == "bfp":       # above BFP_MAX_T a bfp pick runs as cuBLAS (measured in the chain)
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    if T > 32:
        out = torch.mm(x, w.t(), out_dtype=torch.float32)
        return out.view(1, T, N), 1
    BN, BK = 64, 128
    BT = 16 if T <= 16 else 32
    tiles = (N // BN) * triton.cdiv(T, BT)
    S = 1
    while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
        S *= 2
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    grid = (N // BN, S, triton.cdiv(T, BT))
    _gemm_splitk[grid](x, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return P, S


@triton.jit
def _f32_to_e4m3(x):
    """float32 -> e4m3fn bits (int32), round to nearest even, saturating to +-448."""
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    bits = x.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 0x80
    a = tl.abs(x)
    sub = (a * 512.0 + 0.5).to(tl.int32)  # round half up (ties are measure-zero)
    ab = a.to(tl.int32, bitcast=True)
    r = ab + 0x7FFFF + ((ab >> 20) & 1)
    r = r & 0x7FF00000
    e = (r >> 23) - 120
    m = (r >> 20) & 7
    nrm = (e << 3) | m
    nrm = tl.minimum(nrm, 0x7E)
    code = tl.where(a < 0.015625, sub, nrm)
    return code | sign


@triton.jit
def _qk_prep(P, S: tl.constexpr, T, POS, LOC, INVF, KC, VC, KS, VS, QOUT, Q8OUT, GOUT, qk_factor,
             ROPE: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    base = P + t * 2304
    hq = tl.arange(0, 16)[:, None]
    hk = tl.arange(0, 2)[:, None]
    i = tl.arange(0, 32)[None, :]
    d = tl.arange(0, 64)[None, :]
    qe = tl.zeros([16, 32], dtype=tl.float32)
    qo = tl.zeros([16, 32], dtype=tl.float32)
    ke = tl.zeros([2, 32], dtype=tl.float32)
    ko = tl.zeros([2, 32], dtype=tl.float32)
    v = tl.zeros([2, 64], dtype=tl.float32)
    g = tl.zeros([1024], dtype=tl.float32)
    gi = tl.arange(0, 1024)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()
    for s in range(S):
        b = base + s * T * 2304
        qe += tl.load(b + hq * 64 + 2 * i)
        qo += tl.load(b + hq * 64 + 2 * i + 1)
        ke += tl.load(b + 1024 + hk * 64 + 2 * i)
        ko += tl.load(b + 1024 + hk * 64 + 2 * i + 1)
        v += tl.load(b + 1152 + hk * 64 + d)
        g += tl.load(b + 1280 + gi)
    rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, 1) / 64.0 + 1e-5)[:, None]
    rk = tl.rsqrt(tl.sum(ke * ke + ko * ko, 1) / 64.0 + 1e-5)[:, None]
    qe = qe * rq
    qo = qo * rq
    ke = ke * rk
    ko = ko * rk
    if ROPE:
        pos = tl.load(POS + t).to(tl.float32)
        ang = pos * tl.load(INVF + i)
        c = tl.cos(ang)
        sn = tl.sin(ang)
        qe, qo = qe * c - qo * sn, qe * sn + qo * c
        ke, ko = ke * c - ko * sn, ke * sn + ko * c
    qb = QOUT + t * 1024 + hq * 64 + 2 * i
    tl.store(qb, (qe * qk_factor).to(tl.bfloat16))
    tl.store(qb + 1, (qo * qk_factor).to(tl.bfloat16))
    q8b = Q8OUT + t * 1024 + hq * 64 + 2 * i
    tl.store(q8b, _f32_to_e4m3(qe).to(tl.uint8))
    tl.store(q8b + 1, _f32_to_e4m3(qo).to(tl.uint8))
    tl.store(GOUT + t * 1024 + gi, g)
    loc = tl.load(LOC + t).to(tl.int64)
    kinv = 1.0 / tl.load(KS)
    vinv = 1.0 / tl.load(VS)
    kb = KC + loc * 128 + hk * 64 + 2 * i
    tl.store(kb, _f32_to_e4m3(ke * kinv).to(tl.uint8))
    tl.store(kb + 1, _f32_to_e4m3(ko * kinv).to(tl.uint8))
    tl.store(VC + loc * 128 + hk * 64 + d, _f32_to_e4m3(v * vinv).to(tl.uint8))


@triton.jit
def _attn_split(Q, KC, VC, PT, SEQ, KS, ACC, ML, pt_stride, page_size, window, sm_scale,
                L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SLIDING: tl.constexpr):
    """One program = one request, both KV heads (128-byte K/V rows), one key split."""
    b = tl.program_id(0)
    s = tl.program_id(1)
    seq = tl.load(SEQ + b)
    m_off = tl.arange(0, BM)
    l_idx = m_off // 8
    hh = m_off % 8
    row_ok = m_off < L * 8
    qpos = seq + l_idx
    d = tl.arange(0, 64)
    qrow = (b * L + l_idx) * 16 + hh
    q0 = tl.load(Q + qrow[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    q1 = tl.load(Q + (qrow + 8)[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
    total = seq + L
    if SLIDING:
        lo = tl.maximum(seq - window + 1, 0)
    else:
        lo = seq * 0
    n = total - lo
    per = tl.cdiv(tl.cdiv(n, NSPLIT), BN) * BN
    kst = lo + s * per
    ken = tl.minimum(total, kst + per)
    scale = sm_scale * 256.0 * tl.load(KS) * 1.4426950408889634
    m0 = tl.full([BM], float("-inf"), dtype=tl.float32)
    m1 = tl.full([BM], float("-inf"), dtype=tl.float32)
    l0 = tl.zeros([BM], dtype=tl.float32)
    l1 = tl.zeros([BM], dtype=tl.float32)
    a0 = tl.zeros([BM, 64], dtype=tl.float32)
    a1 = tl.zeros([BM, 64], dtype=tl.float32)
    for kb in range(kst, ken, BN):
        offs = kb + tl.arange(0, BN)
        kv_ok = offs < ken
        page = tl.load(PT + b * pt_stride + offs // page_size, mask=kv_ok, other=0)
        row = page.to(tl.int64) * page_size + offs % page_size
        kptr = row[:, None] * 128 + d[None, :]
        ok = kv_ok[None, :] & (offs[None, :] <= qpos[:, None])
        if SLIDING:
            ok = ok & (offs[None, :] > qpos[:, None] - window)
        k8 = tl.load(KC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16 = (((k8 & 0x7F) << 7) | ((k8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        k8b = tl.load(KC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        k16b = (((k8b & 0x7F) << 7) | ((k8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8 = tl.load(VC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16 = (((v8 & 0x7F) << 7) | ((v8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        v8b = tl.load(VC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
        v16b = (((v8b & 0x7F) << 7) | ((v8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
        sc = tl.where(ok, tl.dot(q0, tl.trans(k16)) * scale, float("-inf"))
        mn = tl.maximum(m0, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m0 - mu)
        l0 = l0 * al + tl.sum(p, 1)
        a0 = a0 * al[:, None] + tl.dot(p.to(tl.float16), v16)
        m0 = mn
        sc = tl.where(ok, tl.dot(q1, tl.trans(k16b)) * scale, float("-inf"))
        mn = tl.maximum(m1, tl.max(sc, 1))
        mu = tl.where(mn == float("-inf"), 0.0, mn)
        p = tl.exp2(sc - mu[:, None])
        al = tl.exp2(m1 - mu)
        l1 = l1 * al + tl.sum(p, 1)
        a1 = a1 * al[:, None] + tl.dot(p.to(tl.float16), v16b)
        m1 = mn
    ob0 = ((b * 2 + 0) * NSPLIT + s) * BM + m_off
    ob1 = ((b * 2 + 1) * NSPLIT + s) * BM + m_off
    tl.store(ACC + ob0[:, None] * 64 + d[None, :], a0)
    tl.store(ACC + ob1[:, None] * 64 + d[None, :], a1)
    tl.store(ML + ob0 * 2, m0)
    tl.store(ML + ob0 * 2 + 1, l0)
    tl.store(ML + ob1 * 2, m1)
    tl.store(ML + ob1 * 2 + 1, l1)


@triton.jit
def _attn_combine(ACC, ML, GATE, VS, OUT, L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr,
                  SB: tl.constexpr, NORMGATE: tl.constexpr = True):
    """One program per (token, q head): merge the splits, per-head RMSNorm, * sigmoid(gate)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = t // L
    l = t % L
    g = h // 8
    m_row = l * 8 + h % 8
    d = tl.arange(0, 64)
    base = (b * 2 + g) * NSPLIT
    M = tl.full([SB], float("-inf"), dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        M = tl.maximum(M, tl.load(ML + ((base + sidx) * BM + m_row) * 2, mask=sm, other=float("-inf")))
    Mx = tl.max(M, 0)
    Mx = tl.where(Mx == float("-inf"), 0.0, Mx)
    den = tl.zeros([SB], dtype=tl.float32)
    num = tl.zeros([64], dtype=tl.float32)
    for s0 in range(0, NSPLIT, SB):
        sidx = s0 + tl.arange(0, SB)
        sm = sidx < NSPLIT
        rows = (base + sidx) * BM + m_row
        m_s = tl.load(ML + rows * 2, mask=sm, other=float("-inf"))
        l_s = tl.load(ML + rows * 2 + 1, mask=sm, other=0.0)
        w = tl.exp2(m_s - Mx)
        den += w * l_s
        a = tl.load(ACC + rows[:, None] * 64 + d[None, :], mask=sm[:, None], other=0.0)
        num += tl.sum(w[:, None] * a, 0)
    o = num / tl.sum(den, 0) * (256.0 * tl.load(VS))
    if NORMGATE:
        o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
        gt = tl.load(GATE + t * 1024 + h * 64 + d)
        o = o * tl.sigmoid(gt)
    tl.store(OUT + t * 1024 + h * 64 + d, o.to(tl.bfloat16))


@triton.jit
def _gemm_o(X, G, W, AOUT, SUMS, STREAM, A1, B1, T, N, K: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
            BK: tl.constexpr, NORMGATE: tl.constexpr):
    """o projection without a K split (K = 1024), so each program owns complete rows: it writes a and
    accumulates the four per-token sums the post-attention boundary needs (a^2, (a1 s)^2, (a1 s)(b1 a), (b1 a)^2)."""
    pn = tl.program_id(0)
    pt = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN)
    rt = pt * BT + tl.arange(0, BT)
    rk = tl.arange(0, BK)
    tmask = rt[None, :] < T
    acc = tl.zeros([BN, BT], dtype=tl.float32)
    w_ptr = W + rn[:, None].to(tl.int64) * K + rk[None, :]
    x_ptr = X + rt[None, :] * K + rk[:, None]
    for _k in range(0, K, BK):
        w = tl.load(w_ptr)
        if NORMGATE:
            xv = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
            gv = tl.load(G + rt[None, :] * K + (rk[:, None] + _k))
            xr = tl.reshape(xv, [BK // 64, 64, BT])
            rs = tl.rsqrt(tl.sum(xr * xr, 1) / 64.0 + 1e-5)
            xv = tl.reshape(xr * rs[:, None, :], [BK, BT]) * tl.sigmoid(gv)
            x = xv.to(tl.bfloat16)
        else:
            x = tl.load(x_ptr, mask=tmask, other=0.0)
        acc = tl.dot(w, x, acc)
        w_ptr += BK
        x_ptr += BK
    tl.store(AOUT + rt[None, :] * N + rn[:, None], acc, mask=tmask)
    st = tl.load(STREAM + rt[None, :] * N + rn[:, None], mask=tmask, other=0.0)
    a1v, b1v = _gate_ab(A1, rn)
    as_ = a1v[:, None] * st
    ba = b1v[:, None] * acc
    ok = (rt[None, :] < T) + tl.zeros([BN, BT], tl.int1)
    q0 = tl.sum(tl.where(ok, acc * acc, 0.0), 0)
    q1 = tl.sum(tl.where(ok, as_ * as_, 0.0), 0)
    q2 = tl.sum(tl.where(ok, as_ * ba, 0.0), 0)
    q3 = tl.sum(tl.where(ok, ba * ba, 0.0), 0)
    m = rt < T
    tl.atomic_add(SUMS + rt * 4 + 0, q0, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 1, q1, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 2, q2, mask=m)
    tl.atomic_add(SUMS + rt * 4 + 3, q3, mask=m)


@triton.jit
def _bnd1_apply(AOUT, SUMS, STREAM, A1, B1, G, S1OUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    """s1 = a1 * stream + b1 * a * rstd(a), h = s1 * rstd(s1) * g, from the four sums (one block per column slice)."""
    t = tl.program_id(0)
    c0 = tl.program_id(1) * BC
    offs = c0 + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r1 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a = tl.load(AOUT + t * N + offs)
    a1v, b1v = _gate_ab(A1, offs)
    s1 = a1v * tl.load(STREAM + t * N + offs) + b1v * a * r
    tl.store(S1OUT + t * N + offs, s1)
    tl.store(HOUT + t * N + offs, (s1 * r1 * (1.0 + tl.load(G + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _route_a(P, S: tl.constexpr, T, BIAS, IDS, WTS, LS, LIDS, POOL, PSUM, BC: tl.constexpr):
    """Reduce the pooler/router partials: one block per (token, pooled slice); slice 0 also does the top-8."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    c = j * BC + tl.arange(0, BC)
    pooled = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        pooled += tl.load(P + (s * T + t) * 4608 + c)
    tl.store(POOL + t * 4096 + c, pooled)
    tl.atomic_add(PSUM + t, tl.sum(pooled * pooled, 0))
    if j == 0:
        e = tl.arange(0, 256)
        logit = tl.zeros([256], dtype=tl.float32)
        for s in range(S):
            b = P + (s * T + t) * 4608
            logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        ids = tl.zeros([8], dtype=tl.int32)
        wv = tl.zeros([8], dtype=tl.float32)
        for k in tl.static_range(8):
            idx = tl.argmax(sel, 0)
            val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
            ids = tl.where(k8 == k, idx, ids)
            wv = tl.where(k8 == k, val, wv)
            sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


@triton.jit
def _route_b(POOL, PSUM, GPRE, XOUT, BC: tl.constexpr, PERMUTED: tl.constexpr):
    """x = rmsnorm(pooled) * (1 + pre_expert_norm), in the expert kernel's layout."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    r = tl.rsqrt(tl.load(PSUM + t) / 4096.0 + 1e-5)
    x = tl.load(POOL + t * 4096 + c) * r * (1.0 + tl.load(GPRE + c).to(tl.float32))
    if PERMUTED:
        ch = c // 64
        kk = c % 64
        kb = kk // 16
        rr = kk % 16
        po = ((t * 64 + ch) * 4 + (rr % 8) // 2) * 16 + kb * 4 + (rr // 8) * 2 + (rr % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _bnd2_sums(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, FOUT, SUMS, BC: tl.constexpr):
    """f = (unpooler partials summed) * g_postffn, and the four sums for the FFN boundary."""
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < NU
    f = tl.zeros([BC], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + c, mask=m, other=0.0)
    f = f * (1.0 + tl.load(GPOST + c).to(tl.float32))
    tl.store(FOUT + t * 8192 + c, f)
    st = tl.load(S1 + t * 8192 + c)
    a2v, b2v = _gate_ab(A2, c)
    as_ = a2v * st
    bf = b2v * f
    tl.atomic_add(SUMS + t * 4 + 0, tl.sum(f * f, 0))
    tl.atomic_add(SUMS + t * 4 + 1, tl.sum(as_ * as_, 0))
    tl.atomic_add(SUMS + t * 4 + 2, tl.sum(as_ * bf, 0))
    tl.atomic_add(SUMS + t * 4 + 3, tl.sum(bf * bf, 0))


@triton.jit
def _bnd2_apply(FOUT, SUMS, S1, A2, B2, GNEXT, SOUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    s0 = tl.load(SUMS + t * 4 + 0)
    s1q = tl.load(SUMS + t * 4 + 1)
    s2q = tl.load(SUMS + t * 4 + 2)
    s3q = tl.load(SUMS + t * 4 + 3)
    r = tl.rsqrt(s0 / N + 1e-8)
    r2 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
    a2v, b2v = _gate_ab(A2, c)
    s2 = a2v * tl.load(S1 + t * N + c) + b2v * tl.load(FOUT + t * N + c) * r
    tl.store(SOUT + t * N + c, s2)
    tl.store(HOUT + t * N + c, (s2 * r2 * (1.0 + tl.load(GNEXT + c).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _gate_ab(G, offs):
    """alpha, beta of a residual gate, from the checkpoint's BF16 weight (as SGLang's fused module does)."""
    w = tl.load(G + offs).to(tl.float32) / 0.3
    beta = tl.sigmoid(w)
    alpha = tl.sqrt(tl.maximum(tl.sigmoid(-w) * (1.0 + beta), 1e-3))
    return alpha, beta


@triton.jit
def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, ROUTER_OUT, WRITE_ROUTER: tl.constexpr = False, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    a1v, b1v = _gate_ab(A1, offs)              # weights and the layer's input stream: no producer dependency
    stv = tl.load(STREAM + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the o-projection) is the producer's output
    a = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        a += tl.load(P + (s * T + t) * 8192 + offs)
    n = a * tl.rsqrt(tl.sum(a * a, 0) / 8192.0 + 1e-8)
    s1 = a1v * stv + b1v * n
    tl.store(S1OUT + t * 8192 + offs, s1)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))
    if WRITE_ROUTER:
        tl.store(ROUTER_OUT + t * 8192 + offs, hh)


@triton.jit
def _route(P, S: tl.constexpr, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    logit = tl.zeros([256], dtype=tl.float32)
    pooled = tl.zeros([4096], dtype=tl.float32)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (pooler + router) is the producer's output
    for s in range(S):
        b = P + (s * T + t) * 4608
        logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
        pooled += tl.load(b + c)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _route_split(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
    """Same as _route but the pooler output (bf16) and the router logits (fp32) arrive as separate tensors."""
    t = tl.program_id(0)
    e = tl.arange(0, 256)
    c = tl.arange(0, 4096)
    biasv = tl.load(BIAS + e)                      # weights and local_start: no producer dependency
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    lsv = tl.load(LS)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # pooled / logits are the producer's output
    logit = tl.load(LOGITS + t * 256 + e)
    pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + biasv
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - lsv + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    r = kk % 16
    if PERMUTED:
        po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
        tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
    else:
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _pexp_norm_sum(x_ptr, w_ptr, s_ptr, out_ptr, ids_ptr, TOPK: tl.constexpr, H: tl.constexpr, eps,
                   BLOCK_H: tl.constexpr):
    """post-expert norm (weight first) * routing weight, summed over slots. Same kernel as SGLang's
    fused_post_expert_norm_weighted_sum (Apache 2.0), called directly: its wrapper falls back to a chain of
    ~8 torch ops below 32 tokens, which costs 39-43 us on the judge instead of ~6."""
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    w = 1.0 + tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)   # the checkpoint's BF16 w; 1 + w formed here
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        if tl.load(ids_ptr + tok * TOPK + k) >= 0:
            x = tl.load(x_ptr + (tok * TOPK + k) * H + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * w
            y = y * tl.rsqrt(tl.sum(y * y, axis=0) / H + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _boundary2(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, GNEXT, SOUT, HOUT,
               USE_PDL: tl.constexpr = False):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    m = offs < NU
    gp = 1.0 + tl.load(GPOST + offs).to(tl.float32)      # weights, and S1 from _boundary1: every kernel between them (pooler/router, route,
    gn = 1.0 + tl.load(GNEXT + offs).to(tl.float32)      # moe, unpooler) triggers only after its own wait, so _boundary1 completed before this kernel began
    a2v, b2v = _gate_ab(A2, offs)
    s1v = tl.load(S1 + t * 8192 + offs)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()   # trigger early: dependents may become resident now
        tl.extra.cuda.gdc_wait()   # P (the unpooler) is the producer's output
    f = tl.zeros([8192], dtype=tl.float32)
    for s in range(S):
        f += tl.load(P + (s * T + t) * NU + offs, mask=m, other=0.0)
    f = f * gp
    n = f * tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
    s2 = a2v * s1v + b2v * n
    tl.store(SOUT + t * 8192 + offs, s2)
    hh = s2 * tl.rsqrt(tl.sum(s2 * s2, 0) / 8192.0 + 1e-5) * gn
    tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))





def _marlin_experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs + post-expert-norm finalize (as baseline.py).
    moe_block_size and use_atomic_add come from the prepare()-time sweep in _tune."""
    from sgl_kernel.scalar_type import scalar_types
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = topk_ids  # already (expert - local_start) % 256 mapped to 0..31 / -1 by the router kernel
    mc = state.get("marlin_cfg", {})
    cfgm = mc.get(num_tokens, mc.get(_bucket(num_tokens)))
    if cfgm is None:
        for block_size_m in (8, 16, 32, 48, 64):
            if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
                break
        atomic = False
    else:
        block_size_m, atomic = cfgm
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    m = torch.empty((num_tokens, size_k), device=x.device, dtype=x.dtype)
    _pexp_norm_sum[(num_tokens,)](y, state["g_postexp"], topk_weights, m, local_ids, TOPK=TOP_K,
                                  H=size_k, eps=1e-8, BLOCK_H=4096, num_warps=8, num_stages=2)
    return m


def _graph_time(fn, flush, reps=9):
    """Median graph-replay time with a cold L2. The flush is a READ sweep: zeroing a buffer leaves ~120 MB of
    dirty lines to write back, which inflates the next kernel by 15-25% (measured on the judge's H200) and not
    evenly across kernels, so it would skew every choice below."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        _FLUSH_SINK.copy_(flush.sum())
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


_FLUSH_SINK = None


def _calib_bias(st, dev, g, log=False):
    """Give the 32 local experts a skewed bias so that synthetic routing matches the judge's busiest-range load:
    ~1.6 local routes per token and ~15.7 distinct local experts at 32 tokens (reference statistics). Two
    parameters (skew width, bias size) found by a torch simulation of the real router; no input data involved."""
    router = st["pr"][D_EXPERT:].float()  # [256 hi; 256 lo] rows of width HIDDEN
    router = router[:N_EXPERTS] + router[N_EXPERTS:]
    gc = torch.Generator(device=dev).manual_seed(987654321)
    h = torch.randn(512, HIDDEN, generator=gc, device=dev)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
    sc = torch.sigmoid(h.double() @ router.double().T * ROUTER_MULT).float()
    idx = torch.arange(LOCAL_EXPERTS, device=dev).float()

    def stats(shape, delta):
        sel = sc + st["bias"]
        sel[:, :LOCAL_EXPERTS] += delta * shape
        ids = sel.topk(TOP_K, dim=-1).indices
        loc = ids < LOCAL_EXPERTS
        rpt = float(loc.float().sum(-1).mean())
        counts = [torch.unique(ids[i:i + 32][loc[i:i + 32]]).numel() for i in range(0, 512, 32)]
        d32 = sum(counts) / len(counts)
        return rpt, d32

    best = None
    for width in (2.0, 3.0, 4.5, 6.0, 9.0):
        shape = torch.exp(-idx / width)
        lo, hi = 0.0, 3.0
        for _ in range(14):
            mid = 0.5 * (lo + hi)
            if stats(shape, mid)[0] < 1.6:
                lo = mid
            else:
                hi = mid
        rpt, d32 = stats(shape, hi)
        if best is None or abs(d32 - 15.7) < abs(best[3] - 15.7):
            best = (shape, hi, rpt, d32)
    st["bias"][:LOCAL_EXPERTS] += best[1] * best[0]
    if log:
        print(f"calib delta {best[1]:.3f} routes/token {best[2]:.2f} experts@32 {best[3]:.1f}", flush=True)


def _tune(st, log=False):
    """prepare()-time choice of the post-attention path per token count: the persistent megakernel or the
    multi-kernel pipeline, and which routed-expert kernel. Uses synthetic activations only (never input data)."""
    dev = st["qkv"].device
    global _FLUSH_SINK
    flush = torch.zeros(int(60e6), dtype=torch.float32, device=dev)  # 240 MB, read-swept between replays
    _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(12345)
    bias0 = st["bias"].clone()
    _calib_bias(st, dev, g, log)
    st["moe_pick"] = {}
    st["gluev2"] = {}
    st["marlin_cfg"] = {}
    st["mm_pick"] = {}
    for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        data = {"stream": torch.randn(T, HIDDEN, generator=g, device=dev),
                "local_start": torch.zeros(1, dtype=torch.int32, device=dev)}
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        if T <= 4:  # redraw until the token(s) route to about as many local experts as the judge's inputs do
            want = max(2, int(round(2.57 * T / 1.0)) if T == 1 else 2 * T)
            for _ in range(40):
                _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                if int(((_ids >= 0) & (_ids < LOCAL_EXPERTS)).sum()) >= min(want, 8 * T):
                    break
                data["stream"] = torch.randn(T, HIDDEN, generator=g, device=dev)
                o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        best, bestt, bestv2 = None, 1e30, True
        times, dfltv2 = {}, False
        msg = []
        nloc = ndis = -1
        for name, w, K in (("qkv", st["qkv"], HIDDEN), ("o", st["o"], 1024), ("pr", st["pr"], HIDDEN),
                           ("un", st["unpool_slice"] if T <= 8 else st["unpool"], D_EXPERT)):
            xd = (torch.randn(T, K, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            if st.get("bfp") is not None:
                st["mm_pick"][(T, name)] = "bfp"
                try:
                    if name == "pr":
                        t_bfp = _graph_time(lambda: (_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                                     _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                            flush, reps=5)
                    else:
                        nn = w.shape[0]
                        t_bfp = _graph_time(lambda: _bfp_gemm(st, name, xd, T, nn, K), flush, reps=5)
                except Exception:  # noqa: BLE001
                    t_bfp = 1e30
            else:
                t_bfp = 1e30
            st["mm_pick"][(T, name)] = False
            try:
                t_tri = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
            except Exception:  # noqa: BLE001
                t_tri = 1e30
            st["mm_pick"][(T, name)] = True
            try:
                t_mm = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
            except Exception:  # noqa: BLE001
                t_mm = 1e30
            t_bfpf = 1e30
            if name == "pr" and T <= 32 and st.get("bfp") is not None and "bfp_fused_pooler_router" in globals():
                try:
                    t_bfpf = _graph_time(lambda: bfp_fused_pooler_router(xd, st["bfp"]), flush, reps=5)
                except Exception:  # noqa: BLE001  (sm_80 or an older bf16_proj file: not offered)
                    t_bfpf = 1e30
            opts = {False: t_tri, True: t_mm, "bfp": t_bfp, "bfpf": t_bfpf}
            bestp = min(opts, key=lambda p: opts[p])
            st["mm_pick"][(T, name)] = bestp
            if log:
                msg.append(f"{name}{ {False: 'T', True: 'M', 'bfp': 'B', 'bfpf': 'F'}[bestp] }:{opts[bestp]:.0f}")
        if st["bank"] is not None and T <= 384:
            xm = (torch.randn(T, D_EXPERT, generator=g, device=dev) * 0.1).to(torch.bfloat16)
            raw = torch.randint(0, N_EXPERTS, (T, 8), generator=g, device=dev, dtype=torch.int32)
            mids = torch.where(raw < LOCAL_EXPERTS, raw, torch.full_like(raw, -1)).contiguous()
            mwt = torch.full((T, 8), 0.125, device=dev)
            bestm, bestmt = None, 1e30
            for bs in (8, 16, 32, 48, 64):
                for atomic in (False, True):
                    st["marlin_cfg"][T] = (bs, atomic)
                    try:
                        tm = _graph_time(lambda: _marlin_experts(xm, mwt, mids, st), flush, reps=5)
                    except Exception:  # noqa: BLE001
                        continue
                    if tm < bestmt:
                        bestm, bestmt = (bs, atomic), tm
            if bestm is not None:
                st["marlin_cfg"][T] = bestm
                if log:
                    msg.append(f"mcfg{bestm[0]}{'a' if bestm[1] else ''}:{bestmt:.0f}")
            else:
                st["marlin_cfg"].pop(T, None)
        cands = [c for c in ("marlin", "agent") if (c != "marlin" or st["bank"] is not None)
                 and (c != "agent" or st["ma_ok"])]
        for moe in cands:
            if moe == "marlin" and st["bank"] is None:
                continue
            if moe == "agent" and not st["ma_ok"]:
                continue
            st["moe_pick"][T] = moe
            for v2 in (False,):  # Parallel boundary search lost every complete-chain sweep.
                st["gluev2"][T] = v2
                try:
                    t = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
                except Exception as ex:  # noqa: BLE001
                    if log:
                        msg.append(f"{moe[:3]}{2 if v2 else 1}:X{repr(ex)[:40]}")
                    continue
                if log:
                    msg.append(f"{moe[:3]}{2 if v2 else 1}:{t:.0f}")
                    if nloc < 0:
                        _o, _s, _ids = _after_attention(data, st, o_in, None, T)
                        loc = _ids[(_ids >= 0) & (_ids < LOCAL_EXPERTS)]
                        nloc, ndis = int(loc.numel()), int(torch.unique(loc).numel())
                if t < times.get(moe, 1e30):
                    times[moe] = t
                    if moe == DEFAULT_MOE:
                        dfltv2 = v2
                if t < bestt:
                    best, bestt, bestv2 = moe, t, v2
        if (DEFAULT_MOE in times and best is not None and best != DEFAULT_MOE
                and times[best] > TUNE_MARGIN * times[DEFAULT_MOE]):
            best, bestv2 = DEFAULT_MOE, dfltv2
        st["moe_pick"][T] = best if best is not None else DEFAULT_MOE
        st["gluev2"][T] = bestv2
        if log:
            print(f"T{T}/{nloc}r{ndis}e " + " ".join(msg), flush=True)
    st["bias"].copy_(bias0)
    st["pbmax"] = 0 if FORCE_PBMAX is None else FORCE_PBMAX
    if log:
        print("tune: pbmax", st["pbmax"], "moe", st["moe_pick"], "gluev2", st["gluev2"], flush=True)
    del flush


def prepare(weights):
    dev = weights["qkv_gate"].device
    router = weights["router"].float()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    st = {}
    st["router_fp32"] = router.contiguous()
    st["qkv"] = weights["qkv_gate"].contiguous()
    st["o"] = weights["o"].contiguous()
    st["pr"] = torch.cat([weights["pooler"], hi, lo], 0).contiguous()
    st["unpool"] = weights["unpooler"].contiguous()
    st["unpool_slice"] = weights["unpooler"][:1024].contiguous()
    st["gate1"] = weights["attn_residual_gate"].contiguous()
    st["gate2"] = weights["ffn_residual_gate"].contiguous()
    st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
    st["g_preexp"] = weights["pre_expert_norm"].contiguous()
    st["g_postexp"] = weights["post_expert_norm"].contiguous()   # BF16 as given; _pexp_norm_sum forms 1 + w
    st["g_postffn"] = weights["post_ffn_norm"].contiguous()
    st["g_next"] = weights["next_pre_attn_norm"].contiguous()
    st["bias"] = weights["router_bias"].float().contiguous()
    st["w13q"] = _repack_q(weights["w13"].view(torch.uint8), True)
    st["w13s"] = _repack_s(weights["w13_sf"].view(torch.uint8), True)
    st["gs13"] = weights["w13_gs"].float().contiguous()
    st["w2q"] = _repack_q(weights["w2"].view(torch.uint8), False)
    st["w2s"] = _repack_s(weights["w2_sf"].view(torch.uint8), False)
    st["gs2"] = weights["w2_gs"].float().contiguous()
    st["invf0"] = torch.zeros(32, device=dev)
    st["invf"] = (1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=dev).float() / HEAD_DIM))).contiguous()
    errs = []

    def _build(fn):
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            errs.append(ex)

    global _MA_SFX, _MA_SC6W
    _MA_SFX = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
    _MA_SC6W = bool(SC6_WGS and _MA_SFX and _wgs_on() and _sc6_ok(weights["w13_sf"], weights["w2_sf"]))
    _p9_sc5w_decide(weights)
    print(f"[sc6] scales ok -> wgs sc6 {_MA_SC6W}", flush=True)
    st["ma_sfx"] = tuple(_MA_SFX)
    st["ma_ext"] = {}
    ths = [threading.Thread(target=_build, args=(f,)) for f in (_att7, _moe, _moe_tiny, _moe_tiny2, _moe_mid, bfp__get_ext, _graft_glue_module, _graft_b2_module, _sl_cluster_module, lambda: _moe(32))]  # ~85 s of nvcc, in parallel
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    if errs:
        raise errs[0]
    _ma_prepare(weights, st)
    if _MA_SC6W:
        try:
            for d in (tuple(MOE_DEFINES), tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)):
                print("[sc6] regs/local/smem (<false>, <true>)", d[-1], _ma_ext(d + tuple(_MA_SFX)).okc_attrs(), flush=True)
        except Exception as ex:
            print(f"[sc6] attrs failed: {ex!r}"[:200], flush=True)
    _bfp_prepare(weights, st)
    st["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count
    st["fa3"] = None
    st["bank"] = None
    st["att_ok"] = True
    st["ma_ok"] = True
    st["pbmax"] = 2
    st["moe_pick"] = {}
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        st["fa3"] = flash_attn_with_kvcache
        st["bank"] = None  # Native experts cover supported shapes; omit unused Marlin packing.
    except Exception as ex:  # noqa: BLE001  (local GPUs without sgl_kernel: fall back to our own kernels)
        print("fallback (no sgl_kernel/sglang):", repr(ex)[:200])
    try:
        _tune(st, log=_TUNE_LOG)
    except Exception as ex:  # noqa: BLE001
        print("tune failed:", repr(ex)[:200], flush=True)
        st["pbmax"] = 2
        st["moe_pick"] = {}
        st["gluev2"] = {}
        st["marlin_cfg"] = {}
        st["mm_pick"] = {}
    for k, v in PIN_PICK.items():      # in-chain winners the isolated tuner sometimes misses (no PDL in its timing)
        st.setdefault("mm_pick", {})[k] = v
    torch.cuda.synchronize()
    if st.get("att_ok"):
        _attn_selftest(st)     # every attention plan family once, on synthetic inputs
    _check_guards()            # after the tuner (every moe build) and the attention self-test
    return st


def _attn_cfg(B, L, sliding):
    BM = 16 if L == 1 else 64
    if sliding:
        nsplit = max(1, min(32, 264 // B))
    else:
        nsplit = max(1, min(132, 528 // B))
    return BM, nsplit


_INV_FREQ_CACHE = {}


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    t = _INV_FREQ_CACHE.get(key)
    if t is None:
        if theta:
            t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
        else:
            t = torch.zeros(32, device=device)
        _INV_FREQ_CACHE[key] = t.float().contiguous()
    return t


def _launch7(data, dbg=None, _mod=None, pdl=None, **override):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    qkv = data["qkv"]
    device = qkv.device
    pairs = hkv // 2
    window = cfg["window"]
    page_table = data["page_table"]
    out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
    plan = _plan7(B, pairs, L, window, page_table.shape[1], P, device)
    tuned = _TUNED7.get((B, L, bool(window), pairs))
    if tuned and not override:
        override = tuned
    if override:
        plan = dict(plan, **override)
        if "cs" in override and override["cs"] > 1:
            plan["flat"] = 0
            if "ncl" not in override:
                plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
            plan["n_splits"] = plan["cs"] * plan["ncl"]
            if "nc" not in override:
                plan["nc"] = max(1, min(plan["ncl"], 4))
        elif "n_splits" in override and "nc" not in override:
            plan["nc"] = 1 if plan["flat"] else max(1, min(plan["n_splits"], 4))
    n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
    if flat:   # grid-wide finalize after all segments (the kernel re-checks residency)
        coop = int(plan.get("coop", int(OKC_FLATCOOP and L == 8 and
                                        n_splits * pairs <= 2 * _num_sms7(device))))
    else:
        coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                            int(n_splits * B * pairs <= 2 * _num_sms7(device))))
    use_pdl = OKC_PDL if pdl is None else bool(pdl)
    tma = (coop << 1) | (int(use_pdl) << 16)   # board kernel bits: 1 = cooperative finalize, 16 = PDL
    if flat or cs <= 1:
        cs, ncl = 0, 1
    ws_acc, ws_l, ws_cnt = _workspace7(B * pairs * nc * L * 16, device)
    scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    (_mod or _att7()).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                  data["k_scale"], data["v_scale"], _inv_freq(cfg["rope_theta"], device), out, ws_acc, ws_l,
                  ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                  dbg, cs, ncl, tma, *(_PF7 or (0, 0)))
    return out


def _supported7(cfg):
    P = cfg["page_size"]
    return (cfg["head_dim"] == 64 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
            and cfg["q_len"] in (1, 8) and cfg["batch"] <= 512 and P > 0 and (P & (P - 1)) == 0
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)

ATT_HAS_PDL = True   # does the embedded attention launcher take pdl= (their v63+)?


def _decode_attention7(data, state, qkv):
    """The attention launcher _launch7: plan and workspace sizes as the board file t853_both's _plan / _launch
    (s302: the embedded kernel is the board build; its mode bits are coop (1) and PDL (16), no w0 field).
    This layer's config already carries their keys (heads, head_dim, page_size, window, rope_theta, scales)."""
    cfg = data["config"]
    pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"])} if ATT_HAS_PDL else {}   # their launcher's PDL switch
    global _PF7
    bst = state.get("bfp")
    _PF7 = None
    if bst is not None and "pooler" in bst.get("sk2", {}) and cfg["batch"] * cfg["q_len"] in (1, 8):
        _ps = bst["sk2"]["pooler"][1]
        _PF7 = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))
    out = _launch7({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                    "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                    "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)
    _PF7 = None   # the pointer belongs to this call's state only
    return out.view(cfg["batch"] * cfg["q_len"], Q_DIM)


_GUARDS = []
_GUARDED_IDS = set()


def _guarded(t):
    nbytes = t.numel() * t.element_size()
    words = (nbytes + 3) // 4
    full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL, dtype=torch.int32, device=t.device)
    full.view(torch.uint8)[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
    v = full.view(torch.uint8)[:nbytes].view(t.dtype).view(t.shape)
    _GUARDS.append((full, words))
    _GUARDED_IDS.add(id(v))
    return v


def _check_guards():
    torch.cuda.synchronize()
    for full, words in _GUARDS:
        bad = int((full[words:] != SCRATCH_SENTINEL).sum())
        if bad:
            raise RuntimeError(f"a component kernel wrote {bad} words past a {words * 4}-byte buffer")


def _attn_selftest(st):
    """Run the attention kernel once per plan family on synthetic inputs (problem constants, no data), so its
    lazily allocated buffers exist and are guarded before any timed call, and the guards see every path."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(4242)
    nq = st["qkv"].shape[0]
    for window, L, B, ctx, P in ((2048, 1, 1, 3000, 1), (2048, 1, 16, 2500, 1), (2048, 1, 32, 2500, 1),
                                 (0, 1, 1, 3000, 1), (0, 1, 16, 1500, 1), (0, 1, 33, 1000, 1),
                                 (0, 8, 1, 3000, 1), (0, 8, 16, 1500, 1), (0, 8, 32, 800, 1),
                                 (2048, 1, 4, 3000, 64), (0, 8, 4, 3000, 64)):
        cfg = dict(batch=B, q_len=L, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, page_size=P, window=window,
                   rope_theta=ROPE_THETA if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125)
        if not _supported7(cfg):
            continue
        seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
        per = (ctx + L + P - 1) // P
        pt = torch.arange(B * per, dtype=torch.int32, device=dev).view(B, per)
        pos = (seq[:, None].long() + torch.arange(L, device=dev)).view(B, L)
        loc = (torch.gather(pt.long(), 1, pos // P) * P + pos % P).reshape(-1)
        kc = torch.zeros((B * per, P, 2, 64), dtype=torch.float8_e4m3fn, device=dev)
        vc = torch.zeros_like(kc)
        qkv = (torch.randn(B * L, nq, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        one = torch.ones((), dtype=torch.float32, device=dev)
        _decode_attention7({"config": cfg, "seq_lens": seq, "cache_loc": loc, "k_cache": kc, "v_cache": vc,
                            "page_table": pt, "k_scale": one, "v_scale": one}, st, qkv)

def _att_fits7(data, cfg):
    """The attention entry's launcher host-checks these sizes (32-bit index math, OKC_OFF32) and raises otherwise;
    mirrored here so such shapes take SGLang's FA3 instead (fuzz: batch 64 x 150k context exceeds the caches' 4 GB)."""
    kc, vc = data["k_cache"], data["v_cache"]
    rows = kc.numel() // max(1, cfg["num_kv_heads"] * cfg["head_dim"])
    return (kc.numel() < (1 << 32) and vc.numel() < (1 << 32) and rows < (1 << 31)
            and data["page_table"].numel() < (1 << 31)
            and cfg["batch"] * cfg["q_len"] * QKV_ROWS < (1 << 31)
            and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def _kernel_fn_fa3(data, state):
    st2 = dict(state)
    st2["att_ok"] = False
    return kernel_fn(data, st2)


def kernel_fn(data, state):
    cfg = data["config"]
    B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
    agent_attn = state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
    if agent_attn:
        qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
        if qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
            qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
        elif qpick is True or qpick == "bfp":
            qkv = torch.mm(data["attn_input"], state["qkv"].t())
        else:
            Pq, Sq = dense(data["attn_input"], state["qkv"], T)
            qkv = Pq[0] if Sq == 1 else Pq.sum(0)
            qkv = qkv.to(torch.bfloat16)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    dev = data["attn_input"].device
    x = data["attn_input"]
    sliding = cfg["window"] > 0
    use_fa3 = state["fa3"] is not None
    P, S = dense(x, state["qkv"], T)
    q = torch.empty((T, Q_DIM), device=dev, dtype=torch.bfloat16)
    q8 = torch.empty((T, Q_DIM), device=dev, dtype=torch.uint8)
    gate = torch.empty((T, 1024), device=dev, dtype=torch.float32)
    kc = data["k_cache"].view(torch.uint8)
    vc = data["v_cache"].view(torch.uint8)
    _qk_prep[(T,)](P, S, T, data["positions"], data["cache_loc"], state["invf"], kc, vc,
                   data["k_scale"], data["v_scale"], q, q8, gate, cfg["qk_scale_factor"],
                   ROPE=bool(cfg["rope_theta"]), num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_fa3:
        descale = {}
        if cfg["kv_scale"] != 1.0:
            descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                           v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
        window = cfg["window"]
        out = state["fa3"](q=q8.view(torch.float8_e4m3fn).view(T, 16, 64), k_cache=data["k_cache"],
                           v_cache=data["v_cache"], page_table=data["page_table"],
                           cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
                           max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                           window_size=(window - 1, 0) if window else (-1, -1),
                           scheduler_metadata=data["scheduler_metadata"], **descale)
        if isinstance(out, tuple):
            out = out[0]
        o_in = out.reshape(T, 1024)
    else:
        BM, NS = _attn_cfg(B, L, sliding)
        acc = torch.empty((B * 2 * NS * BM, 64), device=dev, dtype=torch.float32)
        ml = torch.empty((B * 2 * NS * BM, 2), device=dev, dtype=torch.float32)
        pt = data["page_table"]
        _attn_split[(B, NS)](q, kc, vc, pt, data["seq_lens"], data["k_scale"], acc, ml,
                             pt.stride(0), cfg["page_size"], cfg["window"], cfg["sm_scale"],
                             L=L, NSPLIT=NS, BM=BM, BN=64, SLIDING=sliding, num_warps=4, num_stages=3)
        o_in = torch.empty((T, Q_DIM), device=dev, dtype=torch.bfloat16)
        _attn_combine[(T, 16)](acc, ml, gate, data["v_scale"], o_in, L=L, NSPLIT=NS, BM=BM,
                               SB=min(64, triton.next_power_of_2(NS)), NORMGATE=False, num_warps=4)
    return _after_attention(data, state, o_in, gate, T)


def _pick_expert_backend(pick, marlin_available, agent_available):
    """The expert-backend fallback, factored out of _after_attention: Marlin
    without its bank falls back to the agent kernel, and vice versa. The
    three steps are kept in order (not a loop): when neither backend is
    available the original code terminates with "agent", and so does this."""
    if pick == "marlin" and not marlin_available:
        pick = "agent"
    if pick != "marlin" and not agent_available:
        pick = "marlin"
    if pick == "marlin" and not marlin_available:
        pick = "agent"
    return pick


def _after_attention(data, state, o_in, gate, T):
    """o projection onwards: residual boundary, router, experts, unpooler, final boundary.
    gate is None when o_in already carries rmsnorm(o) * sigmoid(gate). state["gluev2"] picks between the
    one-program-per-token boundary kernels and the parallel (atomic-sum) ones; the tuner chooses per T."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    gv = state.get("gluev2", {})
    v2 = gv.get(T, gv.get(_bucket(T), False)) if FORCE_GLUEV2 is None else FORCE_GLUEV2
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if v2:
        a_out = torch.empty((T, HIDDEN), **f32)
        sums = torch.zeros((T, 4), **f32)
        BN, BT, BK = 64, (16 if T <= 16 else 32), 128
        _gemm_o[(HIDDEN // BN, triton.cdiv(T, BT))](
            o_in, gate, state["o"], a_out, sums, data["stream"], state["gate1"], state["gate1"], T, HIDDEN,
            K=1024, BN=BN, BT=BT, BK=BK, NORMGATE=gate is not None, num_warps=4, num_stages=3)
        BC1 = 2048
        _bnd1_apply[(T, HIDDEN // BC1)](a_out, sums, data["stream"], state["gate1"], state["gate1"], state["g_preffn"],
                                        s1, h, N=HIDDEN, BC=BC1, num_warps=8)
    else:
        if gate is None:
            P, S = dense(o_in, state["o"], T, state, "o")
        else:
            P, S = dense_ng(o_in, gate, state["o"], T, state)
        _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,
                         WRITE_ROUTER=T <= 8, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if T <= 8 and v2:
        _router_input_norm[(T,)](s1, state["g_preffn"], hr, num_warps=8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:  # pooler and router came from the bf16_proj kernels as two separate tensors
        pooled, logits = P
        _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                           data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    elif v2:
        pool = torch.empty((T, D_EXPERT), **f32)
        psum = torch.zeros((T,), **f32)
        BCR = 1024
        _route_a[(T, D_EXPERT // BCR)](P, S, T, state["bias"], ids, wts, data["local_start"], lids, pool, psum,
                                       BC=BCR, num_warps=8)
        _route_b[(T, D_EXPERT // BCR)](pool, psum, state["g_preexp"], xm, BC=BCR,
                                       PERMUTED=not (use_marlin or use_ma), num_warps=8)
    elif (T > 32 and S == 1 and state.get("compose_route_graft", {}).get(T, 0) and P.shape[-1] == 4608
          and P.dtype == torch.float32 and P.is_contiguous()):
        # g121: base boundaries + the grafted route (same FP32 [T, 4608] pooler | router hi | lo input as _route
        # with S == 1; same ids / lids / wts / non-permuted bf16 xm outputs). 1: plain launch, 2: PDL launch.
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        _graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=_GRAFT_ROUTE_EARLY, PACKED=1, num_warps=4,
                             launch_pdl=state["compose_route_graft"][T] == 2)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)   # the build their kernel_fn picks for this M, then at most MMAX tokens per launch, as theirs
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):      # (their scratch is sized for MMAX: T > 384 would overrun it)
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])     # lids: local expert index, -1 = another GPU's expert
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if v2:
        fbuf = torch.zeros((T, HIDDEN), **f32)
        sums2 = torch.zeros((T, 4), **f32)
        BC2 = 1024
        _bnd2_sums[(T, HIDDEN // BC2)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], fbuf, sums2,
                                       BC=BC2, num_warps=8)
        _bnd2_apply[(T, HIDDEN // BC2)](fbuf, sums2, s1, state["gate2"], state["gate2"], state["g_next"], s2, out,
                                        N=HIDDEN, BC=BC2, num_warps=8)
    else:
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids


# A_graft_glue_d1_v1_v32 graft block (appended to the vjeux base, everything above is unchanged).
# Portions from vjeux/okc-2026 (Apache-2.0): the base above, and _graft_after_attention below, which is
# Grafted parts are our team's own code (Apache-2.0 submission): glue_cluster_boundary (8-CTA cluster
import ctypes as _C
import hashlib as _hashlib
from functools import lru_cache as _lru_cache
from triton.language.extra.cuda import gdc_wait as _gdc_wait, gdc_launch_dependents as _gdc_launch
from triton.experimental import gluon as _glx
from triton.experimental.gluon import language as _gl

_GRAFT_GLUE_T = (1, 8, 16, 32, 128, 256)      # token counts that take the grafted boundary + route kernels
_GRAFT_LT_T = (128, 256)          # token counts that take the cuBLASLt picks
_GRAFT_ROUTE_EARLY = False   # route kernel: trigger dependents before (True) or after (False) its wait

_GRAFT_GLUE_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
namespace cg = cooperative_groups;
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

struct GlueArgs {
  const BF *branch, *post, *pre;
  const float *part, *stream, *alpha, *beta;
  BF *out;
  float *next, *router;
  int rows;
};

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
template<int V> __device__ __forceinline__ void glf(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
      :"=f"(x[0]),"=f"(x[1]),"=f"(x[2]),"=f"(x[3]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];"
      :"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
  }
}
template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
template<int V> __device__ __forceinline__ void gsf(float* p, const float* x) {
  if constexpr (V == 4) {
    asm volatile("st.global.v4.f32 [%0],{%1,%2,%3,%4};"
      ::"l"(p),"f"(x[0]),"f"(x[1]),"f"(x[2]),"f"(x[3]):"memory");
  } else {
    asm volatile("st.global.v2.f32 [%0],{%1,%2};"
      ::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
  }
}
template<int V> __device__ __forceinline__ void gsb(BF* p, const float* x) {
  unsigned u[V/2];
  #pragma unroll
  for(int j=0;j<V/2;j++) {
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u[j]):"f"(x[2*j+1]),"f"(x[2*j]));
  }
  if constexpr(V==4) {
    asm volatile("st.global.v2.b32 [%0],{%1,%2};"::"l"(p),"r"(u[0]),"r"(u[1]):"memory");
  } else {
    asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u[0]):"memory");
  }
}

template<int N> __device__ __forceinline__ float local_squares(const float (&x)[N]) {
  float q[N/2];
  #pragma unroll
  for(int j=0;j<N/2;j++) q[j]=gf(x[2*j],x[2*j],gm(x[2*j+1],x[2*j+1]));
  #pragma unroll
  for(int width=N/2;width>1;width/=2) {
    #pragma unroll
    for(int j=0;j<width/2;j++) q[j]=ga(q[2*j],q[2*j+1]);
  }
  return q[0];
}
__device__ __forceinline__ float butterfly(float x, int first) {
  #pragma unroll
  for(int d=first;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares(const float (&x)[N], float* sm,
                                                cg::cluster_group cluster, int rank) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sum=butterfly(local_squares(x),16);
  if(lane<8 && warp<warps_per_cta) {   // a202: lane d stores to rank d (every lane holds the same sum)
    volatile float* p=cluster.map_shared_rank(sm,lane);
    p[rank*warps_per_cta+warp]=sum;
  }
  cluster.sync();
  float all=sm[lane&(NW-1)];
  return butterfly(all,NW/2);
}

template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>
__global__ __launch_bounds__(128) void glue_cluster_boundary(GlueArgs a) {
  constexpr int logical_threads=NW*32;
  constexpr int per_cta=logical_threads/8;
  constexpr int N=8192/logical_threads;
  constexpr int repeats=N/V;
  cg::cluster_group cluster=cg::this_cluster();
  int rank=cluster.block_rank(), tid=threadIdx.x;
  int logical=rank*per_cta+(tid%per_cta);
  int row=blockIdx.x/8;
  bool active=tid<per_cta;
  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N];
  __shared__ float partials[2][NW];
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    glf<V>(a.stream+row*8192+col,s+r*V);
    glf<V>(a.alpha+col,alpha+r*V);
    glf<V>(a.beta+col,beta+r*V);
    glb<V>(a.pre+col,pre+r*V);
    if constexpr(POST) glb<V>(a.post+col,post+r*V);
    if constexpr(SLICED) {
      if(col>=1024) {
        #pragma unroll
        for(int j=0;j<V;j++) {
          int k=r*V+j;
          ns[k]=gf(alpha[k],s[k],gm(beta[k],gm(0.f,post[k])));
        }
        if(active) gsf<V>(a.next+row*8192+col,ns+r*V);
      }
    }
  }
  cluster.sync();
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if constexpr(SLICED) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=0.f;
      if(col<1024) {
        #pragma unroll
        for(int sk=0;sk<8;sk++) {
          float part[V];glf<V>(a.part+(sk*a.rows+row)*1024+col,part);
          #pragma unroll
          for(int j=0;j<V;j++) b[r*V+j]=ga(b[r*V+j],part[j]);
        }
      }
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(b[r*V+j]);
    } else {
      glb<V>(a.branch+row*8192+col,b+r*V);
    }
    if constexpr(POST) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(gm(b[r*V+j],post[r*V+j]));
    }
  }
  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if(!SLICED || col<1024) {
      #pragma unroll
      for(int j=0;j<V;j++) {
        int k=r*V+j;
        float bn=gb(gm(b[k],r1));
        if(SLICED || (PRELOADED_BASE && col<4096))
          ns[k]=gf(alpha[k],s[k],gm(beta[k],bn));
        else
          ns[k]=gf(beta[k],bn,gm(alpha[k],s[k]));
      }
    }
  }
  // a201: the `next` stores follow the second reduction (its cluster barrier's MEMBAR.ALL.GPU would wait for them)
  float r2=gr(cluster_squares<NW>(ns,partials[1],cluster,rank),1.e-5f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    float h[V];
    #pragma unroll
    for(int j=0;j<V;j++) h[j]=gm(gm(ns[r*V+j],r2),pre[r*V+j]);
    if(active) {
      gsb<V>(a.out+row*8192+col,h);
      if(!SLICED || col<1024) gsf<V>(a.next+row*8192+col,ns+r*V);
      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
    }
  }
}

template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(a.rows*8);cfg.blockDim=dim3(128);cfg.stream=stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim={8,1,1};
  attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=2;
  auto e=cudaLaunchKernelEx(&cfg,fn,a);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV router,bool write_router) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0)};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b1: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(write_router) {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,true,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,true,false>,s);
  } else {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,false,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,false,false>,s);
  }
}
void b2c(TV branch,TV stream,TV post,TV pre,TV alpha,TV beta,TV out,TV next) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)post.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)next.data_ptr(),(int)stream.size(0)};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b2c: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,true,false,false,true>,s);
  else glue_launch(a,glue_cluster_boundary<4,32,true,false,false,false>,s);
}
'''
_GRAFT_B2_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
__device__ __forceinline__ void lf2(const float* p, float* x) {
  float2 v = *reinterpret_cast<const float2*>(p); x[0]=v.x; x[1]=v.y;
}
__device__ __forceinline__ void lb2(const BF* p, float* x) {
  unsigned u = *reinterpret_cast<const unsigned*>(p);
  x[0]=__uint_as_float(u<<16); x[1]=__uint_as_float(u&0xffff0000u);
}
__device__ __forceinline__ void glf2(const float* p, float* x) {
  asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
}
__device__ __forceinline__ void gsf2(float* p, const float* x) {
  asm volatile("st.global.v2.f32 [%0],{%1,%2};"::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
}
__device__ __forceinline__ void gsb2(BF* p, const float* x) {
  unsigned u;
  asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u):"f"(x[1]),"f"(x[0]));
  asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u):"memory");
}
__device__ __forceinline__ float pair_sq(float x0, float x1) { return gf(x0,x0,gm(x1,x1)); }
__device__ __forceinline__ float butterfly(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}

template<int P,bool SBF>
__global__ __launch_bounds__(P) void b2_single(const float* part, const float* stream, const BF* post,
    const BF* pre, const float* alpha, const float* beta, BF* out, float* next, int rows) {
  constexpr int J=1024/P, JL=512/P, WP=P/32;
  const int t=threadIdx.x, lane=t&31, w=t>>5, row=blockIdx.x;
  __shared__ float sm1[32], sm2[32];
  const float* srow=stream+(size_t)row*8192;
  float* nrow=next+(size_t)row*8192;
  float ns[J][8], pr[J][8];
  float s0[JL][2], al[JL][2], be[JL][2], po[JL][2];
  float q1[JL], q23[JL];
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) {
      const int col=2*L+2048*r;
      lb2(pre+col,pr[j]+2*r);
      if(j<JL && r==0) {
        lf2(srow+col,s0[j]); lf2(alpha+col,al[j]); lf2(beta+col,be[j]); lb2(post+col,po[j]);
      } else {
        float s[2],a[2],b[2],p[2];
        lf2(srow+col,s); lf2(alpha+col,a); lf2(beta+col,b); lb2(post+col,p);
        #pragma unroll
        for(int k=0;k<2;k++) ns[j][2*r+k]=gf(a[k],s[k],gm(b[k],gm(0.f,p[k])));
      }
    }
  }
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) if(!(j<JL && r==0)) gsf2(nrow+2*L+2048*r,ns[j]+2*r);
  }
  #pragma unroll
  for(int j=0;j<J;j++) {
    if(j<JL) {
      q1[j]=pair_sq(ns[j][2],ns[j][3]);
      q23[j]=ga(pair_sq(ns[j][4],ns[j][5]),pair_sq(ns[j][6],ns[j][7]));
    } else {
      float v=ga(ga(pair_sq(ns[j][0],ns[j][1]),pair_sq(ns[j][2],ns[j][3])),
                 ga(pair_sq(ns[j][4],ns[j][5]),pair_sq(ns[j][6],ns[j][7])));
      v=butterfly(v);
      if(lane==0) { sm2[w+WP*j]=v; sm1[w+WP*j]=0.f; }
    }
  }
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  float b[JL][2];
  #pragma unroll
  for(int j=0;j<JL;j++) {
    const int col=2*(t+P*j);
    if constexpr(SBF) {
      unsigned u;
      asm volatile("ld.global.b32 %0,[%1];":"=r"(u):"l"(reinterpret_cast<const BF*>(part)+(size_t)row*1024+col):"memory");
      b[j][0]=gb(gm(__uint_as_float(u<<16),po[j][0]));
      b[j][1]=gb(gm(__uint_as_float(u&0xffff0000u),po[j][1]));
    } else {
    float pp[8][2];
    #pragma unroll
    for(int sk=0;sk<8;sk++) glf2(part+((size_t)(sk*rows+row))*1024+col,pp[sk]);
    #pragma unroll
    for(int k=0;k<2;k++) {
      float x=0.f;
      #pragma unroll
      for(int sk=0;sk<8;sk++) x=ga(x,pp[sk][k]);
      x=gb(x);
      b[j][k]=gb(gm(x,po[j][k]));
    }
    }
    const float z=pair_sq(0.f,0.f);
    float v=ga(ga(pair_sq(b[j][0],b[j][1]),z),ga(z,z));
    v=butterfly(v);
    if(lane==0) sm1[w+WP*j]=v;
  }
  __syncthreads();
  const float r1=gr(butterfly(sm1[lane]),1.e-8f);
  #pragma unroll
  for(int j=0;j<JL;j++) {
    const int col=2*(t+P*j);
    #pragma unroll
    for(int k=0;k<2;k++) {
      float bn=gb(gm(b[j][k],r1));
      ns[j][k]=gf(al[j][k],s0[j][k],gm(be[j][k],bn));
    }
    gsf2(nrow+col,ns[j]);
    float v=ga(ga(pair_sq(ns[j][0],ns[j][1]),q1[j]),q23[j]);
    v=butterfly(v);
    if(lane==0) sm2[w+WP*j]=v;
  }
  __syncthreads();
  const float r2=gr(butterfly(sm2[lane]),1.e-5f);
  BF* orow=out+(size_t)row*8192;
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) {
      float h[2];
      #pragma unroll
      for(int k=0;k<2;k++) h[k]=gm(gm(ns[j][2*r+k],r2),pr[j][2*r+k]);
      gsb2(orow+2*L+2048*r,h);
    }
  }
}


void b2s(TV branch,TV stream,TV post,TV pre,TV alpha,TV beta,TV out,TV next) {
  int rows=(int)stream.size(0);
  if(stream.size(1)!=8192 || branch.size(0)!=rows || branch.size(1)!=1024) throw std::runtime_error("graft b2s: shape");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(rows);cfg.blockDim=dim3(512);cfg.stream=s;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=1;
  auto e=cudaLaunchKernelEx(&cfg,b2_single<512,true>,(const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
      (const BF*)post.data_ptr(),(const BF*)pre.data_ptr(),(const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
      (BF*)out.data_ptr(),(float*)next.data_ptr(),rows);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
'''


@_lru_cache(None)
def _graft_glue_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='graft_glue_' + _hashlib.md5(_GRAFT_GLUE_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_GRAFT_GLUE_CUDA, functions=['b1', 'b2c'],
                            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                               '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])


@_lru_cache(None)
def _graft_b2_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='graft_b2_' + _hashlib.md5(_GRAFT_B2_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_GRAFT_B2_CUDA, functions=['b2s'],
                            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                               '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])


@_glx.jit
def _w7_l2pf(P, N: _gl.constexpr, STEP: _gl.constexpr, NW: _gl.constexpr):
    """w7 (EROREx, 2026-11-02): prefetch.global.L2 of P[0 : N], one per STEP elements (32 B); a cache hint only."""
    o = _gl.arange(0, N // STEP, layout=_gl.BlockedLayout([1], [32], [NW], [0])) * STEP
    _gl.inline_asm_elementwise("prefetch.global.L2 [$1]; mov.u32 $0, 0;", "=r,l", [P + o], dtype=_gl.int32,
                               is_pure=False, pack=1)


@_glx.jit
def _w7_row_boundary(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                     H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    """w7 (EROREx, 2026-11-02): _graft_row_boundary (T 33..192) in the _graft_row_boundary_pre structure."""
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    pw = _gl.load(PreW + c).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _gdc_wait()
    _gdc_launch()
    s = _gl.load(Stream + row * H + c)
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.float32)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    ns = _gl.fma(beta, b, a * s)   # = a * s + beta * b in the old kernel's FMA contraction (bitwise: tools/bnd_eq.py)
    _gl.store(NewStream + row * H + c, ns)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    h = ns * r * pw
    _gl.store(Out + row * H + c, h)


@_glx.jit
def _graft_row_boundary(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                        H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    row = _gl.program_id(0)
    _w7_l2pf(PreW, H, 16, NW)   # w7: weight vectors into L2 before the wait (their loads stay where they were)
    _w7_l2pf(Alpha, H, 8, NW)
    _w7_l2pf(Beta, H, 8, NW)
    if POST:
        _w7_l2pf(PostW, H, 16, NW)
    _w7_l2pf(Stream + row * H, H, 8, NW)   # w7 pf: stream row -> L2 before the wait (a hint: no register gets data)
    _gdc_wait()
    _gdc_launch()
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.float32)
    if POST:
        w = _gl.load(PostW + c).to(_gl.float32)
        b = (b * w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    s = _gl.load(Stream + row * H + c)
    ns = a * s + beta * b
    _gl.store(NewStream + row * H + c, ns)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    w = _gl.load(PreW + c).to(_gl.float32)
    h = ns * r * w
    _gl.store(Out + row * H + c, h)


@_glx.jit
def _graft_row_boundary_pre(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                            H: _gl.constexpr, F: _gl.constexpr, POST: _gl.constexpr, V: _gl.constexpr, NW: _gl.constexpr):
    row = _gl.program_id(0)
    c = _gl.arange(0, H, layout=_gl.BlockedLayout([V], [32], [NW], [0]))
    pw = _gl.load(PreW + c).to(_gl.float32)
    a = _gl.load(Alpha + c)
    beta = _gl.load(Beta + c)
    if POST:
        post_w = _gl.load(PostW + c).to(_gl.float32)
    _w7_l2pf(Stream + row * H, H, 8, NW)   # w7 pf: stream row -> L2 before the wait (a hint: no register gets data)
    _gdc_wait()
    _gdc_launch()
    b = _gl.load(Branch + row * F + c, c < F, 0).to(_gl.float32)
    if POST:
        b = (b * post_w).to(_gl.bfloat16).to(_gl.float32)
    r = _gl.rsqrt(_gl.sum(b * b, 0) / H + 1.e-8)
    b = (b * r).to(_gl.bfloat16).to(_gl.float32)
    s = _gl.load(Stream + row * H + c)
    ns = a * s + beta * b
    _gl.store(NewStream + row * H + c, ns)
    r = _gl.rsqrt(_gl.sum(ns * ns, 0) / H + 1.e-5)
    h = ns * r * pw
    _gl.store(Out + row * H + c, h)


@triton.jit
def _graft_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, PACKED: tl.constexpr, LSTR: tl.constexpr = 256):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if EARLY:
        _gdc_launch()
    _gdc_wait()
    if not EARLY:
        _gdc_launch()
    if kind == 0:
        c = tl.arange(0, 4096)
        if SPLIT:
            pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
        else:
            pooled = tl.load(ALL + t * 4608 + c)
        gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
        x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if SPLIT:
            if LSTR == 512:   # dn pr: [T, 512] FP32 hi | lo router logits (a separate cuBLASLt router GEMM)
                logit = tl.load(LOGITS + t * 512 + e) + tl.load(LOGITS + t * 512 + 256 + e)
            else:
                logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        if PACKED != 0:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if PACKED == 1:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


_GP, _GI, _GSZ = _C.c_void_p, _C.c_int, _C.c_size_t


class _GraftAlgo(_C.Structure):
    _fields_ = [("data", _C.c_uint64 * 8)]


class _GraftHeur(_C.Structure):
    _fields_ = [("algo", _GraftAlgo), ("workspaceSize", _C.c_size_t), ("state", _C.c_int),
                ("wavesCount", _C.c_float), ("reserved", _C.c_int * 4)]


class _GraftLt:
    def __init__(self, transpose_a):
        self.lib = _C.CDLL("libcublasLt.so.13")
        sigs = {
            "cublasLtCreate": [_C.POINTER(_GP)],
            "cublasLtMatmulDescCreate": [_C.POINTER(_GP), _GI, _GI],
            "cublasLtMatmulDescSetAttribute": [_GP, _GI, _GP, _GSZ],
            "cublasLtMatrixLayoutSetAttribute": [_GP, _GI, _GP, _GSZ],
            "cublasLtMatrixLayoutCreate": [_C.POINTER(_GP), _GI, _C.c_uint64, _C.c_uint64, _C.c_int64],
            "cublasLtMatmul": [_GP] * 12 + [_C.POINTER(_GraftAlgo), _GP, _GSZ, _GP],
            "cublasLtMatmulAlgoInit": [_GP, _GI, _GI, _GI, _GI, _GI, _GI, _GI, _C.POINTER(_GraftAlgo)],
            "cublasLtMatmulAlgoConfigSetAttribute": [_C.POINTER(_GraftAlgo), _GI, _GP, _GSZ],
            "cublasLtMatmulAlgoCheck": [_GP, _GP, _GP, _GP, _GP, _GP, _C.POINTER(_GraftAlgo), _C.POINTER(_GraftHeur)],
        }
        for name, types_ in sigs.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = types_, _GI
        self.handle = _GP()
        self.call("cublasLtCreate", _C.byref(self.handle))
        self.desc = _GP()
        self.call("cublasLtMatmulDescCreate", _C.byref(self.desc), 68, 0)
        op = _GI(1 if transpose_a else 0)
        self.call("cublasLtMatmulDescSetAttribute", self.desc, 3, _C.byref(op), _C.sizeof(op))
        self.workspace = torch.empty(64 << 20, device="cuda", dtype=torch.uint8)
        self.alpha, self.beta = _C.c_float(1), _C.c_float(0)

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name} status {code}")

    def __deepcopy__(self, memo):
        import copy
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.lib, result.handle, result.desc = self.lib, self.handle, self.desc
        result.alpha, result.beta = self.alpha, self.beta
        result.workspace = copy.deepcopy(self.workspace, memo)
        return result


class _GraftPlan(dict):
    def __deepcopy__(self, memo):
        import copy
        result = type(self)()
        memo[id(self)] = result
        for key, value in self.items():
            result[key] = value if key == "layouts" else copy.deepcopy(value, memo)
        return result



def _graft_layouts(lt, m, n, k, mode, batch, fp32):
    wa = (k, n // batch, k, k * n // batch) if mode == "tn" else (n // batch, k, n, n // batch)
    result = []
    for index, (rows, cols, ld, stride) in enumerate((wa, (k, m, k, 0), (n // batch, m, n, n // batch))):
        desc = _GP()
        lt.call("cublasLtMatrixLayoutCreate", _C.byref(desc), 0 if fp32 and index == 2 else 14, rows, cols, ld)
        count, offset = _GI(batch), _C.c_int64(stride)
        lt.call("cublasLtMatrixLayoutSetAttribute", desc, 5, _C.byref(count), _C.sizeof(count))
        lt.call("cublasLtMatrixLayoutSetAttribute", desc, 6, _C.byref(offset), _C.sizeof(offset))
        result.append(desc)
    return result


_GRAFT_LT_PICKS = {
    ("pr", 256): ('tn', 2, True, [785979015234, 4294967331, 0, 562949953421312, 558779540176897, 60129542158, 292057776128, 0]),
    ("pr", 128): ('tn', 3, True, [575525617730, 4294967331, 0, 1970324836974594, 558779540176897, 60129542158, 292057776128, 0]),   # s302: N-batch 3 (bf16_deep sweep, -0.7..-0.9 us)
    ("o", 256): ('nn', 2, False, [85899345986, 4294967331, 0, 1407374883553286, 558779540176897, 60129542158, 292058693646, 0]),
    ("un", 256): ('nn', 2, False, [85899345986, 4294967331, 0, 1407374883553287, 558779540176897, 60129542158, 292058693646, 0]),
    ("un", 128): ('nn', 2, False, [77309411394, 4294967331, 0, 1407374883553283, 558779540176897, 60129542158, 292058693646, 0]),
    ("qkv", 128): ('tn', 16, False, [395136991298, 4294967331, 0, 1407374883553282, 558779540176897, 60129542158, 292058693646, 0]),
    ("o", 128): ('nn', 8, False, [77309411394, 4294967331, 0, 1407374883553283, 558779540176897, 60129542158, 292058693646, 0]),
}
# lead_kq256 (our head d0e53792 _QKV256_CONFIG): qkv m256, algo 66 built from attributes.
_GRAFT_QKV256 = {"id": 66, "tile": 124, "stages": 35, "splitk": 1, "red": 0, "swz": 0, "custom": 0, "cluster": 2}
_GRAFT_QKV256_ATTRS = (("tile", 1, _C.c_uint32), ("splitk", 2, _C.c_int32), ("red", 3, _C.c_uint32),
                       ("swz", 4, _C.c_uint32), ("custom", 5, _C.c_uint32), ("stages", 6, _C.c_uint32),
                       ("cluster", 8, _C.c_uint16))


def _graft_lt_plan(st):
    lt_tn, lt_nn = _GraftLt(True), _GraftLt(False)
    plans, status = {}, {}
    src = {"pr": st["pr"], "o": st["o"], "un": st["unpool"], "qkv": st["qkv"]}
    for (name, m), (mode, batch, fp32, words) in _GRAFT_LT_PICKS.items():
        if m not in _GRAFT_LT_T:
            continue
        lt = lt_tn if mode == "tn" else lt_nn
        if mode == "tn":
            weight = src[name]
            n, k = weight.shape
        else:
            key = "graft_t_" + name
            if key not in st:
                st[key] = src[name].t().contiguous()
            weight = st[key]
            k, n = weight.shape
        assert weight.is_contiguous() and weight.data_ptr() % 128 == 0, (name, weight.data_ptr() % 128)
        assert lt.workspace.data_ptr() % 256 == 0 and n % batch == 0 and (n // batch * 2) % 16 == 0
        layouts = _graft_layouts(lt, m, n, k, mode, batch, fp32)
        algo = _GraftAlgo((_C.c_uint64 * 8)(*words))
        heur = _GraftHeur()
        code = lt.lib.cublasLtMatmulAlgoCheck(lt.handle, lt.desc, *layouts, layouts[2], _C.byref(algo), _C.byref(heur))
        status[(name, m)] = (code, heur.workspaceSize)
        if code or heur.workspaceSize > lt.workspace.numel():
            continue
        plans[(name, m)] = _GraftPlan(lt=lt, layouts=layouts, algo=algo, weight=weight, n=n, fp32=fp32)
    if 256 in _GRAFT_LT_T:
        algo = _GraftAlgo()
        lt_tn.call("cublasLtMatmulAlgoInit", lt_tn.handle, 68, 0, 14, 14, 14, 14, _GRAFT_QKV256["id"], _C.byref(algo))
        for name, attr, ctype in _GRAFT_QKV256_ATTRS:
            value = ctype(_GRAFT_QKV256[name])
            lt_tn.call("cublasLtMatmulAlgoConfigSetAttribute", _C.byref(algo), attr, _C.byref(value), _C.sizeof(value))
        weight = st["qkv"]
        n, k = weight.shape
        assert weight.is_contiguous() and weight.data_ptr() % 128 == 0
        layouts = _graft_layouts(lt_tn, 256, n, k, "tn", 1, False)
        heur = _GraftHeur()
        code = lt_tn.lib.cublasLtMatmulAlgoCheck(lt_tn.handle, lt_tn.desc, *layouts, layouts[2], _C.byref(algo),
                                                 _C.byref(heur))
        status[("qkv", 256)] = (code, heur.workspaceSize)
        if not code and heur.workspaceSize <= lt_tn.workspace.numel():
            plans[("qkv", 256)] = _GraftPlan(lt=lt_tn, layouts=layouts, algo=algo, weight=weight, n=n, fp32=False)
    st["graft_lt"] = plans
    st["graft_lt_status"] = {f"{k[0]}:{k[1]}": v for k, v in status.items()}


def _graft_lt_run(x, plan):
    x = x.contiguous()
    assert x.data_ptr() % 16 == 0
    out = torch.empty((x.shape[0], plan["n"]), device=x.device,
                      dtype=torch.float32 if plan["fp32"] else torch.bfloat16)
    lt = plan["lt"]
    a, b, c = plan["layouts"]
    lt.call("cublasLtMatmul", lt.handle, lt.desc, _C.byref(lt.alpha), _GP(plan["weight"].data_ptr()), a,
            _GP(x.data_ptr()), b, _C.byref(lt.beta), _GP(out.data_ptr()), c, _GP(out.data_ptr()), c,
            _C.byref(plan["algo"]), _GP(lt.workspace.data_ptr()), lt.workspace.numel(),
            _GP(torch.cuda.current_stream().cuda_stream))
    return out


_vj_dense = dense


def dense(x, w, T, state=None, name=None):
    plans = state.get("graft_lt") if state is not None else None
    plan = plans.get((name, T)) if plans and T > 32 else None
    if plan is not None:
        rq = _rq_gemm(state, name, x, T)
        if rq is not None:
            return rq.view(1, T, rq.shape[1]), 1
        out = _graft_lt_run(x, plan)
        return out.view(1, T, plan["n"]), 1
    return _vj_dense(x, w, T, state, name)


def _graft_al(*ts):
    for t in ts:
        assert t.is_contiguous() and t.data_ptr() % 16 == 0, (t.shape, t.data_ptr() % 16)


def _graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=HIDDEN, router_out=None):
    """Returns True when a grafted kernel handled the boundary, else False (caller runs the base kernel)."""
    g = st["graft"]
    if S != 1:
        return False
    br = P.view(T, -1)
    width = br.shape[1]
    pfx = "2" if post else "1"
    if T <= 16 and br.dtype == torch.bfloat16:
        _graft_al(br, stream, s_out, h_out)
        if not post and width == HIDDEN:
            _graft_glue_module().b1(br, stream, g["pre1"], g["alpha1"], g["beta1"], h_out, s_out,
                                    router_out if router_out is not None else s_out, router_out is not None)
            return True
        if post and width == 1024 and nu == 1024:
            _graft_b2_module().b2s(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            return True
        if post and width == HIDDEN and nu == HIDDEN:
            _graft_glue_module().b2c(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
            return True
        return False
    if 16 < T <= 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                      g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 32,
                                      num_warps=32, launch_pdl=True)
        return True
    if T > 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        (_w7_row_boundary if T <= 192 else _graft_row_boundary)[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                  g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,
                                  num_warps=16 if T > 192 else 32, launch_pdl=True)
        return True
    return False


def _graft_after_attention(data, state, o_in, gate, T):
    """Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if not _graft_boundary(P, S, T, data["stream"], state, False, s1, h, router_out=hr):
        _boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,
                         WRITE_ROUTER=T <= 8, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        if T in (16, 32):
            nw, parts = {16: (8, 1), 32: (8, 1)}[T]
            _sl_tile_route[(T, parts + 1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                data["local_start"], lids, SPLIT=True, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
                num_warps=nw, launch_pdl=True)
        else:
            _graft_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=_GRAFT_ROUTE_EARLY, PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, LSTR=int(logits.shape[-1]), launch_pdl=True)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        if T in (16, 32):
            nw, parts = {16: (8, 1), 32: (8, 1)}[T]
            _sl_tile_route[(T, parts + 1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                data["local_start"], lids, SPLIT=False, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True,
                num_warps=nw, launch_pdl=True)
        else:
            _graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=_GRAFT_ROUTE_EARLY, PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=True)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if not _graft_boundary(P, S, T, s1, state, True, s2, out, nu=nu):
        _boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                         s2, out, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return out, s2, ids


_vj_after_attention = _after_attention


def _after_attention(data, state, o_in, gate, T):
    if "graft" in state and T in _GRAFT_GLUE_T and state.get("compose_graft_enabled", {}).get(T, T in (1, 8, 128, 256)):
        return _graft_after_attention(data, state, o_in, gate, T)
    return _vj_after_attention(data, state, o_in, gate, T)


_vj_kernel_fn = kernel_fn


def kernel_fn(data, state):
    cfg = data["config"]
    T = cfg["tokens"]
    plan = state.get("graft_lt", {}).get(("qkv", T)) if T in (128, 256) else None
    if (plan is not None and state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
            and state.get("mm_pick", {}).get((T, "qkv"), True) is True):
        qkv = _rq_gemm(state, "qkv", data["attn_input"], T)
        if qkv is None:
            qkv = _graft_lt_run(data["attn_input"], plan)
        try:
            o_in = _decode_attention7(data, state, qkv)
        except RuntimeError:
            return _kernel_fn_fa3(data, state)
        return _after_attention(data, state, o_in, None, T)
    return _vj_kernel_fn(data, state)


_vj_prepare = prepare


def prepare(weights):
    st = _vj_prepare(weights)   # the base tuner runs first, on the base kernels (no "graft" keys yet)
    if _GRAFT_GLUE_T:
        if st.get("bfp") is not None:
            for T in _GRAFT_GLUE_T:
                if T <= 16:
                    st["mm_pick"][(T, "o")] = "bfp"
        g = {}
        for i, name in ((1, "attn_residual_gate"), (2, "ffn_residual_gate")):
            gw = weights[name].float() / 0.3
            beta = torch.sigmoid(gw)
            g["alpha%d" % i] = torch.sqrt(torch.clamp(torch.sigmoid(-gw) * (1. + beta), min=1.e-3)).contiguous()
            g["beta%d" % i] = beta.contiguous()
        g["pre1"] = (weights["pre_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        g["post2"] = (weights["post_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        g["pre2"] = (weights["next_pre_attn_norm"].float() + 1).to(torch.bfloat16).contiguous()
        for v in g.values():
            assert v.data_ptr() % 128 == 0, v.data_ptr() % 128
        st["graft"] = g
        _graft_glue_module()
        _graft_b2_module()
    if _GRAFT_LT_T:
        _graft_lt_plan(st)
    torch.cuda.synchronize()
    return st



_MA_ALT_SRCS["moe_helper0"] = '\n#include <cuda_bf16.h>\n#include <type_traits>\n#include <cuda_runtime.h>\n#include <stdint.h>\n#define OKC_DEBUG 0\n#define OKC_USE_TMA 1\n#define OKC_RLOAD 1\n#define OKC_F1EARLY 1  // tiny builds: FAST1 warp issues its topk loads before lane 0\'s barrier inits\n#define OKC_WGSFINSS 1  // WGS: the final pass forms each route\'s sum of squares (the phase-B epilogue only writes y)\n#ifndef OKC_FINCOL\n#define OKC_FINCOL 1\n#endif\n#define OKC_WGMIN 9\n#define OKC_XPRE 1\n#define FAST1_ARCH 1\n#ifndef OKC_FAST1\n#define OKC_FAST1 3\n#endif\n#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)\n#ifndef OKC_CPFF\n#define OKC_CPFF 1\n#endif\n#ifndef OKC_FCOSTBB\n#define OKC_FCOSTBB OKC_FCOSTB\n#endif\n#define OKC_FCOSTSB OKC_FCOST\n#define OKC_DYN 1\n#ifndef OKC_DYNK\n#define OKC_DYNK 4\n#endif\n#define OKC_TDYN 0\n#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)\n#define OKC_FT_T 64\n#ifndef OKC_SPLIT\n#define OKC_SPLIT 0  // skinny-only build with experts of > SK_T tokens split into <= SK_T-token jobs (MID, Oct 26)\n#endif\n#ifndef OKC_TINY\n#define OKC_TINY 0\n#endif\n#ifndef OKC_NW\n#define OKC_NW 8\n#endif\n#ifndef OKC_EARLYN\n#define OKC_EARLYN 1\n#endif\n#define OKC_BIGM 64\n#define OKC_XPF 0\n#ifndef OKC_CPF\n#define OKC_CPF 1\n#endif\n#define OKC_FATSTATIC 1\n#define OKC_FCOST 40\n#ifndef OKC_FCOSTB\n#define OKC_FCOSTB 40\n#endif\n#ifndef OKC_WS\n#define OKC_WS 0  // tiny build: one extra helper warp issues the ring\'s TMA copies and runs every skinny epilogue\n#endif\n#define OKC_WGSX 0\n#ifndef OKC_WGS\n#define OKC_WGS 0\n#endif\n#define SILU_DIV(a, b) __fdividef((a), (b))\n#ifndef OKC_UWARP\n#define OKC_UWARP 1\n#endif\n#if OKC_UWARP\n#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)\n#else\n#define WARP_IDX(t) ((t) >> 5)\n#endif\n#ifndef OKC_PRMTSC\n#define OKC_PRMTSC 0\n#endif\n#ifndef OKC_FINPF\n#define OKC_FINPF 0   // skinny builds: final item\'s topk_w / pos before the barrier, ss + ybuf loads batched (Oct 26, bf16)\n#endif\n#define OKC_WSNR 4  // partial-sum buffers between the compute warps and the helper warp\n#define OKC_NOLOAD 0\n\nnamespace okc {\n\n#ifndef OKC_L2H\n#define OKC_L2H 1\n#endif\n#if OKC_L2H\n// L2 policies (CUTLASS CacheHintSm90): weights stream through once (evict first); inter and ybuf are re-read later in\n// the same call (evict last). Hints only: the math and the memory traffic\'s results are unchanged.\n#define L2H_SUFFIX ".L2::cache_hint"\n#define L2H_OPERAND ", %4"\n#define L2H_ARG , "l"(0x12F0000000000000ull)\n__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {\n  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");\n}\n__device__ __forceinline__ void st_keep_f32(float* a, float v) {\n  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");\n}\n#else\n#define L2H_SUFFIX ""\n#define L2H_OPERAND ""\n#define L2H_ARG\n__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }\n__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }\n#endif\n\nconstexpr int HD = 4096, ID = 4096;\nconstexpr int NW = OKC_NW, NT = NW * 32;\nconstexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)\n#define TIDX ((int)((threadIdx.x + NT) % NTL))\nconstexpr int NRB = OKC_WS ? OKC_WSNR : 2;\n#if OKC_WS || OKC_WGS\n#define CTA_SYNC() asm volatile("bar.sync 1, %0;\\n" ::"n"(NT) : "memory")\n#define WS_JOIN() asm volatile("bar.sync 2, %0;\\n" ::"n"(NTL) : "memory")\n#else\n#define CTA_SYNC() __syncthreads()\n#endif\nconstexpr int NCH = 64 / NW;     // 64-K chunks per warp in a skinny unit (4096 / NW K per warp)\n#define OKC_SKT 8\nconstexpr int SK_T = OKC_SKT;   // max tokens of a skinny job (8: 1 n-tile; 16: 2 n-tiles via smem x)\nconstexpr int RED_W = OKC_SKT > 8 ? 8 : 4;\nstatic_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");\nconstexpr int FT_T = OKC_FT_T;   // max tokens of a fat job (longer jobs are split into chunks)\nconstexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;   // 16-row blocks\nconstexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;\nconstexpr int RB_A_LOG = 9, RB_B_LOG = 8;\nstatic_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");\nconstexpr int BFLAG = 1 << 30;  // uslot: a phase-B unit prefetched at the end of phase A\n// dynamic tail: the last pool units of a phase\'s skinny work are claimed at run time (OKC_DYNK per CTA)\n__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }\nstatic_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");\nstatic_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");   // 256-row blocks (fat units)\nconstexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;\nconstexpr int XP_SK = HD + 8;   // skinny x row pitch (bf16)\nconstexpr int KS = 256;         // fat: K per x stage (4 chunks)\nconstexpr int XP_FT = KS + 8;   // fat x row pitch (bf16)\nconstexpr int NST = 3;          // fat x stages\nconstexpr float TWO62 = 4.611686018427388e18f;\n\nstruct Params {\n  const uint4* wA; const uint4* sA; const float* gsA;\n  const uint4* wB; const uint4* sB; const float* gsB;\n  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here\n  const unsigned char* wAf; const unsigned char* wBf;  // fat-order weight copies (36 KB per k-stage)\n  const unsigned char* wAu; const unsigned char* wBu;  // unit-contiguous copies (36 KB per 16-row block)\n  const unsigned char* wAn; const unsigned char* wBn;\n  const unsigned char* wAs; const unsigned char* wBs;  // WGS copies: [256-row unit][stage][warpgroup][9 KB]\n  __nv_bfloat16* xA; __nv_bfloat16* xB;  // WGS: per job, per stage x / inter in wgmma B layout [stage][n8][k8][8][8]\n  int* xready;                           // WGS: per job, rows of xA written\n  int* doneAs;\n  uint32_t zero;  // always 0 (opaque to the compiler)\n  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;\n  int M, lo, nloc;\n  int* ctr;\n  int* doneA;      // [JMAX]\n  __nv_bfloat16* inter;  // [EMAX, ID]\n  float* ybuf;     // [EMAX, HD]\n  float* ss;\n  float* ss1;      // [EMAX]\n  __nv_bfloat16* out;\n  unsigned long long* dbg;\n};\n\nstruct Smem {\n  uint32_t lut[256];\n#if !OKC_WGS\n  float red[NRB][NW][32][RED_W];\n#else\n  float red[1][1][1][RED_W];      // (the skinny paths that use it are not run in WGS builds)\n#endif\n#if OKC_WS\n  uint64_t wse[8], wsrf[8], wsre[8];\n  int wsr[4];                          // WS: this CTA\'s phase A and phase B unit ranges\n#endif\n#if OKC_WGS\n  uint64_t bars[24];  // WGS: 0..3 weights full, 4..7 weights empty, 8..15 x full, 16..23 x empty\n  int wglast[4];      // WGS: this warpgroup completed a split unit\n  int wcp[JMAX + 1];\n  int wcpB[JMAX + 1];\n  int jrow[JMAX + 1];\n  uint32_t wtot;\n  uint32_t wtotB;\n#else\n  uint64_t bars[12];  // 0..4 skinny ring (NS <= 5), 5..11 fat ring (NSF <= 7)\n#endif\n#if OKC_WGS\n  int wgr[4];         // WGS: this CTA\'s phase A and phase B unit ranges\n#endif\n  int seq, fseq, preissued, preissuedB;\n  int fcnt[32];     // OKC_SPLIT: FAST1 per-expert entry counts (the preissue\'s job -> expert map)\n  void* xbuf_ptr;\n  int cnt[64], off[64];\n  int wc[NW][32];\n  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];\n  int njF, njS, unit;\n  int gen;          // the grid barrier\'s generation at launch start (+1 per launch): picks this launch\'s ss copy\n  float* ssp;       // this launch\'s ss copy\n  int uslot[8];    // unit id per skinny ring slot (-1: none; BFLAG | v: phase-B unit prefetched in phase A)\n  int dync[JMAX];  // phase A: pool units this CTA processed, per job (credited to doneA at the end of phase A)\n  int unit_next, unit_next2;\n  signed char eid[EMAX];\n  short ent[EMAX];\n  short pos[EMAX];\n};\n\n// subroutines cost ~1 us): a * c < 2^24, rG ~ 1 / G, then an exact +-1 fix-up\n__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {\n  const int p = a * c;\n  int q = __float2int_rz((float)p * rG);\n  if (q * G > p) --q;\n  else if ((q + 1) * G <= p) ++q;\n  return q;\n}\n__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n// a x (b.lo, b.lo) and a x (b.hi, b.hi): the broadcast folds into the HMUL2 operand (.H0_H0 / .H1_H1)\n__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {\n  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));\n  return *reinterpret_cast<uint32_t*>(&r);\n}\n\n__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,\n                                         uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {\n  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);\n  int sz = valid ? 16 : 0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" ::"r"(s), "l"(gmem), "r"(sz));\n}\n__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\\n" ::); }\ntemplate <int N>\n__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\\n" ::"n"(N)); }\n__device__ __forceinline__ unsigned long long gtimer() {\n  unsigned long long t;\n  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));\n  return t;\n}\n#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\nextern __shared__ __align__(16) unsigned char smem_raw[];\n__device__ __forceinline__ float* ss_cur();  // this launch\'s ss copy (defined after Smem users)\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\\n" : "=r"(v) : "l"(p));\n  return v;\n}\n\n// Dequantize one 32-bit word (8 E2M1 codes) into the 4 bf16x2 A-fragment registers (values e2m1 * 2^-126).\n// Prepacked bit layout (see _pack_codes): r0 = q & M, r1 = rotr(q,6) & M, r2 = rotr(q,10) & M,\n// r3 = (rotl(q,4) & 0x81808180) | (rotr(q,4) & 0x00400040), M = 0x81C081C0.\n__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {\n  constexpr uint32_t M = 0x81C081C0u;\n  r0 = q & M;\n  r1 = __funnelshift_r(q, q, 6) & M;\n  r2 = __funnelshift_r(q, q, 10) & M;\n  // DQ7 bit layout (template DQ7 = True): 7 ALU + 2 IMAD.SHL per 8 weights. Register sources partition the word:\n  // r0 bits as stored, r1 = word << 4, r2 = rotl 10, r3 = word << 6 (6 bits) | rotl 14 (the two e1 bits)\n  r1 = (q * 16u) & M;\n  r2 = __funnelshift_l(q, q, 10) & M;\n  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);\n  return;\n  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);\n}\n\n\n// ---------------------------------------------------------------- routing prologue (every CTA, identical result)\n__device__ void early_issue(const Params& p, Smem& S);\n__device__ void route(const Params& p, Smem& S) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const int E = p.M * 8;\n  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;\n  const int i0 = warp * CH, i1 = min(E, i0 + CH);\n  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;\n#if OKC_RLOAD && OKC_WGS\n  {\n    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;\n    int4 ev[LPT];\n    const int E4 = E >> 2;\n#pragma unroll\n    for (int k = 0; k < LPT; ++k) {\n      const int i4 = tid + k * NT;\n      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);\n    }\n#pragma unroll\n    for (int k = 0; k < LPT; ++k) {\n      const int i4 = tid + k * NT;\n      if (i4 < E4) {\n        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};\n        char4 c;\n        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);\n        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);\n        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);\n        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);\n        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;\n      }\n    }\n  }\n#else\n#pragma unroll 8\n  for (int i = tid; i < E; i += NT) {\n    int e = __ldg(p.topk_ids + i) - p.lo;\n    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);\n  }\n#endif\n  CTA_SYNC();\n  for (int base = i0; base < i1; base += 32) {\n    const int i = base + lane;\n    const int e = (i < i1) ? (int)S.eid[i] : -1;\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);\n  }\n  CTA_SYNC();\n  if (warp == 0) {\n    int c[NW];\n    int tot = 0;\n#pragma unroll\n    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }\n    int incl = tot;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }\n    int off = incl - tot;\n    S.cnt[lane] = tot;\n    S.off[lane] = off;\n    int o2 = off;\n#pragma unroll\n    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }\n    // jobs: fat (cnt > SK_T, chunks of FT_T) first, then skinny (0 < cnt <= SK_T); expert order within each\n#if OKC_SPLIT\n    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");\n    const int nfj = 0;\n    const int nsj = (tot + SK_T - 1) / SK_T;   // ceil(tot / 8) jobs of <= 8 tokens; a split expert re-streams its weights\n#else\n    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;\n    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;\n#endif\n    int fi = nfj, si = nsj;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);\n      if (lane >= d) { fi += v1; si += v2; }\n    }\n    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);\n    int fj = fi - nfj, sj = nfTot + si - nsj;\n    for (int q = 0; q < nfj; ++q) {\n      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);\n    }\n#if OKC_SPLIT\n    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }\n#else\n    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }\n#endif\n    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }\n    __syncwarp();\n#if !OKC_WGS\n    int pos = 0;\n    for (int key = FT_T / 8; key >= 1; --key) {\n      for (int f0 = 0; f0 < nfTot; f0 += 32) {\n        const int f = f0 + lane;\n        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;\n        const unsigned bal = __ballot_sync(0xffffffffu, m);\n        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;\n        pos += __popc(bal);\n      }\n    }\n#endif\n  }\n  CTA_SYNC();\n  early_issue(p, S);\n  for (int base = i0; base < i1; base += 32) {\n    const int i = base + lane;\n    const int e = (i < i1) ? (int)S.eid[i] : -1;\n    const unsigned peers = __match_any_sync(0xffffffffu, e);\n    if (e >= 0) {\n      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));\n      S.ent[q] = (short)i;\n      S.pos[i] = (short)q;\n    } else if (i < i1) {\n      S.pos[i] = -1;\n    }\n    __syncwarp();\n    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);\n    __syncwarp();\n  }\n  CTA_SYNC();\n}\n\n\n\n// ---------------------------------------------------------------- epilogues (one 16-row block, tokens 2t,2t+1 of n-tile)\n// r[0..1] rows g (gate / out row n0), r[2..3] rows g+8 (up / out row n1), tokens tk0 = 2t, 2t+1 relative to tile\n__device__ __forceinline__ float probe_q(float v) {\n  return v;\n}\n__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    int tk = tok0 + h;\n    if (tk < ntok) {\n      float gt = r[h] * gsc, up = r[2 + h] * gsc;\n      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));\n    }\n  }\n}\n// returns per-token partial sum of squares (2 values) for this lane\'s two tokens\n__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,\n                                      float* sq) {\n  const float w0 = 1.f + __bfloat162float(p.normw[n0]), w1 = 1.f + __bfloat162float(p.normw[n0 + 8]);\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    int tk = tok0 + h;\n    float v0 = r[h] * gsc * w0, v1 = r[2 + h] * gsc * w1;\n    sq[h] = v0 * v0 + v1 * v1;\n    if (tk < ntok) {\n      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);\n      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);\n    }\n  }\n}\n__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {\n#pragma unroll\n  for (int h = 0; h < 2; ++h) {\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);\n    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);\n  }\n  if (g == 0) {\n#pragma unroll\n    for (int h = 0; h < 2; ++h)\n      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);\n  }\n}\n\n\n\n// ---------------------------------------------------------------- skinny path: shared-memory weight ring\n// A unit = (job <= 16 tokens, 16-row block) x full K = 32 KB codes + 4 KB scales, contiguous in global memory.\n// Units of the CTA\'s range stream through OKC_NS smem slots (TMA bulk copies on sm_90, cp.async elsewhere).\n// All 8 warps work on the same unit, each on its own 512-K range, with x for that K range held in registers.\nconstexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;\n\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n#if OKC_PRMTSC\n#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)\n#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)\n#endif\n__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(count));\n}\n__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\\n" ::"r"(smem_u32(bar)) : "memory");\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {\n  asm volatile(\n      "{\\n .reg .pred p;\\n WAIT_%=:\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\\n @!p bra WAIT_%=;\\n}\\n"\n      ::"r"(smem_u32(bar)), "r"(parity) : "memory");\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n  const size_t blk = (size_t)S.jE[j] * rbs + rb;\n  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {\n  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);\n}\n\ntemplate <bool PB>\n__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,\n                                            uint4 (&x)[2 * NCH]) {\n  const int g = lane >> 2, t = lane & 3;\n  const int cnt = S.jC[j], st = S.jS[j];\n  {\n    const int r = g;\n    if (r < cnt) {\n      const __nv_bfloat16* row;\n      if (PB) row = p.inter + (size_t)(st + r) * ID;\n      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;\n      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);\n#pragma unroll\n      for (int c = 0; c < NCH; ++c) {\n        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);\n        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);\n      }\n    } else {\n#pragma unroll\n      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);\n    }\n  }\n}\n\nconstexpr int X1P = 512 + 8;  // per-warp smem x rows for the 2nd n-tile (tokens 8..15), pitch in bf16\ntemplate <bool PB>\n__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,\n                                             __nv_bfloat16* x1s) {\n  // warp-private: rows 8..15 of the job, K range [512*warp, +512): 8 rows x 64 uint4\n  const int cnt = S.jC[j], st = S.jS[j];\n  for (int i = lane; i < 8 * 64; i += 32) {\n    const int r = i >> 6, c = i & 63;\n    uint4 v = make_uint4(0, 0, 0, 0);\n    if (8 + r < cnt) {\n      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;\n      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;\n      v = PB ? __ldcg(src) : __ldg(src);\n    }\n    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;\n  }\n  __syncwarp();\n}\n\n#if SKDYN\ntemplate <bool PB>\n__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,\n                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;  // static units [v0, v1); the run-time pool [poolBase, nSall) follows\n  const int rsh = PB ? RB_B_LOG : RB_A_LOG;\n  const int seq0 = seq;\n  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  int preB = 0;\n  const bool cpf = OKC_CPF && !PB && cpf_ok;\n  const int v0B = cB0, nB = cBn;\n  const int pre = PB ? S.preissuedB : S.preissued;\n  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;\n  bool exh = poolBase >= nSall;\n  int* pctr = p.ctr + (PB ? 1 : 0);\n  auto take = [&]() -> int {\n    if (snext < v1) return snext++;\n    if (exh) return -1;\n    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);\n    const int v = poolBase + cl;\n    if (v >= nSall) { exh = true; pend = -2; return -1; }\n    pend = atomicAdd(pctr, 1);\n    if (!PB) {\n      const int jj = njF + (v >> rsh);\n      S.dync[jj] += 1;\n      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);\n    }\n    return v;\n  };\n  auto refill = [&](int sl) {\n    const int v = take();\n    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }\n    else if (cpf && preB < nB) {\n      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);\n      S.uslot[sl] = BFLAG | (v0B + preB);\n      ++preB;\n    } else {\n      S.uslot[sl] = -1;\n    }\n  };\n  if (tid == 0)\n    for (int i = pre; i < OKC_NS; ++i) {\n      const int sl = (seq0 + i) % OKC_NS;\n      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }\n      else refill(sl);\n    }\n  if (nU < OKC_NS) CTA_SYNC();\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  bool two = false;\n  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;\n  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit\'s rows one unit ahead\n  float nwc0 = 0.f, nwc1 = 0.f;\n  const int egg = (tid & 31) >> 2;\n  if (PB && tid < 128) {\n    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);\n    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);\n    nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n  }\n  int nproc = 0;\n  for (int i = 0;; ++i) {\n    int v;\n    if (i < nU) {\n      v = v0 + i;\n    } else {\n      v = S.uslot[(seq0 + i) % OKC_NS];\n      if (v < 0 || (!PB && (v & BFLAG))) break;  // pool exhausted (or phase B\'s prefetched units follow)\n      v &= ~BFLAG;\n    }\n    const int j = njF + (v >> rsh), rb = v & (rbs - 1);\n    float nwn0 = 0.f, nwn1 = 0.f;\n    int vn = -1;\n    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];\n    if (PB && tid < 128 && vn >= 0) {\n      const int rbq = (vn & ~BFLAG) & (rbs - 1);\n      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n    }\n    if (j != curj) {\n      curj = j;\n      two = (OKC_SKT > 8) && S.jC[j] > 8;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);\n    }\n    const int sl = (seq0 + i) % OKC_NS;\n    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);\n    mbar_wait(bars + sl, par);\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n#if OKC_NW == 16\n    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);\n    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);\n#else\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n#endif\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n      if (!two) {\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);\n        }\n      } else {\n        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);\n          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);\n        }\n      }\n    }\n#undef MULG\n#undef MULG8\n    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }\n    float* rd = &S.red[i & 1][warp][lane][0];\n    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();\n    if (tid == 0) {\n      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }\n      else refill(sl);\n    }\n    if (tid < (two ? 128 : 64)) {\n      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;\n      const int gg = ln >> 2, tt = ln & 3;\n      float s0 = 0.f, s1 = 0.f;\n#pragma unroll\n      for (int ww = 0; ww < NW; ++ww) {\n        const float* q = &S.red[i & 1][ww][ln][nt * 4];\n        s0 += q[h];\n        s1 += q[2 + h];\n      }\n      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n      const int tk = nt * 8 + 2 * tt + h;\n      if (!PB) {\n        if (tk < cnt) {\n          const float gsc = p.gsA[ce] * TWO62;\n          const float gt = s0 * gsc, up = s1 * gsc;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      } else {\n        const float gsc = p.gsB[ce] * TWO62;\n        const int n0 = rb * 16 + gg;\n        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n    nwc0 = nwn0; nwc1 = nwn1;\n    ++nproc;\n  }\n  seq = seq0 + nproc;\n  if (!PB) {\n    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      if (nU > 0) {\n        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);\n        for (int jj = jf; jj <= jl; ++jj) {\n          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);\n          atomicAdd(p.doneA + jj, b - a);\n        }\n      }\n      for (int jj = jdmin; jj <= jdmax; ++jj) {\n        const int k = S.dync[jj];\n        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }\n      }\n    }\n  }\n  if (!PB && tid == 0) S.preissuedB = preB;\n  if (PB && tid == 0) S.preissuedB = 0;\n  CTA_SYNC();\n}\n#else\ntemplate <bool PB>\n__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,\n                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;\n  if (nU <= 0) return;\n  const int seq0 = seq;\n  // order earlier generic-proxy accesses of this smem region (fat path) before async-proxy (TMA) writes\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  int preB = 0;\n  const bool cpf = OKC_CPF && !PB && cpf_ok;\n  const int v0B = cB0, nB = cBn;\n  const int pre = PB ? S.preissuedB : S.preissued;\n  for (int i = pre; i < OKC_NS; ++i) {\n    const int sl = (seq0 + i) % OKC_NS;\n    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);\n    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }\n  }\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  bool two = false;\n  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;\n  // phase B epilogue: prefetch (1 + post_norm_w) of the next unit\'s rows one unit ahead\n  float nwc0 = 0.f, nwc1 = 0.f;\n  const int egg = (tid & 31) >> 2;\n  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }\n  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n    float nwn0 = 0.f, nwn1 = 0.f;\n    if (PB && tid < 128 && i + 1 < nU) {\n      const int rbq = (v + 1) % rbs;\n      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));\n    }\n    if (j != curj) {\n      curj = j;\n      two = (OKC_SKT > 8) && S.jC[j] > 8;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);\n    }\n    const int sl = (seq0 + i) % OKC_NS;\n    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);\n    mbar_wait(bars + sl, par);\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#define MULG(r) bmul2((r), sg)\n#define MULG8(r) bmul2((r), sg8)\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n      if (!two) {\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);\n        }\n      } else {\n        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);\n        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);\n        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};\n#pragma unroll\n        for (int st = 0; st < 4; ++st) {\n          uint32_t r0, r1, r2, r3;\n          dq4(ww[st], r0, r1, r2, r3);\n          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);\n          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);\n          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);\n        }\n      }\n    }\n#undef MULG\n#undef MULG8\n    if (!two) {\n#pragma unroll\n      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    }\n    float* rd = &S.red[i & 1][warp][lane][0];\n    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();\n    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);\n    else if (cpf && i + OKC_NS - nU < nB) {\n      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);\n      ++preB;\n    }\n    if (tid < (two ? 128 : 64)) {\n      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;\n      const int gg = ln >> 2, tt = ln & 3;\n      float s0 = 0.f, s1 = 0.f;\n#pragma unroll\n      for (int ww = 0; ww < NW; ++ww) {\n        const float* q = &S.red[i & 1][ww][ln][nt * 4];\n        s0 += q[h];\n        s1 += q[2 + h];\n      }\n      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n      const int tk = nt * 8 + 2 * tt + h;\n      if (!PB) {\n        if (tk < cnt) {\n          const float gsc = p.gsA[ce] * TWO62;\n          const float gt = s0 * gsc, up = s1 * gsc;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      } else {\n        const float gsc = p.gsB[ce] * TWO62;\n        const int n0 = rb * 16 + gg;\n        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n    nwc0 = nwn0; nwc1 = nwn1;\n  }\n  seq = seq0 + nU;\n  if (!PB) {\n    __threadfence();\n    CTA_SYNC();\n    if (tid == 0) {\n      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;\n      for (int jj = jf; jj <= jl; ++jj) {\n        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);\n        atomicAdd(p.doneA + jj, b - a);\n      }\n    }\n  }\n  if (!PB && tid == 0) S.preissuedB = preB;\n  if (PB && tid == 0) S.preissuedB = 0;\n  CTA_SYNC();\n}\n#endif\n\n// ---------------------------------------------------------------- fat unit: CTA = 256 rows (warp = 2 x 16 rows) x full K, <=64 tokens\n// ---------------------------------------------------------------- fat unit through the smem ring\n// A fat unit is 256 rows x full K for <= 64 tokens. Its weights arrive in 16 k-stages of 36 KB (16 row blocks x\n// 4 chunks + their scales, contiguous in the fat-order copy) through slots 0..2 of the ring region; the x tile\n// for the stage is staged by cp.async into two buffers that live in slots 3..4.\nconstexpr int NSF = OKC_NSF;\ntemplate <bool PB>\n__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {\n  const int nub = PB ? UB_B : UB_A;\n  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;\n  if (TIDX == 0) {\n    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n  }\n}\n\n// ---------------------------------------------------------------- fat units: one continuous stream per CTA\n// Units come from a dynamic queue; the next unit is claimed when the current one starts, so the ring keeps\n// streaming across unit boundaries (the refills of the last k-stages already fetch the next unit\'s weights).\n// ---------------------------------------------------------------- wgmma (sm90a): A from registers, B (x) from smem\n__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {  // K-major, 128 B swizzle, 8-row groups 1 KB apart\n  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);\n}\n__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {\n  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);\n}\n__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory"); }\n__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory"); }\ntemplate <int N>\n__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\\n" ::"n"(N) : "memory"); }\ntemplate <int NTT> struct WG;\ntemplate <> struct WG<1> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<2> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<3> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<4> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<5> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<6> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<7> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\ntemplate <> struct WG<8> {\n  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {\n    asm volatile(\n        "{\\n .reg .pred p;\\n setp.ne.b32 p, 1, 0;\\n"\n        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\\n}\\n"\n        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])\n        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)\n        : "l"(desc));\n  }\n};\n\n// one 36 KB k-stage of a fat unit with wgmma: A = dequantized weights (natural k order) from registers, B = x tile from\n// shared memory in the K-major no-swizzle core-matrix layout [n8][k8][8 rows][16 B] (LBO 128 B along K, SBO 4 KB along N)\ntemplate <int NTT>\n__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,\n                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {\n  const int g = lane >> 2;\n  const uint32_t xb = smem_u32(xs);\n  uint32_t A[2][8][4];\n  uint32_t keep = 0;\n#pragma unroll\n  for (int cc = 0; cc < 4; ++cc) {\n    const int sb = cc & 1;\n    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);\n    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);\n    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);\n    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);\n#pragma unroll\n    for (int st = 0; st < 4; ++st) {\n      const uint32_t K92 = 0x6D806D80u;\n      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);\n      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);\n      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);\n      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);\n#define FMG0(r) bmul2((r), sg0)\n#define FMG80(r) bmul2((r), sg80)\n#define FMG1(r) bmul2((r), sg1)\n#define FMG81(r) bmul2((r), sg81)\n      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;\n      dq4((&w0.x)[st], r0, r1, r2, r3);\n      dq4((&w1.x)[st], q0, q1, q2, q3);\n      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);\n      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);\n      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);\n      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);\n#undef FMG0\n#undef FMG80\n#undef FMG1\n#undef FMG81\n    }\n    wg_fence();\n#pragma unroll\n    for (int st = 0; st < 4; ++st) {\n      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);\n      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);\n      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);\n    }\n    wg_commit();\n    if (cc >= 1) {\n      wg_wait<1>();  // the previous chunk\'s group retired: its register set may be rewritten\n#pragma unroll\n      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];\n    }\n  }\n  wg_wait<0>();\n#pragma unroll\n  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];\n  acc0[0][0] += __uint_as_float(keep & zero);\n}\n\ntemplate <int NTT>\n__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,\n                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {\n  const int g = lane >> 2, t = lane & 3;\n  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);\n#if OKC_NW == 8\n  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);\n#endif\n#pragma unroll\n  for (int cc = 0; cc < 4; ++cc) {\n    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);\n#if OKC_NW == 8\n    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);\n#endif\n#if OKC_PRMTSC\n    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);\n#if OKC_NW == 8\n    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);\n#endif\n#else\n    const int sh = (cc & 1) * 16;\n    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;\n    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;\n    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];\n    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];\n#endif\n    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;\n#pragma unroll\n    for (int sp = 0; sp < 2; ++sp) {\n      uint32_t a0[2][4], a1[2][4];\n      (void)a1;\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int s2 = 2 * sp + h;\n        uint32_t r0, r1, r2, r3;\n        dq4((&w0.x)[s2], r0, r1, r2, r3);\n        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);\n#if OKC_NW == 8\n        dq4((&w1.x)[s2], r0, r1, r2, r3);\n        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);\n#endif\n      }\n#pragma unroll\n      for (int nt = 0; nt < NTT; ++nt) {\n        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);\n        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);\n#if OKC_NW == 8\n        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);\n#endif\n        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);\n#if OKC_NW == 8\n        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);\n#endif\n      }\n    }\n  }\n}\n\ntemplate <bool PB, bool BIG>\n__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const int g = lane >> 2, t = lane & 3;\n  int* ctr = p.ctr + (PB ? 1 : 0);\n  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n  uint64_t* fb = S.bars + 5;\n  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);\n  constexpr int NKS = HD / KS;  // 16\n  int u = blockIdx.x;\n  (void)ctr;\n  if (u >= nF) return;\n  constexpr int UBS = PB ? UB_B : UB_A;\n  (void)ubs;\n  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };\n  auto wait_A = [&](int j) {\n    if (PB) {\n      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }\n      CTA_SYNC();\n    }\n  };\n  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };\n  auto stage_x = [&](int j, int ks, int xsl) {\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int ntl = (cnt + 7) >> 3;\n    const int nrow = ntl * 8;\n    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);\n    const int per = nrow * (KS / 8);\n    if (is_nat(j)) {\n      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);\n      for (int i = tid; i < per; i += NT) {\n        const int r = i >> 5, ch = i & 31;\n        const bool v = r < cnt;\n        const __nv_bfloat16* src;\n        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;\n        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;\n        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);\n      }\n    } else {\n      for (int i = tid; i < per; i += NT) {\n        int r = i / (KS / 8), c = i - r * (KS / 8);\n        bool v = r < cnt;\n        const __nv_bfloat16* src;\n        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;\n        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;\n        cp_async16(xs + r * XP_FT + c * 8, src, v);\n      }\n    }\n    cp_commit();\n  };\n  // BIG kernel: per-thread x staging addresses computed once per unit (a stage only adds its k offset)\n  const __nv_bfloat16* xsrc[8];\n  uint32_t xdst[8];\n  uint32_t xval = 0;\n  int xn = 0;\n  auto prep_x = [&](int j) {\n    const int cnt = S.jC[j], st = S.jS[j];\n    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);\n    xval = 0;\n    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;\n#pragma unroll\n    for (int k = 0; k < 8; ++k) {\n      const int i = tid + k * NT;\n      const int r = i >> 5, c = i & 31;\n      const bool v = i < per && r < cnt;\n      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;\n      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;\n      xdst[k] = r * XP_FT + c * 8;\n      if (v) xval |= 1u << k;\n    }\n  };\n  auto stage_xp = [&](int ks, int xsl) {\n    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);\n#pragma unroll\n    for (int k = 0; k < 8; ++k)\n      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);\n    cp_commit();\n  };\n  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  int f = S.fseq;  // ring position of the current unit\'s k-stage 0\n  int xq = 0;\n  int j, ub;\n  job_of(u, j, ub);\n#pragma unroll\n  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));\n  wait_A(j);\n  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);\n  while (true) {\n    const int un = u + gridDim.x;\n    int jn = 0, ubn = 0;\n    if (un < nF) job_of(un, jn, ubn);\n    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];\n    const int ntile = (cnt + 7) >> 3;\n    const int ntc = ntile;\n    const bool nat = is_nat(j);\n    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;\n    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];\n#pragma unroll\n    for (int i = 0; i < FT_T / 8; ++i)\n#pragma unroll\n      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }\n    for (int ks = 0; ks < NKS; ++ks) {\n      if (OKC_XPRE) {\n        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);\n        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }\n        else cp_commit();\n      } else {\n        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);\n        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }\n        else cp_commit();\n      }\n      cp_wait<1>();\n      const int q = f + ks, sl = q % NSF;\n      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));\n      if (nat) asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");  // cp.async x writes -> wgmma reads\n      CTA_SYNC();\n      const unsigned char* slot = ring + sl * UNIT_BYTES;\n      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);\n      if (nat) {\n        switch (ntc) {\n          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;\n        }\n      } else {\n        switch (ntc) {\n          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;\n          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;\n          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;\n        }\n      }\n      CTA_SYNC();\n      ++xq;\n      const int kn = ks + NSF;\n      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);\n      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));\n    }\n    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;\n    if (!PB) {\n#pragma unroll\n      for (int nt = 0; nt < FT_T / 8; ++nt)\n        if (nt < ntile) {\n          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);\n#if OKC_NW == 8\n          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);\n#endif\n        }\n      __threadfence();\n      CTA_SYNC();\n      if (tid == 0) atomicAdd(p.doneA + j, 1);\n    } else {\n#pragma unroll\n      for (int nt = 0; nt < FT_T / 8; ++nt)\n        if (nt < ntile) {\n          float sq0[2], sq1[2];\n          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);\n#if OKC_NW == 8\n          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);\n          sq0[0] += sq1[0]; sq0[1] += sq1[1];\n#else\n          (void)sq1;\n#endif\n          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);\n        }\n    }\n    f += NKS;\n    if (un >= nF) break;\n    u = un; j = jn; ub = ubn;\n  }\n  cp_wait<0>();\n  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n  if (tid == 0) S.fseq = f;\n  CTA_SYNC();\n}\n\ntemplate <bool PB>\n__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {\n  fat_stream<PB, false>(p, S, xbuf, nF, ubs);\n}\n\n\n\n// issue the phase-A skinny ring prologue as soon as the job table exists (before the positions pass)\n__device__ void early_issue(const Params& p, Smem& S) {\n#if FAST1_ON || OKC_WGS\n  return;\n#endif\n  if (OKC_FATSTATIC && S.njF > 0) {  // fat units run first and own the ring region\n    if (TIDX == 0) S.preissued = 0;\n    return;\n  }\n  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);\n  const float rG = __fdividef(1.f, (float)gridDim.x);\n  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);\n  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);\n  // only the first unit, issued by the last warp so warp 0 (routing) is not held up; the ring prologue in\n  // skinny_ring issues the rest after routing, so every CTA\'s first unit is served ahead of the second ones\n  const int n = min(OKC_EARLYN, v1 - v0);\n  if (TIDX == NT - 32) {\n    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }\n    S.preissued = n > 0 ? n : 0;\n  }\n}\n\n// skinny share [v0, v1) of CTA c when nF fat units go round robin first: every CTA ends at the same modeled cost\n#if OKC_WS\n// ---------------------------------------------------------------- warp-specialized skinny ring (tiny build)\n// The NW compute warps only wait for data, compute, release the slot and hand their partial sums over; the helper\n// the last WSBLK units keep their partial buffers until the compute warps\' own epilogues: WSBLK > WSNR deadlocks the\n// partial ring (tuner, Oct 26: every WSNR=2 / WSBLK=3 config hung)\nstatic_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");\n#define OKC_WSRF 0\n__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);\ntemplate <bool PB>\n__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {\n  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);\n  const uint32_t k16 = 16u + p.zero;\n  (void)k16;\n  const int rbs = PB ? RB_B : RB_A;\n  const int njF = S.njF;\n  const int nU = v1 - v0;\n  const int seq0 = seq;\n  uint4 x0[2 * NCH];\n  int curj = -1;\n  float lgs = 0.f;                             // the last phase-B unit\'s epilogue constants (warp 0)\n  uint32_t lnw0 = 0u, lnw1 = 0u;                // (norm weights as raw bf16 bits)\n  for (int i = 0; i < nU; ++i) {\n    const int v = v0 + i;\n    const int j = njF + v / rbs;\n    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {\n      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);\n      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);\n      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);\n    }\n    if (j != curj) {\n      curj = j;\n      if (PB) {\n        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }\n        __syncwarp();\n      }\n      load_x_regs<PB>(p, S, j, lane, warp, x0);\n    }\n    const int q = seq0 + i;\n    const int sl = q % OKC_NS;\n    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n    const unsigned char* slot = ring + sl * UNIT_BYTES;\n    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);\n    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};\n    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n#pragma unroll\n    for (int c = 0; c < NCH; ++c) {\n      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);\n#if OKC_PRMTSC\n      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);\n      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n#else\n      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);\n      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n#endif\n      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};\n      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,\n                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};\n#pragma unroll\n      for (int st = 0; st < 4; ++st) {\n        uint32_t r0, r1, r2, r3;\n        dq4(ww[st], r0, r1, r2, r3);\n        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n    const int bb = q % NRB;\n    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));  // the helper has consumed this buffer\'s previous unit\n    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);\n    CTA_SYNC();\n    if (tid == 0) {\n      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);\n      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);\n    }\n  }\n  if (PB && warp < OKC_WSBLK && warp < nU) {  // the CTA\'s last units: epilogues here, not behind the helper\'s queue\n    const int ul = nU - 1 - warp;\n    const int q = seq0 + ul, bb = q % NRB;\n    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;\n    const int gg = lane >> 2, tt = lane & 3;\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    const int n0 = rb * 16 + gg;\n    const float gsc = lgs * TWO62;\n    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = 2 * tt + h;\n      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;\n      float sq = v0f * v0f + v1f * v1f;\n      if (tk < cnt) {\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n      }\n      sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n      sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n    }\n  }\n  seq = seq0 + nU;\n}\n\n__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {\n  const int sl = q % OKC_NS;\n  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);\n  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);\n}\n\n__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n  const int lane = TIDX & 31;\n  WS_JOIN();  // routing tables and this CTA\'s ranges are ready\n  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];\n  const int nA = a1 - a0, N = nA + (b1 - b0);\n  const int njF = S.njF;\n  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n  const int gg = lane >> 2, tt = lane & 3;\n  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;\n  for (int q = 0; q < Nh; ++q) {\n    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {  // unit q + NS goes into unit q\'s slot once every compute warp released it\n      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));\n      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);\n    }\n    const bool PB = q >= nA;\n    const int v = PB ? b0 + (q - nA) : a0 + q;\n    const int rbs = PB ? RB_B : RB_A;\n    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;\n    const int bb = q % NRB;\n    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));\n    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int w = 0; w < NW; ++w) {\n      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);\n      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;\n    }\n    __syncwarp();\n    if (lane == 0) mbar_arrive1(S.wsre + bb);\n    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];\n    if (!PB) {\n      const float gsc = p.gsA[ce] * TWO62;\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        if (tk < cnt) {\n          const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;\n          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));\n        }\n      }\n      if (q == nA - 1) {  // this CTA\'s phase A is done: credit its units to the jobs\' counters\n        __threadfence();\n        __syncwarp();\n        if (lane == 0) {\n          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;\n          for (int jj = jf; jj <= jl; ++jj) {\n            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);\n            atomicAdd(p.doneA + jj, b - a);\n          }\n        }\n      }\n    } else {\n      const float gsc = p.gsB[ce] * TWO62;\n      const int n0 = rb * 16 + gg;\n      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);\n#pragma unroll\n      for (int h = 0; h < 2; ++h) {\n        const int tk = 2 * tt + h;\n        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;\n        float sq = v0f * v0f + v1f * v1f;\n        if (tk < cnt) {\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);\n          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);\n        }\n        sq += __shfl_xor_sync(0xffffffffu, sq, 4);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 8);\n        sq += __shfl_xor_sync(0xffffffffu, sq, 16);\n        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);\n      }\n    }\n  }\n  __threadfence();\n  WS_JOIN();\n}\n#endif\n\ntemplate <bool BIG, bool PB>\n__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {\n  const float rG = __fdividef(1.f, (float)G);\n  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;\n  auto fcost = [](long long k) -> long long {\n    // phase-specific cost of one fat unit in skinny-unit equivalents (phase B\'s skinny units are slower, so its ratio is lower)\n    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);\n    return k > 0 ? (long long)fc * k : 0;\n  };\n  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);\n  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;\n  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;\n  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };\n  const long long WG = W(G);\n  if (nF > 0 && WG > 0) {\n    const float sc = __fdividef((float)nS, (float)WG);\n    auto split = [&](int x) -> int {\n      if (x >= G) return nS;\n      const int v = __float2int_rz((float)W(x) * sc);\n      return v < nS ? v : nS;\n    };\n    v0 = split(c); v1 = split(c + 1);\n  } else {\n    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);\n  }\n}\n\n#if OKC_WGS\n// ---------------------------------------------------------------- WGS: RS-wgmma for every job (row split, stream-K)\n// Unit = (job, 256-row block): phase A 32 per job, phase B 16, each 16 stages of 256 K. Stage = 36 KB of weights,\n// for each warpgroup q its 64 rows: 8 KB codes [grp][warp][lane][4 words] (natural mma k order) + 1 KB scales\n// (A = dequantized codes in registers, N = 8 * n-tiles) and runs its own epilogue (rows g / g + 8 are gate / up).\n// Work is split by cost-weighted stage ranges (stream-K): a unit cut between two CTAs adds its partial sums into a\n// global buffer (indexed by the CTA boundary) and the second contributor runs the epilogue.\n#define OKC_WGSNX 3                             // x ring slots\nconstexpr int WGS_NS = 3;                       // weight ring slots (36 KB; freed as soon as the codes are in registers)\nconstexpr int WGS_NX = OKC_WGSNX;               // x ring slots (freed when the stage\'s last wgmma retired)\nconstexpr int WGS_NT = FT_T / 8;                // max n-tiles of a job\nconstexpr int WGS_PIECE = 9216;                 // one warpgroup\'s weights in a stage\nconstexpr int WGS_XS = WGS_NT * 4096;           // an x slot: up to FT_T rows x 256 K\nconstexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;\nstatic_assert(BX_E + WGS_NX <= 24, "WGS barriers");\nconstexpr int WGS_SLOTS = 160;\nconstexpr int WGS_FIXW = WGS_NT * 512;\nconstexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;  // + doneAs\n__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {\n  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;\n}\n__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {\n  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;\n}\n#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207\n#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\nconstexpr unsigned long long wgs_pack8(const int (&c)[8]) {\n  unsigned long long r = 0;\n  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);\n  return r;\n}\nconstexpr int WGS_CA[8] = {OKC_WGSA};\nconstexpr int WGS_CB[8] = {OKC_WGSB};\nconstexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);\nstatic_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");\n__device__ __forceinline__ int wgs_cost(int pb, int cnt) {\n  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);\n}\n#define OKC_WGSREORD 1\n#define OKC_GATHB 8\n#define OKC_WGSPR 1  // WGS: the x producer warp computes the stage ranges while the consumers gather\n#define OKC_WGSDFIN 2  // WGS: stream-K split units finished by a designated share (see wgs_flush): bit0 phase A, bit1 phase B\n#define OKC_WGSFCI 3\n#define OKC_WGSGS 2  // WGS: k16 steps per wgmma commit group (4 serializes the N=16..32 variants at 96 registers: C7512)\n__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {\n  (void)tag;\n  mbar_wait(bar, parity);\n}\ntemplate <int NTT> struct WGSMMA;\ntemplate <> struct WGSMMA<1> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<2> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<3> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<4> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<5> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<6> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<7> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\ntemplate <> struct WGSMMA<8> {\n  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\\n"\n                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {\n    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\\n"\n                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));\n  }\n};\n// two producer warps: the weights warp streams every stage\'s 36 KB through the weight ring (TMA, lane 0), the x warp\n// gathers every stage\'s x slice into the x ring (cp.async) and, in phase B, first waits for the job\'s phase-A units;\n// so the weights run ahead by the whole ring even while x waits\n// Phase A runs a CTA\'s trailing partial unit FIRST, then the rest of its range in order: the CTA that holds a unit\'s\n// first stages otherwise reaches them last, finishes the unit (read back + epilogue) and pays it exposed at the phase\n// end. Returns where the first sub-range starts: [sp, g1) then [g0, sp) (sp == g0: no reordering). Phase B keeps the\n// order (its first stages wait for phase A\'s units in order).\n__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {\n  const int ue = ((g1 - 1) >> 4) << 4;  // first stage of the unit holding the range\'s last stage\n  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;\n}\n__device__ void wgs_ranges(const Params& p, Smem& S);  // below (needs the cost tables)\n__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\n  WS_JOIN();  // routing tables and this CTA\'s stage ranges are ready\n  if ((TIDX & 31) != 0) return;\n  int ws = 0, n = 0;\n  uint32_t wph = 0;\n#pragma unroll 1\n  for (int pb = 0; pb < 2; ++pb) {\n    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);\n    const int ubl = pb ? 4 : 5;\n    const unsigned char* base = pb ? p.wBs : p.wAs;\n#pragma unroll 1\n    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {\n      if (gs == g1) gs = g0;\n      const int u = gs >> 4, st = gs & 15;\n      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);\n      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);\n      uint64_t* bar = S.bars + BW_F + ws;\n      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");\n      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }\n    }\n  }\n}\nconstexpr int WGS_XROWS = EMAX + 8 * JMAX;  // padded rows of all jobs, upper bound\n__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {\n  // the routing tables are ready (the consumers arrive on barrier 7 right after routing): this idle warp computes the\n  asm volatile("bar.sync 7, %0;\\n" ::"n"(NT + 32) : "memory");\n  if ((TIDX & 31) == 0) wgs_ranges(p, S);\n  __syncwarp();\n  WS_JOIN();  // routing tables and this CTA\'s stage ranges are ready\n  if ((TIDX & 31) != 0) return;\n  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;\n  int xs = 0, n = 0;\n  uint32_t xph = 0;\n#pragma unroll 1\n  for (int pb = 0; pb < 2; ++pb) {\n    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);\n    const int ubl = pb ? 4 : 5;\n    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;\n    int curj = -1, npad = 8;\n    const __nv_bfloat16* jb = xsrc;\n#pragma unroll 1\n    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {\n      if (gs == g1) gs = g0;\n      const int u = gs >> 4, st = gs & 15;\n      const int j = u >> ubl;\n      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);\n      if (j != curj) {\n        curj = j;\n        npad = ((S.jC[j] + 7) >> 3) << 3;\n        jb = xsrc + (size_t)S.jrow[j] * HD;\n        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }\n      }\n      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }\n      uint64_t* bar = S.bars + BX_F + xs;\n      const uint32_t bytes = (uint32_t)npad * 512u;\n      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");\n      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"\n                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");\n      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }\n    }\n  }\n}\n// copied into xA\'s per-stage wgmma layout, then counted into xready (padding rows are counted, not written)\n__device__ void wgs_gather(const Params& p, Smem& S) {\n  const int tid = TIDX, nj = S.njF + S.njS;\n  const int R = S.jrow[nj];\n  const float rG = __fdividef(1.f, (float)gridDim.x);\n  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);\n  int j = 0;\n  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");\n  constexpr int GB = OKC_GATHB;\n  for (int qb = q0; qb < q1; qb += GB) {\n    uint4 v[GB];\n    uint4* d[GB];\n#pragma unroll\n    for (int k = 0; k < GB; ++k) {\n      const int q = qb + k;\n      d[k] = nullptr;\n      if (q < q1) {\n        while (S.jrow[j + 1] <= q) ++j;\n        const int r = q - S.jrow[j], cnt = S.jC[j];\n        if (r < cnt) {\n          const int ntt = (cnt + 7) >> 3;\n          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));\n          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);\n        }\n      }\n    }\n#pragma unroll\n    for (int k = 0; k < GB; ++k)\n      if (d[k]) *d[k] = v[k];\n  }\n  CTA_SYNC();\n  if (tid == 0) {  // count this CTA\'s rows per job (release: the CTA\'s stores are ordered before by the barrier)\n    int jj = 0;\n    for (int q = q0; q < q1;) {\n      while (S.jrow[jj + 1] <= q) ++jj;\n      const int e = min(q1, S.jrow[jj + 1]);\n      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");\n      q = e;\n    }\n  }\n}\nstruct WgsPend { int kind, u, slot, nst; };\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {\n  const int lane = TIDX & 31, g = lane >> 2, t = lane & 3;\n  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];\n  if (!PB) {\n    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;\n    const float gsc = p.gsA[e] * TWO62;\n    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = nt * 8 + 2 * t + h;\n      if (tk < cnt) {\n        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;\n        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),\n                     __float2bfloat16(probe_q(v)));\n      }\n    }\n  } else {\n    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;\n    const float gsc = p.gsB[e] * TWO62;\n    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;\n#pragma unroll\n    for (int h = 0; h < 2; ++h) {\n      const int tk = tok0 + h;\n      if (tk < cnt) {\n        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);\n        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);\n      }\n    }\n  }\n}\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {\n  if (pd.kind == 0) return;\n  const int tid = TIDX, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;\n  constexpr int UBL = PB ? 4 : 5;\n  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  if (pd.kind == 1) {\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");\n    pd.kind = 0;\n    return;\n  }\n  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;\n  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;\n  if constexpr (DF) {\n  if (pd.kind == 2) {\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");\n    pd.kind = 0;\n    return;\n  }\n  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  pd.kind = 0;\n  } else {\n  if (wt == 0) {\n    int old;\n    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");\n    S.wglast[wg] = old + pd.nst == 16;\n  }\n  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n  const bool last = S.wglast[wg];\n  pd.kind = 0;\n  if (!last) return;\n  }\n  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;\n  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;\n  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {\n    float4 v[2];\n#pragma unroll\n    for (int k = 0; k < 2; ++k)\n      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));\n#pragma unroll\n    for (int k = 0; k < 2; ++k)\n      if (nt0 + k < ntt) {\n        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);\n        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};\n        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);\n      }\n  }\n  if (wt == 0) *fc = 0;\n  if (!PB) {\n    asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");\n    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");\n  }\n}\n// one segment (stages [s0, s1) of unit u) on NTT n-tiles: accumulate, then finish (whole unit) or hand off (split)\ntemplate <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,\n                                        int& seq, WgsPend& pd) {\n  const int tid = TIDX, lane = tid & 31;\n  const int warp = WARP_IDX(tid);\n  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;\n  constexpr int UBL = PB ? 4 : 5;\n  const uint32_t k16c = 16u + p.zero;  // opaque 16: the scale decode\'s multiply stays an IMAD (FMA pipe)\n  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);\n  float d[WGS_NT][4];\n  // k16 steps per commit group: the whole function shares one register budget (96 at 544 threads) and ptxas\n  // serializes every wgmma if any pipeline does not fit (C7512), so the wide-N variants commit 2 steps at a time\n  constexpr int GS = OKC_WGSGS;\n  uint32_t af[2][GS][4];\n  int prev = -1;\n  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);\n  const unsigned char* pc0 = ring + wg * WGS_PIECE;\n  auto stage = [&](auto first) {\n    constexpr bool FIRST = decltype(first)::value;\n    const int ws = seq % WGS_NS, xs = seq % WGS_NX;\n    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);\n    const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);\n    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));\n    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);\n    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);\n#pragma unroll\n    for (int lg = 0; lg < 4; ++lg) {\n      const uint4 cwg = cwn;\n      const uint2 scg = scn;\n      if (lg < 3) {\n        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);\n        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);\n      } else if (lane == 0) {\n        mbar_arrive1(S.bars + BW_E + ws);  // every code and scale of this stage is in registers: weights slot free\n      }\n#pragma unroll\n      for (int h = 0; h < 4 / GS; ++h) {\n        const int ci = lg * (4 / GS) + h;  // commit group index in the stage\n#pragma unroll\n        for (int jj = 0; jj < GS; ++jj) {\n          const int js = h * GS + jj;\n          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;\n          uint32_t r0, r1, r2, r3;\n          dq4(q, r0, r1, r2, r3);\n          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;\n          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;\n          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);\n          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);\n        }\n        wg_fence();\n#pragma unroll\n        for (int jj = 0; jj < GS; ++jj) {\n          if (FIRST && ci == 0 && jj == 0)\n            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);\n          else\n            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],\n                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));\n        }\n        wg_commit();\n        wg_wait<1>();\n        if (ci == 0) {\n          if (prev >= 0) {  // the previous stage\'s last group just retired: its x slot is free\n            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n            prev = -1;\n          }\n        }\n        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);\n      }\n    }\n    prev = xs;\n    ++seq;\n  };\n  stage(std::true_type{});\n#pragma unroll 1\n  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});\n  wg_wait<0>();\n  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);\n  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }\n  if (slot >= 0) {\n    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;\n#pragma unroll\n    for (int nt = 0; nt < NTT; ++nt)\n      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),\n                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");\n    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};\n    return;\n  }\n  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;\n#pragma unroll\n  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);\n  if (!PB) pd = WgsPend{1, u, -1, 0};\n}\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {\n  // (__forceinline__: outlined by NVVM when it grows, a CALL that makes ptxas serialize every wgmma, C7510)\n  constexpr int UBL = PB ? 4 : 5;\n  WgsPend pd{0, 0, 0, 0};\n#pragma unroll 1\n  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {\n    if (gs >= ge) {\n      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);\n      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;\n    }\n    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));\n    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));\n    const int ntt = (S.jC[u >> UBL] + 7) >> 3;\n    switch (ntt) {\n      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n    }\n    gs += s1 - s0;\n  }\n  wgs_flush<PB>(p, S, pd);\n}\n// cost-weighted stage ranges: per-stage cost by the job\'s n-tiles, a prefix per job and phase (S.wcp, S.wcpB)\n// all 32-bit (tot * c < 2^32: tot <= 136 jobs x 512 stages x 255 x 132): no 64-bit divide subroutine calls, which run\n// from a cold instruction cache at the start of every launch\n__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {\n  uint32_t acc = 0, accB = 0;\n  for (int j = 0; j < nj; ++j) {\n    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;\n    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;\n    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;\n  }\n  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;\n  S.wtot = acc; S.wtotB = accB;\n}\n// stage at cost position x (phase pb), the stage whose cost interval holds x\n__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {\n  const int* pre = pb ? S.wcpB : S.wcp;\n  int lo = 0, hi = nj;\n  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }\n  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);\n  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);\n}\n__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {\n  const uint32_t tot = pb ? S.wtotB : S.wtot;\n  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);\n  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);\n}\n// this CTA\'s stage ranges (stream-K over 256-row units x 16 stages), one thread: the producers stream the same\n__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {\n  const int nj = S.njF + S.njS;\n  wgs_prefix(S, nj);\n  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);\n  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n}\n// the CTA whose range holds stage s (the fixup slot of a split unit is the CTA holding its first stage)\n__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {\n  const uint32_t tot = pb ? S.wtotB : S.wtot;\n  const int j = s >> (9 - pb);\n  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);\n  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);\n  if (c > G - 1) c = G - 1;\n  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;\n  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;\n  return c;\n}\n#endif\n\ntemplate <bool PB, bool BIG>\n__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {\n  const int njF = S.njF, njS = S.njS;\n  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;\n  const int nS = njS * rbs, nF = njF * ubs;\n  // fat units first (round robin over CTAs), then a skinny share sized so every CTA carries the same cost\n#if OKC_TINY\n  if (nF > 0) __trap();  // tiny build (M <= 8): a job never has more than 8 tokens, so there are no fat units\n#else\n  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }\n#endif\n  const int G = gridDim.x, c = blockIdx.x;\n  int v0, v1;\n  const int nSs = nS - dyn_pool(nS, G);  // static skinny units; [nSs, nS) is the run-time pool\n  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);\n  // cross-phase prefill: a CTA without phase-B fat units starts phase B with its skinny share, so phase A\'s last ring\n  // refills already fetch those units (same split function as phase B uses)\n  int cB0 = 0, cBn = 0;\n  bool cpf = false;\n  if (!PB) {\n    const int nFB = njF * UB_B;\n    if (c >= nFB) {\n      int b1;\n      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);\n      cBn = b1 - cB0;\n      cpf = OKC_CPFF || njF == 0;\n    }\n  }\n#if OKC_WS\n  (void)cpf; (void)cBn;\n  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);\n#else\n  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,\n                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);\n#endif\n}\n\n// cap: spills)\n__device__ __forceinline__ float* ss_cur() {\n#if OKC_WGS\n  unsigned long long v;\n  asm volatile("ld.volatile.shared.u64 %0, [%1];\\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));\n  return reinterpret_cast<float*>(v);\n#else\n  return reinterpret_cast<const Smem*>(smem_raw)->ssp;  // the skinny / fat builds have registers to spare: let it hoist\n#endif\n}\ntemplate <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {\n  Smem& S = *reinterpret_cast<Smem*>(smem_raw);\n  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);\n  const int tid = TIDX;\n  DBG(0);\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#if OKC_WS\n  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n#endif\n#if OKC_WGS\n  if (tid >= NT) {\n    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));\n    return;\n  }\n#endif\n#if FAST1_ON\n  // tiny build (M <= 8): every touched expert is one skinny job, in expert order, so this CTA\'s first units follow\n  // from the touched-expert mask alone; the last warp issues them before routing, off the cold-start critical path\n  if (tid >= NT - 32) {\n    const int ln = tid & 31;\n#if OKC_SPLIT\n    S.fcnt[ln] = 0;\n    __syncwarp();\n    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {\n      const int i = i0 + ln;\n      int e = -1;\n      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }\n      const unsigned peers = __match_any_sync(0xffffffffu, e);\n      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));\n    }\n    __syncwarp();\n    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }\n    const int njob = __shfl_sync(0xffffffffu, jincl, 31);\n    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);\n    const float rG1 = __fdividef(1.f, (float)gridDim.x);\n    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);\n    const int n1 = min(OKC_FAST1, v11 - v01);\n    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];\n#pragma unroll\n    for (int i = 0; i < OKC_FAST1; ++i) {\n      const int jj = (v01 + i) >> RB_A_LOG;\n      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;\n    }\n#elif OKC_F1EARLY\n    uint32_t m = 0;\n    const int E8 = p.M * 8;\n    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;\n    if (ln == 0) {\n      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);\n      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }\n      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n    }\n    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }\n    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }\n    for (int i = ln + 64; i < E8; i += 32) {\n      const int e = __ldg(p.topk_ids + i) - p.lo;\n      if (e >= 0 && e < p.nloc) m |= 1u << e;\n    }\n    m = __reduce_or_sync(0xffffffffu, m);\n#endif\n    if (ln == 0) {\n#if OKC_SPLIT || !OKC_F1EARLY\n      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);\n      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }\n      asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n#endif\n#if OKC_SPLIT\n      const int v0 = v01, n = n1;\n#else\n      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);\n      const float rG = __fdividef(1.f, (float)gridDim.x);\n      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);\n      const int n = min(OKC_FAST1, v1 - v0);\n#endif\n      asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n#if OKC_SPLIT\n#pragma unroll\n      for (int i = 0; i < OKC_FAST1; ++i) {\n        if (i >= n) break;\n        const int v = v0 + i;\n        const int e = fex[i];\n#else\n      for (int i = 0; i < n; ++i) {\n        const int v = v0 + i;\n        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);\n#endif\n        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;\n        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;\n        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");\n        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"\n                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");\n      }\n      S.preissued = n > 0 ? n : 0;\n    }\n  }\n#endif\n  // pull every token row of x into L2 now (one 128 B line per thread-iteration); the skinny jobs read them later\n  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)\n    asm volatile("prefetch.global.L2 [%0];\\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));\n  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {\n    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;\n    uint32_t b;\n    if (ex == 15 && mt == 7) {\n      b = 0xFFFFu;\n    } else {\n      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);\n      v *= 0x1p64f;\n      b = __float_as_uint(sgn ? -v : v) >> 16;\n    }\n    S.lut[i] = b | (b << 16);\n  }\n  if (tid == 0) {\n#if !FAST1_ON\n#if OKC_WGS\n    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);\n#else\n    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);\n#endif\n    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");\n    S.preissued = 0;\n#endif\n    S.seq = 0;\n    S.fseq = 0;\n    S.preissuedB = 0;\n    S.xbuf_ptr = xbuf;\n    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);\n    S.gen = gen;\n    S.ssp = (gen & 1) ? p.ss1 : p.ss;\n  }\n#if SKDYN\n  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;\n#endif\n  route(p, S);\n#if OKC_WGS && OKC_WGSPR\n  __threadfence_block();\n  asm volatile("bar.arrive 7, %0;\\n" ::"n"(NT + 32) : "memory");  // the x producer computes the stage ranges\n#endif\n  DBG(1);\n  {  // the previous launch\'s ss copy and barrier counter are the next launch\'s: zero them now (this CTA\'s share)\n    float* other = (S.gen & 1) ? p.ss : p.ss1;\n    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;\n    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n  }\n#if OKC_WS\n  if (tid == 0) {  // this CTA\'s unit ranges for the helper warp (the same exact split run_phase uses: no fat jobs)\n    const int G = gridDim.x, c = blockIdx.x;\n    const float rG = __fdividef(1.f, (float)G);\n    const int nA = S.njS * RB_A, nB = S.njS * RB_B;\n    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);\n    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n  }\n  WS_JOIN();\n#endif\n#if OKC_WGS\n  if (tid < 32) {\n    const int nj = S.njF + S.njS;\n    int carry = 0;\n    for (int j0 = 0; j0 < nj; j0 += 32) {\n      const int j = j0 + tid;\n      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;\n      int incl = v;\n#pragma unroll\n      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }\n      if (j < nj) S.jrow[j] = carry + incl - v;\n      carry += __shfl_sync(0xffffffffu, incl, 31);\n    }\n    if (tid == 0) S.jrow[nj] = carry;\n  }\n  CTA_SYNC();\n  // phase A\'s x into xA (this CTA\'s share of the rows), before the producers start: its x reads do not queue behind\n  // every CTA\'s first weight stages\n  wgs_gather(p, S);\n  DBG(7);\n  WS_JOIN();\n  DBG(6);\n  {\n    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);\n    int seq = 0;\n    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n    DBG(2);\n    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);\n    DBG(3);\n  }\n#else\n  run_phase<false, BIG>(p, S, xbuf);\n  DBG(2);\n  run_phase<true, BIG>(p, S, xbuf);\n  DBG(3);\n#endif\n#if OKC_WS\n  WS_JOIN();  // the helper\'s last epilogue stores are visible\n#endif\n  asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\n#if OKC_FINPF && OKC_FINCOL && !OKC_WGS\n  // the CTA\'s first final item: its routing (pos) and route weights are inputs, so they load before the barrier\n  constexpr int FSEGP = HD / (NT * 4);\n  const int itp = blockIdx.x;\n  const bool hasp = itp < p.M * FSEGP;\n  const int tkp = hasp ? itp / FSEGP : 0;\n  int qvp[8];\n  float wvp[8];\n#pragma unroll\n  for (int s = 0; s < 8; ++s) {\n    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;\n    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;\n  }\n#endif\n  CTA_SYNC();\n  if (tid == 0) {\n    int* bc = p.ctr + 4 + (S.gen & 1);\n    __threadfence();\n    const int old = atomicAdd(bc, 1);\n    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);\n    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);\n  }\n  CTA_SYNC();\n  DBG(4);\n  {\n    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;\n    if (i0 < 2) p.ctr[i0] = 0;\n    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;\n#if OKC_WGS\n    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;\n    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;\n#endif\n  }\n  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n#if OKC_WGS && OKC_WGSFINSS\n  {\n    static_assert(HD == NT * 8, "8 columns per thread");\n    float* red = reinterpret_cast<float*>(xbuf);\n    const int lane = tid & 31, warp = WARP_IDX(tid);\n    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {\n      int qv[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];\n      float ps[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) {\n        ps[sl] = 0.f;\n        if (qv[sl] >= 0) {\n          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;\n        }\n      }\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl)\n        if (qv[sl] >= 0) {\n#pragma unroll\n          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);\n        }\n      if (lane == 0) {\n#pragma unroll\n        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];\n      }\n      CTA_SYNC();\n      if (tid < 8) {\n        float tsum = 0.f;\n        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];\n        red[NW * 8 + tid] = tsum;\n      }\n      CTA_SYNC();\n      float acc[8];\n#pragma unroll\n      for (int i = 0; i < 8; ++i) acc[i] = 0.f;\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl)\n        if (qv[sl] >= 0) {\n          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);\n          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;\n          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;\n        }\n      uint4 o;\n      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);\n#pragma unroll\n      for (int i = 0; i < 4; ++i) {\n        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);\n        ow[i] = *reinterpret_cast<uint32_t*>(&v);\n      }\n      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;\n      CTA_SYNC();\n    }\n  }\n#elif OKC_FINCOL\n  constexpr int FSEG = HD / (NT * 4);\n  int it0 = blockIdx.x;\n#if OKC_FINPF\n  if (hasp) {   // first item: weights / positions from before the barrier; all 16 loads issued before the math\n    const int col = (itp % FSEG) * (NT * 4) + tid * 4;\n    float ssv[8];\n    float4 yv[8];\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;\n      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);\n    }\n    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      if (qvp[s] >= 0) {\n        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);\n        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;\n      }\n    }\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);\n    uint2 o;\n    o.x = *reinterpret_cast<uint32_t*>(&o0);\n    o.y = *reinterpret_cast<uint32_t*>(&o1);\n    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;\n  }\n  it0 += gridDim.x;\n#endif\n  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {\n    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;\n    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      const int q = S.pos[tk * 8 + s];\n      if (q >= 0) {\n        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);\n        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));\n        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;\n      }\n    }\n    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);\n    uint2 o;\n    o.x = *reinterpret_cast<uint32_t*>(&o0);\n    o.y = *reinterpret_cast<uint32_t*>(&o1);\n    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;\n  }\n#else\n  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {\n    float acc[16];\n#pragma unroll\n    for (int i = 0; i < 16; ++i) acc[i] = 0.f;\n#pragma unroll\n    for (int s = 0; s < 8; ++s) {\n      const int q = S.pos[tk * 8 + s];\n      if (q >= 0) {\n        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);\n        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;\n#pragma unroll\n        for (int i = 0; i < 4; ++i) {\n          float4 v = __ldcg(src + i);\n          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;\n        }\n      }\n    }\n    uint4 o[2];\n    uint32_t* ow = reinterpret_cast<uint32_t*>(o);\n#pragma unroll\n    for (int i = 0; i < 8; ++i) {\n      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);\n      ow[i] = *reinterpret_cast<uint32_t*>(&v);\n    }\n    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);\n    dst[0] = o[0]; dst[1] = o[1];\n  }\n#endif\n  DBG(5);\n}\n\n}\n\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n\nstatic size_t okc_smem_bytes() {\n  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;\n#if OKC_WGS\n  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;\n#endif\n  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);\n  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;\n  return base + (xa > xb ? xa : xb);\n}\n\nstatic int g_okc_pdl = 0;  // launch with programmatic stream serialization (PDL); off unless okc_set_pdl(1)\n\nvoid okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }\n\nvoid okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {\n  okc::Params p;\n  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();\n  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();\n  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();\n  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();\n  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();\n  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }\n  else { p.wAn = p.wAf; p.wBn = p.wBf; }\n#if OKC_WGS\n  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");\n  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,\n              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");\n  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();\n#else\n  p.wAs = nullptr; p.wBs = nullptr;\n#endif\n  p.zero = 0;\n  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();\n  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;\n#if OKC_WGS && OKC_RLOAD\n  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");\n#endif\n  int* ib = (int*)scratch[0].data_ptr();\n  p.ctr = ib; p.doneA = ib + 8;\n  p.ss = (float*)scratch[1].data_ptr();\n  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");\n  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);\n  p.ybuf = p.ss + okc::EMAX;\n  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();\n#if OKC_WGS\n  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,\n              "OKC_WGS needs xA / xB after inter (scratch[2])");\n  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;\n  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;\n  p.xready = ib + 8 + 512;\n  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;\n#else\n  p.xA = p.xB = nullptr;\n  p.xready = nullptr;\n  p.doneAs = nullptr;\n#endif\n  p.out = (__nv_bfloat16*)out.data_ptr();\n  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;\n  static bool attr = false;\n  size_t sm = okc_smem_bytes();\n  if (!attr) {\n    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);\n    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);\n    (void)e0; (void)e1;\n    attr = true;\n  }\n  cudaStream_t st = at::cuda::getCurrentCUDAStream();\n  // large batches run the variant with the fat path inlined (and wgmma); small ones keep it out of line, which\n  // leaves the skinny loop\'s code and registers as tuned\n  if (g_okc_pdl) {\n    cudaLaunchConfig_t cfg = {};\n    cfg.gridDim = dim3((unsigned)grid);\n    cfg.blockDim = dim3(okc::NTL);\n    cfg.dynamicSmemBytes = sm;\n    cfg.stream = st;\n    cudaLaunchAttribute at[1];\n    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n    at[0].val.programmaticStreamSerializationAllowed = 1;\n    cfg.attrs = at;\n    cfg.numAttrs = 1;\n    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);\n    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);\n    return;\n  }\n  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);\n  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);\n}\n\n\n'
MOE_MID_DEFINES += ("@moe_helper0",)

def _df_patch(c):
    def s1(s, old, new):
        assert s.count(old) == 1, (old[:80], s.count(old))
        return s.replace(old, new)
    c = s1(c, "#define OKC_NOLOAD 0\n", """#define OKC_NOLOAD 0
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#define OKC_DF OKC_PRMTSC   // data-flag A->B hand-off (needs the PRMT scale decode builds)
#define INTER_SENT_ROWS 128
""")
    c = s1(c, "constexpr int X1P = 512 + 8;", """// DF: phase-B x rows (the job's inter) straight from L2 until every value is written (sentinel 0xFFFF = bf16 NaN)
__device__ __forceinline__ uint32_t df_bad(const uint4& v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t b = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) b |= (uint32_t)(((w[i] & 0xFFFFu) == 0xFFFFu) | ((w[i] >> 16) == 0xFFFFu));
  return b;
}
__device__ __forceinline__ void load_x_inter_df(const Params& p, const Smem& S, int j, int lane, int warp, uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  const bool act = g < cnt;
  const uint4* src = reinterpret_cast<const uint4*>(p.inter + (size_t)(st + (act ? g : 0)) * ID + warp * (64 * NCH) + 16 * t);
  for (int spin = 0;; ++spin) {
    uint32_t bad = 0;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      x[2 * c] = act ? __ldcg(src + c * 8) : make_uint4(0, 0, 0, 0);
      x[2 * c + 1] = act ? __ldcg(src + c * 8 + 1) : make_uint4(0, 0, 0, 0);
      bad |= df_bad(x[2 * c]) | df_bad(x[2 * c + 1]);
    }
    if (!__any_sync(0xffffffffu, bad != 0) || spin > 200000) break;
    __nanosleep(32);
  }
}

constexpr int X1P = 512 + 8;""")
    a = c.index("__device__ void skinny_ring_ws(const Params& p")
    b = c.index("__device__ void ws_helper(")
    seg = c[a:b]
    seg = s1(seg, """      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);""", """      if (PB && OKC_DF) {
        load_x_inter_df(p, S, j, lane, warp, x0);
      } else {
        if (PB) {
          if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
          __syncwarp();
        }
        load_x_regs<PB>(p, S, j, lane, warp, x0);
      }""")
    c = c[:a] + seg + c[b:]
    c = s1(c, "if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters",
           "if (!OKC_DF && q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters")
    c = s1(c, "  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n", """#if !OKC_WGS
  if (OKC_DF) {  // every CTA is past phase B: inter is dead until the next launch; give it back its "not yet written" sentinel
    const int rows = min(INTER_SENT_ROWS, p.M * 8);
    uint4* ip = reinterpret_cast<uint4*>(p.inter);
    for (int i = blockIdx.x * NT + tid; i < rows * (ID / 8); i += gridDim.x * NT) ip[i] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
  }
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
""")
    c = s1(c, "  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();\n", """  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_DF
  p.inter += (size_t)128 * 4096;   // DF sentinel rows [128, 256) of the scratch: only DF launches (MID) use them
#endif
""")
    return c


_MA_ALT_SRCS["moe_helper0df"] = _df_patch(_MA_ALT_SRCS["moe_helper0"])
MOE_MID_DEFINES = tuple("@moe_helper0df" if d == "@moe_helper0" else d for d in MOE_MID_DEFINES)


G121_ROUTE = True       # g121: race the grafted route on the base boundary chain (T > 32)
G121_MARGIN = 0.4       # us the grafted route must win by (same threshold as chain_policy)
G121_PDL_MARGIN = 0.3   # us the PDL launch must beat the plain launch by (T=256 glue PDL lost ~1% before)
PIN_UN_CHAIN_US = 0.0  # p001: T = 1 unpooler cgemv in-chain margin (cgemv won 9/9 raw races)


def _compose_times(fns, reps=16):
    graphs, outputs = [], []
    for fn in fns:
        fn()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out=fn()
        graphs.append(g);outputs.append(out)
    flush=torch.ones((65536,1024),device='cuda',dtype=torch.float32)
    sink=torch.empty(1024,device='cuda',dtype=torch.float32)
    events=[]
    for r in range(reps):
        order=range(len(graphs)) if r%2==0 else reversed(range(len(graphs)))
        for i in order:
            torch.sum(flush,dim=0,out=sink)
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();graphs[i].replay();b.record()
            events.append((i,a,b))
    torch.cuda.synchronize()
    samples=[[] for _ in graphs]
    for i,a,b in events:samples[i].append(a.elapsed_time(b)*1000)
    return [sorted(s)[len(s)//2] for s in samples]



_stack_unpinned_prepare = prepare

def prepare(weights):
    st = _stack_unpinned_prepare(weights)
    bfp = st.get("bfp")
    if bfp is not None:
        for T in (1, 8, 16, 32):
            for name, layer in _BFP_LAYER.items():
                bfp["choice"][(layer, T)] = bfp__static_choice(layer, T, bfp)
            for name in ("qkv", "o", "un"):
                st["mm_pick"][(T, name)] = "bfp"
            can_fuse = (bfp["choice"][("pooler", T)][0] == "sk2"
                        and bfp["choice"][("router", T)][0] == "sk2")
            st["mm_pick"][(T, "pr")] = "bfpf" if T <= 32 and can_fuse else "bfp"
    for T in (128, 256):
        for name in ("qkv", "o", "pr", "un"):
            st["mm_pick"][(T, name)] = True
    return st

_policy_parent_prepare=prepare


def _policy_variant(st,T,mode):
    ss=dict(st)
    ss['mm_pick']=dict(st['mm_pick'])
    ss['graft_lt']=dict(st.get('graft_lt',{}))
    ss['compose_graft_enabled']=dict(st.get('compose_graft_enabled',{}))
    ss['compose_boundary_warps']=dict(st.get('compose_boundary_warps',{}))
    graft,bw,drops=mode
    ss['compose_graft_enabled'][T]=graft
    ss['compose_boundary_warps'][T]=bw
    for name in drops:ss['graft_lt'].pop((name,T),None)
    return ss


def prepare(weights):
    st=_policy_parent_prepare(weights)
    st['compose_base_mm']=dict(st['mm_pick'])
    st['compose_base_graft_lt']=dict(st['graft_lt'])
    st['compose_active']=set()
    st['compose_selection']=[]
    st['compose_graft_enabled']={T:T in (1,8,128,256) for T in (1,8,16,32,128,256)}
    st['compose_boundary_warps']={}
    gen=torch.Generator(device=st['qkv'].device).manual_seed(1618033)
    for T in (1,8,16,32,128,256):
        original=st['compose_graft_enabled'][T]
        modes=[(original,16,()),(True,16,()),(False,16,()),(False,8,()),(False,4,())]
        modes=list(dict.fromkeys(modes))
        data={'stream':torch.randn(T,HIDDEN,device=st['qkv'].device,generator=gen),
              'local_start':torch.zeros(1,device=st['qkv'].device,dtype=torch.int32)}
        o_in=(.1*torch.randn(T,1024,device=st['qkv'].device,generator=gen)).to(torch.bfloat16)
        ids=_after_attention(data,st,o_in,None,T)[2]
        counts=torch.bincount(ids.flatten().long()//32,minlength=8)
        data['local_start'].fill_(32*int(counts.argmax()))
        variants=[_policy_variant(st,T,mode) for mode in modes]
        ts=_compose_times([lambda ss=ss:_after_attention(data,ss,o_in,None,T) for ss in variants],reps=32)
        best=min(range(len(ts)),key=ts.__getitem__)
        if ts[best]>=ts[0]-.4:best=0
        print(f"[pin] T={T} chain_policy "+" ".join(f"{'g' if m[0] else 'b'}{m[1]}:{t:.2f}" for m,t in zip(modes,ts))+f" -> {'g' if modes[best][0] else 'b'}{modes[best][1]}",flush=True)
        picked=variants[best]
        st['compose_graft_enabled'][T]=picked['compose_graft_enabled'][T]
        st['compose_boundary_warps'][T]=picked['compose_boundary_warps'][T]
        st['graft_lt']=picked['graft_lt']
        st['compose_selection'].append({'T':T,'region':'chain_policy','us':ts,'modes':modes,'choice':modes[best]})
        if G121_ROUTE and T > 32 and not st['compose_graft_enabled'][T]:
            rvs=[]
            for rg in (0,1,2):
                ss=dict(st)
                ss['compose_route_graft']=dict(st.get('compose_route_graft',{}))
                ss['compose_route_graft'][T]=rg
                rvs.append(ss)
            rts=_compose_times([lambda ss=ss:_after_attention(data,ss,o_in,None,T) for ss in rvs],reps=32)
            rb=0
            if min(rts[1],rts[2])<rts[0]-G121_MARGIN:
                rb=2 if rts[2]<rts[1]-G121_PDL_MARGIN else 1
            st['compose_route_graft']=dict(st.get('compose_route_graft',{}))
            st['compose_route_graft'][T]=rb
            st['compose_selection'].append({'T':T,'region':'route_split','us':rts,'choice':rb})
            print(f"[pin] T={T} route_split "+" ".join(f"{t:.2f}" for t in rts)+f" -> {rb}",flush=True)
    return st


_SL_GLUE_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
namespace cg = cooperative_groups;
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

struct GlueArgs {
  const BF *branch, *post, *pre;
  const float *part, *stream, *alpha, *beta;
  BF *out;
  float *next, *router;
  int rows;
  const float* alpha2;
  float* tail;
};

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
template<int V> __device__ __forceinline__ void glf(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
      :"=f"(x[0]),"=f"(x[1]),"=f"(x[2]),"=f"(x[3]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];"
      :"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
  }
}
template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
template<int V> __device__ __forceinline__ void gsf(float* p, const float* x) {
  if constexpr (V == 4) {
    asm volatile("st.global.v4.f32 [%0],{%1,%2,%3,%4};"
      ::"l"(p),"f"(x[0]),"f"(x[1]),"f"(x[2]),"f"(x[3]):"memory");
  } else {
    asm volatile("st.global.v2.f32 [%0],{%1,%2};"
      ::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
  }
}
template<int V> __device__ __forceinline__ void gsb(BF* p, const float* x) {
  unsigned u[V/2];
  #pragma unroll
  for(int j=0;j<V/2;j++) {
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u[j]):"f"(x[2*j+1]),"f"(x[2*j]));
  }
  if constexpr(V==4) {
    asm volatile("st.global.v2.b32 [%0],{%1,%2};"::"l"(p),"r"(u[0]),"r"(u[1]):"memory");
  } else {
    asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u[0]):"memory");
  }
}

template<int N> __device__ __forceinline__ float local_squares(const float (&x)[N]) {
  float q[N/2];
  #pragma unroll
  for(int j=0;j<N/2;j++) q[j]=gf(x[2*j],x[2*j],gm(x[2*j+1],x[2*j+1]));
  #pragma unroll
  for(int width=N/2;width>1;width/=2) {
    #pragma unroll
    for(int j=0;j<width/2;j++) q[j]=ga(q[2*j],q[2*j+1]);
  }
  return q[0];
}
__device__ __forceinline__ float butterfly(float x, int first) {
  #pragma unroll
  for(int d=first;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares(const float (&x)[N], float* sm,
                                                cg::cluster_group cluster, int rank) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sum=butterfly(local_squares(x),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* p=cluster.map_shared_rank(sm,dst);
      p[rank*warps_per_cta+warp]=sum;
    }
  }
  cluster.sync();
  float all=sm[lane&(NW-1)];
  return butterfly(all,NW/2);
}

template<int NW,int N>
__device__ __forceinline__ float cluster_squares_pair(const float (&x)[N], const float (&y)[N],
    float* sm, float* smy, cg::cluster_group cluster, int rank, float& tail) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sx=butterfly(local_squares(x),16);
  float sy=butterfly(local_squares(y),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* px=cluster.map_shared_rank(sm,dst);
      volatile float* py=cluster.map_shared_rank(smy,dst);
      px[rank*warps_per_cta+warp]=sx;
      py[rank*warps_per_cta+warp]=sy;
    }
  }
  cluster.sync();
  tail=butterfly(smy[lane&(NW-1)],NW/2);
  return butterfly(sm[lane&(NW-1)],NW/2);
}

template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>
__global__ __launch_bounds__(128) void glue_cluster_boundary(GlueArgs a) {
  constexpr int logical_threads=NW*32;
  constexpr int per_cta=logical_threads/8;
  constexpr int N=8192/logical_threads;
  constexpr int repeats=N/V;
  cg::cluster_group cluster=cg::this_cluster();
  int rank=cluster.block_rank(), tid=threadIdx.x;
  int logical=rank*per_cta+(tid%per_cta);
  int row=blockIdx.x/8;
  bool active=tid<per_cta;
  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];
  __shared__ float partials[3][NW];
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    glf<V>(a.stream+row*8192+col,s+r*V);
    glf<V>(a.alpha+col,alpha+r*V);
    glf<V>(a.beta+col,beta+r*V);
    glf<V>(a.alpha2+col,a2+r*V);
    glb<V>(a.pre+col,pre+r*V);
    if constexpr(POST) glb<V>(a.post+col,post+r*V);
    if constexpr(SLICED) {
      if(col>=1024) {
        #pragma unroll
        for(int j=0;j<V;j++) {
          int k=r*V+j;
          ns[k]=gf(alpha[k],s[k],gm(beta[k],gm(0.f,post[k])));
        }
      }
    }
  }
  cluster.sync();
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if constexpr(SLICED) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=0.f;
      if(col<1024) {
        #pragma unroll
        for(int sk=0;sk<8;sk++) {
          float part[V];glf<V>(a.part+(sk*a.rows+row)*1024+col,part);
          #pragma unroll
          for(int j=0;j<V;j++) b[r*V+j]=ga(b[r*V+j],part[j]);
        }
      }
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(b[r*V+j]);
    } else {
      glb<V>(a.branch+row*8192+col,b+r*V);
    }
    if constexpr(POST) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(gm(b[r*V+j],post[r*V+j]));
    }
  }
  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if(!SLICED || col<1024) {
      #pragma unroll
      for(int j=0;j<V;j++) {
        int k=r*V+j;
        float bn=gb(gm(b[k],r1));
        if(SLICED || (PRELOADED_BASE && col<4096))
          ns[k]=gf(alpha[k],s[k],gm(beta[k],bn));
        else
          ns[k]=gf(beta[k],bn,gm(alpha[k],s[k]));
      }
    }
  }
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    #pragma unroll
    for(int j=0;j<V;j++) {
      int k=r*V+j;
      carry[k]=gm(a2[k],ns[k]);
      masked[k]=(col+j>=1024) ? carry[k] : 0.f;
    }
  }
  float tail;
  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);
  if(rank==0 && tid==0) a.tail[row]=tail;
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    float h[V];
    #pragma unroll
    for(int j=0;j<V;j++) h[j]=gm(gm(ns[r*V+j],r2),pre[r*V+j]);
    if(active) {
      gsb<V>(a.out+row*8192+col,h);
      gsf<V>(a.next+row*8192+col,carry+r*V);
      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
    }
  }
}

template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(a.rows*8);cfg.blockDim=dim3(128);cfg.stream=stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim={8,1,1};
  attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=2;
  auto e=cudaLaunchKernelEx(&cfg,fn,a);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0),(const float*)alpha2.data_ptr(),(float*)tail.data_ptr()};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b1: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(write_router) {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,true,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,true,false>,s);
  } else {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,false,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,false,false>,s);
  }
}
'''

@_lru_cache(None)
def _sl_cluster_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    return _tvm_load_inline(name='sl_early_tail_' + _hashlib.md5(_SL_GLUE_CUDA.encode()).hexdigest()[:12],
                            cuda_sources=_SL_GLUE_CUDA, functions=['b1'],
                            extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math'])

@triton.jit
def _sl_cluster_finish(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT,
                       BC: tl.constexpr, USE_PDL: tl.constexpr = True):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0,1024)
    beta = tl.load(BETA+c)
    gp = tl.load(POST+c).to(tl.float32)
    carry = tl.load(CARRY+t*8192+c)
    tail = tl.load(TAIL+t)
    o = tile*BC+tl.arange(0,BC)
    outcarry = tl.load(CARRY+t*8192+o)
    gn = tl.load(NEXT+o).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    f = (tl.load(P+t*1024+c).to(tl.float32)*gp).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(tl.sum(f*f,0)/8192.0+1e-8)
    n = (f*inv).to(tl.bfloat16).to(tl.float32)
    head = carry+beta*n
    invnext = tl.rsqrt((tail+tl.sum(head*head,0))/8192.0+1e-5)
    result = tl.where(o<1024,tl.gather(head,o%1024,0),outcarry)
    tl.store(SOUT+t*8192+o,result)
    tl.store(HOUT+t*8192+o,(result*invnext*gn).to(tl.bfloat16))

SL_CLUSTER_CONFIG={1: (8, 256), 8: (16, 1024)}
def _sl_cluster_after_attention(data, state, o_in, gate, T):
    """Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if S != 1 or P.dtype != torch.bfloat16:
        return _graft_after_attention(data,state,o_in,gate,T)
    g=state['graft']
    _sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        _sl_graft_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], num_warps=SL_ROUTE_CONFIG[T][0], launch_pdl=True)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        _sl_graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], num_warps=SL_ROUTE_CONFIG[T][0], launch_pdl=True)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if S != 1 or P.dtype != torch.bfloat16:
        raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=True)
    return out, s2, ids


_sl_previous_after_attention=_after_attention

def _after_attention(data,state,o_in,gate,T):
    if T in SL_CLUSTER_CONFIG and 'graft' in state and state.get('compose_graft_enabled', {}).get(T, True):
        return _sl_cluster_after_attention(data,state,o_in,gate,T)
    return _sl_previous_after_attention(data,state,o_in,gate,T)

_sl_previous_prepare=prepare

def prepare(weights):
    st=_sl_previous_prepare(weights)
    _sl_cluster_module()
    return st
@triton.jit
def _sl_graft_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if EARLY:
        _gdc_launch()
    _gdc_wait()
    if not EARLY:
        _gdc_launch()
    if kind == 0:
        c = tl.arange(0, 4096)
        if SPLIT:
            pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
        else:
            pooled = tl.load(ALL + t * 4608 + c)
        gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
        x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
        tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + tl.load(BIAS + e)
        k8 = tl.arange(0, 8)
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - tl.load(LS) + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)

SL_ROUTE_CONFIG={1: (8, 0, 0), 8: (4, 0, 0)}
@triton.jit
def _sl_tile_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr, PARTS: tl.constexpr, PRELOAD: tl.constexpr):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    if kind<PARTS:
        c=tl.arange(0,4096)
        oc=kind*(4096//PARTS)+tl.arange(0,4096//PARTS)
        if PRELOAD:
            gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
        _gdc_launch()
        _gdc_wait()
        if SPLIT:
            pooled=tl.load(POOLED+t*4096+c).to(tl.float32)
        else:
            pooled=tl.load(ALL+t*4608+c)
        inv=tl.rsqrt(tl.sum(pooled*pooled,0)/4096.0+1e-5)
        if not PRELOAD:
            gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
        if PARTS==1:
            vals=pooled
        elif SPLIT:
            vals=tl.load(POOLED+t*4096+oc).to(tl.float32)
        else:
            vals=tl.load(ALL+t*4608+oc)
        tl.store(XOUT+t*4096+oc,(vals*inv*gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        if PRELOAD:
            bias=tl.load(BIAS+e)
            ls=tl.load(LS)
        _gdc_launch()
        _gdc_wait()
        if not PRELOAD:
            bias=tl.load(BIAS+e)
            ls=tl.load(LS)
        if SPLIT:
            logit = tl.load(LOGITS + t * 256 + e)
        else:
            logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias
        k8 = tl.arange(0, 8)
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)

SL_TILE_CONFIG={1:(4,1,True),8:(4,4,True)}
def _sl_tile_after_attention(data, state, o_in, gate, T):
    """Copy of the base _after_attention (one-program-per-token path) with the grafted boundary/route kernels.
    Every kernel the graft does not cover falls back to the base kernel with the base arguments."""
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    tail = torch.empty((T,), **f32)
    s1 = torch.empty((T, HIDDEN), **f32)
    h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    hr = torch.empty((T, HIDDEN), **f32) if T <= 8 else None
    if gate is None:
        P, S = dense(o_in, state["o"], T, state, "o")
    else:
        P, S = dense_ng(o_in, gate, state["o"], T, state)
    if S != 1 or P.dtype != torch.bfloat16:
        return _graft_after_attention(data,state,o_in,gate,T)
    g=state['graft']
    _sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
    P, S = _pool_router_fp32(h, hr, T, state, ppick)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    use_marlin = moe == "marlin"
    use_ma = moe == "agent"
    if not (use_marlin or use_ma):
        raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    if S == -1:
        pooled, logits = P
        _graft_al(pooled, logits, xm, ids, wts, lids)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=True, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
        allp = P.view(T, 4608)
        _graft_al(allp, xm, ids, wts, lids)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                             data["local_start"], lids, SPLIT=False, EARLY=SL_ROUTE_CONFIG[T][2], MODE=SL_ROUTE_CONFIG[T][1], PARTS=SL_TILE_CONFIG[T][1], PRELOAD=SL_TILE_CONFIG[T][2], num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    else:
        _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                     PERMUTED=False, num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    if use_ma:
        m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
        ext = _moe(T, state)
        if MOE_HAS_PDL:
            ext.okc_set_pdl(1 if _pdl(T) else 0)
        for m0 in range(0, T, _MA_MMAX):
            m1 = min(T, m0 + _MA_MMAX)
            ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                        state["nsm"])
    elif use_marlin:
        m = _marlin_experts(xm, wts, lids, state)
    wu = state["unpool_slice"] if T <= 8 else state["unpool"]
    P, S = dense(m, wu, T, state, "un")
    nu = wu.shape[0]
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    if S != 1 or P.dtype != torch.bfloat16:
        raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
    nw,bc=SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=True)
    return out, s2, ids


_sl_tile_prior_after=_after_attention
def _after_attention(data,state,o_in,gate,T):
    if T in SL_TILE_CONFIG and 'graft' in state and state.get('compose_graft_enabled', {}).get(T, True):
        return _sl_tile_after_attention(data,state,o_in,gate,T)
    return _sl_tile_prior_after(data,state,o_in,gate,T)

MOE_TINY2_DEFINES += ("@moe_helper0",)
MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1")


_DE_EXT=None
_DE_RUN=bfp__run_sk2
_DE_PREPARE=prepare

def _de_source():
 cfg=('qkv8n_r144','qkv_gate',2304,24,3,1,4,2,2,1)
 _,nstage=bfp__sk2_geom(cfg,8)
 dispatch=f'launch_sk2<1, 8, 3, 1, 4, 2, {nstage}, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return;'
 source=bfp_SK2_SRC.replace('@@SK2_DISPATCH@@',dispatch).replace('@@SK2F_DISPATCH@@','')
 source=source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)','4').replace('NW + (COMP ? NESC : 0)','NW')
 source=source.replace('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");','')
 a=source.index('      int pe = 1;');b=source.index('\n    }\n    mbar_arrive(cbar);',a)
 source=source[:a]+r"""
      for(int pe=1+ew;pe<=cnt;pe+=NESC) {
        const int2 e=ptab_s[pe];const int kk=e.x&65535;
        if(lane<M && kk>=gs0*KST && kk<(gs0+nst)*KST)
          cr[lane*CP+(e.x>>16)] += __int_as_float(e.y)*__bfloat162float(x[(size_t)lane*K+kk]);
      }
"""+source[b:]
 shared=bfp_CUDA_SRC.replace('@@MBAR@@',bfp_MBAR_HELPERS).replace('@@BODY@@','')
 return shared+bfp_MAP_SRC+source

def _de_ext():
 global _DE_EXT
 if _DE_EXT is None:
  import hashlib
  source=_de_source();tag=hashlib.md5(source.encode()).hexdigest()[:12];source=source.replace('sk2_kernel','sk2_kernel_narrow_'+tag).replace('launch_sk2','launch_sk2_narrow_'+tag);cpp=bfp_CPP_SRC[bfp_CPP_SRC.index('void sk2('):bfp_CPP_SRC.index('void sk2f(')]+'void sk2_set_pdl(int64_t n);'
  _DE_EXT=load_inline('proj_dl_escape_'+hashlib.md5(source.encode()).hexdigest()[:12],'#include <torch/extension.h>\n'+cpp,
    cuda_sources=source,functions=['sk2','sk2_set_pdl'],no_implicit_headers=True,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
 return _DE_EXT

def prepare(weights):
 _de_ext()
 return _DE_PREPARE(weights)

def bfp__run_sk2(x,layer,m,n,k,state,dg=0,dbg=None):
 if layer!='qkv_gate' or not 2<=m<=8:return _DE_RUN(x,layer,m,n,k,state,dg,dbg)
 name=bfp__sk2_name(layer,m);ci,stream,tab,pmax,b4,n_rows=state['sk2'][name]
 out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 ext=_de_ext();ext.sk2_set_pdl(bfp_SK2_PDL)
 ext.sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,bfp__sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)
 return out

# Fresh-thread primary-context binding, following matvey's helper and the
import threading as _GlueContextThreading
_GLUE_CONTEXT_TLS = _GlueContextThreading.local()
_GLUE_CONTEXT_API = None

def _glue_context_api():
    global _GLUE_CONTEXT_API
    if _GLUE_CONTEXT_API is None:
        api = _C.CDLL("libcuda.so.1")
        for name, args in (
            ("cuCtxGetCurrent", [_C.POINTER(_C.c_void_p)]),
            ("cuCtxGetDevice", [_C.POINTER(_C.c_int)]),
            ("cuDeviceGet", [_C.POINTER(_C.c_int), _C.c_int]),
            ("cuDevicePrimaryCtxRetain", [_C.POINTER(_C.c_void_p), _C.c_int]),
            ("cuCtxSetCurrent", [_C.c_void_p]),
        ):
            function = getattr(api, name)
            function.argtypes, function.restype = args, _C.c_int
        _GLUE_CONTEXT_API = api
    return _GLUE_CONTEXT_API

def _glue_context_check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA driver status {status}")

def _glue_bind_context(x):
    index = x.device.index if x.device.index is not None else 0
    if getattr(_GLUE_CONTEXT_TLS, "device", None) == index:
        return
    api = _glue_context_api()
    context = _C.c_void_p()
    _glue_context_check(api.cuCtxGetCurrent(_C.byref(context)), "cuCtxGetCurrent")
    current_device = _C.c_int(-1)
    if context.value:
        _glue_context_check(api.cuCtxGetDevice(_C.byref(current_device)), "cuCtxGetDevice")
    if not context.value or current_device.value != index:
        device, primary = _C.c_int(), _C.c_void_p()
        _glue_context_check(api.cuDeviceGet(_C.byref(device), index), "cuDeviceGet")
        _glue_context_check(api.cuDevicePrimaryCtxRetain(_C.byref(primary), device), "cuDevicePrimaryCtxRetain")
        _glue_context_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    _GLUE_CONTEXT_TLS.device = index

_glue_context_parent_kernel = kernel_fn
def kernel_fn(data, state):
    _glue_bind_context(data["attn_input"])
    return _glue_context_parent_kernel(data, state)



@triton.jit
def _router_input_norm(S1, G, OUT):
    t = tl.program_id(0)
    offs = tl.arange(0, 8192)
    s1 = tl.load(S1 + t * 8192 + offs)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
    tl.store(OUT + t * 8192 + offs, hh)


def _pool_router_fp32(h, hr, T, state, ppick):
    if T <= 8:
        if state.get("bfp") is not None and ppick in ("bfp", "bfpf"):
            if ppick == "bfpf":
                _bfp_pdl(T, "pool")
            pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
        else:
            pooled = F.linear(h, state["pr"][:D_EXPERT])
        logits = _router_fp32_gemv(hr, state)
        return (pooled, logits), -1
    if ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:
        _bfp_pdl(T, "pool")
        return bfp_fused_pooler_router(h, state["bfp"]), -1
    if ppick == "bfp" and state.get("bfp") is not None:
        pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
        logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)
        return (pooled, logits), -1
    return dense(h, state["pr"], T, state, "pr")


_tune_with_bias_restore = _tune


def _tune(st, log=False):
    bias0 = st["bias"].clone()
    try:
        return _tune_with_bias_restore(st, log=log)
    finally:
        st["bias"].copy_(bias0)


_moe_global_cache = _moe


class _RouterExt:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __deepcopy__(self, memo):
        return self


def _moe(M=None, state=None):
    if state is None:
        return _moe_global_cache(M)
    sfx = tuple(state["ma_sfx"])
    if M is not None and MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
        key, defines = "tiny", MOE_TINY_DEFINES
    elif M is not None and MOE_TINY2_DEFINES and M <= MOE_TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES)):
        key, defines = "tiny2", MOE_TINY2_DEFINES
    elif M is not None and MOE_MID_DEFINES and M <= MOE_MID_MAX_M:
        key, defines = "mid", MOE_MID_DEFINES
    elif M is not None and 16 < M <= 32 and tuple(_main_defines(sfx)) == tuple(MOE_DEFINES):
        key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)
    else:
        key, defines = "main", _main_defines(sfx)
    ext = state["ma_ext"].get(key)
    if ext is None:
        ext = _RouterExt(_ma_ext(tuple(defines) + sfx))
        if MOE_HAS_PDL:
            ext.okc_set_pdl(0)
        state["ma_ext"][key] = ext
    return ext


_router_fp32_parent_prepare = prepare


def prepare(weights):
    st = _router_fp32_parent_prepare(weights)
    for T in (1, 8, 16, 32, 128, 256):
        _moe(T, st)
    return st



_ROUTER_FP32_CUDA = r"""
#include <cuda_runtime.h>
#include <stdint.h>

__global__ __launch_bounds__(256) void router_fp32_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int m) {
  const int expert = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* row = w + (size_t)expert * 8192;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int k = tid * 4; k < 8192; k += 1024) {
    const float4 b = *(const float4*)(row + k);
#pragma unroll
    for (int t = 0; t < 8; ++t) if (t < m) {
      const float4 a = *(const float4*)(x + (size_t)t * 8192 + k);
      acc[t] = fmaf(a.x, b.x, acc[t]);
      acc[t] = fmaf(a.y, b.y, acc[t]);
      acc[t] = fmaf(a.z, b.z, acc[t]);
      acc[t] = fmaf(a.w, b.w, acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int d = 16; d; d >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], d);
  }
  __shared__ float partial[8][8];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < 8; ++t) partial[warp][t] = acc[t];
  }
  __syncthreads();
  if (warp < m) {
    float v = lane < 8 ? partial[lane][warp] : 0.f;
#pragma unroll
    for (int d = 4; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    if (lane == 0) out[(size_t)warp * 256 + expert] = v;
  }
}

int router_fp32_launch(int64_t x, int64_t w, int64_t out, int m, int64_t stream) {
  if (m < 1 || m > 8) return -1;
  router_fp32_kernel<<<256, 256, 0, (cudaStream_t)stream>>>(
      (const float*)x, (const float*)w, (float*)out, m);
  return (int)cudaGetLastError();
}
"""
_ROUTER_FP32_EXT = None


def _router_fp32_ext():
    global _ROUTER_FP32_EXT
    if _ROUTER_FP32_EXT is None:
        import hashlib
        digest = hashlib.md5(_ROUTER_FP32_CUDA.encode()).hexdigest()[:12]
        cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint router_fp32_launch(int64_t,int64_t,int64_t,int,int64_t);'
        _ROUTER_FP32_EXT = load_inline(
            name='router_fp32_' + digest,
            cpp_sources=cpp,
            cuda_sources=_ROUTER_FP32_CUDA,
            functions=['router_fp32_launch'],
            no_implicit_headers=True,
            with_pytorch_error_handling=False,
            verbose=False,
            extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'])
    return _ROUTER_FP32_EXT


def _router_fp32_gemv(x, state):
    m = x.shape[0]
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    rc = state['router_fp32_ext'].router_fp32_launch(
        x.data_ptr(), state['router_fp32'].data_ptr(), out.data_ptr(), m,
        torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f'FP32 router launch failed: {rc}')
    return out


_router_fp32_native_parent_prepare = prepare


def prepare(weights):
    state = _router_fp32_native_parent_prepare(weights)
    state['router_fp32_ext'] = _RouterExt(_router_fp32_ext())
    return state


_kfix_router_fp32_gemv = _router_fp32_gemv
def _router_fp32_gemv(x, state):
    if state.get('router_fp32_ext') is None:
        state['router_fp32_ext'] = _RouterExt(_router_fp32_ext())
    return _kfix_router_fp32_gemv(x, state)


GLE_T = {1: True}   # T = 8 keeps 0389's b1
GLE_FRT = True      # T <= 8: the fused pooler + FP32 router kernel (else 0389's split pooler / FP32 GEMV)
GLE_ILV = 1         # its 4 FP32 router stages interleave the pooler stream: 1 last slot of each period, 2 first slot after the PDL pre-issue, 0 off
GLE_CAPTURE = None   # diagnostics: a dict receives this call's pooler / router outputs and partials
GLE_RC = {1: 8}   # T -> NCH: NCH * T extra route programs (no wait) form the carry, 8192 / (NCH * T) columns x T tokens each
_GLE_EXT = None


def _gle_source():
    cfg = next(c for c in bfp_SK2_CONFIGS if c[0] == bfp_SK2_PICK[("o", 8)])
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    assert (layer, R, NWR * WSK, CL) == ("o", 32, 4, 1), cfg
    _, nstage = bfp__sk2_geom(cfg, 8)
    s = bfp_SK2_SRC
    s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\n"
                  "struct GleEpi { const float *s, *a, *b, *a2; const __nv_bfloat16* p; __nv_bfloat16* uv; float* part; float* uvf; const float *x32, *w32; int fpr, ilv; };\n"
                  "static GleEpi g_gle;", 1)
    s = s.replace("           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {",
                  "           float* __restrict__ part, int* __restrict__ rcnt, int pdl, const GleEpi ep) {", 1)
    pre = r"""  __shared__ __align__(16) float4 gx_s[64], gx_a[64], gx_b[64], gx_a2[64];
  __shared__ __align__(16) uint2 gx_p[64];
  if (!FUSE && tid < 8 * M) {
    const size_t col = (size_t)cta * R + (tid & 7) * 4;
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&gx_s[tid]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d0), "l"(ep.s + (size_t)(tid >> 3) * N + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a[tid])), "l"(ep.a + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_b[tid])), "l"(ep.b + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a2[tid])), "l"(ep.a2 + col) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_p[tid])), "l"(ep.p + col) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
"""
    anchor = '  __syncthreads();\n  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread'
    assert s.count(anchor) == 1
    s = s.replace(anchor, pre + anchor, 1)
    epi = r"""  constexpr int R4 = R / 4;
  if constexpr (!FUSE) {
  static_assert(R4 == 8 && CL == 1 && sizeof(OutT) == 2, "gle epilogue: o8h geometry, BF16 out");
    const int tt = tid >> 3, row = (tid & 7) * 4;
    const bool on = tid < R4 * M;
    float q[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (on) {
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if constexpr (COMP) {
#pragma unroll
        for (int c = 0; c < NESC; ++c) {
          v.x += corr[c * CSZ + tt * CP + row];
          v.y += corr[c * CSZ + tt * CP + row + 1];
          v.z += corr[c * CSZ + tt * CP + row + 2];
          v.w += corr[c * CSZ + tt * CP + row + 3];
        }
      }
#pragma unroll
      for (int w = 0; w < WSK; ++w) {
        const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
        v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
      }
      const size_t col = (size_t)cta * R + row;
      OutT* o = out + (size_t)tt * N + col;
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
      const float pf[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
      asm volatile("cp.async.wait_all;" ::: "memory");   // this thread's own copies
      const uint2 xp = gx_p[tid];
      const float4 xs = gx_s[tid], xa = gx_a[tid], xb = gx_b[tid], xa2 = gx_a2[tid];
      const float4 xas = make_float4(__fmul_rn(xa.x, xs.x), __fmul_rn(xa.y, xs.y), __fmul_rn(xa.z, xs.z), __fmul_rn(xa.w, xs.w));
      const float pp[4] = {__fadd_rn(1.f, __uint_as_float(xp.x << 16)), __fadd_rn(1.f, __uint_as_float(xp.x & 0xffff0000u)),
                           __fadd_rn(1.f, __uint_as_float(xp.y << 16)), __fadd_rn(1.f, __uint_as_float(xp.y & 0xffff0000u))};
      const float ex_as[4] = {xas.x, xas.y, xas.z, xas.w}, ex_b[4] = {xb.x, xb.y, xb.z, xb.w};
      const float ex_a2[4] = {xa2.x, xa2.y, xa2.z, xa2.w};
      float uu[4], vv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float as = ex_as[j], bp = __fmul_rn(ex_b[j], pf[j]);
        uu[j] = __fmul_rn(as, pp[j]);
        vv[j] = __fmul_rn(bp, pp[j]);
        q[0] = __fmaf_rn(pf[j], pf[j], q[0]);
        q[1] = __fmaf_rn(as, as, q[1]);
        q[2] = __fmaf_rn(as, bp, q[2]);
        q[3] = __fmaf_rn(bp, bp, q[3]);
        if (col >= 1024) {
          const float x = __fmul_rn(ex_a2[j], as), y = __fmul_rn(ex_a2[j], bp);
          q[4] = __fmaf_rn(x, x, q[4]);
          q[5] = __fmaf_rn(x, y, q[5]);
          q[6] = __fmaf_rn(y, y, q[6]);
        }
      }
      auto put = [&](const float* f, size_t r) {
        __nv_bfloat162 h0 = __floats2bfloat162_rn(f[0], f[1]), h1 = __floats2bfloat162_rn(f[2], f[3]);
        uint2 hk;
        hk.x = *reinterpret_cast<uint32_t*>(&h0); hk.y = *reinterpret_cast<uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(ep.uv + r * N + col) = hk;
        *reinterpret_cast<float4*>(ep.uvf + r * N + col) = make_float4(f[0], f[1], f[2], f[3]);
      };
      put(uu, (size_t)tt);
      put(vv, (size_t)(M + tt));
    }
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 4);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 2);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 1);
    }
    if (on && (tid & 7) == 0) {
      float4* pq = reinterpret_cast<float4*>(ep.part + ((size_t)cta * M + tt) * 8);
      pq[0] = make_float4(q[0], q[1], q[2], q[3]);
      pq[1] = make_float4(q[4], q[5], q[6], 0.f);
    }
  }
  else
  for (int idx = tid; idx < R4 * M; idx += NC) {
"""
    anchor = '  constexpr int R4 = R / 4;\n  for (int idx = tid; idx < R4 * M; idx += NC) {'
    assert s.count(anchor) == 1
    s = s.replace(anchor, epi, 1)
    fpr_stage = (
        "        } else if (ep.fpr) {   // FP32 router stage j = st - nst: columns rq * 2048 + 512 j .. + 511\n"
        "          const int kb = rq * 2048 + (st - nst) * 512;\n"
        "          mbar_expect_tx(full0 + 8 * slot, 8 * 2048 + M * 2048);\n"
        "          for (int r = 0; r < 8; ++r)\n"
        "            bulk_g2s(s0 + r * 2048, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb), 2048, full0 + 8 * slot);\n"
        "          for (int t = 0; t < M; ++t)\n"
        "            bulk_g2s(s0 + 16384 + t * 2048, reinterpret_cast<const uint8_t*>(ep.x32 + (size_t)t * 8192 + kb), 2048, full0 + 8 * slot);\n"
        "        } else {   // router piece: global router stage rq * 4 + j of tile rt")
    fpr_cons = (
        "  if constexpr (FUSE) {\n    if (ep.fpr) {\n"
        "      static_assert(!FUSE || (NW == 8 && SST >= 32768), \"fp32 router stages: 8 consumer warps, 32 KB slots\");\n"
        "      // thread: column group cg = tid & 127 (4 of the stage's 512 columns) x all 8 router rows x tokens 4h .. 4h + 3\n"
        "      // (h = tid >> 7): every shared-memory float4 is read by 2 threads, not 8 warps; FP32 FMAs\n"
        "      const int cg = tid & 127, th = tid >> 7;\n"
        "      float af[32];\n"
        "#pragma unroll\n"
        "      for (int i = 0; i < 32; ++i) af[i] = 0.f;\n"
        "      for (int st = nst; st < nstt; ++st) {\n"
        "        const int slot = st % NSTAGE;\n"
        "        mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n"
        "        const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + cg * 4;\n"
        "        const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384) + cg * 4;\n"
        "        float4 xa[4];\n"
        "#pragma unroll\n"
        "        for (int u = 0; u < 4; ++u)\n"
        "          xa[u] = (th * 4 + u < M) ? *reinterpret_cast<const float4*>(xs + (th * 4 + u) * 512) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
        "#pragma unroll\n"
        "        for (int r = 0; r < 8; ++r) {\n"
        "          const float4 bw = *reinterpret_cast<const float4*>(ws + r * 512);\n"
        "#pragma unroll\n"
        "          for (int u = 0; u < 4; ++u) {\n"
        "            float v = af[r * 4 + u];\n"
        "            v = fmaf(xa[u].x, bw.x, v); v = fmaf(xa[u].y, bw.y, v); v = fmaf(xa[u].z, bw.z, v); v = fmaf(xa[u].w, bw.w, v);\n"
        "            af[r * 4 + u] = v;\n"
        "          }\n"
        "        }\n"
        "        __syncwarp();\n"
        "        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n"
        "      }\n"
        "      // reduction over the 128 column groups through the (now idle) ring: red[th][cg][33] (static indices only)\n"
        "      if constexpr (COMP) mbar_wait(cbar, 0);   // as the BF16 router: the escape warps are done with the ring\n"
        "      named_bar(1, NC);                         // and every consumer is done reading its last stage\n"
        "      float* red32 = reinterpret_cast<float*>(gbase);\n"
        "#pragma unroll\n"
        "      for (int i = 0; i < 32; ++i) red32[(th * 128 + cg) * 33 + i] = af[i];\n"
        "      named_bar(1, NC);\n"
        "      float* red2 = red32 + 256 * 33;\n"
        "      {\n"
        "        const int o = tid & 63, pq = tid >> 6, h = o >> 5, e = o & 31;\n"
        "        float v = 0.f;\n"
        "        for (int c = pq * 32; c < pq * 32 + 32; ++c) v += red32[(h * 128 + c) * 33 + e];\n"
        "        red2[pq * 64 + o] = v;\n"
        "      }\n"
        "      named_bar(1, NC);\n"
        "      if (tid < 64) {\n"
        "        const int h = tid >> 5, e = tid & 31, r = e >> 2, t = h * 4 + (e & 3);\n"
        "        const float v = red2[tid] + red2[64 + tid] + red2[128 + tid] + red2[192 + tid];\n"
        "        if (t < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + t * 8 + r, v);\n"
        "      }\n"
        "      named_bar(1, NC);\n"
        "    } else {\n    // router piece:")
    ilv_edits = (
        ("  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n",
         "  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n"
         "  const bool ilv = FUSE && ep.fpr && ep.ilv && (nstt % 4) == 0 && nstt / 4 > P;\n"
         "  const int PER = ilv ? nstt / 4 : (1 << 30);\n"
         "  const int OFF = ilv ? (ep.ilv == 2 ? P : PER - 1) : 0;   // router stage slot inside each period\n"),
        ("        if (!FUSE || st < nst) {\n"
         "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
         "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);\n"
         "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n",
         "        const bool isr = ilv ? (st % PER == OFF) : (FUSE && st >= nst);\n"
         "        const int ps = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\n"
         "        if (!isr) {\n"
         "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
         "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + ps) * wstride, WBY, full0 + 8 * slot);\n"
         "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + ps) * UPS);\n"),
        ("          const int kb = rq * 2048 + (st - nst) * 512;\n",
         "          const int kb = rq * 2048 + (ilv ? st / PER : st - nst) * 512;\n"),
        ("        if (FUSE && st >= nst) {   // router stages:",
         "        if (ilv ? (st % PER == OFF) : (FUSE && st >= nst)) {   // router stages:"),
        ("        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;\n",
         "        const int psq = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\n"
         "        const int kbeg = (gs0 + psq) * KST, kend = kbeg + KST;\n"),
        ("        if ((st % NESC) != ew) {", "        if ((psq % NESC) != ew) {"),
        ("  for (int st = 0; st < nst; ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n",
         "  float afi[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};   // ilv: FP32 router accumulators (warp = router row)\n"
         "  for (int st = 0; st < (ilv ? nstt : nst); ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n"
         "    if constexpr (FUSE) if (ilv && st % PER == OFF) {   // an interleaved FP32 router stage\n"
         "      const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + warp * 512;\n"
         "      const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384);\n"
         "#pragma unroll\n"
         "      for (int i = 0; i < 4; ++i) {\n"
         "        const int k = lane * 4 + i * 128;\n"
         "        const float4 bw = *reinterpret_cast<const float4*>(ws + k);\n"
         "#pragma unroll\n"
         "        for (int t = 0; t < 8; ++t) if (t < M) {\n"
         "          const float4 a = *reinterpret_cast<const float4*>(xs + t * 512 + k);\n"
         "          afi[t] = fmaf(a.x, bw.x, afi[t]); afi[t] = fmaf(a.y, bw.y, afi[t]);\n"
         "          afi[t] = fmaf(a.z, bw.z, afi[t]); afi[t] = fmaf(a.w, bw.w, afi[t]);\n"
         "        }\n"
         "      }\n"
         "      __syncwarp();\n"
         "      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n"
         "      continue;\n"
         "    }\n"),
        ("  if constexpr (FUSE) {\n    if (ep.fpr) {\n",
         "  if constexpr (FUSE) {\n    if (ilv) {   // interleaved FP32 router: warp butterflies of afi, then the 4-quarter reduction below\n"
         "#pragma unroll\n"
         "      for (int t = 0; t < 8; ++t)\n"
         "#pragma unroll\n"
         "        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);\n"
         "      float mine = afi[0];\n"
         "#pragma unroll\n"
         "      for (int t = 1; t < 8; ++t) mine = lane == t ? afi[t] : mine;\n"
         "      if (lane < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + lane * 8 + warp, mine);\n"
         "      if constexpr (COMP) mbar_wait(cbar, 0);\n"
         "      named_bar(1, NC);\n"
         "    } else if (ep.fpr) {\n"))
    for old, new in (
            ("        } else {   // router piece: global router stage rq * 4 + j of tile rt", fpr_stage),
            ("  if constexpr (FUSE) {\n    // router piece:", fpr_cons),
            ("    named_bar(1, NC);\n    __shared__ int s_rlast;",
             "    named_bar(1, NC);\n    }\n    __shared__ int s_rlast;")):
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, new, 1)
    for old, new in ilv_edits:
        assert s.count(old) == 1, old[:70]
        s = s.replace(old, new, 1)
    s = s.replace("wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl, g_gle));", 1)
    s = s.replace("wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, 0, g_gle);", 1)
    host = (f"void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,\n"
            f"         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {{")
    assert s.count(host) == 1
    disp = (f"if (tokr == 8 && N == 8192 && K == 1024 && M <= 8) {{ launch_sk2<1, 8, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
            f"{NACC}, {CL}, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, 0, nullptr); return; }}")
    pcfg = next(c for c in bfp_SK2_CONFIGS if c[0] == "pooler")
    flines = []
    for tokr in (8,):   # the fused pooler + FP32 router at M <= 8
        _, pns = bfp__sk2_geom(pcfg, tokr)
        _, _, _, pR, pNT, pNWR, pWSK, pUPW, pNACC, pCL = pcfg
        flines.append(f"if (tokr == {tokr}) {{ launch_sk2<1, {tokr}, {pNT}, {pNWR}, {pWSK}, {pUPW}, {pns}, true, false, "
                      f"{pNACC}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                      f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                      f"rcnt.data_ptr<int>()); return; }}")
    s = s.replace("@@SK2_DISPATCH@@", disp).replace("@@SK2F_DISPATCH@@", "\n  ".join(flines))
    s += r"""
void gle_o(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
           at::Tensor s, at::Tensor a, at::Tensor b, at::Tensor a2, at::Tensor p, at::Tensor uv, at::Tensor part, at::Tensor uvf) {
  TORCH_CHECK(M >= 1 && M <= 8 && s.size(0) == M && uv.size(0) == 2 * M && uvf.size(0) == 2 * M && part.numel() >= 256 * M * 8, "gle_o: shapes");
  g_gle = GleEpi{s.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), a2.data_ptr<float>(),
                 reinterpret_cast<const __nv_bfloat16*>(p.data_ptr()), reinterpret_cast<__nv_bfloat16*>(uv.data_ptr()),
                 part.data_ptr<float>(), uvf.data_ptr<float>()};
  sk2(Ws, x, out, ptab, pmax, base4, M, 8192, 1024, 0, 8, 0, ptab);
}
void gle_pr(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
            int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, at::Tensor x32, at::Tensor w32, int64_t wpf) {
  TORCH_CHECK(M >= 1 && M <= 8 && x.size(0) == M && x32.size(0) == M && x32.scalar_type() == at::kFloat &&
              w32.scalar_type() == at::kFloat && w32.size(0) == 256 && w32.size(1) == 8192, "gle_pr: shapes");
  g_gle = GleEpi{};
  g_gle.x32 = x32.data_ptr<float>(); g_gle.w32 = w32.data_ptr<float>(); g_gle.fpr = 1; g_gle.ilv = (int)wpf;
  sk2f(Ws, Wsr, x, out, outr, ptab, pmax, base4, part, rcnt, M, 8);
  g_gle = GleEpi{};
}
"""
    shared = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", "")
    return shared + bfp_MAP_SRC + s


def _gle_ext():
    global _GLE_EXT
    if _GLE_EXT is None:
        source = _gle_source()
        tag = _hashlib.md5(source.encode()).hexdigest()[:12]
        source = (source.replace("sk2_kernel", "sk2_kernel_gle" + tag).replace("launch_sk2", "launch_sk2_gle" + tag)
                  .replace("void sk2(", "void sk2_gle" + tag + "(").replace("  sk2(Ws,", "  sk2_gle" + tag + "(Ws,")
                  .replace("void sk2f(", "void sk2f_gle" + tag + "(").replace("  sk2f(Ws,", "  sk2f_gle" + tag + "(Ws,"))
        cpp = ("void gle_o(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,"
               " int64_t M, torch::Tensor s, torch::Tensor a, torch::Tensor b, torch::Tensor a2, torch::Tensor p,"
               " torch::Tensor uv, torch::Tensor part, torch::Tensor uvf);\n"
               "void gle_pr(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr,"
               " torch::Tensor ptab, int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M,"
               " torch::Tensor x32, torch::Tensor w32, int64_t wpf);\nvoid sk2_set_pdl(int64_t n);\n")
        _GLE_EXT = load_inline("gle_o_" + tag, "#include <torch/extension.h>\n" + cpp, cuda_sources=source,
                               functions=["gle_o", "gle_pr", "sk2_set_pdl"], no_implicit_headers=True,
                               extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)
    return _GLE_EXT


@triton.jit
def _gle_carry(PO, S, ALPHA, BETA, ALPHA2, t, cols, r1, ROWS1: tl.constexpr):
    p = tl.load(PO + t * 8192 + cols).to(tl.float32)
    s = tl.load(S + t * 8192 + cols)
    a = tl.load(ALPHA + cols)
    b = tl.load(BETA + cols)
    bn = (p * r1).to(tl.bfloat16).to(tl.float32)
    if ROWS1:
        ns = tl.fma(b, bn, a * s)
    else:
        ns = tl.where(cols < 4096, tl.fma(a, s, b * bn), tl.fma(b, bn, a * s))
    return tl.load(ALPHA2 + cols) * ns


@triton.jit
def _gle_route(POOLED, LOGITS, PART, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, RT,
               PO, S, ALPHA, BETA, ALPHA2, S1,
               MODE: tl.constexpr, PARTS: tl.constexpr, NCH: tl.constexpr, TT: tl.constexpr):
    t = tl.program_id(0)
    kind = tl.program_id(1)
    k8 = tl.arange(0, 8)
    cta = tl.arange(0, 256)
    q = tl.sum(tl.load(PART + (cta[:, None] * T + t) * 8 + k8[None, :]), 0)
    q0 = tl.sum(tl.where(k8 == 0, q, 0.0), 0)
    q1 = tl.sum(tl.where(k8 == 1, q, 0.0), 0)
    q2 = tl.sum(tl.where(k8 == 2, q, 0.0), 0)
    q3 = tl.sum(tl.where(k8 == 3, q, 0.0), 0)
    r1 = tl.rsqrt(q0 / 8192.0 + 1e-8)
    r2 = tl.rsqrt((q1 + 2.0 * r1 * q2 + r1 * r1 * q3) / 8192.0 + 1e-5)
    if kind > PARTS:
        cc = ((kind - PARTS - 1) * TT + t) * (8192 // (NCH * TT)) + tl.arange(0, 8192 // (NCH * TT))
        for tok in tl.static_range(TT):
            r1t = tl.rsqrt(tl.sum(tl.load(PART + (cta * TT + tok) * 8), 0) / 8192.0 + 1e-8)
            tl.store(S1 + tok * 8192 + cc, _gle_carry(PO, S, ALPHA, BETA, ALPHA2, tok, cc, r1t, TT == 1))
    elif kind < PARTS:
        c = tl.arange(0, 4096)
        oc = kind * (4096 // PARTS) + tl.arange(0, 4096 // PARTS)
        gpre = 1.0 + tl.load(GPRE + oc).to(tl.float32)
        _gdc_wait()
        _gdc_launch()
        pooled = (tl.load(POOLED + t * 4096 + c).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + c).to(tl.float32) * r1) * r2
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
        if PARTS == 1:
            vals = pooled
        else:
            vals = (tl.load(POOLED + t * 4096 + oc).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + oc).to(tl.float32) * r1) * r2
        tl.store(XOUT + t * 4096 + oc, (vals * inv * gpre).to(tl.bfloat16))
    else:
        e = tl.arange(0, 256)
        bias = tl.load(BIAS + e)
        ls = tl.load(LS)
        q4 = tl.sum(tl.where(k8 == 4, q, 0.0), 0)
        q5 = tl.sum(tl.where(k8 == 5, q, 0.0), 0)
        q6 = tl.sum(tl.where(k8 == 6, q, 0.0), 0)
        tail = q4 + 2.0 * r1 * q5 + r1 * r1 * q6
        _gdc_wait()
        _gdc_launch()
        tl.store(RT + t, r1)
        tl.store(RT + T + t, tail)
        logit = (tl.load(LOGITS + t * 256 + e) + tl.load(LOGITS + (T + t) * 256 + e) * r1) * r2
        sc = tl.sigmoid(logit * 0.1767766952966369)
        sel = sc + bias
        if MODE != 2:
            bits = sel.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
            if MODE == 0:
                picked = tl.topk(packed, 8)
            else:
                sorted_values = tl.sort(packed, descending=True)
                picked = tl.gather(sorted_values, k8, 0)
            ids = (255 - (picked & 255)).to(tl.int32)
            wv = tl.gather(sc, ids, 0)
        else:
            ids = tl.zeros([8], dtype=tl.int32)
            wv = tl.zeros([8], dtype=tl.float32)
            for k in tl.static_range(8):
                idx = tl.argmax(sel, 0)
                val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                ids = tl.where(k8 == k, idx, ids)
                wv = tl.where(k8 == k, val, wv)
                sel = tl.where(e == idx, float("-inf"), sel)
        wv = wv / (tl.sum(wv, 0) + 1e-15)
        tl.store(IDS + t * 8 + k8, ids)
        lid = (ids - ls + 256) % 256
        tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
        tl.store(WTS + t * 8 + k8, wv)


def _gle_frt(xb, x32, state):
    """Fused BF16 pooler + FP32 router for M = xb.shape[0] <= 8 rows: (pooled bf16 [M, 4096], logits fp32 [M, 256])."""
    bst = state["bfp"]
    m = xb.shape[0]
    dev = xb.device
    if "fused_part" not in bst:
        bst["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
        bst["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
    pci, pstream, ptab_, ppmax, pb4, _ = bst["sk2"]["pooler"]
    pooled = torch.empty((m, 4096), device=dev, dtype=torch.bfloat16)
    logits = torch.empty((m, 256), device=dev, dtype=torch.float32)
    _bfp_pdl(m, "pool")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_pr(pstream, bst["sk2"]["router"][1], xb, pooled, logits, ptab_, ppmax, pb4, bst["fused_part"], bst["fused_cnt"],
               m, x32, state["router_fp32"], int(GLE_ILV))
    return pooled, logits


_gle_prior_pool_router_fp32 = _pool_router_fp32
def _pool_router_fp32(h, hr, T, state, ppick):
    if (GLE_FRT and T <= 8 and hr is not None and ppick in ("bfp", "bfpf") and state.get("bfp") is not None
            and state.get("router_fp32") is not None and hr.is_contiguous() and h.is_contiguous()):
        return _gle_frt(h, hr, state), -1
    return _gle_prior_pool_router_fp32(h, hr, T, state, ppick)


def _gle_o_choice(state, T):
    bst = state.get("bfp")
    if bst is None or not bst.get("sm90"):
        return False
    ch = bst.get("choice", {}).get(("o", T)) or bfp__nearby_choice("o", T, bst) or bfp__static_choice("o", T, bst)
    return ch[0] == "sk2" and bfp__sk2_name("o", T) == bfp_SK2_PICK[("o", 8)]


def _gle_after_attention(data, state, o_in, gate, T):
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    g = state['graft']
    bst = state["bfp"]
    ci, stream, tab, pmax, b4, n_rows = bst["sk2"][bfp__sk2_name("o", T)]
    P = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uv = torch.empty((2 * T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uvf = torch.empty((2 * T, HIDDEN), **f32)
    part = torch.empty((256 * T * 8,), **f32)
    rt = torch.empty((2, T), **f32)
    nch = GLE_RC.get(T, GLE_RC.get(str(T), 0))
    s1 = torch.empty((T, HIDDEN), **f32)
    _bfp_pdl(T, "o")
    ext = _gle_ext()
    ext.sk2_set_pdl(bfp_SK2_PDL)
    ext.gle_o(stream, o_in, P, tab, pmax, b4, T, data['stream'], g['alpha1'], g['beta1'], g['alpha2'], state['g_preffn'], uv, part,
              uvf)
    if GLE_FRT:   # fused pooler + FP32 router (one launch)
        pooled, logits = _gle_frt(uv, uvf, state)
    else:         # 0389's T <= 8 split: BF16 pooler (bf16_proj sk2), FP32 router GEMV, here on the two rows per token
        pooled = _bfp_gemm(state, "pool", uv, 2 * T, D_EXPERT, HIDDEN)
        logits = _router_fp32_gemv(uvf, state)
    if GLE_CAPTURE is not None:
        GLE_CAPTURE.update(logits=logits, pooled=pooled, part=part, P=P)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    _graft_al(pooled, logits, xm, ids, wts, lids, part, rt)
    _gle_route[(T, SL_TILE_CONFIG[T][1] + 1 + nch)](pooled, logits, part, T, state["bias"], state["g_preexp"], ids, wts, xm,
                                                    data["local_start"], lids, rt, P, data['stream'], g['alpha1'], g['beta1'],
                                                    g['alpha2'], s1, MODE=SL_ROUTE_CONFIG[T][1],
                                                    PARTS=SL_TILE_CONFIG[T][1], NCH=nch, TT=T,
                                                    num_warps=SL_TILE_CONFIG[T][0], launch_pdl=True)
    m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    mx = _moe(T, state)
    if MOE_HAS_PDL:
        mx.okc_set_pdl(1 if _pdl(T) else 0)
    for m0 in range(0, T, _MA_MMAX):
        m1 = min(T, m0 + _MA_MMAX)
        mx.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32, state["nsm"])
    U, S = dense(m, state["unpool_slice"], T, state, "un")
    if S != 1 or U.dtype != torch.bfloat16:
        raise RuntimeError('gle: BF16 sliced unpooler expected')
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    nw, bc = SL_CLUSTER_CONFIG[T]
    _sl_cluster_finish[(T, 8192 // bc)](U, T, s1, rt[1], g['beta2'], g['post2'], g['pre2'], s2, out, BC=bc, num_warps=nw,
                                        launch_pdl=True)
    return out, s2, ids


_gle_prior_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    if (GLE_T.get(T, GLE_T.get(str(T), False)) and GLE_RC.get(T, 0) > 0 and gate is None and _pdl(T) and 'graft' in state
            and state.get('compose_graft_enabled', {}).get(T, True) and T in SL_TILE_CONFIG and T in SL_CLUSTER_CONFIG):
        mmp = state.get("mm_pick", {})
        ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
        opick = mmp.get((T, "o"), mmp.get((_bucket(T), "o"), False))
        moe = _pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent")),
                                   marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
        gw = state.get('g_preffn')
        if (ppick in ("bfp", "bfpf") and opick == "bfp" and moe == "agent" and state.get("bfp") is not None
                and state.get("router_fp32") is not None and state.get("router_fp32_ext") is not None
                and gw is not None and gw.dtype == torch.bfloat16 and gw.numel() == HIDDEN and gw.is_contiguous()
                and _gle_o_choice(state, T)):
            return _gle_after_attention(data, state, o_in, gate, T)
    return _gle_prior_after(data, state, o_in, gate, T)


_gle_prior_prepare = prepare
def prepare(weights):
    _gle_ext()
    return _gle_prior_prepare(weights)


GLE_B1X = True
_GLE_W_CUR = None


def _gle_b1x_source():
    s = _SL_GLUE_CUDA
    for old, new in (
            ("float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];\n  __shared__ float partials[3][NW];",
             "float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N],nsx[N];\n  __shared__ float partials[4][NW];"),
            ("    if constexpr(POST) glb<V>(a.post+col,post+r*V);\n",
             "    if constexpr(POST || ROUTER) glb<V>(a.post+col,post+r*V);   // ROUTER: post = w (pre_ffn_norm)\n"),
            ("        float bn=gb(gm(b[k],r1));\n",
             "        float bn=gb(gm(b[k],r1));\n"
             "        if constexpr(ROUTER) {   // the router's FP32 copy: no BF16 rounding of r1 * P\n"
             "          float bx=gm(b[k],r1);\n"
             "          nsx[k]=(SLICED || (PRELOADED_BASE && col<4096)) ? gf(alpha[k],s[k],gm(beta[k],bx)) : gf(beta[k],bx,gm(alpha[k],s[k]));\n"
             "        }\n"),
            ("  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);\n",
             "  float r2, r2x=0.f;\n"
             "  if constexpr(ROUTER) {   // the router copy's sum of squares rides the same cluster exchange\n"
             "    float zs;\n"
             "    r2=gr(cluster_squares_triple<NW>(ns,masked,nsx,partials[1],partials[2],partials[3],cluster,rank,tail,zs),1.e-5f);\n"
             "    r2x=gr(zs,1.e-5f);\n"
             "  } else {\n"
             "    r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);\n"
             "  }\n"),
            ("template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n",
             "template<int NW,int N>\n"
             "__device__ __forceinline__ float cluster_squares_triple(const float (&x)[N], const float (&y)[N], const float (&z)[N],\n"
             "    float* sm, float* smy, float* smz, cg::cluster_group cluster, int rank, float& tail, float& zsum) {\n"
             "  constexpr int warps_per_cta=NW/8;\n"
             "  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;\n"
             "  float sx=butterfly(local_squares(x),16);\n"
             "  float sy=butterfly(local_squares(y),16);\n"
             "  float sz=butterfly(local_squares(z),16);\n"
             "  if(lane==0 && warp<warps_per_cta) {\n"
             "    #pragma unroll\n"
             "    for(int dst=0;dst<8;dst++) {\n"
             "      volatile float* px=cluster.map_shared_rank(sm,dst);\n"
             "      volatile float* py=cluster.map_shared_rank(smy,dst);\n"
             "      volatile float* pz=cluster.map_shared_rank(smz,dst);\n"
             "      px[rank*warps_per_cta+warp]=sx;\n"
             "      py[rank*warps_per_cta+warp]=sy;\n"
             "      pz[rank*warps_per_cta+warp]=sz;\n"
             "    }\n"
             "  }\n"
             "  cluster.sync();\n"
             "  tail=butterfly(smy[lane&(NW-1)],NW/2);\n"
             "  zsum=butterfly(smz[lane&(NW-1)],NW/2);\n"
             "  return butterfly(sm[lane&(NW-1)],NW/2);\n"
             "}\n\n"
             "template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n"),
            ("      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);\n",
             "      if constexpr(ROUTER) {\n"
             "        float hx[V];\n"
             "        #pragma unroll\n"
             "        for(int j=0;j<V;j++) hx[j]=gm(gm(nsx[r*V+j],r2x),ga(1.f,post[r*V+j]));\n"
             "        gsf<V>(a.router+row*8192+col,hx);\n"
             "      }\n"),
            ("void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {\n"
             "  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),",
             "void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router,TV w) {\n"
             "  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)w.data_ptr(),(const BF*)pre.data_ptr(),")):
        assert s.count(old) == 1, old[:70]
        s = s.replace(old, new, 1)
    return s


@_lru_cache(None)
def _gle_b1x_module():
    from tvm_ffi.cpp import load_inline as _tvm_load_inline
    src = _gle_b1x_source()
    return _tvm_load_inline(name='gle_b1x_' + _hashlib.md5(src.encode()).hexdigest()[:12], cuda_sources=src,
                            functions=['b1'], extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math'])


class _GleB1x:
    """_sl_cluster_module() stand-in: the router-writing b1 goes to the exact-FP32 copy (w from the calling state)."""
    def __init__(self, base):
        self.base = base

    def b1(self, branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, write_router):
        if write_router and GLE_B1X and _GLE_W_CUR is not None:
            return _gle_b1x_module().b1(branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, True, _GLE_W_CUR)
        return self.base.b1(branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, write_router)


_gle_b1x_prior_module = _sl_cluster_module
def _sl_cluster_module():
    return _GleB1x(_gle_b1x_prior_module())


_gle_b1x_prior_after = _after_attention
def _after_attention(data, state, o_in, gate, T):
    global _GLE_W_CUR
    gw = state.get('g_preffn')
    _GLE_W_CUR = gw if (gw is not None and gw.dtype == torch.bfloat16 and gw.numel() == HIDDEN and gw.is_contiguous()) else None
    return _gle_b1x_prior_after(data, state, o_in, gate, T)


_gle_b1x_prior_prepare = prepare
def prepare(weights):
    _gle_b1x_module()
    return _gle_b1x_prior_prepare(weights)


_pin_parent_prepare = prepare
def prepare(weights):
    st = _pin_parent_prepare(weights)
    cge = st.setdefault('compose_graft_enabled', {})
    for T in (1, 8):
        cge[T] = True
    return st


# c001: _make_chen (rs128 / rq closure, used only by the rq bake above RQ_USE) removed; its attribution:
# (embedded: chenwainuo muse_bf16_proj)
# Based on github.com/chenwainuo/okc muse_bf16_proj/submission.py @ 3d418e1 (Apache-2.0). Copied for the OKC starting point.


# ---------------------------------------------------------------------------------------------- rs128 (chenwainuo)
_CHEN = {}  # c001: no rs128 closure (RQ_USE is empty, st["chen_rq"] is always {})
_rq_prior_prepare = prepare


def prepare(weights):
    # c001: the rs128 rq bake (pw57_flush build, FLUSH_VALIDATE, FIXED_COST, rq extension thread, BAKE / PICK) is
    # gone: its picks were filtered by RQ_USE, which is empty, so st["chen_rq"] was always {} and _rq_gemm
    # always returned None (the layer's own qkv / unpooler / pooler paths ran at every T).
    assert not RQ_USE
    st = _rq_prior_prepare(weights)
    st["chen_rq"] = {}
    torch.cuda.synchronize()
    return st


RQ_USE = set()  # f005: no BF16 split-K partial reduce (precision rule)


def _rq_gemm(state, name, x, T):
    p = state.get("chen_rq", {}).get((name, T)) if state is not None else None
    if p is None:
        return None
    out = torch.empty((T, p["n"]), dtype=torch.bfloat16, device=x.device)
    if _CHEN["_rqcall"](state["chen_st"], p, x.contiguous(), out) != 0:
        return None
    return out


# h501 EX (EROREx, 2026-11-01): T = 2..8 run EROREx's GLE chain (from EROREx's v480_cpbx decode layer, Apache-2.0;
# its own design notes: the deferred boundary-1 normalization follows alygg77/okc-muse-spark-kernels, Apache-2.0) on
EX_T = tuple(range(2, 9))
EX_NCH = 8            # carry programs per token
EX_ROUTE_WARPS = 4
EX_TOPK = 1           # 1: tl.topk on packed keys (8 x max where Triton has none)
EX_FIN_CFG = ((1, 8, 256), (2, 8, 512), (4, 8, 1024), (8, 16, 1024))
EX_PDL_N = 2          # pooler + router launch: weight stages issued before griddepcontrol.wait
EX_ILR_RING_KB = 216
EX_SEEDS = 4
EX_FLIP_TOL = 1e-2    # biased-score margin a flipped expert may have (official: ROUTE_TOL 5e-2, KEEP_TOL 3e-2)
EX_AGREE_MIN = 0.75   # share of self-test tokens routed exactly like this file's chain
EX_ROW_TOL = 4e-2     # per row (official compare: 5 x tol, tol >= 3e-2)
EX_AGG_TOL = 2e-2
EX_TIME_T = tuple(range(2, 8))   # nearby (unscored) T: EX only where it times no slower
EX_SLOW = 1.005
_EX_EXT = None
_EX_ERR = None

_EX_SK2_SRC = r"""

#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
static int g_sk2_pdl = 0;
void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }
#define SK2_DIAG 0
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK, bool FUSE, bool ILR>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,
           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {
  const int dgv = SK2_DIAG ? dg : 0;
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;
  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);
  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;
  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;
  if (trace && threadIdx.x == 0) tr[0] = t_start;
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int WBYI = ILR ? NW * 512 : 0;
  constexpr int SST = ((XBY + WBY + WBYI) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
  constexpr int NSTR = (FUSE && !ILR) ? 4 : 0;
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");
  static_assert(!ILR || (FUSE && NWR == 1 && NW == UPS), "sk2 in-loop router: one 64-k unit per consumer warp");
  const int rt = cta >> 2, rq = cta & 3;
  const uint8_t* wsr = FUSE ? Wsr + (size_t)(ILR ? cta * WBYI : rt * WBYR) : nullptr;
  const size_t wstrr = (size_t)gridDim.x * WBYI;
  const int nstt = nst + NSTR;
  const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;
  const int etst = ((dg >> 16) & 255) ? max(0, nst - ((dg >> 16) & 255)) : -1;
#ifdef SK2_ET
#define SK2_ET_TRIG(s_, e_) asm volatile("{\n .reg .pred p;\n setp.eq.s32 p, %0, %1;\n @p griddepcontrol.launch_dependents;\n}\n" :: "r"(s_), "r"(e_));
#else
#define SK2_ET_TRIG(s_, e_) ((void)0)
#endif

  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL == 1)   // PDL: the first P stages' weights go out before the wait; their barriers first
      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int st = 0; st < P; ++st) {
      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY + WBYI);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
      if constexpr (ILR) bulk_g2s(sbase + st * SST + XBY + WBY, wsr + (size_t)(gs0 + st) * wstrr, WBYI, full0 + 8 * st);
    }
    if (COMP) {   // this CTA's escape table (EROREx v367: weight data, so before the wait, right behind the pre-wait
      if constexpr (CL == 1) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (!(dgv & 4))
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) {
        if (i >= P) mbar_init(full0 + 8 * i, 1);
        mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));
      }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    if (lane == 0) {
      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY + WBYI);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if constexpr (ILR) bulk_g2s(s0 + XBY + WBY, wsr + (size_t)(gs0 + st) * wstrr, WBYI, full0 + 8 * slot);
          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
        } else {
          const int gsr = rq * NSTR + (st - nst);
          mbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);
          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);
        }
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      int pe = 1;
      for (int st = 0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        SK2_ET_TRIG(st, etst);
        if (FUSE && st >= nst) {
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;
  float accr2[ILR ? MT : 1][4];
#pragma unroll
  for (int mt = 0; mt < (ILR ? MT : 1); ++mt) accr2[mt][0] = accr2[mt][1] = accr2[mt][2] = accr2[mt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    SK2_ET_TRIG(st, etst);
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();
    const uint32_t s0 = sbase + slot * SST;
    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      uint32_t rb[ILR ? 8 : 1];
      if constexpr (ILR) {   // this warp's unit of the CTA's router tile: 32 B per lane, lanes 16..31 are zero columns
        const uint32_t ra = s0 + XBY + WBY + ui * 512 + (lane & 15) * 32;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(rb[0]), "=r"(rb[1]), "=r"(rb[2]), "=r"(rb[3]) : "r"(ra));
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(rb[4]), "=r"(rb[5]), "=r"(rb[6]), "=r"(rb[7]) : "r"(ra + 16));
        if (lane >= 16) {
#pragma unroll
          for (int i = 0; i < 8; ++i) rb[i] = 0u;
        }
      }
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      if (dgv & 8) {
        const int nh = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
          }
        continue;
      }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
      if constexpr (ILR) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            mma_bf16(accr2[mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], rb[2 * j], rb[2 * j + 1]);
      }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();
  }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the next kernel may start its prologue
  if constexpr (FUSE && !ILR) {
    float accr[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
    for (int st = nst; st < nstt; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const uint32_t s0 = sbase + slot * SST;
      const int ui = wk;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;
      uint32_t bh[8], bo[8];
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3]) : "r"(a0));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7]) : "r"(a0 + 16));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3]) : "r"(a0 + 1024));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7]) : "r"(a0 + 1040));
      uint32_t a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4((PK && trow[mt] >= M) ? zrow
                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
        }
    }
    if constexpr (COMP) mbar_wait(cbar, 0);
    named_bar(1, NC);
    float* redr = reinterpret_cast<float*>(gbase);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = 2 * t4, tok = mt * 16 + g;
      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
    }
    named_bar(1, NC);
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
    for (int i = tid; i < TOKS * 8; i += NC) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
      __stcg(pp + i, v);
    }
    named_bar(1, NC);
    __shared__ int s_rlast;
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");
      s_rlast = (old & 3) == 3;
      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (s_rlast) {
      const float* pr = part + (size_t)rt * 4 * 32 * 8;
      for (int i = tid; i < M * 8; i += NC) {
        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);
        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();

  if (dgv & 1) {
    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);
    return;
  }
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    if (trace && tid == 0) tr[15] = gtime();
    return;
  }
  named_bar(1, NC);
  if (trace && tid == 0) tr[9] = gtime();
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      if (trace && mt == 0 && nt == 0) {
        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
        if (tid == 0) tr[12] = gtime();
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  float* const redi = red + WSK * TOKS * RP;   // ILR: [WSK][TOKS][4] router k-warp partials after the pooler's
  if constexpr (ILR) {
    if (t4 < 2) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int tok = mt * 16 + g;
        *reinterpret_cast<float2*>(redi + (wk * TOKS + tok) * 4 + 2 * t4) = make_float2(accr2[mt][0], accr2[mt][1]);
        *reinterpret_cast<float2*>(redi + (wk * TOKS + tok + 8) * 4 + 2 * t4) = make_float2(accr2[mt][2], accr2[mt][3]);
      }
    }
  }
  if (trace && tid == 0) tr[11] = gtime();
  named_bar(1, NC);
  if (trace && tid == 0) tr[10] = gtime();
  if (trace && tid == 0) tr[13] = gtime();
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  if (trace && tid == 0) tr[14] = gtime();
  if constexpr (ILR) {   // the CTA's 2 router rows over all of K: k-warps in order, then hi + lo; FP32 out
    if (tid < 2 * M) {
      const int tt = tid >> 1, r = tid & 1;
      float h = 0.f, l = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) {
        h += redi[(w * TOKS + tt) * 4 + r];
        l += redi[(w * TOKS + tt) * 4 + 2 + r];
      }
      outr[(size_t)tt * 256 + 2 * cta + r] = h + l;
    }
  }
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
  if (trace && tid == 0) tr[15] = gtime();
  if ((dgv & 32) && dbg != nullptr && tid == 0) {
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();
  }
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false, bool ILR = false>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg,
                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int WBYI = ILR ? NW * 512 : 0;
  constexpr int SST = ((XBY + WBY + WBYI) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE, ILR>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    const int pdl = g_sk2_pdl;
    if (CL > 1 || pdl > 0) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[2];
      int na = 0;
      if (CL > 1) {
        at_[na].id = cudaLaunchAttributeClusterDimension;
        at_[na].val.clusterDim.x = CL;
        at_[na].val.clusterDim.y = 1;
        at_[na].val.clusterDim.z = 1;
        ++na;
      }
      if (pdl > 0) {
        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at_[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
      }
      lc.attrs = at_;
      lc.numAttrs = na;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg,
                                        wsr, outr, part, rcnt, pdl));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg,
                                                                               wsr, outr, part, rcnt, 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void sk2f(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
          int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr, int64_t ilr) {
  @@SK2F_DISPATCH@@
  TORCH_CHECK(false, "sk2f: no config");
}

void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""

_EX_GLE_STRUCT = r"""
// GLE (deferred boundary-1 normalization): the o epilogue's extra inputs / outputs, set by gle_o for its launch
struct GleEpi { const float *s, *a1, *b1, *a2; const __nv_bfloat16* pre; __nv_bfloat16* uv; float* part; };
static GleEpi g_gle = {};
"""

_EX_GOS_CUDA = r"""
__device__ __forceinline__ uint4 gos_ldw4(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 gos_ldw2(const uint8_t* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 gos_ldx(const __nv_bfloat16* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}

__global__ void __launch_bounds__(256, 2)
gos_kernel(const uint8_t* __restrict__ Ws, const int* __restrict__ esc, const __nv_bfloat16* x,
           __nv_bfloat16* __restrict__ out, uint32_t base4, int M, int pdl, const GleEpi ge) {
  constexpr int N = 8192, K = 1024, R = 32, RP = 36;
  __shared__ __align__(16) float red[8 * 8 * RP];
  __shared__ __align__(16) float4 gx_s[64], gx_a[64], gx_b[64], gx_a2[64];
  __shared__ __align__(16) uint2 gx_p[64];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* wb = Ws + ((size_t)cta * 8 + warp) * (8 * 768);
  if (pdl) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(wb + lane * 128) : "memory");
    if (lane < 16) asm volatile("prefetch.global.L2 [%0];" :: "l"(wb + (32 + lane) * 128) : "memory");
  }
  uint4 lw[2][4];
  uint2 cw[2][4];
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const uint8_t* p = wb + (s * 4 + nt) * 768;
      lw[s][nt] = gos_ldw4(p + lane * 16);
      cw[s][nt] = gos_ldw2(p + 512 + lane * 8);
    }
  const uint32_t myrec = (uint32_t)__ldg(esc + ((size_t)cta * 8 + warp) * 32 + lane);
  if (tid < 8 * M) {
    const size_t col = (size_t)cta * R + (tid & 7) * 4;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_s[tid])),
                 "l"(ge.s + (size_t)(tid >> 3) * N + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a[tid])),
                 "l"(ge.a1 + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_b[tid])),
                 "l"(ge.b1 + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a2[tid])),
                 "l"(ge.a2 + col) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_p[tid])),
                 "l"(ge.pre + col) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  uint32_t b[2][4][8];
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      decode8(lw[s][nt].x, lw[s][nt].y, cw[s][nt].x, base4, &b[s][nt][0]);
      decode8(lw[s][nt].z, lw[s][nt].w, cw[s][nt].y, base4, &b[s][nt][4]);
    }
  {
    int ei = 0;
    uint32_t cur = __shfl_sync(0xffffffffu, myrec, 0);
    while ((cur >> 30) == 0u) {
      const uint32_t rs = (cur >> 29) & 1u, rn = (cur >> 27) & 3u, rl = (cur >> 22) & 31u, rr = (cur >> 19) & 7u;
      const uint32_t msk = (uint32_t)lane == rl ? (cur & 0xFFFFu) << (((cur >> 18) & 1u) * 16u) : 0u;
#pragma unroll
      for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int r = 0; r < 8; ++r)
            b[s][nt][r] ^= ((uint32_t)s == rs && (uint32_t)nt == rn && (uint32_t)r == rr) ? msk : 0u;
      ++ei;
      cur = ei < 32 ? __shfl_sync(0xffffffffu, myrec, ei) : 0xFFFFFFFFu;
    }
  }
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // x and the outputs are ours from here
  const int xr = g < M ? g : M - 1;
  const __nv_bfloat16* xp = x + (size_t)xr * K + warp * 128 + 8 * t;
  uint4 xv[2][2];
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int p = 0; p < 2; ++p) xv[s][p] = gos_ldx(xp + s * 64 + p * 32);
  float acc[2][2][4];
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int i = 0; i < 2; ++i) acc[s][i][0] = acc[s][i][1] = acc[s][i][2] = acc[s][i][3] = 0.f;
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint4 xq = xv[s][j >> 1];
      const uint32_t b0 = (j & 1) ? xq.z : xq.x, b1 = (j & 1) ? xq.w : xq.y;
#pragma unroll
      for (int i = 0; i < 2; ++i)
        mma_bf16(acc[s][i], b[s][2 * i][2 * j], b[s][2 * i + 1][2 * j], b[s][2 * i][2 * j + 1], b[s][2 * i + 1][2 * j + 1],
                 b0, b1);
    }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");   // the pooler may start its prologue
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int tok = 2 * t + (e & 1), row = 16 * i + g + 8 * (e >> 1);
      if (tok < M) red[(warp * 8 + tok) * RP + row] = acc[0][i][e] + acc[1][i][e];
    }
  __syncthreads();
  const int tt = tid >> 3, row = (tid & 7) * 4;
  const bool on = tid < 8 * M;
  float gq[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (on) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const float4 q = *reinterpret_cast<const float4*>(red + (w * 8 + tt) * RP + row);
      v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
    }
    const size_t col = (size_t)cta * R + row;
    __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
    uint2 pk;
    pk.x = *reinterpret_cast<uint32_t*>(&lo);
    pk.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(out + (size_t)tt * N + col) = pk;
    const float pf[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
    asm volatile("cp.async.wait_all;" ::: "memory");   // this thread's own pre-wait copies
    const float4 xs = gx_s[tid], xa = gx_a[tid], xb = gx_b[tid], xa2 = gx_a2[tid];
    const uint2 xpp = gx_p[tid];
    const float sv[4] = {xs.x, xs.y, xs.z, xs.w}, av[4] = {xa.x, xa.y, xa.z, xa.w};
    const float bv[4] = {xb.x, xb.y, xb.z, xb.w}, a2v[4] = {xa2.x, xa2.y, xa2.z, xa2.w};
    const float pp[4] = {1.f + __uint_as_float(xpp.x << 16), 1.f + __uint_as_float(xpp.x & 0xffff0000u),
                         1.f + __uint_as_float(xpp.y << 16), 1.f + __uint_as_float(xpp.y & 0xffff0000u)};
    const bool tail = col >= 1024;
    float uu[4], vv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float as = av[j] * sv[j], bp = bv[j] * pf[j];
      uu[j] = as * pp[j];
      vv[j] = bp * pp[j];
      gq[0] = fmaf(pf[j], pf[j], gq[0]);
      gq[1] = fmaf(as, as, gq[1]);
      gq[2] = fmaf(as, bp, gq[2]);
      gq[3] = fmaf(bp, bp, gq[3]);
      if (tail) {
        const float x2 = a2v[j] * as, y2 = a2v[j] * bp;
        gq[4] = fmaf(x2, x2, gq[4]);
        gq[5] = fmaf(x2, y2, gq[5]);
        gq[6] = fmaf(y2, y2, gq[6]);
      }
    }
    const __nv_bfloat162 u0 = __floats2bfloat162_rn(uu[0], uu[1]), u1 = __floats2bfloat162_rn(uu[2], uu[3]);
    const __nv_bfloat162 v0 = __floats2bfloat162_rn(vv[0], vv[1]), v1 = __floats2bfloat162_rn(vv[2], vv[3]);
    uint2 uk, vk;
    uk.x = *reinterpret_cast<const uint32_t*>(&u0);
    uk.y = *reinterpret_cast<const uint32_t*>(&u1);
    vk.x = *reinterpret_cast<const uint32_t*>(&v0);
    vk.y = *reinterpret_cast<const uint32_t*>(&v1);
    *reinterpret_cast<uint2*>(ge.uv + (size_t)tt * N + col) = uk;
    *reinterpret_cast<uint2*>(ge.uv + (size_t)(M + tt) * N + col) = vk;
  }
#pragma unroll
  for (int k = 0; k < 7; ++k) {
    gq[k] += __shfl_xor_sync(0xffffffffu, gq[k], 4);
    gq[k] += __shfl_xor_sync(0xffffffffu, gq[k], 2);
    gq[k] += __shfl_xor_sync(0xffffffffu, gq[k], 1);
  }
  if (on && (tid & 7) == 0) {
    float4* pq = reinterpret_cast<float4*>(ge.part + ((size_t)cta * M + tt) * 8);
    pq[0] = make_float4(gq[0], gq[1], gq[2], gq[3]);
    pq[1] = make_float4(gq[4], gq[5], gq[6], 0.f);
  }
}

void gle_os(at::Tensor Ws, at::Tensor esc, at::Tensor x, at::Tensor out, int64_t base4, int64_t M, at::Tensor s,
            at::Tensor ab, at::Tensor pre, at::Tensor uv, at::Tensor part, int64_t pdl) {
  TORCH_CHECK(M >= 1 && M <= 8 && x.dim() == 2 && x.size(0) == M && x.size(1) == 1024 && x.scalar_type() == at::kBFloat16 &&
              out.size(0) == M && out.size(1) == 8192 && out.scalar_type() == at::kBFloat16 &&
              s.size(0) == M && s.size(1) == 8192 && s.scalar_type() == at::kFloat && ab.numel() == 3 * 8192 &&
              ab.scalar_type() == at::kFloat && pre.numel() == 8192 && pre.scalar_type() == at::kBFloat16 &&
              uv.size(0) == 2 * M && uv.size(1) == 8192 && part.numel() >= 256 * M * 8 && part.scalar_type() == at::kFloat &&
              Ws.numel() == 8192 * 1024 * 3 / 2 && esc.numel() == 256 * 8 * 32 &&
              Ws.is_contiguous() && esc.is_contiguous() && x.is_contiguous() && out.is_contiguous() && s.is_contiguous() &&
              ab.is_contiguous() && pre.is_contiguous() && uv.is_contiguous() && part.is_contiguous() &&
              (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 && (reinterpret_cast<uintptr_t>(Ws.data_ptr()) & 15) == 0,
              "gle_os: shapes");
  float* abp = ab.data_ptr<float>();
  const GleEpi ge{s.data_ptr<float>(), abp, abp + 8192, abp + 2 * 8192, reinterpret_cast<const __nv_bfloat16*>(pre.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(uv.data_ptr()), part.data_ptr<float>()};
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(256);
  lc.blockDim = dim3(256);
  lc.dynamicSmemBytes = 0;
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[1];
  at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at_[0].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = at_;
  lc.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, gos_kernel, reinterpret_cast<const uint8_t*>(Ws.data_ptr()),
                                    reinterpret_cast<const int*>(esc.data_ptr()),
                                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                    reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), (uint32_t)base4, (int)M, (int)pdl, ge));
}
"""


def _ex_gos_perm(k, dev):
    """k' -> the x column of k': lane t's B-fragment pairs of k-steps 2p, 2p + 1 of a 64-k unit are x columns
    64 u + 32 p + 8 t .. + 7 (one 16-byte vector)."""
    kp = torch.arange(k, device=dev)
    r = kp % 64
    j, hb, tq, p = r // 16, (r % 16) // 8, (r % 8) // 2, r % 2
    return (kp - r) + 32 * (j // 2) + 8 * tq + 4 * (j % 2) + 2 * hb + p



def _ex_gos_pack(w):
    """o [8192, 1024] BF16 -> (stream uint8 [cta][warp][s][tile][768], escape records int32 [2048 warps][32], base4),
    or None when the 12-bit format or the 32 records per warp do not hold. Lossless: decode + xor = the BF16 bits."""
    n, k = w.shape
    if (n, k) != (8192, 1024) or w.dtype != torch.bfloat16:
        return None
    dev = w.device
    perm = _ex_gos_perm(k, dev)
    if not bool((torch.sort(perm).values == torch.arange(k, device=dev)).all()):
        return None
    wp = w[:, perm].contiguous()
    comp = bfp_sk2_compress(wp)
    if comp is None:
        return None
    base = int(comp["base4"]) & 0xFF
    stream = comp["packed"].view(256, 4, 8, 2, 768).permute(0, 2, 3, 1, 4).contiguous().view(-1)
    rows, cols = comp["rows"].long(), comp["cols"].long()
    bits = wp.view(torch.int16)[rows, cols].to(torch.int64) & 0xFFFF
    e71, sign = (bits >> 8) & 0x7F, (bits >> 15) & 1
    dec = (((sign << 7) | (base + (e71 - base).clamp(0, 7))) << 8) | (bits & 0xFF)
    msk = (bits ^ dec) & 0xFFFF
    g, nt, c = rows % 8, (rows % 32) // 8, rows // 32
    u, rr = cols // 64, cols % 64
    jj, hb, tq, pp = rr // 16, (rr % 16) // 8, (rr % 8) // 2, rr % 2
    lane, v = g * 4 + tq, 4 * jj + 2 * hb + pp
    wid, s = c * 8 + u // 2, u % 2
    rec = (s << 29) | (nt << 27) | (lane << 22) | ((v >> 1) << 19) | ((v & 1) << 18) | msk
    tab = torch.full((2048, 32), -1, dtype=torch.int32, device=dev)
    if rec.numel():
        order = torch.argsort(wid * 2 + s, stable=True)   # per warp: unit 0's records first (the kernel walks them once)
        wid, rec = wid[order], rec[order]
        cnt = torch.bincount(wid, minlength=2048)
        if int(cnt.max()) > 32:
            return None
        start = torch.cumsum(cnt, 0) - cnt
        slot = torch.arange(wid.numel(), device=dev) - start[wid]
        tab[wid, slot] = rec.to(torch.int32)
    return stream, tab.contiguous(), int(comp["base4"])



@triton.jit
def _ex_top8_max(key, k8):
    """The 8 largest of 256 distinct u64 keys, descending: 8 x (max, drop it); the expert is 255 - the low byte."""
    ids = tl.zeros([8], dtype=tl.int32)
    for k in tl.static_range(8):
        mk = tl.max(key, 0)
        ids = tl.where(k8 == k, (255 - (mk & 255)).to(tl.int32), ids)
        key = tl.where(key == mk, 0, key)
    return ids


if hasattr(tl, "topk"):
    @triton.jit
    def _ex_top8(key, k8):
        return (255 - (tl.topk(key, 8) & 255)).to(tl.int32)
else:
    _ex_top8 = _ex_top8_max


@triton.jit
def _ex_route(POOLED, LOGITS, PART, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, RT, PO, STREAM, AB, CARRY,
               CW: tl.constexpr, FIN: tl.constexpr, PDL: tl.constexpr = True, TOPK: tl.constexpr = 2):
    """Program (t, 0): pooled = r2 (Wu + r1 Wv) -> pre-expert norm -> MoE x; (t, 1): logits = r2 (Ru + r1 Rv) -> top 8
    (as _route_split); (t, 2 + c): carry columns c * CW .. of a2 s1 (FIN) or s1. r1 / r2 from o's partial sums."""
    t = tl.program_id(0)
    kind = tl.program_id(1)
    cta = tl.arange(0, 256)
    if kind >= 2:
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        r1c = tl.rsqrt(tl.sum(tl.load(PART + (cta * T + t) * 8), 0) / 8192.0 + 1e-8)
        c = (kind - 2) * CW + tl.arange(0, CW)
        p = tl.load(PO + t * 8192 + c).to(tl.float32)
        s1 = tl.load(AB + c) * tl.load(STREAM + t * 8192 + c) + tl.load(AB + 8192 + c) * (p * r1c)
        if FIN:
            s1 = tl.load(AB + 16384 + c) * s1
        tl.store(CARRY + t * 8192 + c, s1)
    else:
        k8 = tl.arange(0, 8)
        q = tl.sum(tl.load(PART + (cta[:, None] * T + t) * 8 + k8[None, :]), 0)
        q0 = tl.sum(tl.where(k8 == 0, q, 0.0), 0)
        q1 = tl.sum(tl.where(k8 == 1, q, 0.0), 0)
        q2 = tl.sum(tl.where(k8 == 2, q, 0.0), 0)
        q3 = tl.sum(tl.where(k8 == 3, q, 0.0), 0)
        r1 = tl.rsqrt(q0 / 8192.0 + 1e-8)
        r2 = tl.rsqrt((q1 + (2.0 * r1) * q2 + (r1 * r1) * q3) / 8192.0 + 1e-5)
        if kind == 0:
            c = tl.arange(0, 4096)
            gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
                tl.extra.cuda.gdc_wait()   # the pooler's output
            pooled = (tl.load(POOLED + t * 4096 + c).to(tl.float32)
                      + tl.load(POOLED + (T + t) * 4096 + c).to(tl.float32) * r1) * r2
            x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
            tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
        else:
            e = tl.arange(0, 256)
            biasv = tl.load(BIAS + e)
            lsv = tl.load(LS)
            if FIN:   # sum over the columns >= 1024 of (a2 s1)^2, for the sliced boundary 2
                q4 = tl.sum(tl.where(k8 == 4, q, 0.0), 0)
                q5 = tl.sum(tl.where(k8 == 5, q, 0.0), 0)
                q6 = tl.sum(tl.where(k8 == 6, q, 0.0), 0)
                tl.store(RT + t, q4 + (2.0 * r1) * q5 + (r1 * r1) * q6)
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
                tl.extra.cuda.gdc_wait()   # the router's output
            logit = (tl.load(LOGITS + t * 256 + e) + tl.load(LOGITS + (T + t) * 256 + e) * r1) * r2
            sc = tl.sigmoid(logit * 0.1767766952966369)
            sel = sc + biasv
            if TOPK == 2:
                ids = tl.zeros([8], dtype=tl.int32)
                wv = tl.zeros([8], dtype=tl.float32)
                for k in tl.static_range(8):
                    idx = tl.argmax(sel, 0)
                    val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
                    ids = tl.where(k8 == k, idx, ids)
                    wv = tl.where(k8 == k, val, wv)
                    sel = tl.where(e == idx, float("-inf"), sel)
            else:
                bits = sel.to(tl.uint32, bitcast=True)
                ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
                key = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
                if TOPK == 1:
                    ids = _ex_top8(key, k8)
                else:
                    ids = _ex_top8_max(key, k8)
                wv = tl.gather(sc, ids, 0)
            wv = wv / (tl.sum(wv, 0) + 1e-15)
            tl.store(IDS + t * 8 + k8, ids)
            lid = (ids - lsv + 256) % 256
            tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
            tl.store(WTS + t * 8 + k8, wv)



@triton.jit
def _ex_finish(U, CARRY, RT, A2, GPOST, GNEXT, SOUT, HOUT, BC: tl.constexpr, PDL: tl.constexpr = True):
    """Boundary 2 at T <= 8 on the carry a2 s1: f = U (1 + w) in the first 1024 columns, zero beyond; program
    (t, tile) writes columns tile * BC .. of s2 = carry + b2 rms(f) and the next layer's input. Each program forms
    the 1024-column sums itself; the sum over the columns >= 1024 of carry^2 is the route's RT (o's partials)."""
    t = tl.program_id(0)
    tile = tl.program_id(1)
    c = tl.arange(0, 1024)
    _, b2 = _gate_ab(A2, c)
    gp = 1.0 + tl.load(GPOST + c).to(tl.float32)
    hc = tl.load(CARRY + t * 8192 + c)      # the route's carry (complete: every kernel between them triggers only
    tail = tl.load(RT + t)                  # after its own wait), weights: no producer dependency
    o = tile * BC + tl.arange(0, BC)
    lo = o < 1024
    om = tl.where(lo, o, 0)
    oc = tl.load(CARRY + t * 8192 + o)
    gn = 1.0 + tl.load(GNEXT + o).to(tl.float32)
    _, b2o = _gate_ab(A2, om)
    gpo = 1.0 + tl.load(GPOST + om).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()   # U (the unpooler slice) is the producer's output
    f = tl.load(U + t * 1024 + c).to(tl.float32) * gp
    inv = tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
    head = hc + b2 * (f * inv)
    nrm = tl.rsqrt((tail + tl.sum(head * head, 0)) / 8192.0 + 1e-5)
    fo = tl.load(U + t * 1024 + om, mask=lo, other=0.0).to(tl.float32) * gpo
    res = tl.where(lo, oc + b2o * (fo * inv), oc)
    tl.store(SOUT + t * 8192 + o, res)
    tl.store(HOUT + t * 8192 + o, (res * nrm * gn).to(tl.bfloat16))



_EX_CPP = """
void sk2xf(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,
           int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr, int64_t ilr);
void sk2x_set_pdl(int64_t n);
void gle_os(torch::Tensor Ws, torch::Tensor esc, torch::Tensor x, torch::Tensor out, int64_t base4, int64_t M,
            torch::Tensor s, torch::Tensor ab, torch::Tensor pre, torch::Tensor uv, torch::Tensor part, int64_t pdl);
"""


def _ex_ilr_nstage(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    ups = WSK * UPW
    sst = (ups * tokr * 128 + NWR * WSK * UPW * NT * 768 + NWR * WSK * 512 + 1023) // 1024 * 1024
    toks = 32 if tokr == 32 else 16
    extra = 1024 + 16 * 16 + 32 + 4 * ((toks * (R + 1) + 3) // 4 * 4) * 4 + 64 * 8 + 128
    ring = min(EX_ILR_RING_KB * 1024, 232448 - extra)
    return max(2, min(16, (bfp_SK2_K[layer] // (ups * 64)), ring // sst))


def _ex_ilr_smem(cfg, tokr, pmax):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    ns = _ex_ilr_nstage(cfg, tokr)
    sst = (WSK * UPW * tokr * 128 + NWR * WSK * UPW * NT * 768 + NWR * WSK * 512 + 1023) // 1024 * 1024
    toks = 32 if tokr == 32 else 16
    nesc = 4 if ns >= 4 else ns
    return 1024 + ns * sst + 16 * ns + 32 + nesc * ((toks * (R + 1) + 3) // 4 * 4) * 4 + pmax * 8 + 128


def _ex_ilr_stream(w):
    """FP32 router [256, 8192] -> the ILR stream of the 128-CTA fused pooler (CTA c: rows 2c, 2c + 1 as hi, hi, lo, lo
    columns of one n8 B tile, lanes 0..15), [16 stages][128 CTAs][8 units][16][16] BF16."""
    n, k = w.shape
    assert (n, k) == (256, 8192), w.shape
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    t = torch.zeros(n // 2, 8, k, dtype=torch.bfloat16, device=w.device)
    t[:, 0], t[:, 1], t[:, 2], t[:, 3] = hi[0::2], hi[1::2], lo[0::2], lo[1::2]
    tu = bfp_sk2_tileunits(t.view(n // 2 * 8, k))[:, :, :16]
    s = tu.reshape(n // 2, k // 512, 8, 16, 16).permute(1, 0, 2, 3, 4).contiguous()
    return s.view(torch.uint8).reshape(-1)


def _ex_src():
    """EROREx's sk2 source (with the in-loop router), instantiated only for the fused pooler + router at tokr 8 / 16,
    plus gos_kernel; every symbol renamed (sk2x_*) so nothing clashes with this file's bf16_proj builds."""
    cfg = next(c for c in bfp_SK2_CONFIGS if c[0] == "pooler")
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    fl = []
    for tokr in (8, 16):
        ns = _ex_ilr_nstage(cfg, tokr)
        fl.append(f"if (tokr == {tokr} && ilr == 1) {{ launch_sk2<1, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {ns}, true, false, "
                  f"{NACC}, 1, __nv_bfloat16, true, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                  f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                  f"rcnt.data_ptr<int>()); return; }}")
    s = _EX_SK2_SRC
    for a, b in (("#define SK2_DIAG 0\n", "#define SK2_DIAG 0\n" + _EX_GLE_STRUCT),
                 ("@@SK2_DISPATCH@@", ""), ("@@SK2F_DISPATCH@@", "\n  ".join(fl))):
        if s.count(a) != 1:
            raise RuntimeError(f"ex source: {s.count(a)} matches of {a[:60]!r}")
        s = s.replace(a, b)
    s += _EX_GOS_CUDA
    for a, b in (("sk2_kernel", "sk2x_kernel"), ("launch_sk2", "launch_sk2x"), ("g_sk2_pdl", "g_sk2x_pdl"),
                 ("void sk2_set_pdl(", "void sk2x_set_pdl("), ("void sk2f(", "void sk2xf("), ("void sk2(", "void sk2x(")):
        s = s.replace(a, b)
    common = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", "")
    return common + bfp_MAP_SRC + s


def _ex_build():
    global _EX_EXT, _EX_ERR
    try:
        import hashlib as _ex_hl
        major, minor = torch.cuda.get_device_capability()
        if major < 9:
            raise RuntimeError("ex: sm_90 only")
        arch = f"{major}{minor}a"
        src = _ex_src()
        _EX_EXT = load_inline(
            name=f"dl_ex_gle_{_ex_hl.md5(src.encode()).hexdigest()[:12]}_{arch}",
            cpp_sources="#include <torch/extension.h>\n" + _EX_CPP,
            cuda_sources=src,
            functions=["sk2xf", "sk2x_set_pdl", "gle_os"],
            extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            no_implicit_headers=True,
            verbose=False,
        )
    except Exception as ex:  # noqa: BLE001  (EX stays off: this file's chain at every T)
        _EX_EXT = None
        _EX_ERR = repr(ex)[-400:]


def _ex_pooler_router(x, st, T):
    """pooler [m, 4096] bf16 + router [m, 256] fp32 of the m = 2T GLE rows in one launch (in-loop router)."""
    bst = st["bfp"]
    m = x.shape[0]
    ci, stream, tab, pmax, b4, n_rows = bst["sk2"]["pooler"]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    _EX_EXT.sk2x_set_pdl(EX_PDL_N if _pdl(T) else 0)
    _EX_EXT.sk2xf(stream, st["ex_ilr"], x, out, outr, tab, pmax, b4, st["ex_part"], st["ex_cnt"], m, bfp__sk2_tokr(m), 1)
    return out, outr


def _ex_fin_cfg(T):
    for tmax, nw, bc in EX_FIN_CFG:
        if T <= tmax:
            return nw, bc
    return EX_FIN_CFG[-1][1:]


def _ex_after_attention(data, st, o_in, T):
    dev = o_in.device
    f32 = dict(device=dev, dtype=torch.float32)
    gos = st["ex_gos"]
    P = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    uv = torch.empty((2 * T, HIDDEN), device=dev, dtype=torch.bfloat16)
    part = torch.empty((256 * T * 8,), **f32)
    _EX_EXT.gle_os(gos[0], gos[1], o_in, P, gos[2], T, data["stream"], st["ex_ab"], st["ex_pre"], uv, part,
                   1 if _pdl(T) else 0)
    pooled, logits = _ex_pooler_router(uv, st, T)
    ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((T, 8), **f32)
    xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    rt = torch.empty((T,), **f32)
    carry = torch.empty((T, HIDDEN), **f32)
    _ex_route[(T, 2 + EX_NCH)](pooled, logits, part, T, st["bias"], st["g_preexp"], ids, wts, xm,
                               data["local_start"], lids, rt, P, data["stream"], st["ex_ab"], carry,
                               CW=HIDDEN // EX_NCH, FIN=True, TOPK=EX_TOPK, num_warps=EX_ROUTE_WARPS, launch_pdl=True)
    m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    mx = _moe(T, st)
    if MOE_HAS_PDL:
        mx.okc_set_pdl(1 if _pdl(T) else 0)
    mx.okc_run(st["ma_w"], st["ma_scratch"], xm, lids, wts, m, 0, 32, st["nsm"])
    un, uk = st["unpool_slice"].shape
    U = _bfp_gemm(st, "un", m, T, un, uk)
    if un != 1024 or U.dtype != torch.bfloat16 or U.numel() != T * 1024:
        raise RuntimeError("ex: BF16 sliced unpooler expected")
    s2 = torch.empty((T, HIDDEN), **f32)
    out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
    nw, bc = _ex_fin_cfg(T)
    _ex_finish[(T, HIDDEN // bc)](U.view(T, 1024), carry, rt, st["gate2"], st["g_postffn"], st["g_next"], s2, out,
                                  BC=bc, num_warps=nw, launch_pdl=True)
    return out, s2, ids


def _ex_gos_ok(st, w, pk, T):
    """gos at T against an FP32 matmul (P) and a torch evaluation of its epilogue (u / v rows, partial sums)."""
    dev = w.device
    ab, pre = st["ex_ab"], st["ex_pre"]
    wf = w.float()
    worst = 0.0
    for seed in range(2):
        g = torch.Generator(device=dev).manual_seed(4600 + 31 * T + seed)
        x = (0.1 * torch.randn(T, 1024, device=dev, generator=g)).to(torch.bfloat16)
        if seed == 1:   # a spike per row: every weight column's escapes weigh in
            x[:, torch.randint(1024, (T,), device=dev, generator=g)] = 4.0
        s = torch.randn(T, HIDDEN, device=dev, generator=g)
        P = torch.full((T, HIDDEN), float("nan"), device=dev, dtype=torch.bfloat16)
        uv = torch.full((2 * T, HIDDEN), float("nan"), device=dev, dtype=torch.bfloat16)
        part = torch.full((256 * T * 8,), float("nan"), device=dev, dtype=torch.float32)
        _EX_EXT.gle_os(pk[0], pk[1], x, P, pk[2], T, s, ab, pre, uv, part, 0)
        torch.cuda.synchronize()
        ref = x.float() @ wf.t()
        Pf = P.float()
        e_ref = float(((Pf - ref).abs() / ref.abs().amax(1, keepdim=True).clamp_min(1e-20)).max())
        pp = 1.0 + pre.float()
        a_s, bp = ab[0] * s, ab[1] * Pf
        uv_ref = torch.cat([(a_s * pp).to(torch.bfloat16), (bp * pp).to(torch.bfloat16)]).float()
        e_uv = float(((uv.float() - uv_ref).abs() / uv_ref.abs().amax(1, keepdim=True).clamp_min(1e-20)).max())
        tail = (torch.arange(HIDDEN, device=dev) >= 1024).float()
        x2, y2 = ab[2] * a_s * tail, ab[2] * bp * tail
        qr = torch.stack([Pf * Pf, a_s * a_s, a_s * bp, bp * bp, x2 * x2, x2 * y2, y2 * y2], -1)   # [T, 8192, 7]
        qr = qr.view(T, 256, 32, 7).sum(2).permute(1, 0, 2)                                         # [256, T, 7]
        q1 = part.view(256, T, 8)[:, :, :7].float()
        e_q = float(((q1 - qr).abs().sum(0) / qr.abs().sum(0).clamp_min(1e-20)).max())
        fin = bool(torch.isfinite(Pf).all() and torch.isfinite(uv.float()).all() and torch.isfinite(q1).all())
        worst = max(worst, e_ref, e_uv, e_q)
        if not (fin and e_ref < 8e-3 and e_uv < 8e-3 and e_q < 4e-3):
            return False, worst
    return True, worst


def _ex_ilr_ok(weights, st):
    """The ILR launch at m = 2T rows: router vs FP64 (rel <= 2e-5), pooled vs FP32 (<= 8e-3 of the row max)."""
    bst = st["bfp"]
    cfg = next(c for c in bfp_SK2_CONFIGS if c[0] == "pooler")
    pm = int(bst["sk2"]["pooler"][3])
    optin = int(torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin)
    for tokr in (8, 16):
        if _ex_ilr_smem(cfg, tokr, pm) > optin - 256:
            return False, f"smem tokr {tokr} pmax {pm}"
    wr = weights["router"].double()
    wp = weights["pooler"].float()
    dev = wp.device
    g = torch.Generator(device=dev).manual_seed(4242)
    worst = 0.0
    for m in (4, 6, 8, 10, 12, 14, 16):
        x = torch.randn(m, HIDDEN, device=dev, generator=g).to(torch.bfloat16)
        pooled, logits = _ex_pooler_router(x, st, 0)
        torch.cuda.synchronize()
        ref = x.double() @ wr.T
        er = float((logits.double() - ref).norm() / ref.norm())
        pref = x.float() @ wp.t()
        ep = float(((pooled.float() - pref).abs() / pref.abs().amax(1, keepdim=True).clamp_min(1e-20)).max())
        worst = max(worst, er)
        if not (er < 2e-5 and ep < 8e-3 and bool(torch.isfinite(pooled.float()).all())):
            return False, f"m={m} router {er:.1e} pooled {ep:.1e}"
    return True, f"router rel <= {worst:.1e}"


def _ex_setup(weights, st):
    st["ex_on"] = set()
    if _EX_EXT is None:
        print(f"[ex] extension not built: {_EX_ERR}", flush=True)
        return
    try:
        bst = st.get("bfp")
        if bst is None or not bst.get("sm90") or "pooler" not in bst.get("sk2", {}):
            print("[ex] bf16_proj sk2 pooler unavailable: off", flush=True)
            return

        def gab(gw):
            w = gw.float() / 0.3
            beta = torch.sigmoid(w)
            return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)), beta
        a1, b1 = gab(st["gate1"])
        a2, _ = gab(st["gate2"])
        st["ex_ab"] = torch.stack([a1, b1, a2]).contiguous()
        st["ex_pre"] = st["g_preffn"].to(torch.bfloat16).contiguous().clone()
        dev = st["ex_ab"].device
        st["ex_part"] = torch.zeros(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
        st["ex_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
        st["ex_ilr"] = _ex_ilr_stream(weights["router"].float().contiguous())
        ok, msg = _ex_ilr_ok(weights, st)
        print(f"[ex] ILR pooler + router: {'ok' if ok else 'FAIL'} ({msg})", flush=True)
        if not ok:
            return
        w = weights["o"].contiguous()
        pk = _ex_gos_pack(w)
        if pk is None:
            print("[ex] o weights not packable: off", flush=True)
            return
        st["ex_gos"] = pk
        gos_ok = {}
        for T in EX_T:
            gos_ok[T] = _ex_gos_ok(st, w, pk, T)
        print("[ex] gos per T (ok / worst rel): " + ", ".join(f"{T}:{int(o)}/{e:.1e}" for T, (o, e) in gos_ok.items()),
              flush=True)
        for T in EX_T:
            if not gos_ok[T][0]:
                continue
            try:
                ok, msg = _ex_chain_ok(weights, st, T)
                if ok and T in EX_TIME_T:
                    tb, te = _ex_times(st, T)
                    ok = te <= tb * EX_SLOW
                    msg += f"; base {tb:.1f} us / EX {te:.1f} us"
                elif ok:
                    tb, te = _ex_times(st, T)
                    msg += f"; base {tb:.1f} us / EX {te:.1f} us (log only)"
                if ok:
                    st["ex_on"].add(T)
                print(f"[ex] T={T}: {'EX chain' if ok else 'base chain'} ({msg})", flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"[ex] T={T}: self-test failed, base chain: {repr(ex)[:300]}", flush=True)
    except Exception as ex:  # noqa: BLE001
        st["ex_on"] = set()
        print("[ex] setup failed, base chain:", repr(ex)[:300], flush=True)
    torch.cuda.synchronize()


def _ex_front_ref(weights, stream, o_in):
    """FP64 router scores of the reference's front (o -> boundary 1 -> pre-FFN norm -> router + bias), [T, 256]."""
    a = o_in.double() @ weights["o"].double().t()
    w = weights["attn_residual_gate"].double() / 0.3
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    s1 = alpha * stream.double() + beta * a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + 1e-8)
    h = s1 * torch.rsqrt(s1.pow(2).mean(-1, keepdim=True) + 1e-5) * (1.0 + weights["pre_ffn_norm"].double())
    return torch.sigmoid((h @ weights["router"].double().t()) * ROUTER_MULT) + weights["router_bias"].double()


def _ex_row_err(g, r):
    d = (g - r).norm(dim=-1)
    n = r.norm(dim=-1)
    return d / torch.maximum(n, (0.1 * n.mean()).clamp_min(1e-30))


def _ex_chain_ok(weights, st, T):
    """EX chain vs this file's chain at T, per token, over EX_SEEDS seeds (see EX_HEAD)."""
    dev = st["ex_ab"].device
    a12 = st["ex_ab"][0] * st["ex_ab"][2]
    agree = total = 0
    worst_flip = worst_row = worst_agg = 0.0
    for seed in range(EX_SEEDS):
        g = torch.Generator(device=dev).manual_seed(5150 + 97 * T + seed)
        data = {"stream": torch.randn(T, HIDDEN, device=dev, generator=g),
                "local_start": torch.zeros(1, device=dev, dtype=torch.int32)}
        o_in = (0.1 * torch.randn(T, 1024, device=dev, generator=g)).to(torch.bfloat16)
        r_out, r_s2, r_ids = _ex_prior_after(data, st, o_in, None, T)
        g_out, g_s2, g_ids = _ex_after_attention(data, st, o_in, T)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(g_out.float()).all() and torch.isfinite(g_s2).all()):
            return False, f"seed {seed}: non-finite output"
        gi = g_ids.long()
        if int(gi.min()) < 0 or int(gi.max()) >= 256 or bool((gi.sort(-1).values[:, 1:] == gi.sort(-1).values[:, :-1]).any()):
            return False, f"seed {seed}: invalid expert ids"
        b = _ex_front_ref(weights, data["stream"], o_in)
        top = b.topk(9, dim=-1)
        s8, s9 = top.values[:, 7:8], top.values[:, 8:9]
        pick_gap = float((s8 - b.gather(1, gi)).max())
        kept = (top.indices[:, :8, None] == gi[:, None, :]).any(-1)
        drop_gap = float(torch.where(kept, torch.full((T, 8), -1.0, dtype=b.dtype, device=b.device), top.values[:, :8] - s9).max())
        worst_flip = max(worst_flip, pick_gap, drop_gap)
        if pick_gap > EX_FLIP_TOL or drop_gap > EX_FLIP_TOL:
            return False, f"seed {seed}: expert margin {max(pick_gap, drop_gap):.1e} > {EX_FLIP_TOL:g}"
        same = (gi.sort(-1).values == r_ids.long().sort(-1).values).all(-1)
        ns = int(same.sum())
        agree += ns
        total += T
        if ns == 0:
            continue
        upd_r = (r_s2 - a12 * data["stream"])[same]
        upd_g = (g_s2 - a12 * data["stream"])[same]
        hr, hg = r_out.float()[same], g_out.float()[same]
        row = max(float(_ex_row_err(hg, hr).max()), float(_ex_row_err(upd_g, upd_r).max()))
        agg = max(float((hg - hr).norm() / hr.norm().clamp_min(1e-30)),
                  float((upd_g - upd_r).norm() / upd_r.norm().clamp_min(1e-30)))
        worst_row, worst_agg = max(worst_row, row), max(worst_agg, agg)
        if row > EX_ROW_TOL or agg > EX_AGG_TOL:
            return False, f"seed {seed}: rows {row:.1e} / aggregate {agg:.1e} vs base chain"
    msg = (f"{agree}/{total} tokens routed as base, rows <= {worst_row:.1e}, aggregate <= {worst_agg:.1e}, "
           f"expert margin <= {worst_flip:.1e}")
    return agree >= EX_AGREE_MIN * total, msg


def _ex_times(st, T):
    """Median us of this file's chain and the EX chain after attention at T (paired CUDA-graph replays, cold L2)."""
    dev = st["ex_ab"].device
    g = torch.Generator(device=dev).manual_seed(7070 + T)
    data = {"stream": torch.randn(T, HIDDEN, device=dev, generator=g),
            "local_start": torch.zeros(1, device=dev, dtype=torch.int32)}
    o_in = (0.1 * torch.randn(T, 1024, device=dev, generator=g)).to(torch.bfloat16)
    ids = _ex_prior_after(data, st, o_in, None, T)[2]
    counts = torch.bincount(ids.flatten().long() // 32, minlength=8)
    data["local_start"].fill_(32 * int(counts.argmax()))   # the busiest EP range, as keell's chain policy
    ts = _compose_times([lambda: _ex_prior_after(data, st, o_in, None, T),
                         lambda: _ex_after_attention(data, st, o_in, T)], reps=32)
    torch.cuda.synchronize()
    return float(ts[0]), float(ts[1])


_ex_prior_after = _after_attention


def _after_attention(data, state, o_in, gate, T):
    if (gate is None and T in state.get("ex_on", ()) and _EX_EXT is not None and o_in.is_contiguous()
            and o_in.data_ptr() % 16 == 0):
        return _ex_after_attention(data, state, o_in, T)
    return _ex_prior_after(data, state, o_in, gate, T)


_ex_prior_prepare = prepare


def prepare(weights):
    th = threading.Thread(target=_ex_build, daemon=True)   # nvcc for the EX extension while the layer prepares
    th.start()
    st = _ex_prior_prepare(weights)
    th.join(timeout=150.0)
    if th.is_alive():
        print("[ex] extension build still running: off", flush=True)
        st["ex_on"] = set()
        return st
    _ex_setup(weights, st)
    return st


# h51x (EROREx, 2026-11-01): asserted string patches of keell's T >= 9 MoE sources (KEELL/muse_decode_layer/moe/h5lib.py)
def _h5_sub(src, old, new, tag, count=1):
    n = src.count(old)
    if n != count:
        raise RuntimeError(f"h5 patch {tag}: {n} matches (want {count}) for {old[:80]!r}")
    return src.replace(old, new)


_H5_YLD = "          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);\n"


def _h5_fin1(c):
    """WGSFINSS final (T >= 33): each route's bf16 y row is read once (kept in registers for the second pass)."""
    a = ("      float ps[8];\n#pragma unroll\n      for (int sl = 0; sl < 8; ++sl) {\n        ps[sl] = 0.f;\n"
         "        if (qv[sl] >= 0) {\n#if OKC_YBF\n" + _H5_YLD)
    c = _h5_sub(c, a, a.replace("      float ps[8];\n", "      uint4 yk[8];\n      float ps[8];\n") + "          yk[sl] = yr;\n", "fin1.a")
    b = "rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);\n#if OKC_YBF\n" + _H5_YLD
    c = _h5_sub(c, b, b.replace(_H5_YLD, "          const uint4 yr = yk[sl];\n"), "fin1.b")
    return c


_H5_HSPH = r"""#if OKC_HSP
// h5 HSP (EROREx, 2026-11-01: our gen_v070 HSP ported to keell's MID helper): four helper warps (physical warps 0..3
// atomic per job, token and helper warp). SiLU exactly as keell's helper (__expf); DF builds credit nothing (their
static_assert(OKC_WSB && !OKC_WSRF && RED_W == 4, "HSP: thread 0 refills the ring, 4-float partials");
static_assert((1 << RB_B_LOG) == RB_B, "HSP: phase-B row blocks by shifts");
__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31;
  const int hw = (TIDX - NT) >> 5;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (hw == 0 && lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int Nh = OKC_WSBL ? N - min(OKC_WSBLK, b1 - b0) : N;
  const int cl = 8 * hw + (lane >> 2), k = lane & 3;
  const int gg = cl >> 2, tt = cl & 3, h = k & 1, hi = k >> 1, tk = 2 * tt + h;
  float ssq = 0.f;
  int cj = -1, cnt = 0, st = 0;
  float gsc = 0.f;
  auto flush = [&]() {  // phase B: the job's sums of squares (rows gg, gg + 8 and the warp's two gg), one atomic per token
    float s = ssq + __shfl_xor_sync(0xffffffffu, ssq, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 16);
    if ((lane & 0x12) == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], s);
    ssq = 0.f;
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
    float s = S.red[bb][0][cl][k];
#pragma unroll
    for (int w = 1; w < NW; ++w) s += S.red[bb][w][cl][k];
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);  // this warp's quarter is read (the buffer is free after all four)
    if (!PB) {
      const float up = __shfl_down_sync(0xffffffffu, s, 2);
      if (hi == 0 && tk < cnt) {
        const float gt = s * gsc, u = up * gsc;
        st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * u)));
      }
      if (!OKC_DF && q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
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
    } else {
      const int n0 = rb * 16 + gg + 8 * hi;
      const float y = s * gsc * (1.f + __bfloat162float(p.normw[n0]));
      ssq += y * y;
      if (tk < cnt) st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), y);
    }
  }
  if (Nh > nA) flush();
  __threadfence();
  WS_JOIN();
}
#else
"""


def _h5_hsp(c):
    """MID (keell's WS helper build): HSP, four helper warps share each unit's epilogue (build with -DOKC_HSP=1)."""
    c = _h5_sub(c, "mbar_init(S.wsre + k, 1); }", "mbar_init(S.wsre + k, WS_NHW); }", "hsp.wsre", 2)
    c = _h5_sub(c, "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);",
                "#ifndef OKC_HSP\n#define OKC_HSP 0  // h5 HSP: four helper warps (one per SMSP) share every unit's epilogue\n"
                "#endif\n#define WS_NHW (OKC_HSP ? 4 : 1)  // helper warps (WS)\n"
                "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 * WS_NHW : 0);", "hsp.ntl")
    a = "__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n"
    c = _h5_sub(c, a, _H5_HSPH + a, "hsp.helper")
    return _h5_sub(c, "  __threadfence();\n  WS_JOIN();\n}\n#endif\n", "  __threadfence();\n  WS_JOIN();\n}\n#endif  // OKC_HSP\n#endif\n",
                   "hsp.tail")




# its source; KEELL/muse_decode_layer/prolog/p6lib.py).
MOE_CUDA_SRC = _h5_fin1(MOE_CUDA_SRC)
_MA_ALT_SRCS["moe_helper0df"] = _h5_hsp(_MA_ALT_SRCS["moe_helper0df"])
MOE_MID_DEFINES = tuple(MOE_MID_DEFINES) + ("-DOKC_HSP=1",)


class _H5Pdl1:   # launches always with PDL (the kernel reads no input before griddepcontrol.wait)
    def __init__(self, real):
        self.real = real

    def okc_set_pdl(self, on):
        return self.real.okc_set_pdl(1)

    def __getattr__(self, k):
        return getattr(self.real, k)

    def __deepcopy__(self, memo):
        return self


_p6_base_moe = _moe


def _moe(M=None, state=None):
    ext = _p6_base_moe(M, state)
    if state is not None and M is not None and MOE_HAS_PDL and not _pdl(M):   # T > PDL_MAX_T: the main WGS build
        return _H5Pdl1(ext)
    return ext



# safe (EROREx, 2026-11-02; KEELL/muse_decode_layer/safe/gen_safe.py): the in-layer MoE's cross-CTA hand-offs made
def _sf_sub(c, old, new, tag, count):
    n = c.count(old)
    if n != count:
        raise RuntimeError(f"safe patch {tag}: {n} matches (want {count})")
    return c.replace(old, new) if count else c


_SF_DF = 0   # MID data-flag A->B hand-off: 0 = the doneA counters (safe default), 1 = keell's DF (trap on timeout)
_SF_HSP8 = True   # TINY2 (T 5..8): four helper warps (HSP2)

# --- HSP2 for the TINY2 build (a copy of moe_helper0, so the TINY build keeps keell's helper) --------------------------
_SF_HSP2 = r"""#if OKC_HSP >= 2
// safe HSP2 (EROREx, 2026-11-02): port of EROREx's own gen_v070 hsp2 helper (BABAGAZ/muse_moe_w4a16) to keell's
// keell's helper (__expf). Phase A credits doneA after every helper warp's stores are fenced (bar 3, 128 threads).
static_assert(OKC_WSB && !OKC_WSRF && RED_W == 4, "HSP2: thread 0 refills the ring, 4-float partials");
static_assert((1 << RB_A_LOG) == RB_A && (1 << RB_B_LOG) == RB_B && (NRB & (NRB - 1)) == 0, "HSP2: shifts and masks");
__device__ __forceinline__ void hsp2_wait(uint32_t a, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(a), "r"(parity) : "memory");
}
__device__ __forceinline__ void hsp2_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(a) : "memory");
}
__device__ __forceinline__ float hsp2_lds(uint32_t a) {
  float v;
  asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(v) : "r"(a));
  return v;
}
#define HSP2_SUM(q)                                                                       \
  const uint32_t bb = (uint32_t)(q) & (NRB - 1);                                          \
  hsp2_wait(rf0 + bb * 8u, ((uint32_t)(q) / NRB) & 1u);                                   \
  const uint32_t ra = rd0 + bb * RBUF;                                                    \
  float s = hsp2_lds(ra);                                                                 \
  _Pragma("unroll") for (int w = 1; w < NW; ++w) s += hsp2_lds(ra + (uint32_t)w * RWARP); \
  __syncwarp();                                                                           \
  if (lane == 0) hsp2_arrive(re0 + bb * 8u);

__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31, hw = (TIDX - NT) >> 5;
  WS_JOIN();  // routing tables and this CTA's ranges are ready
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (hw == 0 && lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int Nh = OKC_WSBL ? N - min(OKC_WSBLK, b1 - b0) : N;
  const int cl = 8 * hw + (lane >> 2), k = lane & 3;
  const int gg = cl >> 2, tt = cl & 3, h = k & 1, hi = k >> 1, tk = 2 * tt + h;
  constexpr uint32_t RWARP = 32u * RED_W * 4u, RBUF = (uint32_t)NW * RWARP;
  const uint32_t rf0 = smem_u32(S.wsrf), re0 = smem_u32(S.wsre), rd0 = smem_u32(&S.red[0][0][cl][k]);
  if (nA > 0) {
    int j = njF + (a0 >> RB_A_LOG), rb = a0 & (RB_A - 1);
    int cnt = S.jC[j], st = S.jS[j];
    float gsc = p.gsA[S.jE[j]] * TWO62;
    for (int q = 0; q < nA; ++q) {
      HSP2_SUM(q)
      const float up = __shfl_down_sync(0xffffffffu, s, 2);
      if (hi == 0 && tk < cnt) {
        const float gt = s * gsc, u = up * gsc;
        st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * u)));
      }
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
  if (Nh > nA) {
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
  WS_JOIN();
}
#undef HSP2_SUM
#else
"""


def _sf_hsp2(c):
    c = _sf_sub(c, "mbar_init(S.wsre + k, 1); }", "mbar_init(S.wsre + k, WS_NHW); }", "hsp2.wsre", 2)
    c = _sf_sub(c, "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);",
                "#ifndef OKC_HSP\n#define OKC_HSP 0  // safe HSP2 (>= 2): four helper warps (one per SMSP) share every unit's epilogue\n"
                "#endif\n#define WS_NHW (OKC_HSP ? 4 : 1)  // helper warps (WS)\n"
                "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 * WS_NHW : 0);", "hsp2.ntl", 1)
    a = "__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {\n"
    c = _sf_sub(c, a, _SF_HSP2 + a, "hsp2.helper", 1)
    return _sf_sub(c, "  __threadfence();\n  WS_JOIN();\n}\n#endif\n",
                   "  __threadfence();\n  WS_JOIN();\n}\n#endif  // OKC_HSP >= 2\n#endif\n", "hsp2.tail", 1)


if _SF_HSP8:
    _MA_ALT_SRCS["moe_helper0h2"] = _sf_hsp2(_MA_ALT_SRCS["moe_helper0"])
    assert MOE_TINY2_DEFINES.count("@moe_helper0") == 1, MOE_TINY2_DEFINES
    MOE_TINY2_DEFINES = tuple("@moe_helper0h2" if d == "@moe_helper0" else d for d in MOE_TINY2_DEFINES) + ("-DOKC_HSP=2",)

_SF_ACQ_OLD = "if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }"
_SF_ACQ_NEW = "{ while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }  // safe: every lane acquires (each reads other CTAs' inter rows)"
_SF_MFIN_OLD = "for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }"
_SF_MFIN_NEW = "while (ld_acquire(cp) < target) { if (OKC_MFS) __nanosleep(OKC_MFS); }  // safe: no bound, never proceeds early"
_SF_PXY_OLD = "      uint64_t* bar = S.bars + BX_F + xs;\n      const uint32_t bytes = (uint32_t)npad * 512u;\n"
_SF_PXY_NEW = ("      asm volatile(\"fence.proxy.async.global;\\n\" ::: \"memory\");  // safe: other CTAs' generic stores seen by the"
               " acquires above -> this bulk copy (async proxy)\n" + _SF_PXY_OLD)
_SF_OCC_OLD = "  cudaStream_t st = at::cuda::getCurrentCUDAStream();\n"
_SF_OCC_NEW = r"""  {  // safe: the persistent kernel spins on other CTAs' flags, so every CTA of the grid must be resident at once;
    static int okc_cap[2] = {-1, -1};
    const int bi = p.M >= OKC_BIGM ? 1 : 0;
    if (okc_cap[bi] < 0) {
      int dev = 0, nsm = 0, nb = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
      if (bi) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, okc::moe_fused<true>, okc::NTL, sm);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, okc::moe_fused<false>, okc::NTL, sm);
      okc_cap[bi] = nb * nsm;
    }
    TORCH_CHECK(grid >= 1 && grid <= okc_cap[bi], "okc_run: grid ", grid, " CTAs cannot all be co-resident (", okc_cap[bi], ")");
  }
""" + _SF_OCC_OLD
_SF_DF_OLD = "#define OKC_DF OKC_PRMTSC   // data-flag A->B hand-off (needs the PRMT scale decode builds)\n"
_SF_DF_NEW = "#define OKC_DF 0   // safe: the doneA-counter A->B hand-off (keell's data-flag DF polls rows still being written)\n"
_SF_DFT_OLD = "    if (!__any_sync(0xffffffffu, bad != 0) || spin > 200000) break;\n"
_SF_DFT_NEW = ("    if (!__any_sync(0xffffffffu, bad != 0)) break;\n"
               "    if (spin > 200000) __trap();  // safe: never go on with unwritten rows\n")

_SF_PATCHES = [
    (_SF_ACQ_OLD, _SF_ACQ_NEW, "acq", {"main": 3, "h0": 3, "h0df": 3, "h2": 3}),
    (_SF_MFIN_OLD, _SF_MFIN_NEW, "mfin", {"main": 2, "h0": 0, "h0df": 0, "h2": 0}),
    (_SF_PXY_OLD, _SF_PXY_NEW, "pxy", {"main": 1, "h0": 1, "h0df": 1, "h2": 1}),
    (_SF_OCC_OLD, _SF_OCC_NEW, "occ", {"main": 1, "h0": 1, "h0df": 1, "h2": 1}),
    (_SF_DFT_OLD, _SF_DFT_NEW, "dftrap", {"main": 0, "h0": 0, "h0df": 1, "h2": 0}),
]
if not _SF_DF:
    _SF_PATCHES.append((_SF_DF_OLD, _SF_DF_NEW, "df", {"main": 0, "h0": 0, "h0df": 1, "h2": 0}))


def _sf_safe(c, key):
    for old, new, tag, cnt in _SF_PATCHES:
        c = _sf_sub(c, old, new, f"{tag}/{key}", cnt[key])
    return c


MOE_CUDA_SRC = _sf_safe(MOE_CUDA_SRC, "main")
for _sf_k, _sf_key in (("moe_helper0", "h0"), ("moe_helper0df", "h0df"), ("moe_helper0h2", "h2")):
    if _sf_k in _MA_ALT_SRCS:
        _MA_ALT_SRCS[_sf_k] = _sf_safe(_MA_ALT_SRCS[_sf_k], _sf_key)
assert set(_MA_ALT_SRCS) <= {"moe_helper0", "moe_helper0df", "moe_helper0h2"}, sorted(_MA_ALT_SRCS)



# dn (EROREx, 2026-11-02; KEELL/muse_decode_layer/dense/gen_dn.py): the dense projections offered the kernels that the
# bf16_proj board file (KEELL/muse_bf16_proj/h110a_kclean.py) runs on the same shapes. Each one is validated and raced
DN_CG = True     # T = 1 qkv / unpooler slice: cgemv (bf16_proj; vjeux/okc-2026 lineage via team keell0renz, Apache-2.0)
DN_OL = True     # o at T = 128 / 256: ol_projection (NathenFernandes, MIT, notice below) / bf16_proj's ag cuBLASLt pin
DN_PR = True     # T = 128: rq pooler (chenwainuo, Apache-2.0) + a separate FP32-out cuBLASLt router over [hi; lo]
DN_GAIN = 0.97     # isolated pre-filter: a candidate must time under 97 % of the incumbent
DN_CHAIN_US = 0.3  # in-chain: the post-attention chain must get at least this many us faster (median of 32 paired replays)
DN_EXT = [None, "not built"]


def _dn_say(s):
    print("[dn] " + s, flush=True)


def _dn_chain_data(st, T, seed):
    """synthetic post-attention inputs at T; local_start = the busiest EP range of their routing (as the compose race)"""
    dev = st["qkv"].device
    g = torch.Generator(device=dev).manual_seed(seed)
    data = {"stream": torch.randn(T, HIDDEN, device=dev, generator=g),
            "local_start": torch.zeros(1, device=dev, dtype=torch.int32)}
    o_in = (0.1 * torch.randn(T, 1024, device=dev, generator=g)).to(torch.bfloat16)
    ids = _after_attention(data, st, o_in, None, T)[2]
    counts = torch.bincount(ids.flatten().long() // 32, minlength=8)
    data["local_start"].fill_(32 * int(counts.argmax()))
    return data, o_in


def _dn_chain_ok(st, ss, data, o_in, T):
    """variant ss vs the incumbent st on one input: finite, >= 90 % of tokens routed alike, and those tokens' output rows
    within 2e-2 (aggregate 1e-2) of the incumbent's"""
    a = _after_attention(data, st, o_in, None, T)
    b = _after_attention(data, ss, o_in, None, T)
    torch.cuda.synchronize()
    if not (bool(torch.isfinite(b[0].float()).all()) and bool(torch.isfinite(b[1]).all())):
        return False, "non-finite"
    same = (a[2].long().sort(-1).values == b[2].long().sort(-1).values).all(-1)
    ns = int(same.sum())
    if ns < 0.9 * T:
        return False, f"{ns}/{T} routed alike"
    ha, hb = a[0].float()[same], b[0].float()[same]
    row = float(((hb - ha).norm(dim=-1) / ha.norm(dim=-1).clamp_min(1e-30)).max())
    agg = float((hb - ha).norm() / ha.norm().clamp_min(1e-30))
    return row < 2e-2 and agg < 1e-2, f"{ns}/{T} alike, rows {row:.1e}, agg {agg:.1e}"


def _dn_race(st, T, variants, seed):
    """variants = [(tag, state)], the incumbent first: the in-chain race; returns the winning (tag, state)"""
    data, o_in = _dn_chain_data(st, T, seed)
    live = [variants[0]]
    for tag, ss in variants[1:]:
        try:
            ok, msg = _dn_chain_ok(st, ss, data, o_in, T)
        except Exception as ex:  # noqa: BLE001
            ok, msg = False, repr(ex)[:160]
        _dn_say(f"T={T} chain check {tag}: {'ok' if ok else 'REJECT'} ({msg})")
        if ok:
            live.append((tag, ss))
    if len(live) < 2:
        return live[0]
    ts = _compose_times([lambda ss=ss: _after_attention(data, ss, o_in, None, T) for _, ss in live], reps=32)
    best = min(range(len(ts)), key=ts.__getitem__)
    pick = live[best] if (best and ts[best] <= ts[0] - DN_CHAIN_US) else live[0]
    _dn_say(f"T={T} chain race " + " ".join(f"{t}:{x:.2f}" for (t, _), x in zip(live, ts)) + f" -> {pick[0]}")
    return pick


def _dn_lin_ok(fn, w, m, seed, tol=6e-3, ptol=3e-3):
    """fn(x) [m, n] bf16 vs x @ w^T in FP32: normal and outlier inputs, two one-hot probe rows (m >= 2)"""
    dev = w.device
    g = torch.Generator(device=dev).manual_seed(seed)
    n, k = w.shape
    wf = w.float()
    for outl in (False, True):
        x = torch.randn(m, k, device=dev, generator=g)
        if outl:
            x[:, torch.randperm(k, device=dev, generator=g)[: k // 100]] *= 30
        rows = sorted({0, m - 1}) if m > 1 else []
        for j, r in enumerate(rows):
            x[r] = 0
            x[r, (r * 977 + 31 * j + 5) % k] = 64.0 if j == 0 else -32.0
        x = x.to(torch.bfloat16)
        o = fn(x)
        torch.cuda.synchronize()
        if tuple(o.shape) != (m, n) or o.dtype != torch.bfloat16 or not bool(torch.isfinite(o.float()).all()):
            return False, "shape / dtype / non-finite"
        ref = x.float() @ wf.t()
        o = o.float()
        err = float((o - ref).norm() / ref.norm())
        pr = float(((o[rows] - ref[rows]).norm(dim=1) / ref[rows].norm(dim=1)).max()) if rows else 0.0
        if not (err < tol and pr < ptol):
            return False, f"rel {err:.1e} probe {pr:.1e}"
    return True, f"rel {err:.1e}"

_DN_CG_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
// dn cgemv (m = 1): bf16_proj's cgemv (KEELL/muse_bf16_proj/h110a_kclean.py) with PDL. Lossless 12-bit planes,
__device__ __forceinline__ void dn_gdc_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void dn_gdc_launch() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ uint4 dn_ldx16(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 dn_ldw16(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 dn_ldw8(const void* p) {
  uint2 v;
  asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ float dn_ldx1(const __nv_bfloat16* p) {
  unsigned short u;
  asm volatile("ld.global.u16 %0, [%1];" : "=h"(u) : "l"(p) : "memory");
  return __uint_as_float(((uint32_t)u) << 16);
}
__device__ __forceinline__ float dn_esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
__device__ __forceinline__ uint32_t dn_cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K, bool PREW>
__global__ void __launch_bounds__(NTH + 32) dn_cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* x, const int2* __restrict__ etab, int pmax, __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) { const uint2 t = dn_ldw8(&tb[lane]); mine = make_int2((int)t.x, (int)t.y); }
    dn_gdc_wait();
    dn_gdc_launch();
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = dn_ldx1(x + (en.x & 0xFFFF)) * dn_esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = dn_ldw16(L + e0);
      cw[r][j] = dn_ldw8(C + e0 / 2);
    }
  if (PREW) {
    uint32_t dep = 0;
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int j = 0; j < PER; ++j) dep ^= lw[r][j].x ^ lw[r][j].w ^ cw[r][j].x ^ cw[r][j].y;
    asm volatile("{.reg .pred p; setp.eq.u32 p, %0, 0x9E3779B9; @p nanosleep.u32 1;}" :: "r"(dep));
  }
  dn_gdc_wait();
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = dn_ldx16(xp);
    xv[j][1] = dn_ldx16(xp + 8);
  }
  dn_gdc_launch();
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        const uint32_t ha = dn_cg_hib(cws[g] & 0x0F0F0F0Fu, base4);
        const uint32_t hb = dn_cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}
template <int ROWS, int NTH, int K, bool PREW>
static int dn_cg_go(int64_t L, int64_t C, int64_t x, int64_t tab, int pmax, int64_t out, uint32_t b4, int N, int pdl,
                    cudaStream_t st) {
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cfg.gridDim = dim3(N / ROWS); cfg.blockDim = dim3(NTH + 32); cfg.dynamicSmemBytes = 0; cfg.stream = st;
  cfg.attrs = at; cfg.numAttrs = 1;
  return (int)cudaLaunchKernelEx(&cfg, dn_cgemv_kernel<ROWS, NTH, K, PREW>, (const uint8_t*)L, (const uint8_t*)C,
                                 (const __nv_bfloat16*)x, (const int2*)tab, pmax, (__nv_bfloat16*)out, b4);
}
int dn_cgemv(int64_t L, int64_t C, int64_t x, int64_t tab, int64_t pmax, int64_t out, int64_t base4, int64_t N,
             int64_t K, int64_t rows, int64_t nth, int64_t pdl, int64_t prew, int64_t stream) {
  if (N % rows) return -98;
  const cudaStream_t st = (cudaStream_t)stream;
  const uint32_t b4 = (uint32_t)base4;
  if (K == 8192 && rows == 2 && nth == 256 && !prew) return dn_cg_go<2, 256, 8192, false>(L, C, x, tab, (int)pmax, out, b4, (int)N, (int)pdl, st);
  if (K == 4096 && rows == 2 && nth == 128 && prew) return dn_cg_go<2, 128, 4096, true>(L, C, x, tab, (int)pmax, out, b4, (int)N, (int)pdl, st);
  return -99;
}
"""
_DN_CG_CPP = "int dn_cgemv(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\n"

# ---- Third-party notice: the ol_projection kernel in _DN_OL_SRC is from
# https://github.com/NathenFernandes/muse-bf16-proj (submission.py @ 7385e5d), as carried by
# KEELL/muse_bf16_proj/h110a_kclean.py (_make_nb, kernel source unchanged), used under the MIT License below:
# MIT License
# Copyright (c) 2026 Nathen Fernandes
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# ---- end of third-party notice
_DN_OL_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b,int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");
}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
OD void ol_mma128(float* a,uint64_t w,uint64_t x) {
          asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 0;" : "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]),"+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]),"+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]),"+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]),"+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39]),"+f"(a[40]),"+f"(a[41]),"+f"(a[42]),"+f"(a[43]),"+f"(a[44]),"+f"(a[45]),"+f"(a[46]),"+f"(a[47]),"+f"(a[48]),"+f"(a[49]),"+f"(a[50]),"+f"(a[51]),"+f"(a[52]),"+f"(a[53]),"+f"(a[54]),"+f"(a[55]),"+f"(a[56]),"+f"(a[57]),"+f"(a[58]),"+f"(a[59]),"+f"(a[60]),"+f"(a[61]),"+f"(a[62]),"+f"(a[63]) : "l"(w),"l"(x));
        }

template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void ol_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        if constexpr(BN==128) ol_mma128(a,aw,ax);
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);
      uint32_t d=base+2*(cb*72+rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  for(int q=tid;q<T*(R/8);q+=TH) {
    int token=q/(R/8),row=(q%(R/8))*8;
    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);
  }
}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  return int(cudaGetLastError());
}
int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {case 0: return ol_launch_t<128,128,4,3,1,1,1>(x,w,out,m,stream,query);
default:return -99;}
}
"""
_DN_OL_CPP = "int ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n"


def _dn_build():
    """one extension (dn cgemv and / or ol_projection), nvcc on a CPU thread while the layer prepares"""
    try:
        import hashlib
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}a" if major >= 9 else f"{major}{minor}"
        src, cpp, fns = "", "#include <pybind11/pybind11.h>\n#include <stdint.h>\n", []
        if DN_CG:
            src, cpp, fns = src + _DN_CG_SRC, cpp + _DN_CG_CPP, fns + ["dn_cgemv"]
        if DN_OL and major >= 9:
            src, cpp, fns = src + _DN_OL_SRC, cpp + _DN_OL_CPP, fns + ["ol_launch"]
        if not fns:
            DN_EXT[1] = "nothing to build"
            return
        tag = hashlib.md5((src + cpp).encode()).hexdigest()[:12]
        DN_EXT[0] = load_inline(name=f"okc_dn_{tag}_{arch}", cpp_sources=cpp, cuda_sources=src, functions=fns,
                                extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                                no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        DN_EXT[1] = None
    except Exception as ex:  # noqa: BLE001  (no dn kernels: every path above stays)
        DN_EXT[0], DN_EXT[1] = None, (str(ex).splitlines()[-1] if str(ex) else repr(ex))[:300]


_DN_CG_CFG = {"qkv": (2, 256, 0), "un": (2, 128, 1)}   # name -> (rows per CTA, threads, PREW): bf16_proj CG_CONFIGS qkv_gate /


def _dn_cg_pack(w, rows_per_cta):
    """bf16_proj cg_compress: row-major 12-bit planes (lossless; escapes corrected exactly from a per-CTA table)"""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    hist = torch.bincount(e71.flatten(), minlength=128)
    base = int(torch.stack([hist[b:b + 8].sum() for b in range(121)]).argmax())
    o = (e71 - base).clamp(0, 7)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = (e71 < base) | (e71 > base + 7)
    if int(esc.sum()) > n * k // 200:
        return None
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)
    dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
    if not bool(((dec + (w[rows, cols].float() - dec)) == w[rows, cols].float()).all()):
        return None
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101, "n": n, "k": k,
            "rows": rows_per_cta, "esc_cols": torch.unique(cols)[:16].tolist()}


def _dn_cg_run(p, x, pdl):
    out = torch.empty((1, p["n"]), dtype=torch.bfloat16, device=x.device)
    rc = DN_EXT[0].dn_cgemv(p["L"].data_ptr(), p["C"].data_ptr(), x.data_ptr(), p["tab"].data_ptr(), p["pmax"],
                            out.data_ptr(), p["base4"], p["n"], p["k"], p["rows"], p["nth"], 1 if pdl else 0, p["prew"],
                            torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"dn cgemv launch {rc}")
    return out


_dn_prior_bfp_gemm = _bfp_gemm


def _bfp_gemm(st, name, x, m, n, k):
    p = st.get("dn_cg", {}).get(name) if m == 1 else None
    if (p is not None and p["n"] == n and p["k"] == k and x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.data_ptr() % 16 == 0):
        return _dn_cg_run(p, x, _pdl(1))
    return _dn_prior_bfp_gemm(st, name, x, m, n, k)


def _dn_cg_ok(p, w):
    """random / outlier rows vs FP32 (rel <= 3e-3: exact weights, FP32 accumulation, BF16 output) and one-hot columns bit-exact
    (escape columns first)"""
    ok, msg = _dn_lin_ok(lambda x: _dn_cg_run(p, x, False), w, 1, 9091, tol=3e-3)
    if not ok:
        return False, msg
    n, k = w.shape
    g = torch.Generator(device=w.device).manual_seed(9092)
    cols = p["esc_cols"] + torch.randint(0, k, (16,), device=w.device, generator=g).tolist()
    x = torch.zeros(1, k, device=w.device, dtype=torch.bfloat16)
    for c in cols:
        x.zero_()
        x[0, c] = 8.0
        got = _dn_cg_run(p, x, False)
        if not torch.equal(got[0], (w[:, c].float() * 8.0).to(torch.bfloat16)):
            return False, f"one-hot column {c} not exact"
    return True, msg + f", {len(cols)} one-hot columns exact"


def _dn_cg_setup(weights, st):
    st["dn_cg"] = {}
    if DN_EXT[0] is None or not hasattr(DN_EXT[0], "dn_cgemv"):
        _dn_say(f"cg off: extension {DN_EXT[1]}")
        return
    bst = st.get("bfp")
    if bst is None or st.get("mm_pick", {}).get((1, "qkv")) != "bfp" or st.get("mm_pick", {}).get((1, "un")) != "bfp":
        _dn_say("cg off: T = 1 qkv / un are not bf16_proj picks")
        return
    packs = {}
    for name, w in (("qkv", st["qkv"]), ("un", st["unpool_slice"])):
        rows, nth, prew = _DN_CG_CFG[name]
        p = _dn_cg_pack(w, rows)
        if p is None:
            _dn_say(f"cg {name}: not packable")
            continue
        p["nth"], p["prew"] = nth, prew
        ok, msg = _dn_cg_ok(p, w)
        _dn_say(f"cg {name}: {'valid' if ok else 'INVALID'} ({msg}; pmax {p['pmax']})")
        if ok:
            packs[name] = p
    if "qkv" in packs:   # qkv is the layer's first kernel: isolated race (paired cold-L2 replays, PDL launch as in the layer)
        x = (0.5 * torch.randn(1, HIDDEN, device=st["qkv"].device,
                               generator=torch.Generator(device=st["qkv"].device).manual_seed(9093))).to(torch.bfloat16)
        n = st["qkv"].shape[0]
        sa, sb = dict(st), dict(st)
        sb["dn_cg"] = {"qkv": packs["qkv"]}
        ts = _compose_times([lambda: _bfp_gemm(sa, "qkv", x, 1, n, HIDDEN), lambda: _bfp_gemm(sb, "qkv", x, 1, n, HIDDEN)],
                            reps=40)
        use = ts[1] < DN_GAIN * ts[0]
        _dn_say(f"cg qkv T=1 isolated: sk2 {ts[0]:.2f} us, cgemv {ts[1]:.2f} us -> {'cgemv' if use else 'sk2'}")
        if use:
            st["dn_cg"]["qkv"] = packs["qkv"]
    if "un" in packs:    # the unpooler slice sits between the MoE and the last boundary: in-chain race at T = 1
        sb = dict(st)
        sb["dn_cg"] = dict(st["dn_cg"], un=packs["un"])
        global DN_CHAIN_US
        _keep_us, DN_CHAIN_US = DN_CHAIN_US, PIN_UN_CHAIN_US   # p001: cgemv kept unless slower (chain check still gates)
        try:
            tag, _ = _dn_race(st, 1, [("sk2", st), ("cgemv", sb)], 9094)
        finally:
            DN_CHAIN_US = _keep_us
        if tag == "cgemv":
            st["dn_cg"]["un"] = packs["un"]
    _dn_say(f"cg picks {sorted(st['dn_cg'])}")


_DN_O_T = (128, 256)
_DN_AG_O = {"tile": 20, "stages": 35, "splitk": 1, "red": 0, "swz": 0, "custom": 6, "cluster": 2}   # bf16_proj ag pin ("o", 256)


def _dn_ol_pack(w):
    """bf16_proj _ol_pack: W [8192, 1024] -> [128 row tiles, 16 k chunks, 64, 64] with the 128B swizzle (lossless repack)"""
    c = torch.arange(64, device=w.device)[None, :] ^ ((torch.arange(64, device=w.device)[:, None] & 7) * 8)
    return w.reshape(128, 64, 16, 64).permute(0, 2, 1, 3).gather(-1, c.expand(128, 16, 64, 64)).contiguous()


def _dn_ol_run(x, st):
    out = torch.empty((x.shape[0], 8192), device=x.device, dtype=torch.bfloat16)
    rc = DN_EXT[0].ol_launch(0, x.data_ptr(), st["dn_olw"].data_ptr(), out.data_ptr(), x.shape[0],
                             torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f"dn ol_projection launch {rc}")
    return out


_dn_prior_dense = dense


def dense(x, w, T, state=None, name=None):
    if (name == "o" and state is not None and T in state.get("dn_ol", ()) and x.dtype == torch.bfloat16
            and x.is_contiguous() and x.data_ptr() % 16 == 0 and tuple(x.shape) == (T, 1024) and tuple(w.shape) == (8192, 1024)):
        return _dn_ol_run(x, state).view(1, T, 8192), 1
    return _dn_prior_dense(x, w, T, state, name)


def _dn_ag_plan(lt, w, m, cfg):
    """bf16_proj's ag pin (cuBLASLt algo 66 built from attributes) on this file's tn layouts; None if the library rejects it"""
    algo = _GraftAlgo()
    lt.call("cublasLtMatmulAlgoInit", lt.handle, 68, 0, 14, 14, 14, 14, 66, _C.byref(algo))
    for nm, attr, ctype in _GRAFT_QKV256_ATTRS:
        v = ctype(cfg[nm])
        lt.call("cublasLtMatmulAlgoConfigSetAttribute", _C.byref(algo), attr, _C.byref(v), _C.sizeof(v))
    n, k = w.shape
    layouts = _graft_layouts(lt, m, n, k, "tn", 1, False)
    heur = _GraftHeur()
    code = lt.lib.cublasLtMatmulAlgoCheck(lt.handle, lt.desc, *layouts, layouts[2], _C.byref(algo), _C.byref(heur))
    if code or heur.workspaceSize > lt.workspace.numel():
        return None
    return _GraftPlan(lt=lt, layouts=layouts, algo=algo, weight=w, n=n, fp32=False)


def _dn_o_setup(weights, st):
    st["dn_ol"] = set()
    w = st["o"]
    ol_ok = DN_EXT[0] is not None and hasattr(DN_EXT[0], "ol_launch")
    if ol_ok:
        st["dn_olw"] = _dn_ol_pack(w)
        for T in _DN_O_T:   # first launch per instantiation sets its smem attribute, outside any capture
            _dn_ol_run(torch.zeros((T, 1024), device=w.device, dtype=torch.bfloat16), st)
        torch.cuda.synchronize()
    else:
        _dn_say(f"ol off: extension {DN_EXT[1]}")
    lt = None
    try:
        lt = _GraftLt(True)
    except Exception as ex:  # noqa: BLE001
        _dn_say(f"ag pin off: {ex!r}"[:200])
    for T in _DN_O_T:
        cands = []
        if ol_ok:
            ss = dict(st)
            ss["dn_ol"] = set(st["dn_ol"]) | {T}
            cands.append(("ol", ss))
        if lt is not None:
            try:
                plan = _dn_ag_plan(lt, w, T, _DN_AG_O)
            except Exception as ex:  # noqa: BLE001
                plan = None
                _dn_say(f"ag pin T={T}: {ex!r}"[:200])
            if plan is None:
                _dn_say(f"ag pin T={T}: not available (cublasLtMatmulAlgoCheck)")
            else:
                ss = dict(st)
                ss["graft_lt"] = dict(st.get("graft_lt", {}))
                ss["graft_lt"][("o", T)] = plan
                cands.append(("ag", ss))
        live = []
        for tag, ss in cands:
            try:
                ok, msg = _dn_lin_ok(lambda x, ss=ss: dense(x, w, T, ss, "o")[0][0], w, T, 9100 + T)
            except Exception as ex:  # noqa: BLE001
                ok, msg = False, repr(ex)[:160]
            _dn_say(f"o T={T} {tag}: {'valid' if ok else 'INVALID'} ({msg})")
            if ok:
                live.append((tag, ss))
        if not live:
            continue
        x = (0.1 * torch.randn(T, 1024, device=w.device, generator=torch.Generator(device=w.device).manual_seed(9101))).to(torch.bfloat16)
        allv = [("base", st)] + live
        ts = _compose_times([lambda ss=ss: dense(x, w, T, ss, "o") for _, ss in allv], reps=24)
        _dn_say(f"o T={T} isolated " + " ".join(f"{t}:{v:.2f}" for (t, _), v in zip(allv, ts)))
        keep = [c for c, v in zip(live, ts[1:]) if v < DN_GAIN * ts[0]]
        if not keep:
            continue
        tag, ss = _dn_race(st, T, [("base", st)] + keep, 9102 + T)
        if tag == "ol":
            st["dn_ol"].add(T)
        elif tag == "ag":
            st["graft_lt"] = dict(st["graft_lt"])
            st["graft_lt"][("o", T)] = ss["graft_lt"][("o", T)]
    _dn_say(f"o picks: ol at {sorted(st['dn_ol'])}")


_DN_PR_T = (128,)
_DN_IN_GRAFT = [False]
_DN_HEUR_SIG = []


def _dn_lt_heur(lt, layouts, count=8):
    """cublasLtMatmulAlgoGetHeuristic: up to `count` algorithms for these layouts within the plan workspace"""
    lib = lt.lib
    if not _DN_HEUR_SIG:
        lib.cublasLtMatmulPreferenceCreate.argtypes, lib.cublasLtMatmulPreferenceCreate.restype = [_C.POINTER(_GP)], _GI
        lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [_GP, _GI, _GP, _GSZ]
        lib.cublasLtMatmulPreferenceSetAttribute.restype = _GI
        lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [_GP] * 7 + [_GI, _C.POINTER(_GraftHeur), _C.POINTER(_GI)]
        lib.cublasLtMatmulAlgoGetHeuristic.restype = _GI
        lib.cublasLtMatmulPreferenceDestroy.argtypes, lib.cublasLtMatmulPreferenceDestroy.restype = [_GP], _GI
        _DN_HEUR_SIG.append(True)
    pref = _GP()
    lt.call("cublasLtMatmulPreferenceCreate", _C.byref(pref))
    try:
        ws = _C.c_uint64(lt.workspace.numel())
        lt.call("cublasLtMatmulPreferenceSetAttribute", pref, 1, _C.byref(ws), _C.sizeof(ws))
        res = (_GraftHeur * count)()
        nret = _GI(0)
        lt.call("cublasLtMatmulAlgoGetHeuristic", lt.handle, lt.desc, *layouts, layouts[2], pref, count, res, _C.byref(nret))
    finally:
        lib.cublasLtMatmulPreferenceDestroy(pref)
    return [_GraftAlgo((_C.c_uint64 * 8)(*res[i].algo.data)) for i in range(nret.value)
            if res[i].state == 0 and res[i].workspaceSize <= lt.workspace.numel()]


def _dn_router_plan(st, T):
    """the fastest valid FP32-out cuBLASLt plan for x [T, 8192] @ [hi; lo]^T -> [T, 512] (cold-L2 replays)"""
    lt = _GraftLt(True)
    w = st["pr"][D_EXPERT:]
    n, k = w.shape
    dev = w.device
    g = torch.Generator(device=dev).manual_seed(9200 + T)
    x = torch.randn(T, k, device=dev, generator=g)
    x[:, torch.randperm(k, device=dev, generator=g)[: k // 100]] *= 30
    x = x.to(torch.bfloat16)
    ref = x.double() @ st["router_fp32"].double().t()
    plans = []
    for batch in (1, 2, 4):
        layouts = _graft_layouts(lt, T, n, k, "tn", batch, True)
        for algo in _dn_lt_heur(lt, layouts):
            p = _GraftPlan(lt=lt, layouts=layouts, algo=algo, weight=w, n=n, fp32=True)
            try:
                o = _graft_lt_run(x, p)
                torch.cuda.synchronize()
                lg = o[:, :256].double() + o[:, 256:].double()
                err = float((lg - ref).norm() / ref.norm())
            except Exception:  # noqa: BLE001
                continue
            if bool(torch.isfinite(o).all()) and err < 2e-5:
                plans.append((f"b{batch}a{len(plans)}", p))
    if not plans:
        return None, "no valid heuristic plan"
    ts = _compose_times([lambda p=p: _graft_lt_run(x, p) for _, p in plans], reps=16)
    i = min(range(len(ts)), key=ts.__getitem__)
    return plans[i][1], f"{len(plans)} valid, best {plans[i][0]} {ts[i]:.2f} us (worst {max(ts):.2f})"


_dn_prior_graft_after = _graft_after_attention


def _graft_after_attention(data, state, o_in, gate, T):
    _DN_IN_GRAFT[0] = True    # only the grafted chain's route reads the [T, 512] router logits (LSTR)
    try:
        return _dn_prior_graft_after(data, state, o_in, gate, T)
    finally:
        _DN_IN_GRAFT[0] = False


_dn_prior_pool_router = _pool_router_fp32


def _pool_router_fp32(h, hr, T, state, ppick):
    pl = state.get("dn_pr", {}).get(T) if _DN_IN_GRAFT[0] else None
    if pl is not None and ppick is True and h.is_contiguous() and h.data_ptr() % 16 == 0:
        pooled = _rq_gemm(state, "pool", h, T)
        if pooled is not None:
            return (pooled, _graft_lt_run(h, pl)), -1
    return _dn_prior_pool_router(h, hr, T, state, ppick)


def _dn_pr_setup(weights, st):
    st["dn_pr"] = {}
    for T in _DN_PR_T:
        if ("pool", T) not in st.get("chen_rq", {}):
            _dn_say(f"pr T={T} off: no rq pooler pick")
            continue
        if st.get("mm_pick", {}).get((T, "pr")) is not True or ("pr", T) not in st.get("graft_lt", {}):
            _dn_say(f"pr T={T} off: the layer's pooler + router is not its cuBLASLt pin")
            continue
        try:
            plan, msg = _dn_router_plan(st, T)
        except Exception as ex:  # noqa: BLE001
            plan, msg = None, repr(ex)[:200]
        _dn_say(f"pr T={T} router: {msg}")
        if plan is None:
            continue
        ss = dict(st)
        ss["dn_pr"] = {T: plan}
        tag, _ = _dn_race(st, T, [("lt4608", st), ("rq+lt512", ss)], 9300 + T)
        if tag != "lt4608":
            st["dn_pr"][T] = plan
    _dn_say(f"pr picks {sorted(st['dn_pr'])}")


_dn_prior_prepare = prepare


def prepare(weights):
    th = None
    if DN_CG or DN_OL:
        th = threading.Thread(target=_dn_build, daemon=True)   # nvcc for the dn extension while the layer prepares
        th.start()
    st = _dn_prior_prepare(weights)
    if th is not None:
        th.join(timeout=90.0)
        if th.is_alive():
            DN_EXT[0], DN_EXT[1] = None, "build still running"
    lt0 = dict(st.get("graft_lt", {}))
    for flag, fn, keys in ((DN_CG, "_dn_cg_setup", ("dn_cg",)), (DN_OL, "_dn_o_setup", ("dn_ol",)),
                           (DN_PR, "_dn_pr_setup", ("dn_pr",))):
        if not flag:
            continue
        try:
            globals()[fn](weights, st)
        except Exception as ex:  # noqa: BLE001  (that part off; the paths above stand)
            for k in keys:
                st[k] = {} if k != "dn_ol" else set()
            if fn == "_dn_o_setup":
                st["graft_lt"] = lt0
            _dn_say(f"{fn} failed: " + (str(ex).splitlines()[-1] if str(ex) else repr(ex))[:300])
        torch.cuda.synchronize()
    return st



# ======================================================================================================================
# p9 (EROREx, 2026-11-02; KEELL/muse_decode_layer/p9/gen_p9.py): SC5W: lossless 5-bit WGS block scales (group minimum + 5-bit offsets, 35552 B per stage; SC6 / 8-bit fallback when a group of the actual weights spans 32+ codes). Applied after every patch set above; each
# patch asserts its count. Small builds (T <= 16, _MA_ALT_SRCS) untouched.
# ======================================================================================================================
_P9_D4 = False
_P9_SC5W = True
_MA_SC5W = False   # set in prepare(): SC5 WGS stages (every group of the actual weights spans < 32 codes)


def _p9_sub(c, old, new, tag, count=1):
    n = c.count(old)
    if n != count:
        raise RuntimeError(f"p9 patch {tag}: {n} matches (want {count})")
    return c.replace(old, new)


# c001: `if _P9_D4:` block removed (_P9_D4 is False: never ran)

if _P9_SC5W:
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, "#ifndef OKC_SC6W\n",
                           "#ifndef OKC_SC5W\n#define OKC_SC5W 0   // p9: WGS stages carry SC5 5-bit scale offsets (35552 B per stage)\n"
                           "#endif\n#ifndef OKC_SC6W\n", "sc5.def")
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, "constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;",
                           "constexpr int WGS_PIECE = OKC_SC5W ? 8880 : OKC_SC6W ? 8960 : 9216;", "sc5.piece")
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, "constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;",
                           "constexpr int WGS_STG = OKC_SC5W ? 35552 : OKC_SC6W ? 35840 : 36864;\n"
                           "constexpr int WGS_STGP = OKC_SC5W ? 35584 : WGS_STG;  // p9: stage pitch in global memory (128 B aligned)\n"
                           "static_assert(!OKC_SC5W || (!OKC_SC6W && OKC_PRMTSC && 4 * WGS_PIECE + 32 == WGS_STG), \"p9 SC5W stage\");", "sc5.stg")
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, "(size_t)WGS_STG;", "(size_t)WGS_STGP;", "sc5.pitch")
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, r'''  const unsigned char* pc0 = ring + wg * WGS_PIECE;
''', r'''  const unsigned char* pc0 = ring + wg * WGS_PIECE;
#if OKC_SC5W
  // p9 SC5W, once per segment: this thread's X word / bit slot and its warp's base pair (after the 4 pieces)
  const int p9xo = 8704 + ((wp * 32 + lane) / 3) * 4, p9ho = 4 * WGS_PIECE - wg * WGS_PIECE + (wg * 4 + wp) * 2;
  const uint32_t p9xs = 5u * (uint32_t)((wp * 32 + lane) % 3);
#endif
''', "sc5.hoist")
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, "#if OKC_SC6W\n    uint32_t sgs[4];", r'''#if OKC_SC5W
    uint32_t sgs[4];
    {  // p9 SC5W: pairs j = 0..2 from this thread's W word, j = 3 from the shared X word; the warp's bases (row halves)
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + (wp * 32 + lane) * 4);
      const uint32_t X = *reinterpret_cast<const uint32_t*>(pc + p9xo);
      uint32_t K2 = __byte_perm((uint32_t)*reinterpret_cast<const unsigned short*>(pc + p9ho), 0u, 0x4140u) * k16c + 0x5C005C00u;
      asm("" : "+r"(K2));  // opaque: one IMAD per pair
      sgs[0] = (W & 0x001F001Fu) * k16c + K2;
      sgs[1] = ((W >> 5) & 0x001F001Fu) * k16c + K2;
      sgs[2] = ((W >> 10) & 0x001F001Fu) * k16c + K2;
      sgs[3] = ((X >> p9xs) & 0x001F001Fu) * k16c + K2;
    }
#elif OKC_SC6W
    uint32_t sgs[4];''', "sc5.dec")
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, "#if !OKC_SC6W\n      const uint2 scg = scs;\n#endif",
                           "#if !OKC_SC6W && !OKC_SC5W\n      const uint2 scg = scs;\n#endif", "sc5.scg")
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, "#if OKC_SC6W\n          const uint32_t sgp = sgs[js];",
                           "#if OKC_SC6W || OKC_SC5W\n          const uint32_t sgp = sgs[js];", "sc5.sgp")
    MOE_CUDA_SRC = _p9_sub(MOE_CUDA_SRC, "#if OKC_SC6W\n    uint32_t sgs[4];", "#if OKC_SC6W\n    uint32_t sgs[4];", "sc5.left", 0)

    def _p9_sc5w_groups(sf, rows):
        # code span (max - min) of every SC5 group (stage x warpgroup x warp x row half: 8 rows x 256 K), experts sf [E, R, 256]
        s = sf.contiguous().view(torch.uint8).index_select(1, rows).to(torch.int16)
        s = s.view(s.shape[0], -1, 4, 4, 2, 8, 16, 4, 4)   # e, ub, q, warp, h, g, st, t, j (_wgs_copy's scale order)
        return s.amax(dim=(5, 7, 8)) - s.amin(dim=(5, 7, 8))

    def _p9_sc5w_decide(weights):
        global _MA_SC5W, _MA_SC6W
        ok = False
        if _P9_SC5W and _MA_SFX and _wgs_on():
            dev = weights["w13_sf"].device
            rb, g = torch.arange(512, device=dev), torch.arange(8, device=dev)
            rowsA = torch.stack((8 * rb[:, None] + g[None, :], 4096 + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
            sp = max(int(_p9_sc5w_groups(weights["w13_sf"], _kperm_rows(rowsA)).max()),
                     int(_p9_sc5w_groups(weights["w2_sf"], torch.arange(4096, device=dev)).max()))
            ok = sp <= 31
            print(f"[p9] sc5w widest group {sp} codes", flush=True)
        _MA_SC5W = bool(ok)
        if _MA_SC5W:
            _MA_SC6W = False
        print(f"[p9] wgs sc5 {_MA_SC5W} sc6 {_MA_SC6W}", flush=True)

    def _p9_sc5w_copy(codes, sf, rows):
        # SC5 WGS copy: [256-row unit][stage (16)][35584 B]: 4 pieces of [8 KB codes][W 512 B][X 176 B] + 32 B bases + pad
        b = _p9_prev_wgs_copy(codes, sf, rows, False).view(-1, 16, 4, 9216)
        nub = b.shape[0]
        v = b[..., 8192:].reshape(nub, 16, 4, 4, 32, 4, 2).to(torch.int32)   # ub, st, q, warp, lane, j, h
        base = v.amin(dim=(4, 5))                                             # ub, st, q, warp, h
        f = (v - base[:, :, :, :, None, None, :]).reshape(nub, 16, 4, 128, 4, 2)
        if int(f.max()) > 31:
            raise RuntimeError("p9 SC5W: a scale group spans more than 32 codes (prepare's check disagrees)")
        W = f[..., 0, 0] | (f[..., 0, 1] << 16)
        for jj in (1, 2):
            W = W | (f[..., jj, 0] << (5 * jj)) | (f[..., jj, 1] << (5 * jj + 16))
        p3 = f[..., 3, 0] | (f[..., 3, 1] << 16)
        p3 = torch.cat((p3, torch.zeros_like(p3[..., :4])), -1).view(nub, 16, 4, 44, 3)
        X = p3[..., 0] | (p3[..., 1] << 5) | (p3[..., 2] << 10)
        piece = torch.cat((b[..., :8192], W.contiguous().view(torch.uint8).view(nub, 16, 4, 512),
                           X.contiguous().view(torch.uint8).view(nub, 16, 4, 176)), -1)
        hdr = base.reshape(nub, 16, 32).to(torch.uint8)                      # byte 2 (q * 4 + warp) + h
        pad = torch.zeros(nub, 16, 32, dtype=torch.uint8, device=b.device)
        return torch.cat((piece.reshape(nub, 16, 4 * 8880), hdr, pad), -1).contiguous().view(-1)

    _p9_prev_wgs_copy = _wgs_copy

    def _wgs_copy(codes, sf, rows, sc6=False):
        if _MA_SC5W:
            return _p9_sc5w_copy(codes, sf, rows)
        return _p9_prev_wgs_copy(codes, sf, rows, sc6)

    _p9_prev_ma_ext = _ma_ext

    def _ma_ext(defines=None):
        d = tuple(MOE_DEFINES if defines is None else defines)
        if _MA_SC5W and "-DOKC_WGS=1" in d and "-DOKC_SC5W=1" not in d:
            d = d + ("-DOKC_SC5W=1",)
        return _p9_prev_ma_ext(d)

# c001: `if _P9_D4:` block removed (_P9_D4 is False: never ran)
