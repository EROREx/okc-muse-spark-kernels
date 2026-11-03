# Muse Spark Open — H200 CUDA kernels (Open Kernel Challenge 2026)

My best kernel submissions for the [Open Kernel Challenge](https://openkernelchallenge.dev) (Meta × Paradigm × Andromeda, 2026).
The challenge: make the inference kernels of **Muse Spark Open**, Meta's open mixture-of-experts LLM (256 experts, NVFP4 / 4-bit weights), run faster on **NVIDIA H200** than SGLang's production kernels.
Every submission is checked against a reference on fresh inputs and timed against SGLang's own kernel in the same run, with CUDA-graph replays and a cold L2 cache.

Team / GitHub handle on the board: **EROREx**

## Results (official leaderboard, 2026-11-03)

**Overall: rank 110 of 1,915 teams (top 6%)**

| Problem | What it is | Speedup vs SGLang | Board rank | Top % | #1 on board |
|---|---|---:|---:|---:|---:|
| `muse_attention_decode` | Paged FP8-KV attention for decode / speculative verify (sliding + full layers, 80k context) | **2.68×** | 102 / 1,303 | 7.8% | 3.34× |
| `muse_moe_w4a16` | 4-bit (NVFP4) mixture-of-experts layer, expert-parallel, BF16 activations | **2.42×** | 103 / 1,358 | 7.6% | 2.76× |
| `muse_decode_layer` | A whole fused transformer decode layer (projections + attention + router + MoE) | **2.27×** | 110 / 1,234 | 8.9% | 2.49× |
| `muse_prefill_layer` | A whole transformer prefill layer (753-token turns over an 80k-token cache) | **1.81×** | 178 / 841 | 21.2% | 2.25× |
| `muse_bf16_proj` | BF16 dense projections (QKV/gate, O, pooler, unpooler, router) at decode batch sizes | **1.56×** | 241 / 1,690 | 14.3% | 1.69× |

Speedup = geometric mean over the problem's scored shapes of (SGLang time / our time), as measured by the official judge.
The source of each row is in [`kernels/`](kernels/), one file per problem, byte-for-byte the file behind that leaderboard entry.

## What's in the kernels

All kernels are single Python files that build CUDA C++ / inline PTX at `prepare()` time (`torch.utils.cpp_extension`) and launch it inside a captured CUDA graph. Highlights of my own work:

- **MoE (W4A16)** — warp-group-specialized (WGS) persistent kernel: TMA bulk copies + mbarriers stream 4-bit weights while tensor cores run; K-permuted weight packing; **lossless 5/6-bit compression of the FP8 block scales** (group minimum + small offsets) to cut HBM bytes; per-expert phase counters instead of a grid-wide barrier; cooperative launch with a prepare-time self-test of every build.
- **Decode layer** — one stacked pipeline for the whole layer: programmatic dependent launch (PDL) chaining across kernels, pre-wait L2 prefetch of the next stage's weights and page tables, per-batch-size route races in `prepare()` (the fastest correct kernel is picked per shape), split-merge of attention partials with one bulk async reduction (`cp.reduce.async.bulk`) per CTA, FP32-exact router.
- **Attention decode** — split-KV FP8 attention with TMA bulk-reduction merge, page-table L2 prefetch at entry, and an FP16-FMA bit-trick `exp2` for the softmax probabilities (only the attention P matrix; every other exponential is exact).
- **Prefill / BF16 projections** — hybrid dispatch: custom GEMV / split-K kernels and pinned cuBLASLt plans raced per shape, custom FMHA prefill attention with an exact hardware `ex2` softmax.
- **Rules-safe engineering** — everything is CUDA-graph capturable (no host syncs, no value-dependent allocation), weights keep their precision (repacked losslessly only), attention attends to every key.

Development loop: a local replica of the official judge, run on rented H200s (Modal) for compile/correctness, plus the organizer's off-board check runs as timing ground truth; hundreds of A/B-tested variants, only measured wins kept.

## Credits and licensing

The challenge encouraged building on public work, and these kernels do — each file's header lists exactly what came from where, and those notices are kept verbatim. Upstream work this builds on, all Apache-2.0 unless noted:

- [vjeux/okc-2026](https://github.com/vjeux/okc-2026)
- [alygg77/okc-muse-spark-kernels](https://github.com/alygg77/okc-muse-spark-kernels)
- [keell0renz/muse-open-kernel-challenge](https://github.com/keell0renz/muse-open-kernel-challenge) (team keell0renz, Bohdan Agarkov)
- ZhimaoL/sglang (`okc_patch` CUDA sources), Madhavan113/ParadigmKernelChallenge, chenwainuo/okc
- [NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj) — MIT, see [`LICENSE-MIT-NathenFernandes.txt`](LICENSE-MIT-NathenFernandes.txt)
- helper code posted by Matsvei ("matvey") on agents.kernarc.com

My modifications are described in each file's header (`EROREx, <date>` lines). This repository is distributed under the [Apache License 2.0](LICENSE), the same license all challenge submissions are released under.

Not affiliated with or endorsed by Meta, Paradigm, Andromeda or the SGLang project.
