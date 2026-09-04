# RHIZOME — Specification

This document is the source of truth. Every deviation from it is an ADR under
`docs/adr/`. The parameter table below is generated: the `spec-sync` CI job
asserts it equals `rhizome params --all --format=markdown`.

Priority order when goals conflict:
`correctness > reproducibility > memory safety > performance > portability`.

## 1. Glossary

- **block** — one residual unit: mixer (optional) + FFN, pre-norm and sandwich
  norm.
- **stratum** — 5 Local blocks followed by 1 Global block.
- **Local block** — Gated Delta-Rule mixer + FFN.
- **Global block** — Multi-head Latent Attention (MLA) + FFN.
- **Think Core** — a weight-shared 3-block unit executed `1..L_max` times per
  token.
- **prelude / coda** — trunk strata before / after the Think Core.
- **segment** — one document inside a packed sequence; recurrent state resets
  at segment boundaries.
- **plan** — a static, serialized execution graph for one `(phase, shape)`.

## 2. Non-negotiable rules

R1 Production code is Rust. R2 No `todo!()`/`unimplemented!()`/`dbg!` on the
default branch. R3 Every op has an f64 reference, an optimised backend, parity
tests, hand-written adjoints and finite-difference gradient checks. R4 All
backends pass one conformance suite; unsupported `(op, dtype, device)` triples
are hard errors at plan time. R5 CI is hermetic — no network in tests, and no
downloads in `build.rs`. R6 `unsafe` only in kernel/FFI/allocator crates.
R7 SPEC.md matches the implementation. R8 Milestones are complete only when a
workflow run proves them. R9 Deterministic mode is bit-identical and
batch-invariant. R10 Typed errors, no panics on user input. R11 All randomness
comes from counter-based Philox keyed by `(seed, stream, counter)`.
R12 Never loosen a tolerance to go green without an ADR.

## 3. Model architecture

### 3.1 Top level

```
Input -> front end -> prelude strata 1..P -> Think Core (3 shared blocks x L)
      -> coda strata P+1..N -> back end + MTP heads
```

Residual stream in bf16 (f32 in reference mode). Pre-norm RMSNorm
(`eps = 1e-6`) on every branch input, sandwich norm on every branch output. No
positional encoding in Local blocks; decoupled RoPE (base `1e6`) in attention.

### 3.2 Local block — Gated Delta-Rule mixer

Per head, with `q` L2-normalised and rescaled by `sqrt(d_h)`, `k` L2-normalised,
`beta = sigmoid(beta_logit)`, `alpha = exp(-softplus(a_h) * softplus(alpha_logit))`:

```
S_t = alpha_t * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
o_t = S_t q_t
```

State `S` is `d_h x d_h` in fp32. A causal depthwise conv1d of kernel 4 with a
3-sample carried state precedes the projections; both `S` and the conv state
reset at segment starts (`alpha := 0`). Training uses a chunk-wise parallel
scan (chunk 64) that is mathematically identical to the sequential recurrence;
inference uses `gated_delta_step` and `gated_delta_multistep` (<= 8 positions,
returning the state after each position for speculative decoding).

### 3.3 Global block — Multi-head Latent Attention

Per token per layer the cache stores `[c_kv (512), k_rope (64)] = 576` values.
Prefill up-projects `K` and `V` from `c_kv` and runs a flash-style tiled,
causal, block-diagonal-by-segment kernel with fp32 softmax. Decode absorbs
`W_UK` into the query and `W_UV` into `W_O` and reduces over 64-token pages
with a fixed split-K order, which makes it batch-invariant. Optional attention
logit softcap (default off) and optional FP8 e4m3 latent cache.

### 3.4 FFN — MoE with shared experts (dropless)

Affinities `s_e = sigmoid(clamp(x . w_e, -30, 30))`; selection score
`s_e + b_e`, where `b_e` is used for selection only. Experts are split into `G`
groups; the group score is the sum of its top-3 selection scores; the top-2
groups are kept and the top-k experts are chosen within them, ties toward the
lower index. Gates are the selected `s_e` renormalised to sum to 1.
Loss-free balancing updates `b_e <- b_e - gamma * sign(load_e - mean_load)` with
`gamma = 1e-3` decayed to zero over the last 10% of pretraining. The
sequence-wise balance loss (weight `1e-4`) is
`sum_e f_e * P_e` with `f_e = (E_r / (k T)) * count_e` and `P_e = mean_t s_e(t)`.
Routing is dropless: grouped GEMM with variable group sizes, no capacity factor.

### 3.5 Think Core

Per token, `r <- h_0`, then for `i = 1, 2, ...`:

```
r <- r + W_in . RMSNorm([r ; h_0]) + e_iter[i] + e_budget[B]
C1: MLA read (default source: the Core-input cache of tokens <= t) + MoE FFN
C2: Gated Delta mixer over per-iteration slab i + MoE FFN
C3: dense SwiGLU FFN, hidden 4*d
p_halt_i = halt_head(r)
```

Inference halts when `p_halt > tau` (default 0.5, fixed in deterministic mode)
or `i == B`. Slabs are double-buffered with a per-sequence `last_halt` table:
the prior state for iteration `i` is `slab[min(i, last_halt)]` as of after the
previous token, so no state copies are ever required.

Training samples `L = round(TruncLogNormal(mu = ln 2, sigma = 0.5))` clamped to
`[1, L_max]` once per optimizer step, backpropagates through the last
`min(L, 4)` iterations, and trains the (detached) halt head with BCE against a
convergence target `||r_i - r_{i-1}|| / ||r_i|| < 0.05` or, on a 1/8 token
subsample, a loss target `l_i - l_L < 0.02` nats. A depth regulariser
`lambda_d * E[depth]` with `E[depth] = sum_i prod_{j<i} (1 - p_j)` flows into
the halt head only.

### 3.6 Product-key memory

4 query heads, `q -> (q1, q2)` in `R^256 x R^256`, key tables
`K1, K2 in R^{1024 x 256}`, top-32 per table -> 1024 candidates -> top-32 ->
softmax -> gather from `V in R^{1048576 x d}`. Values are sharded by slot range;
values use sparse AdamW, keys use AdamW; dead keys are re-initialised.

### 3.7 / 3.8 Front end, back end, MTP

BPE path: 128K vocabulary with byte fallback, untied unembedding, next-token CE
plus z-loss `1e-4`. Byte-latent path: frozen 150M entropy patcher, 4-block byte
encoder with hashed n-gram embeddings, cross-attention patch pooling, 6-block
byte decoder. Two sequential MTP heads with loss weights `0.3 * 0.85^(k-1)`,
reused as speculative drafts at inference.

### 3.9 Numerics

muP over width, depth-scaled init `1 / sqrt(2 N_blocks)` on output projections,
Muon with RMS-matched update scaling `0.2 * sqrt(max(d_out, d_in))`, FP8 e4m3
training GEMMs with fp32 accumulation, MXFP4/MXINT4 deployment with 32-element
blocks and e8m0 shared scales, global grad-norm clip 1.0.

### 3.10 Reference configurations

<!-- BEGIN PARAMS TABLE -->
PLACEHOLDER
<!-- END PARAMS TABLE -->

## 4. Op vocabulary and tolerances

The closed op set and its per-op tolerances are documented in
`docs/kernels.md`. Defaults: f32 CPU `atol 1e-6 / rtol 1e-5`; bf16, fp8 and MX
backends `atol 1e-2 * RMS(ref) / rtol 2e-2`. Gradient checks use f64 central
differences with `h = 1e-5` and require relative error `<= 1e-6`.

## 5. Training system

Static planning (forward graph, derived backward graph, liveness, arena
offsets, activation checkpointing, fusion), ZeRO-3 data parallelism, expert
parallelism bounded to two ranks per token by the top-2-group rule, interleaved
1F1B pipeline parallelism with the Think Core on a balanced stage, sequence
parallelism, Muon plus AdamW plus sparse AdamW, a WSD schedule, and async
sharded topology-agnostic checkpoints.

## 6. Inference engine

Static plans per `(phase, batch bucket in {1,2,4,8,16,32,64,128})`, paged latent
KV (page = 64), fixed-size recurrent slabs, double-buffered Core slabs, a radix
prefix cache over raw bytes, continuous batching with chunked prefill (2048),
self-drafted speculative decoding from the MTP heads, byte-level constrained
decoding, and a deterministic mode with fixed reduction order, batch-invariant
split-K, stable tie-breaks and Philox-seeded sampling.

## 7. Weight format

safetensors-compatible header plus an extension section (MX block layouts, per
tensor BLAKE3, embedded schema, tokenizer/patcher hash, format version) and an
ed25519-signed manifest. The loader never executes code from files and its
parser is fuzzed.

## 8. Evaluation

Perplexity, synthetic tasks (copy, MQAR, modular arithmetic, PCFG entropy gap),
needle retrieval, exact-match QA, sandboxed code execution, think-depth versus
accuracy curves, halt-head calibration (ECE), and memory-edit locality.

## 9. Testing

T1 unit and gradient checks, T2 cross-implementation parity, T3 planner checks,
T4 convergence, T5 distributed, T6 serving integration, T7 fuzz, T8 miri,
T9 benchmarks, T10 GPU conformance.
