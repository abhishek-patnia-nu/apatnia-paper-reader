# Ring Attention with Blockwise Transformers for Near-Infinite Context

- **Authors:** Hao Liu, Matei Zaharia, Pieter Abbeel (UC Berkeley)
- **arXiv:** [2310.01889](https://arxiv.org/abs/2310.01889)
- **Published:** ICLR 2024 (Oct 2023)
- **Code:** [lhao499/llm_large_context](https://github.com/lhao499/llm_large_context) (JAX reference); PyTorch production version lives in [`torch.distributed.tensor.experimental`](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/tensor/experimental/_context_parallel/_attention.py)
- **Topics:** Context parallelism, sequence sharding, ring attention, blockwise attention, communication/computation overlap, long-context training

## Why we're reading this

Foundational paper for **Context Parallelism (CP)** — the technique we want to add to the
aic-nuformer-research training stack (Lightning + FSDP). Nearly every CP implementation
(PyTorch native CP, torchtitan, ring-flash-attn, Megatron CP) is built on this algorithm.
First paper in the CP reading sequence: Ring Attention → Striped Attention → USP →
Llama 3 §3.3 → torch/torchtitan source.

## Abstract

Transformer activation memory scales with sequence length, capping context size at what one
device can hold. Ring Attention shards the *sequence* across devices: each device keeps one
query block and computes blockwise attention while key-value blocks rotate around a ring of
hosts (send to next, receive from previous). Because blockwise attention is permutation-
invariant in the KV blocks (online-softmax rescaling makes any order correct), the rotation
is exact — no approximation. When per-block compute time exceeds block transfer time, the
communication hides entirely behind computation, giving zero-overhead context scaling: max
context grows linearly with device count. Demonstrated up to 100M-token training contexts on
TPU pods.

## Calibration (questionnaire, 2026-06-12)

| Concept | Level | Treatment in notes |
|---------|-------|--------------------|
| Online (incremental) softmax | new | deep background with worked code (§2) |
| FlashAttention / tiled attention | new | deep background with worked code (§2) |
| Activation memory accounting | familiar | light refresh, formulas only |
| Collectives vs p2p/ring communication | familiar (solid on FSDP collectives) | explain p2p/ring, skip all-gather/reduce-scatter basics |
| Compute/comm overlap, arithmetic intensity | new | full walkthrough of the block-size condition (§3) |
| Prior sequence parallelism (Megatron SP, RSA) | new | background where the paper contrasts (§1, §6) |

## Paper Structure

| Section | Topic | Status |
|---------|-------|--------|
| 1 | Introduction | done |
| 2 | Large Context Memory Constraint (activation memory math, BPT) | |
| 3 | Ring Attention (the algorithm: ring rotation, arithmetic intensity, memory) | |
| 4 | Setting (experimental setup) | |
| 5 | Results (max context, MFU, RL, LLM) | |
| 6 | Related Work | |
| 7 | Conclusion | |
| Appendix A | JAX code | |
| Appendix C | Inference requirement | |
| Appendix D | Training FLOPs scaling | |

Sections 2–3 are the core (the memory argument and the algorithm). Sections 4–5 are
TPU-era experiments — worth a skim for the MFU trade-off, not deep notes. Appendix A is
worth reading alongside §3.
