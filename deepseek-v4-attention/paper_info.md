# DeepSeek-V4: Hybrid Attention (CSA + HCA)

> **Scoped reading.** This folder covers only the **attention architecture** of the
> DeepSeek-V4 technical report (§2.3), plus a prerequisite lineage section. The full
> report (58 pages) also covers DeepSeekMoE, the Muon optimizer, FP4 training, and
> OPD distillation — those are out of scope here.

- **Authors:** DeepSeek-AI
- **Type:** Technical report (not arXiv-indexed; hosted on the Hub)
- **Source:** [DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
- **Reference impl:** [DeepSeek-V4-Pro/tree/main/inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference)
- **Published:** 2026
- **Topics:** Long-context efficiency, sparse attention, KV compression, attention architectures

## Abstract (attention focus)

DeepSeek-V4 targets one-million-token context. Its headline efficiency lever is a
**hybrid attention architecture** that interleaves two designs: **Compressed Sparse
Attention (CSA)** — compress every `m` tokens' KV into one entry, then apply
**DeepSeek Sparse Attention (DSA)** top-`k` selection over the compressed entries — and
**Heavily Compressed Attention (HCA)** — a much higher compression rate (`m' ≫ m`) with
no sparse selection. Both add a sliding-window branch, attention sink, partial RoPE,
shared-KV MQA, and grouped output projection. CSA is a *composition* of prior ideas
(MLA-style latent compression, NSA-style compress/select/local structure, the DSA
lightning indexer); the novelty is selecting over *compressed* blocks and the CSA/HCA
hybrid.

## Knowledge calibration (from questionnaire)

| Concept | Level | Treatment |
|---|---|---|
| MHA | solid | skip — **but KV cache mechanics need detail** |
| MQA / GQA | familiar | quick refresher |
| MLA | new | full depth |
| NSA | new | full depth |
| DSA + Lightning Indexer | new | full depth |
| RoPE / partial RoPE | familiar | quick refresher |
| Sliding window + attention sink | new | full depth |
| Sparse / top-k attention | new | full depth |

## Reading path

CSA is unintelligible without its lineage, so we read prerequisites first (per the
agreed plan: MLA → NSA → DSA), then the report's actual §2.3.

## Paper Structure

| Section | Topic | Status |
|---------|-------|--------|
| 0.1 | Prereq: MLA — latent KV compression (DeepSeek-V2) | done |
| 0.2 | Prereq: NSA — native sparse attention (compress + select + local) | |
| 0.3 | Prereq: DSA + Lightning Indexer (DeepSeek-V3.2-Exp) | |
| 2.3 | Hybrid Attention overview (why CSA + HCA, the bottleneck) | |
| 2.3.1 | Compressed Sparse Attention (CSA) | |
| 2.3.2 | Heavily Compressed Attention (HCA) | |
| 2.3.3 | Other details (norm, partial RoPE, sliding window, attention sink) | |
| 2.3.4 | Efficiency discussion | |
| App | KV-cache layout & sparse-attention kernel co-design (Fig. 6) — optional | |

## Figures to clip into `artifacts/`

- `figure_2_architecture.png` — overall DeepSeek-V4 architecture (CSA/HCA placement)
- `figure_3_csa.png` — CSA core architecture (compressor + lightning indexer + MQA)
- `figure_4_hca.png` — HCA core architecture (heavy compression, no sparsity)
- `figure_6_kv_cache_layout.png` — KV cache layout (optional appendix section)
