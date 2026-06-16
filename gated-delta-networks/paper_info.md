# Gated Delta Networks: Improving Mamba2 with Delta Rule

- **Authors:** Songlin Yang (MIT CSAIL), Jan Kautz (NVIDIA), Ali Hatamizadeh (NVIDIA)
- **arXiv:** [2412.06464](https://arxiv.org/abs/2412.06464)
- **Local PDF:** [paper.pdf](paper.pdf)
- **Published:** ICLR 2025 camera ready (arXiv v3: Mar 6, 2025; v1: Dec 9, 2024)
- **Code:** [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- **Topics:** Linear RNNs, gated linear attention, delta rule, hybrid architectures, hardware-efficient training

## Abstract

Linear Transformers are efficient alternatives to softmax attention but historically struggle on retrieval and long-context tasks. Two recent mechanisms address parts of this: **gating** (Mamba2) enables rapid memory erasure, and the **delta rule** (DeltaNet) enables targeted key-value updates. The authors observe these are complementary and propose the **gated delta rule** which fuses both. They derive a hardware-efficient chunkwise training algorithm by extending the WY representation to handle the new gating term, then introduce **Gated DeltaNet** plus two hybrid variants (H1: GDN + sliding window attention; H2: Mamba2 + GDN + SWA). At 1.3B params on 100B tokens, Gated DeltaNet beats Mamba2 and DeltaNet on language modeling, common-sense reasoning, in-context retrieval, length extrapolation, and long-context understanding; the hybrids beat Transformer++ on retrieval-heavy tasks.

## Knowledge calibration

| Concept | Level | Treatment |
|---|---|---|
| Softmax attention / QKV notation | solid | quick notation only |
| Linear attention as a recurrent state | new | full depth |
| Mamba2 gating / selective state decay | new | full depth |
| Delta rule / online-learning view | new | full depth |
| WY representation / chunkwise training | new | full depth |
| Sliding-window attention | familiar | quick refresher when hybrid blocks appear |

## Paper Structure

| Section | Topic | Status |
|---------|-------|--------|
| 1 | Introduction | done |
| 2 | Preliminary | |
| 2.1 | Mamba2: Linear Attention with Decay | |
| 2.2 | DeltaNet: Linear Attention with Delta Rule | |
| 3 | Gated Delta Networks | |
| 3.1 | Formulation: the gated delta rule | |
| 3.2 | Case study: Single Needle in a Haystack | |
| 3.3 | Hardware-efficient chunkwise algorithm | |
| 3.4 | Architecture and hybrid models | |
| 4 | Experiments | |
| 5 | Related Work | |
| 6 | Conclusion | |
| App A | Extended WY representation proof | |
| App B | Evaluation details and ablations | |

## Figures to clip into `artifacts/`

- `figure_1_architecture.png` — hybrid architecture and Gated DeltaNet block design
- `figure_2_length_extrapolation.png` — length extrapolation on long benchmarks
- `figure_3_training_throughput.png` — 1.3B model training throughput on a single H100
