# Pretraining Large Language Models with NVFP4

- **Authors:** NVIDIA (80+ contributors)
- **arXiv:** [2509.25149](https://arxiv.org/abs/2509.25149)
- **Published:** Sep 2025 (v2: Mar 2026)
- **Topics:** Low-precision training, FP4 quantization, LLM pretraining

## Abstract

Introduces a method for stable LLM pretraining using NVFP4 (4-bit floating point). Key techniques: Random Hadamard transforms, 2D quantization, stochastic rounding, selective high-precision layers. Validated on a 12B-parameter model trained on 10T tokens -- matching FP8 baseline accuracy.

## Paper Structure

| Section | Topic | Status |
|---------|-------|--------|
| 1 | Introduction | done |
| 2 | NVFP4 Format | done |
| 3 | Training with NVFP4 (Results) | done |
| 4 | Training Methodology | done |
| 4.1 | Mixed precision | done |
| 4.2 | Random Hadamard Transforms | done |
| 4.3 | 2D scaling | done |
| 4.4 | Stochastic rounding | done |
| 5 | NVFP4 vs MXFP4 | |
| 6 | Conclusions | |
| App A-E | Models, Quantization, Ablations | |
