# PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

- **Authors:** Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, Shen Li (Meta AI)
- **arXiv:** [2304.11277](https://arxiv.org/abs/2304.11277)
- **Published:** VLDB 2023 (Apr 2023)
- **Code:** [pytorch/torch/distributed/fsdp](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/fully_sharded_data_parallel.py)
- **Topics:** Distributed training, model sharding, ZeRO-style parallelism, PyTorch internals, collective communication, mixed precision

## Abstract

Presents PyTorch Fully Sharded Data Parallel (FSDP) as an industry-grade solution for training large models that don't fit on a single GPU. FSDP shards parameters, gradients, and optimizer states across ranks, then communicates parameters on-demand (via AllGather) for forward/backward computation, freeing them immediately after. Inspired by DeepSpeed ZeRO but co-designed with PyTorch internals (Tensor system, dispatcher, CUDA caching allocator) to provide non-intrusive UX. Introduces deferred initialization (for models too large to materialize), configurable sharding factors (full, hybrid, replication) that map to datacenter topology, communication/computation overlap with prefetching, a rate limiter for the caching allocator, and native sharded mixed precision. Evaluated on up to 512 A100 80GB GPUs across T5-11B, GPT-175B, and a 768B-parameter DHEN recommendation model; achieves ~60% A100 utilization on GPT-175B with near-linear scaling.

## Paper Structure

| Section | Topic | Status |
|---------|-------|--------|
| 1 | Introduction | done |
| 2 | Background | done |
| 2.1 | Model Replication (DDP) | done |
| 2.2 | Model Partitioning (pipeline, RPC) | done |
| 2.3 | Model Sharding | done |
| 3 | System Design | done |
| 3.1 | Model Initialization (deferred init) | done |
| 3.2 | Sharding Strategies (Full, Hybrid, Autograd) | done |
| 3.3 | Communication Optimizations (overlap, prefetch, accumulation) | done |
| 3.4 | Memory Management (caching allocator, rate limiter) | done |
| 4 | Implementation | done |
| 4.1 | Initialization (deferred / GPU / CPU) | done |
| 4.2 | Flat Parameters | done |
| 4.3 | Runtime (autograd hooks) | done |
| 4.4 | Native Mixed Precision | done |
| 5 | Evaluation | |
| 5.1 | Setup | |
| 5.2 | Model Scale (vs DDP) | |
| 5.3 | Throttle Communications | |
| 5.4 | Efficient Training for Large Models | |
| 6 | Related Work | |
| 7 | Discussion (Interoperability, Limitations) | |
| 8 | Conclusion | |
