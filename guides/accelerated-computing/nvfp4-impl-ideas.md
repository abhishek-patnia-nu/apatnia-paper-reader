# NVFP4 Training Stack: Implementation Primer

> A minimal Lightning plugin for NVFP4 training: what it does, why
> a tiny `te.Linear` cast wrapper is needed to keep things working, and
> how the resulting setup differs from the `bf16-mixed` baseline.

**Source code references:**

- Reference implementation: [`explore-bf16-mixed.py`](../../../aic-nuformer-research/explore-bf16-mixed.py)
- [Lightning's `MixedPrecision`](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/plugins/precision/amp.html)
- [TE NVFP4 docs](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
- [TE FP8/FP4 primer notebook](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html)
- [TE NVFP4 PR #2177](https://github.com/NVIDIA/TransformerEngine/pull/2177)
- Paper context: [arxiv:2509.25149](https://arxiv.org/abs/2509.25149) (original NVFP4 pretraining), [arxiv:2604.12374](https://arxiv.org/abs/2604.12374) (Nemotron 3 Super refinement)

---

## Table of Contents

1. [The Plugin](#1-the-plugin)
2. [The TeLinearAutocast Wrapper](#2-the-telinearautocast-wrapper)
3. [How It Differs from `bf16-mixed`](#3-how-it-differs-from-bf16-mixed)
4. [What Linear Layers To Replace [TODO]](#4-what-linear-layers-to-replace-todo)
5. [Distributed Considerations](#5-distributed-considerations)
6. [How It Plugs Into Lightning](#6-how-it-plugs-into-lightning)
7. [Software Requirements and Constraints](#7-software-requirements-and-constraints)
8. [Open Implementation Questions](#8-open-implementation-questions)

---

## 1. The Plugin

A faithful NVFP4 plugin on Lightning is a thin subclass of
`MixedPrecision` with two overrides: `forward_context` enters
`te.autocast` next to `torch.autocast`, and `convert_module` swaps
eligible `nn.Linear` for `TeLinearAutocast`. Everything else (gradient
scaling, optimizer step, clipping, state dict) inherits unchanged.

Reference: [`explore-bf16-mixed.py`](../../../aic-nuformer-research/explore-bf16-mixed.py).

```97:166:../../../aic-nuformer-research/explore-bf16-mixed.py
class NVFP4MixedPrecision(MixedPrecision):
    """Lightning's ``bf16-mixed`` (or ``16-mixed``) plugin with two NVFP4
    bells-and-whistles bolted on:
    ...
    """

    def __init__(
        self,
        *args: Any,
        replace_linear: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.recipe = NVFP4BlockScaling()
        self.replace_linear = replace_linear

    @override
    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        with self.autocast_context_manager(), te.autocast(enabled=True, recipe=self.recipe):
            yield

    @override
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        # ... walks tree, dim-checks against NVFP4_BLOCK_SCALING_SIZE,
        #     swaps eligible nn.Linear for TeLinearAutocast ...
```

What this does:

- **`forward_context`** enters both `torch.autocast` (Lightning's
  `MixedPrecision` default for non-TE ops) and `te.autocast` with the
  NVFP4 recipe. The `te.autocast` is a no-op for `nn.Linear`-only
  models — only `te.Linear` (and our wrapper around it) reads the
  thread-local recipe state.
- **`convert_module`** walks the model and swaps every dim-eligible
  `nn.Linear` for the wrapper. The dim check uses TE's own constant
  `NVFP4_BLOCK_SCALING_SIZE` (=16, imported from
  `transformer_engine.pytorch.constants`) so we don't hardcode it.
- **The swap is a clone-copy** of the FP32 weight/bias (no dtype
  change). The optimizer keeps FP32 master weights and FP32 moments
  for replaced layers exactly as it would for the unreplaced
  `nn.Linear`.

What this deliberately does *not* do (as compared to the more
elaborate plugin variants you'll find in NVIDIA's tutorials):

- **No `weights_dtype` argument.** Parameters stay in whatever dtype
  the model was built in (FP32 by default). FP32 master weights are
  the paper-faithful default; introducing a `weights_dtype` knob lets
  you trade master-weight precision for storage, but obscures the
  apples-to-apples comparison with `bf16-mixed`.
- **No predicate filter** (yet). Every dim-eligible Linear gets
  replaced — see [§4](#4-what-linear-layers-to-replace-todo) for why
  production models will want a structural filter.
- **No dynamic recipe / step tracking.** The recipe is fixed at
  construction. Mid-training NVFP4 → MXFP8 switching (described in
  [arxiv:2509.25149](https://arxiv.org/abs/2509.25149) Appendix D as a
  loss-mitigation trick) is a future extension that would add a step
  hook plus an `active_recipe` property.

### Run modes

Two CLI modes on the reference script:

```bash
uv run python explore-bf16-mixed.py            # plain bf16-mixed (every Linear stays nn.Linear)
uv run python explore-bf16-mixed.py --use-te   # FP4 GEMM via TeLinearAutocast for every dim-eligible Linear
```

That's it. No `--paper-recipe`, no `--weights-dtype`, no
`--switch-step`. Adding any of those is its own follow-on task; for
now the script is purely a dtype-tracing harness so we can verify the
two-mode behavior matches expectations.

---

## 2. The TeLinearAutocast Wrapper

This is the load-bearing detail of the plugin: `te.Linear` does not
do its own input cast on the FP8/FP4 path. Without our wrapper, an
FP32 activation arriving from upstream RMSNorm/LayerNorm (FP32 always
under `bf16-mixed`, since norms are autocast-excluded) trips a hard
assertion deep in TE.

### The crash

```
RuntimeError: RHT is only supported for bfloat16 input
```

Triggers in `transformer_engine.pytorch.module.linear._Linear.forward`
the first time a `te.Linear` runs under
`te.autocast(recipe=NVFP4BlockScaling())` with an FP32 input.

### Why

Two observations from reading TE source:

1. **`torch.autocast` doesn't intercept `te.Linear`.**
   `te.Linear._Linear.forward` is a `torch.autograd.Function` without
   `@torch.amp.custom_fwd`, so the autocast dispatch table never sees
   it. The outer `torch.autocast` set up by `MixedPrecision` casts
   `nn.Linear`, `nn.LayerNorm`, `nn.functional.silu`, etc., but skips
   anything dispatched through a custom `Function`.

2. **`te.Linear`'s FP4/FP8 path skips `cast_if_needed`.** In
   `transformer_engine/pytorch/module/linear.py:226-240`, the non-FP8
   `else` branch calls `cast_if_needed(inp, self.activation_dtype)` to
   align activation dtype with whatever
   `set_activation_dtype` decided. The FP8/FP4 branch doesn't —
   it passes `inp` straight to `input_quantizer`, and once the recipe
   has `random_hadamard_transform=True` (the NVFP4 default for
   activations), the quantizer asserts BF16.

So under `bf16-mixed`:

- `nn.LayerNorm` / `RMSNorm` produce FP32 outputs (norms are
  autocast-excluded for numerical stability — sums over the hidden
  dim need FP32 dynamic range).
- That FP32 output flows into `te.Linear`.
- TE's FP4 quantizer asks RHT for BF16 input, gets FP32, asserts.

### The wrapper

Reference: [`explore-bf16-mixed.py`](../../../aic-nuformer-research/explore-bf16-mixed.py).

```169:209:../../../aic-nuformer-research/explore-bf16-mixed.py
def _autocast_dtype(x: torch.Tensor) -> torch.dtype:
    """Match TE's own ``set_activation_dtype`` logic: if
    ``torch.autocast`` is active, use its dtype; otherwise fall back
    to the input's own dtype."""
    if torch.is_autocast_enabled():
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


class TeLinearAutocast(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.te_linear = te.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        h = cast_if_needed(x, _autocast_dtype(x))
        return self.te_linear(h)
```

Two things to highlight:

- **The cast dtype is dynamic** (`torch.get_autocast_dtype(device)` if
  autocast active, else `x.dtype`). This mirrors TE's own
  `set_activation_dtype` rule (see
  `transformer_engine/pytorch/module/base.py:928-945`). Means the
  wrapper works under `bf16-mixed`, `fp16-mixed`, and no-autocast
  without changes — we don't hardcode `torch.bfloat16`.
- **The cast happens INSIDE the wrapper**, not as a hook on the parent
  module. Localizes the dtype change so upstream and downstream
  tensors keep whatever dtype they had — only the boundary into
  `te.Linear` is forced to the autocast dtype.

This is the minimum cast machinery needed to make NVFP4 training
work under Lightning's stock `bf16-mixed` precision plugin. If TE
ever adds the missing `cast_if_needed` to its FP4 path
(`linear.py:226-240`), the wrapper becomes redundant.

---

## 3. How It Differs from `bf16-mixed`

Same model, same data, same optimizer, same `bf16-mixed` precision
plugin — what changes when you flip `--use-te` on?

### 3.1 Module Construction

| | `bf16-mixed` (`nn.Linear`) | NVFP4 (`TeLinearAutocast`) |
|---|---|---|
| `convert_module` | no-op | walks tree, swaps every dim-eligible `nn.Linear` for the wrapper |
| Param dtype on device | FP32 (model default) | FP32 (model default) |
| Module class for transformer Linears | `nn.Linear` | `TeLinearAutocast` (which holds a `te.Linear`) |

### 3.2 forward_context

```python
# bf16-mixed (Lightning default)
torch.autocast("cuda", dtype=torch.bfloat16)

# NVFP4 (NVFP4MixedPrecision)
torch.autocast("cuda", dtype=torch.bfloat16)
te.autocast(enabled=True, recipe=NVFP4BlockScaling())
```

The outer `torch.autocast` is identical. The added `te.autocast` is
what tells `te.Linear` to actually run an FP4 GEMM rather than
falling through to its non-FP8 path.

### 3.3 At a Linear's Forward

| Stage | `bf16-mixed`, `nn.Linear` | NVFP4, `TeLinearAutocast` |
|---|---|---|
| Cast at op boundary | autocast: `x.to(bf16)`, `w.to(bf16)` | wrapper: `x.to(bf16)`; then TE quantizes `x` (1×16 NVFP4) and `w` (16×16 NVFP4 with RHT) |
| GEMM dtype | **BF16 × BF16 → BF16** on Tensor Cores | **FP4 × FP4 → BF16** on Tensor Cores (E4M3 block scales per block, FP32 global scale post-GEMM) |
| Saved-for-backward | BF16 `x` and `w` views (16 bits/elt) | FP4 rowwise+columnwise + E4M3 scales (~4.5 bits/elt) |

Two asymmetries worth highlighting:

- **Where precision loss happens.** `bf16-mixed`'s drop is a boundary
  cast (FP32 → BF16; near-trivial since BF16 has the same 8-bit
  exponent as FP32). NVFP4 applies an RHT to the weight, then 16×16
  block-amax scaling, then 4-bit quantization with stochastic rounding
  on the gradient path. The numerics machinery is doing real work for
  outlier suppression — that's where the paper's three features (RHT,
  2D block scaling, stochastic rounding) live.
- **What backward saves.** `bf16-mixed` saves BF16 activations
  (16 bits/elt). NVFP4 saves FP4 (~4.5 bits/elt). Activation memory
  drops by ~3.5×, which is where FP4's batch-size headroom comes from.

### 3.4 Backward

| | `bf16-mixed` | NVFP4 |
|---|---|---|
| dgrad GEMM (`grad_out × wᵀ`) | BF16 GEMM | FP4 GEMM using saved FP4 columnwise weight, with stochastic rounding on `grad_out` |
| wgrad GEMM (`xᵀ × grad_out`) | BF16 GEMM | FP4 GEMM using saved FP4 columnwise activation, with RHT on inputs |
| Where `param.grad` lands | FP32 — autograd casts BF16 grad to FP32 at the param boundary | FP32 — same boundary cast |

In both setups the gradient *the optimizer sees* is FP32. The
difference is how that FP32 gradient was computed (BF16 vs FP4 GEMMs).

### 3.5 Optimizer Step

| | `bf16-mixed` | NVFP4 |
|---|---|---|
| Param dtype | FP32 | FP32 |
| Grad dtype | FP32 | FP32 |
| AdamW `m`, `v` state | FP32 | FP32 |
| GradScaler | None (BF16 has FP32-equivalent dynamic range) | None |

**Optimizer-side they're identical.** The clone-copy in
`convert_module` preserves the parameter as FP32, so AdamW sees the
same FP32 master weights and FP32 moments either way.

### 3.6 Memory per Parameter

For one logical parameter, ignoring activations:

| Bucket | `bf16-mixed` | NVFP4 |
|---|---|---|
| Weight on `Module` | 4 B (FP32) | 4 B (FP32) |
| `param.grad` | 4 B (FP32) | 4 B (FP32) |
| AdamW `m` | 4 B (FP32) | 4 B (FP32) |
| AdamW `v` | 4 B (FP32) | 4 B (FP32) |
| **Subtotal: optimizer-related** | **16 B/param** | **16 B/param** |
| TE FP4 row+col cache (forward only) | — | ~1 B/param + small scale tables |
| Saved activations (per step) | BF16 ⇒ 2 B/value | FP4 ⇒ ~0.6 B/value |

Parameters and optimizer state cost the same. The savings come
almost entirely from saved activations, which is why FP4's
batch-size headroom shows up most when `seqlen × hidden` is big (FFN
intermediates).

### 3.7 Benchmarking

For a meaningful tokens/sec or loss comparison: same batch, same seq
length on both modes. Activation-memory savings let NVFP4 fit a
larger batch — separate "what's the speedup at equal batch" from
"what's the batch-size headroom." Reporting `tokens/sec` rather than
`ms/step` makes the two regimes comparable.

---

## 4. What Linear Layers To Replace [TODO]

> **TODO** — the current plugin replaces every dim-eligible
> `nn.Linear`. In production we want to be selective: not every Linear
> benefits from NVFP4, and some shouldn't be quantized at all
> (embeddings, `lm_head`, attention components, the final-N% blocks).

The mechanism will be **name-based predicates** plumbed through
`convert_module`: a callable taking `(dotted_name, layer)` and
returning whether to replace. Easy ones: substring matches
(`"embed"`, `"lm_head"`, `"attention"`); structural ones: block-index
awareness for the "final 15% of blocks" rule.

The baseline recipe will come from two NVFP4 papers:

- [arxiv:2509.25149](https://arxiv.org/abs/2509.25149) §4.1 — the
  original NVFP4 pretraining recipe (Nemotron-Nano-12B, 62 blocks):
  skip embeddings, `lm_head`, attention components, plus first 2 +
  last 8 blocks (16% kept BF16).
- [arxiv:2604.12374](https://arxiv.org/abs/2604.12374) Table 3 —
  Nemotron 3 Super refinement (88 blocks): same family of skips but
  only "Final 15% of Network" (drops the first-N guard, since the
  older paper's own ablation showed it was belt-and-suspenders), plus
  model-specific carve-outs for LatentMoE / MTP / Mamba.

The two papers describe the same **algorithm** (FP4 format, RHT, 2D
scaling, stochastic rounding) — the new paper explicitly defers to
the old one for the quantization scheme — with refined
**layer-selection policy**. Predicate API design TBD.

---

## 5. Distributed Considerations

### 5.1 Amax Synchronization

NVFP4's block-level scale factors (`s_block`) are local — no
cross-rank communication. But the global scale factor (`s_global`)
requires the global amax across the full tensor.

For tensors sharded by tensor or sequence parallelism, TE performs
an all-reduce of amax before quantization:

```
Per-rank local amax  →  all-reduce(max)  →  shared global amax  →  s_global
```

Automatic for `te.Linear` layers configured with TP/SP.

### 5.2 Quantized All-Gather

TE supports all-gather of NVFP4-quantized tensors:

```
High-precision tensor
  → compute amax
  → synchronize amax across ranks
  → compute s_global (identical on all ranks)
  → scale + cast to FP4 (with s_block per block)
  → all-gather the FP4 tensor + scale arrays
```

All ranks must share the same `s_global`, guaranteed by the amax
sync. This avoids gathering in high precision and then quantizing
(which would be more expensive).

### 5.3 Lightning Strategy Interaction

- **FSDP sharding:** Parameters are sharded in FP32. TE quantizes to
  FP4 inside `te.Linear.forward()` on the already-gathered parameters
  — FSDP's all-gather happens before quantization.
- **DDP gradient sync:** Gradients are accumulated in FP32. TE's FP4
  is only used for forward/backward GEMMs, not for gradient
  communication.
- **Tensor parallelism:** Handled by TE's own parallel linear layers
  (`te.ColumnParallelLinear`, `te.RowParallelLinear`), which integrate
  amax sync internally. Using these requires building the model with
  TE parallel layers rather than relying on `convert_module`. Out of
  scope for the current toy plugin.

---

## 6. How It Plugs Into Lightning

This section covers the underlying mechanism — how the precision
plugin interface works, when each method gets called, and how
`te.autocast` activates FP4 inside `te.Linear`.

### 6.1 The Precision Plugin Interface

Lightning splits precision handling into a framework-agnostic core
(Fabric) and a trainer-aware wrapper (PyTorch Lightning):

```
lightning.fabric.plugins.precision.Precision       (core interface)
    │
    ├── forward_context()          → context manager wrapping forward
    ├── convert_module(module)     → one-time model surgery
    ├── tensor_init_context()      → default dtype for new tensors
    └── module_init_context()      → intercept nn.Module construction
          │
lightning.pytorch.plugins.precision.amp.MixedPrecision
    │
    ├── pre_backward / backward / post_backward
    ├── optimizer_step(optimizer, model, closure)
    └── {train,val,test,predict}_step_context() → all delegate to forward_context()
```

`forward_context()` is the primary extension point. Every training and
eval step runs inside the context manager it returns. For FP8/FP4,
this is where `te.autocast` is activated.

`convert_module(module)` runs once during `trainer.fit()` setup,
before training starts:

```
trainer.fit()
  └─ _call_setup_hook()
       └─ strategy.setup()
            └─ precision.convert_module(model)   # nn.Linear → TeLinearAutocast
            └─ move model to device
```

This is the right place for the swap — the model exists as
`nn.Linear`s up to this point, and after the swap they're
`TeLinearAutocast` for the rest of the run.

We deliberately don't override `module_init_context`: with no
predicate filter, the simple `convert_module` pass over
`model.named_modules()` is enough. (Stock `TransformerEnginePrecision`
overrides `module_init_context` to monkey-patch `nn.Linear` →
`te.Linear` globally during model construction — useful as a fast
path, but it bypasses any predicate filter and complicates the
mental model. We pay the one-time `convert_module` walk and skip the
indirection.)

### 6.2 Trainer Loop Hook Order

For a single training step:

```
trainer.fit()
  └─ training_step loop:
       1. precision.train_step_context().__enter__()  # activates forward_context()
       2. model.training_step(batch)                  # user code runs here
       3. precision.pre_backward(loss, model)
       4. precision.backward(loss, model, optimizer)  # loss.backward()
       5. precision.post_backward(loss, model)
       6. precision.optimizer_step(optimizer, model, closure)
       7. precision.train_step_context().__exit__()
```

Critical detail: **the backward pass runs outside `forward_context()`.**
TE handles this correctly because the autograd backward graph was
built during forward inside the context — TE layers record the
recipe and apply it during backward automatically.

### 6.3 `te.autocast` and `te.Linear`

`te.autocast` (despite the legacy name `te.fp8_autocast`, it also
handles FP4) sets a thread-local recipe; `te.Linear.forward()` checks
this thread-local to decide whether to quantize:

```python
# Simplified TE pseudocode
class Linear(torch.nn.Module):
    def forward(self, inp):
        recipe = get_current_recipe()
        if recipe is None:
            return F.linear(inp, self.weight, self.bias)   # standard path

        # Quantize weight + activation per recipe
        w_fp4_row, w_fp4_col, w_scales = nvfp4_quantize_2d(self.weight)
        x_fp4_row, x_fp4_col, x_scales = nvfp4_quantize_1d(inp)

        # FP4 GEMM on Tensor Cores → BF16/FP32 output
        out = nvfp4_gemm(x_fp4_row, w_fp4_col, x_scales, w_scales)
        ctx.save_for_backward(x_fp4_col, w_fp4_row, ...)
        return out
```

Backward computes two GEMMs (dgrad and wgrad), uses the saved
rowwise/columnwise quantized copies, applies stochastic rounding when
quantizing `grad_output` to FP4, and applies RHT to wgrad inputs.

**Nested autocasts stack — innermost recipe wins.** Useful if a
model wants to override the plugin's recipe for a specific block
range.

### 6.4 The `NVFP4BlockScaling` Recipe

```python
from transformer_engine.common.recipe import NVFP4BlockScaling

# All defaults — enables the full paper methodology
recipe = NVFP4BlockScaling()

# Selective feature control
recipe = NVFP4BlockScaling(
    disable_rht=False,                  # Random Hadamard Transform on Wgrad
    disable_2d_quantization=False,      # 2D weight scaling (16×16 blocks)
    disable_stochastic_rounding=False,  # Stochastic rounding on gradients
)
```

For one tensor being quantized to NVFP4:

```
Input tensor (FP32/BF16)
    │
    ├─ [If RHT enabled] Reshape to tiles of (m·k/16, 16)
    │   Multiply by cached 16×16 Hadamard matrix; reshape back
    │
    ├─ Compute global amax = max(|x|) across the full tensor
    │   [Distributed: all-reduce amax across TP/SP ranks (§5.1)]
    │
    ├─ s_global = global_amax / (448.0 · 6.0)   [FP32]
    │
    ├─ For each block of 16 contiguous elements:
    │   ├─ block_amax = max(|x_scaled_i|)
    │   ├─ s_block = round_to_e4m3(block_amax / 6.0)
    │   └─ x_fp4_i = quantize_e2m1(x_scaled_i / s_block)
    │       [stochastic rounding for gradients, nearest-even for forward]
    │
    └─ Output: x_fp4 + s_block array + s_global scalar
```

For 2D weight quantization the same flow applies but blocks are
16×16 instead of 1×16. NVFP4 GEMMs only support the **TN layout**
(both operands transposed for the non-reduction dimension), which is
why TE stores both rowwise and columnwise quantized copies.

Other recipe classes you can pass to the same plugin:

```python
from transformer_engine.common.recipe import (
    DelayedScaling,        # FP8 per-tensor delayed scaling (Hopper+)
    Float8CurrentScaling,  # FP8 per-tensor just-in-time scaling
    Float8BlockScaling,    # FP8 block scaling (DeepSeek-v3 style)
    MXFP8BlockScaling,     # MXFP8 OCP microscaling (Blackwell)
    NVFP4BlockScaling,     # NVFP4 (Blackwell, SM 10.0+)
)
```

---

## 7. Software Requirements and Constraints

| Component | Minimum Version | Notes |
|-----------|----------------|-------|
| GPU | Blackwell (SM 10.0) | GB200, GB300. Not available on Hopper. |
| Transformer Engine | ≥ 2.8.0 | NVFP4 recipe merged via PR #2177 |
| PyTorch | Per TE compat matrix | Must have Blackwell kernel support |
| Lightning | 2.6+ | `MixedPrecision` plugin used as the parent class |
| CUDA toolkit | 12.8+ | Blackwell arch support |

### Dimension Constraints

- `te.Linear` for NVFP4 requires both `in_features` and
  `out_features` divisible by **`NVFP4_BLOCK_SCALING_SIZE`** (= 16,
  the block size). The plugin's `convert_module` honors this by
  importing the constant from `transformer_engine.pytorch.constants`
  and skipping any Linear that doesn't satisfy it.
- Scale tensor padding: first dim to multiple of 128, second dim to
  multiple of 4.
- Sequence length should be a multiple of 16 for efficient tiling.

### Memory Overhead

NVFP4 stores both rowwise and columnwise quantized copies of each
tensor (TN-layout GEMMs in forward and backward). This doubles
quantized-tensor memory compared to a single copy, but FP4 is so
small that the total is well below the BF16 baseline.

Effective storage per value: ~4.5 bits (4 bits data + 0.5 bits
amortized scale overhead).

---

## 8. Open Implementation Questions

1. **Layer-replacement recipe.** TBD; the current plugin replaces
   every dim-eligible `nn.Linear`. Baseline policy will come from
   [§4](#4-what-linear-layers-to-replace-todo) once we wire in the
   predicate layer.

2. **LayerNorm replacement.** §1 deliberately leaves `nn.LayerNorm`
   alone. If profiling shows the LN kernel matters, the right fix is
   `te.LayerNormLinear` fusion at model build time rather than
   independent LN replacement (which only buys a slightly faster norm
   kernel).

3. **Monitoring quantization error.** TE doesn't expose per-layer
   quantization error metrics out of the box. For debugging
   convergence issues, we may need hooks that compare pre-quantization
   and dequantized values.
