# NVFP4 Training Stack: Implementation Primer

> How PyTorch Lightning precision plugins work, how Transformer Engine
> implements NVFP4 under the hood, and what we need to build to experiment
> with FP4 training.

**Source code references:**

- [Fabric `Precision` base](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/plugins/precision/precision.html)
- [Fabric `TransformerEnginePrecision`](https://lightning.ai/docs/fabric/stable/_modules/lightning/fabric/plugins/precision/transformer_engine.html) — where all the real logic lives
- [PyTorch `TransformerEnginePrecision`](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/plugins/precision/transformer_engine.html) — thin trainer-aware wrapper
- [TE NVFP4 docs](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
- [TE FP8/FP4 primer notebook](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html)
- [TE NVFP4 PR #2177](https://github.com/NVIDIA/TransformerEngine/pull/2177)
- Paper context: [arXiv 2509.25149](https://arxiv.org/abs/2509.25149)

---

## Table of Contents

1. [Lightning Precision Plugin Architecture](#1-lightning-precision-plugin-architecture)
2. [Annotated Source: TransformerEnginePrecision](#2-annotated-source-transformerengineprecision)
3. [How te.autocast and Recipes Wire Into te.Linear](#3-how-teautocast-and-recipes-wire-into-telinear)
4. [Transformer Engine's NVFP4BlockScaling Recipe](#4-transformer-engines-nvfp4blockscaling-recipe)
5. [Integration Patterns for NVFP4 Training](#5-integration-patterns-for-nvfp4-training)
6. [Distributed Considerations](#6-distributed-considerations)
7. [Software Requirements and Constraints](#7-software-requirements-and-constraints)
8. [Open Implementation Questions](#8-open-implementation-questions)

---

## 1. Lightning Precision Plugin Architecture

### 1.1 Two-Layer Design

Lightning splits precision handling into a framework-agnostic core (Fabric)
and a trainer-aware wrapper (PyTorch Lightning):

```
lightning.fabric.plugins.precision.Precision       (core interface)
    │
    ├── forward_context()          → context manager wrapping forward passes
    ├── convert_module(module)     → one-time model surgery
    ├── convert_input(data)        → cast inputs to working dtype
    ├── convert_output(data)       → cast outputs back to default dtype
    ├── tensor_init_context()      → sets default dtype for tensor allocation
    └── module_init_context()      → intercepts nn.Module construction
          │
lightning.pytorch.plugins.precision.Precision(FabricPrecision, CheckpointHooks)
    │
    ├── connect(model, optimizers, lr_schedulers)  → wire into trainer
    ├── pre_backward(tensor, module)               → fires on_before_backward
    ├── backward(tensor, model, optimizer)          → calls model.backward()
    ├── post_backward(tensor, module)               → fires on_after_backward, detaches loss
    ├── optimizer_step(optimizer, model, closure)   → wraps closure + gradient clipping
    ├── _clip_gradients(...)                        → delegates to configure_gradient_clipping
    └── {train,val,test,predict}_step_context()    → all delegate to forward_context()
```

The key insight: **`forward_context()` is the primary extension point.** Every
training/eval step runs inside whatever context manager it returns. For FP8/FP4,
this is where autocast gets activated.

### 1.2 Trainer Integration Points

The trainer calls precision plugin methods at specific points in the loop. Here
is the call sequence for a single training step:

```
trainer.fit()
  └─ training_step loop:
       1. precision.train_step_context().__enter__()     # activates forward_context()
       2. model.training_step(batch)                     # user code runs here
       3. precision.pre_backward(loss, model)            # fires callbacks
       4. precision.backward(loss, model, optimizer)     # loss.backward()
       5. precision.post_backward(loss, model)           # detaches, fires callbacks
       6. precision.optimizer_step(optimizer, model, closure)
            └─ closure runs steps 2-5 internally
            └─ _after_closure fires on_before_optimizer_step
            └─ _clip_gradients runs
            └─ optimizer.step()
       7. precision.train_step_context().__exit__()
```

### 1.3 How `convert_module` Gets Called

`convert_module` runs once during `trainer.fit()` setup, before training starts:

```
Trainer._run()
  └─ _call_setup_hook()
       └─ strategy.setup()
            └─ precision.convert_module(model)   # layer replacement happens here
            └─ move model to device
```

This is the right place for replacing `nn.Linear` with `te.Linear`.

### 1.4 How `module_init_context` Gets Called

If the model is constructed **inside** Fabric's `init_module` context, the
plugin's `module_init_context()` is active during construction. This allows
intercepting `nn.Linear(...)` calls before they even create parameters:

```python
with fabric.init_module():
    model = MyModel()  # nn.Linear calls get redirected to te.Linear
```

In the Lightning Trainer path, this happens if you use
`Trainer(plugins=[...])` with a `LightningModule` that builds layers in
`__init__`.

---

## 2. Annotated Source: TransformerEnginePrecision

The existing FP8 plugin is the template for any NVFP4 work. Below is the
complete Fabric-layer source, annotated.

### 2.1 Class and __init__

```python
class TransformerEnginePrecision(Precision):
    precision: Literal["transformer-engine", ...] = "transformer-engine"

    def __init__(self, *, weights_dtype, recipe=None, replace_layers=None,
                 fallback_compute_dtype=None):
        # Gate on TE availability
        if not _TRANSFORMER_ENGINE_AVAILABLE:
            raise ModuleNotFoundError(str(_TRANSFORMER_ENGINE_AVAILABLE))

        from transformer_engine.common.recipe import DelayedScaling

        # Recipe normalization: accept dict or dataclass
        if recipe is None:
            recipe = DelayedScaling()          # <-- default: FP8 delayed scaling
        elif isinstance(recipe, Mapping):
            recipe = dict(recipe)
            if "fp8_format" in recipe:
                from transformer_engine.common.recipe import Format
                recipe["fp8_format"] = getattr(Format, recipe["fp8_format"])
            recipe = DelayedScaling(**recipe)

        self.weights_dtype = weights_dtype
        self.recipe = recipe                   # <-- stored, passed to te.fp8_autocast
        self.replace_layers = replace_layers
        self.fallback_compute_dtype = fallback_compute_dtype or weights_dtype
```

**Key detail:** The `recipe` parameter accepts *any* TE recipe object. The
code defaults to `DelayedScaling` but nothing prevents passing
`NVFP4BlockScaling()` — the `__init__` only does `isinstance(recipe, Mapping)`
to handle dict configs. If you pass a pre-built recipe dataclass, it goes
through unchanged.

**However:** The `isinstance(recipe, Mapping)` → `DelayedScaling(**recipe)`
path hardcodes `DelayedScaling`. Passing an NVFP4 recipe as a dict would fail
here. You must pass the recipe as a pre-constructed object, not a dict.

### 2.2 forward_context — The Heart of the Plugin

```python
def forward_context(self):
    dtype_ctx = _DtypeContextManager(self.weights_dtype)
    fallback_autocast_ctx = torch.autocast(
        device_type="cuda", dtype=self.fallback_compute_dtype
    )
    import transformer_engine.pytorch as te
    autocast_ctx = te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)

    stack = ExitStack()
    stack.enter_context(dtype_ctx)                # Layer 1: default dtype
    stack.enter_context(fallback_autocast_ctx)    # Layer 2: torch mixed precision
    stack.enter_context(autocast_ctx)             # Layer 3: TE FP8/FP4
    return stack
```

Three nested context managers, from outermost to innermost:

1. **`_DtypeContextManager`** — sets `torch.get_default_dtype()` to
   `weights_dtype` (e.g., bfloat16). Ensures new tensor allocations use the
   right type.

2. **`torch.autocast("cuda", dtype=...)`** — standard PyTorch AMP for
   operations that don't have TE equivalents (attention, norms, etc.). Acts as
   the fallback.

3. **`te.fp8_autocast(enabled=True, fp8_recipe=self.recipe)`** — the TE
   context. Any `te.Linear`, `te.LayerNorm` etc. inside this context will
   execute their GEMMs using the recipe's precision format. For
   `NVFP4BlockScaling`, this means FP4 GEMMs with 2D weight scaling,
   stochastic rounding, and RHT.

**Important behavior:** `te.fp8_autocast` (despite the name) also handles FP4.
The recipe object determines the actual format. The naming is a TE legacy.

**Important constraint for backward:** The backward pass must execute
**outside** the autocast context. Lightning handles this correctly because
`backward()` is called as a separate method, not inside `forward_context()`.
But the actual autograd backward graph was built during forward inside the
context, so TE layers record the recipe and apply it during backward
automatically.

### 2.3 convert_module — Layer Replacement

```python
def convert_module(self, module):
    # Skip if model already has TE layers (user built the model with TE)
    if any("transformer_engine.pytorch" in m.__module__ for m in module.modules()):
        if self.replace_layers is True:
            rank_zero_info("...Skipping")
    elif self.replace_layers in (None, True):
        _convert_layers(module)          # <-- recursive replacement
    module = module.to(dtype=self.weights_dtype)
    return module
```

The `_convert_layers` helper walks the module tree recursively:

```python
def _convert_layers(module):
    import transformer_engine.pytorch as te

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            # Dimension check: TE requires in_features % 8 == 0, out_features % 16 == 0
            if child.in_features % 8 != 0 or child.out_features % 16 != 0:
                rank_zero_warn("...skipping layer {name}")
                continue
            has_bias = child.bias is not None
            replacement = te.Linear(child.in_features, child.out_features, bias=has_bias)
            replacement.weight.data = child.weight.data.clone()
            if has_bias:
                replacement.bias.data = child.bias.data.clone()
            module.__setattr__(name, replacement)

        elif isinstance(child, torch.nn.LayerNorm):
            replacement = te.LayerNorm(child.normalized_shape[0], eps=child.eps)
            replacement.weight.data = child.weight.data.clone()
            if child.bias is not None and replacement.bias is not None:
                replacement.bias.data = child.bias.data.clone()
            module.__setattr__(name, replacement)
        else:
            _convert_layers(child)     # recurse into submodules
```

**What this means for NVFP4:**
- `te.Linear` is the same class for FP8 and FP4 — the recipe determines
  which precision format is used at runtime.
- Layer replacement is independent of the recipe. You can replace layers once
  and swap recipes later.
- The dimension constraints (divisible by 8/16) apply to FP8. NVFP4 requires
  dimensions divisible by 16 (block size). TE handles padding internally, but
  misaligned layers may not get FP4 acceleration.

### 2.4 module_init_context — Build-Time Interception

```python
def module_init_context(self):
    dtype_ctx = self.tensor_init_context()
    stack = ExitStack()
    if self.replace_layers:
        import transformer_engine.pytorch as te
        context_manager = _ClassReplacementContextManager({
            "torch.nn.Linear": te.Linear,
            "torch.nn.LayerNorm": te.LayerNorm,
        })
        stack.enter_context(context_manager)
    stack.enter_context(dtype_ctx)
    return stack
```

`_ClassReplacementContextManager` monkey-patches `torch.nn.Linear` with
`te.Linear` for the duration of the context. Any model code that does
`nn.Linear(...)` will actually construct a `te.Linear`. This is the
alternative to post-hoc `convert_module` — it avoids the clone-and-replace
overhead.

### 2.5 convert_input / convert_output

```python
def convert_input(self, data):
    return apply_to_collection(
        data, function=_convert_fp_tensor, dtype=Tensor, dst_type=self.weights_dtype
    )

def convert_output(self, data):
    return apply_to_collection(
        data, function=_convert_fp_tensor, dtype=Tensor, dst_type=torch.get_default_dtype()
    )
```

These recursively walk dataclasses/dicts/tuples and cast floating-point tensors.
Inputs go to `weights_dtype` (e.g. bf16); outputs go back to default (e.g. f32).

### 2.6 The PyTorch-Layer Wrapper (Essentially Empty)

```python
# lightning/pytorch/plugins/precision/transformer_engine.py
from lightning.fabric.plugins.precision.transformer_engine import (
    TransformerEnginePrecision as FabricTEPrecision,
)
from lightning.pytorch.plugins.precision.precision import Precision

class TransformerEnginePrecision(Precision, FabricTEPrecision):
    """Plugin for training with fp8 precision via nvidia's Transformer Engine."""
    pass  # body is just the docstring
```

Python MRO composes `Precision`'s trainer hooks (backward, optimizer_step,
etc.) with `FabricTEPrecision`'s core logic (forward_context, convert_module,
etc.). No code needed in the subclass — it's pure mixin composition.

---

## 3. How te.autocast and Recipes Wire Into te.Linear

Understanding the TE internals is important for debugging and for knowing what
the plugin's `forward_context()` actually triggers.

### 3.1 te.autocast (Context Manager)

```python
# Simplified TE pseudocode
@contextmanager
def autocast(enabled=True, recipe=None):
    prev = get_current_recipe()
    set_current_recipe(recipe if enabled else None)
    try:
        yield
    finally:
        set_current_recipe(prev)
```

`te.autocast` sets a thread-local recipe. `te.Linear.forward()` checks this
thread-local to decide whether to quantize its operands.

### 3.2 te.Linear Forward Pass (Simplified)

```python
class Linear(torch.nn.Module):
    def forward(self, inp):
        recipe = get_current_recipe()
        if recipe is None:
            return F.linear(inp, self.weight, self.bias)   # standard path

        # Quantize weight based on recipe type
        if isinstance(recipe, NVFP4BlockScaling):
            # Quantize weight with 2D block scaling (16x16) → NVFP4
            # Quantize activation with 1D block scaling (1x16) → NVFP4
            # Both rowwise and columnwise copies are created for backward
            w_fp4_row, w_fp4_col, w_scales = nvfp4_quantize_2d(self.weight)
            x_fp4_row, x_fp4_col, x_scales = nvfp4_quantize_1d(inp)
        elif isinstance(recipe, DelayedScaling):
            # FP8 path with delayed per-tensor scaling
            ...

        # GEMM in low precision → output in BF16/FP32
        out = nvfp4_gemm(x_fp4_row, w_fp4_col, x_scales, w_scales)

        # Save tensors for backward (grad computation needs the columnwise copies)
        ctx.save_for_backward(x_fp4_col, w_fp4_row, ...)
        return out
```

### 3.3 te.Linear Backward Pass

The backward computes three GEMMs:
- **Dgrad:** activation gradients = `grad_output × weight^T` (uses saved
  `w_fp4_row` transposed)
- **Wgrad:** weight gradients = `activation^T × grad_output` (uses saved
  `x_fp4_col` transposed)

For NVFP4, the backward additionally:
1. Applies **stochastic rounding** when quantizing `grad_output` to FP4
2. Applies **Random Hadamard Transform** to Wgrad inputs before quantization
3. Uses the saved rowwise/columnwise copies to avoid re-quantizing (the 2D
   weight scaling ensures forward and backward see the same quantized weights)

### 3.4 Nested Autocast for Mixed Recipes

TE autocasts are stackable. The innermost active recipe wins:

```python
with te.autocast(recipe=NVFP4BlockScaling()):     # default for whole model
    x = layer_0(x)   # runs in NVFP4
    x = layer_1(x)   # runs in NVFP4
    ...
    with te.autocast(recipe=MXFP8BlockScaling()):  # override for final layers
        x = layer_N(x)   # runs in MXFP8
```

This is how TE implements the "keep last N layers in higher precision" pattern
without any per-layer annotation or special plugin logic. The model's
`forward()` method just wraps the sensitive layers in a second autocast.

---

## 4. Transformer Engine's NVFP4BlockScaling Recipe

### 4.1 Recipe API

```python
from transformer_engine.common.recipe import NVFP4BlockScaling

# All defaults — enables the full paper methodology
recipe = NVFP4BlockScaling()

# Selective feature control
recipe = NVFP4BlockScaling(
    disable_rht=False,                  # Random Hadamard Transform on Wgrad
    disable_2d_quantization=False,      # 2D weight scaling (16x16 blocks)
    disable_stochastic_rounding=False,  # Stochastic rounding on gradients
)
```

Defaults correspond exactly to the paper's recommendations. All three features
are enabled by default.

### 4.2 Other Recipe Classes for Reference

```python
from transformer_engine.common.recipe import (
    DelayedScaling,        # FP8 per-tensor delayed scaling (Hopper+)
    Float8CurrentScaling,  # FP8 per-tensor just-in-time scaling
    Float8BlockScaling,    # FP8 block scaling (DeepSeek-v3 style)
    MXFP8BlockScaling,     # MXFP8 OCP microscaling (Blackwell)
    NVFP4BlockScaling,     # NVFP4 (Blackwell, SM 10.0+)
)
```

All of these are valid `recipe` arguments to both `te.autocast` and
Lightning's `TransformerEnginePrecision(recipe=...)`.

### 4.3 NVFP4 Quantization Flow Inside TE

For a single tensor being quantized to NVFP4:

```
Input tensor (BF16/FP32)
    │
    ├─ [If RHT enabled] Reshape to tiles of (m*k/16, 16)
    │   Multiply by cached 16x16 Hadamard matrix H = (1/√16) · S · H_16
    │   Reshape back to (m, k)
    │
    ├─ Compute global amax = max(|x_i|) over entire tensor
    │   [Distributed: all-reduce amax across TP/SP ranks]
    │
    ├─ Compute s_global = global_amax / (448.0 * 6.0)   [FP32]
    │
    ├─ Scale tensor: x_scaled = x / s_global
    │
    ├─ For each block of 16 contiguous elements:
    │   ├─ block_amax = max(|x_scaled_i|) within block
    │   ├─ s_block = round_to_e4m3(block_amax / 6.0)    [E4M3]
    │   └─ x_fp4_i = quantize_e2m1(x_scaled_i / s_block)
    │       [stochastic rounding if gradient, nearest-even otherwise]
    │
    └─ Output: x_fp4 tensor + s_block array + s_global scalar
```

For **2D weight quantization**, the same flow applies but blocks are 16x16
(16 along input channels × 16 along output channels) instead of 1x16. The
resulting quantized tensor and its transpose share the same values.

### 4.4 NVFP4 GEMM Execution on Tensor Cores

```
Tensor Core reads:
  - FP4 operand tiles (A, B)
  - E4M3 block scale factors for each tile
  
Computes per-block:
  partial = s_block_A * s_block_B * Σ(a_k * b_k)   [k over block of 16]
  
Accumulates partials in FP32 across all blocks

After GEMM: multiply output by s_global_A * s_global_B
→ Final result in BF16/FP32
```

NVFP4 GEMMs only support the **TN layout** (both operands transposed for
the non-reduction dimension). TE handles this by storing both rowwise and
columnwise quantized copies of each tensor.

### 4.5 Scale Factor Storage Layout

```
Rowwise:  data shape [A, B],  scales shape [A, ceil(B/16)]
Colwise:  data shape [B, A],  scales shape [B, ceil(A/16)]   (transposed storage)

Both scale tensors are padded:
  - 1st dim → round up to multiple of 128
  - 2nd dim → round up to multiple of 4
```

Scale factors are stored as E4M3 and must be **swizzled** before GEMM
(hardware-specific memory layout). TE's quantization kernels handle this
automatically.

---

## 5. Integration Patterns for NVFP4 Training

### 5.1 Minimal: Use Existing Lightning Plugin As-Is

The current `TransformerEnginePrecision` already works with NVFP4 — just
pass the recipe:

```python
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.precision import TransformerEnginePrecision
from transformer_engine.common.recipe import NVFP4BlockScaling

plugin = TransformerEnginePrecision(
    weights_dtype=torch.bfloat16,
    recipe=NVFP4BlockScaling(),
    replace_layers=True,
)

trainer = Trainer(plugins=[plugin], accelerator="gpu", devices=8)
trainer.fit(model, datamodule)
```

**Limitation:** Applies NVFP4 uniformly to every `te.Linear`. No per-layer
precision control.

### 5.2 Model-Level Mixed Precision via Nested Autocast

The paper requires ~15% of layers (typically the last N blocks) in higher
precision. Implement this in the model's `forward()`:

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling, MXFP8BlockScaling

class MyModel(L.LightningModule):
    def __init__(self, num_layers=62, highprec_start=54):
        super().__init__()
        self.layers = nn.ModuleList([...])  # will be converted to te.Linear by plugin
        self.highprec_start = highprec_start
        self.highprec_recipe = MXFP8BlockScaling()

    def forward(self, x):
        # Lightning's forward_context already has te.autocast(recipe=NVFP4BlockScaling)
        # active from the plugin. We override for the last layers:
        for i, layer in enumerate(self.layers):
            if i >= self.highprec_start:
                with te.autocast(recipe=self.highprec_recipe):
                    x = layer(x)
            else:
                x = layer(x)
        return x
```

The inner `te.autocast` overrides the outer one set by the plugin's
`forward_context()`. No changes to the plugin itself.

### 5.3 Mid-Training Precision Switching via Callback

The paper shows switching from NVFP4 → BF16 during LR decay closes the loss
gap. Implement as a Lightning callback:

```python
from transformer_engine.common.recipe import NVFP4BlockScaling, MXFP8BlockScaling

class PrecisionSwitchCallback(L.Callback):
    """Switch TE recipe at a given training step."""

    def __init__(self, switch_step: int, new_recipe=None):
        self.switch_step = switch_step
        self.new_recipe = new_recipe or MXFP8BlockScaling()
        self._switched = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self._switched and trainer.global_step >= self.switch_step:
            trainer.precision_plugin.recipe = self.new_recipe
            self._switched = True
            rank_zero_info(
                f"Switched precision recipe at step {trainer.global_step} "
                f"to {type(self.new_recipe).__name__}"
            )
```

This mutates `trainer.precision_plugin.recipe` in-place. Since
`forward_context()` reads `self.recipe` fresh each step, the change takes
effect immediately.

**Caveat:** This is a mutable-state approach. A more robust alternative is to
subclass the plugin and implement `forward_context()` to check the current
step and select the recipe dynamically.

### 5.4 Custom Plugin with Dynamic Recipe Selection

```python
import torch
from contextlib import ExitStack
from lightning.fabric.plugins.precision.transformer_engine import (
    TransformerEnginePrecision as FabricTEPrecision,
)
from lightning.fabric.plugins.precision.utils import _DtypeContextManager
from lightning.pytorch.plugins.precision.precision import Precision


class NVFP4DynamicPrecision(Precision, FabricTEPrecision):
    """NVFP4 plugin with step-based recipe switching."""

    precision = "nvfp4-dynamic"

    def __init__(
        self,
        *,
        weights_dtype=torch.bfloat16,
        nvfp4_recipe=None,
        decay_recipe=None,
        switch_step=None,
    ):
        from transformer_engine.common.recipe import NVFP4BlockScaling, MXFP8BlockScaling
        super().__init__(
            weights_dtype=weights_dtype,
            recipe=nvfp4_recipe or NVFP4BlockScaling(),
        )
        self.decay_recipe = decay_recipe or MXFP8BlockScaling()
        self.switch_step = switch_step
        self._current_step = 0

    def on_step(self, step: int):
        """Call from a callback or training loop to update step counter."""
        self._current_step = step

    @property
    def active_recipe(self):
        if self.switch_step and self._current_step >= self.switch_step:
            return self.decay_recipe
        return self.recipe

    def forward_context(self):
        import transformer_engine.pytorch as te
        stack = ExitStack()
        stack.enter_context(_DtypeContextManager(self.weights_dtype))
        stack.enter_context(
            torch.autocast(device_type="cuda", dtype=self.fallback_compute_dtype)
        )
        stack.enter_context(
            te.fp8_autocast(enabled=True, fp8_recipe=self.active_recipe)
        )
        return stack
```

### 5.5 Selective Layer Replacement (Skip Layers That Stay in BF16)

If certain layers should never be converted to `te.Linear` (because they
always run in BF16), override `convert_module`:

```python
class NVFP4SelectivePrecision(Precision, FabricTEPrecision):

    def __init__(self, *, weights_dtype, recipe, skip_layer_prefixes=()):
        super().__init__(weights_dtype=weights_dtype, recipe=recipe)
        self.skip_layer_prefixes = skip_layer_prefixes

    def convert_module(self, module):
        # Replace only layers NOT in the skip list
        import transformer_engine.pytorch as te

        for name, child in module.named_modules():
            if any(name.startswith(p) for p in self.skip_layer_prefixes):
                continue
            if isinstance(child, torch.nn.Linear):
                if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                    continue
                parent_name, _, attr = name.rpartition(".")
                parent = module.get_submodule(parent_name) if parent_name else module
                has_bias = child.bias is not None
                replacement = te.Linear(
                    child.in_features, child.out_features, bias=has_bias
                )
                replacement.weight.data = child.weight.data.clone()
                if has_bias:
                    replacement.bias.data = child.bias.data.clone()
                setattr(parent, attr, replacement)

        module = module.to(dtype=self.weights_dtype)
        return module
```

Usage:

```python
plugin = NVFP4SelectivePrecision(
    weights_dtype=torch.bfloat16,
    recipe=NVFP4BlockScaling(),
    skip_layer_prefixes=("layers.54.", "layers.55.", ..., "layers.61."),
)
```

Layers that stay as `nn.Linear` (not `te.Linear`) will be unaffected by
`te.autocast` — they execute in whatever dtype `torch.autocast` provides
(the fallback).

---

## 6. Distributed Considerations

### 6.1 Amax Synchronization

NVFP4's block-level scale factors (`s_block`) are local — no cross-rank
communication needed. But the global scale factor (`s_global`) requires the
global amax across the full tensor.

For tensors sharded by tensor or sequence parallelism, TE performs an
**all-reduce of amax** before quantization:

```
Per-rank local amax  →  all-reduce(max)  →  shared global amax  →  s_global
```

This is automatic for `te.Linear` layers configured with TP/SP.

### 6.2 Quantized All-Gather

TE supports all-gather of NVFP4-quantized tensors. The flow:

```
High-precision tensor
  → compute amax
  → synchronize amax across ranks
  → compute s_global (identical on all ranks)
  → scale + cast to FP4 (with s_block per block)
  → all-gather the FP4 tensor + scale arrays
```

All ranks must share the same `s_global`, which is guaranteed by the amax
sync. This avoids gathering in high precision and then quantizing (which
would be more expensive).

### 6.3 Lightning Strategy Interaction

Lightning's `FSDPStrategy` and `DDPStrategy` operate at a higher level than
the precision plugin. The key interactions:

- **FSDP sharding:** Parameters are sharded in their high-precision form
  (BF16). TE quantizes to FP4 inside `te.Linear.forward()` on the
  already-gathered parameters. FSDP's all-gather happens before quantization.
- **DDP gradient sync:** Gradients are accumulated in FP32 (optimizer states).
  TE's FP4 is only used for the forward/backward GEMMs, not for gradient
  communication.
- **Tensor parallelism:** Handled by TE's own parallel linear layers
  (`te.ColumnParallelLinear`, `te.RowParallelLinear`), which integrate amax
  sync internally. Using these requires building the model with TE's parallel
  layers rather than relying on `convert_module`.

---

## 7. Software Requirements and Constraints

| Component | Minimum Version | Notes |
|-----------|----------------|-------|
| GPU | Blackwell (SM 10.0) | GB200, GB300. Not available on Hopper. |
| Transformer Engine | ≥ 2.8.0 | NVFP4 recipe merged via PR #2177 |
| PyTorch | Per TE compat matrix | Must have Blackwell kernel support |
| Lightning | 2.6+ | `TransformerEnginePrecision` exists |
| CUDA toolkit | 12.8+ | Blackwell arch support |

### Dimension Constraints

- `te.Linear` for NVFP4 requires both `in_features` and `out_features` to be
  divisible by **16** (the block size).
- Scale tensor padding: first dim to multiple of 128, second dim to multiple
  of 4.
- Sequence length should be a multiple of 16 for efficient tiling.

### Memory Overhead

NVFP4 stores **both rowwise and columnwise** quantized copies of each tensor
(needed for TN-layout GEMMs in forward and backward). This doubles the
quantized-tensor memory compared to a single copy, but FP4 is so small that
the total is still well below the BF16 baseline.

Effective storage per value: **~4.5 bits** (4 bits data + 0.5 bits amortized
scale overhead).

---

## 8. Open Implementation Questions

1. **Recipe type checking in `__init__`:** The current plugin defaults unknown
   recipes to `DelayedScaling` via the dict-to-dataclass path. An NVFP4-aware
   plugin should also accept `NVFP4BlockScaling` dicts, or we simply always
   pass pre-built recipe objects.

2. **`convert_module` granularity:** The current implementation replaces *all*
   `nn.Linear` layers. For NVFP4, we want selective replacement (skip final
   blocks). This is either a plugin override (§5.5) or a model-level concern
   (§5.2). Need to decide which pattern we standardize on.

3. **Precision switching state:** Mutating `self.recipe` mid-training (§5.3)
   works because `forward_context()` reads it fresh each call. But this is
   implicit. Should we formalize a `set_recipe()` API with validation?

4. **Checkpoint compatibility:** Switching from NVFP4 to MXFP8 mid-training
   changes the quantization state. TE layers cache quantized copies —
   verify that recipe changes correctly invalidate these caches.

5. **TE parallel layers vs `convert_module`:** For tensor-parallel setups,
   `convert_module`'s `nn.Linear` → `te.Linear` swap isn't sufficient — we
   need `te.ColumnParallelLinear` / `te.RowParallelLinear`. This likely
   means the model must be built with TE layers from the start, not
   converted.

6. **Monitoring quantization error:** TE doesn't expose per-layer
   quantization error metrics out of the box. For debugging convergence
   issues, we may need to add hooks that compare pre-quantization and
   dequantized values.
