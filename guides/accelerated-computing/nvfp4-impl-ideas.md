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
   - 5.1 Minimal: Use Existing Lightning Plugin As-Is
   - 5.2 Model-Level Mixed Precision via Nested Autocast
   - 5.3 Mid-Training Precision Switching via Callback
   - 5.4 Custom Plugin with Dynamic Recipe Selection
   - 5.5 Selective Layer Replacement (Skip Layers That Stay in BF16)
   - 5.6 Predicate-Based Linear Layer Filtering
   - 5.7 Putting It All Together: A Production-Style Recipe
   - 5.8 How NVFP4 Paper-Mode Differs from `bf16-mixed`
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

### 5.6 Predicate-Based Linear Layer Filtering

§5.5 hard-codes a name-prefix skip list, which is good for "freeze these
specific blocks in BF16" but awkward for everything else. A more flexible
pattern uses a **layer predicate** — a callable that decides per-`nn.Linear`
whether to replace it. Predicates are easy to compose (size + name + block
index, etc.) and cover the most useful selection patterns with one plugin
class.

#### 5.6.1 Why You'd Want This

Profiling shows that NVFP4 wall-clock wins on real workloads come from
GEMM speedup minus TE wrapper overhead. Both scale per-`te.Linear`:

- **Per-call CPU cost** of `te.Linear` is ~3–5x higher than `nn.Linear`
  (the TE forward pass goes through a custom autograd Function with
  recipe/scale state management, plus quantize/dequantize plumbing).
- **Per-kernel GPU speedup** from FP4 grows with the GEMM `M`-dim. Tiny
  Linears barely benefit because their kernel is already dominated by
  launch overhead, not arithmetic.

Net: wrapping a 1024×128 projection in `te.Linear` costs more dispatch
than it saves on the GEMM. Wrapping a 1024×3072 FFN expansion saves
substantial wall-clock. A predicate lets you keep the big wins and drop
the small losses.

In hybrid GDN/Mamba/transformer models the picture sharpens further: the
GDN gating projections (`a_proj`, `b_proj`) often have `out_features=12`
or similar small values that don't even satisfy TE's
divisible-by-16 dimension constraint; they get skipped automatically.
But the `q_proj`/`k_proj`/`v_proj` projections inside GDN blocks are
typically smaller than the FFN GEMMs and may not pay back the wrapper
cost — those are the ones a predicate can target.

#### 5.6.2 Plugin with Predicate Support

```python
from typing import Callable

import torch
from lightning.fabric.plugins.precision.transformer_engine import (
    TransformerEnginePrecision as FabricTEPrecision,
)
from lightning.pytorch.plugins.precision.precision import Precision


# A predicate decides whether a given Linear should be REPLACED with te.Linear.
# Returning True → replace (low precision). False → leave as nn.Linear (BF16).
LinearPredicate = Callable[[str, torch.nn.Linear], bool]


class NVFP4FilteredPrecision(Precision, FabricTEPrecision):
    """NVFP4 plugin with predicate-based per-Linear replacement.

    ``replacement_filter`` is called for every ``nn.Linear`` in the model.
    It receives the dotted module name and the layer; return True to
    convert to ``te.Linear``, False to leave alone. ``None`` → replace all
    (matching the default ``TransformerEnginePrecision`` behavior).
    """

    precision = "nvfp4-filtered"

    def __init__(
        self,
        *,
        weights_dtype: torch.dtype,
        recipe,
        replacement_filter: LinearPredicate | None = None,
    ):
        super().__init__(weights_dtype=weights_dtype, recipe=recipe)
        self.replacement_filter = replacement_filter

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        import transformer_engine.pytorch as te

        # Snapshot names first since we'll mutate the tree as we iterate.
        candidates = [
            (n, m) for n, m in module.named_modules() if isinstance(m, torch.nn.Linear)
        ]
        for name, child in candidates:
            if (
                self.replacement_filter is not None
                and not self.replacement_filter(name, child)
            ):
                continue
            # Same shape constraints as TE's internal _convert_layers.
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

The plugin reuses `forward_context()` from the base class, so `te.autocast`
behavior is unchanged — we only narrow which Linears go through the
recipe's quantization path.

#### 5.6.3 Predicate Building Blocks

Small, single-purpose predicates that compose well:

```python
def by_min_size(min_params: int) -> LinearPredicate:
    """Keep NVFP4 only on Linears with at least ``min_params`` parameters."""
    def pred(name: str, layer: torch.nn.Linear) -> bool:
        return layer.in_features * layer.out_features >= min_params
    return pred


def by_name_pattern(*substrings: str) -> LinearPredicate:
    """Keep NVFP4 on Linears whose dotted name contains any substring."""
    def pred(name: str, layer: torch.nn.Linear) -> bool:
        return any(s in name for s in substrings)
    return pred


def not_by_name_pattern(*substrings: str) -> LinearPredicate:
    """Skip Linears whose name contains any of these substrings."""
    def pred(name: str, layer: torch.nn.Linear) -> bool:
        return not any(s in name for s in substrings)
    return pred


def by_block_index(
    max_block_idx: int,
    block_segment: str = "blocks",
) -> LinearPredicate:
    """Keep NVFP4 on Linears in transformer blocks 0..max_block_idx-1.

    Looks for a path segment matching ``block_segment`` followed by an
    integer index. Layers that aren't inside a numbered block (embeddings,
    final norm, lm_head) return True by default — compose with another
    predicate if you want different behavior for them.
    """
    def pred(name: str, layer: torch.nn.Linear) -> bool:
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == block_segment:
                try:
                    return int(parts[i + 1]) < max_block_idx
                except ValueError:
                    return True
        return True
    return pred


def all_of(*predicates: LinearPredicate) -> LinearPredicate:
    def pred(name: str, layer: torch.nn.Linear) -> bool:
        return all(p(name, layer) for p in predicates)
    return pred


def any_of(*predicates: LinearPredicate) -> LinearPredicate:
    def pred(name: str, layer: torch.nn.Linear) -> bool:
        return any(p(name, layer) for p in predicates)
    return pred
```

#### 5.6.4 Useful Filter Recipes

**Recipe A — "FFN-only" NVFP4.** Many models have FFN GEMMs that are 3–4x
bigger than attention projections (e.g., hidden=1024 → ffn=3072). FFN
Linears typically dominate FLOPs; quantizing only those keeps most of
the GPU savings while halving the TE wrapper count.

```python
plugin = NVFP4FilteredPrecision(
    weights_dtype=torch.bfloat16,
    recipe=NVFP4BlockScaling(),
    # FFN linears in many models are named w1/w2/w3 (SwiGLU),
    # gate_proj/up_proj/down_proj (Llama style), or fc1/fc2 (GPT style).
    replacement_filter=by_name_pattern(
        "w1", "w2", "w3",
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2",
    ),
)
```

**Recipe B — "Skip small projections in hybrid models."** GDN-hybrid
models have many small projections per block (q/k/v/g/o, plus dt-bias
projections). Skipping the ones below an empirical size threshold keeps
the FFN expansion + softmax-attention QKV in NVFP4 and lets the small
projections stay in BF16:

```python
plugin = NVFP4FilteredPrecision(
    weights_dtype=torch.bfloat16,
    recipe=NVFP4BlockScaling(),
    replacement_filter=by_min_size(min_params=512 * 1024),  # ~0.5M params
)
```

**Recipe C — "Last 15% in higher precision" (paper pattern).** The NVFP4
paper recommends keeping the final ~15% of transformer blocks in higher
precision. The plugin replaces only the early blocks; later blocks stay
as `nn.Linear` and run in the BF16 fallback set by `forward_context()`.
For an MXFP8 tail (instead of BF16) compose with the model-level nested
autocast pattern from §5.2.

```python
n_blocks = 28
high_prec_start = int(n_blocks * 0.85)  # 24

plugin = NVFP4FilteredPrecision(
    weights_dtype=torch.bfloat16,
    recipe=NVFP4BlockScaling(),
    replacement_filter=all_of(
        by_block_index(max_block_idx=high_prec_start, block_segment="blocks"),
        # Also keep lm_head out of NVFP4 even though it's shape-eligible.
        not_by_name_pattern("lm_head"),
    ),
)
```

**Recipe D — "Skip GDN gating + lm_head + sub-threshold."** Combines
several heuristics to maximize NVFP4 coverage on payoff-positive
Linears only:

```python
plugin = NVFP4FilteredPrecision(
    weights_dtype=torch.bfloat16,
    recipe=NVFP4BlockScaling(),
    replacement_filter=all_of(
        # Skip the small dt-bias / gating projections inside GDN blocks
        # that don't satisfy NVFP4's divisible-by-16 anyway. Listing them
        # by name here means TE doesn't have to log a warning for each.
        not_by_name_pattern("a_proj", "b_proj"),
        not_by_name_pattern("lm_head"),
        by_min_size(min_params=256 * 1024),
    ),
)
```

#### 5.6.5 Audit Helper: Preview Before Training

Predicate bugs are silent — a typo can leave 90% of the model in BF16 and
you won't notice until you compare against an unfiltered baseline. Always
audit the filter against the actual model before training:

```python
import json
from collections import Counter

def audit_replacements(
    model: torch.nn.Module,
    plugin: NVFP4FilteredPrecision,
) -> dict:
    """Return what ``plugin.convert_module`` would do, without mutating ``model``.

    Mirrors the plugin's decision tree (predicate first, then shape
    constraints) so the audit is consistent with what training will see.
    """
    replaced, skipped = [], []
    for name, child in model.named_modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        params = child.in_features * child.out_features
        entry = {
            "name": name,
            "shape": [child.in_features, child.out_features],
            "params": params,
        }
        if (
            plugin.replacement_filter is not None
            and not plugin.replacement_filter(name, child)
        ):
            entry["reason"] = "predicate rejected"
            skipped.append(entry)
        elif child.in_features % 16 or child.out_features % 16:
            entry["reason"] = (
                f"dim not divisible by 16 "
                f"(in%16={child.in_features % 16}, out%16={child.out_features % 16})"
            )
            skipped.append(entry)
        else:
            replaced.append(entry)

    leaf_name = lambda e: e["name"].rsplit(".", 1)[-1]
    return {
        "summary": {
            "replaced_count": len(replaced),
            "skipped_count": len(skipped),
            "replaced_params": sum(e["params"] for e in replaced),
            "skipped_params": sum(e["params"] for e in skipped),
            "replaced_param_fraction": (
                sum(e["params"] for e in replaced)
                / max(1, sum(e["params"] for e in replaced + skipped))
            ),
            "replaced_by_leaf_name": Counter(leaf_name(e) for e in replaced),
            "skipped_by_leaf_name": Counter(leaf_name(e) for e in skipped),
        },
        "replaced": replaced,
        "skipped": skipped,
    }
```

Use it as a sanity check before kicking off `trainer.fit`:

```python
diag = audit_replacements(model, plugin)
print(json.dumps(diag["summary"], indent=2, default=dict))
# {
#   "replaced_count": 196,
#   "skipped_count": 64,
#   "replaced_params": 432000000,
#   "skipped_params": 18000000,
#   "replaced_param_fraction": 0.96,
#   "replaced_by_leaf_name": {"w1": 28, "w2": 28, "w3": 28, ...},
#   "skipped_by_leaf_name":  {"q_proj": 21, "a_proj": 21, "lm_head": 1, ...}
# }
```

If `replaced_param_fraction` is, say, < 0.7, most of the model's weight
is still in BF16 and you'll see a much smaller speedup than the per-kernel
FP4 advantage suggests. Use that fraction (and the per-leaf-name
breakdown) to tune the predicate before paying for a full training run.

#### 5.6.6 Pairing With Profiling

The audit tells you *which* layers will be replaced. Profiling tells you
*which replacements paid off*. The right loop:

1. Pick a predicate. Audit. Confirm fraction looks right.
2. Train for ~200 steps at production seq/batch with profiler enabled
   (with `record_module_names=False` — see §6 for why this matters for
   `torch.compile`).
3. Compute tokens/sec for the filtered run vs an unfiltered NVFP4 run vs
   BF16.
4. If filtered NVFP4 ≥ unfiltered NVFP4 in tokens/sec, the predicate is
   net positive — the dispatch overhead saved exceeds the GEMM speedup
   given up. Otherwise tighten or relax the filter and repeat.

The convergence side (loss curves) is a separate question — predicates
that exclude too many layers may also degrade gradient quality. Always
pair throughput audits with a short loss-quality check on a small
training run.

### 5.7 Putting It All Together: A Production-Style Recipe

§5.1–5.6 each cover one knob. A faithful implementation of the paper's
headline training recipe needs all of them at once. This section combines
them into a single plugin + callback + model pattern.

#### 5.7.1 The Three Orthogonal Choices

Adopting NVFP4 in a real training stack involves three independent
decisions. Each maps to a different mechanism in the Lightning + TE
stack:

| Decision                                               | Mechanism                              | Section |
|--------------------------------------------------------|----------------------------------------|---------|
| Which Linears become `te.Linear` at all                | Plugin's `convert_module` + predicate  | §5.6    |
| What precision each `te.Linear` runs in *per step*     | Plugin's `forward_context` recipe      | §5.1    |
| How the recipe changes *over training*                 | Step-aware `active_recipe`             | §5.4    |
| Per-layer recipe override (e.g., MXFP8 tail blocks)    | Nested `te.autocast` in model.forward  | §5.2    |
| Sanity-check coverage before paying for a long run     | `audit_replacements()`                 | §5.6.5  |

The combined plugin below handles the first three; the fourth is opt-in
model code; the fifth is a pre-flight check.

#### 5.7.2 The Combined Plugin

```python
from contextlib import ExitStack
from typing import Callable

import torch
from lightning.fabric.plugins.precision.transformer_engine import (
    TransformerEnginePrecision as FabricTEPrecision,
)
from lightning.fabric.plugins.precision.utils import _DtypeContextManager
from lightning.pytorch.plugins.precision.precision import Precision


LinearPredicate = Callable[[str, torch.nn.Linear], bool]


class NVFP4ProductionPrecision(Precision, FabricTEPrecision):
    """Filter + dynamic-recipe NVFP4 plugin.

    - ``weights_dtype`` is the on-device parameter dtype. Use ``float32``
      for FP32 master weights (paper-faithful, what the optimizer sees);
      use ``bfloat16`` for BF16-throughout (smaller, less stable).
    - ``fallback_compute_dtype`` is the autocast dtype for Linears that
      ``replacement_filter`` skips (and any non-TE op). Pass
      ``torch.bfloat16`` explicitly when ``weights_dtype=torch.float32``,
      otherwise the parent class defaults it to ``weights_dtype`` and the
      skipped Linears silently run in FP32 (see §5.8.8).
    - ``replacement_filter`` chooses which ``nn.Linear`` modules get
      converted to ``te.Linear`` (§5.6).
    - ``nvfp4_recipe`` is used while ``current_step < switch_step``.
    - ``decay_recipe`` is used once ``current_step >= switch_step``
      (§5.3, §5.4).
    - The current step is pushed in by ``StepTracker`` (§5.7.3) so the
      plugin doesn't need to inspect the trainer.
    """

    precision = "nvfp4-production"

    def __init__(
        self,
        *,
        weights_dtype: torch.dtype = torch.bfloat16,
        fallback_compute_dtype: torch.dtype | None = None,
        nvfp4_recipe=None,
        decay_recipe=None,
        switch_step: int | None = None,
        replacement_filter: LinearPredicate | None = None,
    ):
        from transformer_engine.common.recipe import (
            NVFP4BlockScaling,
            MXFP8BlockScaling,
        )
        super().__init__(
            weights_dtype=weights_dtype,
            recipe=nvfp4_recipe or NVFP4BlockScaling(),
            fallback_compute_dtype=fallback_compute_dtype,
        )
        self.decay_recipe = decay_recipe or MXFP8BlockScaling()
        self.switch_step = switch_step
        self.replacement_filter = replacement_filter
        self._current_step = 0

    # --- step tracking (called from a callback) ---
    def on_step(self, step: int) -> None:
        self._current_step = step

    @property
    def active_recipe(self):
        if self.switch_step is not None and self._current_step >= self.switch_step:
            return self.decay_recipe
        return self.recipe

    # --- forward_context: NVFP4 → MXFP8 swap based on step ---
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

    # --- convert_module: predicate-based replacement ---
    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        import transformer_engine.pytorch as te

        candidates = [
            (n, m) for n, m in module.named_modules() if isinstance(m, torch.nn.Linear)
        ]
        for name, child in candidates:
            if (
                self.replacement_filter is not None
                and not self.replacement_filter(name, child)
            ):
                continue
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

The class is mostly composition: `convert_module` is §5.6's predicate
filter, `forward_context` is §5.4's dynamic recipe selector. The two
features are independent — set `replacement_filter=None` for full
coverage, or `switch_step=None` for a static recipe.

#### 5.7.3 Step-Tracking Callback

The plugin doesn't depend on the trainer; a callback feeds the current
step in once per batch:

```python
import lightning.pytorch as L

class StepTracker(L.Callback):
    """Push ``trainer.global_step`` into a step-aware precision plugin."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        plugin = trainer.precision_plugin
        if hasattr(plugin, "on_step"):
            plugin.on_step(trainer.global_step)
```

This keeps `trainer.precision_plugin` decoupled from Lightning's
internals — easier to unit-test, and the same plugin works under Fabric
without changes.

#### 5.7.4 Optional: Model-Side Nested Autocast for an MXFP8 Tail

The plugin alone leaves the last `(1 - 0.85) * n_blocks` blocks running
in *whatever the predicate excludes them into*. If you skip them via
`by_block_index` (§5.6.4 Recipe C), they stay as `nn.Linear` → BF16
fallback. To run them in MXFP8 instead (closer to the paper), let the
plugin convert them to `te.Linear` and override the recipe locally in
the model's forward (the §5.2 pattern):

```python
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling

class MyModel(L.LightningModule):
    HIGH_PREC_START = 24      # last 4 of 28 blocks → MXFP8
    HIGH_PREC_RECIPE = MXFP8BlockScaling()

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if i >= self.HIGH_PREC_START:
                # Inner autocast wins over the plugin's outer one (§3.4).
                with te.autocast(recipe=self.HIGH_PREC_RECIPE):
                    x = block(x)
            else:
                x = block(x)
        return x
```

The inner `te.autocast` overrides whatever recipe the plugin's
`forward_context()` set for the outer scope, so the tail blocks run in
MXFP8 while the bulk runs in NVFP4 (or MXFP8 after the mid-training
switch — the plugin's behavior is unchanged).

#### 5.7.5 Pre-Flight Audit + Trainer Wiring

```python
import json
import torch
from lightning.pytorch import Trainer
from transformer_engine.common.recipe import NVFP4BlockScaling, MXFP8BlockScaling

# --- Filter: skip lm_head, very small projections, and the last 15% of blocks ---
n_blocks = 28
high_prec_start = int(n_blocks * 0.85)  # 24

replacement_filter = all_of(
    by_block_index(max_block_idx=high_prec_start, block_segment="blocks"),
    not_by_name_pattern("lm_head"),
    by_min_size(min_params=256 * 1024),
)

# --- Plugin: NVFP4 for ~85% of training, MXFP8 thereafter ---
total_steps = 100_000
plugin = NVFP4ProductionPrecision(
    # FP32 master weights → optimizer keeps FP32 m/v + FP32 params.
    # TE quantizes FP32 → FP4 at GEMM time. ``weights_dtype=bf16`` would
    # break the master-weights property and store BF16 optimizer state.
    weights_dtype=torch.float32,
    # Skipped Linears (tail blocks, lm_head) run in BF16 via autocast.
    # Without this, ``fallback_compute_dtype`` defaults to ``weights_dtype``
    # and the skipped Linears would run in FP32 (see §5.8.8).
    fallback_compute_dtype=torch.bfloat16,
    nvfp4_recipe=NVFP4BlockScaling(),
    decay_recipe=MXFP8BlockScaling(),
    switch_step=int(total_steps * 0.85),     # 85_000
    replacement_filter=replacement_filter,
)

# --- Audit: catch filter bugs before paying for the run ---
model = MyModel(...)
diag = audit_replacements(model, plugin)
print(json.dumps(diag["summary"], indent=2, default=dict))
assert diag["summary"]["replaced_param_fraction"] >= 0.70, (
    f"filter too aggressive: only {diag['summary']['replaced_param_fraction']:.0%} "
    f"of params will run in NVFP4"
)

# --- Wire it up ---
trainer = Trainer(
    plugins=[plugin],
    callbacks=[StepTracker(), ...],
    max_steps=total_steps,
    accelerator="gpu",
    devices=8,
)
trainer.fit(model, datamodule)
```

#### 5.7.6 Mapping Back to the Paper

| Paper element                                       | Where it lives in this recipe                                |
|-----------------------------------------------------|--------------------------------------------------------------|
| FP32 master weights, FP32 optimizer state           | `weights_dtype=torch.float32` (see §5.8 for what this means) |
| BF16 fallback for skipped Linears                   | `fallback_compute_dtype=torch.bfloat16` (§5.8.8 gotcha)      |
| NVFP4 GEMMs in most layers                          | `nvfp4_recipe` + plugin's `convert_module`                   |
| Stochastic rounding, RHT, 2D weight scaling         | `NVFP4BlockScaling()` defaults (§4.1)                        |
| Last ~15% of blocks in higher precision             | `by_block_index` filter (BF16 tail) **or** §5.7.4 (MXFP8 tail)|
| `lm_head` excluded from FP4                         | `not_by_name_pattern("lm_head")` filter                      |
| NVFP4 → MXFP8 mid-training switch                   | `switch_step` + `decay_recipe` + `StepTracker`               |
| TP/SP all-reduce of amax                            | Automatic in `te.Linear` (see §6)                            |
| FSDP-friendly                                       | TE quantizes post-gather; nothing to do (see §6.3)           |
| Skip layers below the FP4 payback threshold         | `by_min_size` filter (empirical threshold)                   |
| Verify coverage before training                     | `audit_replacements()` pre-flight                            |

#### 5.7.7 Knobs to Tune

Each filter parameter and step boundary is empirical. Reasonable starting
points and what to watch:

- **`weights_dtype`** — `torch.float32` for paper-faithful FP32 master
  weights (the optimizer keeps FP32 `m`, `v`, params); `torch.bfloat16`
  for BF16-throughout if you're willing to trade stability for memory.
  Cost of FP32: 2× param storage and 2× optimizer state vs BF16. See
  §5.8 for the full lifecycle implications.
- **`fallback_compute_dtype`** — set explicitly to `torch.bfloat16`
  whenever `weights_dtype` is `torch.float32`. Otherwise it defaults to
  `weights_dtype` and skipped Linears silently run in FP32 (§5.8.8).
- **`high_prec_start = int(n_blocks * 0.85)`** — paper recommends 15%.
  Watch downstream eval; if loss diverges from BF16 reference late in
  training, increase the high-precision tail.
- **`switch_step = 0.85 * total_steps`** — anchored to the LR decay
  start. Too early loses the FP4 throughput win; too late lets the
  quantization noise compound through decay.
- **`min_params = 256 * 1024`** — purely throughput-driven. Sweep
  thresholds (`128 * 1024`, `512 * 1024`, `1024 * 1024`) and pick the
  one with the best `tokens/sec` from the §5.6.6 profile loop. Has no
  effect on convergence (the skipped layers run in BF16 fallback, same
  as they would without TE).
- **`replaced_param_fraction` floor** — the assert in §5.7.5 catches
  filter regressions. 70% is a soft floor; under that you've effectively
  trained a BF16 model with extra TE overhead.

### 5.8 How NVFP4 Paper-Mode Differs from `bf16-mixed`

The §5.7 plugin is most useful when you can articulate exactly *what
changes* between a `bf16-mixed` baseline and a paper-faithful NVFP4 run.
With the right config (`weights_dtype=torch.float32`,
`fallback_compute_dtype=torch.bfloat16`) the optimizer-side semantics
are identical; the differences are concentrated in `convert_module`,
the `forward_context` stack, and what gets saved for backward. Walking
through each lifecycle stage clarifies which knob is doing what.

#### 5.8.1 Module Construction

| | `bf16-mixed` (Lightning's `MixedPrecision`) | NVFP4 paper-mode (`NVFP4ProductionPrecision`) |
|---|---|---|
| `convert_module` | no-op — `MixedPrecision` doesn't override it | walks the tree, swaps eligible `nn.Linear` → `te.Linear` for the bulk of blocks, then `module.to(weights_dtype)` casts every parameter |
| Param dtype on device | FP32 (whatever the model was built with) | FP32 (when `weights_dtype=torch.float32`) |
| Module class for transformer Linears | `nn.Linear` everywhere | `te.Linear` for the early blocks, `nn.Linear` for the high-precision tail + `lm_head` |

#### 5.8.2 forward_context (wraps every train step)

```python
# bf16-mixed
torch.autocast("cuda", dtype=torch.bfloat16)

# NVFP4 paper-mode
ExitStack:
    _DtypeContextManager(weights_dtype)                       # default tensor dtype
    torch.autocast("cuda", dtype=fallback_compute_dtype)      # fallback for skipped layers
    te.fp8_autocast(enabled=True, fp8_recipe=active_recipe)   # NVFP4 or MXFP8
```

#### 5.8.3 At a Linear's Forward Call

| Stage | `bf16-mixed`, `nn.Linear` (FP32 weight) | NVFP4, `te.Linear` (FP32 weight) — eligible block | NVFP4, `nn.Linear` (FP32 weight) — skipped tail / `lm_head` |
|---|---|---|---|
| Input | activations from previous op | activations | activations |
| Cast at op boundary | `x.to(bf16)`, `w.to(bf16)` (autocast) | TE quantizes `x` (1×16 NVFP4) and `w` (16×16 NVFP4 with RHT) | depends on `fallback_compute_dtype` |
| GEMM dtype | **BF16 × BF16 → BF16** on Tensor Cores | **FP4 × FP4 → BF16** on Tensor Cores (E4M3 block scales applied per block, FP32 global scale post-GEMM) | matches fallback dtype |
| Saved-for-backward | BF16 `x` and `w` views (16 bits/elt) | FP4 rowwise + columnwise copies + E4M3 scales (~4.5 bits/elt) | matches fallback dtype |

Two asymmetries worth highlighting:

- **Where precision loss happens.** `bf16-mixed`'s precision drop is a
  boundary cast (FP32 → BF16, near-trivial in practice — BF16 has the
  same 8-bit exponent as FP32). NVFP4 applies an RHT to the weight,
  then 16×16 block-amax scaling, then 4-bit quantization with
  stochastic rounding on the gradient path. The numerics machinery is
  doing real work for outlier suppression — that's where the paper's
  three features (RHT, 2D block scaling, stochastic rounding) live.
- **What backward saves.** `bf16-mixed` saves BF16 activations
  (16 bits/elt). NVFP4 saves FP4 activations (~4.5 bits/elt). Activation
  memory drops by ~3.5×, which is where most of the FP4 batch-size
  headroom comes from on long-sequence training.

#### 5.8.4 Backward

| | `bf16-mixed` | NVFP4 paper-mode |
|---|---|---|
| Backward inside autocast? | No (Lightning exits before `loss.backward()`) | No (same), but TE's saved tensors carry the recipe info |
| dgrad GEMM (`grad_out × wᵀ`) | BF16 GEMM | FP4 GEMM using saved FP4 columnwise weight, with stochastic rounding when re-quantizing `grad_out` |
| wgrad GEMM (`xᵀ × grad_out`) | BF16 GEMM | FP4 GEMM using saved FP4 columnwise activation, with RHT applied to inputs before quantization |
| Where `param.grad` lands | FP32 — autograd accumulates BF16 grad onto FP32 `nn.Parameter.grad` (boundary cast) | FP32 — same boundary cast, even though the GEMM was FP4 |

In both setups the gradient *the optimizer sees* is FP32. The difference
is purely in *how* that FP32 gradient was computed (BF16 GEMM vs FP4
GEMM with stochastic rounding).

#### 5.8.5 Optimizer Step

| | `bf16-mixed` | NVFP4 paper-mode |
|---|---|---|
| Param dtype | FP32 | FP32 |
| Grad dtype | FP32 | FP32 |
| AdamW `m`, `v` state | FP32 | FP32 |
| GradScaler | None (BF16 has FP32-equivalent dynamic range) | None |

Optimizer-side they're identical — both have FP32 master weights with
FP32 optimizer state. `weights_dtype=torch.float32` on the NVFP4 plugin
is what makes this true; `weights_dtype=torch.bfloat16` would store BF16
params and BF16 optimizer state, breaking the master-weights property.

#### 5.8.6 Memory per Parameter

For one logical parameter, ignoring activations:

| Bucket | `bf16-mixed` | NVFP4 paper-mode |
|---|---|---|
| Weight on `Module` | 4 B (FP32) | 4 B (FP32) |
| `param.grad` | 4 B (FP32) | 4 B (FP32) |
| AdamW `m` | 4 B (FP32) | 4 B (FP32) |
| AdamW `v` | 4 B (FP32) | 4 B (FP32) |
| **Subtotal: optimizer-related** | **16 B/param** | **16 B/param** |
| TE FP4 row+col cache (forward only, freed after backward) | — | ~1 B/param + small scale tables |
| Saved activations (per step, dominates at long seqlen) | BF16 ⇒ 2 B / stored value | FP4 ⇒ ~0.6 B / stored value |

Parameters and optimizer state cost the same. The savings come almost
entirely from saved activations, which is why FP4's batch-size headroom
shows up most when `seqlen × hidden` is big (the FFN intermediates).

#### 5.8.7 Behavior Over Training Time

- **`bf16-mixed`:** static. Same plugin behavior at step 0 and step
  100k.
- **NVFP4 paper-mode:** `te.fp8_autocast` reads `self.active_recipe`
  fresh each step (a `StepTracker` callback pushes `global_step` into
  the plugin). At `switch_step`, the property flips from
  `NVFP4BlockScaling` to `MXFP8BlockScaling` and the next forward and
  backward run in MXFP8. Stored params and optimizer state aren't
  touched — only the *quantization recipe applied at GEMM time*.

#### 5.8.8 The `fallback_compute_dtype` Gotcha

`TransformerEnginePrecision`'s parent has a subtle default:

```python
# In the parent's __init__:
self.fallback_compute_dtype = fallback_compute_dtype or weights_dtype
```

If you set `weights_dtype=torch.float32` (for FP32 master weights) and
**don't** override `fallback_compute_dtype`, the outer
`torch.autocast("cuda", dtype=fp32)` becomes a no-op. The skipped
Linears (high-precision tail blocks, `lm_head`) then run in **FP32, not
BF16** — slower than the `bf16-mixed` baseline for those layers, and
farther from the paper than necessary.

Pass `fallback_compute_dtype=torch.bfloat16` explicitly:

```python
plugin = NVFP4ProductionPrecision(
    weights_dtype=torch.float32,
    fallback_compute_dtype=torch.bfloat16,   # tail blocks + lm_head run in BF16
    nvfp4_recipe=NVFP4BlockScaling(),
    decay_recipe=MXFP8BlockScaling(),
    switch_step=int(0.85 * total_steps),
    replacement_filter=replacement_filter,
)
```

With this, the tail blocks behave **identically to `bf16-mixed`** (FP32
weights, BF16 GEMM via autocast), while the bulk runs in NVFP4 with
FP32 master weights. The only thing that differs between the two runs
is the GEMM precision in the inner ~85% of blocks — which is the
comparison the paper's throughput numbers are actually claimed against.

#### 5.8.9 What This Means for Benchmarking

If you want a meaningful tokens/sec or loss comparison between
`bf16-mixed` and NVFP4:

1. **Keep `weights_dtype=torch.float32` on the NVFP4 plugin.** Otherwise
   you're measuring FP4-GEMM-vs-BF16-GEMM **and** BF16-master-vs-FP32-
   master in one mixed signal.
2. **Pass `fallback_compute_dtype=torch.bfloat16`.** Otherwise the
   skipped Linears have a different compute dtype than the baseline,
   confounding the per-block comparison.
3. **Match the optimizer config.** Both runs should use the same
   AdamW settings, same gradient clipping, same LR schedule. With (1)
   and (2) above, the optimizer state is byte-for-byte the same.
4. **Match the seq/batch geometry.** The activation-memory savings let
   NVFP4 fit larger batch — so report tokens/sec, not ms/step, when
   batch size differs.

What you're left measuring is the throughput delta from FP4 GEMMs in
the early ~85% of the model, plus or minus the per-`te.Linear` wrapper
overhead, plus or minus the activation-memory headroom you spend on
larger batches. That's what's actually attributable to the precision
choice.

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
