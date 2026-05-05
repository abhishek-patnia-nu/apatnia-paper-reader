# NVFP4 Training Stack: Implementation Primer

> A production-style PyTorch Lightning plugin for NVFP4 training, what
> it actually does at each step, and how it differs from the
> `bf16-mixed` baseline.

**Source code references:**

- [Fabric `Precision` base](https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/plugins/precision/precision.html)
- [Fabric `TransformerEnginePrecision`](https://lightning.ai/docs/fabric/stable/_modules/lightning/fabric/plugins/precision/transformer_engine.html) — where the real logic lives
- [TE NVFP4 docs](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)
- [TE FP8/FP4 primer notebook](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html)
- [TE NVFP4 PR #2177](https://github.com/NVIDIA/TransformerEngine/pull/2177)
- Paper context: [arXiv 2509.25149](https://arxiv.org/abs/2509.25149)

---

## Table of Contents

1. [The Production Plugin](#1-the-production-plugin)
2. [How It Differs from `bf16-mixed`](#2-how-it-differs-from-bf16-mixed)
3. [Distributed Considerations](#3-distributed-considerations)
4. [How It Plugs Into Lightning](#4-how-it-plugs-into-lightning)
5. [Software Requirements and Constraints](#5-software-requirements-and-constraints)
6. [Open Implementation Questions](#6-open-implementation-questions)

---

## 1. The Production Plugin

A faithful NVFP4 training stack on Lightning needs to handle three
orthogonal concerns:

| Concern | Mechanism |
|---|---|
| Which `nn.Linear` modules become `te.Linear` at all | `convert_module` + a layer predicate |
| What precision each `te.Linear` runs in *per step* | `forward_context` recipe |
| How the recipe changes *over training* | Step-aware `active_recipe` (NVFP4 → MXFP8 mid-training) |

The plugin below packages all three into one class. Plus a
**pre-flight audit** (§1.4) for sanity-checking coverage before
paying for a full run.

### 1.1 Layer Predicates

A `LinearPredicate` is a callable taking a module's dotted name and
the `nn.Linear` itself; returning `True` keeps the layer in the FP4
path, `False` leaves it as `nn.Linear`. Small composable building
blocks cover the common selection patterns:

```python
from typing import Callable

import torch

LinearPredicate = Callable[[str, torch.nn.Linear], bool]


def by_min_size(min_params: int) -> LinearPredicate:
    """Keep NVFP4 only on Linears with at least ``min_params`` parameters."""
    def pred(name, layer):
        return layer.in_features * layer.out_features >= min_params
    return pred


def by_name_pattern(*substrings: str) -> LinearPredicate:
    """Keep NVFP4 on Linears whose name contains any substring."""
    def pred(name, layer):
        return any(s in name for s in substrings)
    return pred


def not_by_name_pattern(*substrings: str) -> LinearPredicate:
    """Skip Linears whose name contains any of these substrings."""
    def pred(name, layer):
        return not any(s in name for s in substrings)
    return pred


def by_block_index(
    max_block_idx: int,
    block_segment: str = "blocks",
) -> LinearPredicate:
    """Keep NVFP4 on Linears whose containing block has index
    ``< max_block_idx``.

    Implementation parses the layer's dotted name. Assumes the
    standard transformer-block convention:

      * Blocks are stored under a container named ``block_segment``
        on the model — typically ``model.blocks``.
      * That container is either an ``nn.ModuleDict`` keyed by
        ``str(block_idx)`` (so paths look like ``blocks.0.<...>``,
        ``blocks.1.<...>``, …) or an ``nn.ModuleList`` indexed by
        int (which produces the same dotted-path layout via
        ``named_modules()``).
      * Each block sets ``self.block_idx = block_idx`` in
        ``__init__``. This is a common pattern in transformer
        codebases — the block needs to pass its index down to
        sub-modules (e.g. for RoPE caches keyed by layer, or for
        debugging) and storing it as an attribute is the obvious way.

    The dotted-name path and the ``block_idx`` attribute both
    encode the same information, since the storage container is
    keyed by the same index. We parse the path here rather than
    reading the attribute so the predicate stays a pure function
    of ``(name, layer)`` — no need to take a model handle at
    construction time. If your blocks live somewhere unusual
    (e.g. nested under ``model.transformer.layers``) override
    ``block_segment`` accordingly.

    Layers that aren't inside a numbered block (embeddings, final
    norm, ``lm_head``) return ``True`` by default — compose with
    another predicate to control those cases.
    """
    def pred(name, layer):
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
    def pred(name, layer):
        return all(p(name, layer) for p in predicates)
    return pred


def any_of(*predicates: LinearPredicate) -> LinearPredicate:
    def pred(name, layer):
        return any(p(name, layer) for p in predicates)
    return pred
```

Why this matters: profiling shows NVFP4 wall-clock wins come from
GEMM speedup minus per-`te.Linear` wrapper overhead. Both scale per
layer:

- **Per-call CPU cost** of `te.Linear` is several times higher than
  `nn.Linear` (custom autograd Function, recipe state management,
  quantize/dequantize plumbing).
- **Per-kernel GPU speedup** from FP4 grows with the GEMM `M`-dim.
  Tiny Linears barely benefit — their kernel is already dominated by
  launch overhead, not arithmetic.

A predicate lets you keep the big-FFN wins and drop the
small-projection losses. Hybrid models (GDN, Mamba, MoE) with many
small per-block projections benefit most from this.

### 1.2 The Plugin

```python
from collections.abc import Mapping
from contextlib import AbstractContextManager, ExitStack

import torch
from lightning.fabric.plugins.precision.transformer_engine import (
    TransformerEnginePrecision as FabricTEPrecision,
)
from lightning.fabric.plugins.precision.utils import _DtypeContextManager
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.pytorch.plugins.precision.precision import Precision


class NVFP4ProductionPrecision(Precision, FabricTEPrecision):
    """Filter + dynamic-recipe NVFP4 plugin.

    Differences from the stock ``TransformerEnginePrecision``:
      * Predicate-based ``convert_module`` (only replace Linears that
        pass ``replacement_filter``).
      * Dynamic recipe via ``active_recipe`` — supports a single
        mid-training switch from NVFP4 → MXFP8.
      * Stricter dim check: NVFP4 needs both dims divisible by 16
        (parent uses the looser FP8 rule of in%8 / out%16).
      * ``module_init_context`` disabled — see method docstring.
      * ``convert_input`` / ``convert_output`` disabled. The parent
        casts every batch to ``weights_dtype`` and every output back
        to ``torch.get_default_dtype()`` on the way out. ``torch.autocast``
        inside ``forward_context`` already handles op-boundary casts
        for the layers that need it; this extra blanket cast is
        wasteful (it touches every per-batch tensor on the host loop)
        and also surprising — e.g. with ``weights_dtype=torch.float32``
        the parent silently up-casts BF16 inputs to FP32. Disable both
        and let the model decide what to do with its inputs.
      * Only ``nn.Linear`` is replaced; ``nn.LayerNorm`` is left alone
        (parent replaces both). LN runs in FP32 outside autocast either
        way, so the only delta is ``te.LayerNorm``'s slightly faster
        kernel — keep it explicit if you want LN-Linear fusion via
        ``te.LayerNormLinear`` at model build time.

    ``te.Linear`` does **not** subclass ``torch.nn.Linear``; downstream
    code that does ``isinstance(layer, nn.Linear)`` will silently skip
    converted layers. Run any inspection helpers (e.g. the audit in
    §1.4) on a pristine pre-conversion model.

    Args:
        weights_dtype: On-device parameter dtype. ``torch.float32`` for
            FP32 master weights (paper-faithful, what the optimizer
            sees); ``torch.bfloat16`` for BF16-throughout (smaller,
            less stable).
        fallback_compute_dtype: Autocast dtype for Linears that
            ``replacement_filter`` skips (and any non-TE op). Pass
            ``torch.bfloat16`` explicitly when
            ``weights_dtype=torch.float32``, otherwise the parent
            class defaults it to ``weights_dtype`` and skipped Linears
            silently run in FP32 (see §2.8).
        nvfp4_recipe: Recipe object (e.g. ``NVFP4BlockScaling()``) used
            while ``current_step < switch_step``. Must be a recipe
            instance, **not** a dict — the parent class hardcodes
            dict normalization to FP8 ``DelayedScaling`` and would
            silently downgrade an NVFP4 dict to FP8.
        decay_recipe: Recipe used once ``current_step >= switch_step``.
            Defaults to ``MXFP8BlockScaling()``.
        switch_step: Global step at which to swap recipes. ``None`` =
            never swap, stays on ``nvfp4_recipe`` for the whole run.
        replacement_filter: Optional ``LinearPredicate``. ``None``
            replaces every shape-eligible Linear, matching stock
            ``TransformerEnginePrecision``.
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
        # Defensive: parent's __init__ has a ``isinstance(recipe, Mapping)
        # → DelayedScaling(**recipe)`` branch that hardcodes FP8. Passing
        # a dict here would silently train in FP8 instead of NVFP4.
        if isinstance(nvfp4_recipe, Mapping):
            raise TypeError(
                "nvfp4_recipe must be a recipe object (e.g. "
                "NVFP4BlockScaling()), not a Mapping/dict — the parent "
                "class would normalize a dict to DelayedScaling (FP8)."
            )
        if isinstance(decay_recipe, Mapping):
            raise TypeError(
                "decay_recipe must be a recipe object (e.g. "
                "MXFP8BlockScaling()), not a Mapping/dict."
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

    def on_step(self, step: int) -> None:
        """Called by ``StepTracker`` (§1.3) once per training batch."""
        self._current_step = step

    @property
    def active_recipe(self):
        if self.switch_step is not None and self._current_step >= self.switch_step:
            return self.decay_recipe
        return self.recipe

    def forward_context(self):
        """The three-layer context stack that every training step runs in.

        Each manager touches a different piece of global state, and
        all three are needed for an FP4 training step:

        1. ``_DtypeContextManager(weights_dtype)`` sets
           ``torch.get_default_dtype()`` to ``weights_dtype``. This
           affects *new tensor allocations inside the forward* — e.g.
           if the model does ``torch.zeros(...)`` for a buffer or a
           position-encoding cache, that tensor will be created in
           ``weights_dtype``. It does **not** cast existing tensors.

        2. ``torch.autocast("cuda", dtype=fallback_compute_dtype)``
           is the **fallback for any op not handled by te.fp8_autocast**
           — not only ``nn.Linear`` modules we skipped from
           ``te.Linear`` conversion, but also attention (softmax,
           scaled-dot-product), norms (LayerNorm, RMSNorm),
           elementwise ops on outputs of ``te.Linear``, etc. PyTorch's
           autocast registry catches these at the op boundary and
           casts inputs to ``fallback_compute_dtype``. With
           ``fallback_compute_dtype=torch.bfloat16`` this matches
           ``MixedPrecision("bf16-mixed")`` semantics for those ops,
           which is what makes the apples-to-apples comparison in
           §2 valid.

        3. ``te.fp8_autocast(enabled=True, fp8_recipe=active_recipe)``
           sets a TE thread-local recipe that ``te.Linear.forward()``
           reads to decide its quantization path (NVFP4 vs MXFP8 vs
           plain FP8 etc.). ``self.active_recipe`` is read fresh on
           every call to ``forward_context``, which is what makes the
           mid-training NVFP4 → MXFP8 switch work without rebuilding
           any module.

        Order of ``enter_context`` calls doesn't matter for
        correctness (each manager touches independent state), but
        matches the stock plugin's order for readability: dtype
        context outermost, ``torch.autocast`` middle, ``te.fp8_autocast``
        innermost.

        Critical detail: the backward pass runs **outside** this
        context (Lightning exits the context after the training_step,
        before ``loss.backward()``). TE handles this correctly
        because the saved tensors and autograd Function objects
        already carry the recipe that was active during forward —
        backward re-applies it automatically.
        """
        import transformer_engine.pytorch as te

        stack = ExitStack()

        # (1) Default dtype for *new* tensor allocations during forward.
        # Affects e.g. ``torch.zeros(...)`` calls inside the model.
        # Does NOT cast existing parameters or activations.
        stack.enter_context(_DtypeContextManager(self.weights_dtype))

        # (2) Fallback autocast for ops without an FP4/FP8 path in TE
        # (attention, norms, elementwise ops, skipped ``nn.Linear``s,
        # etc.). Matches what ``MixedPrecision("bf16-mixed")`` does
        # when ``fallback_compute_dtype=torch.bfloat16``.
        stack.enter_context(
            torch.autocast(device_type="cuda", dtype=self.fallback_compute_dtype)
        )

        # (3) TE quantization recipe. Read ``self.active_recipe``
        # fresh so a recipe swap (NVFP4 → MXFP8 at ``switch_step``)
        # takes effect on the next training step without any
        # ``convert_module`` rerun. Innermost wins for nested
        # ``te.autocast`` overrides (see §4.3).
        stack.enter_context(
            te.fp8_autocast(enabled=True, fp8_recipe=self.active_recipe)
        )

        return stack

    def module_init_context(self) -> AbstractContextManager:
        """Disable the parent's global ``nn.Linear`` → ``te.Linear``
        class replacement.

        The parent uses ``_ClassReplacementContextManager`` to monkey-
        patch ``torch.nn.Linear`` at construction time. That happens
        *before* ``convert_module``, so the predicate filter would
        never get a chance to run — every Linear built under
        ``fabric.init_module()`` would become ``te.Linear`` regardless
        of ``replacement_filter``. Force everyone through the
        ``convert_module`` path so the predicate is always honored.
        """
        return self.tensor_init_context()

    def convert_input(self, data: Any) -> Any:
        """Pass batches through untouched.

        The parent casts every floating-point tensor in the batch to
        ``weights_dtype``. ``torch.autocast`` inside ``forward_context``
        already does op-boundary casts where they're needed; this
        blanket pre-cast adds host-loop overhead and can surprise the
        model (e.g. up-casting BF16 numerical features to FP32 when
        ``weights_dtype=torch.float32``).
        """
        return data

    def convert_output(self, data: Any) -> Any:
        """Pass outputs through untouched (see ``convert_input``)."""
        return data

    def convert_module(self, module: torch.nn.Module) -> torch.nn.Module:
        import transformer_engine.pytorch as te

        # Defensive guard from stock plugin: don't try to re-replace
        # modules that are already TE (e.g. loaded from a checkpoint
        # built with TE layers). Just cast and return.
        if any(
            "transformer_engine.pytorch" in m.__module__
            for m in module.modules()
        ):
            rank_zero_info(
                "NVFP4ProductionPrecision: model already contains "
                "TransformerEngine layers; skipping convert_module."
            )
            return module.to(dtype=self.weights_dtype)

        candidates = [
            (n, m) for n, m in module.named_modules()
            if isinstance(m, torch.nn.Linear)
        ]
        for name, child in candidates:
            if (
                self.replacement_filter is not None
                and not self.replacement_filter(name, child)
            ):
                continue
            # Stricter than the parent's FP8 check (in%8 / out%16) —
            # NVFP4 block size is 16 along both dims.
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

The class is mostly composition: `convert_module` is predicate-filtered
layer replacement; `forward_context` is dynamic recipe selection. The
two features are independent — set `replacement_filter=None` for full
coverage, or `switch_step=None` for a static recipe.

Four guards inherited from analyzing the stock plugin's behavior:

- `module_init_context` is overridden to disable global
  `nn.Linear` → `te.Linear` interception during model construction.
  Without this override, building the model under
  `fabric.init_module()` would silently replace **every** Linear,
  ignoring `replacement_filter`.
- `convert_input` / `convert_output` overridden to no-ops. The
  parent's per-batch dtype cast is redundant with `torch.autocast`,
  costs host-loop time on every batch, and can surprise the model
  (e.g. up-casting BF16 numerical features to FP32 when
  `weights_dtype=torch.float32`).
- `convert_module` short-circuits if the model already contains TE
  modules. Matches the stock plugin's behavior.
- `__init__` rejects dict recipes. The parent class hardcodes
  `dict → DelayedScaling` (FP8); accepting the dict would silently
  downgrade an NVFP4 config to FP8.

### 1.3 Step-Tracking Callback

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

Decoupling keeps the plugin testable and lets the same plugin work
under Fabric (which doesn't have callbacks) by calling `on_step`
directly from the training loop.

### 1.4 Pre-Flight Audit

Predicate bugs are silent — a typo can leave 90% of the model in BF16
and you won't notice until you compare against an unfiltered baseline.
Audit the filter against the actual model before training.

> **Run the audit on a pristine pre-conversion model.** `te.Linear`
> does not subclass `torch.nn.Linear`, so the
> `isinstance(child, torch.nn.Linear)` check below silently skips
> every layer that has already been converted. Build a fresh model
> for the audit; do **not** reuse a model that has already been
> through `plugin.convert_module()`.

```python
import json
from collections import Counter


def audit_replacements(
    model: torch.nn.Module,
    plugin: NVFP4ProductionPrecision,
) -> dict:
    """Return what ``plugin.convert_module`` would do, without mutating ``model``.

    Mirrors the plugin's decision tree (predicate first, then shape
    constraints) so the audit matches what training will see.
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

    leaf = lambda e: e["name"].rsplit(".", 1)[-1]
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
            "replaced_by_leaf_name": Counter(leaf(e) for e in replaced),
            "skipped_by_leaf_name": Counter(leaf(e) for e in skipped),
        },
        "replaced": replaced,
        "skipped": skipped,
    }
```

If `replaced_param_fraction < 0.7` you've effectively trained a BF16
model with extra TE overhead. Use the fraction (and the per-leaf
breakdown) to tune the predicate before paying for a full run.

### 1.5 Wiring It Up: The Paper-Faithful Config

```python
import torch
from lightning.pytorch import Trainer
from transformer_engine.common.recipe import NVFP4BlockScaling, MXFP8BlockScaling

# Filter: skip lm_head, very small projections, and the last 15% of blocks.
n_blocks = 28
high_prec_start = int(n_blocks * 0.85)  # 24

replacement_filter = all_of(
    by_block_index(max_block_idx=high_prec_start, block_segment="blocks"),
    not_by_name_pattern("lm_head"),
    by_min_size(min_params=256 * 1024),
)

total_steps = 100_000
plugin = NVFP4ProductionPrecision(
    # FP32 master weights → optimizer keeps FP32 m/v + FP32 params.
    # TE quantizes FP32 → FP4 at GEMM time. ``weights_dtype=bf16`` would
    # break the master-weights property.
    weights_dtype=torch.float32,
    # Skipped Linears (tail blocks, lm_head) run in BF16 via autocast.
    # Without this, ``fallback_compute_dtype`` defaults to
    # ``weights_dtype`` and skipped Linears would run in FP32 (§2.8).
    fallback_compute_dtype=torch.bfloat16,
    nvfp4_recipe=NVFP4BlockScaling(),
    decay_recipe=MXFP8BlockScaling(),
    switch_step=int(total_steps * 0.85),
    replacement_filter=replacement_filter,
)

# Pre-flight audit
model = MyModel(...)
diag = audit_replacements(model, plugin)
print(json.dumps(diag["summary"], indent=2, default=dict))
assert diag["summary"]["replaced_param_fraction"] >= 0.70

trainer = Trainer(
    plugins=[plugin],
    callbacks=[StepTracker(), ...],
    max_steps=total_steps,
    accelerator="gpu",
    devices=8,
)
trainer.fit(model, datamodule)
```

### 1.6 Mapping to the Paper

| Paper element | Plugin knob |
|---|---|
| FP32 master weights, FP32 optimizer state | `weights_dtype=torch.float32` |
| BF16 fallback for skipped Linears | `fallback_compute_dtype=torch.bfloat16` |
| NVFP4 GEMMs in most layers | `nvfp4_recipe` + `convert_module` |
| Stochastic rounding, RHT, 2D weight scaling | `NVFP4BlockScaling()` defaults |
| Last ~15% of blocks in higher precision | `by_block_index` filter (skipped Linears run in BF16 fallback) |
| `lm_head` excluded from FP4 | `not_by_name_pattern("lm_head")` filter |
| NVFP4 → MXFP8 mid-training switch | `switch_step` + `decay_recipe` + `StepTracker` |
| TP/SP all-reduce of amax | Automatic in `te.Linear` (§3) |
| FSDP-friendly | TE quantizes post-gather; nothing to do (§3) |
| Skip layers below the FP4 payback threshold | `by_min_size` filter (empirical) |
| Verify coverage before training | `audit_replacements()` pre-flight |

### 1.7 Knobs to Tune

- **`weights_dtype`** — `torch.float32` for paper-faithful FP32
  master weights; `torch.bfloat16` for BF16-throughout. Cost of FP32:
  2× param storage and 2× optimizer state vs BF16.
- **`fallback_compute_dtype`** — set to `torch.bfloat16` whenever
  `weights_dtype=torch.float32`. Otherwise it defaults to
  `weights_dtype` and skipped Linears silently run in FP32 (§2.8).
- **`high_prec_start = int(n_blocks * 0.85)`** — paper recommends
  15%. Watch downstream eval; if loss diverges from BF16 reference
  late in training, increase the high-precision tail.
- **`switch_step = 0.85 * total_steps`** — anchored to the LR decay
  start. Too early loses the FP4 throughput win; too late lets
  quantization noise compound through decay.
- **`min_params = 256 * 1024`** — purely throughput-driven. Sweep
  thresholds (`128 * 1024`, `512 * 1024`, `1024 * 1024`) and pick the
  one with best `tokens/sec`. Has no effect on convergence (skipped
  layers run in BF16 fallback, same as without TE).
- **`replaced_param_fraction` floor** — the audit assert. 70% is a
  soft floor; under that you've effectively trained a BF16 model with
  extra TE overhead.

---

## 2. How It Differs from `bf16-mixed`

The interesting comparison: same model, same data, same optimizer
config — what changes between Lightning's stock
`MixedPrecision("bf16-mixed", "cuda")` and the plugin from §1 with
paper-faithful config (`weights_dtype=fp32`,
`fallback_compute_dtype=bf16`)?

### 2.1 Module Construction

| | `bf16-mixed` | NVFP4 paper-mode |
|---|---|---|
| `convert_module` | no-op | walks the tree, swaps eligible `nn.Linear` → `te.Linear` for the bulk; `module.to(weights_dtype)` casts every param |
| Param dtype on device | FP32 (whatever the model was built with) | FP32 (when `weights_dtype=torch.float32`) |
| Module class for transformer Linears | `nn.Linear` everywhere | `te.Linear` for early blocks, `nn.Linear` for tail + `lm_head` |

### 2.2 forward_context

```python
# bf16-mixed
torch.autocast("cuda", dtype=torch.bfloat16)

# NVFP4 paper-mode (innermost → outermost effect; see §1.2 forward_context docstring)
ExitStack:
    _DtypeContextManager(weights_dtype)                       # default dtype for new tensor allocs in forward
    torch.autocast("cuda", dtype=fallback_compute_dtype)      # fallback for ops not handled by te.fp8_autocast
                                                              #   (attention, norms, skipped nn.Linears, etc.)
    te.fp8_autocast(enabled=True, fp8_recipe=active_recipe)   # NVFP4 or MXFP8 — only te.Linear reads this
```

### 2.3 At a Linear's Forward

| Stage | `bf16-mixed`, `nn.Linear` | NVFP4, `te.Linear` (eligible) | NVFP4, `nn.Linear` (skipped tail/`lm_head`) |
|---|---|---|---|
| Cast at op boundary | `x.to(bf16)`, `w.to(bf16)` (autocast) | TE quantizes `x` (1×16 NVFP4) and `w` (16×16 NVFP4 with RHT) | matches `fallback_compute_dtype` |
| GEMM dtype | **BF16 × BF16 → BF16** on Tensor Cores | **FP4 × FP4 → BF16** on Tensor Cores (E4M3 block scales per block, FP32 global scale post-GEMM) | matches fallback |
| Saved-for-backward | BF16 `x` and `w` views (16 bits/elt) | FP4 rowwise+columnwise + E4M3 scales (~4.5 bits/elt) | matches fallback |

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

### 2.4 Backward

| | `bf16-mixed` | NVFP4 paper-mode |
|---|---|---|
| dgrad GEMM (`grad_out × wᵀ`) | BF16 GEMM | FP4 GEMM using saved FP4 columnwise weight, with stochastic rounding on `grad_out` |
| wgrad GEMM (`xᵀ × grad_out`) | BF16 GEMM | FP4 GEMM using saved FP4 columnwise activation, with RHT on inputs |
| Where `param.grad` lands | FP32 — autograd casts BF16 grad to FP32 at the param boundary | FP32 — same boundary cast |

In both setups the gradient *the optimizer sees* is FP32. The
difference is how that FP32 gradient was computed (BF16 vs FP4 GEMMs).

### 2.5 Optimizer Step

| | `bf16-mixed` | NVFP4 paper-mode |
|---|---|---|
| Param dtype | FP32 | FP32 |
| Grad dtype | FP32 | FP32 |
| AdamW `m`, `v` state | FP32 | FP32 |
| GradScaler | None (BF16 has FP32-equivalent dynamic range) | None |

**Optimizer-side they're identical.** `weights_dtype=torch.float32`
on the NVFP4 plugin is what makes this true.
`weights_dtype=torch.bfloat16` would store BF16 params and BF16
optimizer state, breaking the master-weights property.

### 2.6 Memory per Parameter

For one logical parameter, ignoring activations:

| Bucket | `bf16-mixed` | NVFP4 paper-mode |
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

### 2.7 Behavior Over Training Time

- **`bf16-mixed`:** static. Same plugin behavior at step 0 and step
  100k.
- **NVFP4 paper-mode:** `te.fp8_autocast` reads `self.active_recipe`
  fresh each step. At `switch_step`, the property flips from
  `NVFP4BlockScaling` to `MXFP8BlockScaling` and the next step runs
  in MXFP8. Stored params and optimizer state aren't touched — only
  the *quantization recipe applied at GEMM time*.

### 2.8 The `fallback_compute_dtype` Gotcha

`TransformerEnginePrecision`'s parent has a subtle default:

```python
self.fallback_compute_dtype = fallback_compute_dtype or weights_dtype
```

If you set `weights_dtype=torch.float32` and **don't** override
`fallback_compute_dtype`, the outer `torch.autocast("cuda", dtype=fp32)`
is a no-op. Skipped Linears (tail blocks, `lm_head`) then run in
**FP32, not BF16** — slower than `bf16-mixed` for those layers, and
farther from the paper than necessary.

Pass `fallback_compute_dtype=torch.bfloat16` explicitly. With this,
the tail blocks behave **identically to `bf16-mixed`** (FP32 weights,
BF16 GEMM via autocast), while the bulk runs in NVFP4 with FP32
master weights. The only difference between the two runs is the GEMM
precision in the inner ~85% of blocks — which is the comparison the
paper's throughput numbers are claimed against.

### 2.9 Implications for Benchmarking

For a meaningful tokens/sec or loss comparison:

1. **Keep `weights_dtype=torch.float32` on the NVFP4 plugin.**
   Otherwise you're measuring FP4-GEMM-vs-BF16-GEMM **and**
   BF16-master-vs-FP32-master in one mixed signal.
2. **Pass `fallback_compute_dtype=torch.bfloat16`.** Otherwise
   skipped Linears have a different compute dtype than the baseline,
   confounding the per-block comparison.
3. **Match the optimizer config.** Both runs should use the same
   AdamW settings, gradient clipping, LR schedule. With (1) and (2)
   above, the optimizer state is byte-for-byte the same.
4. **Match the seq/batch geometry.** Activation-memory savings let
   NVFP4 fit a larger batch — report `tokens/sec`, not ms/step, when
   batch differs.

What's left is the throughput delta from FP4 GEMMs in the early
~85% of the model, plus or minus per-`te.Linear` wrapper overhead,
plus or minus activation-memory headroom. That's what's actually
attributable to the precision choice.

---

## 3. Distributed Considerations

### 3.1 Amax Synchronization

NVFP4's block-level scale factors (`s_block`) are local — no
cross-rank communication. But the global scale factor (`s_global`)
requires the global amax across the full tensor.

For tensors sharded by tensor or sequence parallelism, TE performs
an all-reduce of amax before quantization:

```
Per-rank local amax  →  all-reduce(max)  →  shared global amax  →  s_global
```

Automatic for `te.Linear` layers configured with TP/SP.

### 3.2 Quantized All-Gather

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

### 3.3 Lightning Strategy Interaction

- **FSDP sharding:** Parameters are sharded in their high-precision
  form (FP32 in paper-mode). TE quantizes to FP4 inside
  `te.Linear.forward()` on the already-gathered parameters — FSDP's
  all-gather happens before quantization.
- **DDP gradient sync:** Gradients are accumulated in FP32. TE's
  FP4 is only used for forward/backward GEMMs, not for gradient
  communication.
- **Tensor parallelism:** Handled by TE's own parallel linear layers
  (`te.ColumnParallelLinear`, `te.RowParallelLinear`), which integrate
  amax sync internally. Using these requires building the model with
  TE's parallel layers rather than relying on `convert_module`.

---

## 4. How It Plugs Into Lightning

This section covers the underlying mechanism — how the precision
plugin interface works, when each method gets called, and how
`te.fp8_autocast` activates FP4 inside `te.Linear`.

### 4.1 The Precision Plugin Interface

Lightning splits precision handling into a framework-agnostic core
(Fabric) and a trainer-aware wrapper (PyTorch Lightning):

```
lightning.fabric.plugins.precision.Precision       (core interface)
    │
    ├── forward_context()          → context manager wrapping forward
    ├── convert_module(module)     → one-time model surgery
    ├── convert_input(data)        → cast inputs to working dtype
    ├── convert_output(data)       → cast outputs back to default dtype
    ├── tensor_init_context()      → default dtype for new tensors
    └── module_init_context()      → intercept nn.Module construction
          │
lightning.pytorch.plugins.precision.Precision(FabricPrecision, ...)
    │
    ├── connect(model, optimizers, ...)  → wire into trainer
    ├── pre_backward / backward / post_backward
    ├── optimizer_step(optimizer, model, closure)
    └── {train,val,test,predict}_step_context() → all delegate to forward_context()
```

`forward_context()` is the primary extension point. Every training and
eval step runs inside the context manager it returns. For FP8/FP4,
this is where `te.fp8_autocast` is activated.

`convert_module(module)` runs once during `trainer.fit()` setup,
before training starts:

```
trainer.fit()
  └─ _call_setup_hook()
       └─ strategy.setup()
            └─ precision.convert_module(model)   # nn.Linear → te.Linear
            └─ move model to device
```

This is the right place for the `nn.Linear` → `te.Linear` swap.
`te.Linear` is the same class for FP8 and FP4 — the recipe (set by
`forward_context`) decides which precision format runs at GEMM time.

`module_init_context()` is a *second* entry point for layer
replacement, fired when the user constructs a model under
`fabric.init_module()`:

```python
with fabric.init_module():     # plugin's module_init_context is active
    model = MyModel(...)       # nn.Linear constructions get redirected
```

The stock `TransformerEnginePrecision` overrides this hook to
monkey-patch `torch.nn.Linear` → `te.Linear` globally for the
duration of the context (via `_ClassReplacementContextManager`). The
upside is no clone-and-replace pass later; the downside is that any
predicate filter on `convert_module` is bypassed entirely — every
Linear becomes `te.Linear` regardless of `replacement_filter`.

For predicate-based plugins (§1.2) the right fix is to override
`module_init_context()` to return only the dtype context manager,
*not* the class replacement. That forces every layer through
`convert_module` so the predicate is always honored, at the cost of
the init-time fast path. The §1.2 plugin does this.

### 4.2 Trainer Loop Hook Order

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

Critical detail: **the backward pass runs outside `forward_context()`**.
TE handles this correctly because the autograd backward graph was
built during forward inside the context — TE layers record the recipe
and apply it during backward automatically.

For our plugin (§1.2) the relevant flow is:

- Step 1: `forward_context()` returns an `ExitStack` that activates
  `_DtypeContextManager` → `torch.autocast` → `te.fp8_autocast`. The
  `te.fp8_autocast` reads `self.active_recipe` fresh, so a recipe
  swap at `switch_step` takes effect on the very next step.
- Step 2: `te.Linear.forward()` checks the thread-local recipe set
  by `te.fp8_autocast` and quantizes operands accordingly. Skipped
  Linears (still `nn.Linear`) just run under the outer
  `torch.autocast(fallback_compute_dtype)`.
- Step 4: `loss.backward()` walks the autograd graph. TE's saved
  tensors carry the recipe info; FP4 dgrad/wgrad GEMMs run here.
  Gradients accumulate onto FP32 `nn.Parameter.grad`.
- Step 6: `optimizer.step()` runs against FP32 params and FP32 grads.
  `StepTracker.on_train_batch_start` had already pushed the new step
  index into the plugin, so the `active_recipe` for the *next*
  training step is already correct.

### 4.3 `te.fp8_autocast` and `te.Linear`

`te.fp8_autocast` (despite the name, it also handles FP4) sets a
thread-local recipe; `te.Linear.forward()` checks this thread-local
to decide whether to quantize:

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
range (e.g. running the tail in MXFP8 while the bulk stays in
NVFP4):

```python
with te.autocast(recipe=NVFP4BlockScaling()):     # outer (plugin)
    x = layer_0(x)   # NVFP4
    with te.autocast(recipe=MXFP8BlockScaling()):  # inner (model)
        x = layer_N(x)   # MXFP8
```

### 4.4 The `NVFP4BlockScaling` Recipe

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
    │   [Distributed: all-reduce amax across TP/SP ranks (§3.1)]
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

## 5. Software Requirements and Constraints

| Component | Minimum Version | Notes |
|-----------|----------------|-------|
| GPU | Blackwell (SM 10.0) | GB200, GB300. Not available on Hopper. |
| Transformer Engine | ≥ 2.8.0 | NVFP4 recipe merged via PR #2177 |
| PyTorch | Per TE compat matrix | Must have Blackwell kernel support |
| Lightning | 2.6+ | `TransformerEnginePrecision` exists |
| CUDA toolkit | 12.8+ | Blackwell arch support |

### Dimension Constraints

- `te.Linear` for NVFP4 requires both `in_features` and
  `out_features` divisible by **16** (the block size).
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

## 6. Open Implementation Questions

1. **Step-aware recipe via callback feels indirect.** The §1.3
   `StepTracker` callback pushes `trainer.global_step` into the
   plugin once per batch; `active_recipe` then reads that field. This
   works but couples the plugin to a specific callback. A cleaner
   alternative would be a `set_step()` API that the trainer calls
   directly, or having the plugin look at
   `trainer.precision_plugin._connected_trainer.global_step` (if
   such a back-reference existed). Worth revisiting if the callback
   approach turns out to be brittle.

2. **Checkpoint compatibility across recipe switches.** Switching
   from NVFP4 to MXFP8 mid-training changes the quantization state.
   TE layers cache quantized copies — verify that recipe changes
   correctly invalidate these caches, especially across a
   save/restore cycle that crosses `switch_step`.

3. **TE parallel layers vs `convert_module`.** For tensor-parallel
   setups, `convert_module`'s `nn.Linear` → `te.Linear` swap isn't
   sufficient — we need `te.ColumnParallelLinear` /
   `te.RowParallelLinear`. This likely means the model must be built
   with TE parallel layers from the start, not converted. The §1.2
   plugin's `module_init_context` override (which disables TE class
   replacement at init time) does **not** help here — TE parallel
   layers aren't drop-in replacements for `nn.Linear`.

4. **LayerNorm replacement.** §1.2 deliberately leaves
   `nn.LayerNorm` alone; the stock plugin replaces it with
   `te.LayerNorm`. If profiling shows the LayerNorm kernel matters,
   the right fix is `te.LayerNormLinear` fusion at model build time
   rather than independent LN replacement (which only buys a slightly
   faster norm kernel).

5. **Monitoring quantization error.** TE doesn't expose per-layer
   quantization error metrics out of the box. For debugging
   convergence issues, we may need hooks that compare pre-quantization
   and dequantized values.
