# Section 4: Implementation

> **Paper reference:** Section 4, pages 7–8

## What this section covers

§3 explained *what* FSDP does. §4 explains *how it does it in PyTorch* -- the API surface users see, the data structures behind the scenes, and the specific autograd hooks that drive correctness.

| §4.X | Topic | Why it matters |
|---|---|---|
| 4.1 | Three ways to initialize a large model | When deferred init doesn't work (cross-unit init deps), what are the fallbacks? |
| 4.2 | The `FlatParameter` class and `FlatParamHandle` | The concrete class structure behind the §3.2 algorithm |
| 4.3 | Runtime autograd hook attachment | Three different hook types, each used for a specific purpose |
| 4.4 | Native mixed precision | A subtle win: FSDP **decreases** peak memory in mixed precision instead of doubling it |

### The two FSDP APIs

Before diving in, the paper distinguishes two user-facing APIs:

1. **`FullyShardedDataParallel`** -- the **wrapper** style. You write `model = FSDP(model)` and FSDP replaces sub-modules with FSDP-wrapped versions.
   - Pros: explicit, easy to introspect.
   - Cons: changes the module hierarchy. Parameter names become `_fsdp_wrapped_module.layer.weight` instead of `layer.weight`, which breaks things like `state_dict` keying and code that walks the module tree.

2. **`fully_shard`** -- the **functional / annotator** style. You write `fully_shard(model.layer)` and FSDP installs `nn.Module` forward/backward *hooks* on the existing module without replacing it.
   - Pros: preserves the module structure and parameter fully-qualified names.
   - Cons: less explicit; mechanism is hooks-based.

Both produce the same algorithm at runtime; the difference is purely in how the FSDP logic gets attached. `fully_shard` is the newer, generally preferred API for new code. The paper covers both because both were/are supported.

---

## 4.1 Initialization (three options, decreasing preference)

§3.1 introduced deferred initialization as the answer when models are too big to materialize on one GPU. §4.1 acknowledges that deferred init has a corner case it can't handle, and gives two fallback options.

### The corner case for deferred init

Recap: deferred init records each tensor's init ops on the meta device, then replays them one FSDP unit at a time on real GPU.

This works **as long as each unit's init is self-contained.** Imagine block 5's `__init__` does:

```python
self.weight = nn.Parameter(some_init())
# Now reach into another block and use its weight:
self.adapter = block_2.weight @ some_matrix    # ← cross-unit dependency
```

When FSDP gets to unit 5 and tries to replay this op, `block_2.weight` is **sharded** -- there's no unsharded copy on the device. The replay can't access the unsharded tensor it needs. Deferred init breaks.

The paper offers two fallbacks for this case.

### Option A: Initialize unsharded on GPU

Skip deferred init entirely. Materialize the full model on a single GPU at construction time, then hand it to FSDP, which immediately shards it.

```python
model = MyHuge175BModel().cuda()    # ← unsharded, on GPU
model = FSDP(model)                  # ← shards immediately
optimizer = Adam(model.parameters()) # ← optimizer over sharded params
```

When this works: when the **unsharded model** fits on one GPU but the **training state** (unsharded model + grads + optimizer + activations) doesn't.

For example, a 12B-param BF16 model:
- Unsharded model: 24 GB (fits on a 40GB A100).
- Training: 24 (BF16 params) + 24 (BF16 grads) + 48 (FP32 master copy) + 96 (Adam moments) + activations = >190 GB. Doesn't fit.

You can pre-load the 24 GB model, hand it to FSDP, which immediately shards. Then construct the optimizer -- it'll see only the sharded params, so its state is also sharded, and you're in business.

**Important**: instantiate the optimizer **after** FSDP wrapping, not before. Otherwise the optimizer captures the unsharded parameters and you've defeated the point.

### Option B: Initialize unsharded on CPU, stream to GPU

When the unsharded model doesn't fit on one GPU but does fit on CPU (which has way more RAM):

```python
model = MyHuge175BModel()           # ← built on CPU, unsharded
model = FSDP(model, device_id=0)    # ← FSDP streams unit-by-unit to GPU
```

FSDP walks the model unit by unit:
1. Move this unit's parameters to GPU (now unsharded on device).
2. Shard them across ranks.
3. Free the unsharded copy.
4. Repeat for next unit.

Cross-unit init dependencies are fine here because the entire unsharded model exists in CPU memory throughout. Slower than deferred init due to limited CPU memory bandwidth and the host-to-device copies, but it works on basically any model that fits on a single host's RAM.

### Recommendation order

1. **Deferred init** -- fastest, works for most models with self-contained per-unit init.
2. **Unsharded on GPU** -- if the model just barely fits to construct but not to train.
3. **Unsharded on CPU** -- last resort for models too big even for one GPU's init, with cross-unit init dependencies.

### Choosing the FSDP unit boundary

A separate but related decision: how to chunk the model into FSDP units (which determines what gets coalesced into one FlatParameter, and what gets AllGathered together).

Two mechanisms:

1. **Manual** -- the user explicitly wraps sub-modules:
   ```python
   for block in model.blocks:
       block = FSDP(block)
   model = FSDP(model)
   ```
   Invasive to model code but maximal control.

2. **`auto_wrap_policy`** -- the user provides a function that decides for each `nn.Module` whether to make it its own FSDP unit:
   ```python
   def policy(module, recurse, **kwargs):
       return isinstance(module, TransformerBlock)
   model = FSDP(model, auto_wrap_policy=policy)
   ```
   Non-invasive. Typical practice: one unit per transformer block.

The paper notes there's no universally optimal policy -- "selecting the optimal wrapping approach typically requires some experiments and measurements."

---

## 4.2 Flat Parameters (the class structure)

§3.2.1 introduced the FlatParameter as an algorithmic concept -- "flatten all the unit's params into one 1D tensor, pad, and chunk evenly." §4.2 names the actual classes.

### Two classes, separation of concerns

```
FlatParameter   (inherits from nn.Parameter)
    Just a 1D tensor that holds the concatenated/padded parameter data.
    Behaves like any other nn.Parameter -- autograd sees it, optimizer
    sees it (when sharded), etc.

FlatParamHandle
    The "manager" object that holds one FlatParameter and knows:
      - the original parameter shapes & names (for restoring views)
      - the sharding metadata (which rank holds which slice)
      - how to perform the AllGather/ReduceScatter
      - when to materialize unsharded views vs free them
```

The user-facing wrapper (`FullyShardedDataParallel` or `fully_shard`) only ever talks to `FlatParamHandle`, never directly to the FlatParameter. This **separates the storage** (FlatParameter, a pure data object) from the **logic** (FlatParamHandle, which orchestrates communication and view management).

Why this matters: it makes the algorithm cleanly testable and reusable. A future API change can switch from a wrapper to a hook-based approach (which is exactly what `fully_shard` did) without touching the storage logic.

### How wrap-policy maps to FlatParameter construction

§3.2.1 said the user controls how parameters are coalesced. The concrete rule from the paper:

> All parameters in the annotated `nn.Module` are assigned to one FlatParameter, **excluding those parameters already assigned**.

This rule is recursive. If you annotate both an outer module and an inner module, the inner one wins (its params are assigned first); the outer module's FlatParameter gets the *residual* params not already taken.

```
class Model(nn.Module):
    def __init__(self):
        self.embed = Embedding(...)        # ← residual
        self.blocks = [Block(...) for _ in range(N)]
        self.head = Linear(...)            # ← residual

# With auto_wrap_policy = "wrap each Block":
#   block_0's params      → FlatParameter_0  (handle: block_0)
#   block_1's params      → FlatParameter_1
#   ...
#   block_N-1's params    → FlatParameter_{N-1}
#   embed + head params   → FlatParameter_N  (residual on outer Model)
```

This nicely matches what you'd want for a transformer: each block is one FlatParameter (so AllGather/ReduceScatter overlap nicely with per-block compute), and the embedding + LM head -- which are usually used at the very start and very end of forward -- get bundled into one outer FlatParameter that lives unsharded across the whole iteration.

### An alternative the team explored (but didn't ship)

The paper mentions an interesting alternative briefly: **use execution order to reconstruct FlatParameters dynamically.** Start with very small FlatParameters, run one iteration, observe which sub-modules execute in what order, then coalesce small adjacent FlatParameters into bigger ones for subsequent iterations.

The pro: the FlatParameter boundaries would match actual execution order, not nn.Module nesting order (which can mismatch -- e.g. if a module's `forward()` calls its children in non-declaration order).

The con: the first iteration is inefficient (many small collectives), and dynamic reshuffling has its own complexity. They didn't ship this; the user-controlled nn.Module-based approach won out.

---

## 4.3 Runtime (the autograd hooks)

The actual moment of attaching FSDP to PyTorch's training loop happens through `nn.Module` hooks and autograd hooks. You marked autograd hooks as familiar -- here's the specific menu FSDP uses.

### `nn.Module` hooks (forward path)

For the **forward pass**, both APIs add logic before and after each FSDP unit's forward:

```
FullyShardedDataParallel wrapper:
    Overrides nn.Module.forward() with custom logic that:
      1. AllGathers the unit's params (or schedules it via prefetching)
      2. Sets up views (so original Parameter names work)
      3. Calls the original forward
      4. Frees the unsharded copy (unless NRAF mode)

fully_shard annotator:
    Uses register_forward_pre_hook() and register_forward_hook()
    on the existing nn.Module. Same logic, different attachment.
```

These are normal nn.Module hooks -- the same mechanism users use to attach debug prints or quantization wrappers. PyTorch fires them automatically when `module(input)` runs.

### Autograd hooks (backward path)

The **backward pass** is trickier because PyTorch's autograd engine handles it automatically and transparently. There's no "backward of unit N" entry point in user code -- it all happens inside the autograd graph traversal. So FSDP uses three different types of autograd hooks, each for a different purpose.

#### Hook type 1: `Tensor.register_hook` (on activations)

This hook fires when a specific tensor's gradient is computed (during backward).

```python
# At the *output* of an FSDP unit's forward pass:
def pre_backward_hook(grad):
    # Backward is about to enter this unit. AllGather params now.
    schedule_allgather_for_this_unit()
    return grad   # don't modify the gradient

output_tensor.register_hook(pre_backward_hook)
```

FSDP attaches this hook to the **forward output** of every FSDP unit. When autograd processes that output during backward, the hook fires -- giving FSDP a precise moment to schedule the AllGather *just before* the unit's backward computation begins.

This is the activation-side anchor: "I want a callback right before this unit's backward runs."

#### Hook type 2: `Tensor.register_hook` on `AccumulateGrad` (on parameters)

This is subtler. In PyTorch's autograd graph, every parameter has an `AccumulateGrad` node at the end of its gradient computation -- this is the function that actually writes the gradient into `param.grad`. You can hook this node to run code right when accumulation finishes.

```python
# On each FlatParameter:
def post_accumulate_grad_hook(param):
    # FlatParameter.grad is now fully written. Time to reduce.
    issue_reduce_scatter(param.grad)
    # After ReduceScatter, each rank only keeps its 1/N slice.

flat_param.register_post_accumulate_grad_hook(post_accumulate_grad_hook)
```

This fires as **soon as the FlatParameter's gradient is fully accumulated.** Critically, it doesn't wait for the rest of the backward pass to finish -- the ReduceScatter fires as early as possible, overlapping with the still-running backward of earlier layers.

The paper specifically notes that the activation-side hook (type 1) *could* in principle do the same job, but it'd be **late** -- the activation gradient is computed before the parameter gradient is fully accumulated, so firing on the activation means waiting an extra moment. The `AccumulateGrad` hook is the tight version.

#### Hook type 3: `queue_callback` (on the whole backward)

This runs **right before the autograd backward pass exits**, after all gradient computations have finished. FSDP uses it as a final "wait barrier":

```python
def end_of_backward_callback():
    # Wait for all pending ReduceScatters to complete
    # so the optimizer step doesn't read in-flight grads.
    cuda_synchronize_with_nccl_stream()

torch.autograd._engine.queue_callback(end_of_backward_callback)
```

Why is this necessary? Recall that ReduceScatters are issued on a separate CUDA stream (the NCCL stream). They run concurrently with the rest of backward. At some point we need to be **sure** they've all completed before the optimizer step reads `param.grad`. The `queue_callback` mechanism is autograd's hook for "run this after backward is fully done."

### Summary of the three hooks

| Hook | Fires when | FSDP uses it for |
|---|---|---|
| `tensor.register_hook` on activations | gradient of activation tensor is computed | **schedule AllGather** for unit's backward |
| `flat_param.register_post_accumulate_grad_hook` | FlatParameter's gradient is fully accumulated | **issue ReduceScatter** (as early as possible) |
| `autograd.queue_callback` | backward pass is about to exit | **synchronize NCCL stream** so optimizer sees finished grads |

The paper's larger point: each of these is a **standard PyTorch hook**, not a private internal API. FSDP gets correctness for free for things like double-backward, retain_graph, gradient checkpointing, etc. -- because autograd is what's driving the hook firing, and autograd already handles those cases correctly.

> **Paper ref:** "The aforementioned methodologies collectively integrate the FSDP algorithm with the PyTorch nn.Module and autograd engine in a non-intrusive and efficient manner." (page 8)

---

## 4.4 Native Mixed Precision

You're solid on mixed precision in general (BF16/FP16 for fwd/bwd + FP32 master copy + gradient scaler for FP16). This subsection focuses on **the FSDP-specific twist**, which is genuinely clever.

### The standard mixed-precision memory overhead

Normal mixed precision (PyTorch's `torch.amp.autocast` style):

```
Per parameter element:
  FP32 master copy:    K_full = 4 bytes
  FP16/BF16 copy:      K_low  = 2 bytes   (added by mixed precision)
  Total:                       6 bytes

Plus optimizer state (Adam):
  m, v in FP32:                8 bytes

So mixed precision *increases* per-param memory from 4 (FP32 only) to 6 bytes,
or from 12 (FP32 + Adam) to 14 bytes total for the parameter-related state.
```

Mixed precision is normally a memory **cost** -- a 50% increase in parameter-related memory -- in exchange for faster compute (BF16/FP16 matmuls).

### Why FSDP can do better

Recall FSDP's memory equation from §3.2.1:

```
Peak param memory per rank = Ψ/F + max_i ψ_i
                              ↑      ↑
                              sharded current unsharded
                              model  unit
```

The **sharded model** part lives in GPU memory throughout training. The **current unsharded unit** is only allocated during one unit's compute, then freed.

Now, the FP32 master copy is what the optimizer uses. The optimizer only sees the *sharded* model (because we instantiate the optimizer after FSDP wrapping). So the FP32 master copy only needs to be `Ψ/F` per rank -- not `Ψ` -- and it only needs to be in FP32 for the sharded part.

The **low-precision (BF16) unsharded copy** only needs to exist during one unit's compute. After the unit finishes, we free it.

Putting it together, FSDP's peak parameter memory in mixed precision is:

```
       K_full         K_low
  ───────────────── ────────────
  Ψ/F (FP32 sharded) + max_i ψ_i (BF16 unsharded current unit)

  = (4/F) Ψ + 2 max_i ψ_i bytes
```

Compare this to FP32-only FSDP, which would be:

```
  K_full · (Ψ/F) + K_full · max_i ψ_i  =  (4/F) Ψ + 4 max_i ψ_i bytes
```

The second term **decreases** from `4 max ψ_i` to `2 max ψ_i`. Mixed precision, in FSDP, **reduces** peak memory rather than increasing it. The first term (sharded FP32 model) is unchanged.

```
                  No mixed prec     Standard mixed prec    FSDP mixed prec
                  (FP32 only)       (DDP/single-GPU)       (this paper)
Sharded part      4 Ψ / F           4 Ψ / F                4 Ψ / F        unchanged
Unsharded unit    4 max_i ψ_i       6 max_i ψ_i            2 max_i ψ_i    ↓
                                    (BF16+FP32)            (BF16 only)
```

The key insight: the FP32 master copy of the *unsharded current unit* is **never needed**, because the optimizer only operates on the *sharded* state. So FSDP can skip allocating it. The unsharded copy can stay in BF16 throughout forward+backward.

This is a real, non-trivial win that the paper deserves credit for.

> **Paper ref:** "the parameter peak memory contribution for FSDP actually **decreases** from `K_full/F · Σ ψ_i + K_full · max ψ_i` to `K_full/F · Σ ψ_i + K_low · max ψ_i`." (page 8, emphasis added)

### Operator-level cast vs FlatParameter-level cast

Standard PyTorch mixed precision (`torch.amp.autocast`) casts operands **at every operator** -- just-in-time. Each matmul takes its FP32 inputs, casts to BF16, computes, returns BF16.

FSDP's mixed precision casts **once per FlatParameter**, in the pre-forward step (and pre-backward, if reshard-after-forward is enabled):

```
Pre-forward of unit:
  1. AllGather sharded BF16 FlatParameter  ← yes, the AllGather itself is in BF16!
  2. Set up views
  3. Run the unit's forward (all in BF16)
```

This has two benefits:

1. **AllGather/ReduceScatter run in BF16**, halving communication volume vs FP32 collectives. This is a significant throughput improvement for comm-bound configurations.
2. **One cast per FlatParameter**, not one cast per op. Less casting overhead.

The trade-off: less granular control. If a specific operator needed FP32 precision for numerical stability (e.g. softmax over very large logits), `torch.amp.autocast` would handle it automatically; FSDP's coarser cast can't. The user has to know about and configure precision exceptions explicitly.

### The sharded gradient scaler (FP16 only)

One subtlety: FP16 (not BF16) needs gradient scaling because its dynamic range is small and gradients can underflow. The standard `torch.amp.GradScaler` works by:

1. Multiplying the loss by a large scalar before backward.
2. Checking grads for infs/nans after backward.
3. If clean, divide grads back by the scalar and step optimizer. If dirty, skip the step and reduce the scalar.

The "check for infs/nans" is a global operation -- a single inf anywhere invalidates the step. In FSDP, gradients are **sharded across ranks** after ReduceScatter. So you can't check infs locally; you need an **AllReduce** to globally OR the inf-flags across ranks before deciding whether to step.

FSDP ships its own `ShardedGradScaler` that does this. The math is the same as `GradScaler` but with an extra cross-rank reduction.

(BF16 doesn't need any of this because its dynamic range matches FP32 -- no underflow, no scaling, no sharded scaler needed. Most modern training uses BF16 for exactly this reason.)

---

## Key takeaways from Section 4

1. **Two FSDP APIs.** `FullyShardedDataParallel` (wrapper) replaces sub-modules; `fully_shard` (annotator) preserves module structure via hooks. Same algorithm, different attachment.

2. **Three init options.** Deferred init handles most cases; "unsharded on GPU" works when the model construction fits but training doesn't; "unsharded on CPU" works when not even construction fits.

3. **FlatParameter + FlatParamHandle.** A clean separation: FlatParameter is pure data (1D tensor); FlatParamHandle is the orchestrator. Wrap policies are recursive with "first match wins."

4. **Three autograd hooks**, each with a specific job:
   - Tensor hook on activations → **schedule AllGather** before a unit's backward.
   - Post-accumulate-grad hook on FlatParameter → **issue ReduceScatter** as soon as the gradient is ready.
   - `queue_callback` → **synchronize streams** at end of backward.

5. **FSDP's mixed precision saves memory instead of costing it.** By keeping the unsharded unit in BF16 (no FP32 master copy needed for transient compute) and the sharded model in FP32, FSDP **decreases** the `max_i ψ_i` term from `4 max ψ_i` to `2 max ψ_i`.

6. **Collectives run in BF16** when mixed precision is on, halving cross-rank comm volume vs FP32 collectives. Big throughput win.

7. **`ShardedGradScaler` for FP16** -- handles the sharded grad layout when checking infs/nans. BF16 doesn't need any of this.

---

*Previous: [Section 3 -- System Design](section_3_system_design.md)*
*Next: [Section 5 -- Evaluation](section_5_evaluation.md)* -- the empirical results. Three model families (T5 from 611M to 11B, GPT-175B, DHEN-768B recsys) on 8 to 512 A100 80GB GPUs. Three categories of experiments: how FSDP compares to DDP at small sizes, when the rate limiter actually helps, and how FSDP scales on truly large models with linear TFLOPS scaling.
