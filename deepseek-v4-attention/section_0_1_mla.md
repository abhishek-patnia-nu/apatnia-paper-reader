# Section 0.1 — Prerequisite: MLA (Multi-head Latent Attention)

> **Lineage position:** This is the *root* of CSA's family tree. MLA (DeepSeek-V2, 2024)
> introduced the idea that the KV cache should be a **compressed latent**, not raw keys
> and values. CSA inherits MLA's query/KV down-projection wholesale — the $c^Q_t$ latent
> in CSA eq. 13 is literally MLA's query compression. Read this first.

## What this section covers

- **The KV cache** — why it exists, and why *it* (not compute) is the long-context
  bottleneck. You asked for detail here.
- **MQA/GQA** — one-paragraph refresher, just to place them in the lineage.
- **MLA** — latent compression, the **weight-absorption** trick that makes it work, and
  the decoupled-RoPE fix. Full depth.
- A worked memory example with real DeepSeek-V2 numbers.

## Notation

All formulas in this folder use these symbols (values are DeepSeek-V2's, since that's
where MLA shipped):

| Symbol | Meaning | DeepSeek-V2 value |
|---|---|---|
| $d$ | model hidden dimension | 5120 |
| $n_h$ | number of attention heads | 128 |
| $d_h$ | dimension per head | 128 |
| $d_c$ | MLA's KV latent dimension | 512 |
| $d^R_h$ | decoupled-RoPE dimension per head | 64 |
| $L$ | sequence length | — |

---

## Background: the KV cache is the bottleneck

### Why a cache exists

Start with what decoding costs **without** one. The model takes raw token IDs as input,
so to produce token $t+1$ you must run a forward pass over the entire prefix
$1 \dots t$. Inside every layer, that pass computes attention for *every* prefix token
against *every* token before it — the full triangular $t \times t$ score matrix. You
can't shortcut it to "just the last token, please": token $t$'s representation at layer
$\ell$ needs the keys/values of tokens $1 \dots t-1$ *at layer $\ell$*, those need the
same tokens' attention **outputs** at layer $\ell - 1$, and so on all the way down. So
each decode step costs $O(t^2)$ — and almost all of it is **identical to what you
computed the step before**. With causal masking, a past token never attends to anything
new, so token 5's key, value, and attention output are bit-for-bit the same whether the
sequence is 6 tokens long or 6000.

The KV cache simply stops redoing that work. Compute each token's $k_t, v_t$ once, when
it's generated, and store them. A decode step then computes q/k/v for **the one new
token only**, appends $k_t, v_t$ to the cache, and computes a single new **row** of the
attention matrix: $q_t$ against all $t$ cached keys. That's $O(t)$ per step — and the
time it takes is dominated by *reading* the cache out of GPU memory, not by the
arithmetic. (That fact is the centerpiece of the next subsection.)

The accounting, per step and summed over generating a $T$-token sequence:

| | step $t$ costs | whole generation |
|---|---|---|
| no cache | $O(t^2)$ | $O(T^3)$ |
| KV cache | $O(t)$ | $O(T^2)$ |

Note what the cache does **not** do: attention is still quadratic in total. Every pair
of tokens still meets exactly once — each step contributes one new row, and over the
full generation those rows stack into the same triangular $T \times T$ score matrix,
computed exactly once instead of over and over. The cache eliminates
*re*-computation, not computation. Shrinking the genuinely-quadratic part is a
different problem — that's what the sparse-attention lineage (§0.2 onward) attacks.

### Why it dominates

Count the scalars stored:

$$\text{cache size} = \underbrace{2}_{K\text{ and }V} \cdot n_{\text{layers}} \cdot \underbrace{n_h \cdot d_h}_{\text{per-token width}} \cdot L \cdot \text{batch}$$

Two facts make this the cost that matters:

1. **It grows linearly with $L$.** At DeepSeek-V4's 1M-token target, the cache dwarfs the
   model weights.
2. **Decoding is memory-bandwidth-bound.** Generating one token does almost no
   arithmetic, but must stream the *entire cache* through the GPU's memory bus. Cache
   size directly sets tokens/sec and how many requests fit in a batch — halve the cache
   and you roughly double throughput.

> **Worked number.** $n_{\text{layers}} = 60$, $n_h = 128$, $d_h = 128$, bf16 (2 bytes),
> $L = 131072$ (128K), batch 1:
> $2 \cdot 60 \cdot 128 \cdot 128 \cdot 131072 \cdot 2 \text{ bytes} \approx \mathbf{515\ GB}$ —
> for *one* sequence. The whole field of efficient attention exists because of this number.

Every technique in this reading path shrinks one factor of that product. This is the map
for the entire folder:

```mermaid
flowchart TB
    formula["cache = 2 · layers · (heads · head-dim) · seq-len"]
    formula --> ax1["shrink heads · head-dim<br/>by sharing across heads"]
    formula --> ax2["shrink heads · head-dim<br/>by low-rank compression"]
    formula --> ax3["shrink seq-len<br/>(fewer entries to read)"]
    ax1 --> mqa["MQA / GQA"]
    ax2 --> mla["MLA — this section"]
    ax3 --> nsa["NSA, DSA → CSA / HCA<br/>(§0.2 onward)"]
    style mla fill:#a4d4a4,color:#000
```

---

## Refresher: MQA and GQA

You're familiar with these, so only the framing matters. The cache stores a separate K
and V **per head** — MQA/GQA attack the $n_h$ factor directly. **MQA:** all query heads
share one K/V head ($n_h \to 1$, up to $128\times$ smaller, but quality suffers).
**GQA:** $g$ groups of query heads each share one K/V head ($n_h \to g$, e.g. 8 — the
Llama/Mistral standard). The trade is explicit: heads lose their *distinct* keys and
values, so you buy memory with expressiveness. MLA's claim is that you don't have to
make that trade.

---

## MLA: cache one latent per token

MLA's bet: the $n_h \cdot d_h = 16384$ scalars of per-token keys (and another 16384 of
values) are highly redundant — they can be regenerated from a single **512-dim latent**.
Store only the latent; reconstruct K and V on demand. A small extra matmul at read time
for a $\sim 57\times$ smaller cache is exactly the right trade when you're
bandwidth-bound, and the absorption trick below makes even that matmul disappear.

### The compression

Down-project each token's hidden state to the latent; that latent is **the only thing
cached**:

$$c^{KV}_t = h_t W^{DKV} \in \mathbb{R}^{d_c} \qquad (5120 \to 512)$$

Per-head keys and values are up-projected from it when needed — for head $i$:

$$k_{t,i} = c^{KV}_t W^{UK}_i, \qquad v_{t,i} = c^{KV}_t W^{UV}_i \qquad (512 \to 128)$$

Queries get the same low-rank treatment (it saves activation memory in training, and —
the reason we care — **this exact latent reappears in CSA eq. 13**):

$$c^Q_t = h_t W^{DQ}, \qquad q_{t,i} = c^Q_t W^{UQ}_i$$

### The key trick: weight absorption

Reconstructing K per head sounds like extra work per decode step. The trick is that you
never do it. Write out the attention score for head $i$ between query $t$ and a cached
token $s$ (row-vector convention, scaling omitted):

$$q_{t,i}\, k_{s,i}^\top = \big(c^Q_t W^{UQ}_i\big)\big(c^{KV}_s W^{UK}_i\big)^\top = c^Q_t \underbrace{\big(W^{UQ}_i W^{UK\top}_i\big)}_{\text{fixed: precompute once}} c^{KV\top}_s$$

The bracketed product doesn't depend on $t$ or $s$, so fold it into the query path and
define an **effective query** per head:

$$\tilde q_{t,i} = c^Q_t \big(W^{UQ}_i W^{UK\top}_i\big) \in \mathbb{R}^{d_c} \qquad \Rightarrow \qquad \text{score} = \tilde q_{t,i}\, c^{KV\top}_s$$

Attention now runs **directly against the cached latents** — keys are never
materialized. The same absorption folds $W^{UV}$ into the output projection $W^O$, so
values aren't materialized either.

Here is the reframe worth remembering: **at inference, MLA *is* MQA with a 512-dim
shared key — the latent itself.** Every head reads the same cached vector, like MQA. But
unlike MQA, each head applies its *own* absorbed matrix $W^{UQ}_i W^{UK\top}_i$, so each
head still attends with a genuinely different function. MQA's cache size, MHA's
per-head expressiveness.

### The RoPE wrinkle

Absorption needs the two weight matrices to sit *adjacent* in the score so they can be
premultiplied. RoPE breaks that: it inserts a position-dependent rotation $R_t$ after
each projection, and the score becomes

$$q_{t,i}\, k_{s,i}^\top = c^Q_t\, W^{UQ}_i\, \underbrace{R_t R_s^\top}_{=\,R_{t-s}} W^{UK\top}_i\, c^{KV\top}_s$$

The rotation between the weights changes with relative position $t-s$, so there's no
fixed matrix to precompute. MLA's fix is **decoupled RoPE**: split each head's score
into two slices that are summed before the softmax —

$$\text{score}(t, s, i) = \underbrace{\tilde q_{t,i}\, c^{KV\top}_s}_{\text{content slice: NoPE, absorbed, latent space}} + \underbrace{q^R_{t,i}\, k^{R\top}_s}_{\text{position slice: RoPE, tiny}}$$

- The **content slice** is everything above — no positional encoding, absorption works.
- The **position slice** is small ($d^R_h = 64$ dims): per-head RoPE queries $q^R_{t,i}$,
  and a single RoPE key $k^R_t$ **shared by all heads** (MQA-style), cached alongside
  the latent.

So the full cache per token per layer is $d_c + d^R_h = 512 + 64 = 576$ scalars.

> Hold onto "RoPE on a small slice of dims, NoPE on the rest" — DeepSeek-V4 keeps exactly
> this idea under the name *partial RoPE* (§2.3.3).

### The whole machine in one picture

```mermaid
flowchart LR
    subgraph write["Write path — once per token t"]
        h["h_t (5120)"] --> ckv["c_KV_t (512)<br/>via W_DKV"]
        h --> kr["k_R_t (64)<br/>RoPE, shared by all heads"]
    end
    subgraph cache["KV cache: 576 scalars / token / layer"]
        cyl1[("c_KV — content latent")]
        cyl2[("k_R — position key")]
    end
    ckv --> cyl1
    kr --> cyl2
    subgraph read["Read path — every decode step, per head i"]
        qt["effective query q̃_t,i (512)<br/>absorbed: W_UQ · W_UK already folded in"]
        qr["RoPE query q_R_t,i (64)"]
        score["scores = content + position<br/>softmax over s = 1..t"]
        qt --> score
        qr --> score
        out["output — W_UV absorbed into W_O"]
        score --> out
    end
    cyl1 --> score
    cyl2 --> score
```

Keys and values exist only implicitly — nothing in the read path ever has shape
$n_h \times d_h$ per cached token.

---

## Worked example: the memory win (DeepSeek-V2 numbers)

KV cache per token, per layer, in scalars:

| Scheme | Per-token-per-layer cache | Relative |
|---|---|---|
| MHA | $2\, n_h d_h = 2 \cdot 128 \cdot 128 = 32768$ | 1× |
| GQA (8 groups) | $2\, g\, d_h = 2 \cdot 8 \cdot 128 = 2048$ | 16× smaller |
| **MLA** | $d_c + d^R_h = 512 + 64 = 576$ | **~57× smaller** |

MLA beats even aggressive GQA *and* keeps per-head expressiveness via absorption rather
than collapsing heads. Small cache **and** strong quality is why it became the DeepSeek
house style and the foundation everything in this folder builds on.

---

## Key takeaways

- The KV cache grows linearly with context, and decoding is **memory-bandwidth-bound**:
  streaming the cache *is* the bottleneck, so cache size sets throughput.
- **MQA/GQA** shrink the cache by sharing K/V across heads — trading expressiveness for
  memory.
- **MLA** caches one low-rank latent $c^{KV}_t$ per token. **Weight absorption** lets
  attention run directly on the cached latents: at inference it behaves like MQA with the
  latent as the shared key, but each head keeps its own absorbed projection — no
  expressiveness trade.
- RoPE can't be absorbed, so MLA splits each score into a NoPE content slice plus a tiny
  **decoupled RoPE** slice ($+64$ dims, shared key) — the direct ancestor of
  DeepSeek-V4's *partial RoPE*.
- The query latent $c^Q_t$ is **reused verbatim in CSA** (eq. 13), shared between the
  sparse-selection indexer and the main attention.
- MQA/GQA cut the head axis; MLA cuts the rank axis. The remaining axis is **sequence
  length** — that's NSA, next.

---

← Previous: [Reading path & overview](paper_info.md) · Next: [§0.2 — NSA (Native Sparse Attention)](section_0_2_nsa.md) →
