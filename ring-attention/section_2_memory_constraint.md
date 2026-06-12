# §2 Large Context Memory Constraint

**What this section covers:** the memory accounting that motivates Ring Attention. Two
background concepts you flagged as new get the full treatment here — **online softmax**
(why attention can be computed in pieces and still be exact) and **tiled attention**
(FlashAttention / blockwise attention, which uses it). Then the paper's actual argument:
blockwise compute gets per-layer activations down to $2bsh$ bytes, but the layer-output
wall remains, and that wall is what forces multi-device sequence sharding.

Paper notation used throughout: batch $b$, sequence length $s$, hidden size $h$, head
dimension $d$, block/tile size $c$, number of heads $n$. Byte counts assume bf16
(2 bytes/element).

## Background: online softmax

Softmax over a vector of scores normally needs the whole vector at once — you can't
normalize until you've seen every term in the denominator:

```python
def softmax(x):                  # (s,)  needs ALL s scores in memory
    e = exp(x - x.max())         # (s,)  subtract max for numerical safety
    return e / e.sum()           # (s,)
```

The online trick: walk the scores chunk by chunk, carrying just two scalars — the running
max $m$ and the running denominator $l$. When a new chunk raises the max, the old
denominator was computed against a stale max, so multiply it by `exp(m_old - m_new)` to
re-express it against the new one:

```python
m = -inf                                   # ()    running max
l = 0.0                                    # ()    running denominator
for x_c in chunks(x):                      # (c,)  one chunk of scores at a time
    m_new = max(m, x_c.max())              # ()
    l = l * exp(m - m_new) + exp(x_c - m_new).sum()   # ()  rescale old, add new
    m = m_new                              # ()
# afterwards: softmax(x)[i] == exp(x[i] - m) / l   -- exact, not approximate
```

The rescale works because `exp(x - m_old) * exp(m_old - m_new) == exp(x - m_new)` — every
term already accumulated into `l` gets silently re-based onto the new max. Two properties
fall out, and **both are load-bearing for this paper**:

1. **Exactness** — the final `(m, l)` is identical to the all-at-once computation.
2. **Order independence** — chunks can arrive in *any* order and the final `(m, l)` is the
   same. This is the "permutation invariance" §3 leans on: KV blocks arriving around a
   ring is just one particular order.

## Background: tiled attention (FlashAttention / blockwise attention)

Attention applies softmax row-wise to the score matrix, then averages values:

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d}}\right)V$$

Materializing $QK^T$ costs $(s, s)$ memory — that's the $2bns^2$ in Table 1, and at
$s=$ 1M it's a petabyte-scale tensor. Tiled attention extends the online-softmax state
per query row with a running *output* `o`, so the score matrix only ever exists one
$(c, c)$ tile at a time:

```python
q = Q[i]                               # (c, d)  one query tile, stays put
o = zeros(c, d)                        # (c, d)  running UNNORMALIZED output
m = full((c, 1), -inf)                 # (c, 1)  running row max
l = zeros(c, 1)                        # (c, 1)  running row denominator

for k, v in kv_tiles():                # (c, d), (c, d)  one KV tile at a time
    s_ = q @ k.T / sqrt(d)             # (c, c)  scores for THIS tile only
    m_new = maximum(m, s_.max(-1, keepdim=True))     # (c, 1)
    p = exp(s_ - m_new)                # (c, c)  tile probs, unnormalized
    alpha = exp(m - m_new)             # (c, 1)  rescale factor for old state
    o = o * alpha + p @ v              # (c, d)  rescale old output, add new
    l = l * alpha + p.sum(-1, keepdim=True)          # (c, 1)
    m = m_new                          # (c, 1)

out = o / l                            # (c, d)  exact attention rows -- bit-equal
```

Peak extra memory is the tile state — independent of $s$. The inner loop is exactly what
runs inside the flash kernels our stack already dispatches to in
`F.scaled_dot_product_attention`. And note where Ring Attention is heading: nothing in
the loop body cares *where* `k, v` came from. Swap `kv_tiles()` for "tiles arriving from
the previous device in a ring" and the math is unchanged — that's the whole paper.

## The paper's accounting: what's left after the kernels

With the score matrix gone, the paper tallies what each transformer layer still stores
(the "familiar" refresher — formulas only). The FFN is the next offender: its
intermediate is $4h$ wide,

$$\mathrm{FFN}(x)=\max(0,\,xW_{1}+b_{1})\,W_{2}+b_{2}$$

```python
x   = ...                  # (b, s, h)    layer input
mid = relu(x @ W1)         # (b, s, 4h)   -> 8bsh bytes in bf16: the FFN peak
y   = mid @ W2             # (b, s, h)    -> 2bsh bytes
```

**Blockwise Parallel Transformers (BPT)** — Liu & Abbeel 2023, the same first author —
applies the tiling idea to the FFN too: since FFN is position-wise, run it on one
$(b, c, h)$ block at a time, so the $4h$-wide intermediate only exists for $c$ positions.
Per-layer peak activation drops from $8bsh$ to $2bsh$ — just the layer output itself.
That's the state of the art the paper starts from (first three rows of Table 1; the Ring
Attention row, $6bch$, is §3's result):

| Layer type | Peak activation (bytes/layer) |
| --- | --- |
| Vanilla | $2bhs^{2}$ |
| Memory-efficient attention (FlashAttention) | $8bsh$ |
| + blockwise FFN (BPT) | $2bsh$ |
| Ring Attention (preview of §3) | $6bch$ — no $s$ at all |

## The wall: storing each layer's output

Here's the argument that no further kernel trick can beat. Self-attention is all-to-all
over positions, so layer $\ell+1$ needs **all** of layer $\ell$'s output. Skipping that
storage means recomputing each output for every query that needs it — cubic compute. So
on a single device, $2bsh$ bytes *per layer* is a floor:

```python
# paper's example: b=1, s=100M tokens, h=1024, bf16
one_layer_out = 1 * 100e6 * 1024 * 2   # ~205 GB  for ONE layer's (b, s, h)
# any real depth stores L of these for backward -> >1000 GB by L=5
# A100/H100 HBM: 80 GB. Off by orders of magnitude.
```

The floor scales linearly in $s$ with no device-count term — bigger GPUs won't arrive
fast enough, so the sequence dimension itself has to be cut across devices. §3 shows how
to do that without paying any communication overhead.

## Key takeaways

- Online softmax carries `(m, l)` running statistics and rescales by `exp(m_old - m_new)`;
  it is **exact** and **order-independent** — the two properties Ring Attention is built on.
- Tiled attention adds a running output `o` to that state; peak memory becomes independent
  of $s$. This is what flash kernels (including our SDPA path) implement.
- BPT extends tiling to the FFN: per-layer activations drop to $2bsh$ — just the output.
- The remaining $2bsh$-per-layer **layer-output storage is irreducible on one device**
  (all-to-all attention needs it; not storing it costs cubic recompute) — hence: shard
  the sequence across devices. That's §3.

---

Prev: [§1 Introduction](section_1_introduction.md) · Next: §3 Ring Attention *(not yet written)*
