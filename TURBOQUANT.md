# TurboQuant: Deep Technical Reference

## 1. The Vision — Redefining AI Efficiency with Extreme Compression

TurboQuant originates from the paradigm articulated in the [Google Research blog: "TurboQuant: Redefining AI Efficiency with Extreme Compression"](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/). The problem it attacks is simple and brutal: **KV cache grows linearly with context length**. For a model with 96 heads, head_dim=128, at fp16 precision, every token in the KV cache costs **512 bytes**. At one million tokens, that's half a gigabyte just for the KV state — before you even count weights.

The TurboQuant vision: **make context length free**. Reduce KV cache from hundreds of bytes per token to ~90 bytes per token, enabling 5–8x compression with negligible attention quality loss. For autoresearch on Apple Silicon, this is the difference between running at 8K context and hitting 64K+.

TurboQuant achieves this via a three-stage pipeline:

1. **Rotation** — whiten activations into i.i.d. Gaussian so scalar quantization is near-optimal
2. **Lloyd-Max Scalar Quantization** — optimal non-uniform centroids for the Gaussian distribution
3. **QJL Residual Correction** — 1-bit Johnson-Lindenstraurs projection to recover quantization error in the inner product

The insight is elegant: if the rotated vector distribution is approximately Gaussian, then **optimal scalar quantization for a Gaussian is provably near-optimal** for attention — no vector codebooks, no product quantization overhead, just packed bit indices and a small centroid table.

---

## 2. The Math — Rotation + Lloyd-Max + QJL

### 2.1 Rotation (Orthogonal Whitening)

Given a vector `x ∈ R^D`, the goal is to apply an orthogonal transformation `Pi` that makes all coordinates approximately i.i.d. N(0,1). This is done via QR decomposition:

```
Pi, R = qr(RandomNormal(D, D, seed))
# Sign correction: ensure det(Pi) = +1 (proper rotation)
if det(Pi) < 0: Pi[:, 0] *= -1
```

Properties:
- `Pi` is exactly orthogonal: `Pi.T @ Pi = I`
- `x_rot = x @ Pi.T` spreads outlier mass uniformly across all coordinates
- After rotation, scalar MSE quantization performs nearly as well as optimal vector quantization
- Rotation matrix is generated on CPU once per head_dim, then reused

Why QR rotation? A random orthogonal matrix is a *spreading operator*: it maps sparse outliers into dense approximately-Gaussian coordinates. The QR decomposition of a Gaussian random matrix is the canonical way to produce a Haar-distributed random orthogonal matrix.

### 2.2 Lloyd-Max Scalar Quantization

For a random variable X with PDF f(x), Lloyd-Max minimizes:

```
MSE = E[(X - Q(X))^2] = Σ_i ∫_{b_{i-1}}^{b_i} (x - c_i)^2 f(x) dx
```

where `{b_i}` are decision boundaries and `{c_i}` are reconstruction centroids. The optimal solution satisfies two conditions:

**Nearest neighbor condition** (boundaries): Each boundary is the midpoint of adjacent centroids:
```
b_i = (c_i + c_{i+1}) / 2
```

**Centroid condition** (centroids): Each centroid is the conditional mean:
```
c_i = E[X | X ∈ [b_{i-1}, b_i]] = ∫_{b_{i-1}}^{b_i} x·f(x) dx / ∫_{b_{i-1}}^{b_i} f(x) dx
```

Lloyd's algorithm alternates between these two conditions until convergence. For the standard Gaussian PDF, the pre-computed codebook values are in `codebook.py` (see Section 6).

**Scaling for attention**: The codebook is computed for N(0,1) and then scaled by `1/sqrt(head_dim)`. For D=128, this is `0.0883883…`. This scaling matches the expected norm of a rotated activation vector.

### 2.3 QJL (Quantization via Johnson-Lindenstrauss) Residual Correction

After MSE quantization, there is residual error:
```
r = x_rot - dequant_mse(x_rot)
```

QJL captures this residual with just **1 bit per dimension** using the Johnson-Lindenstrauss lemma:

1. **JL Projection**: `p = r @ S.T` where S is a random Gaussian matrix (D, D)
2. **1-bit Packing**: `sign_bits = pack(p >= 0)` — 32 signs per uint32 word
3. **Score Correction**: During attention, the JL lemma guarantees:
   ```
   E[sign(p) · q_sketch] ≈ <q, r> · sqrt(2/π)
   ```

The full QJL score correction for query q against key k:
```
score_qjl = sqrt(π/2) / D · (q @ S.T) @ sign(S @ residual).T · ||residual||
```

Where:
- `q_sketch = q @ (S @ Pi).T` — JL sketch of the query (combined with rotation in one matmul)
- `sign_bits` — packed sign bits of projected residuals for each key
- `γ = ||residual||` — L2 norm stored per token
- `sqrt(π/2)/D` — scaling from the JL inner product estimator

**Combined rotation+JL matrix**: `[Pi; S @ Pi]` of shape (2D, D) lets query rotation and sketching happen in a single matmul: `q_combined = q @ combined_rot_jl.T`, producing shape `(B, H, T, 2D)`.

**For T_q=1 (decode step)**: A fused Metal kernel (`fused_qjl.py`) computes the QJL dot product directly from packed uint32 sign bits without unpacking — avoiding 32x memory blowup.

### 2.4 TurboQuant_mse vs TurboQuant_prod

| Mode | Keys | Values | Total Bits |
|------|------|--------|------------|
| `TurboQuant_mse` | b-bit Lloyd-Max | b-bit Lloyd-Max | 2b |
| `TurboQuant_prod` | (b-1)-bit MSE + QJL | b-bit MSE | ~2b |

The product-inspired mode sacrifices 1 MSE bit on keys to add QJL correction. However, benchmarks show a **critical finding** (see Section 4): at D=128 and D=256, prod mode is **worse** than pure MSE because losing 1 MSE bit (e.g., 8→4 levels for 3→2 bit) costs more than the QJL correction recovers.

---

## 3. The Three Variants

### 3.1 V1 — TurboQuantKVCache (cache.py, 226 lines)

**Storage**: Custom codebook quantization with 2-bit/3-bit packed indices in uint32.

**Constructor**: `TurboQuantKVCache(head_dim=128, mse_bits=2, use_qjl=True, seed=42)`

**Key characteristics**:
- **No pre-allocation** — buffers grow via `mx.concatenate()` every `update_and_fetch()` call, causing O(T) reallocations
- Uses **custom Metal kernels** (`kernels.py`) for binary-search quantization and index packing
- `turboquant_encode()` → L2 normalize → rotate → Metal-kernel binary search on boundaries → pack
- Dequantization: unpack indices → centroid lookup → inverse rotation
- **Storage per token** (D=128, 2-bit with QJL): **92 bytes** vs 512 bytes fp16 = **5.6x compression**
- Returns full dequantized `(keys, values)` from `update_and_fetch()`
- Requires explicit `mx.eval(*state)` after concatenation (Metal memory barrier issue)
- mlx-lm compatible: `state`, `meta_state`, `is_trimmable()`→False, `offset`, `nbytes`
- GQA support via `n_repeats = n_q_heads // n_kv_heads`

**Metal kernels used**: quantize (binary search), pack_signs, pack_2bit, pack_3bit, fused_attention, fused_attention_norot, fused_value

### 3.2 V2 — TurboQuantKVCacheV2 (cache_v2.py, 273 lines)

**Storage**: MLX-native `mx.quantize()` / `mx.dequantize()` / `mx.quantized_matmul()` with affine quantization.

**Constructor**: `TurboQuantKVCacheV2(head_dim=128, bits=3, group_size=64, use_qjl=False, use_rotation=True, use_normalization=True, seed=42)`

**Key characteristics**:
- **Pre-allocation with step=256** — buffers allocated in chunks (O(T/256) reallocations vs O(T))
- Uses `mx.quantize()` returning `(data, scales, biases)` — MLX affine quantization format
- Uses `mx.quantized_matmul(q, *quant_tuple, transpose=True/False, group_size, bits)` for scoring and value output
- **Normalization mode**: L2-normalize before rotation, store norms separately, "bake" norms into scales/biases at retrieval:
  ```
  norm * dequant(data, scale, bias) = dequant(data, norm*scale, norm*bias)
  ```
- **Lean mode** (`use_normalization=False`): direct quantization without L2 normalization
- Optional rotation; optional QJL (stores sign_bits + residual_norms)
- **Returns quantized tuples** from `update_and_fetch()` — attention calls `mx.quantized_matmul()` directly on the tuples
- `is_trimmable()`→True with `trim(n)` support
- No custom Metal kernels for quantization (except `fused_qjl.py` for score correction)

### 3.3 V3 — TurboQuantKVCacheV3 (cache_v3.py, 423 lines)

**Storage**: Lloyd-Max codebook quantization (paper-correct) with **pure MLX ops** — no Metal kernels.

**Constructor**: `TurboQuantKVCacheV3(head_dim=128, bits=2, use_qjl=False, n_outlier=0, outlier_bits=3, seed=42)`

**Key characteristics**:
- **Pre-allocation with step=256** (same as V2)
- Pure MLX ops — no custom Metal kernels for quantization/unpacking
- Lloyd-Max scalar quantization with pre-computed codebook from `codebook.py`
- **Mixed-bit allocation** (channel splitting): first `n_outlier` channels at `outlier_bits`, rest at `bits`:
  ```
  effective_bits = (n_outlier * outlier_bits + n_regular * bits) / head_dim
  ```
  Example: 2.5-bit = 64 channels @ 3-bit + 64 channels @ 2-bit
- **TurboQuant_prod mode**: keys at `(b-1)`-bit MSE + QJL, values at `b`-bit MSE
- **Incremental centroid caching**: `_key_centroids_cache` and `_value_centroids_cache` updated every `update_and_fetch()`, so `get_key_centroids()` returns precomputed O(1) buffers
- Returns full dequantized `(keys, values)` from `update_and_fetch()` (same interface as V1)
- Attention uses cached centroids directly — already dequantized
- `is_trimmable()`→True with `trim()`
- `_quantize_and_pack()` returns `(out_packed, reg_packed, centroid_vals)` — unified for mixed mode

### API Comparison

| Feature | V1 | V2 | V3 |
|---------|----|----|----|
| Quant method | Metal kernel binary search on boundaries | `mx.quantize()` (affine) | Lloyd-Max codebook (pure MLX) |
| Pre-allocation | No (concat every call) | Yes (step=256) | Yes (step=256) |
| Return type | Dequantized `(keys, values)` | Quantized tuples `(q_keys, q_values)` | Dequantized `(keys, values)` |
| Norm handling | Built into encoding | Optional (`use_normalization`) | Always (via `safe_normalize()`) |
| Rotation | Always | Optional | Always |
| QJL | Supported | Supported | Supported |
| Mixed-bit | No | No | Yes |
| TurboQuant_prod | No | No | Yes |
| Centroid caching | No | N/A | Yes (incremental) |
| Trimmable | No | Yes | Yes |
| Metal kernels | Yes | One (fused_qjl) | No |
| Lines of code | 226 | 273 | 423 |

---

## 4. Critical Benchmark Findings

### 4.1 Prod Mode is Worse Than Pure MSE

The most significant empirical finding across all benchmark runs: **TurboQuant_prod mode (keys at b-1 bit + QJL, values at b-bit) consistently underperforms pure TurboQuant_mse mode.**

At D=128 and D=256:
- Dropping keys from 3-bit to 2-bit loses significant precision (8→4 centroid levels)
- The QJL 1-bit correction cannot compensate for this loss
- The JL inner product estimator has variance that scales with `1/D` — at these dimensions the correction is noisy
- Pure MSE at the higher bit rate is more reliable

This was confirmed in `tests/test_turboquant.py` (1089+ lines) across D=128 and D=256, multiple random seeds, and multiple text PPL evaluations. **Recommendation: use pure MSE mode (`use_qjl=False`) unless context is extremely long where memory pressure dominates quality.**

### 4.2 V2 Affine vs V3 Lloyd-Max

`experiment_2bit.py` compared affine quantization (V2's `mx.quantize()`) vs Lloyd-Max (V3's codebook approach):
- Lloyd-Max consistently achieves lower MSE for the same bit budget on Gaussian-like distributions
- The centroid spacing adapts to the actual PDF rather than uniform spacing
- For rotated activations that are approximately Gaussian, Lloyd-Max is provably near-optimal

### 4.3 Long-Context Scaling

`benchmark_longseq.py` measured throughput at context lengths 512/1024/2048/4096/8192:
- V2's pre-allocation (step=256) provides consistent O(T/256) reallocation profile vs V1's O(T)
- At long context, the memory savings from 2-3 bit quantization become decisive
- V3's centroid caching eliminates per-attention dequantization cost

### 4.4 Models Tested

- Llama 3.2 3B Instruct (4-bit weight quantized)
- Llama 3.1 8B
- Mistral 7B
- gemma-3 4B

Across all models, the compression ratio holds (5-8x depending on bits) and PPL degradation is bounded.

---

## 5. Enabling Autoresearch at Longer Context on Apple Silicon

Apple Silicon's Unified Memory Architecture makes it uniquely suited for TurboQuant-MLX:

### Memory Economics

| Config | Bytes/Token | 8K Tokens | 64K Tokens |
|--------|------------|-----------|------------|
| fp16 | 512 | 4 MB | 32 MB |
| 4-bit affine (V2) | ~128 | 1 MB | 8 MB |
| 2-bit Lloyd-Max (V3) | ~64 | 0.5 MB | 4 MB |

For autoresearch experiments that iterate over long documents, the 8x saving from fp16→2-bit frees memory for:
- **Longer prompts**: Process 64K-token research documents instead of hitting 8K limits
- **Larger batch sizes**: Evaluate more configurations simultaneously
- **Multi-model runs**: Compare different architectures within the same memory budget
- **Avoidance of swap**: MLX performance degrades sharply when hitting swap

### Pipeline Integration

TurboQuant integrates into autoresearch via the `patch.py` monkey-patch mechanism:

```python
import turboquant.patch as tq_patch
tq_patch.apply()  # Intercepts mlx-lm SDPA calls
```

The patch intercepts `mlx_lm.models.base.scaled_dot_product_attention` and dispatches based on cache type:
- `TurboQuantKVCacheV3` → `turboquant_v3_sdpa()`
- `TurboQuantKVCacheV2` → `turboquant_v2_sdpa()`
- `TurboQuantKVCache` → `turboquant_fused_sdpa()`
- Other caches → fall through to original

This means autoresearch scripts using standard `mlx_lm.generate()` automatically benefit from TurboQuant compression — no code changes to the generation loop.

### Strategy Selection for Autoresearch

```python
from turboquant.benchmark_common import make_cache

# Best quality: V3 at 3-bit pure MSE
cache = make_cache(model, "tqv3_3bit")

# Maximum compression: V3 at 2-bit
cache = make_cache(model, "tqv3_2bit")

# Balanced: V2 at 4-bit lean
cache = make_cache(model, "tqv2_4bit_lean")

# DO NOT use: prod mode (benchmarks show it's worse than pure MSE)
# cache = make_cache(model, "tqv3_3bit_prod")  # Avoid
```

---

## 6. Lloyd-Max Codebook Tables

All values computed via Lloyd's algorithm with `scipy.integrate.quad` for N(0,1) Gaussian. The codebook is scaled by `1/sqrt(head_dim)` at runtime: for D=128, scale = `0.0883883`.

### 1-bit (2 levels, 1 boundary)

| Index | Centroid | Boundary |
|-------|----------|----------|
| 0 | -0.7978845608028654 | — |
| 1 | +0.7978845608028654 | 0.0 |

### 2-bit (4 levels, 3 boundaries)

| Index | Centroid |
|-------|----------|
| 0 | -1.510417608611893 |
| 1 | -0.4527800346911237 |
| 2 | +0.4527800346911237 |
| 3 | +1.510417608611893 |

Boundaries: `[-0.9815988216515084, 0.0, 0.9815988216515084]`

### 3-bit (8 levels, 7 boundaries)

| Index | Centroid |
|-------|----------|
| 0 | -2.151945705166112 |
| 1 | -1.3439092791423422 |
| 2 | -0.7560052816730181 |
| 3 | -0.2450941791152904 |
| 4 | +0.2450941791152904 |
| 5 | +0.7560052816730181 |
| 6 | +1.3439092791423422 |
| 7 | +2.151945705166112 |

Boundaries: `[-1.74793, -1.04996, -0.50055, 0.0, 0.50055, 1.04996, 1.74793]`

### 4-bit (16 levels, 15 boundaries)

| Index | Centroid |
|-------|----------|
| 0 | -2.732896755154294 |
| 1 | -2.069364258154187 |
| 2 | -1.618400443227723 |
| 3 | -1.2565648452462146 |
| 4 | -0.9426291036999694 |
| 5 | -0.6569817464411519 |
| 6 | -0.3881887141600061 |
| 7 | -0.1284430012487642 |
| 8 | +0.1284430012487642 |
| 9 | +0.3881887141600061 |
| 10 | +0.6569817464411519 |
| 11 | +0.9426291036999694 |
| 12 | +1.2565648452462146 |
| 13 | +1.618400443227723 |
| 14 | +2.069364258154187 |
| 15 | +2.732896755154294 |

Boundaries: `[-2.40113, -1.84388, -1.43748, -1.09960, -0.79981, -0.52259, -0.25832, 0.0, 0.25832, 0.52259, 0.79981, 1.09960, 1.43748, 1.84388, 2.40113]`

**API**:
- `get_codebook(bits, head_dim)` — returns scaled centroids (centroids / sqrt(head_dim))
- `get_codebook_unscaled(bits)` — returns raw N(0,1) centroids
- `quantize_to_indices(values, boundaries)` — assigns each value to the nearest centroid via cascaded boundary comparison (pure MLX, avoids giant broadcast tensor)

---

## 7. Metal Kernel Design Patterns

TurboQuant defines 8 Metal kernel families in `kernels.py`, plus 2 specialized kernels in `fused_qjl.py` and `fused_v2_attn.py`. All use the Metal C++ dialect via `mx.fast.metal_kernel()`.

### 7.1 Elementary Kernels

**turboquant_quantize** — Binary search quantization
- Input: `rotated` (flat float32), `boundaries` (1D float32)
- Output: `indices` (flat uint8)
- Grid: `(flat.size, 1, 1)`, threadgroup: `(min(256, flat.size), 1, 1)`
- Each thread counts how many boundaries its value exceeds — linear scan on small boundary arrays (3 for 2-bit, 7 for 3-bit, 15 for 4-bit)

**turboquant_pack_signs** — Pack 32 sign bits per uint32
- Grid: `(num_words, 1, 1)`, threadgroup: `(min(256, num_words), 1, 1)`
- Each thread processes one uint32 word, packing 32 `(values[idx] > 0)` bits

**turboquant_pack_2bit** — Pack 16 2-bit indices (0–3) per uint32
- Each thread packs `indices[base:base+16]` with 2 bits each
- Unpacking uses pure MLX bitwise ops: `(packed >> shifts) & 0x3`

**turboquant_pack_3bit** — Pack 10 3-bit indices (0–7) per uint32 (2 bits unused)
- Handles padding to multiple of 10, boundary checks
- Unpacking: MLX bitwise ops `(packed >> shifts) & 0x7`, then trim

**turboquant_fused_scores** (DEAD) — Fused MSE + QJL scoring kernel
- Marked `_FUSED_SCORE_SOURCE_DEAD`, wrapper exists but not used in main attention paths

**turboquant_fused_value** — Fused value output
- Decode centroids + inverse rotation + weighted sum in one dispatch
- Grid: `(D, n_repeats, 1)`, threadgroup: `(min(128, D), 1, 1)`

### 7.2 Fused Attention Kernels

**turboquant_fused_attention** — Fully fused decode-step attention
- Rotation, scoring, online softmax, value accumulation, inverse rotation in ONE dispatch
- Uses `<metal_simdgroup>` and `metal::fast::exp()`
- Grid: `(n_q_heads * 32, 1, 1)`, threadgroup: `(32, 1, 1)`
- One simdgroup per query head; each thread handles 4 dimensions (32 × 4 = 128)
- `simd_sum()` for cross-thread dot product reduction
- `shared_q[128]` and `shared_acc[128]` threadgroup memory for rotation

**turboquant_fused_attention_norot** (HIGH-OCCUPANCY) — Rotation done externally via MLX GEMM
- Grid: `(n_q_heads * 1024, 1, 1)`, threadgroup: `(1024, 1, 1)`
- 32 simdgroups × 32 lanes = 1024 threads per query head
- 32 simdgroups process keys IN PARALLEL with stride-32 loop
- Cross-simdgroup reduction via threadgroup memory: `tg_max[32]`, `tg_sum[32]`, `tg_acc[32 * 128]`
- `threadgroup_barrier(mem_flags::mem_threadgroup)` for synchronization

**fused_qjl_scores** (from `fused_qjl.py`) — QJL dot product from packed uint32
- Directly reads packed sign bits without unpacking (avoids 32x memory blowup)
- Grid: `(total_bhr * 32, T_kv, 1)`, threadgroup: `(32, 1, 1)`
- One simdgroup per (batch, kv_head, repeat); each lane processes D/32 dimensions
- `simd_sum()` reduction

**fused_v2_attn** (from `fused_v2_attn.py`) — Fully fused V2 kernel
- Replaces 5-step pipeline: `quantized_matmul(key)` + `fused_qjl` + add + softmax + `quantized_matmul(value)`
- Uses `extract_bits()` helper for cross-word-bit-boundary reads (critical for 3-bit: 30 bits/word means values straddle word boundaries)
- 32 simdgroups × 32 lanes = 1024 threads, matching MLX's sdpa_vector.h pattern
- Online softmax, affine dequant, QJL, and value accumulation in one pass

### 7.3 Design Principles

1. **Metal only for bit-level ops**: Packing, quantization, sign-bit dot products. All matmuls use MLX's optimized GEMM.
2. **Simdgroup parallelism**: 32 lanes per simdgroup, matching Apple Silicon's execution width
3. **Threadgroup memory for reductions**: `tg_max`, `tg_sum`, `tg_acc` arrays enable cross-simdgroup communication
4. **Online softmax**: Numerically stable — avoids materializing full score matrices
5. **Avoid 32x expansion**: Fused kernels work on packed uint32 directly
6. **Fallback paths**: All fused kernels have pure-MLX fallbacks for wider shapes

---

## 8. Using turboquant-mlx in the Autoresearch Pipeline

### 8.1 Quick Start

```python
import turboquant.patch as tq_patch
tq_patch.apply()  # Monkey-patches mlx-lm SDPA

from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# TurboQuant cache is created by mlx-lm's KVCache mechanism
# When the patch is active, SDPA dispatches to TurboQuant attention automatically
response = generate(model, tokenizer, prompt="Research topic: ...", verbose=True)
```

### 8.2 Strategy Selection (benchmark_common.py)

The `make_cache(model, strategy)` factory supports 30+ configurations:

| Strategy | Description | Bits/Token |
|----------|-------------|------------|
| `fp16` | Baseline, no compression | 16 |
| `quant4` | Standard MLX 4-bit quantized cache | 4 |
| `tqv2_4bit` | V2, 4-bit affine, rotation + norm | 4 |
| `tqv2_4bit_lean` | V2, 4-bit affine, no norm | 4 |
| `tqv2_3bit_lean` | V2, 3-bit affine, no norm | 3 |
| `tqv3_3bit` | V3, 3-bit Lloyd-Max, pure MLX | 3 |
| `tqv3_2bit` | V3, 2-bit Lloyd-Max, pure MLX | 2 |
| `tqv3_3bit_prod` | V3, prod mode (AVOID — worse than pure MSE) | ~3 |
| `tqv3_2bit_prod` | V3, prod mode (AVOID) | ~2 |

### 8.3 Manual Cache Construction

```python
from turboquant.cache_v3 import TurboQuantKVCacheV3
from turboquant.benchmark_common import cache_nbytes, make_cache

# V3: 3-bit Lloyd-Max, no QJL (recommended for best quality)
cache_layer = TurboQuantKVCacheV3(head_dim=128, bits=3, use_qjl=False, seed=42)

# V3: 2.5-bit mixed — 64 channels at 3-bit, 64 at 2-bit
cache_layer = TurboQuantKVCacheV3(
    head_dim=128, bits=2, use_qjl=False,
    n_outlier=64, outlier_bits=3, seed=42
)

# Check memory usage
bytes_per_token = cache_layer.nbytes / num_cached_tokens
```

### 8.4 Custom Attention Pipeline

```python
from turboquant.attention_v3 import turboquant_v3_sdpa
from turboquant.cache_v3 import TurboQuantKVCacheV3

# In your generation loop:
cache = TurboQuantKVCacheV3(head_dim=128, bits=3)
keys, values = cache.update_and_fetch(new_keys, new_values)

# Attention on cached (dequantized) centroids
output = turboquant_v3_sdpa(queries, cache, scale=1.0/sqrt(128))
```

### 8.5 Benchmarking

```bash
# Full benchmark with PPL evaluation
python benchmark.py

# Long context throughput: 512 → 8192 tokens
python benchmark_longseq.py

# Multi-model PPL comparison
python benchmark_models.py

# Quantization error analysis
python experiment_2bit.py

# QJL effectiveness study
python experiment_qjl.py
```

### 8.6 Key Recommendations for Autoresearch

1. **Use V3 for experiments**: Pure MLX, no Metal kernel compilation issues, mixed-bit support, centroid caching
2. **Use 3-bit pure MSE as default**: Best quality/compression tradeoff at D=128
3. **Use 2-bit for very long context**: When memory pressure dominates quality concerns
4. **Avoid prod mode**: Benchmarks across D=128 and D=256 show pure MSE is better
5. **Apply patch once at startup**: `tq_patch.apply()` is idempotent
6. **Monitor `nbytes`**: `cache_layer.nbytes` gives exact memory usage for cache management in autoresearch loops

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 3 | Exports: codebook, rotation, cache |
| `attention.py` | — | V1 pure-MLX attention with matrix associativity optimization |
| `attention_fused.py` | — | V1 fused Metal kernel path (fallback to attention.py) |
| `attention_v2.py` | — | V2 attention via `mx.quantized_matmul()` |
| `attention_v3.py` | — | V3 attention with cached centroids |
| `patch.py` | 44 | Monkey-patches mlx-lm SDPA dispatch |
| `cache.py` | 226 | V1: Metal kernel quantization, no pre-alloc |
| `cache_v2.py` | 273 | V2: mx.quantize, pre-alloc, affine |
| `cache_v3.py` | 423 | V3: Lloyd-Max, mixed-bit, centroid caching |
| `codebook.py` | — | Lloyd-Max codebook tables (1-4 bit) |
| `codebook_ops.py` | 132 | Pure MLX bit ops: quantize, pack, unpack |
| `qjl.py` | 64 | QJL encode via pure MLX (no Metal) |
| `rotation.py` | 90 | QR rotation, JL matrix, safe_normalize |
| `kernels.py` | — | 8 Metal kernel families |
| `fused_qjl.py` | 118 | Fused QJL Metal kernel (packed uint32 dot) |
| `fused_v2_attn.py` | 250 | Fully fused V2 Metal attention |
| `benchmark.py` | 137 | Generation + PPL benchmark |
| `benchmark_common.py` | — | Strategy factory, PPL, byte counting |
| `benchmark_longseq.py` | — | Throughput at 512–8192 context |
| `benchmark_models.py` | — | Multi-model PPL |
| `experiment_2bit.py` | — | Affine vs Lloyd-Max error analysis |
| `experiment_qjl.py` | — | QJL effectiveness study |
| `tests/test_turboquant.py` | 1089+ | Comprehensive unit tests |
| `run_llm.py` | 109 | Demo: Llama 3.2 3B with TurboQuant V2 |
