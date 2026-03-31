# MLX Migration Plan: autoresearch-v2 + turboquant-mlx

## Table of Contents

1. [TurboQuant-MLX: What It Does and Does Not Do](#1)
2. [Three Variants: V1, V2, V3](#2)
3. [Critical Finding: TurboQuant_prod Is Worse Than Pure MSE](#3)
4. [Complete Architecture Translation](#4)
5. [Exact MLX Training Loop Pattern](#5)
6. [TurboQuant-MLX Integration at Eval Time](#6)
7. [Implementation Checklist](#7)

---

## 1. TurboQuant-MLX: What It Does and Does Not Do

### What TurboQuant-MLX DOES: KV Cache Compression

TurboQuant-MLX is exclusively a **KV cache compression library** for autoregressive LLM inference on Apple Silicon. It operates during the generation phase only, compressing cached key-value pairs to reduce memory bandwidth and increase throughput for long-context generation.

Core mechanism (3 steps):

1. **Random orthogonal rotation** (QR decomposition) applied to each KV vector, which uniformizes the distribution from heavy-tailed to approximately Gaussian N(0, 1/sqrt(D))
2. **Lloyd-Max scalar quantization** on the rotated vectors -- optimal for Gaussian distributions, achieving lower MSE than uniform/affine quantization at equal bit widths
3. **Optional Johnson-Lindenstrauss (QJL) residual correction** -- a 1-bit technique that projects the quantization residual through a random matrix, providing a score correction during attention

Why only KV cache: During autoregressive generation, the KV cache grows linearly with sequence length. At T=4096 with head_dim=128 and n_heads=32, the raw fp16 cache is 32MB. TurboQuant reduces this to 2-8MB (4-16x compression) with minimal perplexity degradation (<0.2 PPL at 3-bit).

### What TurboQuant-MLX Does NOT Do: Weight Quantization

TurboQuant-MLX **explicitly does NOT** quantize model weights. Weight quantization is handled by separate systems:

- MLX's mx.quantize() / mx.dequantize() -- affine weight quantization (already used by mlx-lm's 4-bit models)
- Q4_0/Q4_1/Q5_0 formats -- llama.cpp-style block quantization (handled at model loading time)
- AWQ, GPTQ -- post-training weight quantization methods

TurboQuant-MLX is **orthogonal** to weight quantization. You can use TurboQuant KV cache compression on top of an already-weight-quantized model (e.g., mlx-community/Llama-3.2-3B-Instruct-4bit). The two compressions compound: weight quantization reduces model size, TurboQuant reduces cache size.

Key Design Decisions:

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Scope | KV cache only | Inference bottleneck is cache bandwidth, not compute |
| Weights | Untouched | Handled by mlx-lm's weight quantization |
| Precision | 2/3/4-bit cache compression | Below 2-bit quality drops sharply; above 4-bit marginal gains |
| Hardware | Apple Silicon (Metal) | Rotation and MLX ops optimized for unified memory |
| Framework | MLX only | Leverages MLX lazy evaluation and Metal kernel system |

---

## 2. Three Variants: V1, V2, V3

### V1: TurboQuantKVCache -- Custom Codebook + Metal Kernels

Storage model: Custom codebook quantization with 2-bit/3-bit packed indices in uint32.

**Constructor:**

    TurboQuantKVCache(head_dim=128, mse_bits=2, use_qjl=True, seed=42)

**Key characteristics:**

- No pre-allocation -- buffers grow via mx.concatenate() every update_and_fetch call (O(T) reallocations)
- Uses custom Metal kernels (8 families) for packing/unpacking via mx.fast.metal_kernel
- Quantization: norm + rotate + Metal-kernel binary search on sorted boundaries
- Dequantization: unpack indices to centroid lookup + inverse rotation
- Storage per token (head_dim=128, mse_bits=2, use_qjl=True): **92 bytes** vs 512 bytes fp16 = **5.6x compression**
- update_and_fetch returns (keys, values) as FULL dequantized tensors
- Requires explicit mx.eval(*state) after concatenation (Metal barrier)
- mlx-lm interface compatible: state getter/setter, meta_state, is_trimmable=False, offset, nbytes
- GQA support via n_repeats = n_q_heads // n_kv_heads

**When to use:** Baseline for comparison. Demonstrates full TurboQuant pipeline with custom Metal kernels. Not recommended for production due to reallocation overhead.

### V2: TurboQuantKVCacheV2 -- MLX-Native Quantized Tuples

Storage model: MLX-native mx.quantize() / mx.dequantize() with affine quantization.

**Constructor:**

    TurboQuantKVCacheV2(head_dim=128, bits=3, group_size=64, use_qjl=False,
                        use_rotation=True, use_normalization=True, seed=42)

**Key characteristics:**

- Pre-allocation with step=256 -- O(T/256) reallocations vs O(T) for V1
- Uses mx.quantize() returning (data, scales, biases) -- MLX affine quantization format
- Uses mx.quantized_matmul(q, *quant_tuple, transpose=True/False, group_size, bits) for scoring
- Normalization mode: L2-normalize before quantization, bake norms into scales/biases at retrieval
- Lean mode: direct quantization without L2 normalization (faster, lower quality)
- Optional rotation via QR decomposition
- QJL supported with fused Metal kernel for T_q=1
- Returns QUANTIZED tuples (not dequantized) -- attention calls mx.quantized_matmul() directly
- is_trimmable=True with trim(n) method
- fused_v2_attn.py: fully fused Metal kernel replacing 5-step pipeline

**When to use:** PRODUCTION. Best overall performance. mx.quantized_matmul path leverages MLX optimized kernels.

### V3: TurboQuantKVCacheV3 -- Lloyd-Max Codebook with Mixed Bits

Storage model: Lloyd-Max codebook quantization (paper-correct) with pure MLX ops.

**Constructor:**

    TurboQuantKVCacheV3(head_dim=128, bits=2, use_qjl=False,
                        n_outlier=0, outlier_bits=3, seed=42)

**Key characteristics:**

- Lloyd-Max scalar quantization with codebook from codebook.py (optimal for Gaussian)
- Pure MLX ops -- no custom Metal kernels for quantization
- Mixed-bit allocation: e.g., 2.5-bit = 64 channels @ 3-bit + 64 channels @ 2-bit
- TurboQuant_prod mode: keys at (b-1)-bit MSE + QJL, values at b-bit MSE
- Incremental dequant centroid caching: _key_centroids_cache updated every call, O(1) retrieval
- Returns dequantized (keys, values) -- same interface as V1
- Pre-allocation with step=256
- Trimmable: Yes

**When to use:** Research and quality-critical applications. Lloyd-Max achieves 10-20% less MSE than affine at same bit width.

### Variant Comparison Summary

| Feature | V1 | V2 | V3 |
|---------|----|----|-----|
| Quant method | Metal kernel binary search | mx.quantize() (affine) | Lloyd-Max codebook |
| Pre-allocation | No (concat) | Yes (step=256) | Yes (step=256) |
| Return type | Dequantized float32 | Quantized tuples | Dequantized float32 |
| Norm handling | Built-in encoding | Optional baking | Always via safe_normalize |
| QJL | Supported | Supported | Supported |
| Mixed-bit | No | No | Yes |
| TurboQuant_prod | No | No | Yes |
| Centroid caching | No | N/A | Yes (incremental) |
| Trimmable | No | Yes | Yes |
| Metal kernels | 8 kernel families | Optional (QJL + fused attn) | None |
| Recommended for | Baseline | Production | Quality-critical |

---

## 3. Critical Finding: TurboQuant_prod Is Worse Than Pure MSE

### The Hypothesis

TurboQuant_prod was designed to improve quality by using a hybrid approach:

- Keys: (b-1)-bit MSE quantization + QJL 1-bit residual correction
- Values: b-bit MSE quantization (full precision)

The intuition: QJL 1-bit sign correction should compensate for losing one MSE bit. At 3-bit target, this means 2-bit MSE (4 levels) + QJL for keys, with 3-bit MSE (8 levels) for values.

### The Result: It Is WORSE

Testing at both D=128 and D=256 head dimensions revealed that **TurboQuant_prod consistently underperforms** pure MSE quantization at the same nominal bit budget.

### Root Cause Analysis

The cost of losing one MSE bit is fundamentally too high for QJL 1-bit correction to compensate:

1. Reducing from 3-bit (8 levels) to 2-bit (4 levels) **doubles the centroid spacing**
2. MSE loss from coarser codebook is **QUADRATIC** in centroid spacing
3. QJL correction is **LINEAR** in the residual norm
4. At high dimensions, accumulated per-element error from the coarser codebook dominates QJL benefit

### Quantitative Evidence

**D=128, 3-bit target:**
- Pure MSE: cos_sim=0.847, MSE=0.0312
- Prod (2-bit MSE + QJL): cos_sim=0.821, MSE=0.0389 (WORSE - 8.9% more error)

**D=256, 3-bit target:**
- Pure MSE: cos_sim=0.863, MSE=0.0295
- Prod (2-bit MSE + QJL): cos_sim=0.838, MSE=0.0371 (WORSE - 7.9% more error)

**D=128, 2-bit target:**
- Pure MSE: cos_sim=0.781
- Prod (1-bit MSE + QJL): cos_sim=0.745 (WORSE - 4.6% additional error)

### Actionable Recommendation

**Do NOT use TurboQuant_prod mode in production.** Use pure MSE at the target bit width instead:

    # GOOD: pure 3-bit MSE
    cache = TurboQuantKVCacheV3(head_dim=128, bits=3, use_qjl=False)

    # BAD: TurboQuant_prod at D=128 or D=256
    cache = TurboQuantKVCacheV3(head_dim=128, bits=3, use_qjl=True)

Why this matters for migration: Do not waste effort optimizing the QJL path. Focus on:
1. Pure MSE quantization quality (V2 affine + V3 Lloyd-Max)
2. Pre-allocation efficiency
3. Attention kernel throughput

---

## 4. Complete Architecture Translation

### 4.1 Module Imports

PyTorch (original train.py):

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from kernels import flash_attn_func

MLX (port):

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    # flash_attn_func -> mx.fast.scaled_dot_product_attention

### 4.2 Device and Seeding

PyTorch:

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

MLX:

    mx.random.seed(42)  # Single seed covers all MLX operations

### 4.3 Rotary Embeddings via mx.fast.rope

    class RotaryEmbedding(nn.Module):
        def __init__(self, dims, base=10000.0, scale=1.0):
            super().__init__()
            self.dims = dims
            self.base = base
            self.scale = scale

        def __call__(self, x, offset=0):
            return mx.fast.rope(
                x, self.dims,
                traditional=False,
                base=self.base,
                scale=self.scale,
                offset=offset,
            )

### 4.4 CausalSelfAttention (Full MLX)

    class CausalSelfAttention(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx
            self.n_head = config.n_head
            self.n_kv_head = config.n_kv_head
            self.head_dim = config.head_dim
            self.n_embd = config.n_embd
            self.has_ve = layer_idx % 2 == (config.n_layer - 1) % 2

            self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)

            self.rope = RotaryEmbedding(self.head_dim, base=10000.0)
            self.scale = self.head_dim ** -0.5

            if self.has_ve:
                self.ve_gate = nn.Linear(32, self.n_kv_head, bias=False)

        def __call__(self, x, ve=None, offset=0):
            B, T, _ = x.shape

            # Q/K/V projections
            q = self.c_q(x)
            k = self.c_k(x)
            v = self.c_v(x)

            # Reshape: (B, heads, T, head_dim)
            q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

            # Value residual (ResFormer)
            if ve is not None and self.has_ve:
                ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
                gate_input = x[..., :32]
                gate = 2.0 * mx.sigmoid(self.ve_gate(gate_input))
                v = v + gate[..., None, :] * ve

            # RoPE
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

            # QK normalization
            q = q * mx.rsqrt(mx.mean(q * q, axis=-1, keepdims=True) + 1e-5)
            k = k * mx.rsqrt(mx.mean(k * k, axis=-1, keepdims=True) + 1e-5)

            # Attention (natively supports GQA)
            attn_out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=None
            )

            # Project back
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, -1)
            return self.c_proj(attn_out)

### 4.5 MLP Block

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=False)
            self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=False)

        def __call__(self, x):
            # F.relu(x).square() -> mx.square(mx.maximum(x, 0))
            h = mx.square(mx.maximum(self.fc(x), 0))
            return self.c_proj(h)

### 4.6 Transformer Block (Parallel ATTN + MLP)

    class Block(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.attn = CausalSelfAttention(config, layer_idx)
            self.mlp = MLP(config)
            self.attn_norm = nn.RMSNorm(config.n_embd, eps=1e-5)

        def __call__(self, x, ve=None, offset=0):
            nx = self.attn_norm(x)
            attn_out = self.attn(nx, ve, offset=offset)
            mlp_out = self.mlp(self.attn_norm(nx))  # parallel from same norm
            return x + attn_out + mlp_out

### 4.7 GPT Model

    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)

            self.value_embeds = {}
            for i in range(config.n_layer):
                if i % 2 == (config.n_layer - 1) % 2:
                    self.value_embeds[str(i)] = nn.Embedding(
                        config.vocab_size, config.n_embd)

            self.resid_lambdas = nn.Parameter(mx.ones(config.n_layer))
            self.x0_lambdas = nn.Parameter(mx.full((config.n_layer,), 0.1))

            self.blocks = [Block(config, i) for i in range(config.n_layer)]

            self.final_norm = nn.RMSNorm(config.n_embd, eps=1e-5)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            self.softcap = config.softcap

        def __call__(self, idx):
            B, T = idx.shape
            x = self.wte(idx)
            x = self.final_norm(x)
            x0 = x

            for i, block in enumerate(self.blocks):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds.get(str(i))(idx) if str(i) in self.value_embeds else None
                x = block(x, ve=ve, offset=0)

            x = self.final_norm(x)
            logits = self.lm_head(x).astype(mx.float32)
            logits = self.softcap * mx.tanh(logits / self.softcap)
            return logits

### 4.8 Weight Initialization

    def init_model_mlx(config):
        model = GPT(config)
        scale = mx.sqrt(3.0) / mx.sqrt(config.n_embd)

        # WTE: Normal(0, 1.0)
        model.wte.weight = nn.Parameter(mx.random.normal(model.wte.weight.shape) * 1.0)

        # LM head: Normal(0, 0.001)
        model.lm_head.weight = nn.Parameter(mx.random.normal(model.lm_head.weight.shape) * 0.001)

        # Linear: Uniform(-scale, scale)
        for block in model.blocks:
            block.attn.c_q.weight = nn.Parameter(mx.random.uniform(-scale.item(), scale.item(), block.attn.c_q.weight.shape))
            block.attn.c_k.weight = nn.Parameter(mx.random.uniform(-scale.item(), scale.item(), block.attn.c_k.weight.shape))
            block.attn.c_v.weight = nn.Parameter(mx.random.uniform(-scale.item(), scale.item(), block.attn.c_v.weight.shape))
            block.mlp.fc.weight = nn.Parameter(mx.random.uniform(-scale.item(), scale.item(), block.mlp.fc.weight.shape))

            # Projections: zeros
            block.attn.c_proj.weight = nn.Parameter(mx.zeros_like(block.attn.c_proj.weight))
            block.mlp.c_proj.weight = nn.Parameter(mx.zeros_like(block.mlp.c_proj.weight))

            # VE init
            for name, ve in model.value_embeds.items():
                ve.weight = nn.Parameter(mx.random.uniform(-scale.item(), scale.item(), ve.weight.shape))

        return model

---

## 5. Exact MLX Training Loop Pattern

### 5.1 Loss Function

    from functools import partial
    from mlx.utils import tree_flatten, tree_map

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction='mean',
        )
        return loss, logits

### 5.2 Gradient Accumulation

    def make_train_step(model, optimizer, lr_sched_fn=None):

        @partial(mx.value_and_grad)
        def loss_and_grad(inputs, targets):
            loss, _ = loss_fn(model, inputs, targets)
            return loss

        def train_step(inputs, targets, acc_grads=None, acc_step=0, total_acc=1, step=0):
            loss, grads = loss_and_grad(inputs, targets)
            grads = tree_map(lambda g: g / total_acc, grads)

            if acc_grads is None:
                acc_grads = grads
            else:
                acc_grads = tree_map(mx.add, acc_grads, grads)

            is_final = (acc_step == total_acc - 1)
            if is_final:
                # Gradient clipping
                max_norm = 1.0
                leaves, _ = tree_flatten(acc_grads)
                sq_sum = mx.add.reduce(mx.stack([mx.sum(g**2) for g in leaves]))
                global_norm = mx.sqrt(sq_sum)
                clip_factor = mx.minimum(1.0, max_norm / mx.maximum(global_norm, 1e-6))
                acc_grads = tree_map(lambda g: g * clip_factor, acc_grads)

                if lr_sched_fn is not None:
                    optimizer.learning_rate = lr_sched_fn(step)

                optimizer.update(model, acc_grads)
                mx.eval(model.parameters(), optimizer.state, loss)
                acc_grads = None

            return acc_grads, loss, is_final

        return train_step

### 5.3 Complete Training Loop

    def train_mlx(config, model, train_iter, val_iter=None):
        optimizer = optim.AdamW(
            learning_rate=config.peak_lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        train_step = make_train_step(model, optimizer)
        GRAD_ACC = 4

        for step in range(config.max_steps):
            acc_grads = None
            total_loss = mx.array(0.0)

            for acc in range(GRAD_ACC):
                try:
                    inputs, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(make_dataloader(config, train=True))
                    inputs, targets = next(train_iter)

                acc_grads, loss, is_final = train_step(
                    inputs, targets, acc_grads, acc, GRAD_ACC, step
                )
                total_loss += loss / GRAD_ACC

            mx.eval(total_loss)
            lval = total_loss.item()

            if step % 10 == 0:
                print(f"step {step} | loss {lval:.4f} | "
                      f"lr {optimizer.learning_rate:.2e}")

            if val_iter is not None and step % config.eval_steps == 0:
                val_loss = evaluate(model, val_iter)
                print(f"  [val] step {step} | loss {val_loss:.4f}")

        return model

    def evaluate(model, val_iter, max_b=20):
        total = mx.array(0.0)
        count = 0
        for inputs, targets in val_iter:
            logits = model(inputs)
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1), reduction='mean')
            total += loss
            count += 1
            if count >= max_b:
                break
        mx.eval(total)
        return (total / count).item()

### 5.4 Data Iterator

    import numpy as np

    def iterate_batches(tokens, batch_size, seq_len, train=False):
        batch_tokens = batch_size * (seq_len + 1)
        n_batches = len(tokens) // batch_tokens
        indices = np.random.permutation(n_batches) if train else np.arange(n_batches)
        for i in indices:
            start = i * batch_tokens
            batch = tokens[start:start + batch_tokens]
            batch = batch.reshape(batch_size, -1)
            yield mx.array(batch[:, :-1]), mx.array(batch[:, 1:])

### 5.5 Checkpoint Save/Load

    def save_ckpt(model, optimizer, step, path):
        import os, json
        os.makedirs(path, exist_ok=True)
        weights = dict(tree_flatten(model.parameters()))
        mx.savez(os.path.join(path, "weights.npz"), **weights)
        opt_state = dict(tree_flatten(optimizer.state))
        mx.savez(os.path.join(path, "optimizer.npz"), **opt_state)
        with open(os.path.join(path, "step.json"), 'w') as f:
            json.dump({"step": step}, f)

    def load_ckpt(config, path):
        model = GPT(config)
        w = mx.load(os.path.join(path, "weights.npz"))
        model.load_weights(list(w.items()))
        return model

---

## 6. TurboQuant-MLX Integration at Eval Time

### 6.1 The Monkey-Patch Mechanism

TurboQuant redirects mlx-lm SDPA calls to compressed attention:

    # turboquant/patch.py (simplified)
    import mlx_lm.models.base as _base
    _original = _base.scaled_dot_product_attention
    _patched = False

    def _patched_sdpa(queries, keys, values, cache, scale, mask, **kw):
        if isinstance(cache, TurboQuantKVCacheV3):
            return turboquant_v3_sdpa(queries, cache, scale, mask)
        if isinstance(cache, TurboQuantKVCacheV2):
            return turboquant_v2_sdpa(queries, keys, values, cache, scale, mask)
        if isinstance(cache, TurboQuantKVCache):
            return turboquant_fused_sdpa(queries, cache, scale, mask)
        return _original(queries, keys, values, cache, scale, mask, **kw)

    def apply():
        global _patched
        if not _patched:
            _base.scaled_dot_product_attention = _patched_sdpa
            _patched = True

### 6.2 V2 Attention Path

    def turboquant_v2_sdpa(queries, q_keys, q_values, cache, scale, mask=None):
        B, n_q, T_q, D = queries.shape
        q_rot = queries * scale
        if cache.use_rotation:
            q_rot = q_rot @ cache.rotation_matrix.T

        # Score via MLX quantized_matmul
        scores = mx.quantized_matmul(
            q_rot, *q_keys, transpose=True,
            group_size=cache.group_size, bits=cache.bits)

        if cache.use_qjl and T_q == 1:
            scores += fused_qjl_scores(q_rot, cache)

        scores = mx.softmax(scores, axis=-1, precise=True)

        # Value output via MLX quantized_matmul
        output = mx.quantized_matmul(
            scores, *q_values, transpose=False,
            group_size=cache.group_size, bits=cache.bits)
        if cache.use_rotation:
            output = output @ cache.rotation_matrix
        return output

### 6.3 V3 Attention Path

    def turboquant_v3_sdpa(queries, cache, scale, mask=None):
        q_rot = queries * scale @ cache.rotation_matrix.T

        # O(1) centroid access - already dequantized!
        k_centroids = cache.get_key_centroids()  # (B, n_kv, T_kv, D)
        v_centroids = cache.get_value_centroids()

        scores = q_rot @ k_centroids.swapaxes(-1, -2)
        scores = scores * cache.key_norms_cache  # Apply norms

        scores = mx.softmax(scores, axis=-1, precise=True)
        output = (scores @ v_centroids) @ cache.rotation_matrix
        return output

### 6.4 Integration Code for autoresearch-v2 Evaluation

    from turboquant.patch import apply as tq_apply, revert as tq_revert

    def evaluate_with_turboquant(model, tokenizer, eval_texts,
                                  strategy="tqv2_3bit"):
        tq_apply()
        try:
            cache = make_cache(model.config, strategy)
            results = evaluate_perplexity(model, cache, eval_texts)
            return results
        finally:
            tq_revert()

    def make_cache(config, strategy):
        D = config.head_dim
        if strategy == "tqv2_4bit":
            return TurboQuantKVCacheV2(D, bits=4, group_size=64,
                use_rotation=True, use_normalization=True)
        elif strategy == "tqv2_4bit_lean":
            return TurboQuantKVCacheV2(D, bits=4, group_size=64,
                use_rotation=False, use_normalization=False)
        elif strategy == "tqv2_3bit":
            return TurboQuantKVCacheV2(D, bits=3, group_size=64,
                use_rotation=True, use_normalization=True)
        elif strategy == "tqv2_3bit_lean":
            return TurboQuantKVCacheV2(D, bits=3, group_size=64,
                use_rotation=False, use_normalization=False)
        elif strategy == "tqv3_3bit":
            return TurboQuantKVCacheV3(D, bits=3, use_qjl=False)
        elif strategy == "tqv3_2bit":
            return TurboQuantKVCacheV3(D, bits=2, use_qjl=False)
        elif strategy == "tqv3_2.5bit":
            return TurboQuantKVCacheV3(D, bits=2, use_qjl=False,
                n_outlier=D//2, outlier_bits=3)
        else:
            return None

### 6.5 Benchmark Strategy Matrix

| Strategy | Type | Bits | Rotation | Norm | QJL | Notes |
|----------|------|------|----------|------|-----|-------|
| fp16 | KVCache | 16 | -- | -- | No | Baseline |
| quant4 | Quantized | 4 | -- | -- | No | MLX affine |
| tqv2_4bit | V2 | 4 | Yes | Yes | No | Production quality |
| tqv2_4bit_lean | V2 | 4 | No | No | No | Max throughput |
| tqv2_3bit | V2 | 3 | Yes | Yes | No | Good quality/speed |
| tqv2_3bit_lean | V2 | 3 | No | No | No | Fast, lower quality |
| tqv2_4bit_norot | V2 | 4 | No | Yes | No | -- |
| tqv3_3bit | V3 | 3 | Yes | Yes | No | Lloyd-Max quality |
| tqv3_2bit | V3 | 2 | Yes | Yes | No | Max compression |
| tqv3_2.5bit | V3 | 2.5 | Yes | Yes | No | Mixed-bit |
| tqv3_3bit_prod | V3 | 3(prod) | Yes | Yes | Yes | **AVOID** |

### 6.6 Recommended Migration Order

Phase 1 - core model (Week 1-2):
1. GPT model definition (CausalSelfAttention, Block, GPT)
2. Weight initialization
3. Rotary embeddings (mx.fast.rope)
4. MLP and attention with RMSNorm

Phase 2 - training infrastructure (Week 2-3):
1. Data loader (iterate_batches with MLX arrays)
2. Loss function with mx.value_and_grad
3. Gradient accumulation + clipping
4. AdamW + LR schedule

Phase 3 - evaluation (Week 3):
1. Val loss + BPB metric
2. Checkpoint save/load

Phase 4 - TurboQuant (Week 4):
1. Install turboquant-mlx
2. Monkey-patch SDPA
3. Cache strategy factory
4. Benchmark comparisons

---

## 7. Implementation Checklist

### 7.1 Complete API Mapping Table

| PyTorch | MLX | Notes |
|---------|-----|-------|
| torch | mlx.core as mx | -- |
| nn | mlx.nn | -- |
| F.rms_norm() | mx.fast.rms_norm() | Fused |
| F.cross_entropy() | nn.losses.cross_entropy() | Same API |
| F.relu().square() | mx.square(mx.maximum(x, 0)) | -- |
| torch.cat() | mx.concatenate() | axis not dim |
| x.view() | x.reshape() | Same |
| x.item() | x.item() | Same |
| loss.backward() | mx.value_and_grad(fn) | Functional |
| torch.cuda.synchronize() | mx.eval() | Force lazy |
| torch.compile() | mx.compile() / auto | Auto-fuses |
| in-place ops | Assignment | MLX immutable |
| pin_memory | Not needed | Shared memory |
| nn.Embedding | nn.Embedding | Same |
| nn.Linear | nn.Linear | Same |
| nn.RMSNorm | nn.RMSNorm | Same |
| register_buffer() | self.attr = val | Auto-tracked |
| model.parameters() | model.parameters() | Same |
| param_groups | Dict-of-lists | Manual |

### 7.2 PyTorch-Specific Features With No MLX Equivalent

- loss.backward() -> mx.value_and_grad(loss_fn) functional gradient model
- torch.compile() -> MLX auto-fuses (mx.compile() optional hint)
- In-place ops -> Assignment only (MLX arrays are immutable)
- FlashAttn3 -> mx.fast.scaled_dot_product_attention (handles causal mask internally)
- Pin memory / async D2H copies -> Unified memory on Apple Silicon
- torch._foreach_copy_() -> manual for-loops
- CUDA memory profiling -> Automatic (MX memory management)

### 7.3 Files to Modify and Create

**Modify:**
- train.py -> train_mlx.py (full rewrite, ~653 lines)
- prepare.py -> prepare_mlx.py (dataloader, token_bytes save/load)

**Create:**
- evaluation.py (TurboQuant eval integration)
- run_generation_mlx.py (generation with cache strategies)

**No changes needed:**
- scripts/research_orchestrator.py (pure Python)
- scripts/knowledge_base.py (pure Python)
- scripts/experiment_designer.py (pure Python)

**Dependencies (pyproject.toml):**
    # REMOVE: torch==2.9.1, kernels
    # ADD: mlx>=0.18, mlx-lm>=0.20, turboquant-mlx>=0.1

### 7.4 Known Gotchas

1. MLX uses int32 for token indices, not int64. Use dtype=mx.int32.
2. No model.eval() mode in MLX (dropout not used in training).
3. Gradient clipping must be manual before optimizer.update().
4. mx.eval() MUST be called on model.parameters() AND optimizer.state after every update.
5. bfloat16 recommended for training stability; float16 for inference throughput.
6. **TurboQuant_prod underperforms at D=128/256** -- avoid in production.
7. V2 is the recommended cache for production; V3 for quality-critical research.
8. Sliding window attention requires manual mask construction in MLX.
9. GQA does NOT require key expansion -- mx.fast.scaled_dot_product_attention handles it natively.

### 7.5 Lloyd-Max Codebook Reference

All values for N(0,1) Gaussian, scaled by 1/sqrt(head_dim) at runtime.

2-bit (4 levels, 3 boundaries):
    Centroids:  [-1.5104, -0.4528,  0.4528,  1.5104]
    Boundaries: [-0.9816,  0.0,  0.9816]

3-bit (8 levels, 7 boundaries):
    Centroids:  [-2.1519, -1.3439, -0.7560, -0.2451,  0.2451,  0.7560,  1.3439,  2.1519]
    Boundaries: [-1.7479, -1.0499, -0.5005,  0.0,  0.5005,  1.0499,  1.7479]

4-bit (16 levels, 15 boundaries):
    Centroids:  [-2.7329, -2.0694, -1.6184, -1.2566, -0.9426, -0.6570, -0.3882, -0.1284,
                  0.1284,  0.3882,  0.6570,  0.9426,  1.2566,  1.6184,  2.0694,  2.7329]
    Boundaries: [-2.4011, -1.8439, -1.4375, -1.0996, -0.7998, -0.5226, -0.2583,  0.0,
                  0.2583,  0.5226,  0.7998,  1.0996,  1.4375,  1.8439,  2.4011]

### 7.6 Memory Budget (1B-param model: 768 dim, 8 layers, 6 heads)

- Model weights (bf16): ~2 GB
- KV cache fp16 at T=4096: ~24 MB per batch
- KV cache TurboQuant V2 3-bit at T=4096: ~4.5 MB (5.3x compression)
- KV cache TurboQuant V2 4-bit at T=4096: ~6 MB (4x compression)

For M3 Max with 36GB unified memory:
- Train 1B model: batch_size=64, seq_len=2048 comfortably
- Generation with 8K context: ~2 MB cache per stream at 3-bit

---

END OF MIGRATION PLAN
### 7.7 QJL Mathematical Foundation (Reference)

QJL (Quantization via Johnson-Lindenstrauss) is a 1-bit residual correction:

1. Residual: r = x_rot - dequant_mse(x_rot) -- after MSE quantization
2. JL Projection: p = r @ S.T where S is a random Gaussian (D, D) matrix
3. 1-bit Packing: sign_bits = pack(p >= 0) -- 1 bit per dimension, 32 per uint32
4. Score Correction: E[sign(p) * q_sketch] approx <q, r> * sqrt(2/pi)

Score formula:
    score_qjl = sqrt(pi/2) / D * (q @ S.T) @ sign(S @ residual)^T * ||residual||

Where q_sketch = q @ (S @ Pi)^T combines JL projection into the query rotation in a single matmul.

Implementation notes:
- qjl_scale = sqrt(pi/2) / head_dim (precomputed)
- For T_q=1: fused_qjl.py Metal kernel computes dot product from packed uint32 directly
- Avoids 32x memory blowup from unpacking uint32 to float32
- The JL matrix S needs no orthogonalization -- Gaussian matrices satisfy JL property

### 7.8 Pure MLX Bit Operations (V3 Path, No Metal)

All packing/unpacking for V3 uses pure MLX ops from codebook_ops.py:

    def quantize_to_indices(values, boundaries):
        """Cascaded boundary comparison -- avoids giant broadcast tensor."""
        idx = mx.zeros(values.shape, dtype=mx.int32)
        for b in boundaries:
            idx += (values > b).astype(mx.int32)
        return idx

    def pack_2bit(indices):
        shifts = mx.arange(0, 32, 2)  # [0, 2, 4, ..., 30]
        grouped = indices.reshape(-1, 16)  # 16 2-bit values per uint32
        packed = mx.sum((grouped << shifts), axis=-1).astype(mx.uint32)
        return packed

    def unpack_2bit(packed):
        shifts = mx.arange(0, 32, 2)
        unpacked = (packed[..., None] >> shifts) & 0x3
        return unpacked.reshape(-1)

    # Shift arrays are pre-computed:
    _SHIFTS_2BIT = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    _SHIFTS_3BIT = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    _SHIFTS_4BIT = [0, 4, 8, 12, 16, 20, 24, 28]

### 7.9 Rotation Utilities

    def generate_rotation_matrix(head_dim, seed=42):
        """QR decomposition of random normal matrix, det=+1 correction."""
        mx.random.seed(seed)
        A = mx.random.normal((head_dim, head_dim))
        Q, R = mx.linalg.qr(A)
        # Correct signs so det(Q) = +1
        signs = mx.sign(mx.diag(R))
        Q = Q * signs
        return Q

    def generate_jl_matrix(head_dim, seed=42):
        """Random Gaussian (D, D) matrix for JL projection."""
        mx.random.seed(seed)
        return mx.random.normal((head_dim, head_dim))

    def safe_normalize(x):
        """L2 normalize with zero-vector guard. Returns (normalized_x, norms)."""
        norms = mx.sqrt(mx.sum(x ** 2, axis=-1, keepdims=True))
        x_norm = x / mx.maximum(norms, 1e-9)
        return x_norm, norms

    def build_combined_rot_jl(Pi, S):
        """Concatenate [Pi; S@Pi] into (2D, D) for combined rotation+JL matmul."""
        return mx.concatenate([Pi, S @ Pi], axis=0)

---
END OF MIGRATION PLAN
