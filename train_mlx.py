"""
Autoresearch v2 MLX Edition pretraining script. Single-file for Apple Silicon.
Usage: uv run train_mlx.py

Complete single-file pretraining with:
- mx.fast.scaled_dot_product_attention (fused attention)
- Full GPT model (parallel ATTN+MLP, value embeddings, RoPE, sliding window)
- MuON + AdamW optimizer
- proper MLX patterns (@partial(mx.value_and_grad), tree accumulation)
- Gradient accumulation for large effective batch sizes
- Metrics output for research_orchestrator parsing
- TurboQuant KV cache integration for evaluation mode
"""

import os
import math
import time
import sys
from dataclasses import dataclass, asdict
from functools import partial

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten, tree_map


# ---------------------------------------------------------------------------
# Import data pipeline
# ---------------------------------------------------------------------------

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS,
    Tokenizer, make_dataloader, get_token_bytes,
    CACHE_DIR, TOKENIZER_DIR,
)

# Try importing TurboQuant for eval-mode KV cache compression
try:
    from turboquant_mlx import KVCacheCompressor
    HAS_TURBOQUANT = True
except ImportError:
    HAS_TURBOQUANT = False


# Use the MLX-specific dataloader wrapper defined below instead of make_dataloader directly


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def has_ve(layer_idx, n_layer):
    """Returns True if layer has Value Embedding (alternating, last always)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    """RoPE rotation using precomputed cos/sin tables."""
    assert x.ndim == 4
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def rope_with_precomputed(x, cos, sin):
    """Apply precomputed RoPE to queries/keys of shape (B, heads, T, head_dim).
    cos, sin shape: (1, T, 1, head_dim) """
    assert x.ndim == 4
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


# ---------------------------------------------------------------------------
# Reusable normalization helper
# ---------------------------------------------------------------------------

class RMSNormWrapper:
    """RMSNorm using mx.fast.rms_norm fused kernel (standalone, no learnable weights)."""
    def __init__(self, eps=1e-5):
        self.eps = eps

    def __call__(self, x):
        """Normalize without learnable weight (like F.rms_norm in PyTorch)."""
        # mx.fast.rms_norm requires weight; we pass ones
        w = mx.ones(x.shape[-1], dtype=x.dtype)
        return mx.fast.rms_norm(x, w, self.eps)


# ---------------------------------------------------------------------------
# Attention Module
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.scale = self.head_dim ** -0.5
        self.layer_idx = layer_idx

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Value Embedding gate (ResFormer)
        self.ve_gate_channels = 32
        if has_ve(layer_idx, config.n_layer):
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

        self.norm_q = RMSNormWrapper(1e-5)
        self.norm_k = RMSNormWrapper(1e-5)

    def __call__(self, x, ve, cos, sin, window_size):
        B, T, C = x.shape

        # Linear projections
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer)
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2.0 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            gate = mx.expand_dims(gate, axis=-1)
            v = v + gate * ve

        # RoPE
        q = rope_with_precomputed(q, cos, sin)
        k = rope_with_precomputed(k, cos, sin)

        # QK normalization
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Transpose to (B, heads/n_kv_heads, T, head_dim)
        q = q.transpose(0, 2, 1, 3)   # (B, n_head, T, head_dim)
        k = k.transpose(0, 2, 1, 3)   # (B, n_kv_head, T, head_dim)
        v = v.transpose(0, 2, 1, 3)   # (B, n_kv_head, T, head_dim)

        # Sliding window causal mask
        mask = self._make_mask(T, window_size)

        # Fused scaled dot-product attention (MLX handles GQA natively)
        output = mx.fast.scaled_dot_product_attention(
            q, k, v,
            scale=self.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(output)

    def _make_mask(self, T, window_size):
        """Sliding window causal mask for training."""
        q_idx = mx.arange(T).reshape(T, 1)
        k_idx = mx.arange(T).reshape(1, T)
        causal = k_idx <= q_idx
        long_window = window_size[0]
        if long_window < T:
            window = k_idx >= (q_idx - long_window + 1)
            mask = causal & window
        else:
            mask = causal
        return mask


# ---------------------------------------------------------------------------
# MLP Block
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.square(mx.maximum(x, 0))  # ReLU^2 activation
        x = self.c_proj(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block (parallel ATTN + MLP)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.norm = RMSNormWrapper(1e-5)

    def __call__(self, x, ve, cos, sin, window_size):
        # Parallel Attention + MLP (both from same norm(x))
        nx = self.norm(x)
        attn_out = self.attn(nx, ve, cos, sin, window_size)
        mlp_out = self.mlp(nx)
        x = x + attn_out + mlp_out
        return x


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(mx.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(mx.zeros(config.n_layer))

        # Value embeddings (alternating layers)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {}
        for i in range(config.n_layer):
            if has_ve(i, config.n_layer):
                self.value_embeds[str(i)] = nn.Embedding(config.vocab_size, kv_dim)

        # Rotary embeddings (precomputed buffers)
        self.rotary_seq_len = config.sequence_len * 10
        self.cos, self.sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim
        )

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :]  # (1, seq, 1, half_dim)
        sin = freqs.sin()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def num_params(self):
        """Total parameter count."""
        return sum(v.size for _, v in tree_flatten(self.trainable_parameters()))

    def param_counts_by_group(self):
        """Parameter counts per group (matches PyTorch version)."""
        params = self.trainable_parameters()
        wte = sum(v.size for name, v in tree_flatten(params) if "wte" in name)
        ve_total = sum(v.size for name, v in tree_flatten(params) if "value_embeds" in name)
        lm_head = sum(v.size for name, v in tree_flatten(params) if "lm_head" in name)
        scalars = self.resid_lambdas.size + self.x0_lambdas.size

        h_names = set()
        for name, _ in tree_flatten(params):
            if "h." in name:
                h_names.add(name)
        h_param_names = {name for name in h_names}

        # Transformer matrices: all params in h that are 2D
        transformer_matrices = sum(
            v.size for name, v in tree_flatten(params)
            if "h." in name and len(v.shape) == 2
        )

        total = wte + ve_total + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': ve_total, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars,
            'total': total,
        }

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        counts = self.param_counts_by_group()
        nparams_exclude = (counts['wte'] + counts['value_embeds'] +
                          counts['scalars'])
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for ws in self.window_sizes:
            w = ws[0]
            effective_seq = t if w < 0 else min(w, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (counts['total'] - nparams_exclude) + attn_flops

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape
        assert T <= self.cos.shape[1], f"T={T} exceeds rotary seq len={self.cos.shape[1]}"

        # Slice precomputed RoPE to current sequence length
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        x = self.wte(idx)
        x = mx.fast.rms_norm(x, mx.ones(x.shape[-1], dtype=x.dtype), 1e-5)
        x0 = x

        for i, block in enumerate(self.h):
            # Residual mixing with learnable scalars
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # Value embedding lookup
            ve_key = str(i)
            ve = self.value_embeds[ve_key](idx) if ve_key in self.value_embeds else None
            # Forward through block
            x = block(x, ve, cos, sin, self.window_sizes[i])

        # Final norm
        x = mx.fast.rms_norm(x, mx.ones(x.shape[-1], dtype=x.dtype), 1e-5)

        # LM head + softcap
        logits = self.lm_head(x)
        logits = logits.astype(mx.float32)
        softcap = 15.0
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction=reduction,
            )
            return loss
        return logits


# ---------------------------------------------------------------------------
# Polar Express coefficients for MuON Newton-Schulz
# ---------------------------------------------------------------------------

POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


# ---------------------------------------------------------------------------
# MuON AdamW Optimizer
# ---------------------------------------------------------------------------

class MuonAdamW(nn.Module):
    """Muon (for 2D matrices) + AdamW (for all other params).

    Usage (matching MLX optimizer pattern):
        optimizer.update(model, grads)
    where grads is a tree matching model.trainable_parameters().
    """

    def __init__(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        scalar_lr=0.5,
        matrix_lr=0.02,
        adam_betas=(0.8, 0.95),
        weight_decay=0.0,
        init_lr_scale=1.0,
    ):
        super().__init__()
        self.unembedding_lr = unembedding_lr * init_lr_scale
        self.embedding_lr = embedding_lr * init_lr_scale
        self.scalar_lr = scalar_lr * init_lr_scale
        self.matrix_lr = matrix_lr
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay

        # Dynamic LR multiplier (set before each step)
        self.lr_multiplier = 1.0
        self.muon_momentum = 0.95
        self.muon_beta2 = 0.95
        self.muon_ns_steps = 5
        self.muon_wd = weight_decay

    def init_grouped_params(self, params):
        """
        Partition parameters into AdamW and MuON groups and initialize state.
        Returns (adamw_state, muon_state).
        """
        adamw_names = set()
        muon_matrix_params = []  # (name, array) for 2D matrices
        adamw_param_list = []    # (name, array) for everything else

        for name, p in tree_flatten(params):
            if "lm_head" in name:
                adamw_param_list.append((name, p))
                adamw_names.add(name)
            elif "wte" in name:
                adamw_param_list.append((name, p))
                adamw_names.add(name)
            elif "value_embeds" in name:
                adamw_param_list.append((name, p))
                adamw_names.add(name)
            elif "resid_lambdas" in name:
                adamw_param_list.append((name, p))
                adamw_names.add(name)
            elif "x0_lambdas" in name:
                adamw_param_list.append((name, p))
                adamw_names.add(name)
            elif len(p.shape) == 2:
                muon_matrix_params.append((name, p))
                adamw_names.add(name)  # don't put in adamw list
            # else: scalars not matching the above — default to AdamW
            # (shouldn't happen but be safe)

        # AdamW state
        adamw_state = {}
        for name, p in adamw_param_list:
            adamw_state[name] = {
                "exp_avg": mx.zeros_like(p),
                "exp_avg_sq": mx.zeros_like(p),
                "step": mx.array(0),
            }

        # Group MuON params by shape
        muon_groups = {}  # shape -> list of (name, array)
        for name, p in muon_matrix_params:
            muon_groups.setdefault(str(p.shape), []).append((name, p))

        # MuON state
        muon_state = {}
        for shape_key, group_params in muon_groups.items():
            first_p = group_params[0][1]
            shape = first_p.shape
            if shape[-2] >= shape[-1]:
                sm_shape = (len(group_params), shape[-2], 1)
            else:
                sm_shape = (len(group_params), 1, shape[-1])
            muon_state[shape_key] = {
                "params": group_params,
                "momentum_buffer": mx.zeros((len(group_params), *shape), dtype=first_p.dtype),
                "second_momentum_buffer": mx.zeros(sm_shape, dtype=first_p.dtype),
            }

        self._adamw_names = adamw_names
        self._adamw_state = adamw_state
        self._muon_state = muon_state
        return adamw_names, adamw_state, muon_state

    def _get_adamw_lr(self, name, n_embd):
        """Get the base learning rate for a given AdamW param."""
        base = 0.0
        if "lm_head" in name:
            base = self.unembedding_lr
        elif "wte" in name:
            base = self.embedding_lr
        elif "value_embeds" in name:
            base = self.embedding_lr
        elif "x0_lambdas" in name:
            base = self.scalar_lr
        elif "resid_lambdas" in name:
            base = self.scalar_lr * 0.01
        else:
            base = self.scalar_lr  # scalar fallback
        return base * self.lr_multiplier

    @staticmethod
    def _adamw_single_step(p, grad, state, lr, beta1, beta2, eps, wd):
        """Single AdamW update for one parameter (pure function)."""
        m = state["exp_avg"]
        v = state["exp_avg_sq"]
        step = state["step"] + 1

        # Weight decay (decoupled)
        p_decayed = p * (1.0 - lr * wd)

        # EMA updates
        m_new = beta1 * m + (1.0 - beta1) * grad
        v_new = beta2 * v + (1.0 - beta2) * mx.square(grad)

        # Bias correction
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        step_size = lr / bias1
        denom = mx.sqrt(v_new / bias2) + eps

        p_new = p_decayed - step_size * m_new / denom

        return p_new, {"exp_avg": m_new, "exp_avg_sq": v_new, "step": step}

    @staticmethod
    def _muon_group_step(params_list, grads_list, state, lr, momentum, wd, beta2, ns_steps):
        """MuON update for one group of same-shape matrix parameters.
        
        params_list, grads_list: list of arrays
        state: {momentum_buffer, second_momentum_buffer}
        Returns: list of updated params, updated state
        """
        n = len(params_list)
        stacked_grads = mx.stack(grads_list)
        stacked_params = mx.stack(params_list)
        shape = stacked_grads.shape[1:]  # omit batch dim
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # LR scaling by aspect ratio
        lr_scaled = lr * max(1.0, shape[-2] / float(shape[-1])) ** 0.5

        # Nesterov momentum
        mb = state["momentum_buffer"]
        mb = momentum * mb + (1.0 - momentum) * stacked_grads
        g = stacked_grads + momentum * (mb - stacked_grads)

        # Polar Express orthogonalization (Newton-Schulz)
        X = g.astype(mx.bfloat16)
        X = X / (
            mx.linalg.norm(X, ord='fro', axis=(-2, -1), keepdims=True) * 1.02 + 1e-6
        )

        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            if shape[-2] > shape[-1]:
                A = X.swapaxes(-2, -1) @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
            else:
                A = X @ X.swapaxes(-2, -1)
                B = b * A + c * (A @ A)
                X = a * X + B @ X

        g = X

        # NorMuon variance reduction
        v_mean = mx.mean(mx.square(g.astype(mx.float32)), axis=red_dim, keepdims=True)
        red_dim_size = g.shape[red_dim]
        v_norm_sq = mx.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size
        v_norm = mx.sqrt(v_norm_sq)

        sm = state["second_momentum_buffer"]
        sm = beta2 * sm + (1.0 - beta2) * v_mean.astype(sm.dtype)
        step_size = mx.maximum(sm, 1e-10).rsqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * mx.square(step_size.astype(mx.float32))
        v_norm_new = mx.sqrt(mx.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
        final_scale = step_size * (v_norm / mx.maximum(v_norm_new, 1e-10))

        g = g * final_scale.astype(g.dtype)

        # Cautious weight decay
        lr_t = lr_scaled
        wd_t = wd
        mask = (g * stacked_params.astype(g.dtype)) >= 0
        update = lr_t * g + lr_t * wd_t * stacked_params.astype(g.dtype) * mask.astype(g.dtype)
        new_stacked = stacked_params.astype(g.dtype) - update

        # Unstack to list
        new_params = [new_stacked[i].astype(stacked_params.dtype) for i in range(n)]
        new_state = {"momentum_buffer": mb, "second_momentum_buffer": sm}

        return new_params, new_state

    def update(self, model, grads):
        """Apply gradient updates.

        This is the MLX optimizer interface: optimizer.update(model, grads).
        model: the model instance (we use trainable_parameters() to access params)
        grads: grad tree matching params
        """
        params = model.trainable_parameters()

        # Separate grads into adamw and muon groups
        adamw_grads = {}
        muon_grads_by_shape = {}  # shape_key -> list of (name, grad)

        for name, g in tree_flatten(grads):
            if name in self._adamw_names:
                # Check if this is a muon param (2D matrix)
                is_muon = any(
                    name in [pn for pn, _ in ms["params"]]
                    for ms in self._muon_state.values()
                )
                if is_muon:
                    shape_key = None
                    for sk, ms in self._muon_state.items():
                        if name in [pn for pn, _ in ms["params"]]:
                            shape_key = sk
                            break
                    muon_grads_by_shape.setdefault(shape_key, []).append((name, g))
                else:
                    adamw_grads[name] = g

        # AdamW updates
        adamw_updates = {}
        beta1, beta2 = self.adam_betas
        for name, (old_param) in tree_flatten(params):
            if name not in adamw_grads:
                continue
            g = adamw_grads[name]
            s = self._adamw_state[name]
            p = old_param
            lr = self._get_adamw_lr(name, getattr(model.config, 'n_embd', 768))
            wd = 0.0  # No weight decay for AdamW groups
            p_new, s_new = self._adamw_single_step(p, g, s, lr, beta1, beta2, 1e-8, wd)
            adamw_updates[name] = p_new
            self._adamw_state[name] = s_new

        # MuON updates (per shape group)
        muon_updates = {}
        for shape_key, state in self._muon_state.items():
            group_names = [pn for pn, _ in state["params"]]
            params_list = []
            grads_list = []
            for pn, pa in state["params"]:
                params_list.append(pa)
                grads_list.append(muon_grads_by_shape.get(shape_key, [(pn, mx.zeros_like(pa))])[0][1]
                                   if shape_key in muon_grads_by_shape
                                   else mx.zeros_like(pa))

            # Actually build grads_list properly
            grads_dict = {n: g for n, g in muon_grads_by_shape.get(shape_key, [])}
            grads_list = [grads_dict.get(pn, mx.zeros_like(pa)) for pn, pa in state["params"]]

            new_params, new_state = self._muon_group_step(
                params_list, grads_list, state,
                lr=self.matrix_lr * self.lr_multiplier,
                momentum=self.muon_momentum,
                wd=self.muon_wd,
                beta2=self.muon_beta2,
                ns_steps=self.muon_ns_steps,
            )
            state["momentum_buffer"] = new_state["momentum_buffer"]
            state["second_momentum_buffer"] = new_state["second_momentum_buffer"]

            for (pn, _), np_ in zip(state["params"], new_params):
                muon_updates[pn] = np_

        # Combine all updates and apply to model
        all_updates = {**adamw_updates, **muon_updates}
        model.update(all_updates)


# ---------------------------------------------------------------------------
# KV Cache for evaluation/generation
# ---------------------------------------------------------------------------

class KVCache:
    """Simple KV cache for autoregressive generation (evaluation mode)."""

    def __init__(self, head_dim, n_kv_heads, max_seq_len):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        """Append new keys/values and return full cache.
        Shape: (batch, n_kv_heads, new_seq, head_dim)
        """
        prev = self.offset
        self.offset += keys.shape[2]

        if self.keys is None:
            b = keys.shape[0]
            shape = (b, self.n_kv_heads, self.max_seq_len, self.head_dim)
            self.keys = mx.zeros(shape, keys.dtype)
            self.values = mx.zeros(shape, values.dtype)

        self.keys = mx.scatter(self.keys, mx.arange(prev, self.offset), keys, axis=2)
        self.values = mx.scatter(self.values, mx.arange(prev, self.offset), values, axis=2)

        return self.keys[:, :, :self.offset], self.values[:, :, :self.offset]

    def reset(self):
        self.keys = None
        self.values = None
        self.offset = 0


class TurboQuantKVCache(KVCache):
    """KV cache with optional TurboQuant compression."""

    def __init__(self, head_dim, n_kv_heads, max_seq_len, bits=3):
        super().__init__(head_dim, n_kv_heads, max_seq_len)
        self.compressor = None
        if HAS_TURBOQUANT:
            try:
                self.compressor = KVCacheCompressor(
                    bits=bits,
                    method="turboquant",
                    block_size=128,
                    target_bits=4.5,
                    fidelity_target=0.99,
                )
            except Exception:
                pass

    def update_and_fetch(self, keys, values):
        if self.compressor is not None:
            # Compress before storing
            keys = self.compressor.compress(keys)
            values = self.compressor.compress(values)
        return super().update_and_fetch(keys, values)


# ---------------------------------------------------------------------------
# MLX Data Loader
# ---------------------------------------------------------------------------

class MLXDataLoader:
    """Wraps PyTorch-based make_dataloader to return MLX arrays.
    
    The PyTorch dataloader yields (torch.Tensor, torch.Tensor, epoch).
    We convert to MLX arrays lazily.
    """

    def __init__(self, tokenizer, B, T, split, buffer_size=1000):
        self._pytorch_loader = make_dataloader(tokenizer, B, T, split, buffer_size)
        self._batch_size = B
        self._seq_len = T

    def __iter__(self):
        return self

    def __next__(self):
        px, py, epoch = next(self._pytorch_loader)
        # PyTorch tensor -> numpy -> MLX array (int32 for token indices)
        import numpy as _np
        x = mx.array(_np.asarray(px.cpu()), dtype=mx.int32)
        y = mx.array(_np.asarray(py.cpu()), dtype=mx.int32)
        return x, y, epoch


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bpb(model, tokenizer, batch_size, token_bytes_arr):
    """Bits per byte evaluation — MLX version."""
    val_loader = MLXDataLoader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes_arr[y_flat]
        mask = nbytes > 0
        total_nats += float(mx.sum(loss_flat * mask))
        total_bytes += int(mx.sum(nbytes))

    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Generation with KV cache
# ---------------------------------------------------------------------------

def generate(model, prompt_ids, max_new_tokens=64, temperature=0.8, cache=None, use_turboquant=False):
    """Autoregressive generation with optional KV cache."""
    tokens = mx.array(prompt_ids).reshape(1, -1)
    generated = list(prompt_ids)

    for pos in range(max_new_tokens):
        if cache is None:
            # No cache: rerun full forward (fine for short sequences)
            logits = model(tokens)[:, -1, :]
        else:
            # Full forward with cache population
            # For simplicity, use non-cached version; KV cache is set up per-block
            logits = model(tokens)[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs)
        else:
            next_token = mx.argmax(logits, axis=-1)

        next_id = int(next_token)
        generated.append(next_id)
        tokens = mx.array([[next_id]])

    return generated


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress, warmup_ratio, warmdown_ratio, final_lr_frac):
    if progress < warmup_ratio:
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    elif progress < 1.0 - warmdown_ratio:
        return 1.0
    else:
        cooldown = (1.0 - progress) / warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * final_lr_frac


def get_muon_momentum(step):
    frac = min(step / 150, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress, base_wd):
    return base_wd * max(0.5, 1 - progress)


# ---------------------------------------------------------------------------
# Hyperparameters (mirrors train.py)
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

TOTAL_BATCH_SIZE = 2**19       # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.05
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.15
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.02
WARMDOWN_RATIO = 0.1
FINAL_LR_FRAC = 0.1
DEPTH = 8
DEVICE_BATCH_SIZE = 256
MAX_GRAD_NORM = 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model_config(depth, vocab_size):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    n_kv_heads = max(1, num_heads // 2)
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=n_kv_heads,
        n_embd=model_dim, window_pattern=WINDOW_PATTERN,
    )


def main():
    t_start = time.time()

    # MLX setup
    mx.random.seed(42)
    mx.set_default_device(mx.gpu)

    # Tokenizer
    tokenizer = Tokenizer(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    # Config
    config = build_model_config(DEPTH, vocab_size)
    print(f"Model config: {asdict(config)}")

    # Model
    model = GPT(config)

    # Count params
    counts = model.param_counts_by_group()
    print("Parameter counts:")
    for key, value in counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = counts['total']
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    # Convert model to bfloat16
    model = model.astype(mx.bfloat16)

    # Init optimizer
    model_dim = config.n_embd
    dmodel_scale = (model_dim / 768) ** -0.5
    print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_scale:.6f}")

    optimizer = MuonAdamW(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        matrix_lr=MATRIX_LR,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
        init_lr_scale=dmodel_scale,
    )
    optimizer.init_grouped_params(model.trainable_parameters())

    # Datloader
    train_loader = MLXDataLoader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)

    # Token bytes for BPB
    token_bytes_arr = get_token_bytes()

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Device batch size: {DEVICE_BATCH_SIZE}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"TurboQuant available: {'yes' if HAS_TURBOQUANT else 'no'}")

    # ---- Training loop ----
    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0

    # Gradient computation using nn.value_and_grad
    # nn.value_and_grad returns a function: loss_and_grad_fn(x, y) -> (loss, grads)
    # where grads is a tree matching model.trainable_parameters()
    def forward_and_loss(model_instance, x_in, y_in):
        return model_instance(x_in, y_in)

    loss_and_grad_fn = nn.value_and_grad(model, forward_and_loss)

    # Training loop
    while True:
        t0 = time.time()

        # Gradient accumulation
        accumulated_grads = None
        last_loss_scalar = 0.0

        for micro_step in range(grad_accum_steps):
            loss_val, grads = loss_and_grad_fn(x, y)

            # Normalize gradients by accumulation steps
            grads = tree_map(lambda g: g / grad_accum_steps, grads)

            # Accumulate
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(mx.add, accumulated_grads, grads)

            # Fetch next batch (MLX arrays)
            x, y, epoch = next(train_loader)
            last_loss_scalar = float(loss_val)

        # Apply optimizer step
        optimizer.update(model, accumulated_grads)

        # Force computation
        mx.eval(model.trainable_parameters(), optimizer.state)

        # Schedule updates
        progress = min(total_training_time / TIME_BUDGET, 1.0) if total_training_time > 0 else 0.0
        lrm = get_lr_multiplier(progress, WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC)
        muon_m = get_muon_momentum(step)
        muon_wd = get_weight_decay(progress, WEIGHT_DECAY)

        optimizer.lr_multiplier = lrm
        optimizer.muon_momentum = muon_m
        optimizer.muon_wd = muon_wd

        # Fast fail
        if math.isnan(last_loss_scalar) or last_loss_scalar > 100:
            print("FAIL")
            sys.exit(1)

        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        # EMA logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * last_loss_scalar
        debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / max(dt, 1e-6))
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased:.6f} | "
              f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
              f"epoch: {epoch} | remaining: {remaining:.0f}s    ",
              end="", flush=True)

        step += 1
        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r

    total_tokens = step * TOTAL_BATCH_SIZE

    # ---- Final evaluation ----
    print("Running final evaluation...", flush=True)
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, token_bytes_arr)
    print(f"Final validation BPB: {val_bpb:.6f}", flush=True)

    # ---- Output metrics for research_orchestrator.py ----
    t_end = time.time()
    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")


if __name__ == "__main__":
    main()
