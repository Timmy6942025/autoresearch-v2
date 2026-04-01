"""
Framework-agnostic data pipeline for autoresearch experiments.

Shared between PyTorch (prepare.py) and MLX (train_mlx.py) training scripts.
Contains: tokenizer wrapper, parquet reading, BOS-aligned packing, token_bytes via numpy.

Usage:
    from shared_prepare import Tokenizer, make_dataloader_numpy, get_token_bytes_numpy
"""

import os
import math
import pickle
import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # the last datashard is shard_06542.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled by prepare.py."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


# ---------------------------------------------------------------------------
# Token bytes (numpy-based, no torch)
# ---------------------------------------------------------------------------

def ensure_token_bytes_npy():
    """
    Ensure token_bytes.npy exists alongside token_bytes.pt.
    Regenerates from the tokenizer pickle if the .npy file is missing.
    This allows shared_prepare to work without any torch dependency.
    """
    npy_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    if os.path.exists(npy_path):
        return

    # Need to regenerate from tokenizer pickle
    pkl_path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Tokenizer pickle not found at {pkl_path}. Run prepare.py first."
        )

    with open(pkl_path, "rb") as f:
        enc = pickle.load(f)

    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))

    token_bytes_arr = np.array(token_bytes_list, dtype=np.int32)
    np.save(npy_path, token_bytes_arr)


def get_token_bytes_numpy(device="cpu"):
    """Load token_bytes as a numpy array (no torch dependency)."""
    ensure_token_bytes_npy()
    npy_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    return np.load(npy_path)


# ---------------------------------------------------------------------------
# Document batches (framework-agnostic)
# ---------------------------------------------------------------------------

def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Numpy-based dataloader (BOS-aligned, best-fit packing)
# ---------------------------------------------------------------------------

def make_dataloader_numpy(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing — numpy version.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).

    Yields: (inputs_np, targets_np, epoch) where inputs/targets are numpy arrays
    of shape (B, T) with dtype int64.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    # Pre-allocate row buffer: numpy array of shape (B, T+1)
    row_buffer = np.zeros((B, row_capacity), dtype=np.int64)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = doc[:remaining]
                    pos += remaining

        inputs = row_buffer[:, :-1].copy()
        targets = row_buffer[:, 1:].copy()
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (framework-agnostic)
# ---------------------------------------------------------------------------

def evaluate_bpb_numpy(model_fn, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Takes a model_fn(x, y, reduction) that accepts numpy arrays and returns
    a numpy array of losses.

    model_fn: callable that takes (inputs_np, targets_np, reduction) and returns
              loss as numpy array (same shape as targets if reduction='none').
    """
    token_bytes_arr = get_token_bytes_numpy()
    val_loader = make_dataloader_numpy(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model_fn(x, y, reduction='none').reshape(-1)
        y_flat = y.reshape(-1)
        nbytes = token_bytes_arr[y_flat]
        mask = nbytes > 0
        total_nats += float(np.sum(loss_flat * mask))
        total_bytes += int(np.sum(nbytes))

    return total_nats / (math.log(2) * total_bytes)
