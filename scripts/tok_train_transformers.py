"""
Train a tokenizer that is fully compatible with HuggingFace `transformers`
(loadable via `AutoTokenizer.from_pretrained(...)`), while keeping the exact same
nanochat interface (encode / decode / render_conversation / ...).

Under the hood this trains a GPT-4-style ByteLevel BPE with the `tokenizers` (Rust)
backend and wraps it in a `transformers.PreTrainedTokenizerFast` (see
`nanochat/tokenizer_transformers.py`). `transformers` itself cannot train from
scratch, so this is the standard supported path.

Usage:
    python -m scripts.tok_train_transformers --max-chars=2000000000 --vocab-size=32768
"""
import os
import time
import argparse

from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import compute_token_bytes, get_tokenizer_dir
from nanochat.tokenizer_transformers import TransformersTokenizer

# -----------------------------------------------------------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a transformers-compatible BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator (identical logic to scripts/tok_train.py)
def text_iterator():
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc[:args.doc_cap] if len(doc) > args.doc_cap else doc
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = TransformersTokenizer.train_from_iterator(text_iterator(), args.vocab_size)
train_time = time.time() - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk (writes tokenizer.json / tokenizer_config.json / ...)
tokenizer_dir = get_tokenizer_dir("transformers")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check (mixed English / Chinese round-trip)
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, "Round-trip encode/decode mismatch!"

# -----------------------------------------------------------------------------
# Verify it loads back through the plain transformers AutoTokenizer path
from transformers import AutoTokenizer
reloaded = AutoTokenizer.from_pretrained(tokenizer_dir)
assert reloaded.encode(test_text, add_special_tokens=False) == encoded, \
    "AutoTokenizer reload does not match the trained tokenizer!"
print(f"Verified AutoTokenizer.from_pretrained('{tokenizer_dir}') matches ✓")

# -----------------------------------------------------------------------------
# Cache token_bytes.pt for the bits-per-byte metric (same as scripts/tok_train.py)
import torch
token_bytes = compute_token_bytes(tokenizer, device="cpu")
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

