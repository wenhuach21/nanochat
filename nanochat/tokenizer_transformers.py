"""
Transformers-based tokenizer, mirroring nanochat's RustBPETokenizer / HuggingFaceTokenizer API.

This wraps `transformers.PreTrainedTokenizerFast` so you can:
- train a fresh GPT-4-style ByteLevel BPE (via the `tokenizers` backend), and
- run inference with the exact same nanochat interface (encode / decode /
  encode_special / get_bos_token_id / render_conversation / ...).

IMPORTANT NOTE ON TRAINING
--------------------------
The `transformers` library itself CANNOT train a BPE tokenizer from scratch.
Training always happens in the `tokenizers` (Rust) backend. So this class trains
with `tokenizers` (identical config to nanochat's HuggingFaceTokenizer) and then
*wraps* the result in a `PreTrainedTokenizerFast`. This is the standard and only
supported way to get a trained-from-scratch tokenizer into `transformers`.
"""

import os
import copy

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast

# Same special tokens and split pattern as nanochat.tokenizer.
# (defined locally to avoid importing nanochat.tokenizer, which pulls in rustbpe/tiktoken)
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Qwen / GPT-4 (cl100k) style split pattern, works well for mixed Chinese-English.
# Notes vs SPLIT_PATTERN above:
# - `\p{L}+` already matches CJK characters, so Chinese needs no special segmentation:
#   byte-level BPE + Chinese training data + a large-enough vocab is what actually matters.
# - Qwen splits numbers digit-by-digit (`\p{N}`), which is friendlier to large vocabs.
#   If you keep a small vocab you may prefer `\p{N}{1,2}` (as in SPLIT_PATTERN).
QWEN_SPLIT_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


# chat template mirroring RustBPETokenizer.render_conversation (user/assistant turns)
CHAT_TEMPLATE = (
    "{{ '<|bos|>' }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|user_start|>' + message['content'] + '<|user_end|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant_start|>' + message['content'] + '<|assistant_end|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant_start|>' }}{% endif %}"
)


class TransformersTokenizer:
    """nanochat-style wrapper around transformers.PreTrainedTokenizerFast."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, hf_path):
        # load any transformers tokenizer (local dir or hub id, e.g. "gpt2")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # load from a directory previously written by .save()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size, pattern=QWEN_SPLIT_PATTERN):
        # 1) Build + train the raw tokenizers BPE (transformers can't train from scratch).
        #    Config is identical to nanochat.HuggingFaceTokenizer, except the split pattern
        #    defaults to the Qwen/cl100k pattern so mixed Chinese-English works well.
        #    For CJK: `\p{L}+` already captures Chinese; what really matters is (a) having
        #    Chinese text in `text_iterator` and (b) a large-enough `vocab_size` so common
        #    Chinese chars/words become tokens instead of falling back to raw UTF-8 bytes.
        backend = HFTokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
        backend.normalizer = None
        split_regex = Regex(pattern)  # tokenizers requires the Regex wrapper
        backend.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        backend.decoder = decoders.ByteLevel()
        backend.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        backend.train_from_iterator(text_iterator, trainer)

        # 2) Wrap the trained backend into a transformers fast tokenizer.
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="<|bos|>",
            additional_special_tokens=SPECIAL_TOKENS,
            model_max_length=int(1e30),
            clean_up_tokenization_spaces=False,
            chat_template=CHAT_TEMPLATE,
        )
        return cls(tokenizer)

    # ------------------------------------------------------------------
    # metadata helpers (mirror RustBPETokenizer)
    # ------------------------------------------------------------------
    def get_vocab_size(self):
        return self.tokenizer.vocab_size + len(self.tokenizer.get_added_vocab())

    def get_special_tokens(self):
        return list(self.tokenizer.get_added_vocab().keys())

    def id_to_token(self, id):
        return self.tokenizer.convert_ids_to_tokens(id)

    def encode_special(self, text):
        # exact-match a single special token -> id
        return self.tokenizer.convert_tokens_to_ids(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        if bos is None or bos == self.tokenizer.unk_token_id:
            bos = self.encode_special("<|endoftext|>")
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    # ------------------------------------------------------------------
    # encode / decode (nanochat interface: returns raw list[int] / list[list[int]])
    # ------------------------------------------------------------------
    def _encode_one(self, text, prepend=None, append=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            ids.append(prepend if isinstance(prepend, int) else self.encode_special(prepend))
        # add_special_tokens=False => we manage BOS/special tokens ourselves
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
        if append is not None:
            ids.append(append if isinstance(append, int) else self.encode_special(append))
        return ids

    def encode(self, text, prepend=None, append=None, num_threads=None):
        # num_threads is ignored (kept for API compatibility with RustBPETokenizer)
        if isinstance(text, str):
            return self._encode_one(text, prepend=prepend, append=append)
        elif isinstance(text, list):
            return [self._encode_one(t, prepend=prepend, append=append) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_dir)
        print(f"Saved transformers tokenizer to {tokenizer_dir}")

    # ------------------------------------------------------------------
    # conversation rendering (identical logic to RustBPETokenizer)
    # ------------------------------------------------------------------
    def render_conversation(self, conversation, max_tokens=2048):
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"
            content = message["content"]
            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    add_tokens(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()
        ids, mask = self.render_conversation(conversation)
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids


if __name__ == "__main__":
    # tiny smoke test: train on a small mixed Chinese-English corpus, encode/decode round-trip
    corpus = [
        "Hello world! The capital of France is Paris.",
        "你好，世界！法国的首都是巴黎。",
        "机器学习和深度学习是人工智能的重要方向。",
        "def add(a, b):\n    return a + b  # 求和 123 + 456",
        "今天天气很好，我们一起去公园散步吧。",
        "Café naïve résumé — Ünïcödé 你好，世界 🚀",
    ] * 100
    # use a larger vocab so common Chinese words become single tokens
    tok = TransformersTokenizer.train_from_iterator(iter(corpus), vocab_size=2000)
    for text in ["Hello world!", "你好，世界！", "机器学习真有意思"]:
        ids = tok.encode(text, prepend="<|bos|>")
        print(f"{text!r:28} -> {len(ids)} ids: {ids}")
        print(f"{'decoded':28} -> {tok.decode(ids)!r}")
    print("bos id:", tok.get_bos_token_id())
    print("vocab size:", tok.get_vocab_size())

