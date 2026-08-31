"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import json
import pickle
import rustbpe
import tiktoken


# ---- Helpers to convert a tiktoken encoding into a HuggingFace `tokenizers` BPE ----
# (needed so we can export a transformers-compatible tokenizer.json)

def _bytes_to_unicode():
    """GPT-2/ByteLevel reversible byte<->unicode mapping (same as tokenizers.ByteLevel)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _token_bytes_to_string(token_bytes, byte_encoder):
    return "".join(byte_encoder[b] for b in token_bytes)


def _bpe_split(mergeable_ranks, token, max_rank):
    """Re-run BPE on a single token to find the two parts it was merged from."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx, min_rank = None, None
        for i in range(len(parts) - 1):
            rank = mergeable_ranks.get(parts[i] + parts[i + 1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx, min_rank = i, rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def _recover_merges(mergeable_ranks):
    """Recover the ordered BPE merge rules from tiktoken's mergeable_ranks."""
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = _bpe_split(mergeable_ranks, token, max_rank=rank)
        if len(pair) != 2:
            continue
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def save_pretrained(self, save_dir, verify=True):
        """Export a transformers-compatible tokenizer (loadable via AutoTokenizer.from_pretrained).

        Writes tokenizer.json (HuggingFace `tokenizers` fast format), tokenizer_config.json and
        special_tokens_map.json. Internally converts the tiktoken BPE (mergeable_ranks) into an
        equivalent ByteLevel BPE with the same split pattern used at training time.

        If `verify` is True, the exported tokenizer is reloaded and its encodings are compared
        against this tokenizer on a few sample strings; a mismatch raises AssertionError. This
        guards against subtle bugs in the BPE merge recovery. If transformers isn't installed,
        verification is skipped with a warning.
        """
        os.makedirs(save_dir, exist_ok=True)

        mergeable_ranks = self.enc._mergeable_ranks  # dict[bytes, int]
        special_tokens = self.enc._special_tokens    # dict[str, int]

        byte_encoder = _bytes_to_unicode()
        id_to_bytes = {v: k for k, v in mergeable_ranks.items()}

        # 1) Build the vocab in ByteLevel-unicode space
        vocab = {_token_bytes_to_string(b, byte_encoder): i for b, i in mergeable_ranks.items()}
        # 2) Recover the ordered merge rules
        merges_dict = _recover_merges(mergeable_ranks)
        merges = []
        for (ix0, ix1), _rank in sorted(merges_dict.items(), key=lambda kv: kv[1]):
            s0 = _token_bytes_to_string(id_to_bytes[ix0], byte_encoder)
            s1 = _token_bytes_to_string(id_to_bytes[ix1], byte_encoder)
            merges.append((s0, s1))

        # 3) Assemble the fast tokenizer with the same pre-tokenizer/decoder as training
        hf_tokenizer = HFTokenizer(BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        hf_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        hf_tokenizer.decoder = decoders.ByteLevel()
        # add special tokens in ascending id order so their ids match the tiktoken ids
        specials_sorted = [name for name, _id in sorted(special_tokens.items(), key=lambda kv: kv[1])]
        hf_tokenizer.add_special_tokens(specials_sorted)

        tokenizer_path = os.path.join(save_dir, "tokenizer.json")
        hf_tokenizer.save(tokenizer_path)

        # 4) Determine BOS token name (nanochat uses <|bos|>, GPT-2 style uses <|endoftext|>)
        bos_token = None
        for cand in ("<|bos|>", "<|endoftext|>"):
            if cand in special_tokens:
                bos_token = cand
                break

        # 5) transformers metadata files
        special_tokens_map = {"additional_special_tokens": specials_sorted}
        if bos_token is not None:
            special_tokens_map["bos_token"] = bos_token
        with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

        # chat template mirroring RustBPETokenizer.render_conversation (user/assistant turns)
        chat_template = (
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
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": False,
            "model_max_length": int(1e30),
            "additional_special_tokens": specials_sorted,
            "chat_template": chat_template,
        }
        if bos_token is not None:
            tokenizer_config["bos_token"] = bos_token
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

        print(f"Saved transformers-compatible tokenizer to {save_dir}")

        if verify:
            self._verify_pretrained(save_dir)

    def _verify_pretrained(self, save_dir):
        """Reload the exported tokenizer and assert it encodes identically to this one."""
        try:
            from transformers import PreTrainedTokenizerFast
        except ImportError:
            print("[save_pretrained] transformers not installed; skipping verification.")
            return
        hf = PreTrainedTokenizerFast(tokenizer_file=os.path.join(save_dir, "tokenizer.json"))
        # Note: avoid strings that literally contain special tokens (e.g. "<|bos|>"), since the
        # fast tokenizer would split them out as special tokens while encode_ordinary would not.
        samples = [
            "Hello world! The capital of France is Paris.",
            "If 5*x + 3 = 13, then x is 2.",
            "def add(a, b):\n    return a + b  # sum 123 + 456",
            "Numbers 007 42 100 and symbols @#$%^&*()",
            "  leading spaces\tand\ttabs\nand newlines\n\n",
            "Café naïve résumé — Ünïcödé 你好，世界 🚀",
            "CamelCaseAndsnake_case_MixED 3.14159",
        ]
        mismatches = []
        for text in samples:
            expected = self.encode(text)
            got = hf.encode(text, add_special_tokens=False)
            if expected != got:
                mismatches.append((text, expected, got))
        if mismatches:
            lines = [f"  {text!r}\n    expected={exp}\n    got     ={got}" for text, exp, got in mismatches]
            raise AssertionError(
                "Exported tokenizer does not match the original on "
                f"{len(mismatches)}/{len(samples)} samples:\n" + "\n".join(lines)
            )
        print(f"[save_pretrained] Verified {len(samples)} samples encode identically ✓")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

# Directory names (under get_base_dir()) for each tokenizer backend.
TOKENIZER_DIRS = {
    "rustbpe": "tokenizer",              # trained by scripts/tok_train.py (RustBPE + tiktoken)
    "transformers": "tokenizer_transformers",  # transformers PreTrainedTokenizerFast
}

def get_tokenizer_dir(backend="rustbpe"):
    """Return the on-disk directory for a given tokenizer backend."""
    from nanochat.common import get_base_dir
    assert backend in TOKENIZER_DIRS, f"Unknown tokenizer backend: {backend} (choices: {list(TOKENIZER_DIRS)})"
    return os.path.join(get_base_dir(), TOKENIZER_DIRS[backend])

def get_tokenizer(backend="rustbpe"):
    """
    Load a tokenizer by backend:
    - "rustbpe":      the RustBPE/tiktoken tokenizer trained by scripts/tok_train.py
    - "transformers": a transformers PreTrainedTokenizerFast (nanochat.tokenizer_transformers)
    Both expose the same nanochat interface (encode/decode/get_bos_token_id/...).
    """
    tokenizer_dir = get_tokenizer_dir(backend)
    if backend == "rustbpe":
        return RustBPETokenizer.from_directory(tokenizer_dir)
    elif backend == "transformers":
        # imported lazily so that rustbpe-only workflows don't require transformers
        from nanochat.tokenizer_transformers import TransformersTokenizer
        return TransformersTokenizer.from_directory(tokenizer_dir)
    else:
        raise ValueError(f"Unknown tokenizer backend: {backend}")

def compute_token_bytes(tokenizer, device="cpu"):
    """
    Compute the (vocab_size,) int32 tensor of per-token byte lengths for ANY tokenizer
    that exposes get_vocab_size()/get_special_tokens()/decode(). Special tokens get 0 bytes
    (excluded from the bits-per-byte metric). Mirrors the logic in scripts/tok_train.py so
    that bpb can be evaluated for tokenizers that don't ship a cached token_bytes.pt.
    """
    import torch
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))
    return torch.tensor(token_bytes, dtype=torch.int32, device=device)

def get_token_bytes(device="cpu", backend="rustbpe"):
    import torch
    tokenizer_dir = get_tokenizer_dir(backend)
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    if os.path.exists(token_bytes_path):
        with open(token_bytes_path, "rb") as f:
            token_bytes = torch.load(f, map_location=device)
        return token_bytes
    # No cached file (e.g. transformers backend): compute on the fly from the tokenizer.
    if backend == "rustbpe":
        raise AssertionError(f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py")
    tokenizer = get_tokenizer(backend)
    return compute_token_bytes(tokenizer, device=device)
