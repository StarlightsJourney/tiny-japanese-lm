"""Three ways to split Japanese text into tokens, all lossless.

Every tokenizer can encode ANY string (unknown characters fall back to their
UTF-8 bytes), so bits-per-byte is computed on exactly the same text for all of
them. Each tokenizer also exposes `byte_lens`: how many UTF-8 bytes each token
id stands for. That lets us turn a summed token loss into bits per byte exactly.

Specs:
  "byte"      256 tokens, one per byte (a Japanese character = 3 tokens)
  "char"      the most frequent characters + 256 byte fallbacks
  "bpe4k"     byte-level BPE with 4,096 tokens (learned word pieces)
  "bpe16k"    byte-level BPE with 16,384 tokens
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


class ByteTokenizer:
    name = "byte"

    def __init__(self):
        self.vocab_size = 256
        self.byte_lens = np.ones(256, dtype=np.int64)

    def train(self, texts):
        return self

    def encode(self, text: str) -> np.ndarray:
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.uint16)

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"kind": "byte"}))


class CharTokenizer:
    """Ids 0-255 are raw bytes (fallback); ids 256+ are whole characters."""

    name = "char"

    def __init__(self, max_chars: int = 8000, min_count: int = 5):
        self.max_chars = max_chars
        self.min_count = min_count
        self.chars: list[str] = []

    def train(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(t)
        keep = [c for c, n in counts.most_common(self.max_chars) if n >= self.min_count]
        self._set_chars(keep)
        return self

    def _set_chars(self, chars):
        self.chars = list(chars)
        self.vocab_size = 256 + len(self.chars)
        self.table = np.full(0x110000, -1, dtype=np.int64)
        for i, c in enumerate(self.chars):
            self.table[ord(c)] = 256 + i
        self.byte_lens = np.array(
            [1] * 256 + [len(c.encode("utf-8")) for c in self.chars], dtype=np.int64
        )

    def encode(self, text: str) -> np.ndarray:
        cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        ids = self.table[cps]
        miss = ids < 0
        if not miss.any():
            return ids.astype(np.uint16)
        # Unknown characters become their UTF-8 bytes (vectorised expansion).
        lens = np.ones(len(cps), dtype=np.int64)
        fallback = {int(cp): np.frombuffer(chr(cp).encode("utf-8"), dtype=np.uint8)
                    for cp in np.unique(cps[miss])}
        for cp, b in fallback.items():
            lens[cps == cp] = len(b)
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        out = np.empty(int(lens.sum()), dtype=np.uint16)
        out[offsets[~miss]] = ids[~miss]
        for cp, b in fallback.items():
            where = offsets[cps == cp]
            for k, byte in enumerate(b):
                out[where + k] = byte
        return out

    def decode(self, ids) -> str:
        buf = bytearray()
        for i in ids:
            i = int(i)
            buf += bytes([i]) if i < 256 else self.chars[i - 256].encode("utf-8")
        return buf.decode("utf-8", errors="replace")

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"kind": "char", "chars": self.chars}, ensure_ascii=False))


class BPETokenizer:
    """Byte-level BPE (GPT-2 style) trained with Hugging Face `tokenizers`.

    The tokenizer algorithm/implementation is reused; the merges are learned
    from our own training text, so nothing pretrained is involved.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.name = f"bpe{vocab_size // 1024}k"
        self.tok = None

    def train(self, texts):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=False,
        )
        lines = (line for t in texts for line in t.splitlines(keepends=True))
        tok.train_from_iterator(lines, trainer=trainer)
        self._set(tok)
        return self

    def _set(self, tok):
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()
        # In byte-level BPE each symbol of a token string is exactly one byte.
        self.byte_lens = np.zeros(self.vocab_size, dtype=np.int64)
        for s, i in tok.get_vocab().items():
            self.byte_lens[i] = len(s)

    def encode(self, text: str) -> np.ndarray:
        lines = text.splitlines(keepends=True)
        parts = []
        for start in range(0, len(lines), 100_000):
            encs = self.tok.encode_batch(lines[start:start + 100_000], add_special_tokens=False)
            parts.append(np.fromiter((i for e in encs for i in e.ids), dtype=np.uint16))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)

    def decode(self, ids) -> str:
        return self.tok.decode([int(i) for i in ids])

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"kind": "bpe", "tokenizer": self.tok.to_str()}))


SPECS = ["byte", "char", "bpe4k", "bpe16k"]


def new_tokenizer(spec: str):
    if spec == "byte":
        return ByteTokenizer()
    if spec == "char":
        return CharTokenizer()
    if spec.startswith("bpe") and spec.endswith("k"):
        return BPETokenizer(int(spec[3:-1]) * 1024)
    raise ValueError(f"unknown tokenizer spec {spec!r}; choose from {SPECS}")


def load_tokenizer(path: Path):
    d = json.loads(Path(path).read_text())
    if d["kind"] == "byte":
        return ByteTokenizer()
    if d["kind"] == "char":
        t = CharTokenizer()
        t._set_chars(d["chars"])
        return t
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(d["tokenizer"])
    t = BPETokenizer(tok.get_vocab_size())
    t._set(tok)
    return t
