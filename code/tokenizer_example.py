"""Show how each tokenizer splits one sentence, for the report's tokenizer table.

The trained tokenizers live on the Modal volume. Copy them down first:

  for t in char bpe4k bpe16k; do
    modal volume get tiny-japanese-lm data/tok/$t/tokenizer.json data/tok/$t/tokenizer.json
  done
  python code/tokenizer_example.py      -> results/tokenizer_example.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers_ja import ByteTokenizer, CharTokenizer, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
SENTENCE = "心配しないで、俺がやるから。"  # "Don't worry, I'll do it."


def _gpt2_byte_decoder() -> dict[str, int]:
    """Byte-level BPE stores each byte as one printable character (GPT-2 scheme); this undoes it."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _gpt2_byte_decoder()


def token_bytes(tok, i: int) -> bytes:
    """The exact UTF-8 bytes one token id stands for."""
    if isinstance(tok, ByteTokenizer) or (isinstance(tok, CharTokenizer) and i < 256):
        return bytes([i])
    if isinstance(tok, CharTokenizer):
        return tok.chars[i - 256].encode("utf-8")
    return bytes(BYTE_DECODER[c] for c in tok.tok.id_to_token(i))


def pieces(tok, text: str) -> list[str]:
    """The text each token stands for. A token that holds part of a character shows as <hex>."""
    out = []
    for i in tok.encode(text):
        b = token_bytes(tok, int(i))
        try:
            out.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(f"<{b.hex()}>")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tok_dir", default=str(ROOT / "data" / "tok"))
    a = p.parse_args()
    result = {"sentence": SENTENCE, "bytes": len(SENTENCE.encode("utf-8")), "tokenizers": {}}
    for spec in ("byte", "char", "bpe4k", "bpe16k"):
        tok = ByteTokenizer() if spec == "byte" else load_tokenizer(Path(a.tok_dir) / spec / "tokenizer.json")
        ps = pieces(tok, SENTENCE)
        result["tokenizers"][spec] = {"n_tokens": len(ps), "pieces": ps}
        print(f"{spec:7} {len(ps):2} tokens: {' | '.join(ps)}")
    (ROOT / "results" / "tokenizer_example.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
