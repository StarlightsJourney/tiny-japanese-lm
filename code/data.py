"""Download, clean and split the text, then tokenize it once per tokenizer.

Layout written under DATA_DIR:
  raw/{subs,wiki}_{train,val,test}.txt     one sentence/paragraph per line
  tok/<spec>/tokenizer.json
  tok/<spec>/{subs,wiki}_{train,val,test}.bin   uint16 token ids
  tok/<spec>/meta.json                     token and byte counts per split

Sources
  subs : Japanese side of JESC, ~2.8M lines of film/TV subtitles (Pryzant et al. 2018)
  wiki : Japanese Wikipedia (wikimedia/wikipedia, 20231101.ja dump), first N MB
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import tarfile
import unicodedata
from pathlib import Path

import numpy as np

from tokenizers_ja import new_tokenizer

SOURCES = ["subs", "wiki"]
SPLITS = ["train", "val", "test"]
JA = re.compile(r"[぀-ヿ一-鿿]")  # hiragana, katakana, kanji


def clean(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).replace("\t", " ").strip()
    return re.sub(r"\s+", " ", line)


def keep(line: str) -> bool:
    return len(line) >= 2 and JA.search(line) is not None


# ---------------------------------------------------------------- downloads
def _jesc_lines():
    """Japanese side of JESC. Tries Hugging Face mirrors, then the Stanford tarball."""
    from datasets import load_dataset

    for repo in ["Hoshikuzu/JESC", "nntsuzu/JESC"]:
        try:
            ds = load_dataset(repo, split="train")
            print(f"[subs] loaded {repo} ({len(ds):,} rows)")
            col = ds.column_names
            for row in ds:
                if "translation" in col:
                    yield row["translation"]["ja"]
                elif "ja" in col:
                    yield row["ja"]
            return
        except Exception as e:  # noqa: BLE001 - try the next mirror
            print(f"[subs] {repo} failed: {e!r}")
    import requests

    url = "https://nlp.stanford.edu/projects/jesc/data/raw.tar.gz"
    print(f"[subs] downloading {url}")
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                for raw in tar.extractfile(m):
                    parts = raw.decode("utf-8", errors="ignore").rstrip("\n").split("\t")
                    if len(parts) == 2:
                        yield parts[1]
                return
    raise RuntimeError("Could not download JESC from any source")


def _wiki_paragraphs(max_mb: int):
    """Yields (article_id, paragraph) from Japanese Wikipedia until max_mb of text."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
    total = 0
    for art in ds:
        for para in art["text"].split("\n"):
            yield art["id"], para
            total += len(para.encode("utf-8"))
        if total >= max_mb * 1_000_000:
            return


def _bucket(key: str) -> str:
    """Deterministic 99/0.5/0.5 split from a hash, so reruns give the same split."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 1000
    return "val" if h < 5 else "test" if h < 10 else "train"


def write_splits(lines_by_split: dict, raw_dir: Path, source: str):
    for split, lines in lines_by_split.items():
        (raw_dir / f"{source}_{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[{source}] {split}: {len(lines):,} lines")


def prepare_raw(data_dir: Path, wiki_mb: int = 600):
    raw = Path(data_dir) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # Subtitles: exact duplicates removed (subtitles repeat "はい" etc. constantly),
    # then each unique line is assigned to a split by its hash.
    seen, subs = set(), {s: [] for s in SPLITS}
    for line in _jesc_lines():
        line = clean(line)
        if keep(line) and line not in seen:
            seen.add(line)
            subs[_bucket(line)].append(line)
    write_splits(subs, raw, "subs")

    # Wikipedia: whole articles go to one split, so test articles are never seen.
    wiki = {s: [] for s in SPLITS}
    for art_id, para in _wiki_paragraphs(wiki_mb):
        para = clean(para)
        if keep(para) and len(para) >= 10:
            wiki[_bucket("wiki" + str(art_id))].append(para)
    write_splits(wiki, raw, "wiki")


# ---------------------------------------------------------------- tokenizing
def read(raw_dir: Path, source: str, split: str) -> str:
    return (Path(raw_dir) / f"{source}_{split}.txt").read_text(encoding="utf-8")


def tokenizer_sample(raw_dir: Path, mb_per_source: int = 40, seed: int = 0) -> list[str]:
    """Equal amounts of subtitle and Wikipedia text for learning the vocabulary."""
    rng = random.Random(seed)
    out = []
    for src in SOURCES:
        lines = read(raw_dir, src, "train").splitlines()
        rng.shuffle(lines)
        budget, chunk = mb_per_source * 1_000_000, []
        for line in lines:
            chunk.append(line)
            budget -= len(line.encode("utf-8"))
            if budget <= 0:
                break
        out.append("\n".join(chunk) + "\n")
    return out


def prepare_tokenizer(data_dir: Path, spec: str, mb_per_source: int = 40):
    data_dir = Path(data_dir)
    raw, out = data_dir / "raw", data_dir / "tok" / spec
    out.mkdir(parents=True, exist_ok=True)
    tok = new_tokenizer(spec).train(tokenizer_sample(raw, mb_per_source))
    tok.save(out / "tokenizer.json")

    meta = {"spec": spec, "vocab_size": tok.vocab_size, "splits": {}}
    for src in SOURCES:
        for split in SPLITS:
            text = read(raw, src, split)
            ids = tok.encode(text)
            assert ids.max() < tok.vocab_size
            ids.astype(np.uint16).tofile(out / f"{src}_{split}.bin")
            n_bytes = len(text.encode("utf-8"))
            assert int(tok.byte_lens[ids].sum()) == n_bytes, "tokenizer is not lossless"
            meta["splits"][f"{src}_{split}"] = {"tokens": int(len(ids)), "bytes": n_bytes,
                                               "bytes_per_token": n_bytes / len(ids)}
            print(f"[{spec}] {src}_{split}: {len(ids):,} tokens, {n_bytes / len(ids):.2f} bytes/token")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
