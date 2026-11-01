"""Experiment 4: does model surprise track how hard a sentence is for a learner?

1. Label sentences with a JLPT level (N5 easiest ... N1 hardest) from word lists:
   a sentence's level is the level of its hardest word. Sentences containing any
   content word that is in no list are dropped, so every label is grounded.
2. Score each sentence with a trained model (bits per byte, lower = less surprising).
3. Measure the rank correlation (Spearman) between score and level, and compare
   it with two simple baselines: sentence length and share of kanji.

Word lists: open-anki-jlpt-decks (MIT), vendored in data/jlpt/.
Sentences: Tatoeba Japanese sentences (CC BY 2.0 FR).
"""
from __future__ import annotations

import bz2
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

LEVELS = {"n5": 1, "n4": 2, "n3": 3, "n2": 4, "n1": 5}  # 1 = easiest
CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞", "形状詞", "連体詞", "感動詞"}
KANJI = re.compile(r"[一-鿿]")
KANA_ONLY = re.compile(r"^[぀-ヿー]+$")
TATOEBA = "https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2"


def load_jlpt(jlpt_dir: Path) -> dict[str, int]:
    """Word -> easiest JLPT level it appears at (a word can be in several lists)."""
    word_level: dict[str, int] = {}
    for name, lvl in LEVELS.items():
        with open(Path(jlpt_dir) / f"{name}.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                forms = [row["expression"].strip()]
                if KANA_ONLY.match(row["reading"].strip() or "x") and KANA_ONLY.match(forms[0] or "x"):
                    forms.append(row["reading"].strip())
                for w in forms:
                    if w and not re.search(r"[～〜()（）\s/・]", w):
                        word_level[w] = min(lvl, word_level.get(w, 99))
    return word_level


def word_levels(sentence: str, tagger, word_level: dict[str, int]):
    """Levels of the content words, or None if any content word is unknown."""
    levels = []
    for w in tagger(sentence):
        f = w.feature
        if f.pos1 not in CONTENT_POS or f.pos2 in ("固有名詞", "数詞"):
            continue
        cands = {w.surface, getattr(f, "orthBase", None) or "", (f.lemma or "").split("-")[0]}
        found = [word_level[c] for c in cands if c in word_level]
        if not found:
            return None
        levels.append(min(found))
    return levels or None


def download_tatoeba(dest: Path) -> list[str]:
    import requests

    r = requests.get(TATOEBA, timeout=300)
    r.raise_for_status()
    lines = bz2.decompress(r.content).decode("utf-8").splitlines()
    sents = [ln.split("\t")[2] for ln in lines if ln.count("\t") >= 2]
    Path(dest).write_text("\n".join(sents) + "\n", encoding="utf-8")
    return sents


def build_set(sentences: list[str], jlpt_dir: Path, out_path: Path,
              per_level: int = 400, seed: int = 0) -> dict:
    import fugashi

    tagger = fugashi.Tagger()
    wl = load_jlpt(jlpt_dir)
    by_level: dict[int, list] = {l: [] for l in LEVELS.values()}
    for s in dict.fromkeys(x.strip() for x in sentences):  # dedupe, keep order
        if not (5 <= len(s) <= 40):
            continue
        lv = word_levels(s, tagger, wl)
        if lv and len(lv) >= 2:
            by_level[max(lv)].append(s)
    rng = random.Random(seed)
    rows = []
    for lvl, sents in by_level.items():
        rng.shuffle(sents)
        for s in sents[:per_level]:
            rows.append({"sentence": s, "level": lvl, "n_chars": len(s),
                         "kanji_ratio": len(KANJI.findall(s)) / len(s)})
    Path(out_path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                              encoding="utf-8")
    counts = {f"N{6 - l}": min(len(v), per_level) for l, v in by_level.items()}
    print(f"[difficulty] kept sentences per level: {counts}")
    return counts


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_ci(a, b, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    vals = [spearman(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate_set(rows: list[dict], scores: list[float]) -> dict:
    levels = [r["level"] for r in rows]
    feats = {"model_bpb": scores,
             "length_chars": [r["n_chars"] for r in rows],
             "kanji_ratio": [r["kanji_ratio"] for r in rows]}
    out = {"n": len(rows), "spearman": {}, "ci95": {}, "mean_by_level": {}}
    for k, v in feats.items():
        out["spearman"][k] = spearman(v, levels)
        out["ci95"][k] = bootstrap_ci(v, levels)
        out["mean_by_level"][k] = {f"N{6 - l}": float(np.mean([x for x, y in zip(v, levels) if y == l]))
                                   for l in sorted(set(levels))}
    return out
