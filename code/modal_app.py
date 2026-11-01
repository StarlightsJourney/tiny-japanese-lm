"""Runs everything on Modal's cloud GPUs. Your laptop only sends commands.

One-time setup (see readme.md):  pip install modal  &&  modal setup

  modal run code/modal_app.py --exp prepare      # download + tokenize data (CPU, ~1 h, ~$1.50)
  modal run code/modal_app.py --exp smoke        # 3-minute GPU check (~$0.05)
  modal run code/modal_app.py --exp lr           # learning-rate sweep (12 x 6 min, ~$1.50)
  modal run code/modal_app.py --exp tokenizer    # experiment 1 (4 x 30 min, ~$2)
  modal run code/modal_app.py --exp size         # experiment 2
  modal run code/modal_app.py --exp data         # experiment 3
  modal run code/modal_app.py --exp seeds        # repeat the baseline with 2 more seeds
  modal run code/modal_app.py --exp difficulty --run base   # experiment 4

Runs in one experiment train in parallel on separate GPUs. Results are written
to results/<run>.json on your computer; runs that already have a result file are
skipped (pass --force to rerun).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GPU = os.environ.get("TJLM_GPU", "L4")  # the declared hardware; keep it fixed for all runs
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.6.0", "numpy==2.2.4", "tokenizers==0.21.1", "datasets==3.6.0",
                    "requests==2.32.3", "fugashi==1.4.0", "unidic-lite==1.0.8")
    .add_local_dir(HERE, "/root/code", ignore=["__pycache__"])
    .add_local_dir(ROOT / "data" / "jlpt", "/root/jlpt")
)
app = modal.App("tiny-japanese-lm", image=image)
volume = modal.Volume.from_name("tiny-japanese-lm", create_if_missing=True)


def _setup():
    sys.path.insert(0, "/root/code")
    volume.reload()


@app.function(cpu=8, memory=32768, timeout=4 * 3600, volumes={VOL: volume})
def prepare_raw(wiki_mb: int = 600):
    _setup()
    from data import prepare_raw as run

    run(Path(VOL) / "data", wiki_mb=wiki_mb)
    volume.commit()


@app.function(cpu=8, memory=32768, timeout=3 * 3600, volumes={VOL: volume})
def prepare_tokenizer(spec: str):
    _setup()
    from data import prepare_tokenizer as run

    meta = run(Path(VOL) / "data", spec)
    volume.commit()
    return meta


@app.function(cpu=4, memory=8192, timeout=3600, volumes={VOL: volume})
def prepare_difficulty():
    _setup()
    from difficulty import build_set, download_tatoeba

    d = Path(VOL) / "data"
    try:
        sents = download_tatoeba(d / "tatoeba_jpn.txt")
        source = "tatoeba"
    except Exception as e:  # noqa: BLE001
        print(f"[difficulty] Tatoeba download failed ({e!r}); using held-out subtitle test lines")
        sents = (d / "raw" / "subs_test.txt").read_text(encoding="utf-8").splitlines()
        source = "subs_test"
    counts = build_set(sents, Path("/root/jlpt"), d / "difficulty.jsonl")
    (d / "difficulty_source.json").write_text(json.dumps({"source": source, "counts": counts}))
    volume.commit()
    return {"source": source, "counts": counts}


@app.function(gpu=GPU, memory=16384, timeout=2 * 3600, volumes={VOL: volume})
def train_remote(cfg: dict):
    _setup()
    from train import train

    result = train(cfg, f"{VOL}/data", VOL, "cuda")
    volume.commit()
    return result


@app.function(gpu=GPU, timeout=1800, volumes={VOL: volume})
def difficulty_remote(run: str):
    _setup()
    import torch

    from difficulty import evaluate_set
    from evaluate import sentence_bits_per_byte
    from model import GPT, GPTConfig
    from tokenizers_ja import load_tokenizer

    ck = torch.load(f"{VOL}/ckpt/{run}.pt", map_location="cuda")
    model = GPT(GPTConfig(**ck["model_config"])).cuda()
    model.load_state_dict(ck["state"])
    tok = load_tokenizer(Path(VOL) / "data" / "tok" / ck["cfg"]["tokenizer"] / "tokenizer.json")
    d = Path(VOL) / "data"
    rows = [json.loads(x) for x in (d / "difficulty.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = sentence_bits_per_byte(model, tok, [r["sentence"] for r in rows], "cuda")
    out = evaluate_set(rows, scores)
    out["run"] = run
    out["sentence_source"] = json.loads((d / "difficulty_source.json").read_text())
    out["per_sentence"] = [{**r, "model_bpb": s} for r, s in zip(rows, scores)]
    return out


@app.local_entrypoint()
def main(exp: str = "smoke", run: str = "base", force: bool = False, wiki_mb: int = 600):
    sys.path.insert(0, str(HERE))
    from configs import experiment
    from tokenizers_ja import SPECS

    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    if exp == "prepare":
        prepare_raw.remote(wiki_mb)
        metas = list(prepare_tokenizer.map(SPECS))
        (results / "tokenizer_stats.json").write_text(json.dumps(metas, indent=2))
        print(json.dumps(prepare_difficulty.remote(), ensure_ascii=False))
        print("Data ready. Next: modal run code/modal_app.py --exp smoke")
        return

    if exp == "difficulty":
        out = difficulty_remote.remote(run)
        (results / f"difficulty_{run}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(json.dumps({k: out[k] for k in ("n", "spearman", "ci95")}, indent=2))
        return

    cfgs = [c for c in experiment(exp) if force or not (results / f"{c['name']}.json").exists()]
    for c in experiment(exp):
        if c not in cfgs:
            print(f"skip {c['name']} (results/{c['name']}.json exists; --force to rerun)")
    print(f"launching {len(cfgs)} run(s) on {GPU}: {[c['name'] for c in cfgs]}")
    for res in train_remote.map(cfgs, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"FAILED: {res!r}")
            continue
        (results / f"{res['name']}.json").write_text(json.dumps(res, indent=2))
        print(f"{res['name']}: test bpb subs {res['test_bpb_subs']:.4f}  wiki {res['test_bpb_wiki']:.4f}")
    print("Make figures with: python code/plot.py")
