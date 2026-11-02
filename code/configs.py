"""Every run in the report, defined in one place.

Edit LR_BY_SIZE after the learning-rate sweep, and BEST_TOKENIZER after
experiment 1, then run the later experiments.
"""
from __future__ import annotations

BUDGET_S = 1800               # declared budget: 30 min of training per run on one L4 GPU
SWEEP_BUDGET_S = 360          # learning-rate sweep runs are preliminary (6 min, not compared)
BEST_TOKENIZER = "bpe16k"     # update after experiment 1 if another tokenizer wins
# Best validation BPB (subtitles) in the 6-minute sweep, each with a worse rate
# on both sides. The first guesses were xs 3e-3, s 2e-3, m 1.5e-3, l 1e-3; s, m
# and l preferred the lowest, so the sweep was rerun centred on these values.
LR_BY_SIZE = {"xs": 3e-3, "s": 1e-3, "m": 7.5e-4, "l": 5e-4}

BASE = dict(tokenizer="bpe16k", size="s", subs_weight="natural", seed=0, budget_s=BUDGET_S)


def run(name: str, **over) -> dict:
    cfg = {**BASE, **over, "name": name}
    cfg.setdefault("lr", LR_BY_SIZE[cfg["size"]])
    return cfg


def experiment(exp: str) -> list[dict]:
    if exp == "smoke":  # 3-minute check that everything works on the GPU
        return [run("smoke", budget_s=180, eval_every_s=60)]
    if exp == "lr":
        return [run(f"lr_{size}_{lr:g}", size=size, lr=lr, tokenizer=BEST_TOKENIZER,
                    budget_s=SWEEP_BUDGET_S)
                for size in LR_BY_SIZE for lr in (LR_BY_SIZE[size] / 2, LR_BY_SIZE[size], LR_BY_SIZE[size] * 2)]
    if exp == "tokenizer":
        return [run("base")] + [run(f"tok_{t}", tokenizer=t) for t in ("byte", "char", "bpe4k")]
    if exp == "size":
        same_as_base = lambda s: s == BASE["size"] and BEST_TOKENIZER == BASE["tokenizer"]  # noqa: E731
        return [run("base") if same_as_base(s) else run(f"size_{s}", size=s, tokenizer=BEST_TOKENIZER)
                for s in ("xs", "s", "m", "l")]
    if exp == "data":
        return [run("base")] + [run(f"data_subs{w:g}", subs_weight=w) for w in (0.0, 0.5, 1.0)]
    if exp == "seeds":
        return [run("base")] + [run(f"base_seed{s}", seed=s) for s in (1, 2)]
    raise ValueError(f"unknown experiment {exp!r}")


EXPERIMENTS = ["smoke", "lr", "tokenizer", "size", "data", "seeds"]
