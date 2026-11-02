"""Follow-up analyses on saved results. No GPU, no training.

  python code/analysis.py   -> results/analysis.json and report/figures/compute.pdf

1. Difficulty: does model surprise add to sentence length? Fit a linear score
   on some sentences, test its rank correlation with JLPT level on the rest
   (5-fold cross-validation), and compare feature sets.
2. Equal compute: the size experiment gave every model the same time, so
   bigger models, which use the GPU better, also got more compute. Compare
   validation BPB at the same number of training FLOPs instead.
3. Memorisation: compare training loss with validation loss in the data-mix
   runs. A growing gap means the model is starting to memorise repeated text.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from difficulty import spearman  # noqa: E402
from plot import MARKERS, PALETTE, H, W  # noqa: E402  (also applies the figure style)

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("TJLM_RESULTS", ROOT / "results"))
FIG = Path(os.environ.get("TJLM_FIGURES", ROOT / "report" / "figures"))


def load(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text())


# ---------------------------------------------------------------- 1. difficulty
FEATURE_SETS = {
    "length": ["n_chars"],
    "length+kanji": ["n_chars", "kanji_ratio"],
    "bpb": ["model_bpb"],
    "length+bpb": ["n_chars", "model_bpb"],
    "length+kanji+bpb": ["n_chars", "kanji_ratio", "model_bpb"],
}


def cv_predictions(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of a least-squares fit of level on features."""
    idx = np.random.default_rng(seed).permutation(len(y))
    pred = np.empty(len(y))
    for k in range(folds):
        test = idx[k::folds]
        train = np.setdiff1d(idx, test)
        mu, sd = X[train].mean(0), X[train].std(0)
        A = np.c_[np.ones(len(train)), (X[train] - mu) / sd]
        w, *_ = np.linalg.lstsq(A, y[train], rcond=None)
        pred[test] = np.c_[np.ones(len(test)), (X[test] - mu) / sd] @ w
    return pred


def partial_spearman(a, b, c) -> float:
    """Rank correlation of a and b after removing the linear effect of c's ranks from both."""
    ra, rb, rc = (np.argsort(np.argsort(v)).astype(float) for v in (a, b, c))
    ea = ra - np.polyval(np.polyfit(rc, ra, 1), rc)
    eb = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
    return float(np.corrcoef(ea, eb)[0, 1])


def difficulty_combo(run: str, n_boot: int = 1000) -> dict:
    rows = load(f"difficulty_{run}")["per_sentence"]
    y = np.array([r["level"] for r in rows], dtype=float)
    bpb = np.array([r["model_bpb"] for r in rows])
    n = np.array([r["n_chars"] for r in rows], dtype=float)
    preds = {k: cv_predictions(np.array([[r[f] for f in fs] for r in rows], dtype=float), y)
             for k, fs in FEATURE_SETS.items()}
    out = {k: spearman(p, y) for k, p in preds.items()}
    out["bpb_vs_length"] = spearman(bpb, n)
    out["partial_bpb_level_given_length"] = partial_spearman(bpb, y, n)
    # Paired bootstrap: is length+bpb really better than length alone? Also the
    # interval of the partial correlation.
    rng = np.random.default_rng(0)
    diffs, partials = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(spearman(preds["length+bpb"][i], y[i]) - spearman(preds["length"][i], y[i]))
        partials.append(partial_spearman(bpb[i], y[i], n[i]))
    out["gain_length+bpb_over_length"] = float(np.mean(diffs))
    out["gain_ci95"] = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    out["partial_ci95"] = (float(np.percentile(partials, 2.5)), float(np.percentile(partials, 97.5)))
    return out


# ---------------------------------------------------------------- 2. compute
SIZE_RUNS = ["size_xs", "base", "size_m", "size_l"]


def equal_compute() -> dict:
    """Validation BPB at the largest FLOP count every size reached.

    Caveat: the learning rate follows each run's own time schedule, so a big
    model measured part-way through has not finished its decay yet. That
    makes this comparison slightly unfair to the bigger models.
    """
    runs = {n: load(n) for n in SIZE_RUNS}
    flops = {n: np.array([6 * r["params_total"] * p["tokens"] for p in r["curve"]]) for n, r in runs.items()}
    budget = min(f[-1] for f in flops.values())
    out = {"flops": float(budget), "val_bpb_subs": {}, "fraction_of_run": {}}
    for n, r in runs.items():
        bpb = np.array([p["val_bpb_subs"] for p in r["curve"]])
        f = flops[n]
        out["val_bpb_subs"][r["config"]["size"]] = float(np.interp(math.log(budget), np.log(f[1:]), bpb[1:]))
        out["fraction_of_run"][r["config"]["size"]] = float(budget / f[-1])

    fig, ax = plt.subplots(figsize=(W, H))
    for i, (n, r) in enumerate(runs.items()):
        c = r["curve"][1:]
        ax.plot(flops[n][1:], [p["val_bpb_subs"] for p in c], color=PALETTE[i], marker=MARKERS[i], markevery=3,
                label=f"{r['config']['size']} ({r['params_total'] / 1e6:.0f}M total)")
    ax.axvline(budget, color="#777777", lw=0.8, ls=":")
    ax.set_xscale("log")
    ticks = [3e15, 1e16, 3e16]
    ax.set_xticks(ticks, [f"{t:.0e}".replace("e+", "e") for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("training FLOPs (6 x params x tokens)")
    ax.set_ylabel("val bits per byte (subtitles)")
    ax.set_ylim(1.3, 1.6)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "compute.pdf")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 3. memorisation
DATA_RUNS = ["data_subs0", "base", "base_seed1", "base_seed2", "data_subs0.5", "data_subs1"]


def memorisation() -> dict:
    """Validation minus training loss at the end of each data-mix run.

    Training loss is a running average over training batches (nats per
    token), drawn from the run's own subtitle/Wikipedia mix. We build the
    matching validation loss from the per-source validation BPB, weighting
    sources by their share of training tokens, and convert the gap to bits
    per byte. The running average lags the model slightly, so the gap is if
    anything an underestimate.
    """
    stats = {s["spec"]: s["splits"] for s in json.loads((RES / "tokenizer_stats.json").read_text())}
    out = {}
    for n in DATA_RUNS:
        r = load(n)
        sp = stats[r["config"]["tokenizer"]]
        w = r["subs_weight_used"]
        bpt = {s: sp[f"{s}_val"]["bytes_per_token"] for s in ("subs", "wiki")}
        last = r["curve"][-1]
        val_nats = (w * last["val_bpb_subs"] * math.log(2) * bpt["subs"]
                    + (1 - w) * last["val_bpb_wiki"] * math.log(2) * bpt["wiki"])
        mix_bpt = w * bpt["subs"] + (1 - w) * bpt["wiki"]
        gap = (val_nats - last["train_loss"]) / (math.log(2) * mix_bpt)
        out[n] = {"subs_weight": w, "epochs_subs": r["epochs_subs"], "epochs_wiki": r["epochs_wiki"],
                  "train_bpb": last["train_loss"] / (math.log(2) * mix_bpt),
                  "val_bpb": val_nats / (math.log(2) * mix_bpt), "gap_bpb": gap}
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    result = {
        "difficulty_combo": {run: difficulty_combo(run) for run in ("base", "data_subs0.5", "data_subs1")},
        "equal_compute": equal_compute(),
        "memorisation": memorisation(),
    }
    (RES / "analysis.json").write_text(json.dumps(result, indent=2))

    print("1. Difficulty, Spearman rho of 5-fold out-of-sample score with JLPT level")
    for run, d in result["difficulty_combo"].items():
        cells = "  ".join(f"{k} {d[k]:.3f}" for k in FEATURE_SETS)
        lo, hi = d["gain_ci95"]
        print(f"  {run:13} {cells}  | length+bpb minus length {d['gain_length+bpb_over_length']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"  | partial {d['partial_bpb_level_given_length']:.3f} {tuple(round(x, 3) for x in d['partial_ci95'])}"
              f"  | bpb vs length {d['bpb_vs_length']:.3f}")
    ec = result["equal_compute"]
    print(f"2. Validation BPB (subtitles) at {ec['flops']:.2e} FLOPs")
    for s, v in ec["val_bpb_subs"].items():
        print(f"  {s:3} {v:.4f}  (at {ec['fraction_of_run'][s]:.0%} of its run)")
    print("3. Validation minus training BPB at the end of each run")
    for n, m in result["memorisation"].items():
        print(f"  {n:13} subs {m['subs_weight']:.2f}  epochs subs {m['epochs_subs']:5.2f} wiki {m['epochs_wiki']:4.2f}"
              f"  train {m['train_bpb']:.3f}  val {m['val_bpb']:.3f}  gap {m['gap_bpb']:+.3f}")
    print(f"wrote {RES / 'analysis.json'} and {FIG / 'compute.pdf'}")


if __name__ == "__main__":
    main()
