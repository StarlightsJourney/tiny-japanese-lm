"""Turn results/*.json into the report's figures and a summary table.

  python code/plot.py            -> report/figures/*.pdf and results/summary.csv
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("TJLM_RESULTS", ROOT / "results"))
FIG = Path(os.environ.get("TJLM_FIGURES", ROOT / "report" / "figures"))
# Categorical colours in fixed order, checked for colour-blind separation.
# The lighter two fall below 3:1 contrast, so every series also gets a marker
# shape and a direct label or legend.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED = "#111111", "#555555"
# Large, dark text: the figures are read on paper and on screens at reduced size.
plt.rcParams.update({"font.size": 9.5, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
                     "lines.linewidth": 1.6, "lines.markersize": 5,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False})
W, H = 3.3, 2.3  # one column of a two-up figure in the report


def load() -> dict[str, dict]:
    runs = {}
    for p in RES.glob("*.json"):
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "test_bpb_subs" in d:
            runs[d["name"]] = d
    return runs


def summary(runs):
    cols = ["name", "tokenizer", "size", "lr", "subs_weight_used", "seed", "params_non_embedding",
            "tokens_seen", "tokens_per_param", "tokens_per_second", "epochs_subs", "epochs_wiki",
            "context_bytes", "test_bpb_subs", "test_bpb_wiki"]
    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(runs.values(), key=lambda r: r["name"]):
            row = {**r["config"], **r}
            w.writerow([round(row[c], 4) if isinstance(row[c], float) else row[c] for c in cols])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def curves(runs, names, label, fname, ylim=(1.2, 1.7)):
    names = [n for n in names if n in runs]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(W, H))
    for i, n in enumerate(names):
        c = runs[n]["curve"][1:]  # skip the untrained step-1 point
        ax.plot([p["train_s"] / 60 for p in c], [p["val_bpb_subs"] for p in c],
                color=PALETTE[i], marker=MARKERS[i], markevery=3, label=label(runs[n]))
    ax.set_xlabel("training time (min)")
    ax.set_ylabel("val bits per byte (subtitles)")
    ax.set_ylim(*ylim)
    ax.legend()
    save(fig, fname)


def learning_curves(runs):
    """One figure with how each experiment's models improve during training.

    (a) the training loss itself (cross-entropy, nats per token) for the three
    baseline seeds; (b)-(d) validation bits per byte on subtitles.
    """
    panels = [
        ("(a) Baseline: training loss, 3 seeds", ["base", "base_seed1", "base_seed2"],
         lambda r: f"seed {r['config']['seed']}", "train_loss", "training loss (nats per token)", (3.4, 6.0), "upper right"),
        ("(b) Tokenizer", ["tok_byte", "tok_char", "tok_bpe4k", "base"],
         lambda r: r["config"]["tokenizer"], "val_bpb_subs", "val bits per byte (subtitles)", (1.25, 1.75), "upper right"),
        ("(c) Model size", ["size_xs", "base", "size_m", "size_l"],
         lambda r: f"{r['config']['size']} ({r['params_non_embedding'] / 1e6:.1f}M)", "val_bpb_subs",
         "val bits per byte (subtitles)", (1.25, 1.75), "upper right"),
        ("(d) Share of subtitles in training", ["data_subs0", "base", "data_subs0.5", "data_subs1"],
         lambda r: f"{r['subs_weight_used']:.0%}", "val_bpb_subs", "val bits per byte (subtitles)", (1.2, 2.35), "upper right"),
    ]
    if not all(n in runs for _, names, *_ in panels for n in names):
        return
    fig, axes = plt.subplots(2, 2, figsize=(2 * W, 2 * H + 0.3))
    for ax, (title, names, label, key, ylabel, ylim, loc) in zip(axes.flat, panels):
        for i, n in enumerate(names):
            c = runs[n]["curve"][1:]  # skip the untrained step-1 point
            ax.plot([p["train_s"] / 60 for p in c], [p[key] for p in c], color=PALETTE[i],
                    marker=MARKERS[i], markevery=3, label=label(runs[n]))
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=8.5, loc=loc)
    for ax in axes[1]:
        ax.set_xlabel("training time (minutes)")
    save(fig, "curves.pdf")


def tokenizer_plot(runs):
    names = [n for n in ("tok_byte", "tok_char", "tok_bpe4k", "base") if n in runs]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(W, H))
    xs = range(len(names))
    for i, (key, lab) in enumerate((("test_bpb_subs", "subtitles"), ("test_bpb_wiki", "Wikipedia"))):
        ys = [runs[n][key] for n in names]
        ax.plot(xs, ys, color=PALETTE[i], marker=MARKERS[i], label=lab)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    ax.set_xticks(list(xs), [f"{runs[n]['config']['tokenizer']}\n{runs[n]['tokens_per_second'] / 1e3:.0f}k tok/s"
                             for n in names])
    ax.set_ylabel("test bits per byte")
    ax.set_ylim(1.1, 1.48)
    ax.legend(loc="upper right")
    save(fig, "tokenizer.pdf")


def size_plot(runs):
    names = [n for n in runs if n.startswith("size_") or n == "base"]
    pts = sorted((runs[n]["params_non_embedding"], runs[n]["test_bpb_subs"], runs[n]["tokens_per_param"])
                 for n in names if runs[n]["config"]["subs_weight"] == "natural" and runs[n]["config"]["seed"] == 0)
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], color=PALETTE[0], marker="o")
    for p in pts:
        ax.annotate(f"{p[2]:.0f} tok/param", (p[0] / 1e6, p[1]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7, color=INK)
    ax.set_xscale("log")
    ticks = [p[0] / 1e6 for p in pts]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}" if v < 10 else f"{v:.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ticks[0] / 1.8, ticks[-1] * 1.5)
    ax.set_ylim(1.325, 1.395)
    ax.set_xlabel("non-embedding parameters (millions)")
    ax.set_ylabel("test bits per byte (subtitles)")
    save(fig, "size.pdf")


def data_plot(runs):
    names = [n for n in runs if n.startswith("data_") or n == "base"]
    pts = sorted((runs[n]["subs_weight_used"], runs[n]["test_bpb_subs"], runs[n]["test_bpb_wiki"],
                  runs[n]["epochs_subs"]) for n in names if runs[n]["config"]["seed"] == 0)
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PALETTE[0], marker=MARKERS[0], label="subtitle test")
    ax.plot([p[0] for p in pts], [p[2] for p in pts], color=PALETTE[1], marker=MARKERS[1], label="Wikipedia test")
    for p in pts:
        if p[3] > 0:  # passes over the subtitle data
            ax.annotate(f"{p[3]:.1f} ep", (p[0], p[1]), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    ax.set_xlabel("share of training sequences from subtitles")
    ax.set_ylabel("test bits per byte")
    ax.set_ylim(1.0, 2.6)
    ax.legend(loc="upper center")
    save(fig, "data.pdf")


def difficulty_plot():
    """Mean model BPB by JLPT level for each scored model, next to mean length."""
    files = sorted(RES.glob("difficulty_*.json"))
    if not files:
        return
    lv = ["N5", "N4", "N3", "N2", "N1"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(2 * W, H))
    for i, p in enumerate(files):
        d = json.loads(p.read_text())
        run = p.stem.removeprefix("difficulty_")
        share = {"base": "16%", "data_subs0.5": "50%", "data_subs1": "100%"}.get(run, run)
        m = d["mean_by_level"]["model_bpb"]
        a.plot(lv, [m[k] for k in lv], color=PALETTE[i], marker=MARKERS[i],
               label=f"{share} subtitles, rho={d['spearman']['model_bpb']:.2f}")
    a.set_xlabel("JLPT level of hardest word (easy to hard)")
    a.set_ylabel("mean bits per byte")
    a.legend(fontsize=7)
    n = d["mean_by_level"]["length_chars"]
    b.plot(lv, [n[k] for k in lv], color=MUTED, marker="o")
    b.set_xlabel("JLPT level of hardest word (easy to hard)")
    b.set_ylabel("mean length (characters)")
    save(fig, "difficulty.pdf")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load()
    if not runs:
        print("no results yet")
        return
    summary(runs)
    learning_curves(runs)
    tokenizer_plot(runs)
    curves(runs, ["tok_byte", "tok_char", "tok_bpe4k", "base"], lambda r: r["config"]["tokenizer"],
           "tokenizer_curves.pdf")
    size_plot(runs)
    curves(runs, ["size_xs", "base", "size_m", "size_l"], lambda r: r["config"]["size"], "size_curves.pdf")
    data_plot(runs)
    seeds = [runs[n]["test_bpb_subs"] for n in ("base", "base_seed1", "base_seed2") if n in runs]
    if len(seeds) > 1:
        print(f"baseline over {len(seeds)} seeds: {statistics.mean(seeds):.4f} "
              f"+/- {statistics.stdev(seeds):.4f} (std) test bpb on subtitles")
    difficulty_plot()
    print(f"wrote {RES / 'summary.csv'} and figures in {FIG}")


if __name__ == "__main__":
    main()
