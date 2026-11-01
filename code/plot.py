"""Turn results/*.json into the report's figures and a summary table.

  python code/plot.py            -> report/figures/*.pdf and results/summary.csv
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("TJLM_RESULTS", ROOT / "results"))
FIG = Path(os.environ.get("TJLM_FIGURES", ROOT / "report" / "figures"))
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False})


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


def curves(runs, names, label, fname):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for n in names:
        if n in runs:
            c = runs[n]["curve"]
            ax.plot([p["train_s"] / 60 for p in c], [p["val_bpb_subs"] for p in c], label=label(runs[n]))
    ax.set_xlabel("training time (min)")
    ax.set_ylabel("validation bits per byte (subtitles)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def bars(runs, names, label, fname, extra=None):
    names = [n for n in names if n in runs]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    xs = range(len(names))
    ax.bar([x - 0.2 for x in xs], [runs[n]["test_bpb_subs"] for n in names], 0.4, label="subtitles")
    ax.bar([x + 0.2 for x in xs], [runs[n]["test_bpb_wiki"] for n in names], 0.4, label="Wikipedia")
    ax.set_xticks(list(xs), [label(runs[n]) for n in names])
    ax.set_ylabel("test bits per byte (lower is better)")
    ax.legend(frameon=False)
    if extra:
        for x, n in zip(xs, names):
            ax.annotate(extra(runs[n]), (x, 0), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def size_plot(runs):
    names = [n for n in runs if n.startswith("size_") or n == "base"]
    pts = sorted((runs[n]["params_non_embedding"], runs[n]["test_bpb_subs"], runs[n]["tokens_per_param"])
                 for n in names if runs[n]["config"]["subs_weight"] == "natural" and runs[n]["config"]["seed"] == 0)
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "o-")
    for p in pts:
        ax.annotate(f"{p[2]:.0f} tok/param", (p[0] / 1e6, p[1]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters (millions)")
    ax.set_ylabel("test bits per byte (subtitles)")
    fig.tight_layout()
    fig.savefig(FIG / "size.pdf")
    plt.close(fig)


def data_plot(runs):
    names = [n for n in runs if n.startswith("data_") or n == "base"]
    pts = sorted((runs[n]["subs_weight_used"], runs[n]["test_bpb_subs"], runs[n]["test_bpb_wiki"],
                  runs[n]["epochs_subs"]) for n in names if runs[n]["config"]["seed"] == 0)
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label="subtitle test")
    ax.plot([p[0] for p in pts], [p[2] for p in pts], "s--", label="Wikipedia test")
    for p in pts:
        ax.annotate(f"{p[3]:.1f} ep", (p[0], p[1]), xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.set_xlabel("share of training drawn from subtitles")
    ax.set_ylabel("test bits per byte")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "data.pdf")
    plt.close(fig)


def difficulty_plot():
    for p in RES.glob("difficulty_*.json"):
        d = json.loads(p.read_text())
        m = d["mean_by_level"]["model_bpb"]
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        lv = ["N5", "N4", "N3", "N2", "N1"]
        ax.plot(lv, [m[k] for k in lv], "o-")
        ax.set_xlabel("JLPT level of hardest word (easy to hard)")
        ax.set_ylabel("mean bits per byte")
        rho = d["spearman"]["model_bpb"]
        ax.set_title(f"Spearman rho = {rho:.2f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / f"{p.stem}.pdf")
        plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load()
    if not runs:
        print("no results yet")
        return
    summary(runs)
    tok_names = ["tok_byte", "tok_char", "tok_bpe4k", "base"]
    bars(runs, tok_names, lambda r: r["config"]["tokenizer"], "tokenizer.pdf",
         extra=lambda r: f"{r['tokens_per_second'] / 1e3:.0f}k tok/s")
    curves(runs, tok_names, lambda r: r["config"]["tokenizer"], "tokenizer_curves.pdf")
    size_plot(runs)
    curves(runs, ["size_xs", "base", "size_s", "size_m", "size_l"], lambda r: r["config"]["size"],
           "size_curves.pdf")
    data_plot(runs)
    seeds = [runs[n]["test_bpb_subs"] for n in ("base", "base_seed1", "base_seed2") if n in runs]
    if len(seeds) > 1:
        import statistics

        print(f"baseline over {len(seeds)} seeds: {statistics.mean(seeds):.4f} "
              f"+/- {statistics.stdev(seeds):.4f} (std) test bpb on subtitles")
    difficulty_plot()
    print(f"wrote {RES / 'summary.csv'} and figures in {FIG}")


if __name__ == "__main__":
    main()
