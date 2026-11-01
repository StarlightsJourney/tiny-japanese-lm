"""Train one model from random weights for a fixed wall-clock budget.

The budget counts only time spent on training steps (data loading, forward,
backward, optimiser). Periodic validation and the final test evaluation are
excluded, so evaluating more often never steals training time.

The learning rate follows warmup + cosine decay over the *time* budget, not a
step count, because under a time budget the number of steps is unknown in
advance (bigger models take fewer, slower steps).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from evaluate import autocast, bits_per_byte
from model import SIZES, build_model
from tokenizers_ja import load_tokenizer

DEFAULTS = dict(
    name="run",
    tokenizer="bpe16k",
    size="s",
    lr=2e-3,
    batch_size=64,
    seq_len=512,
    budget_s=1800,          # 30 minutes of training per run
    subs_weight="natural",  # share of training sequences drawn from subtitles
    seed=0,
    eval_every_s=120,
    warmup_frac=0.02,
    min_lr_frac=0.1,
    grad_clip=1.0,
    weight_decay=0.1,
    val_tokens=200_000,     # per source, for the training curves
    test_tokens=2_000_000,  # per source, cap for the final test score
)


def lr_at(frac: float, cfg: dict) -> float:
    if frac < cfg["warmup_frac"]:
        return cfg["lr"] * frac / cfg["warmup_frac"]
    p = min(1.0, (frac - cfg["warmup_frac"]) / (1 - cfg["warmup_frac"]))
    lo = cfg["min_lr_frac"]
    return cfg["lr"] * (lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * p)))


def load_split(tok_dir: Path, name: str) -> np.ndarray:
    return np.memmap(tok_dir / f"{name}.bin", dtype=np.uint16, mode="r")


def train(cfg: dict, data_dir: str, out_dir: str, device: str | None = None) -> dict:
    cfg = {**DEFAULTS, **cfg}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok_dir = Path(data_dir) / "tok" / cfg["tokenizer"]
    tok = load_tokenizer(tok_dir / "tokenizer.json")
    meta = json.loads((tok_dir / "meta.json").read_text())
    # Training tokens are copied into RAM: random reads from a network volume are slow.
    train_data = {s: np.fromfile(tok_dir / f"{s}_train.bin", dtype=np.uint16) for s in ("subs", "wiki")}

    n_subs, n_wiki = len(train_data["subs"]), len(train_data["wiki"])
    w = cfg["subs_weight"]
    w_subs = n_subs / (n_subs + n_wiki) if w == "natural" else float(w)

    T, B = cfg["seq_len"], cfg["batch_size"]
    model = build_model(cfg["size"], tok.vocab_size, T).to(device)
    opt = model.optimizer(cfg["lr"], cfg["weight_decay"])
    n_params = model.n_params()
    print(f"[{cfg['name']}] {cfg['size']} model, {n_params / 1e6:.2f}M non-embedding params, "
          f"vocab {tok.vocab_size}, subs weight {w_subs:.3f}, device {device}")

    def get_batch():
        from_subs = rng.random(B) < w_subs
        rows = []
        for is_subs in from_subs:
            arr = train_data["subs" if is_subs else "wiki"]
            start = rng.integers(0, len(arr) - T - 1)
            rows.append(arr[start:start + T + 1])
        b = torch.from_numpy(np.stack(rows).astype(np.int64))
        if device == "cuda":
            b = b.pin_memory().to(device, non_blocking=True)
        return b[:, :-1], b[:, 1:], int(from_subs.sum())

    def val_scores():
        return {s: bits_per_byte(model, load_split(tok_dir, f"{s}_val"), tok.byte_lens, T,
                                 device, max_tokens=cfg["val_tokens"]) for s in ("subs", "wiki")}

    train_time, step, seqs_subs, seqs_total = 0.0, 0, 0, 0
    curve, next_eval, loss_ema = [], 0.0, None
    while train_time < cfg["budget_s"]:
        t0 = time.perf_counter()
        lr = lr_at(train_time / cfg["budget_s"], cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, n_sub = get_batch()
        with autocast(device):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        step += 1
        seqs_subs += n_sub
        seqs_total += B
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()

        if train_time >= next_eval or train_time >= cfg["budget_s"]:
            v = val_scores()
            curve.append({"train_s": round(train_time, 1), "step": step,
                          "tokens": seqs_total * T, "train_loss": round(loss_ema, 4),
                          "val_bpb_subs": round(v["subs"], 4), "val_bpb_wiki": round(v["wiki"], 4)})
            print(f"[{cfg['name']}] {train_time:7.0f}s step {step:6d} lr {lr:.2e} "
                  f"loss {loss_ema:.3f} val bpb subs {v['subs']:.4f} wiki {v['wiki']:.4f}", flush=True)
            next_eval += cfg["eval_every_s"]

    tokens = seqs_total * T
    test = {s: bits_per_byte(model, load_split(tok_dir, f"{s}_test"), tok.byte_lens, T,
                             device, max_tokens=cfg["test_tokens"]) for s in ("subs", "wiki")}
    out = Path(out_dir)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "model_config": vars(model.config), "state": model.state_dict()},
               out / "ckpt" / f"{cfg['name']}.pt")

    result = {
        "name": cfg["name"],
        "config": cfg,
        "test_bpb_subs": test["subs"],
        "test_bpb_wiki": test["wiki"],
        "final_val_bpb_subs": curve[-1]["val_bpb_subs"],
        "final_val_bpb_wiki": curve[-1]["val_bpb_wiki"],
        "params_non_embedding": n_params,
        "params_total": model.n_params(non_embedding=False),
        "vocab_size": tok.vocab_size,
        "steps": step,
        "tokens_seen": tokens,
        "tokens_per_param": tokens / n_params,
        "tokens_per_second": tokens / train_time,
        "train_seconds": train_time,
        "subs_weight_used": w_subs,
        "epochs_subs": seqs_subs * T / n_subs,
        "epochs_wiki": (seqs_total - seqs_subs) * T / n_wiki,
        "bytes_per_token_subs_test": meta["splits"]["subs_test"]["bytes_per_token"],
        "context_bytes": T * meta["splits"]["subs_test"]["bytes_per_token"],
        "approx_train_flops": 6 * model.n_params(non_embedding=False) * tokens,
        "curve": curve,
        "device": torch.cuda.get_device_name() if device == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__,
    }
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / f"{cfg['name']}.json").write_text(json.dumps(result, indent=2))
    print(f"[{cfg['name']}] done: test bpb subs {test['subs']:.4f} wiki {test['wiki']:.4f}, "
          f"{tokens / 1e6:.0f}M tokens, {result['tokens_per_param']:.1f} tokens/param")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v) if not isinstance(v, str) or k != "subs_weight" else str,
                       default=v)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default=None)
    a = vars(p.parse_args())
    data_dir, out_dir, device = a.pop("data_dir"), a.pop("out_dir"), a.pop("device")
    assert a["size"] in SIZES
    train(a, data_dir, out_dir, device)
