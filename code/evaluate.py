"""Bits per byte: the tokenizer-independent score used for every comparison.

bits per byte = (total loss in nats over the predicted tokens)
                / (UTF-8 bytes those tokens stand for * ln 2)

Lower is better. Unlike loss per token, it does not reward a tokenizer for
simply making tokens longer, because it is measured per byte of the same text.
"""
from __future__ import annotations

import math

import numpy as np
import torch


@torch.no_grad()
def bits_per_byte(model, ids: np.ndarray, byte_lens: np.ndarray, seq_len: int,
                  device: str, batch_size: int = 32, max_tokens: int | None = None) -> float:
    """Scores every token after the first, in chunks of seq_len with 1 token overlap."""
    model.eval()
    if max_tokens is not None:
        ids = ids[:max_tokens]
    ids = torch.from_numpy(ids.astype(np.int64))
    lens = torch.from_numpy(byte_lens.astype(np.int64))
    starts = list(range(0, len(ids) - 1, seq_len))
    full = [s for s in starts if s + seq_len + 1 <= len(ids)]
    tail = [s for s in starts if s + seq_len + 1 > len(ids)]
    nats, nbytes = 0.0, 0

    def score(batch):
        nonlocal nats, nbytes
        x, y = batch[:, :-1].to(device), batch[:, 1:].to(device)
        with autocast(device):
            _, loss = model(x, y, reduction="sum")
        nats += loss.item()
        nbytes += int(lens[batch[:, 1:]].sum())

    for i in range(0, len(full), batch_size):
        score(torch.stack([ids[s:s + seq_len + 1] for s in full[i:i + batch_size]]))
    for s in tail:  # the shorter last chunk is scored on its own
        score(ids[s:].unsqueeze(0))
    model.train()
    return nats / (nbytes * math.log(2))


@torch.no_grad()
def sentence_bits_per_byte(model, tok, sentences: list[str], device: str) -> list[float]:
    """Per-sentence score: a newline is given as context, then each token is predicted."""
    model.eval()
    scores = []
    nl = tok.encode("\n")
    for s in sentences:
        ids = np.concatenate([nl, tok.encode(s)]).astype(np.int64)
        assert len(ids) <= model.config.block_size + 1, "sentence longer than the context"
        x = torch.from_numpy(ids[:-1]).unsqueeze(0).to(device)
        y = torch.from_numpy(ids[1:]).unsqueeze(0).to(device)
        with autocast(device):
            _, loss = model(x, y, reduction="sum")
        nbytes = int(tok.byte_lens[ids[len(nl):]].sum())
        scores.append(loss.item() / (nbytes * math.log(2)))
    return scores


def autocast(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)
