# Product / code brief: tiny-japanese-lm

## One line

A Japanese language model small enough to ship inside a desktop app, trained
from scratch on 30 GPU-minutes, whose job is to say how surprising (and so how
hard) a sentence is for a learner.

## The problem

Sheng (声), the author's offline-first app for learning Japanese from real
media, shows live subtitles of whatever the user is watching and lets them save
sentences as flashcards. Krashen's i+1 idea says learners progress fastest on
input just above their level, so Sheng would like to know, for every subtitle
line, roughly how hard it is: highlight lines worth saving, sort a card
collection from easy to hard, or warn that a show is far above the user's level.

Constraints from the app:

| Constraint | Consequence for the model |
| --- | --- |
| Offline, runs on the user's laptop | Small (3-60M parameters), CPU-friendly, no API calls |
| Scores every subtitle line as it appears | Milliseconds per sentence |
| Input is spoken, colloquial Japanese | Train and evaluate mainly on subtitles, not just Wikipedia |
| Open-source friendly | Open data, no pretrained weights we cannot redistribute |

## What the model does

Input: one Japanese sentence (a subtitle line).
Output: a score in **bits per byte** (BPB), the model's average surprise per
byte of UTF-8 text. Lower means more predictable.

Hypothesis to test: sentences with rarer words and more complex structure are
more surprising, so BPB rises with JLPT level (N5 easiest, N1 hardest). If this
holds better than cheap baselines (sentence length, share of kanji), the model
is useful to Sheng; if not, that is a useful negative result and Sheng keeps a
simple word-list heuristic.

## What the course project answers

Course rules (see the readme): train from random weights, declare a fixed
per-run budget up front, compare choices on the same held-out test text.

Declared budget: **1,800 s of training-step time per run, one NVIDIA L4 on Modal.**

| # | Question | What varies | Readout |
| --- | --- | --- | --- |
| 1 | Which tokenizer for Japanese? | bytes, characters, BPE 4k, BPE 16k | test BPB, tokens/s, context in bytes |
| 2 | Big-and-short or small-and-long? | 3M / 11M / 25M / 59M non-embedding params | test BPB vs size; tokens per parameter vs Chinchilla's ~20 |
| 3 | How much to repeat scarce subtitle data? | share of subtitles 0 / natural / 0.5 / 1.0 | subtitle and Wikipedia BPB; epochs of subtitles |
| 4 | Does surprise track learner difficulty? | best model, fixed | Spearman rho with JLPT level, 95% bootstrap CI, vs baselines |
| - | How noisy is one run? | 3 seeds of the baseline | std of test BPB, used to judge every difference above |

## How it is built (code map)

```
data.py           download JESC subtitles + Japanese Wikipedia, clean, dedupe, hash-split 99/0.5/0.5
tokenizers_ja.py  4 lossless tokenizers, each knows bytes-per-token -> exact BPB
model.py          nanoGPT-style decoder (pre-norm, GELU, tied embeddings), 5 sizes
train.py          time-budgeted loop; LR warmup + cosine over elapsed *time*; bf16
evaluate.py       corpus BPB and per-sentence BPB
difficulty.py     JLPT-labelled Tatoeba set (fugashi + open word lists), Spearman + bootstrap
configs.py        every run in the report, in one place
modal_app.py      runs data prep and experiments on Modal GPUs, in parallel
plot.py           figures + results/summary.csv
smoke_test.py     2-minute CPU check of the whole pipeline (passes)
```

Design choices worth defending in the report:

- **BPB, not loss per token**, so tokenizers with different vocabularies are
  compared on the same text.
- **Time budget measured on training steps only**: evaluating more often never
  steals training time, and the LR schedule follows elapsed time because the
  number of steps is unknown in advance.
- **Whole Wikipedia articles and unique subtitle lines go to one split**, so
  test text is never seen in training.
- **Every run writes one JSON** (config, curve, scores) so the report can be
  checked against the repository.

## Compute plan and cost

About US$9 of Modal time for everything (see the readme table), inside one
month of the ~US$30 free credit. Set Modal's spend limit to $0 before the first
run so nothing can be charged beyond the credit.

## Optional extension: OpenRouter

The JLPT labels come from word lists and ignore grammar. An OpenRouter key could
add a second, independent difficulty label: ask a large hosted LLM to rate each
test sentence N5-N1, then report how well our model's BPB agrees with both label
sets. This touches evaluation only; no hosted model is used in training, so the
from-scratch rule still holds. The key would live in a local `.env` (already
git-ignored), never in the repository. Not built yet.

## Integration path into Sheng (after the course)

1. Export the best checkpoint (expected: ~10-25M params) to ONNX or a
   quantised format; ship it with Sheng's tokenizer file.
2. Score each finished subtitle line in Sheng's background worker; cache by text.
3. Map BPB to a coarse level using the Experiment 4 calibration (mean BPB per
   JLPT level).
4. Use it to rank saved cards and flag lines near the user's level.

Success for Sheng: rho with JLPT level clearly above the length and kanji
baselines, and under ~10 ms per line on a laptop CPU.

## Out of scope

Text generation quality, chat, translation, and any pretrained or fine-tuned
model. The model is a scorer, not an assistant.
