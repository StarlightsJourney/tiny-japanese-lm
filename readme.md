# tiny-japanese-lm

How good can a Japanese language model get if you train it from scratch for
30 minutes on one GPU?

This repository trains small GPT-style models (3M to 59M parameters, not
counting embeddings) from random weights on Japanese film subtitles and
Wikipedia. Every run gets the same budget: 30 minutes of training on one NVIDIA
L4 GPU. Within that budget I ask four questions:

1. **Tokenizer.** Should the model read bytes, characters, or learned word
   pieces (BPE with 4k or 16k tokens)?
2. **Size.** Is it better to train a big model briefly or a small one for
   longer? (Hoffmann et al., 2022)
3. **Data.** Subtitles are scarce and Wikipedia is plentiful. How often is it
   worth repeating the subtitles? (Muennighoff et al., 2023)
4. **Difficulty.** Is the model more surprised by sentences that are harder
   for learners, as measured by JLPT level?

The report is [report/report.pdf](report/report.pdf). This readme summarises
it and explains how to reproduce every number in it.

## Declared setup

| | |
| --- | --- |
| **Compute budget** | 30 minutes (1,800 s) of training per run on one NVIDIA L4 GPU (24 GB), rented from Modal. The clock runs only while the model is learning: reading a batch, measuring its errors and updating its weights. Scoring held-out text, data preparation, tokenizer training and the preliminary learning-rate search are not counted. |
| **Initialisation** | Every model starts from random weights. No pretrained weights or embeddings. Tokenizers are trained on this project's own training data. |
| **Evaluation objective** | Bits per byte (BPB) on held-out film subtitles (primary) and Wikipedia (secondary); lower is better. BPB is the model's total surprise in bits divided by the length of the text in bytes, so models with different tokenizers are compared on equal terms. Choices are made on a validation set; final numbers come from a separate test set. Every model is scored on the same subtitle test text. |

## Why

I'm also building Sheng (声), an offline app for learning Japanese from films
and TV. It shows live subtitles of whatever you're watching and lets you save
sentences as flashcards. It would be more useful if it knew how hard each
sentence is, so it could point out lines near your level.

That has to work offline, on a laptop, in milliseconds. A model this small
could do it, if its surprise at a sentence tracks how hard the sentence is.
Question 4 tests exactly that. Sheng is a separate, private project; nothing
here depends on it. [docs/brief.md](docs/brief.md) explains the plan in more
detail.

## Results

| Question | Finding | Test BPB, subtitles |
| --- | --- | --- |
| Tokenizer | Learned word pieces beat characters, and characters beat bytes. BPE-16k is best. | bpe16k 1.334, bpe4k 1.344, char 1.362, byte 1.430 |
| Size | The best size is a flat region from 10M to 25M parameters. 3M and 59M are clearly worse. | 3M 1.358, 10.6M 1.334, 25M 1.336, 59M 1.386 |
| Data mix | For subtitle text, subtitles in training matter far more than Wikipedia. Repeating them 11 times still helped slightly. | 0% 1.718, 16% 1.334, 50% 1.279, 100% 1.270 |
| Difficulty | Model surprise adds nothing measurable to sentence length. Length plus kanji share is the better difficulty score. | – |

- **Noise.** Three seeds of the baseline vary by 0.0015 BPB (standard
  deviation). Gaps under about 0.005 on validation are treated as ties, so the
  10.6M and 25M models tie.
- **Size.** The best model read 24 tokens per parameter, or 15 if the
  embedding table is counted. That is consistent with the ~20 of Hoffmann et
  al. (2022), though one budget and four sizes are not a test of their rule.
- **Time is not compute.** Bigger models use the GPU better, so in 30 minutes
  the largest did twice the computation of the smallest. Compared at equal
  computation instead, the 10.6M model clearly beats the 25M one. A time
  budget favours bigger models somewhat.
- **Repetition.** Each extra share of subtitles helps less: 0.38, then 0.055,
  then 0.008. The subtitles-only run read the same text 10.8 times. Its gap
  between validation and training loss nearly tripled, a sign it had begun to
  memorise, though validation was still improving. It also more than doubled its
  Wikipedia BPB. The 50% mix keeps most of the gain without that cost.
- **Difficulty.** Model surprise barely tracks JLPT level (Spearman 0.05 to
  0.12), while sentence length alone reaches 0.27. Easy sentences are short,
  and short sentences score high per byte, likely because the start of any
  sentence is hard to predict. With length held fixed, a weak link remains
  (0.13 to 0.17). But adding the model to a length-based score does not
  measurably improve predictions on held-out sentences, while adding the share
  of kanji raises Spearman from 0.26 to 0.36.

For Sheng, this means difficulty should come from sentence length and kanji
share, not from the model.

## Setup

You need Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git config core.hooksPath .githooks
python code/smoke_test.py
```

The last command trains tiny models on fake data on your CPU in about two
minutes. It should end with `SMOKE TEST PASSED`. The `git config` line turns
on the repository's hooks, which block secrets and AI attribution in commits.

## Keys

There are two, and neither goes in a tracked file.

**Modal** runs the GPU jobs. Its key is created and stored for you:

```bash
modal setup
```

This opens a browser, logs you in, and writes the token to `~/.modal.toml`
in your home folder, outside this repository. You never paste it anywhere.
On a machine without a browser, create a token at
https://modal.com/settings/tokens and set `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` in your shell instead.

**OpenRouter** is optional and nothing uses it yet. It's for a planned
second set of difficulty labels (see the brief). Put it in `.env`:

```bash
cp .env.example .env
# then edit .env:  OPENROUTER_API_KEY=sk-or-v1-...
```

`.env` is git-ignored, and the pre-commit hook refuses to commit it or
anything that looks like a key.

Modal needs a card on file for its free credit (about US$30 a month). Enter
it only on modal.com. Before running anything, go to
https://modal.com/settings/usage and set the spend limit to $0. Then a job
that would cost real money stops instead.

## Running the experiments

Run these from the repository root, in order. Each waits for its runs to
finish and writes results to `results/`. Costs assume about US$0.80 per L4
hour, including start-up and evaluation.

| Step | Command | What it does | Time | Cost |
| --- | --- | --- | --- | --- |
| 1 | `modal run code/modal_app.py --exp prepare` | Download, split and tokenize data; build the JLPT set | ~1 h | ~$1.50 |
| 2 | `modal run code/modal_app.py --exp smoke` | 3-minute GPU check | 5 min | $0.05 |
| 3 | `modal run code/modal_app.py --exp lr` | Learning-rate sweep, 12 runs of 6 min (preliminary) | ~45 min | ~$1.20 |
| 4 | Set `LR_BY_SIZE` in `code/configs.py` to the best rate per size, then run step 3 again: it adds only the runs that are new, so each best rate ends up with a worse rate on both sides | | ~15 min | ~$0.40 |
| 5 | `modal run code/modal_app.py --exp tokenizer` | Question 1, 4 runs | ~1 h | ~$2 |
| 6 | Set `BEST_TOKENIZER` in `code/configs.py` if BPE-16k didn't win on validation (it did) | | | |
| 7 | `modal run code/modal_app.py --exp size` | Question 2, 3 new runs | ~1 h | ~$1.50 |
| 8 | `modal run code/modal_app.py --exp data` | Question 3, 3 new runs | ~1 h | ~$1.50 |
| 9 | `modal run code/modal_app.py --exp seeds` | Baseline with 2 more seeds, to measure noise | ~45 min | ~$1 |
| 10 | `modal run code/modal_app.py --exp difficulty --run base`, then the same with `--run data_subs0.5` and `--run data_subs1` | Question 4, for three data mixes | up to 30 min each, mostly waiting for a GPU | under $0.01 each |
| 11 | `python code/plot.py` | Figures in `report/figures/`, table in `results/summary.csv` | 1 min | free |
| 12 | `python code/analysis.py` | Follow-ups: length plus surprise, equal compute, memorisation (`results/analysis.json`) | 1 min | free |
| 13 | `modal volume get tiny-japanese-lm data/tok/<t>/tokenizer.json data/tok/<t>/tokenizer.json` for `<t>` in char, bpe4k, bpe16k, then `python code/tokenizer_example.py` | How each tokenizer splits one sentence (report Table 2) | 1 min | free |
| 14 | `cd report && tectonic -X compile report.tex` | Builds `report/report.pdf` with XeLaTeX (needed for the Japanese text and the Atkinson Hyperlegible font). On Overleaf, set the compiler to XeLaTeX. | 1 min | free |

The whole project cost US$8.16 on Modal's billing report, all of it free
credit. Times are wall-clock and include waiting: a new Modal account runs only
a few GPUs at once, so the runs in one step queue behind each other.

A run that already has a file in `results/` is skipped; add `--force` to
rerun it. If a GPU is preempted, run the same command again.

## The budget

- **Hardware:** one NVIDIA L4 (24 GB) on Modal. Set by `GPU` in
  `code/modal_app.py`.
- **Budget:** 1,800 seconds of training per run (`BUDGET_S` in
  `code/configs.py`). Only training steps count: loading a batch, forward,
  backward, optimiser. Validation and testing don't, so evaluating more often
  never costs training time.
- **Schedule:** the learning rate warms up and then decays along elapsed
  time, not step count, because you don't know in advance how many steps a
  model will get in 30 minutes.
- **Not counted:** data preparation, tokenizer training and the 6-minute
  learning-rate sweep. The course rules allow this.
- **Learning rates:** chosen per size by the sweep, on validation BPB:
  3e-3 (xs), 1e-3 (s), 7.5e-4 (m), 5e-4 (l). Each has a worse rate on both
  sides. The full table is in the report's appendix.

## Data

- **Subtitles:** the Japanese side of JESC (Pryzant et al., 2018). Duplicate
  lines removed. A hash of each line sends it to train, validation or test
  (99 / 0.5 / 0.5), so the split is the same every time.
- **Wikipedia:** the first ~600 MB of the 2023-11-01 Japanese dump. Whole
  articles go to one split, so no test article is seen in training.
- **Difficulty set:** Japanese sentences from Tatoeba, each labelled with the
  JLPT level of its hardest word. If Tatoeba can't be reached, held-out
  subtitle lines are used instead, and the result file says so.

## Layout

```
code/
  data.py           download, clean, split and tokenize
  tokenizers_ja.py  byte, character and BPE tokenizers, all lossless
  model.py          small GPT (architecture from nanoGPT, random init)
  train.py          time-budgeted training
  evaluate.py       bits per byte, for a corpus or a sentence
  difficulty.py     JLPT-labelled sentences and the correlation analysis
  configs.py        every run in the report
  modal_app.py      runs it all on Modal
  plot.py           figures and summary table
  analysis.py       follow-up analyses on saved results
  tokenizer_example.py  how each tokenizer splits one sentence
  smoke_test.py     2-minute CPU check
data/jlpt/          JLPT word lists (MIT)
docs/brief.md       what the model is for and how it's built
results/            one JSON per run, summary.csv, analysis.json
report/             report.pdf, its LaTeX source, bibliography and figures
.githooks/          blocks secrets and AI attribution in commits
AGENTS.md           rules for AI coding agents working here
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: run the smoke test, commit
under your own name, and don't put AI attribution in commits or PRs.

## Licence and credits

MIT, see [LICENSE](LICENSE). The model architecture is adapted from
[nanoGPT](https://github.com/karpathy/nanoGPT) (MIT). The JLPT word lists are
from [open-anki-jlpt-decks](https://github.com/jamsinclair/open-anki-jlpt-decks)
(MIT). The datasets are downloaded, not redistributed; each keeps its own
licence. No pretrained weights or embeddings are used anywhere.
