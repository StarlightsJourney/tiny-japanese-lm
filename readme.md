# tiny-japanese-lm

How good can a Japanese language model get if you train it from scratch for
30 minutes on one GPU?

This repository trains small GPT-style models (3M to 59M parameters) from
random weights on Japanese film subtitles and Wikipedia. Every run gets the
same budget: 30 minutes of training on one NVIDIA L4 GPU. Within that budget
we ask four questions:

1. **Tokenizer.** Should the model read bytes, characters, or learned word
   pieces (BPE with 4k or 16k tokens)?
2. **Size.** Is it better to train a big model briefly or a small one for
   longer? (Hoffmann et al., 2022)
3. **Data.** Subtitles are scarce and Wikipedia is plentiful. How often is it
   worth repeating the subtitles? (Muennighoff et al., 2023)
4. **Difficulty.** Is the model more surprised by sentences that are harder
   for learners, as measured by JLPT level?

Every score is bits per byte on the same held-out text. Loss per token would
favour tokenizers with long tokens; bits per byte doesn't.

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

The full write-up is [report/report.pdf](report/report.pdf). In short, with 30
minutes on one L4:

| Question | Answer | Test bits per byte, subtitles |
| --- | --- | --- |
| Tokenizer | BPE-16k beats BPE-4k, characters and bytes | 1.334 vs 1.344, 1.362, 1.430 |
| Size | 10.6M and 25M tie; 3M and 59M are worse | 1.334, 1.336 vs 1.358, 1.386 |
| Data mix | More subtitles always helped, even repeated 11 times | 1.270 (all subtitles) vs 1.718 (none) |
| Difficulty | Surprise adds nothing measurable to sentence length | – |

Run-to-run noise is small: the baseline varies by 0.0015 across three seeds.
The best model trained on 24 tokens per parameter, close to the ~20 that
Hoffmann et al. found best for models a thousand times larger.

The difficulty result is negative. Per byte, short sentences look surprising
whatever their content, and easy sentences are short, so the two effects
cancel. A score built from sentence length and the share of kanji tracks JLPT
level better (Spearman 0.36) than anything that uses the model.

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
| 5 | `modal run code/modal_app.py --exp tokenizer` | Question 1, 4 runs in parallel | ~40 min | ~$2 |
| 6 | Set `BEST_TOKENIZER` in `code/configs.py` if BPE 16k didn't win | | | |
| 7 | `modal run code/modal_app.py --exp size` | Question 2 | ~40 min | ~$1.50 |
| 8 | `modal run code/modal_app.py --exp data` | Question 3 | ~40 min | ~$1.50 |
| 9 | `modal run code/modal_app.py --exp seeds` | Baseline with 2 more seeds, to measure noise | ~40 min | ~$1 |
| 10 | `modal run code/modal_app.py --exp difficulty --run base`, then the same with `--run data_subs0.5` and `--run data_subs1` | Question 4, for three data mixes | ~5 min each | ~$0.05 each |
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
results/            one JSON per run
report/             report source (LaTeX)
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
