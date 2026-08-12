# models-for-exercises-dataset

This project fine-tunes a small language model to recommend a real exercise you can do with a
household object (like water bottles or a backpack) instead of the gym equipment you don't
have, or tell you honestly when nothing works.

Built on the [exercises dataset](https://github.com/jayounghoyos/exercises-dataset), 1324
exercises with structured metadata and instructions in 10 languages.

## M1 - Fine-tuning baseline (SI4006)

**Task.** "MacGyver Gym Rat": given a target muscle and a household object, name a real
exercise from the catalog that works with that object, explain the adaptation, and recite the
exercise's real steps, or refuse when nothing in the catalog fits. Full dataset description
and known biases are in [`docs/DATASET.md`](docs/DATASET.md).

**Model.** `Qwen/Qwen3-1.7B`, a decoder. Apache-2.0, no login needed, and small enough to fit
a free Colab T4.

**Baseline.** The same model, same prompt, no adapter, scored on the same validation split
used for the fine-tuned model.

**Method.** LoRA, `r=16`, `alpha=32`, `dropout=0.05`, `target_modules=all-linear`. Config and
reasoning are in `scripts/04_train_lora.py`.

**Results**, validation split (n=154):

| method | constraint satisfaction | step grounding (ROUGE-L) | refusal F1 |
|---|---:|---:|---:|
| zero-shot (no adapter) | 0.0% | n/a | 0.0% |
| LoRA | 13.0% | 0.666 (n=33) | 8.3% |

Zero-shot: the model answers in the right format 100% of the time, but always names a generic
exercise ("Plank") that isn't an exact match to this catalog's specific naming, so it never
passes any of the real-exercise checks.

LoRA: constraint satisfaction goes from 0% to 13.0%, and when it does name a real exercise the
steps it recites are mostly grounded in that exercise's real instructions (ROUGE-L 0.666). It
still mostly guesses instead of refusing: refusal precision is 100% (every refusal it gives is
correct) but recall is only 4.3%, so it says "no valid exercise" on almost none of the cases
that actually call for it. Seen vs unseen objects (11.1% vs 14.8%) don't show a real gap, but
with only 54 examples each that's within noise (about 0.6 points per example). Full numbers in
`reports/zeroshot.json` and `reports/finetuned.json`.

**Notebook.** [`notebooks/M1_macgyver.ipynb`](notebooks/M1_macgyver.ipynb) runs the whole
pipeline on a free Colab T4: model/tokenizer load, dataset build, baseline, LoRA training, and
qualitative examples.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Git

## Setup

Clone the repo together with the dataset submodule:

```bash
git clone --recurse-submodules https://github.com/iamcroody/models-for-exercises-dataset.git
cd models-for-exercises-dataset
```

If you already cloned it without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Install uv, if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the dependencies:

```bash
uv sync
```

## Run

`uv run` checks the lockfile and syncs the environment on every call, so there is no virtualenv
to activate:

```bash
uv run python scripts/01_macgyver_data.py     # builds data/processed/macgyver/{train,val,test}.jsonl
uv run python scripts/02_score.py             # self-test: scorer against gold answers
uv run python scripts/03_eval_zeroshot.py     # zero-shot baseline
uv run python scripts/04_train_lora.py        # trains models/r16-all-linear
uv run python scripts/03_eval_zeroshot.py --adapter models/r16-all-linear --report-name finetuned
```

Add a dependency with `uv add <package>`, it updates `pyproject.toml` and `uv.lock` for you.

## Dataset

The dataset lives in `data/exercises-dataset` as a submodule pinned to a fixed commit, so every
run trains on exactly the same data. See [`docs/DATASET.md`](docs/DATASET.md) for source, task,
how the household-object mapping was built, splits, language, licence, and known biases.

To move the pin to the latest dataset commit:

```bash
git -C data/exercises-dataset pull origin main
git add data/exercises-dataset
git commit -m "build(data): bump dataset pin"
```

Exercise media is © [Gym visual](https://gymvisual.com/) and is referenced, not redistributed
here, see the dataset repo's `NOTICE.md` before reusing it.
