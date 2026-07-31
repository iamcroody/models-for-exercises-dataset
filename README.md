# models-for-exercises-dataset

Fine-tuning experiments on the [exercises dataset](https://github.com/jayounghoyos/exercises-dataset) — 1324 exercises with structured metadata and instructions in 10 languages.

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

`uv run` checks the lockfile and syncs the environment on every call, so there is no virtualenv to activate:

```bash
uv run python -c "import json; print(len(json.load(open('data/exercises-dataset/data/exercises.json'))))"
# 1324
```

Add a dependency with `uv add <package>` — it updates `pyproject.toml` and `uv.lock` for you.

## Dataset

The dataset lives in `data/exercises-dataset` as a submodule pinned to a fixed commit, so every run trains on exactly the same data.

To move the pin to the latest dataset commit:

```bash
git -C data/exercises-dataset pull origin main
git add data/exercises-dataset
git commit -m "build(data): bump dataset pin"
```

Exercise media is © [Gym visual](https://gymvisual.com/) and is referenced, not redistributed here — see the dataset repo's `NOTICE.md` before reusing it.
