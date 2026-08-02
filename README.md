# models-for-exercises-dataset

**This project reads the written description of a gym exercise and automatically fills in
its catalog entry — which muscle it trains and what equipment it needs — so a fitness app
can search, filter and substitute exercises without a human tagging each one by hand.**

It is the M1 deliverable for SI4006: a small open language model
([Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)) fine-tuned with LoRA on 1081
exercises, measured honestly against two baselines on a held-out validation set.

| | target muscle | equipment | both correct |
|---|---|---|---|
| Best rule without a model | 18.2% | 87.6% | 18.2% |
| Same model, **no** fine-tuning | 32.2% | 89.3% | 30.6% |
| **After LoRA fine-tuning** | **85.1%** | **99.2%** | **84.3%** |

The whole pipeline is one notebook:
**[`notebooks/M1_finetuning_qwen3.ipynb`](notebooks/M1_finetuning_qwen3.ipynb)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamcroody/models-for-exercises-dataset/blob/main/notebooks/M1_finetuning_qwen3.ipynb)

It is built for the free Colab T4 and sized for it — 6.7 GB peak against 15 GB available,
and precision is selected from the device at runtime because the T4 has no bf16 units
(details under [LoRA configuration](#lora-configuration)). The committed outputs are from a
local RTX 5060 Ti run; expect roughly 25–30 minutes for training on a T4 rather than 8.

---

## The task

Given an exercise's name and English instructions, produce its catalog fields as JSON:

```
Name: cable incline pushdown
Instructions: Attach a straight bar to a high pulley cable machine. Stand facing
away from the machine with your feet shoulder-width apart. [...]
```
```json
{"target": "lats", "equipment": "cable"}
```

`target` is one of 19 muscles, `equipment` one of 28 types. A third catalog field,
`body_part`, is **derived rather than predicted**: `target → body_part` is a strict
function across all 1324 records (19 targets → 10 body parts), and `category` is a
verbatim copy of it. Generating it would add an accuracy column that is free whenever
`target` is right, which flatters the headline number while measuring nothing.

Full dataset description, licence and known biases: **[`docs/DATASET.md`](docs/DATASET.md)**.

## Base model, and why

**Family: decoder.** Not because it is strongest on this task in isolation — a
DeBERTa-v3-style encoder classifier would very likely beat it on a closed 19-class
problem. It is because M2 mounts RAG on this model and M3 adds a visual component, and an
encoder classifier can host neither. The marginally weaker architecture that survives two
more modules is the cheaper choice.

**Model: `Qwen/Qwen3-1.7B`.**

- **Apache-2.0 and ungated.** No Hugging Face login or licence click-through. Llama-3.2 is
  gated, which alone would break the requirement that anyone can open the notebook and run it.
- **Fits the free tier with headroom** — 6.7 GB peak against a T4's 15 GB, 1.0% of
  parameters trainable under LoRA.
- **Has a forward path.** `Qwen3-VL-2B` shares this tokenizer and chat template, so M3's
  visual component is a family swap rather than a rewrite — and the dataset already ships a
  thumbnail and animation GIF for all 1324 exercises.

### Tokenization check

Before committing to a base model we measured how each candidate splits *our* vocabulary
(`scripts/00_tokenizer_check.py`). That matters more than usual here because the labels are
the **output**: the model must reproduce all 19 target and 28 equipment values verbatim.

Tokens per word — lower is better:

| Tokenizer | Family | `target` | `equipment` | names |
|---|---|---|---|---|
| **Qwen3-1.7B** | decoder | 2.00 | 1.54 | 1.43 |
| SmolLM2-1.7B | decoder | 2.00 | 1.50 | 1.49 |
| gpt2 | decoder | 2.04 | 1.59 | 1.48 |
| bert-base-uncased | encoder | **1.91** | **1.26** | **1.32** |
| flan-t5-base | encoder-decoder | 2.74 | 1.63 | 1.64 |

Read honestly: **BERT's lowercase WordPiece fragments our anatomy vocabulary the least**
and Qwen3 does not win outright. So this study did not pick the model — the family
requirement did. What it establishes is that Qwen3 costs nothing *within* its family, and
it rules out the encoder-decoder option on a measurement rather than on taste. The worst
cases for Qwen3 are `levator scapulae` (5 tokens) and `olympic barbell` (5).

## Baselines, and why these

A single number says nothing, so there are two, measuring different things.

**Rule-based** (`scripts/01_baseline.py`) — majority class, plus a substring rule that
looks for the label verbatim in the exercise name. This measures **the dataset**: how much
of the task needs no model at all. Both are fitted on train and scored on val, exactly like
the model — a baseline that has seen the evaluation set is not a baseline.

**Zero-shot** (`scripts/03_eval_zeroshot.py`) — the same Qwen3-1.7B with no adapter, same
prompts, same split, same greedy decoding. This measures **the model**, and is the only
comparison that isolates what LoRA contributed. It is by far the harder bar: it beats the
rule-based baseline on both fields before any training.

## Results

Validation split, n=121, never trained on. Every row produced by the same scoring function
(`exlib.score_predictions`).

| | target acc | target macro-F1 | equip acc | equip macro-F1 | joint | valid JSON | in-label (target) |
|---|---|---|---|---|---|---|---|
| rule: majority | 13.2% | 0.016 | 30.6% | 0.031 | 9.1% | — | — |
| rule: substring | 18.2% | 0.064 | 87.6% | 0.783 | 18.2% | — | — |
| zero-shot, no LoRA | 32.2% | 0.282 | 89.3% | 0.900 | 30.6% | 100.0% | 70.2% |
| **fine-tuned (LoRA)** | **85.1%** | **0.814** | **99.2%** | **0.924** | **84.3%** | 100.0% | **100.0%** |

**Primary metric is `target` accuracy** — 19 classes, the weakest baseline, the most room.
`macro-F1` sits beside it because accuracy hides the tail and the tail is most of the label
space (`levator scapulae` has 2 records in the entire catalog); macro-F1 weights a rare
class the same as `abs`. `valid JSON` and `in-label` are reported rather than repaired,
because a model that answers 60% of the time must not outscore one that always answers.

### Honest reading

**It improved, a lot, and for exactly the reason the zero-shot analysis predicted.**
`target` accuracy went 32.2% → 85.1%, a +52.9 point gain over the untrained model and
+66.9 over the best rule.

But the interesting number is `in-label`, which went **70.2% → 100.0%**. Before training,
roughly a third of the base model's `target` answers were not values we offered at all — it
echoed the exercise name back (`"target": "dumbbell iron cross"`) or landed one letter off
a real label (`"pectors"` for `"pectorals"`). It already read the domain; what it could not
do was stay inside a closed vocabulary. **LoRA bought vocabulary discipline, not fitness
knowledge.** That reframing is the actual finding here.

**Where it still fails.** Both representative failures are label-convention disputes rather
than comprehension failures:

| Exercise | Catalog says | Model says |
|---|---|---|
| dumbbell incline one arm hammer press | `triceps` | `delts` |
| semi squat jump (male) | `cardiovascular system` | `quads` |

An incline hammer press plausibly targets delts; a squat jump plausibly targets quads. The
model is not confused about the exercise — it disagrees with the catalog's convention. Since
every label comes from a single upstream source with no second annotator, our ceiling is
*agreement with this catalog*, not correctness about exercise physiology. That is roughly
where 85% sits.

**Two caveats against reading these numbers too warmly.** `equipment` at 99.2% is a smaller
achievement than it looks — a substring rule already gets 87.6%, because the equipment name
appears verbatim in the exercise name 60% of the time. And about 1.7% of validation records
share a name and labels with a train record under different wording, a mild upward bias
documented in [`docs/DATASET.md`](docs/DATASET.md).

The **test split (122 records) has never been touched**, deliberately, so M2's rigorous
evaluation is not run on a set already used to make decisions here.

## LoRA configuration

| Hyperparameter | Value | Why |
|---|---|---|
| `r` | 16 | Chosen by measurement, not by prior — see the ablation below, where our initial reasoning turned out to be wrong. |
| `lora_alpha` | 32 | Holds the conventional `alpha = 2r`, keeping the effective scale `alpha/r` at 2.0 so changing `r` does not silently change update magnitude too. |
| `target_modules` | `q,k,v,o,gate,up,down_proj` | All seven linear projections, attention **and** MLP. Our reason for including the MLP did not survive contact with the ablation — see below. |
| `lora_dropout` | 0.05 | Light regularisation on a small dataset. |
| `learning_rate` | 1e-4 | TRL's documented adapter rate, ~5× a full fine-tune's, since only the freshly-initialised low-rank matrices learn. |
| epochs / batch | 3 / 16 effective | Eval loss 0.0403 → 0.0300 → 0.0279, still falling at epoch 3 — not yet overfitting. |

17.4M trainable parameters (**1.0%** of 1.74B). 7m51s and 6.7 GB VRAM on an RTX 5060 Ti.

Batch size is fixed rather than scaled to the GPU on purpose, so a local run and the Colab
run take identical optimisation steps. Precision is chosen from the device
(`exlib.pick_device_dtype`): TRL's `SFTConfig` defaults `bf16=True` whenever `fp16` is
unset, and Colab's T4 is Turing with no bf16 units — a hard-coded config crashes on exactly
the GPU this is required to run on. FlashAttention-2 is left off for the same reason
(Ampere+ only).

### Ablation — and where our reasoning was wrong

Three runs, identical apart from the variable under test, `alpha = 2r` throughout so
changing `r` does not also change the update scale. Same val split, same scoring.

| Config | Trainable | target acc | target macro-F1 | equip acc | joint | eval loss | time |
|---|---|---|---|---|---|---|---|
| r=8, all linear | 8.7M | 77.7% | 0.683 | 99.2% | 76.9% | 0.0319 | 485s |
| **r=16, all linear** | 17.4M | **85.1%** | **0.814** | 99.2% | **84.3%** | **0.0279** | 470s |
| r=16, attention only | 6.4M | 77.7% | 0.679 | 98.4% | 76.0% | 0.0348 | 404s |

We went in with two predictions. **Both were wrong, and the useful part of this experiment
is how.**

**Prediction 1: `r=8` would be enough, and `r=16` risked overfitting.** The reasoning was
that 1081 examples over a closed label space is a narrow target needing little capacity.
Wrong — `r=8` *underfits*, losing 7.4 points of target accuracy and 0.13 macro-F1. Nothing
in any run showed overfitting: eval loss was still falling at epoch 3 in all three.

**Prediction 2: the MLP blocks were doing the work,** because mapping "cable incline
pushdown" → `{lats, cable}` is lexical-semantic. The attention-only run does lose 7.4
points, which looks like confirmation — until you notice it has **fewer** trainable
parameters (6.4M) than the r=8 all-linear run (8.7M), and the two score *identically* at
77.7%. Placement is confounded with capacity, and once you line the budgets up the
placement effect disappears.

**What the data actually supports:** at this scale, the total number of adapter parameters
drove the result, not where they were placed. Two configurations near 6–9M land on 77.7%
whichever modules they adapt; the 17.4M configuration reaches 85.1%. Ranking by eval loss
agrees (0.0279 < 0.0319 < 0.0348).

So `r=16` on all linear layers is kept because it measurably won, not because our story
about MLPs was right. Testing the claim was worth more than the claim.

**What this ablation cannot tell us:** a properly controlled placement test needs
parameter-matched configurations (roughly `r=16` attention-only against `r=6` all-linear),
which we did not run. And with n=121, one record is 0.8 points — the 7.4-point gaps are
real, but the 0.004 macro-F1 difference between the two 77.7% runs is noise.


## Reproducing

Requires [uv](https://docs.astral.sh/uv/) and git. Clone with the dataset submodule:

```bash
git clone --recurse-submodules https://github.com/iamcroody/models-for-exercises-dataset.git
cd models-for-exercises-dataset
uv sync
```

Already cloned without submodules? `git submodule update --init --recursive`

```bash
uv run python scripts/00_tokenizer_check.py    # tokenizer study
uv run python scripts/02_prepare_data.py       # build splits (seed 42)
uv run python scripts/01_baseline.py           # rule-based baseline on val
uv run python scripts/03_eval_zeroshot.py      # zero-shot baseline
uv run python scripts/04_train_lora.py         # ~8 min on a 16 GB GPU
uv run python scripts/05_eval_finetuned.py     # comparison table
```

The trained adapter is committed (`models/r16-all-linear`, 34 MB), so
`05_eval_finetuned.py` reproduces the results table exactly, without retraining.

**Retraining will not reproduce it exactly.** Everything seedable is seeded (`seed = 42`
for data, shuffling and init), but CUDA matmul and gradient-checkpointing kernels are not
bitwise deterministic, so two runs of an identical config land on slightly different
weights. Re-running the notebook end to end gave **84.3%** target accuracy where the
committed run gave 85.1% — one validation record out of 121. Numbers in this README come
from the committed adapter; treat differences of a point as run-to-run noise, which is also
why the ablation only claims the 7.4-point gaps and explicitly discounts the 0.004
macro-F1 one.

`data/processed/` is gitignored — it is fully derived from the pinned submodule by a seeded
script, so it is regenerated rather than committed.

## Layout

```
notebooks/M1_finetuning_qwen3.ipynb   Colab-runnable, end to end
scripts/
  exlib.py                            shared: prompt, parsing, metrics, device
  00_tokenizer_check.py               tokenizer fertility study
  01_baseline.py                      rule-based baseline
  02_prepare_data.py                  stratified splits, seed 42
  03_eval_zeroshot.py                 base model, no adapter
  04_train_lora.py                    LoRA fine-tuning
  05_eval_finetuned.py                evaluation + comparison table
docs/DATASET.md                       dataset description, licence, biases
docs/documentationLinks.md            reference links
models/r16-all-linear/                trained adapter (fp16)
reports/*.json                        every number in this README
data/exercises-dataset/               pinned submodule
```

`exlib.build_prompt` is the single definition of the prompt. Training, the zero-shot
baseline and the fine-tuned evaluation all call it, so the reported delta cannot be
contaminated by prompt drift between the three.

## Licence

Code and tooling: MIT. Exercise metadata and instructions: MIT, from the
[upstream dataset](https://github.com/jayounghoyos/exercises-dataset). Exercise media
(images, GIFs) is © [Gym visual](https://gymvisual.com/) — referenced, not redistributed
here. See the dataset repo's `NOTICE.md` before reusing it.
