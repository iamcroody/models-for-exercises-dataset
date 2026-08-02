# Dataset description

## Source

[`jayounghoyos/exercises-dataset`](https://github.com/jayounghoyos/exercises-dataset), a
public catalog of gym and calisthenics exercises. It is vendored here as a git submodule
at `data/exercises-dataset`, **pinned to commit `8872272`**, so every training run in
this repo sees byte-identical data. Bumping the pin is a deliberate, committed act:

```bash
git -C data/exercises-dataset pull origin main
git add data/exercises-dataset && git commit -m "build(data): bump dataset pin"
```

The catalog itself aggregates a public exercise dataset used to prototype search and
substitution features for a fitness app. We did not scrape it; we consume it as
published.

## Size and splits

**1324 exercise records.** Each carries a name, a target muscle, an equipment type, a
body part, secondary muscles, a 180×180 thumbnail, an animation GIF, and step-by-step
instructions in 10 languages.

| Split | Records | Purpose |
|---|---|---|
| train | 1081 | LoRA fine-tuning |
| val | 121 | every number reported in the README |
| test | 122 | untouched — reserved for M2 |

The split is stratified on `target` with seed 42, built by `scripts/02_prepare_data.py`.
`data/processed/` is gitignored: it is fully derived from the pinned submodule by a
seeded script, so it is regenerated rather than committed.

Two deliberate choices in how the split is made, both because the alternative would
have produced dishonest numbers:

- **Stratified, not random.** `levator scapulae` has 2 records in the entire catalog. A
  plain random split can put both in test, where the model has no chance, or none in
  train, where it can never be learned. Neither outcome says anything about the model.
- **Classes that cannot reach every split go to train.** One record — `trap bar
  deadlift`, the only `trap bar` row in the catalog — drew into val, where it would have
  been graded against a label the model was structurally never shown. It is moved back to
  train and the move is recorded in `data/processed/meta.json` under `rescued_to_train`,
  rather than silently reshaping the split.

## Language and licence

Instructions exist in English, Spanish, Italian, Turkish, Russian, Chinese, Hindi,
Polish, Korean and French. **We train on English only** — see limitations.

- Exercise metadata, instruction text, and repo tooling: **MIT**.
- Images and GIFs: **© [Gym visual](https://gymvisual.com/)**, referenced but *not*
  redistributed by this repo. Nothing in M1 touches the media; the M3 visual component
  will, and needs its own licence check against
  [Gym visual's terms](https://gymvisual.com/content/3-terms-and-conditions-of-use)
  before that.

Nothing in the licence restricts the use made here: non-commercial coursework on the
text fields.

## Task

**Structured extraction.** Given the text describing one exercise, produce a JSON object
with its catalog fields.

**Input** — the exercise name and its English instructions:

```
Name: cable incline pushdown
Instructions: Attach a straight bar to a high pulley cable machine. Stand facing away
from the machine with your feet shoulder-width apart. [...]
```

**Output** — the two fields, restricted to a closed label space:

```json
{"target": "lats", "equipment": "cable"}
```

The prompt lists every valid value for both fields. That is deliberate: the same prompt
grades the untrained model in `03_eval_zeroshot.py`, and a base model that has never seen
our labelling conventions would otherwise be marked down for vocabulary it was never
shown — which measures our conventions, not the model.

### Why two fields and not three

The catalog also carries `body_part`, and an earlier version of this pipeline generated
it as a third output. It should not be:

- `target → body_part` is a **strict function** — all 19 target values map to exactly one
  of 10 body parts, with no exceptions in the catalog.
- `category` is a **verbatim copy** of `body_part` in all 1324 records.

So `body_part` is free whenever `target` is right. Generating it would add an accuracy
column that inflates the headline result while measuring nothing. It is derived from a
lookup table instead, and `exlib.build_body_part_map` raises if a future dataset bump ever
makes `target` ambiguous, so the assumption cannot rot silently.

## Known biases and limitations

**Severe class imbalance.** `target` runs from `abs` (169 records) to `levator scapulae`
(2). `equipment` is worse: `body weight` has 325 records while **8 of the 28 equipment
classes have exactly one record each** and 11 have fewer than five. This is why every
result is reported with macro-F1 next to accuracy — accuracy alone lets a model that
learned six common classes and ignored the tail look competent.

**`equipment` is largely a string-matching problem.** The gold equipment value appears
verbatim in the exercise name in **60%** of records ("**dumbbell** biceps curl"), and a
substring rule with a majority-class fallback already reaches 87.6% on val. Any model
result on `equipment` should be read against that rule, not against the majority class.
`target` leaks at only 6.6%, which is why it is the headline metric.

**English-only training on a 10-language dataset.** We train and evaluate on
`instructions.en`. Nothing here supports the Spanish-speaking users the parent project
targets, and the multilingual instructions are machine translations of the English rather
than independently authored, so training on them would mostly teach translation artefacts.

**Vocabulary skewed to commercial gyms.** Equipment classes are dominated by machines
found in a fully equipped gym — `leverage machine`, `smith machine`, `stepmill machine`.
Home-workout, calisthenics, improvised-equipment and adaptive-athlete phrasing is thin to
absent. The model will most likely fail on exactly the users who are not in a commercial
gym.

**Single-source labels, no annotator agreement.** Every label comes from one upstream
catalog. There is no second annotator, no adjudication, and no measure of how consistent
the labelling is. Where the catalog is wrong or idiosyncratic, the model learns to be
wrong the same way, and our evaluation will score that as correct — the ceiling here is
agreement with this catalog, not correctness about exercise physiology.

**Near-duplicate records straddle the split.** Six exercise names appear twice. Three
pairs are exact duplicates (same name, instructions and labels) and all three landed
entirely in train, so they leak nothing. The other two — `barbell seated calf raise` and
`smith reverse calf raises` — have one copy in train and one in val with the same name and
the same labels but differently worded instructions. That is roughly **2 of 121 val
records (1.7%)** whose name→label mapping the model may have memorised rather than
inferred. Small, but it is a real upward bias on the reported numbers.

**`secondary_muscles` and `muscle_group` are unused.** Both are populated for all 1324
records and both are plausible extraction targets. We left them out to keep M1's scope to
the two fields that are genuinely independent; they are available if M2 wants a harder
task.

**The evaluation set is small.** 121 records. A one-record change moves accuracy by 0.8
points, so differences of a point or two between configurations in the ablation table are
noise, and are not treated as meaningful.
