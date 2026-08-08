"""Build the MacGyver dataset: gym exercises re-asked as household-object questions.

The catalog knows nothing about water bottles or backpacks, so an equivalence
table has to exist for the task to be posable at all. Everything else is
derived from real data by deterministic code — no language model writes any
part of this dataset, which is why the usual synthetic-data failure modes
(inherited generator style, mode collapse, silent fabrication) cannot occur.

What is real and what is ours, exactly:

  real, verbatim from the catalog   exercise name, equipment, target, every
                                    instruction step
  ours, deterministic               the object equivalences (20 phrases), four
                                    adaptation sentences, the prompt wording,
                                    the safety line, the refusal wording
  written by an LLM                 nothing

Three kinds of example come out:

  answerable   the catalog has an exercise for (target, equipment-behind-object)
  refusal      it has none, and the honest answer is to say so
  unseen       an answerable example whose object never appears in training

The unseen split is the interesting one. With a handful of object phrases a
model can memorise "water bottles -> dumbbell" as a lookup instead of learning
"a matched pair you can grip -> dumbbell". Holding two objects per equipment
class out of training turns that risk into a measurement.

Run:
    uv run python scripts/01_macgyver_data.py
    uv run python scripts/01_macgyver_data.py --objects-per-exercise 3
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import macgyver as mg

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Refusals are split more evenly than the exercises. Each impossible
# (target, object) pair yields exactly one example, so an 80/10/10 split would
# leave a handful in val — too few to read a rate off. Weighting val and test
# up costs train a little and buys an evaluation that can be believed.
REFUSAL_SPLITS = {"train": 0.6, "val": 0.2, "test": 0.2}

# Exercises where the equipment is a platform, not a load. The mapping assumes
# the object replaces what the equipment *weighs*; in "kettlebell plyo push-up"
# your hands go on the kettlebells, and no advice about gripping a jug applies.
# Found by reading generated samples, not by reasoning about the schema.
PLATFORM_ROLE = re.compile(
    r"\b(?:on|onto|on top of|across|against)\s+"
    r"(?:the|a|an|your|each|both)?\s*\w*\s*(?:kettlebell|dumbbell)s?\b",
    re.IGNORECASE,
)

# Apparatus the offered object cannot stand in for, named anywhere in the steps.
# The equipment label only records the *primary* implement, so "exercise ball
# supine triceps extension" is filed under `dumbbell` while step 1 asks for an
# exercise ball, and `body weight` means "no external load" rather than "no
# equipment" — 111 of its 325 records still want a pull-up bar or a bench.
#
# The rule this enforces: the answer must need nothing beyond the object the
# user said they had. Telling someone holding two bricks to lie on a bench is
# not adaptation, and in a domain where the failure mode is injury, a
# confidently wrong answer is worse than a refusal.
#
# It costs 329 of 696 mapped exercises. Found by reading generated samples,
# three separate times — the schema gives no hint that any of this is wrong.
APPARATUS = re.compile(
    r"\b(?:pull-?up bar|chin-?up bar|dip bar|parallel bars|horizontal bar|"
    r"wall bar|captain's chair|bench|rings|exercise ball|stability ball|"
    r"swiss ball|box|step|chair|machine|barbell|cable|band)\b",
    re.IGNORECASE,
)


def needs_apparatus(record):
    """True when the exercise needs kit the object cannot stand in for.

    The name is searched as well as the steps: "bench pull-ups" carries its
    requirement in the title while its steps never spell it out.
    """
    text = record["name"] + " " + " ".join(record["instruction_steps"]["en"])
    return bool(PLATFORM_ROLE.search(text) or APPARATUS.search(text))


def stratified_split(records, key, rng):
    """Split by class so every split mirrors the class distribution.

    A plain random split would let a small class land entirely in one split,
    where it is either unlearnable or ungradeable. Classes too small to reach
    every split go to train first: better the model sees them at all than that
    they sit unreachable in val.
    """
    by_class = defaultdict(list)
    for r in records:
        by_class[r[key]].append(r)

    out = {name: [] for name in SPLITS}
    for members in by_class.values():
        members = members[:]
        rng.shuffle(members)
        n = len(members)
        n_val = int(n * SPLITS["val"])
        n_test = int(n * SPLITS["test"])
        out["val"] += members[:n_val]
        out["test"] += members[n_val : n_val + n_test]
        out["train"] += members[n_val + n_test :]
    return out


def answerable_row(record, obj):
    return {
        "prompt": [{"role": "user",
                    "content": mg.build_prompt(record["target"], obj)}],
        "completion": [{"role": "assistant",
                        "content": mg.build_answer(record, obj)}],
        "kind": "answerable",
        "object_seen": not mg.is_holdout(obj),
        "exercise_id": record["id"],
        "exercise_name": record["name"],
        "target": record["target"],
        "equipment": record["equipment"],
        "object": obj,
    }


def refusal_row(target, obj, alternative):
    return {
        "prompt": [{"role": "user", "content": mg.build_prompt(target, obj)}],
        "completion": [{"role": "assistant",
                        "content": mg.build_refusal(target, obj, alternative)}],
        "kind": "refusal",
        "object_seen": not mg.is_holdout(obj),
        "exercise_id": None,
        "exercise_name": None,
        "target": target,
        "equipment": None,
        "object": obj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-per-exercise", type=int, default=2)
    args = parser.parse_args()

    records = mg.load_catalog()
    rng = random.Random(mg.SEED)

    mapped = [r for r in records if r["equipment"] in mg.OBJECT_MAP]
    reachable = [r for r in mapped if not needs_apparatus(r)]
    n_dropped = len(mapped) - len(reachable)

    unknown = {r["equipment"] for r in records} - set(mg.OBJECT_MAP) - set(mg.EXCLUDED)
    if unknown:
        # A dataset bump that adds an equipment class must be an explicit
        # decision, not a silent omission from the training set.
        raise ValueError(
            f"equipment classes with no mapping and no exclusion: {sorted(unknown)}"
        )

    splits = stratified_split(reachable, "target", rng)

    # Whether a question is answerable is a fact about the catalog, not about
    # where a record happened to land, so this is built over every reachable
    # record rather than per split.
    answerable = {
        (r["target"], obj)
        for r in reachable
        for obj in mg.objects_for(r["equipment"], include_holdout=True)
    }
    all_objects = [
        obj for eq in mg.OBJECT_MAP for obj in mg.objects_for(eq, include_holdout=True)
    ]
    all_targets = sorted({r["target"] for r in records})
    impossible = sorted(
        (t, o) for t in all_targets for o in all_objects if (t, o) not in answerable
    )

    # Alternatives named in a refusal always come from train. Pointing at a val
    # exercise would put its name in front of the model during training, which
    # is the leak the split exists to prevent.
    by_target = defaultdict(list)
    for r in splits["train"]:
        by_target[r["target"]].append(r)

    def alternative_for(target):
        pool = by_target.get(target) or [r for r in records if r["target"] == target]
        # Body weight first: the nearest alternative should be the one needing
        # least, not whichever happened to sort first.
        return sorted(
            pool, key=lambda r: (r["equipment"] != "body weight", r["name"])
        )[0]

    # Train never sees a held-out object, in an answerable example or a refusal.
    rng.shuffle(impossible)
    seen_impossible = [p for p in impossible if not mg.is_holdout(p[1])]
    held_impossible = [p for p in impossible if mg.is_holdout(p[1])]
    n_val = round(len(seen_impossible) * REFUSAL_SPLITS["val"])
    n_test = round(len(seen_impossible) * REFUSAL_SPLITS["test"])
    refusals = {
        "val": seen_impossible[:n_val] + held_impossible[: len(held_impossible) // 2],
        "test": seen_impossible[n_val : n_val + n_test]
                + held_impossible[len(held_impossible) // 2 :],
        "train": seen_impossible[n_val + n_test :],
    }

    mg.PROCESSED.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, members in splits.items():
        is_train = name == "train"
        rows = []
        for record in members:
            pools = [mg.objects_for(record["equipment"])]
            if not is_train:
                pools.append(mg.OBJECT_MAP[record["equipment"]]["holdout"])
            for pool in pools:
                for k in range(min(args.objects_per_exercise, len(pool))):
                    # Rotate by id so two exercises sharing an equipment class
                    # do not both get the first object in the list.
                    rows.append(answerable_row(
                        record, pool[(int(record["id"]) + k) % len(pool)]
                    ))

        for target, obj in refusals[name]:
            rows.append(refusal_row(target, obj, alternative_for(target)))

        rng.shuffle(rows)
        path = mg.PROCESSED / f"{name}.jsonl"
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats[name] = {
            "total": len(rows),
            "refusals": sum(r["kind"] == "refusal" for r in rows),
            "unseen_object": sum(not r["object_seen"] for r in rows),
        }

    meta = {
        "seed": mg.SEED,
        "objects_per_exercise": args.objects_per_exercise,
        "counts": stats,
        "reachable_exercises": len(reachable),
        "dropped_needs_apparatus": n_dropped,
        "catalog_exercises": len(records),
        "object_map": mg.OBJECT_MAP,
        "adaptation_templates": mg.ADAPTATION,
        "excluded_equipment": mg.EXCLUDED,
        "impossible_pairs": len(impossible),
        "provenance": {
            "exercise_names_equipment_targets_steps": "catalog, verbatim",
            "object_equivalences_and_wording": "this script, deterministic",
            "llm_generated": "none",
        },
    }
    (mg.PROCESSED / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    n_held = sum(len(e["holdout"]) for e in mg.OBJECT_MAP.values())
    n_seen = sum(len(e["seen"]) for e in mg.OBJECT_MAP.values())
    print(f"catalog     {len(records)} exercises")
    print(f"reachable   {len(reachable)} across {len(mg.OBJECT_MAP)} equipment classes")
    print(f"excluded    {len(records) - len(mapped)} across {len(mg.EXCLUDED)} classes")
    print(f"dropped     {n_dropped} whose steps need kit the object cannot replace")
    print(f"objects     {n_seen} in training, {n_held} held out for the unseen test")
    print(f"impossible  {len(impossible)} (target, object) pairs with no exercise\n")
    for name in SPLITS:
        s = stats[name]
        print(f"{name:<6} {s['total']:>5} examples  "
              f"{s['refusals']:>3} refusals  {s['unseen_object']:>3} unseen-object"
              f"  ->  {mg.PROCESSED / f'{name}.jsonl'}")
    if stats["val"]["refusals"] < 10:
        print(f"\nWARNING: {stats['val']['refusals']} refusals in val is too few to "
              "read a rate off — widen OBJECT_MAP to create more impossible pairs")
    print(f"\nwrote {mg.PROCESSED / 'meta.json'}")


if __name__ == "__main__":
    main()
