"""Build the MacGyver dataset: gym exercises re-asked as household-object questions.

The catalog knows nothing about water bottles or backpacks. This script crosses
it with a hand-written equivalence table so that every training pair is derived
from real data by deterministic code — no language model writes any part of it,
which is why the usual synthetic-data failure modes (inherited generator style,
mode collapse, silent fabrication) cannot occur here.

What is real and what is ours, exactly:

  real, verbatim from the catalog   exercise name, equipment, target, every
                                    instruction step
  ours, deterministic               the equipment <-> object equivalences, the
                                    prompt wording, the Adaptation and Safety
                                    lines, the refusal wording
  written by an LLM                 nothing

Two kinds of example come out:

  answerable   the catalog has an exercise for (target, equipment-behind-object)
  refusal      it has none, and the honest answer is to say so and point at the
               nearest thing that does exist

The refusals are the point of the whole exercise. An untuned model asked to
train lats with two water bottles will invent something, because no dumbbell
exercise in the catalog targets lats. Teaching the model where the catalog
ends is what separates this from a party trick.

Run:
    uv run python scripts/06_macgyver_data.py
    uv run python scripts/06_macgyver_data.py --objects-per-exercise 1
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CATALOG = Path("data/exercises-dataset/data/exercises.json")
OUT_DIR = Path("data/processed/macgyver")
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Refusals are split more evenly than the exercises. Each impossible
# (target, object) pair yields exactly one example and the catalog only
# produces 51 of them, so an 80/10/10 split would leave five in val — too few
# to read a refusal rate off. Weighting val and test up costs train a handful
# of examples and buys an evaluation that can actually be believed.
REFUSAL_SPLITS = {"train": 0.6, "val": 0.2, "test": 0.2}

SEED = 42

# Equipment we are willing to improvise, and what a person plausibly has at
# home. Each entry is (object phrase, the sentence that explains the swap).
#
# The second element is not decoration. "Use a backpack instead of a
# kettlebell" is useless advice; the failure mode of improvised loading is
# always the same — the grip, the balance, or the load — so each note names
# the specific thing that goes wrong.
OBJECT_MAP = {
    "dumbbell": [
        ("two filled water bottles",
         "Fill both bottles to the same level so the load stays even on each "
         "side, and grip them around the middle rather than the neck."),
        ("two full cans of food",
         "Pick two cans of the same size. They are lighter than most "
         "dumbbells, so add repetitions rather than swinging them faster."),
        ("two one-litre juice cartons",
         "Hold each carton by the body, not the cap, and keep them upright so "
         "the contents do not shift mid-repetition."),
    ],
    "kettlebell": [
        ("a filled detergent jug with a handle",
         "The handle sits off to one side, so the jug will pull your wrist "
         "outward. Grip firmly and keep the wrist straight."),
        ("a backpack held by one strap",
         "Loop the strap over your palm rather than your fingers, and pack the "
         "contents tight so nothing shifts as the bag swings."),
    ],
    "weighted": [
        ("a backpack loaded with books",
         "Pack the heaviest books against your spine and tighten the straps so "
         "the load cannot slide during the movement."),
        ("a filled shopping bag",
         "Double-bag it and hold both handles together — a single handle will "
         "dig into your hand before your muscles are done."),
    ],
    # Not an improvisation at all: the honest answer is that nothing is needed.
    # Kept in because "I have nothing" is the most common version of this
    # question, and a model that has only ever seen substitutions will invent
    # one here too.
    "body weight": [
        ("nothing at all",
         "This one needs no equipment. Clear enough floor space to extend "
         "fully in every direction before you start."),
        ("just the floor and a wall",
         "No equipment is required. Use the wall only for balance, never to "
         "push yourself through a repetition you cannot control."),
    ],
}

# Equipment deliberately left out, and why. Printed on every run so the
# decision stays visible rather than living in a comment nobody opens.
EXCLUDED = {
    "cable": "steps say 'attach the bar to the high pulley' — reciting that "
             "to someone holding a towel is nonsense, not adaptation",
    "barbell": "loads and failure modes that improvised equipment cannot "
               "safely reproduce",
    "ez barbell": "same as barbell",
    "olympic barbell": "same as barbell",
    "smith machine": "the machine's fixed bar path is the exercise",
    "leverage machine": "the machine's fixed bar path is the exercise",
    "sled machine": "the machine's fixed bar path is the exercise",
    "assisted": "the assistance is the machine",
    "stability ball": "no household object has its instability",
    "bosu ball": "no household object has its instability",
    "medicine ball": "these are throwing movements; a backpack is not throwable",
    "rope": "the class mixes battling ropes, jump rope and stretches — no "
            "single object substitutes for all three",
    "band": "elastic tension has no reliable household equivalent; tights and "
            "inner tubes fail unpredictably under load",
    "resistance band": "same as band",
    "roller": "no household equivalent",
    "wheel roller": "no household equivalent",
    "hammer": "single record, and the object is already a household one",
    "tire": "single record, not a household object",
    "trap bar": "single record, no household equivalent",
    "elliptical machine": "cardio machine, no household equivalent",
    "stationary bike": "cardio machine, no household equivalent",
    "skierg machine": "cardio machine, no household equivalent",
    "stepmill machine": "cardio machine, no household equivalent",
    "upper body ergometer": "cardio machine, no household equivalent",
}

SAFETY = "Keep the movement controlled and stop if you feel pain."

# Exercises where the equipment is a platform, not a load. The whole mapping
# above assumes the object replaces what the equipment *weighs*; in "kettlebell
# plyo push-up" your hands go on the kettlebells, and no advice about gripping
# a detergent jug applies. Caught by reading generated examples, not by
# reasoning about the schema — six records out of 696, all dropped rather than
# mapped wrongly.
PLATFORM_ROLE = re.compile(
    r"\b(?:on|onto|on top of|across|against)\s+"
    r"(?:the|a|an|your|each|both)?\s*\w*\s*(?:kettlebell|dumbbell)s?\b",
    re.IGNORECASE,
)


def is_platform_role(record):
    return bool(PLATFORM_ROLE.search(" ".join(record["instruction_steps"]["en"])))

PROMPT = (
    "I want to train my {target} but I have no gym equipment. "
    "All I have is {object}.\n"
    "Recommend one safe exercise and explain how to do it.\n\n"
    "Answer in exactly this format:\n"
    "Exercise: <name>\n"
    "Gym equivalent: <equipment>\n"
    "Adaptation: <how to use what I have>\n"
    "Steps:\n"
    "1. <step>\n"
    "Safety: <one line>\n\n"
    "If nothing you know of trains that muscle with that object, say so "
    "instead of inventing one."
)


def build_prompt(target, obj):
    """The one prompt definition. Training, the zero-shot baseline and the
    fine-tuned evaluation must all render this identically, or the reported
    delta measures prompt drift instead of fine-tuning."""
    return PROMPT.format(target=target, object=obj)


def build_answer(record, obj, note):
    """An answerable completion: our two template lines around real steps."""
    steps = "\n".join(
        f"{i}. {s}" for i, s in enumerate(record["instruction_steps"]["en"], 1)
    )
    return (
        f"Exercise: {record['name']}\n"
        f"Gym equivalent: {record['equipment']}\n"
        f"Adaptation: {note}\n"
        f"Steps:\n{steps}\n"
        f"Safety: {SAFETY}"
    )


def build_refusal(target, obj, alternative):
    """A refusal completion, pointing at something the catalog does have."""
    return (
        f"No safe option: I know no {target} exercise that works with "
        f"{obj}.\n"
        f"Closest alternative: {alternative['name']}, which needs "
        f"{alternative['equipment']}."
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects-per-exercise", type=int, default=2)
    args = parser.parse_args()

    records = json.loads(CATALOG.read_text())
    rng = random.Random(SEED)

    mapped = [r for r in records if r["equipment"] in OBJECT_MAP]
    reachable = [r for r in mapped if not is_platform_role(r)]
    n_platform = len(mapped) - len(reachable)

    unknown = {r["equipment"] for r in records} - set(OBJECT_MAP) - set(EXCLUDED)
    if unknown:
        # A dataset bump that adds an equipment class must be an explicit
        # decision, not a silent omission from the training set.
        raise ValueError(
            f"equipment classes with no mapping and no exclusion: {sorted(unknown)}"
        )

    splits = stratified_split(reachable, "target", rng)

    # Which (target, object) pairs the catalog can actually answer. Built over
    # every reachable record, not per split: whether a question is answerable
    # is a fact about the catalog, not about where a record happened to land.
    answerable = {
        (r["target"], obj)
        for r in reachable
        for obj, _ in OBJECT_MAP[r["equipment"]]
    }
    all_objects = [obj for pairs in OBJECT_MAP.values() for obj, _ in pairs]
    all_targets = sorted({r["target"] for r in records})
    impossible = sorted(
        (t, o) for t in all_targets for o in all_objects if (t, o) not in answerable
    )

    # Alternatives named in a refusal always come from train. A refusal that
    # pointed at a val exercise would put that exercise's name in front of the
    # model during training, which is exactly the leak the split exists to
    # prevent.
    by_target = defaultdict(list)
    for r in splits["train"]:
        by_target[r["target"]].append(r)
    fallback = sorted(records, key=lambda r: r["name"])

    def alternative_for(target):
        pool = by_target.get(target) or [r for r in fallback if r["target"] == target]
        # Body weight first: the nearest alternative should be the one needing
        # least, not whichever happened to sort first.
        pool = sorted(pool, key=lambda r: (r["equipment"] != "body weight", r["name"]))
        return pool[0]

    rng.shuffle(impossible)
    n_val = round(len(impossible) * REFUSAL_SPLITS["val"])
    n_test = round(len(impossible) * REFUSAL_SPLITS["test"])
    refusals = {
        "val": impossible[:n_val],
        "test": impossible[n_val : n_val + n_test],
        "train": impossible[n_val + n_test :],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts, refusal_counts = {}, {}
    for name, members in splits.items():
        rows = []
        for record in members:
            options = OBJECT_MAP[record["equipment"]]
            for k in range(min(args.objects_per_exercise, len(options))):
                # Rotate by id so two exercises sharing an equipment class do
                # not both get the first object in the list.
                obj, note = options[(int(record["id"]) + k) % len(options)]
                rows.append({
                    "prompt": [{"role": "user",
                                "content": build_prompt(record["target"], obj)}],
                    "completion": [{"role": "assistant",
                                    "content": build_answer(record, obj, note)}],
                    "kind": "answerable",
                    "exercise_id": record["id"],
                    "exercise_name": record["name"],
                    "target": record["target"],
                    "equipment": record["equipment"],
                    "object": obj,
                })

        take = refusals[name]
        for target, obj in take:
            alt = alternative_for(target)
            rows.append({
                "prompt": [{"role": "user", "content": build_prompt(target, obj)}],
                "completion": [{"role": "assistant",
                                "content": build_refusal(target, obj, alt)}],
                "kind": "refusal",
                "exercise_id": None,
                "exercise_name": None,
                "target": target,
                "equipment": None,
                "object": obj,
            })

        rng.shuffle(rows)
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[name] = len(rows)
        refusal_counts[name] = len(take)

    meta = {
        "seed": SEED,
        "objects_per_exercise": args.objects_per_exercise,
        "counts": counts,
        "refusal_counts": refusal_counts,
        "reachable_exercises": len(reachable),
        "dropped_platform_role": n_platform,
        "catalog_exercises": len(records),
        "object_map": {k: [o for o, _ in v] for k, v in OBJECT_MAP.items()},
        "excluded_equipment": EXCLUDED,
        "impossible_pairs": len(impossible),
        "provenance": {
            "exercise_names_equipment_targets_steps": "catalog, verbatim",
            "object_equivalences_and_wording": "this script, deterministic",
            "llm_generated": "none",
        },
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"catalog     {len(records)} exercises")
    print(f"reachable   {len(reachable)} across {len(OBJECT_MAP)} equipment classes")
    print(f"excluded    {len(records) - len(mapped)} across {len(EXCLUDED)} classes")
    print(f"dropped     {n_platform} where the equipment is a platform, not a load")
    print(f"targets     {len({r['target'] for r in reachable})} of {len(all_targets)} reachable")
    print(f"impossible  {len(impossible)} (target, object) pairs with no exercise\n")
    for name in SPLITS:
        share = refusal_counts[name] / counts[name]
        print(f"{name:<6} {counts[name]:>5} examples "
              f"({refusal_counts[name]} refusals, {share:.1%})  ->  "
              f"{OUT_DIR / f'{name}.jsonl'}")
    if refusal_counts["val"] < 10:
        print(f"\nWARNING: {refusal_counts['val']} refusals in val is too few to "
              "read a rate off — widen OBJECT_MAP to create more impossible pairs")
    print(f"\nwrote {OUT_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
