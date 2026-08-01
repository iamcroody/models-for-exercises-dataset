"""Build the structured-extraction dataset from the exercise catalog.

Each record becomes a prompt-completion pair in TRL's conversational format:
the exercise text goes in as a user turn, the extracted fields come out as a
JSON assistant turn.

The prompt lists the valid values for every field. That is deliberate: the
same prompt is used to evaluate the untrained model in 03, and a base model
with no knowledge of our label space would otherwise be graded on vocabulary
it was never shown.

Run:
    uv run python scripts/02_prepare_data.py
    uv run python scripts/02_prepare_data.py --input-mode instructions
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DATASET = Path("data/exercises-dataset/data/exercises.json")
OUT_DIR = Path("data/processed")
FIELDS = ["target", "body_part", "equipment"]
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def build_prompt(record, label_space, input_mode):
    if input_mode == "name":
        body = f"Name: {record['name']}"
    elif input_mode == "instructions":
        body = f"Instructions: {record['instructions']['en']}"
    else:
        body = f"Name: {record['name']}\nInstructions: {record['instructions']['en']}"

    allowed = "\n".join(f"{f}: {' | '.join(label_space[f])}" for f in FIELDS)
    return (
        "Extract the structured fields for this exercise.\n\n"
        f"{body}\n\n"
        "Reply with JSON only, choosing from these exact values:\n"
        f"{allowed}"
    )


def stratified_split(records, key, rng):
    """Split by class so every split mirrors the class distribution.

    A plain random split would let a 2-example class such as
    `levator scapulae` land entirely in test, where the model has no chance,
    or vanish from train, where it can never be learned. Neither failure is
    informative about the model.

    Classes too small to reach every split go to train first: better that the
    model sees them at all than that they sit unlearnable in test.
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

    for split in out.values():
        rng.shuffle(split)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-mode",
        choices=["name", "instructions", "both"],
        default="both",
        help="which exercise text the model sees (default: both)",
    )
    args = parser.parse_args()

    records = json.loads(DATASET.read_text())
    label_space = {f: sorted({r[f] for r in records}) for f in FIELDS}

    rng = random.Random(SEED)
    splits = stratified_split(records, "target", rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as fh:
            for r in members:
                example = {
                    "prompt": [
                        {
                            "role": "user",
                            "content": build_prompt(r, label_space, args.input_mode),
                        }
                    ],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": json.dumps({f: r[f] for f in FIELDS}),
                        }
                    ],
                    "exercise_id": r["id"],
                }
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"{name:<6} {len(members):>5} examples  ->  {path}")

    meta = {
        "input_mode": args.input_mode,
        "seed": SEED,
        "fields": FIELDS,
        "label_space": label_space,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nlabel space: " + ", ".join(f"{f}={len(v)}" for f, v in label_space.items()))
    print(f"wrote {OUT_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()
