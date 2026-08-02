"""Build the structured-extraction dataset from the exercise catalog.

Each record becomes a prompt-completion pair in TRL's conversational format:
the exercise text goes in as a user turn, the extracted fields come out as a
JSON assistant turn. TRL computes the loss on the completion only for this
format, so the label-space listing in the prompt costs nothing at train time.

The model predicts `target` and `equipment`. `body_part` is derived from
`target` afterwards — see exlib.PREDICT_FIELDS for why.

Run:
    uv run python scripts/02_prepare_data.py
    uv run python scripts/02_prepare_data.py --input-mode instructions
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exlib

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


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

    # Iterate classes in sorted order, not dict order, so the split depends
    # only on the seed and never on the order records happen to appear in.
    out = {name: [] for name in SPLITS}
    for _, group in sorted(by_class.items()):
        members = list(group)
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


def rescue_unseen_classes(splits, fields):
    """Move held-out records whose class never occurs in train back into train.

    The split above stratifies on `target` only, so a rare `equipment` value
    can still land entirely outside train — `trap bar` has exactly one record
    in the catalog and drew into val. Grading the model on a label it was
    structurally never shown measures the split, not the model, and the same
    reasoning already justifies sending tiny classes to train first.

    Returns the records moved, so the caller can report them rather than let
    the split quietly reshape itself.
    """
    moved = []
    for field in fields:
        seen = {r[field] for r in splits["train"]}
        for name in ("val", "test"):
            keep = []
            for r in splits[name]:
                if r[field] in seen:
                    keep.append(r)
                else:
                    seen.add(r[field])
                    splits["train"].append(r)
                    moved.append((name, field, r[field], r["name"]))
            splits[name] = keep
    return moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-mode",
        choices=["name", "instructions", "both"],
        default="both",
        help="which exercise text the model sees (default: both)",
    )
    args = parser.parse_args()

    records = exlib.load_catalog()
    label_space = exlib.build_label_space(records)

    # Raises if a target ever spans two body parts, which would invalidate
    # deriving body_part instead of predicting it.
    body_part_map = exlib.build_body_part_map(records)

    rng = random.Random(exlib.SEED)
    splits = stratified_split(records, "target", rng)
    moved = rescue_unseen_classes(splits, exlib.PREDICT_FIELDS)
    for split_name, field, value, name in moved:
        print(f"moved {split_name} -> train: {field}={value!r} unseen in train ({name})")
    if moved:
        print()

    exlib.PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        path = exlib.PROCESSED / f"{name}.jsonl"
        with path.open("w") as fh:
            for r in members:
                example = {
                    "prompt": [
                        {
                            "role": "user",
                            "content": exlib.build_prompt(
                                r, label_space, args.input_mode
                            ),
                        }
                    ],
                    "completion": [
                        {"role": "assistant", "content": exlib.gold_completion(r)}
                    ],
                    # Gold values are carried alongside so evaluation never has
                    # to re-open the catalog and re-derive the split.
                    "exercise_id": r["id"],
                    "name": r["name"],
                    **{f: r[f] for f in exlib.PREDICT_FIELDS},
                    exlib.DERIVED_FIELD: r[exlib.DERIVED_FIELD],
                }
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"{name:<6} {len(members):>5} examples  ->  {path}")

    meta = {
        "input_mode": args.input_mode,
        "seed": exlib.SEED,
        "predict_fields": exlib.PREDICT_FIELDS,
        "derived_field": exlib.DERIVED_FIELD,
        "label_space": label_space,
        "body_part_map": body_part_map,
        "counts": {k: len(v) for k, v in splits.items()},
        "rescued_to_train": [
            {"from": s, "field": f, "value": v, "exercise": n} for s, f, v, n in moved
        ],
    }
    meta_path = exlib.PROCESSED / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print("\nlabel space: " + ", ".join(f"{f}={len(v)}" for f, v in label_space.items()))
    print(f"body_part derived from target ({len(body_part_map)} targets -> "
          f"{len(set(body_part_map.values()))} body parts)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
