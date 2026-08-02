"""Rule-based baseline for structured field extraction.

Establishes the numbers the fine-tuned model has to beat. Two naive strategies:

  majority   always predict the most frequent value
  substring  predict the field value that appears literally in the exercise
             name, falling back to the majority class

Both are *fitted on train and scored on the evaluation split*, exactly like
the model. Fitting them on the whole catalog — which is what this script used
to do — leaks the evaluation set into the baseline and makes the comparison
meaningless in whichever direction happens to be convenient.

Predictions are scored through exlib.score_predictions, the same function the
zero-shot and fine-tuned evaluations use, so every row of the results table is
computed by identical code.

Run:
    uv run python scripts/01_baseline.py
    uv run python scripts/01_baseline.py --split test
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exlib


def fit_majority(train, field):
    """Most frequent value in train."""
    return Counter(r[field] for r in train).most_common(1)[0][0]


def fit_substring_vocab(train, field):
    """Label values to scan for, longest first.

    Longest first so "lower legs" wins over "legs" and "ez barbell" wins over
    "barbell" — the shorter value is a substring of the longer one, and
    matching it first would silently mislabel every long-form record.
    """
    return sorted({r[field] for r in train}, key=len, reverse=True)


def predict_majority(records, majorities):
    return [dict(majorities) for _ in records]


def predict_substring(records, vocabs, majorities):
    preds = []
    for r in records:
        name = r["name"].lower()
        preds.append(
            {
                field: next(
                    (v for v in vocabs[field] if v.lower() in name), majorities[field]
                )
                for field in exlib.PREDICT_FIELDS
            }
        )
    return preds


def leakage(records, field):
    """Share of records whose gold value appears verbatim in the name.

    High leakage means the field is readable straight off the string, so a
    model that beats the majority class there has proved very little — the
    substring rule is the bar that matters.
    """
    return sum(r[field].lower() in r["name"].lower() for r in records) / len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="split to score on; strategies are always fitted on train (default: val)",
    )
    args = parser.parse_args()

    meta = exlib.load_meta()
    label_space = meta["label_space"]
    body_part_map = meta["body_part_map"]

    train = exlib.load_split("train")
    evalset = exlib.load_split(args.split)

    majorities = {f: fit_majority(train, f) for f in exlib.PREDICT_FIELDS}
    vocabs = {f: fit_substring_vocab(train, f) for f in exlib.PREDICT_FIELDS}

    strategies = {
        "majority": predict_majority(evalset, majorities),
        "substring": predict_substring(evalset, vocabs, majorities),
    }

    report = {
        "split": args.split,
        "fitted_on": "train",
        "n_train": len(train),
        "majority_class": majorities,
        "leakage": {f: round(leakage(evalset, f), 4) for f in exlib.PREDICT_FIELDS},
        "strategies": {
            name: exlib.score_predictions(preds, evalset, label_space, body_part_map)
            for name, preds in strategies.items()
        },
    }

    path = exlib.write_report("baseline", report)

    print(f"fitted on train ({len(train)}), scored on {args.split} ({len(evalset)})\n")
    header = f"{'strategy':<10} {'field':<11} {'accuracy':>9} {'macro-F1':>9} {'leakage':>8}"
    print(header)
    print("-" * len(header))
    for name, scores in report["strategies"].items():
        for field in exlib.PREDICT_FIELDS:
            s = scores["fields"][field]
            print(
                f"{name:<10} {field:<11} {s['accuracy']:>8.1%} "
                f"{s['macro_f1']:>9.3f} {report['leakage'][field]:>7.1%}"
            )
        print(f"{'':<10} {'joint':<11} {scores['joint_accuracy']:>8.1%}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
