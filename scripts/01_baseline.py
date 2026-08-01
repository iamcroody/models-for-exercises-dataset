"""Rule-based baseline for structured field extraction.

Establishes the numbers a fine-tuned model has to beat. Two naive strategies
per field:

  majority   always predict the most frequent value
  substring  predict the field value that appears literally in the exercise
             name, falling back to the majority class

Run:
    uv run python scripts/01_baseline.py
"""

import json
from collections import Counter
from pathlib import Path

DATASET = Path("data/exercises-dataset/data/exercises.json")
REPORT = Path("reports/baseline.json")
FIELDS = ["target", "body_part", "equipment"]


def majority(records, field):
    """Always predict the most frequent value."""
    top = Counter(r[field] for r in records).most_common(1)[0][0]
    correct = sum(1 for r in records if r[field] == top)
    return correct / len(records), top


def substring(records, field):
    """Predict the field value literally present in the name, else majority."""
    # Longest first so "lower legs" wins over "legs".
    values = sorted({r[field] for r in records}, key=len, reverse=True)
    fallback = Counter(r[field] for r in records).most_common(1)[0][0]

    correct = 0
    for r in records:
        name = r["name"].lower()
        pred = next((v for v in values if v.lower() in name), fallback)
        correct += pred == r[field]
    return correct / len(records)


def leakage(records, field):
    """Share of records whose gold value appears verbatim in the name.

    High leakage means the field is readable off the string and a model
    adds little over a regex.
    """
    hits = sum(1 for r in records if r[field].lower() in r["name"].lower())
    return hits / len(records)


def main():
    records = json.loads(DATASET.read_text())

    report = {"n_records": len(records), "fields": {}}
    for field in FIELDS:
        maj_acc, maj_val = majority(records, field)
        report["fields"][field] = {
            "n_classes": len({r[field] for r in records}),
            "leakage": round(leakage(records, field), 4),
            "majority_accuracy": round(maj_acc, 4),
            "majority_class": maj_val,
            "substring_accuracy": round(substring(records, field), 4),
        }

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{len(records)} records\n")
    header = f"{'field':<12} {'classes':>7} {'leakage':>8} {'majority':>9} {'substring':>10}"
    print(header)
    print("-" * len(header))
    for field, s in report["fields"].items():
        print(
            f"{field:<12} {s['n_classes']:>7} {s['leakage']:>7.1%} "
            f"{s['majority_accuracy']:>8.1%} {s['substring_accuracy']:>9.1%}"
        )
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
