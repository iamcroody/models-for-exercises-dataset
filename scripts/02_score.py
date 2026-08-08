"""Score model replies, and self-test the scorer against the gold completions.

Run with no arguments it grades the dataset's own answers. Every number must
come out perfect: the gold completions name real catalog exercises with the
right target and the right equipment, and recite those exercises' steps
verbatim. Anything less means the scorer is broken, and finding that out here
costs a second rather than a training run.

    uv run python scripts/02_score.py                      # self-test on val
    uv run python scripts/02_score.py --replies out.json   # grade a real model
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import macgyver as mg


def print_report(report, title):
    print(f"\n{title}")
    print(f"  {'examples':<26} {report['n']} "
          f"({report['n_answerable']} answerable, {report['n_refusable']} refusable)")
    print(f"  {'format ok':<26} {report['format_ok']:.1%}")
    print(f"  {'constraint satisfaction':<26} {report['constraint_satisfaction']:.1%}"
          "   <- headline")
    print(f"    {'exercise is real':<24} {report['exercise_real']:.1%}")
    print(f"    {'target matches':<24} {report['target_match']:.1%}")
    print(f"    {'equipment matches':<24} {report['equipment_match']:.1%}")
    grounding = report["step_grounding_rouge_l"]
    print(f"  {'step grounding ROUGE-L':<26} "
          f"{grounding:.3f} (n={report['n_grounded']})" if grounding is not None
          else f"  {'step grounding ROUGE-L':<26} n/a")
    r = report["refusal"]
    print(f"  {'refusal P / R / F1':<26} {r['precision']:.1%} / {r['recall']:.1%} "
          f"/ {r['f1']:.1%}")
    for label in ("seen_objects", "unseen_objects"):
        s = report[label]
        value = f"{s['constraint_satisfaction']:.1%}" if s["n"] else "n/a"
        print(f"  {label.replace('_', ' '):<26} {value} (n={s['n']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--replies",
        help="JSON list of model replies, aligned with the split. "
             "Omitted, the gold completions are scored as a self-test.",
    )
    parser.add_argument("--report-name", help="write reports/<name>.json")
    args = parser.parse_args()

    rows = mg.load_split(args.split)
    catalog = mg.load_catalog()

    if args.replies:
        replies = json.loads(Path(args.replies).read_text())
        title = f"{args.replies} on {args.split} ({len(rows)} examples)"
    else:
        replies = [r["completion"][0]["content"] for r in rows]
        title = f"self-test: gold completions on {args.split} ({len(rows)} examples)"

    report = mg.score_predictions(replies, rows, catalog)
    print_report(report, title)

    if not args.replies:
        # The gold answers are correct by construction. If the scorer disagrees
        # it is the scorer that is wrong, and every later number would inherit
        # the error silently.
        perfect = (
            report["format_ok"] == 1.0
            and report["constraint_satisfaction"] == 1.0
            and report["refusal"]["f1"] == 1.0
            and report["step_grounding_rouge_l"] > 0.99
        )
        print("\nself-test", "PASSED" if perfect else "FAILED")
        if not perfect:
            raise SystemExit(1)

    if args.report_name:
        report["split"] = args.split
        print(f"\nwrote {mg.write_report(args.report_name, report)}")


if __name__ == "__main__":
    main()
