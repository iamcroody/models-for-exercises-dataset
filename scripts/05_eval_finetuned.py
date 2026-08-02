"""Evaluate the LoRA-tuned model and compare it against both baselines.

Metrics, and why these:

  accuracy        The task is classification into a closed set, so exact match
                  is the natural primary metric. `target` accuracy is the
                  headline: 19 classes, the weakest baseline, the most room.

  macro-F1        Accuracy alone hides the tail, and the tail is most of the
                  label space — `levator scapulae` has 2 records in the whole
                  catalog. Macro-F1 weights a rare class the same as `abs`,
                  so a model that only learned the six common targets cannot
                  hide behind it.

  JSON valid      Reported rather than repaired. A model that answers 60% of
  in-label        the time must not outscore one that always answers, and
                  "invented a label we never offered" is a different failure
                  from "picked the wrong one from the list". These two are
                  where the untrained model actually loses.

  joint           Both fields right on the same record — what a caller
                  consuming this as a structured extractor actually gets.

Run:
    uv run python scripts/05_eval_finetuned.py
    uv run python scripts/05_eval_finetuned.py --adapter models/r8-all-linear
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exlib


def pick_examples(records, preds, replies, limit=4):
    """A hit, a rare-class case, and a failure — whatever the run provides.

    Chosen by rule rather than by hand so the examples cannot be cherry-picked
    into flattering the model, and so they update automatically when the run
    does.
    """
    from collections import Counter

    freq = Counter(r["target"] for r in records)
    rows = []
    for rec, pred, reply in zip(records, preds, replies):
        gold = {f: rec[f] for f in exlib.PREDICT_FIELDS}
        got = {f: (pred or {}).get(f) for f in exlib.PREDICT_FIELDS}
        rows.append(
            {
                "name": rec["name"],
                "gold": gold,
                "predicted": got,
                "raw_reply": reply.strip(),
                "correct": gold == got,
                "rarity": freq[rec["target"]],
            }
        )

    chosen, used = [], set()

    def take(pool, why):
        for row in pool:
            if row["name"] not in used:
                used.add(row["name"])
                chosen.append({**row, "why_shown": why})
                return

    hits = [r for r in rows if r["correct"]]
    misses = [r for r in rows if not r["correct"]]
    take(hits, "typical correct extraction")
    take(sorted(hits, key=lambda r: r["rarity"]), "correct on a rare target class")
    take(misses, "failure case")
    take(sorted(misses, key=lambda r: r["rarity"]), "failure on a rare target class")
    return chosen[:limit]


def load_if_present(name):
    path = exlib.REPORTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--model", default=exlib.BASE_MODEL)
    parser.add_argument("--adapter", default="models/r16-all-linear")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--report-name",
        default="finetuned",
        help="reports/<name>.json (use a distinct name for ablation runs)",
    )
    args = parser.parse_args()

    adapter = Path(args.adapter)
    if not adapter.is_absolute():
        adapter = exlib.ROOT / adapter
    if not adapter.exists():
        raise FileNotFoundError(
            f"{adapter} is missing. Train an adapter first:\n"
            "    uv run python scripts/04_train_lora.py"
        )

    meta = exlib.load_meta()
    evalset = exlib.load_split(args.split)

    device, dtype = exlib.pick_device_dtype()
    print(f"{args.model} + {adapter.name} on {device} / {dtype}")
    print(f"scoring {len(evalset)} {args.split} examples\n")

    tok = exlib.load_tokenizer(args.model)
    model = exlib.load_model(args.model, adapter=str(adapter))

    replies = exlib.generate_replies(model, tok, evalset, batch_size=args.batch_size)
    preds = [exlib.parse_json_reply(r) for r in replies]

    report = exlib.score_predictions(
        preds, evalset, meta["label_space"], meta["body_part_map"]
    )
    report["model"] = args.model
    report["adapter"] = str(adapter.relative_to(exlib.ROOT))
    report["split"] = args.split
    report["examples"] = pick_examples(evalset, preds, replies)

    summary = adapter / "training_summary.json"
    if summary.exists():
        report["training"] = json.loads(summary.read_text())

    path = exlib.write_report(args.report_name, report)

    # ---- comparison table -------------------------------------------------
    baseline = load_if_present("baseline")
    zeroshot = load_if_present("zeroshot")

    rows = []
    if baseline and baseline["split"] == args.split:
        for name, scores in baseline["strategies"].items():
            rows.append((f"rule: {name}", scores))
    if zeroshot and zeroshot["split"] == args.split:
        rows.append(("zero-shot (no LoRA)", zeroshot))
    rows.append(("fine-tuned (LoRA)", report))

    header = (
        f"{'':<22} {'target':>8} {'equip':>8} {'joint':>8} "
        f"{'t-F1':>7} {'e-F1':>7} {'JSON':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, s in rows:
        t, e = s["fields"]["target"], s["fields"]["equipment"]
        json_rate = s.get("json_valid_rate")
        print(
            f"{name:<22} {t['accuracy']:>7.1%} {e['accuracy']:>7.1%} "
            f"{s['joint_accuracy']:>7.1%} {t['macro_f1']:>7.3f} {e['macro_f1']:>7.3f} "
            f"{'n/a' if json_rate is None else format(json_rate, '.1%'):>7}"
        )

    if zeroshot and zeroshot["split"] == args.split:
        print("\ndelta from fine-tuning (vs the same model, untrained):")
        for field in exlib.PREDICT_FIELDS:
            before = zeroshot["fields"][field]
            after = report["fields"][field]
            print(
                f"  {field:<10} accuracy {before['accuracy']:>6.1%} -> "
                f"{after['accuracy']:>6.1%}   in-label "
                f"{before['in_label_space_rate']:>6.1%} -> "
                f"{after['in_label_space_rate']:>6.1%}"
            )

    print("\nqualitative examples:")
    for ex in report["examples"]:
        mark = "ok  " if ex["correct"] else "MISS"
        print(f"  [{mark}] {ex['name']}  ({ex['why_shown']})")
        print(f"         gold {ex['gold']}")
        print(f"         got  {ex['predicted']}")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
