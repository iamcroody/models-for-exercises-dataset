"""The baseline: the same base model, no fine-tuning, on the same validation set.

This is the comparison the assignment recommends and the only one that isolates
what LoRA contributed. Same prompts, same split, same greedy decoding, same
scorer — the only difference between this and 05 is whether an adapter is
loaded, so the delta cannot be an artefact of the harness.

    uv run python scripts/03_eval_zeroshot.py --limit 8      # quick smoke test
    uv run python scripts/03_eval_zeroshot.py --split val
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import macgyver as mg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=mg.BASE_MODEL)
    parser.add_argument("--adapter", help="LoRA adapter; omit for the baseline")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, help="score only the first N rows")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--save-replies", help="write raw replies to this path")
    args = parser.parse_args()

    rows = mg.load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]

    device, dtype = mg.pick_device_dtype()
    label = f"{args.model}{' + ' + args.adapter if args.adapter else ' (no adapter)'}"
    print(f"{label} on {device} / {'4-bit' if args.four_bit else dtype}")
    print(f"scoring {len(rows)} {args.split} examples")

    tok = mg.load_tokenizer(args.model)
    model = mg.load_model(args.model, adapter=args.adapter, four_bit=args.four_bit)
    replies = mg.generate_replies(model, tok, rows, batch_size=args.batch_size)

    report = mg.score_predictions(replies, rows, mg.load_catalog())
    report.update({"model": args.model, "adapter": args.adapter, "split": args.split,
                   "n_scored": len(rows)})
    mg.print_report(report, label)

    # Three qualitative examples, required by the assignment. Chosen rather
    # than cherry-picked: the first answerable row, the first refusable one,
    # and the first using an object the model was never trained on.
    report["examples"] = []
    wanted = [("answerable", lambda r: r["kind"] == "answerable" and r["object_seen"]),
              ("refusal", lambda r: r["kind"] == "refusal"),
              ("unseen object", lambda r: not r["object_seen"])]
    for why, match in wanted:
        found = next(((row, rep) for row, rep in zip(rows, replies) if match(row)), None)
        if found:
            row, reply = found
            report["examples"].append({
                "why_shown": why,
                "prompt": row["prompt"][0]["content"],
                "gold": row["completion"][0]["content"],
                "generated": reply,
            })

    if args.save_replies:
        Path(args.save_replies).write_text(json.dumps(replies, indent=2))
        print(f"\nwrote {args.save_replies}")
    if args.report_name:
        print(f"wrote {mg.write_report(args.report_name, report)}")


if __name__ == "__main__":
    main()
