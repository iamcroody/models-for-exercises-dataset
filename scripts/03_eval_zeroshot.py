"""Zero-shot baseline: the same base model, before any fine-tuning.

This is the baseline the assignment recommends, and the only one that isolates
what fine-tuning contributed — the rule-based baseline in 01 measures the
dataset, this measures the model. Same prompts, same split, same decoding, no
adapter, so the delta against 05 is attributable to the LoRA weights alone.

The prompt lists every valid label, so the untrained model is not being marked
down for vocabulary it was never shown. What it is being marked on is whether
it can pick from that list and answer in JSON.

Run:
    uv run python scripts/03_eval_zeroshot.py
    uv run python scripts/03_eval_zeroshot.py --split test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--model", default=exlib.BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    meta = exlib.load_meta()
    evalset = exlib.load_split(args.split)

    device, dtype = exlib.pick_device_dtype()
    print(f"{args.model} (no adapter) on {device} / {dtype}")
    print(f"scoring {len(evalset)} {args.split} examples\n")

    tok = exlib.load_tokenizer(args.model)
    model = exlib.load_model(args.model)

    replies = exlib.generate_replies(model, tok, evalset, batch_size=args.batch_size)
    preds = [exlib.parse_json_reply(r) for r in replies]

    report = exlib.score_predictions(
        preds, evalset, meta["label_space"], meta["body_part_map"]
    )
    report["model"] = args.model
    report["adapter"] = None
    report["split"] = args.split
    report["max_new_tokens"] = exlib.MAX_NEW_TOKENS
    # Kept so the failure mode is inspectable rather than just a low number.
    report["sample_replies"] = replies[:5]

    path = exlib.write_report("zeroshot", report)

    print(f"JSON valid       {report['json_valid_rate']:>7.1%}")
    for field in exlib.PREDICT_FIELDS:
        s = report["fields"][field]
        print(
            f"{field:<16} {s['accuracy']:>7.1%}  macro-F1 {s['macro_f1']:.3f}  "
            f"in-label {s['in_label_space_rate']:.1%}"
        )
    print(f"{'joint':<16} {report['joint_accuracy']:>7.1%}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
