"""How badly does each candidate tokenizer fragment our domain vocabulary?

Week 3's Lab A advice: before committing to a base model, look at how it
tokenizes the terms your task actually turns on. A model that shreds
`levator scapulae` into eight subwords has to spend capacity reassembling a
label it must emit verbatim, and every one of those tokens is a place the
generation can go wrong.

That matters more here than usual, because the labels are not just inputs —
they are the output. The model has to reproduce all 19 target values and 28
equipment values exactly, so label fertility is a direct measure of how hard
the output space is.

The candidates are one per architecture family, which is the choice the
assignment asks us to justify, and all of them are ungated so this study
reruns without a Hugging Face login.

Run:
    uv run python scripts/00_tokenizer_check.py
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exlib

CANDIDATES = [
    ("Qwen/Qwen3-1.7B", "decoder", "chosen"),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "decoder", "alternative"),
    ("openai-community/gpt2", "decoder", "reference BPE"),
    ("google-bert/bert-base-uncased", "encoder", "alternative"),
    ("google/flan-t5-base", "encoder-decoder", "alternative"),
]


def fertility(tokenizer, phrases):
    """Tokens per whitespace word, plus the raw token counts.

    Fertility of 1.0 means the tokenizer already knows every word in the
    phrase. Anything much above 2 means it is spelling our vocabulary out of
    fragments.
    """
    counts = [len(tokenizer.encode(p, add_special_tokens=False)) for p in phrases]
    words = [len(p.split()) for p in phrases]
    return counts, sum(counts) / sum(words)


def main():
    from transformers import AutoTokenizer

    records = exlib.load_catalog()
    label_space = exlib.build_label_space(records)

    labels = {f: label_space[f] for f in exlib.PREDICT_FIELDS}
    names = [r["name"] for r in records]
    instructions = [r["instructions"]["en"] for r in records]

    report = {"n_records": len(records), "tokenizers": {}}

    for model_id, family, role in CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception as exc:  # gated repo, no network, missing files
            print(f"{model_id}: skipped ({type(exc).__name__})")
            report["tokenizers"][model_id] = {"skipped": str(exc)[:200]}
            continue

        entry = {"family": family, "role": role, "vocab_size": len(tok), "fields": {}}

        for field, values in labels.items():
            counts, fert = fertility(tok, values)
            worst = sorted(zip(values, counts), key=lambda kv: -kv[1])[:3]
            entry["fields"][field] = {
                "n_labels": len(values),
                "mean_tokens_per_label": round(statistics.mean(counts), 2),
                "max_tokens_per_label": max(counts),
                "fertility": round(fert, 3),
                "worst": [{"label": v, "tokens": c} for v, c in worst],
            }

        name_counts, name_fert = fertility(tok, names)
        instr_counts, _ = fertility(tok, instructions)
        entry["exercise_names"] = {
            "mean_tokens": round(statistics.mean(name_counts), 2),
            "fertility": round(name_fert, 3),
        }
        # Drives max_length and therefore training cost and T4 memory.
        entry["instructions_en"] = {
            "mean_tokens": round(statistics.mean(instr_counts), 1),
            "p95_tokens": int(statistics.quantiles(instr_counts, n=20)[-1]),
            "max_tokens": max(instr_counts),
        }

        report["tokenizers"][model_id] = entry

    path = exlib.write_report("tokenizer_study", report)

    header = (
        f"{'tokenizer':<38} {'vocab':>7} {'target':>8} {'equip':>8} "
        f"{'names':>8} {'instr p95':>10}"
    )
    print(header)
    print("-" * len(header))
    for model_id, e in report["tokenizers"].items():
        if "skipped" in e:
            print(f"{model_id:<38} {'skipped':>7}")
            continue
        print(
            f"{model_id:<38} {e['vocab_size']:>7} "
            f"{e['fields']['target']['fertility']:>8.2f} "
            f"{e['fields']['equipment']['fertility']:>8.2f} "
            f"{e['exercise_names']['fertility']:>8.2f} "
            f"{e['instructions_en']['p95_tokens']:>10}"
        )
    print("\n(numbers are tokens per word — lower is better)")

    chosen = report["tokenizers"].get("Qwen/Qwen3-1.7B", {})
    if "fields" in chosen:
        print("\nworst-fragmented labels for the chosen tokenizer:")
        for field, s in chosen["fields"].items():
            worst = ", ".join(f"{w['label']} ({w['tokens']})" for w in s["worst"])
            print(f"  {field:<10} {worst}")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
