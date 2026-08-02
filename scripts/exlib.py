"""Shared pieces for the M1 extraction experiments.

Everything here is imported by both the CLI scripts and the Colab notebook.
The reason this module exists is `build_prompt`: the training data, the
zero-shot baseline and the fine-tuned evaluation must all show the model the
exact same string, or the reported delta measures prompt drift instead of
fine-tuning. One definition, three call sites.

The numbered scripts can't be imported (module names can't start with a
digit), so anything two of them need lives here.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data/exercises-dataset/data/exercises.json"
PROCESSED = ROOT / "data/processed"
REPORTS = ROOT / "reports"

# The two fields the model actually predicts.
#
# `body_part` is deliberately absent: it is a strict function of `target`
# (19 targets -> 10 body parts, verified in build_body_part_map), and
# `category` is a verbatim copy of it. Asking the model to emit it would add
# a third accuracy column that is free whenever `target` is right, which
# flatters the results without measuring anything.
PREDICT_FIELDS = ["target", "equipment"]
DERIVED_FIELD = "body_part"

SEED = 42


def load_catalog(path=CATALOG):
    """Read the exercise catalog from the pinned dataset submodule."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The dataset is a git submodule:\n"
            "    git submodule update --init --recursive"
        )
    return json.loads(path.read_text())


def build_label_space(records, fields=PREDICT_FIELDS):
    """Sorted set of valid values per field.

    Sorted, not insertion-ordered, so the prompt string is stable across runs
    and machines — otherwise the "identical prompt" guarantee above is a lie.
    """
    return {f: sorted({r[f] for r in records}) for f in fields}


def build_body_part_map(records):
    """target -> body_part, asserted to be single-valued.

    The whole reason `body_part` is derived rather than predicted. If a future
    dataset bump introduces a target that spans two body parts, this raises
    instead of silently picking one and quietly corrupting the derived column.
    """
    seen = defaultdict(set)
    for r in records:
        seen[r["target"]].add(r[DERIVED_FIELD])

    ambiguous = {t: sorted(v) for t, v in seen.items() if len(v) > 1}
    if ambiguous:
        raise ValueError(
            "target -> body_part is no longer a function; body_part can no "
            f"longer be derived and must be predicted: {ambiguous}"
        )
    return {t: v.pop() for t, v in seen.items()}


def build_prompt(record, label_space, input_mode="both"):
    """The one and only prompt the model ever sees.

    The valid values for every field are listed in the prompt on purpose. The
    same prompt grades the untrained model in 03, and a base model that has
    never seen our label space would otherwise be marked down for vocabulary
    it was never shown — which measures our labelling conventions, not the
    model.
    """
    if input_mode == "name":
        body = f"Name: {record['name']}"
    elif input_mode == "instructions":
        body = f"Instructions: {record['instructions']['en']}"
    else:
        body = f"Name: {record['name']}\nInstructions: {record['instructions']['en']}"

    allowed = "\n".join(f"{f}: {' | '.join(label_space[f])}" for f in PREDICT_FIELDS)
    return (
        "Extract the structured fields for this exercise.\n\n"
        f"{body}\n\n"
        "Reply with JSON only, choosing from these exact values:\n"
        f"{allowed}"
    )


def gold_completion(record):
    """The target JSON string for a record."""
    return json.dumps({f: record[f] for f in PREDICT_FIELDS})


def load_meta():
    """Read meta.json written by 02_prepare_data.py."""
    path = PROCESSED / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build the splits first:\n"
            "    uv run python scripts/02_prepare_data.py"
        )
    return json.loads(path.read_text())


def load_split(name):
    """Read one split as a list of records."""
    path = PROCESSED / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build the splits first:\n"
            "    uv run python scripts/02_prepare_data.py"
        )
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_reply(text):
    """Pull a field dict out of whatever the model actually emitted.

    Returns None when nothing parseable comes back. That None is a result, not
    an error: the rate at which the untrained model fails to produce JSON is
    one of the clearest things fine-tuning fixes, so it gets counted rather
    than repaired. The cleanup here is limited to unwrapping — stripping
    reasoning blocks and code fences — and never guesses at field values.
    """
    if not text:
        return None

    text = _THINK.sub("", text)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None

    # Scan for the brace that closes the first object, ignoring braces that
    # appear inside string literals.
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def derive_body_part(target, body_part_map):
    """Look up body_part from a predicted target. None if the target is invalid."""
    return body_part_map.get(target)


def score_predictions(preds, golds, label_space, body_part_map):
    """Metrics for a list of predicted field dicts against gold records.

    `preds` entries are dicts from parse_json_reply, or None where the model
    produced no parseable JSON. Those Nones count as wrong everywhere — they
    are not dropped, because silently excluding unparseable outputs would let
    a model that answers 40% of the time outscore one that always answers.

    macro_f1 is computed over the classes present in the gold labels of this
    split. Classes the split never asks about would otherwise contribute a
    zero to the mean and make every model look equally bad on the tail.
    """
    from sklearn.metrics import f1_score

    n = len(golds)
    if n == 0:
        raise ValueError("nothing to score")
    if len(preds) != n:
        raise ValueError(f"{len(preds)} predictions for {n} gold records")

    # Sentinel for "no answer" / "unparseable". Not a member of any label
    # space, so it can never accidentally match a gold value.
    MISSING = "\x00missing"

    report = {
        "n": n,
        "json_valid_rate": round(sum(p is not None for p in preds) / n, 4),
        "fields": {},
    }

    def answered(pred, field):
        """The model's answer for one field, or MISSING.

        A model can emit valid JSON that omits a field, or answers it with a
        list or a number. All of those are non-answers, not near-misses.
        """
        value = pred.get(field) if isinstance(pred, dict) else None
        return value if isinstance(value, str) and value else MISSING

    per_field_correct = []
    for field in PREDICT_FIELDS:
        y_true = [g[field] for g in golds]
        y_pred = [answered(p, field) for p in preds]

        correct = [t == p for t, p in zip(y_true, y_pred)]
        per_field_correct.append(correct)

        labels = sorted(set(y_true))
        valid = set(label_space[field])
        report["fields"][field] = {
            "accuracy": round(sum(correct) / n, 4),
            "macro_f1": round(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
                4,
            ),
            # How often the model answered with a value that exists in our
            # label space at all — separates "wrong class" from "invented a
            # class we never showed it".
            "in_label_space_rate": round(sum(p in valid for p in y_pred) / n, 4),
        }

    report["joint_accuracy"] = round(
        sum(all(c) for c in zip(*per_field_correct)) / n, 4
    )

    # Derived, not learned. Reported so the pipeline's real output is visible,
    # and labelled in the key so nobody reads it as a third trained field.
    derived_correct = sum(
        derive_body_part(answered(p, "target"), body_part_map) == g[DERIVED_FIELD]
        for p, g in zip(preds, golds)
    )
    report["derived_body_part_accuracy"] = round(derived_correct / n, 4)

    return report


def write_report(name, payload):
    """Write a report to reports/<name>.json and return the path."""
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


BASE_MODEL = "Qwen/Qwen3-1.7B"

# The gold answer is ~15 tokens. 64 is a deliberate, equal budget for both the
# untrained and the fine-tuned model: a model that cannot produce a 15-token
# JSON object within 64 tokens has failed the task, and truncated output
# counts as invalid rather than being quietly retried with more room.
MAX_NEW_TOKENS = 64


def load_tokenizer(model_id=BASE_MODEL):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id=BASE_MODEL, adapter=None, dtype=None):
    """Load the base model, optionally with a LoRA adapter merged on top."""
    import torch
    from transformers import AutoModelForCausalLM

    _, default_dtype = pick_device_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype or default_dtype, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        # Merging folds the adapter into the base weights so generation runs
        # at base-model speed instead of through the LoRA side path.
        model = model.merge_and_unload()

    model.eval()
    torch.set_grad_enabled(False)
    return model


def generate_replies(model, tok, records, batch_size=16, show_progress=True):
    """Greedy-decode one reply per record.

    Greedy, not sampled: the task has one correct answer, and temperature
    would add run-to-run variance to a number we are asking a reader to
    compare across three rows of a table.

    `enable_thinking=False` matters more than it looks. Qwen3's template
    always inserts an empty `<think></think>` block ahead of assistant content
    when rendering a full conversation, which is what the model is trained on
    here; at generation time only `enable_thinking=False` reproduces that same
    prefix. Leaving it True would end the prompt at `assistant\\n` and hand the
    fine-tuned model a suffix it never saw during training.
    """
    import torch

    original_side = tok.padding_side
    tok.padding_side = "left"  # required for batched decoder generation
    replies = []
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [
                tok.apply_chat_template(
                    r["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for r in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            for i in range(len(batch)):
                new_tokens = out[i][enc["input_ids"].shape[1] :]
                replies.append(tok.decode(new_tokens, skip_special_tokens=True))
            if show_progress:
                done = min(start + batch_size, len(records))
                print(f"  generated {done}/{len(records)}", end="\r", flush=True)
    finally:
        tok.padding_side = original_side

    if show_progress:
        print()
    return replies


def pick_device_dtype():
    """bf16 where the GPU has bf16 *hardware*, fp16 otherwise.

    Colab's free tier hands out a T4 (Turing, sm_75), which has no bf16 units.
    TRL's SFTConfig defaults bf16=True whenever fp16 is unset, so precision has
    to be chosen from the actual device. Every script and the notebook route
    through this, so a local run and the Colab run differ in precision only,
    never in code path.

    The obvious check, `torch.cuda.is_bf16_supported()`, is a trap: it returns
    True on a T4, because it also counts bf16 that torch can *emulate* in
    software. A Colab run confirmed it — the notebook reported
    `torch.bfloat16` on a `Tesla T4 (sm_75)`. Emulated bf16 runs, slowly and
    with no tensor-core path, which is the worst outcome: no crash to tell you
    it is wrong.

    Compute capability is unambiguous. Hardware bf16 arrived with Ampere
    (sm_80), so major >= 8 is the real question, and it is what we ask.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu", torch.float32

    major, _ = torch.cuda.get_device_capability()
    return "cuda", torch.bfloat16 if major >= 8 else torch.float16
