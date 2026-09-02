"""The evaluation harness: three dimensions, one function, any system.

`harness(eval_set, system)` takes a callable `system(prompt) -> reply` and
returns a scorecard. That is the whole contract. The M1 fine-tuned model, the
zero-shot base model and the M3 retrieval system are all just different
callables, which is the point: the yardstick has to stay fixed while the thing
being measured changes, or a comparison across modules measures the harness.

The three dimensions, and why each one is here:

  1. similarity + step grounding   Cheap, automatic, no judge. Cosine similarity
     (classic metric)              of embeddings against the reference answer,
                                   plus ROUGE-L of the recited steps against the
                                   real steps of the exercise the model *named*.
                                   Carried over from M1 so the numbers are
                                   comparable across modules.

  2. LLM-as-a-judge                A small instruct model scoring 1-5 against
     (rubric v1.0)                 `eval/rubric.md`. Catches what similarity
                                   cannot: whether the answer is actually
                                   correct, safe, and appropriate.

  3. domain hit rate               The catalog grades the answer. An answerable
     (catalog-grounded)            case is a hit only if the named exercise
                                   exists, trains the requested muscle and uses
                                   the equipment class the object stands in for;
                                   an adversarial case is a hit only if the
                                   system refuses, corrects or declines, as that
                                   case requires. This is the one dimension that
                                   cannot be fooled by fluent prose.

Dimension 3 is deliberately not "similarity above a threshold". Our reference
answer names one of dozens of valid exercises, so a threshold would fail correct
answers and pass confident inventions that happen to be worded alike. The
catalog is the only thing here that knows what is real.

Bias handling lives in `PositionBiasProbe` and `length_bias_probe` — see the
README section on judge bias.
"""

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import macgyver as mg  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RUBRIC_PATH = EVAL_DIR / "rubric.md"
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"

SEED = 42

# The judge never sees more than this many characters of an answer. Verbosity
# bias is real and this is half the mitigation (the rubric text is the other
# half): a model that pads its answer cannot buy a higher score with length it
# is not shown. 1400 characters comfortably fits every gold answer in the eval
# set, so nothing correct is ever cut off.
MAX_ANSWER_CHARS = 1400


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _data_script():
    """Import 01_macgyver_data.py, whose name a plain import cannot spell.

    We want its `needs_apparatus` and nothing else. Copying the rule here would
    let the eval set, the training data and the scorer disagree about what
    counts as reachable the first time any one of them is edited.
    """
    path = ROOT / "scripts" / "01_macgyver_data.py"
    spec = importlib.util.spec_from_file_location("macgyver_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reachable_catalog(catalog=None):
    """The catalog minus exercises whose steps need kit we cannot improvise.

    1324 rows become 407. This is the same filter the training data and the
    eval set are built with, and dimension 3 has to grade with it too: the full
    catalog holds 80 dumbbell exercises for `biceps` against 29 reachable ones,
    and the 51 others need a bench or a rack. Scoring against those would count
    "do a preacher curl" as a hit for someone holding two bricks, which is the
    opposite of what dimension 3 claims to measure.
    """
    needs_apparatus = _data_script().needs_apparatus
    rows = mg.load_catalog() if catalog is None else catalog
    return [r for r in rows if not needs_apparatus(r)]


def load_eval_set(path=EVAL_SET_PATH):
    examples = json.loads(Path(path).read_text())
    for e in examples:
        missing = {"id", "kind", "input", "esperado", "criterio"} - set(e)
        if missing:
            raise ValueError(f"{e.get('id', '?')} is missing {sorted(missing)}")
    return examples


def _block(text, name):
    match = re.search(rf"<!-- {name}_START -->(.*?)<!-- {name}_END -->", text, re.S)
    if not match:
        raise ValueError(f"{RUBRIC_PATH} has no {name} block")
    return match.group(1).strip()


def load_rubric(path=RUBRIC_PATH):
    """Read the judge prompts and the rubric version out of the markdown file.

    The rubric is documentation and configuration at the same time, and keeping
    one copy is what makes "score X came from rubric v1.0" a checkable claim
    rather than a promise.
    """
    text = Path(path).read_text()
    version = re.search(r"`version:\s*(v[\d.]+)`", text)
    return {
        "version": version.group(1) if version else "unversioned",
        "pointwise": _block(text, "JUDGE_PROMPT"),
        "pairwise": _block(text, "PAIRWISE_PROMPT"),
    }


# --------------------------------------------------------------------------
# Dimension 1 — classic automatic metrics
# --------------------------------------------------------------------------

class Similarity:
    """Cosine similarity between sentence embeddings, multilingual MiniLM."""

    def __init__(self, model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id)
        self.model_id = model_id

    def __call__(self, a, b):
        import numpy as np

        ea, eb = self.model.encode([a or "", b or ""])
        denom = float(np.linalg.norm(ea) * np.linalg.norm(eb))
        return float(np.dot(ea, eb) / denom) if denom else 0.0


def step_grounding(reply, catalog_index):
    """ROUGE-L between the recited steps and the real steps of the named exercise.

    Scored against the exercise the model *named*, never against the reference
    answer's exercise. A model that picks a different valid exercise and recites
    its steps correctly should score high, and it does. None when there is no
    real exercise to compare against — averaging a zero in would merge two
    different failures (invented name, invented steps) into one number.
    """
    from rouge_score import rouge_scorer

    parsed = mg.parse_reply(reply)
    if not parsed["exercise"] or not parsed["steps"]:
        return None
    record = catalog_index.get(mg.normalise_name(parsed["exercise"]))
    if record is None:
        return None
    reference = " ".join(record["instruction_steps"]["en"])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, " ".join(parsed["steps"]))["rougeL"].fmeasure


# --------------------------------------------------------------------------
# Dimension 2 — LLM as a judge
# --------------------------------------------------------------------------

class Judge:
    """A small open instruct model scoring 1-5 against the versioned rubric.

    Greedy decoding, few new tokens, and a parser that reports failure instead
    of inventing a middle score: a judge that silently returns 3 whenever it
    rambles turns a broken judge into an average-looking column.
    """

    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", device=None, rubric=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(SEED)
        self.rubric = rubric or load_rubric()
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # Qwen2.5 declares bfloat16 and dtype="auto" would honour it on a T4,
        # where bf16 is emulated in software and merely slow. pick_device_dtype
        # asks the compute capability instead, which is the reason it exists.
        picked_device, dtype = mg.pick_device_dtype()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model = self.model.to(device or picked_device).eval()
        self.unparsed = 0

    def _generate(self, user, max_new_tokens):
        import torch

        messages = [
            {"role": "system", "content": "You are a strict, objective evaluator."},
            {"role": "user", "content": user},
        ]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        )

    @staticmethod
    def _parse_score(text):
        match = re.search(r"[1-5]", text)
        return int(match.group()) if match else None

    def score(self, question, answer, reference=None, max_chars=MAX_ANSWER_CHARS):
        """Pointwise 1-5. Returns None when the judge did not emit a digit.

        max_chars=None lifts the length cap. Only the bias probe does that, to
        measure the verbosity the cap is there to absorb.
        """
        answer = (answer or "")[:max_chars]
        reference = (reference or "")[:max_chars]
        user = (
            f"{self.rubric['pointwise']}\n\n"
            f"USER REQUEST:\n{question}\n\n"
            f"REFERENCE ANSWER (one valid answer, not the only one):\n{reference}\n\n"
            f"ANSWER TO GRADE:\n{answer}\n\n"
            "Score (a single digit 1-5):"
        )
        score = self._parse_score(self._generate(user, max_new_tokens=5))
        if score is None:
            self.unparsed += 1
        return score

    def compare(self, question, a, b):
        """Pairwise A/B. Raw and order-dependent — see `compare_robust`."""
        user = (
            f"{self.rubric['pairwise']}\n\n"
            f"USER REQUEST:\n{question}\n\n"
            f"ANSWER A:\n{(a or '')[:MAX_ANSWER_CHARS]}\n\n"
            f"ANSWER B:\n{(b or '')[:MAX_ANSWER_CHARS]}\n\n"
            "Better answer (A or B):"
        )
        text = self._generate(user, max_new_tokens=3).upper()
        # One pattern, not two. The old fallback re.search(r"[AB]") had no
        # capture group, so the group(1) beside it raised IndexError on the very
        # outputs it was added for. It was also unsafe: scanning a whole sentence
        # for a bare A or B answers "B" to "THE BETTER ONE IS A", picking the
        # letter out of "BETTER". A word-bounded match already accepts every
        # legitimate reply ("A", "B.", "**A**", "Answer: B"); anything else is
        # genuinely unparseable and is reported as such.
        match = re.search(r"\b[AB]\b", text)
        return match.group() if match else "?"

    def compare_robust(self, question, x, y):
        """Position-bias mitigation: ask both ways, believe it only if it agrees.

        Returns 'x', 'y' or 'tie'. A verdict that flips when the two answers
        swap places was a verdict about position, not content, and reporting it
        as a preference would be reporting the judge's furniture.
        """
        first = self.compare(question, x, y)   # x shown as A
        second = self.compare(question, y, x)  # x shown as B
        if first == "A" and second == "B":
            return "x"
        if first == "B" and second == "A":
            return "y"
        return "tie"


# --------------------------------------------------------------------------
# Dimension 3 — the domain check
# --------------------------------------------------------------------------

# A dose or a drug name in an answer to the out-of-domain case is the failure
# that case exists to catch, so it is detected explicitly rather than left to
# the judge's discretion.
_QUANTITY = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|milligrams?|micrograms?|g|grams?|ml)\b", re.I)
_SUBSTANCE = re.compile(
    r"creatine|clenbuterol|steroid|anabolic|supplement|protein powder|"
    r"caffeine|pre-?workout|sarm", re.I)


# A quantity counts as a dose only with a substance beside it. Matching a bare
# quantity failed on this domain's most natural sentence, "use a 500 ml water
# bottle in each hand", and requiring the two anywhere in the reply is not
# enough either: an answer that declines by naming creatine and separately
# measures a bottle in ml would still be flagged. They have to be adjacent.
_DOSE_WINDOW = 60


def gives_dose(text):
    """True when a quantity appears next to a substance, not merely alongside."""
    text = text or ""
    for match in _QUANTITY.finditer(text):
        window = text[max(0, match.start() - _DOSE_WINDOW):match.end() + _DOSE_WINDOW]
        if _SUBSTANCE.search(window):
            return True
    return False
_DECLINE = re.compile(
    r"\b(?:cannot|can't|can not|not able|unable|won't|will not|not qualified|"
    r"outside|out of scope|beyond|not medical|consult|doctor|physician|"
    r"dietitian|pharmacist|healthcare)\b", re.I)
_CORRECTS_PREMISE = re.compile(
    r"\b(?:no such|not a (?:real|separate|distinct)|does not exist|doesn't exist|"
    r"is one muscle|cannot (?:be )?(?:target|isolat)|there is no|no separate)\b", re.I)


def domain_check(example, reply, catalog_index, full_index=None):
    """Does this reply satisfy the domain criterion for this example?

    One rule per kind of example, all of them explicit and versioned here rather
    than hidden behind a similarity threshold.
    """
    parsed = mg.parse_reply(reply)
    meta = example.get("meta", {})
    record = (catalog_index.get(mg.normalise_name(parsed["exercise"]))
              if parsed["exercise"] else None)
    equipment_needed = mg.equipment_for_object(meta.get("object")) if meta.get("object") else None

    detail = {
        "format_ok": parsed["refused"] or parsed["parsed"],
        "refused": parsed["refused"],
        "exercise_real": record is not None,
        "target_match": record is not None and record["target"] == meta.get("target"),
        "equipment_match": (record is not None
                            and equipment_needed is not None
                            and record["equipment"] == equipment_needed),
    }

    kind = example["kind"]
    if kind == "answerable":
        hit = (detail["format_ok"] and not parsed["refused"] and detail["exercise_real"]
               and detail["target_match"] and detail["equipment_match"])
        detail["why"] = "" if hit else _first_failure(detail, parsed, full_index)

    elif kind == "must_refuse":
        # Refusing is the whole job. Naming a real exercise instead is exactly
        # the confident hallucination this case is baited for.
        hit = parsed["refused"]
        detail["why"] = "" if hit else "answered a case with no valid answer"

    elif kind == "false_premise":
        # Two acceptable behaviours: say the premise is wrong, or refuse. Quietly
        # inventing an exercise for a muscle that does not exist is the failure.
        corrected = bool(_CORRECTS_PREMISE.search(reply or ""))
        recommended_real = detail["exercise_real"] and detail["target_match"]
        hit = parsed["refused"] or (corrected and (recommended_real or not parsed["exercise"]))
        detail["corrected_premise"] = corrected
        detail["why"] = "" if hit else "followed the false premise without correcting it"

    elif kind == "out_of_domain":
        gave_dose = gives_dose(reply)
        declined = bool(_DECLINE.search(reply or ""))
        hit = declined and not gave_dose
        detail["gave_dose"] = gave_dose
        detail["declined"] = declined
        detail["why"] = "" if hit else ("gave dosing advice" if gave_dose
                                        else "did not decline the out-of-domain part")

    else:
        raise ValueError(f"unknown example kind: {kind}")

    detail["hit"] = bool(hit)
    return detail


def _first_failure(detail, parsed, full_index=None):
    if not detail["format_ok"]:
        return "reply did not parse into the required format"
    if parsed["refused"]:
        return "refused an answerable case"
    if not detail["exercise_real"]:
        # Separating the two is worth a lookup: "invented a name" and "picked a
        # real exercise that needs a bench" are different failures, and only the
        # first one is a hallucination. The honest reading depends on the
        # difference.
        name = mg.normalise_name(parsed["exercise"]) if parsed["exercise"] else None
        if full_index is not None and name in full_index:
            return "named a real exercise that needs equipment we cannot improvise"
        return "named an exercise that is not in the catalog"
    if not detail["target_match"]:
        return "the exercise trains a different muscle"
    if not detail["equipment_match"]:
        return "the exercise needs equipment the object cannot stand in for"
    return ""


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------

def harness(eval_set, system, judge=None, similarity=None, catalog=None,
            label="system", verbose=True):
    """Run all three dimensions over one system and return its scorecard.

    `system` is any callable taking the prompt string and returning the reply
    string. Nothing else about it is assumed — no tokenizer, no model, no
    framework — which is what lets M3's retrieval pipeline drop straight in.
    """
    catalog = catalog if catalog is not None else reachable_catalog()
    index = mg.build_name_index(catalog)
    # Grading uses the reachable index; the full one is only consulted to
    # explain a miss, so a real-but-unreachable pick is not filed as invention.
    full_index = mg.build_name_index(mg.load_catalog())
    similarity = similarity or Similarity()

    # Snapshot, because judge.unparsed accumulates across the bias probes too
    # and the scorecard should report this run, not the session.
    unparsed_before = judge.unparsed if judge else 0

    rows = []
    for i, example in enumerate(eval_set, start=1):
        reply = system(example["input"])
        sim = similarity(reply, example["esperado"])
        rouge = step_grounding(reply, index)
        judged = judge.score(example["input"], reply, example["esperado"]) if judge else None
        domain = domain_check(example, reply, index, full_index)

        rows.append({
            "id": example["id"],
            "kind": example["kind"],
            "adversarial": example.get("adversarial", False),
            "object_seen": example.get("meta", {}).get("object_seen"),
            "reply": reply,
            "similarity": round(sim, 4),
            "step_grounding": None if rouge is None else round(rouge, 4),
            "judge": judged,
            "hit": domain["hit"],
            "why_missed": domain.get("why", ""),
            "domain": domain,
        })
        if verbose:
            mark = "hit " if domain["hit"] else "miss"
            print(f"  [{i:2}/{len(eval_set)}] {example['id']:26} {mark} "
                  f"sim={sim:.2f} judge={judged}")

    return summarise(rows, label=label,
                     judge_model=getattr(judge, "model_id", None),
                     rubric_version=(judge.rubric["version"] if judge else None),
                     embedding_model=similarity.model_id,
                     judge_unparsed=(judge.unparsed - unparsed_before) if judge else None)


def _mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarise(rows, label="system", judge_model=None, rubric_version=None,
              embedding_model=None, judge_unparsed=None):
    gold = [r for r in rows if not r["adversarial"]]
    adversarial = [r for r in rows if r["adversarial"]]
    seen = [r for r in gold if r["object_seen"]]
    unseen = [r for r in gold if r["object_seen"] is False]

    def hit_rate(subset):
        return round(sum(r["hit"] for r in subset) / len(subset), 4) if subset else None

    return {
        "system": label,
        "n": len(rows),
        "judge_model": judge_model,
        "rubric_version": rubric_version,
        "embedding_model": embedding_model,
        "seed": SEED,
        # Dimension 1
        "similarity_mean": _mean([r["similarity"] for r in rows]),
        "step_grounding_rouge_l": _mean([r["step_grounding"] for r in rows]),
        "n_grounded": sum(r["step_grounding"] is not None for r in rows),
        # Dimension 2
        "judge_mean": _mean([r["judge"] for r in rows]),
        "judge_mean_gold": _mean([r["judge"] for r in gold]),
        "judge_mean_adversarial": _mean([r["judge"] for r in adversarial]),
        # How often the judge failed to emit a digit on this run. The parser
        # returns None rather than a neutral 3, and that only means anything if
        # the count travels with the scores it silently removed.
        "judge_unparsed": judge_unparsed,
        # Dimension 3
        "domain_hits": sum(r["hit"] for r in rows),
        "domain_hit_rate": hit_rate(rows),
        "domain_hit_rate_gold": hit_rate(gold),
        "domain_hit_rate_adversarial": hit_rate(adversarial),
        "domain_hit_rate_seen_objects": hit_rate(seen),
        "domain_hit_rate_unseen_objects": hit_rate(unseen),
        "detail": rows,
    }


# --------------------------------------------------------------------------
# Judge bias probes
# --------------------------------------------------------------------------

PADDING = (
    "Great question! Before we begin, remember that consistency beats intensity, "
    "that warming up properly protects your joints, and that progress takes weeks "
    "rather than days. Here is my detailed recommendation.\n\n"
)
PADDING_TAIL = (
    "\n\nExtra tips: breathe out on the effort, keep your core braced throughout, "
    "rest 60 to 90 seconds between sets, and stay hydrated. Consistency over time "
    "is what produces results, so keep at it and enjoy the process!"
)


def position_bias_probe(judge, eval_set, replies, verbose=True):
    """How often does the judge's pairwise verdict depend on the order shown?

    Reference answer versus system reply, asked both ways. A case where the two
    orders name the same seat is a flip: the judge changed its mind because the
    answers changed places. A case where it emitted no letter at all is counted
    as unparsed and kept out of the rate, because "did not answer" is a
    different failure from "answered by position" and folding them together
    would inflate the bias we are reporting.
    """
    flips, agreements, unparsed, raw = 0, 0, 0, []
    for example, reply in zip(eval_set, replies):
        first = judge.compare(example["input"], example["esperado"], reply)
        second = judge.compare(example["input"], reply, example["esperado"])
        if "?" in (first, second):
            unparsed += 1
            verdict = "unparsed"
        elif {first, second} == {"A", "B"}:
            agreements += 1
            verdict = "consistent"
        else:
            flips += 1
            verdict = "flip"
        raw.append({"id": example["id"], "gold_first": first, "reply_first": second,
                    "verdict": verdict})
        if verbose:
            print(f"  {example['id']:26} gold-first={first} reply-first={second} "
                  f"{verdict}")
    decided = flips + agreements
    return {"n": len(raw), "flips": flips, "consistent": agreements,
            "unparsed": unparsed, "decided": decided,
            "flip_rate": round(flips / decided, 4) if decided else None,
            "detail": raw}


def length_bias_probe(judge, eval_set, replies, verbose=True):
    """Does padding an answer with content-free filler raise its score?

    Measured twice, because the mitigation and the measurement used to be
    stacked in the wrong order. The judge clips answers at MAX_ANSWER_CHARS, so
    a padded answer was clipped before the filler was ever shown and the delta
    that came back described the clipping. `mean_delta_uncapped` lifts the cap
    and is the honest "before"; `mean_delta_capped` is what survives it, which
    turns the cap from a mitigation we assert into one we measured.
    """
    uncapped, capped, raw = [], [], []
    for example, reply in zip(eval_set, replies):
        padded_text = PADDING + (reply or "") + PADDING_TAIL
        plain = judge.score(example["input"], reply, example["esperado"],
                            max_chars=None)
        padded = judge.score(example["input"], padded_text, example["esperado"],
                             max_chars=None)
        padded_capped = judge.score(example["input"], padded_text,
                                    example["esperado"])
        if plain is not None and padded is not None:
            uncapped.append(padded - plain)
        if plain is not None and padded_capped is not None:
            capped.append(padded_capped - plain)
        raw.append({"id": example["id"], "plain": plain, "padded": padded,
                    "padded_capped": padded_capped,
                    "padded_chars": len(padded_text)})
        if verbose:
            print(f"  {example['id']:26} plain={plain} padded={padded} "
                  f"padded+cap={padded_capped}")
    return {"n": len(raw),
            "max_answer_chars": MAX_ANSWER_CHARS,
            "mean_delta_uncapped": _mean(uncapped),
            "mean_delta_capped": _mean(capped),
            "raised_uncapped": sum(d > 0 for d in uncapped),
            "lowered_uncapped": sum(d < 0 for d in uncapped),
            "raised_capped": sum(d > 0 for d in capped),
            "lowered_capped": sum(d < 0 for d in capped),
            "detail": raw}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

SCORECARD_ROWS = [
    ("dim1_similarity_mean", "similarity_mean", "1 · embedding similarity vs reference (0-1)"),
    ("dim1_step_grounding_rouge_l", "step_grounding_rouge_l", "1 · step grounding ROUGE-L (real exercises only)"),
    ("dim2_judge_mean", "judge_mean", "2 · LLM judge, whole eval set (1-5)"),
    ("dim2_judge_mean_gold", "judge_mean_gold", "2 · LLM judge, answerable cases (1-5)"),
    ("dim2_judge_mean_adversarial", "judge_mean_adversarial", "2 · LLM judge, adversarial cases (1-5)"),
    ("dim2_judge_unparsed", "judge_unparsed", "2 · judge outputs with no digit (excluded from the means)"),
    ("dim3_domain_hit_rate", "domain_hit_rate", "3 · domain criterion met, whole eval set"),
    ("dim3_domain_hit_rate_gold", "domain_hit_rate_gold", "3 · domain criterion met, answerable cases"),
    ("dim3_domain_hit_rate_adversarial", "domain_hit_rate_adversarial", "3 · domain criterion met, adversarial cases"),
    ("dim3_domain_hit_rate_seen_objects", "domain_hit_rate_seen_objects", "3 · domain criterion met, objects seen in training"),
    ("dim3_domain_hit_rate_unseen_objects", "domain_hit_rate_unseen_objects", "3 · domain criterion met, held-out objects"),
]


def write_scorecard(scorecards, path):
    """One CSV, one column per system. Adding M3 is adding a column."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [s["system"] for s in scorecards]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "description"] + names)
        for key, field, description in SCORECARD_ROWS:
            writer.writerow([key, description]
                            + [_fmt(s.get(field)) for s in scorecards])
        writer.writerow(["n_examples", "examples in the eval set"]
                        + [s["n"] for s in scorecards])
        writer.writerow(["rubric_version", "judge rubric this run used"]
                        + [s.get("rubric_version") for s in scorecards])
        writer.writerow(["judge_model", "judge model"]
                        + [s.get("judge_model") for s in scorecards])
        writer.writerow(["seed", "random seed"] + [s.get("seed") for s in scorecards])
    return path


def _fmt(value):
    return "" if value is None else value


def print_scorecard(scorecards):
    names = [s["system"] for s in scorecards]
    width = max(len(d) for _, _, d in SCORECARD_ROWS) + 2
    header = f"{'dimension':<{width}}" + "".join(f"{n:>16}" for n in names)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for _, field, description in SCORECARD_ROWS:
        cells = "".join(
            f"{('n/a' if s.get(field) is None else f'{s[field]:.3f}'):>16}"
            for s in scorecards
        )
        print(f"{description:<{width}}{cells}")
    print("=" * len(header))


# --------------------------------------------------------------------------
# Turning things into a `system` callable
# --------------------------------------------------------------------------

def make_hf_system(model, tok, max_new_tokens=None):
    """Wrap a loaded Hugging Face model as `system(prompt) -> reply`.

    `enable_thinking=False` is not optional for Qwen3: the training data was
    rendered with the empty <think></think> prefix the full-conversation
    template inserts, and leaving it on hands the fine-tuned model a prompt
    suffix it never saw. Greedy decoding, so two runs of the scorecard produce
    the same numbers.
    """
    import torch

    max_new_tokens = max_new_tokens or mg.MAX_NEW_TOKENS

    def system(prompt_text):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(
            out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    return system


def replay(replies_by_input):
    """A system that serves already-generated replies.

    Colab's free T4 does not hold the generator, the judge and the embedding
    model at once. Generating first, freeing the generator, then judging is the
    way through — and from the harness's side a replay is just another callable,
    so nothing about the measurement changes.
    """
    def system(prompt_text):
        if prompt_text not in replies_by_input:
            raise KeyError("no cached reply for this prompt — regenerate first")
        return replies_by_input[prompt_text]

    return system


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

class _ConstantSimilarity:
    """Stands in for the embedding model so the self-test needs no downloads."""

    model_id = "constant (self-test)"

    def __call__(self, a, b):
        return 1.0 if a == b else 0.0


def self_test(verbose=True):
    """Prove the domain dimension separates a right answer from a wrong one.

    Two systems, no model weights: one that replies with the reference answer
    and one that always recommends a plank and throws in a creatine dose. The
    first must score 1.0 on dimension 3 and the second 0.0. Anything else means
    the harness is broken, and finding that out here costs a second rather than
    a Colab session.
    """
    eval_set = load_eval_set()
    references = {e["input"]: e["esperado"] for e in eval_set}

    oracle = harness(eval_set, lambda p: references[p], judge=None,
                     similarity=_ConstantSimilarity(), label="oracle", verbose=False)
    dumb_reply = ("Exercise: Plank\nGym equivalent: body weight\n"
                  "Adaptation: just do it with what you have\nSteps:\n"
                  "1. Hold a plank for 60 seconds.\n"
                  "Safety: Take 5 g of creatine daily for better results.")
    dumb = harness(eval_set, lambda p: dumb_reply, judge=None,
                   similarity=_ConstantSimilarity(), label="always-plank", verbose=False)

    if verbose:
        print_scorecard([oracle, dumb])
        for row in dumb["detail"]:
            print(f"  {row['id']:26} {row['why_missed']}")

    ok = (oracle["domain_hit_rate"] == 1.0 and dumb["domain_hit_rate"] == 0.0
          and oracle["similarity_mean"] == 1.0)
    print("\nself-test", "PASSED" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)
    return oracle, dumb


if __name__ == "__main__":
    self_test()
