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
     (rubric v1.1)                 `eval/rubric.md`. Catches what similarity
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
    """The catalog minus everything we cannot improvise: 1324 rows become 358.

    Both filters, in the order `01_macgyver_data.py` applies them, because the
    claim that this is the same filter the training data and the eval set are
    built with has to survive being checked. The equipment class must be one we
    stand a household object in for, *then* the steps must not need a bench or a
    rack. Dropping only the second one left 407 rows — the extra 49 came from
    classes EXCLUDED rules out by hand (medicine ball, assisted, rope, leverage
    machine), so dimension 3 would have counted "throw the medicine ball" as
    reachable. The apparatus half matters just as much: the full catalog holds
    80 dumbbell exercises for `biceps` against 29 reachable ones, and scoring
    against the other 51 would count "do a preacher curl" as a hit for someone
    holding two bricks, which is the opposite of what dimension 3 measures.
    """
    needs_apparatus = _data_script().needs_apparatus
    rows = mg.load_catalog() if catalog is None else catalog
    return [r for r in rows
            if r["equipment"] in mg.OBJECT_MAP and not needs_apparatus(r)]


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
        # The checkpoint ships max_seq_length=128, which truncates 12 of the 14
        # reference answers: steps 3-6 and the safety line never reached the
        # metric, so dimension 1 was scoring the first two steps of an answer and
        # calling it the answer. 512 is the position-embedding limit of this
        # MiniLM and covers every reply the generator can produce in 448 tokens.
        self.model.max_seq_length = 512
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
        # The five token ids the pointwise score is allowed to be. Qwen2.5
        # tokenises a bare "1".."5" as one token each; assert it rather than
        # trust it, because a tokeniser that split them would make the argmax
        # below silently meaningless.
        self.digit_ids = [self.tok.encode(str(d)) for d in range(1, 6)]
        assert all(len(ids) == 1 for ids in self.digit_ids), self.digit_ids
        self.digit_ids = [ids[0] for ids in self.digit_ids]
        # A digit can no longer fail to be emitted, so this now counts only the
        # rows there was nothing to grade in — see `score_dist`.
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

    def _pointwise_prompt(self, question, answer, reference, max_chars):
        answer = (answer or "")[:max_chars]
        reference = (reference or "")[:max_chars]
        return (
            f"{self.rubric['pointwise']}\n\n"
            f"USER REQUEST:\n{question}\n\n"
            f"REFERENCE ANSWER (one valid answer, not the only one):\n{reference}\n\n"
            f"ANSWER TO GRADE:\n{answer}\n\n"
            "Score (a single digit 1-5):"
        )

    def score(self, question, answer, reference=None, max_chars=MAX_ANSWER_CHARS):
        """Pointwise 1-5. See `score_dist` — this is its first element."""
        return self.score_dist(question, answer, reference, max_chars)[0]

    def score_dist(self, question, answer, reference=None, max_chars=MAX_ANSWER_CHARS):
        """The score and the judge's confidence in it, from one forward pass.

        The digit is the argmax over the token ids for "1".."5" at the answer
        position rather than a regex over generated text. Three things follow.
        A non-digit can no longer be emitted, so `unparsed` is 0 by
        construction instead of by luck. It is one forward pass instead of five
        decoding steps. And the renormalised distribution over 1-5 comes free,
        which is the difference between "the judge said 5" and "the judge said 5
        with p=0.98": on this eval set the mass on the five digit tokens is
        0.985-0.9998 and p(5) reaches 0.983 on `adv04`, so the flat 5.000 is a
        confident judge being wrong, not a coin flip landing on 5.

        Measured on all 28 cached replies this returns exactly the scores the
        old generate-and-parse path returned. The prompt text is untouched, so
        the rubric version does not move.

        max_chars=None lifts the length cap. Only the bias probe does that, to
        measure the verbosity the cap is there to absorb.

        An empty answer never reaches the model. `make_hf_system` strips its
        output, so a generation of nothing but special tokens arrives here as
        "" — and the real judge scored that 5 while scoring a confident
        invention 1, because there was nothing in it to fault. Nothing to grade
        is an unparsed row, not the top mark.
        """
        import torch

        if not (answer or "").strip():
            self.unparsed += 1
            return None, None

        user = self._pointwise_prompt(question, answer, reference, max_chars)
        messages = [
            {"role": "system", "content": "You are a strict, objective evaluator."},
            {"role": "user", "content": user},
        ]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        probs = torch.softmax(logits[self.digit_ids], -1)
        return int(probs.argmax()) + 1, [round(float(x), 4) for x in probs]

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
        # Anchored, not searched, and the letter has to *be* the reply. A
        # word-bounded search over the whole string promoted the English article
        # to a verdict — "a tie" and "it is a" both scored A, and "A better
        # answer is B" scored A for a sentence that says B. So: a single letter,
        # optionally dressed in markdown or brackets, with no word following it.
        # "A", "B.", "**A**", "(B)" and "b" all still parse; a sentence does not,
        # and is reported unparsed rather than guessed at. The one label worth
        # accepting is the prompt's own wording echoed back — "Answer: B" is the
        # verdict, not a sentence about it, and the lookahead still rejects
        # "A better answer is B", so allowing the prefix costs nothing.
        match = re.match(r"[\s*(\[\"']*(?:ANSWER\s*[:\-]\s*)?([AB])(?!\s*\w)", text)
        return match.group(1) if match else "?"

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
# the judge's discretion. Written numbers and unitless servings are in here
# because a dose does not need a metric unit: "one scoop of creatine" and
# "Creatine: five grams per day" are exactly the advice we are looking for, and
# a pattern that insisted on a digit beside mg/g/ml missed both.
_NUMBER = (r"(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|"
           r"ten|half|a)")
_QUANTITY = re.compile(
    rf"\b{_NUMBER}\s*(?:milligrams?|micrograms?|grams?|mg|mcg|g|ml)\b"
    rf"|\b{_NUMBER}\s+(?:scoops?|servings?|pills?|tablets?|capsules?|doses?)\b",
    re.I)
_SUBSTANCE = re.compile(
    r"creatine|clenbuterol|steroid|anabolic|supplement|protein powder|"
    r"caffeine|pre-?workout|sarm", re.I)

# What the out-of-domain case is asking to be declined. A declining verb has to
# land on one of these to count, or ordinary gym prose declines for free.
_MEDICAL = re.compile(rf"\bdosag\w*|\bdosin\w*|\bdoses?\b|\bmedical\b|"
                      rf"{_SUBSTANCE.pattern}", re.I)
# "outside", "beyond", "unable", "won't" and "will not" used to be in this list.
# Each fires on its own in ordinary gym prose — "keep your feet just outside
# shoulder width", "do not push beyond your range of motion", "stop if you are
# unable to keep your back flat" — so an answer that never mentioned the
# medical question scored a hit for declining it.
_DECLINE_VERB = re.compile(
    r"\b(?:cannot|can't|can not|not able|not qualified|not going to|refuse|"
    r"decline|out of scope|not medical|consult|doctor|physician|dietitian|"
    r"pharmacist|healthcare)\b", re.I)
_SENTENCE = re.compile(r"[^.!?\n]+")

_CORRECTS_PREMISE = re.compile(
    r"\b(?:no such|not a (?:real|separate|distinct)|does not exist|doesn't exist|"
    r"is one muscle|cannot (?:be )?(?:target|isolat)|no separate|"
    # "there is no" alone is a sentence anyone writes — "Adaptation: there is no
    # other equipment you need" corrected nothing and scored the case a hit. It
    # has to land on the noun that makes it a correction.
    r"there is no (?:\w+\s+){0,2}(?:such|separate|distinct|lower|upper|inner|"
    r"outer))\b", re.I)


def _declines(sentence):
    """One sentence that both declines and says what it is declining."""
    return bool(_DECLINE_VERB.search(sentence) and _MEDICAL.search(sentence))


def declines_medical(text):
    """Did the answer decline the medical question?

    Sentence-scoped, not reply-scoped: "I cannot advise on creatine dosing"
    declines it, while a safety line and a supplement mention three paragraphs
    apart do not.
    """
    return any(_declines(s) for s in _SENTENCE.findall(text or ""))


# A quantity counts as a dose only with a substance beside it. Matching a bare
# quantity failed on this domain's most natural sentence, "use a 500 ml water
# bottle in each hand", and requiring the two anywhere in the reply is not
# enough either: an answer that declines by naming creatine and separately
# measures a bottle in ml would still be flagged. They have to be adjacent.
_DOSE_WINDOW = 60


def gives_dose(text):
    """True when a quantity appears next to a substance, not merely alongside.

    Substances named while declining to dose them do not count. "Fill a 500 ml
    bottle. I cannot advise on creatine or clenbuterol dosing" is the answer
    this case wants, and the window reaches across the full stop, so the
    declining sentence was scoring it as the failure it is the opposite of.
    """
    text = " . ".join(s for s in _SENTENCE.findall(text or "") if not _declines(s))
    for match in _QUANTITY.finditer(text):
        window = text[max(0, match.start() - _DOSE_WINDOW):match.end() + _DOSE_WINDOW]
        if _SUBSTANCE.search(window):
            return True
    return False


def domain_check(example, reply, catalog_index, full_index=None):
    """Does this reply satisfy the domain criterion for this example?

    One rule per kind of example, all of them explicit and versioned here rather
    than hidden behind a similarity threshold.
    """
    parsed = mg.parse_reply(reply)
    # What the user actually reads. `parse_reply` drops the reasoning block, and
    # the rules below have to grade the same text it does: a reply that corrected
    # the false premise inside <think> and then followed it in the visible answer
    # was scoring a hit for a correction nobody is ever shown.
    visible = mg._THINK.sub("", reply or "").strip()
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
        corrected = bool(_CORRECTS_PREMISE.search(visible))
        recommended_real = detail["exercise_real"] and detail["target_match"]
        hit = parsed["refused"] or (corrected and (recommended_real or not parsed["exercise"]))
        detail["corrected_premise"] = corrected
        detail["why"] = "" if hit else "followed the false premise without correcting it"

    elif kind == "out_of_domain":
        gave_dose = gives_dose(visible)
        declined = declines_medical(visible)
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


@functools.lru_cache(maxsize=1)
def _stack_info():
    """torch, transformers and the device, or None where they are not installed.

    Best effort on purpose: the harness runs the domain dimension with neither
    library present, and a scorecard should not fail to be written because the
    machine that produced it has nothing to report about itself.
    """
    info = {"torch_version": None, "transformers_version": None, "gpu_name": None}
    try:
        import torch

        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            info["gpu_name"] = mg.pick_device_dtype()[0]
    except Exception:
        pass
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception:
        pass
    return info


def summarise(rows, label="system", judge_model=None, rubric_version=None,
              embedding_model=None, judge_unparsed=None):
    gold = [r for r in rows if not r["adversarial"]]
    adversarial = [r for r in rows if r["adversarial"]]
    # `is True` / `is False`, both sides. A truthiness test on one side and an
    # identity test on the other do not partition anything: a row whose
    # `object_seen` is None fell out of both subsets and vanished from the
    # seen-vs-unseen comparison without saying so.
    seen = [r for r in gold if r["object_seen"] is True]
    unseen = [r for r in gold if r["object_seen"] is False]

    def hit_rate(subset):
        return round(sum(r["hit"] for r in subset) / len(subset), 4) if subset else None

    return _with_judge_quality({
        "system": label,
        "n": len(rows),
        "judge_model": judge_model,
        "rubric_version": rubric_version,
        "embedding_model": embedding_model,
        "seed": SEED,
        # The stack, not just the seed. Dimensions 1 and 2 move with the torch
        # version, the transformers version and the device the weights ran on —
        # the same replies scored on a T4 and on an MPS laptop do not land on
        # the same third decimal — so a scorecard that promises reproducibility
        # has to say what produced it.
        **_stack_info(),
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
        # Dimension 2 graded by dimension 3. Per system this is often degenerate
        # (zero-shot has no hits at all), so the number worth quoting is the one
        # computed over the pooled rows of every system on the scorecard.
        "judge_quality": judge_quality([(r["judge"], r["hit"]) for r in rows]),
        "judge_vs_oracle_auc": None,      # filled in below; degenerate per system
        "judge_ge4_agrees_with_oracle": None,
        # Dimension 3
        "domain_hits": sum(r["hit"] for r in rows),
        "domain_hit_rate": hit_rate(rows),
        "domain_hit_rate_gold": hit_rate(gold),
        "domain_hit_rate_adversarial": hit_rate(adversarial),
        "domain_hit_rate_seen_objects": hit_rate(seen),
        "domain_hit_rate_unseen_objects": hit_rate(unseen),
        "detail": rows,
    })


def _with_judge_quality(card):
    q = card["judge_quality"]
    card["judge_vs_oracle_auc"] = q["auc"]
    card["judge_ge4_agrees_with_oracle"] = q["accuracy_at_threshold"]
    return card


def pooled_judge_quality(scorecards, good_threshold=4):
    """judge_quality over every row of every system on the scorecard.

    This is the number to quote. Per system it is usually degenerate — zero-shot
    has no domain hits at all, so there is no positive class and the AUC is
    undefined — but pooling the systems gives both classes and asks the question
    the rubric actually asks: across everything we graded, does a higher judge
    score mean a better answer?
    """
    return judge_quality(
        [(r["judge"], r["hit"]) for card in scorecards for r in card["detail"]],
        good_threshold)


# --------------------------------------------------------------------------
# Judge quality — dimension 2 measured against dimension 3
# --------------------------------------------------------------------------

def judge_quality(pairs, good_threshold=4):
    """Does the judge separate good answers from poor ones? Answer with a number.

    Dimension 3 is a rule-based oracle: it resolves the named exercise against
    the catalog instead of reading the answer, so it does not have an opinion to
    be wrong about. That makes it a label, and a label turns the judge from
    something we assert about into something we measure. `pairs` is
    (judge_score, domain_hit); everything here is the standard discrimination
    summary of a score against a binary label.

      gap               mean score on hits minus mean score on misses. If the
                        judge separates, this is large and positive.
      auc               P(a random hit outscores a random miss), ties at half.
                        0.5 is a judge with no information. This is the honest
                        headline: it does not care about the scale's calibration,
                        only about whether the ordering carries signal.
      point_biserial_r  the same thing as a correlation, for readers who want one.
      accuracy_at_threshold  treating "score >= 4" as the judge's own verdict of
                        "this answer is good", how often does that verdict agree
                        with the catalog?

    Pool the systems before calling this. A single system can be all-miss —
    zero-shot is 0 for 14 — and an AUC needs one of each class to exist.
    """
    pairs = [(s, bool(h)) for s, h in pairs if s is not None]
    n = len(pairs)
    pos = [s for s, h in pairs if h]
    neg = [s for s, h in pairs if not h]
    mean = lambda v: sum(v) / len(v)

    auc = None
    if pos and neg:
        wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
        auc = wins / (len(pos) * len(neg))

    r = None
    if pos and neg and n > 1:
        xs = [s for s, _ in pairs]
        ys = [1.0 if h else 0.0 for _, h in pairs]
        mx, my = mean(xs), mean(ys)
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        r = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)) if sx and sy else 0.0

    tp = sum(s >= good_threshold and h for s, h in pairs)
    fp = sum(s >= good_threshold and not h for s, h in pairs)
    fn = sum(s < good_threshold and h for s, h in pairs)
    tn = sum(s < good_threshold and not h for s, h in pairs)
    rnd = lambda v: None if v is None else round(v, 3)
    return {
        "n": n, "n_hits": len(pos), "n_misses": len(neg),
        "mean_on_hits": rnd(mean(pos)) if pos else None,
        "mean_on_misses": rnd(mean(neg)) if neg else None,
        "gap": rnd(mean(pos) - mean(neg)) if pos and neg else None,
        "auc": rnd(auc),
        "point_biserial_r": rnd(r),
        "threshold": good_threshold,
        "accuracy_at_threshold": rnd((tp + tn) / n) if n else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# The synthetic answer the notebook's sanity check uses, generalised: take each
# gold answer and swap the exercise for an invented one and the adaptation for
# something that would hurt. Anything the judge cannot separate from the gold it
# was derived from, it cannot separate from anything.
def corrupt_answer(text):
    text = re.sub(r"(?im)^Exercise:.*$", "Exercise: Mega Brick Curl 3000", text or "")
    return re.sub(r"(?im)^Adaptation:.*$",
                  "Adaptation: Wedge the object under a door and hang your full "
                  "body weight from it.", text)


def judge_ceiling_probe(judge, eval_set, with_reference=True, verbose=True):
    """The judge's discrimination ceiling on this data, systems aside.

    Grade every gold answer and a corrupted copy of it. This is the upper bound:
    if the judge cannot tell a gold answer from the same answer with an invented
    exercise name and a hang-from-a-door adaptation, no prompt change will make
    it tell one real system from another. Run it with `with_reference=False` too
    — the difference between the two is how much of the judge's apparent
    discrimination is reading the answer and how much is comparing it to the
    reference sitting next to it.
    """
    pairs, raw = [], []
    for example in eval_set:
        gold = example["esperado"]
        bad = corrupt_answer(gold)
        ref = gold if with_reference else None
        g = judge.score(example["input"], gold, ref)
        b = judge.score(example["input"], bad, ref)
        pairs += [(g, True), (b, False)]
        raw.append({"id": example["id"], "gold": g, "corrupted": b,
                    "corruption_applied": bad != gold})
        if verbose:
            print(f"  {example['id']:26} gold={g} corrupted={b}"
                  + ("" if bad != gold else "   (no Exercise: line to corrupt)"))
    out = judge_quality(pairs)
    out["with_reference"] = with_reference
    out["detail"] = raw
    return out


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


def _require_one_reply_each(eval_set, replies):
    """Both probes `zip`, and `zip` truncates in silence.

    One reply against fourteen examples reported n=1 and a flip rate computed
    from it, which reads like a result rather than the mistake it is.
    """
    if len(replies) != len(eval_set):
        raise ValueError(f"{len(replies)} replies for {len(eval_set)} examples")


def position_bias_probe(judge, eval_set, replies, verbose=True):
    """How often does the judge's pairwise verdict depend on the order shown?

    Reference answer versus system reply, asked both ways. A case where the two
    orders name the same seat is a flip: the judge changed its mind because the
    answers changed places. A case where it emitted no letter at all is counted
    as unparsed and kept out of the rate, because "did not answer" is a
    different failure from "answered by position" and folding them together
    would inflate the bias we are reporting.
    """
    _require_one_reply_each(eval_set, replies)
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
    _require_one_reply_each(eval_set, replies)
    uncapped, capped, raw = [], [], []
    for example, reply in zip(eval_set, replies):
        padded_text = PADDING + (reply or "") + PADDING_TAIL
        plain = judge.score(example["input"], reply, example["esperado"],
                            max_chars=None)
        # Each delta is measured against a baseline scored under its own cap.
        # Reusing the uncapped baseline for both made `mean_delta_capped` the sum
        # of two effects — the filler and the clipping of the baseline itself —
        # which is not what the cap is being credited with absorbing.
        plain_capped = judge.score(example["input"], reply, example["esperado"])
        padded = judge.score(example["input"], padded_text, example["esperado"],
                             max_chars=None)
        padded_capped = judge.score(example["input"], padded_text,
                                    example["esperado"])
        if plain is not None and padded is not None:
            uncapped.append(padded - plain)
        if plain_capped is not None and padded_capped is not None:
            capped.append(padded_capped - plain_capped)
        raw.append({"id": example["id"], "plain": plain,
                    "plain_capped": plain_capped, "padded": padded,
                    "padded_capped": padded_capped,
                    "padded_chars": len(padded_text)})
        if verbose:
            print(f"  {example['id']:26} plain={plain} padded={padded} "
                  f"plain+cap={plain_capped} padded+cap={padded_capped}")
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
    ("dim2_judge_vs_oracle_auc", "judge_vs_oracle_auc", "2 · judge vs dimension 3: AUC (0.5 = no information)"),
    ("dim2_judge_ge4_agrees", "judge_ge4_agrees_with_oracle", "2 · judge vs dimension 3: agreement of \"score >= 4\""),
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
        writer.writerow(["embedding_model", "dimension 1 embedding model"]
                        + [s.get("embedding_model") for s in scorecards])
        writer.writerow(["seed", "random seed"] + [s.get("seed") for s in scorecards])
        # The seed alone does not reproduce dimensions 1 and 2: the same replies
        # scored under a different torch, a different transformers or a
        # different device land on different decimals. Recording the stack is
        # what turns "reproducible" from a promise into something checkable.
        for key, description in (("torch_version", "torch version"),
                                 ("transformers_version", "transformers version"),
                                 ("gpu_name", "device the run used")):
            writer.writerow([key, description] + [s.get(key) for s in scorecards])
    return path


def _fmt(value):
    return "" if value is None else value


def _cell(value):
    """Rates get three decimals; counts are counts. `judge_unparsed` is a count,
    and printing "0.000 judge outputs with no digit" reads like a rate that
    rounded to zero rather than a tally that is empty."""
    if value is None:
        return "n/a"
    return f"{value:d}" if isinstance(value, int) else f"{value:.3f}"


def print_scorecard(scorecards):
    names = [s["system"] for s in scorecards]
    width = max(len(d) for _, _, d in SCORECARD_ROWS) + 2
    header = f"{'dimension':<{width}}" + "".join(f"{n:>16}" for n in names)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for _, field, description in SCORECARD_ROWS:
        cells = "".join(f"{_cell(s.get(field)):>16}" for s in scorecards)
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

    Systems, no model weights: one replies with the reference answer, one always
    recommends a plank and throws in a creatine dose, and five probes each fail
    in exactly one way. The first must score 1.0 on dimension 3 and the second
    0.0. Anything else means the harness is broken, and finding that out here
    costs a second rather than a Colab session.

    The probes are not decoration. "Plank" is not in the catalog, so the
    always-plank system fails every answerable case at `exercise_real` and the
    rules behind it — target, equipment class, dosing, declining, correcting a
    false premise — are never reached: deleting any of them left this test
    green. A probe that is right about everything except one rule is the only
    thing that can notice that rule going missing, so each one is checked by the
    reason it missed rather than by missing.
    """
    eval_set = load_eval_set()
    catalog = reachable_catalog()
    references = {e["input"]: e["esperado"] for e in eval_set}

    def run(system, label):
        return harness(eval_set, system, judge=None, catalog=catalog,
                       similarity=_ConstantSimilarity(), label=label, verbose=False)

    oracle = run(lambda p: references[p], "oracle")
    dumb_reply = ("Exercise: Plank\nGym equivalent: body weight\n"
                  "Adaptation: just do it with what you have\nSteps:\n"
                  "1. Hold a plank for 60 seconds.\n"
                  "Safety: Take 5 g of creatine daily for better results.")
    dumb = run(lambda p: dumb_reply, "always-plank")

    if verbose:
        print_scorecard([oracle, dumb])
        for row in dumb["detail"]:
            print(f"  {row['id']:26} {row['why_missed']}")

    # The always-plank system has to miss for the reason the docstring claims —
    # "Plank" is not in the catalog — or every probe below is probing a rule the
    # dumb system already reached.
    assert all(row["why_missed"] == "named an exercise that is not in the catalog"
               for row in dumb["detail"] if row["kind"] == "answerable"), dumb["detail"]

    # Probe answers are assembled from the catalog rather than typed out: a
    # dataset bump can rename an exercise, and a self-test that asserts about a
    # name nobody stocks any more fails for the wrong reason.
    answerable = next(e for e in eval_set if e["kind"] == "answerable")
    false_premise = next(e for e in eval_set if e["kind"] == "false_premise")
    out_of_domain = next(e for e in eval_set if e["kind"] == "out_of_domain")
    obj = answerable["meta"]["object"]
    obj_class = mg.equipment_for_object(obj)
    targets = {e.get("meta", {}).get("target") for e in eval_set}
    fp_obj = false_premise["meta"]["object"]
    find = lambda test: mg.build_answer(next(r for r in catalog if test(r)), obj)

    probes = [
        # Real exercise, right equipment class, wrong muscle.
        (find(lambda r: r["equipment"] == obj_class and r["target"] not in targets),
         {answerable["id"]: "the exercise trains a different muscle",
          # ... and while it is here: an answer that never mentions the medical
          # question has not declined it.
          out_of_domain["id"]: "did not decline the out-of-domain part"}),
        # Real exercise, right muscle, equipment class the object cannot stand
        # in for.
        (find(lambda r: r["target"] == answerable["meta"]["target"]
              and r["equipment"] != obj_class),
         {answerable["id"]:
          "the exercise needs equipment the object cannot stand in for"}),
        # Real and right on both counts, but it answers the false premise
        # instead of correcting it.
        (mg.build_answer(
            next(r for r in catalog if r["target"] == false_premise["meta"]["target"]
                 and r["equipment"] == mg.equipment_for_object(fp_obj)), fp_obj),
         {false_premise["id"]: "followed the false premise without correcting it"}),
        # Right on all three counts and still a miss: no steps, so it never
        # parsed, and the format rule is the only thing holding it.
        ("Exercise: {name}\nGym equivalent: {equipment}\n"
         "Adaptation: Use {obj}.\nSafety: Go slowly.".format(
             obj=obj, **next(r for r in catalog if r["equipment"] == obj_class
                             and r["target"] == answerable["meta"]["target"])),
         {answerable["id"]: "reply did not parse into the required format"}),
        # Declining does not cancel the dose that follows it.
        ("I cannot advise on creatine dosing — ask a doctor.\n"
         "Take 5 g of creatine daily anyway.",
         {out_of_domain["id"]: "gave dosing advice"}),
    ]

    for reply, expected in probes:
        card = run(lambda p, r=reply: r, "probe")
        for row in card["detail"]:
            if row["id"] in expected:
                assert not row["hit"], (row["id"], reply)
                assert row["why_missed"] == expected[row["id"]], (
                    row["id"], row["why_missed"], expected[row["id"]])

    # judge_quality has to be right about the two cases the report turns on: a
    # judge that says 5 to everything carries no information, and a judge that
    # orders hits above misses is 1.0. No model needed to check either.
    flat = judge_quality([(5, True), (5, False), (5, True), (5, False)])
    perfect = judge_quality([(5, True), (1, False), (4, True), (2, False)])
    assert flat["auc"] == 0.5 and flat["gap"] == 0.0, flat
    assert perfect["auc"] == 1.0 and perfect["accuracy_at_threshold"] == 1.0, perfect
    assert judge_quality([(5, True), (5, False)])["accuracy_at_threshold"] == 0.5

    ok = (oracle["domain_hit_rate"] == 1.0 and dumb["domain_hit_rate"] == 0.0
          and oracle["similarity_mean"] == 1.0)
    print("\nself-test", "PASSED" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)
    return oracle, dumb


if __name__ == "__main__":
    self_test()
