"""Shared pieces for the MacGyver experiments.

Imported by the numbered scripts and by the Colab notebook. This module exists
mainly for `build_prompt`: the training data, the zero-shot baseline and the
fine-tuned evaluation must render the same string, or the reported delta
measures prompt drift instead of fine-tuning. One definition, three call sites.

The numbered scripts cannot be imported (a module name cannot start with a
digit), so anything two of them need lives here.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data/exercises-dataset/data/exercises.json"
PROCESSED = ROOT / "data/processed/macgyver"
REPORTS = ROOT / "reports"

SEED = 42

# Equipment we are willing to improvise, and the household objects that stand
# in for it. Objects are grouped by the *role* the equipment plays, which is
# what makes a substitution work or fail:
#
#   dumbbell     a matched pair, one per hand
#   kettlebell   a single off-centre load carried by a handle
#   weighted     dead weight added to a bodyweight movement
#   body weight  nothing at all
#
# `holdout` objects never appear in training. They exist to answer one
# question with a number instead of a hope: does the model generalise to
# objects it has never been shown, or has it memorised nine phrases?
OBJECT_MAP = {
    "dumbbell": {
        "seen": [
            "two filled water bottles",
            "two full cans of food",
            "two one-litre juice cartons",
            "two bags of rice",
        ],
        "holdout": ["two bricks", "two full paint cans"],
    },
    "kettlebell": {
        "seen": [
            "a filled detergent jug with a handle",
            "a backpack held by one strap",
            "a bucket with a handle",
        ],
        "holdout": ["a paint can with a wire handle", "a shopping bag of tins"],
    },
    "weighted": {
        "seen": [
            "a backpack loaded with books",
            "a filled shopping bag",
            "a suitcase packed with clothes",
        ],
        "holdout": ["a duffel bag full of laundry", "a crate of bottled water"],
    },
    "body weight": {
        "seen": ["nothing at all", "just the floor and a wall", "an empty room"],
        "holdout": ["a patch of grass in the park", "a bare hallway"],
    },
}

# One note per equipment class, not per object. Written against the *role*, so
# the sentence holds for an object the table has never seen — and so the
# authored surface stays at four sentences instead of twenty.
ADAPTATION = {
    "dumbbell": ("Use {object} in place of the dumbbells, one in each hand. "
                 "Match the load on both sides so the movement stays symmetrical."),
    "kettlebell": ("Use {object} in place of the kettlebell. Grip it firmly and "
                   "keep your wrist straight — an off-centre handle pulls it out "
                   "of line."),
    "weighted": ("Use {object} as the added weight. Secure it against your body "
                 "so the load cannot shift mid-repetition."),
    "body weight": ("No equipment is needed for this one. Clear enough space to "
                    "extend fully in every direction before you start."),
}

# Equipment deliberately left out, and why. Carried into meta.json so the
# decision is auditable rather than buried in a comment.
EXCLUDED = {
    "cable": "steps say 'attach the bar to the high pulley' — reciting that to "
             "someone holding a towel is nonsense, not adaptation",
    "barbell": "loads and failure modes improvised equipment cannot safely "
               "reproduce",
    "ez barbell": "same as barbell",
    "olympic barbell": "same as barbell",
    "smith machine": "the machine's fixed bar path is the exercise",
    "leverage machine": "the machine's fixed bar path is the exercise",
    "sled machine": "the machine's fixed bar path is the exercise",
    "assisted": "the assistance is the machine",
    "stability ball": "no household object has its instability",
    "bosu ball": "no household object has its instability",
    "medicine ball": "throwing movements; a backpack is not throwable",
    "rope": "the class mixes battling ropes, jump rope and stretches — no single "
            "object substitutes for all three",
    "band": "elastic tension has no reliable household equivalent; tights and "
            "inner tubes fail unpredictably under load",
    "resistance band": "same as band",
    "roller": "no household equivalent",
    "wheel roller": "no household equivalent",
    "hammer": "single record, and the object is already a household one",
    "tire": "single record, not a household object",
    "trap bar": "single record, no household equivalent",
    "elliptical machine": "cardio machine, no household equivalent",
    "stationary bike": "cardio machine, no household equivalent",
    "skierg machine": "cardio machine, no household equivalent",
    "stepmill machine": "cardio machine, no household equivalent",
    "upper body ergometer": "cardio machine, no household equivalent",
}

SAFETY = "Keep the movement controlled and stop if you feel pain."

PROMPT = (
    "I want to train my {target} but I have no gym equipment. "
    "All I have is {object}.\n"
    "Recommend one safe exercise and explain how to do it.\n\n"
    "Answer in exactly this format:\n"
    "Exercise: <name>\n"
    "Gym equivalent: <equipment>\n"
    "Adaptation: <how to use what I have>\n"
    "Steps:\n"
    "1. <step>\n"
    "Safety: <one line>\n\n"
    "If nothing you know of trains that muscle with that object, say so instead "
    "of inventing one."
)


def objects_for(equipment, include_holdout=False):
    """Object phrases usable for an equipment class."""
    entry = OBJECT_MAP[equipment]
    return entry["seen"] + (entry["holdout"] if include_holdout else [])


def equipment_for_object(obj):
    """Which equipment class an object stands in for. None if unknown."""
    for equipment, entry in OBJECT_MAP.items():
        if obj in entry["seen"] or obj in entry["holdout"]:
            return equipment
    return None


def is_holdout(obj):
    return any(obj in entry["holdout"] for entry in OBJECT_MAP.values())


def build_prompt(target, obj):
    """The one prompt definition. Never build this string anywhere else."""
    return PROMPT.format(target=target, object=obj)


def build_answer(record, obj):
    """An answerable completion: two template lines around real catalog steps."""
    steps = "\n".join(
        f"{i}. {s}" for i, s in enumerate(record["instruction_steps"]["en"], 1)
    )
    return (
        f"Exercise: {record['name']}\n"
        f"Gym equivalent: {record['equipment']}\n"
        f"Adaptation: {ADAPTATION[record['equipment']].format(object=obj)}\n"
        f"Steps:\n{steps}\n"
        f"Safety: {SAFETY}"
    )


def build_refusal(target, obj, alternative):
    """A refusal, pointing at something the catalog does hold."""
    return (
        f"No safe option: I know no {target} exercise that works with {obj}.\n"
        f"Closest alternative: {alternative['name']}, which needs "
        f"{alternative['equipment']}."
    )


def load_catalog(path=CATALOG):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The dataset is a git submodule:\n"
            "    git submodule update --init --recursive"
        )
    return json.loads(path.read_text())


def load_split(name):
    path = PROCESSED / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build the splits first:\n"
            "    uv run python scripts/01_macgyver_data.py"
        )
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_meta():
    path = PROCESSED / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build the splits first:\n"
            "    uv run python scripts/01_macgyver_data.py"
        )
    return json.loads(path.read_text())


def write_report(name, payload):
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def pick_device_dtype():
    """bf16 where the GPU has bf16 *hardware*, fp16 otherwise.

    `torch.cuda.is_bf16_supported()` is a trap: it returns True on a T4 because
    it counts bf16 that torch can emulate in software, which runs slowly and
    never errors. Hardware bf16 arrived with Ampere (sm_80), so compute
    capability is the unambiguous question and it is what we ask.
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu", torch.float32

    major, _ = torch.cuda.get_device_capability()
    return "cuda", torch.bfloat16 if major >= 8 else torch.float16
