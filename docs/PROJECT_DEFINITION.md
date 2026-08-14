# STAI Integrative Project — Definition Template

## 0 · Team

- **Team name:** MacGyver Gym Rat
- **Members** (name · email):
  - Juan Andrés Young Hoyos · jayoungh@eafit.edu.co
  - José Alejandro Jiménez Vásquez · jajimenez4@eafit.edu.co
  - David Cuadros Mariño · dacuadrosm@eafit.edu.co
  - Alberto Daniel Cervantes Forero · adcervantf@eafit.edu.co
- **Point of contact with the professor:** Juan Andrés Young Hoyos
- **How the team coordinates code:** GitHub, with repos split by responsibility:
  - Dataset: https://github.com/jayounghoyos/exercises-dataset (fork of https://github.com/hasaneyldrm/exercises-dataset)
  - Models and experiments: https://github.com/iamcroody/models-for-exercises-dataset (uses the dataset as a git submodule pinned to a fixed commit, so all 4 members train on exactly the same data)
- **Weekly meeting day/time outside class:** Tuesdays, 3–4 pm

## 1 · Topic and project

**In one sentence, what is your project?**

"MacGyver Gym Rat": a fine-tuned model that, given a muscle you want to train and a household object you have on hand, recommends a real exercise from a 1,324-exercise catalog, explains how to adapt the object, and recites the real steps — or honestly admits when there's no valid option.

**Why this topic?**

The project sits inside Buddy, the "AI trainer" of the Croody ecosystem, focused on physical health, prevention, and balance. We chose this topic because the exercise dataset is very complete and makes a good fine-tuning case: the model has to understand relationships and context between exercise and muscle, and reason across different types of substitutions.

## 2 · User and decision

- **Who would use this system?** Someone who already had a specific muscle group planned to train that day, but is left without the gym equipment they needed — traveling, at home, or anywhere without gym access — and doesn't want to skip the workout or improvise something that doesn't work or is unsafe.
- **What concrete decision or task does the system help with?** Which exercise to do for a specific muscle when the necessary gym equipment isn't available, using only a household object they do have.
- **What does that person do today, without the system?** They probably skip that muscle group for the day, do something generic that doesn't really train it, or search on their own online/on social media for an improvised substitute with no certainty whether it actually works or is safe.

## 3 · Core model task (M1)

- **What type of task is it?** Open-ended generation (not classification): given a target muscle and a household object, the model must write a full recommendation following a fixed format, or refuse if there's no valid option.
- **What is the input? What is the expected output?**
  - Input: a natural-language description of the muscle to train and the object available (e.g. *"I want to train my delts but I have no gym equipment. All I have is two filled water bottles."*), plus the exact response format requested.
  - Output: the name of a real exercise from the catalog, its gym equivalent, the adaptation to the household object, the real steps, and one safety line — or an honest refusal if nothing in the catalog applies.
- **Which candidate base model?** Qwen/Qwen3-1.7B. Chosen after testing the simplest possible fine-tuning directly on a free Colab T4, comparing memory/GPU usage and performance across the sizes in the Qwen3 family: the 1.7B model gave the best balance. The 8B model also ran, but pushed the free T4's resources hard.
  - How Qwen3 tokenizes the domain vocabulary: compared against SmolLM2-1.7B-Instruct and GPT-2, Qwen3-1.7B used 1.43 tokens/word on exercise names and 1.06 on household-object phrases — the best result of the three.

## 4 · Dataset

- **Where does the text come from?** https://github.com/jayounghoyos/exercises-dataset (own fork of https://github.com/hasaneyldrm/exercises-dataset)
- **Roughly how many examples?** The full catalog has 1,324 exercises; of those, 358 are reachable for the "MacGyver Gym Rat" task (the ones that map to one of the 4 equipment classes with a household stand-in). The final training dataset is 662 examples (train) + 154 (val) + 154 (test).
- **What language(s) is it in?** The original catalog has instructions in 10 languages, but the project uses **English only**.
- **Is there any license restricting use?** Exercise metadata (name, equipment, muscle, instructions): MIT. Images and GIFs: © Gym visual — not used in this project.
- **Any known bias or limitation in the dataset?** Only 4 of the catalog's equipment classes have a defined household stand-in (dumbbell, kettlebell, weighted, body weight); the rest (cables, machines, barbells) are left out. The object-to-equipment mapping was chosen by hand, not measured. See `docs/DATASET.md` for the full detail, including how the splits were built.

## 5 · Success metric

- **What main metric will measure whether the system is useful?** Constraint satisfaction: the exercise the model names actually exists in the catalog, and matches both the requested target muscle and equipment class. As a secondary metric, ROUGE-L between the recited steps and that exercise's real steps (measures whether the biomechanics described came from the catalog or were invented), plus refusal precision/recall (whether the model recognizes when there's no valid option).
- **Why that metric and not another?** There are many valid answers to the same question (several different exercises can work for the same muscle+object combo), so scoring against a single "gold" answer would mark most correct answers as wrong. Instead, the metric resolves the exercise the model named against the real catalog and checks whether that choice satisfies the request.
- **What would be a reasonable baseline to compare against?** The same model (Qwen3-1.7B) with no fine-tuning, evaluated with the same prompt, same split, and same greedy decoding — so the only thing that changes is whether a LoRA adapter is loaded.

## 6 · Visual component (M4) — initial plan

- **How could a visual component be integrated into this system?** Show the real GIF of the recommended exercise alongside the text response, so the user sees the execution instead of only reading it. The models repo's README already leaves this path open: `Qwen3-VL-2B` shares the tokenizer and chat template with the current model, so moving to a vision-capable model is a family swap, not a rewrite — it could also be used to verify visually that the user actually has the object they say they have, or that their form resembles the suggested exercise.
- **Do you have representative image examples?** Yes — every exercise in the catalog ships a thumbnail and a GIF. Real example, the same exercise the team already used as a reference case in their design notes ("two filled water bottles" → dumbbell): **dumbbell full can lateral raise** (id `0311`) — `images/0311-AQ0mC4Y.jpg` and `videos/0311-AQ0mC4Y.gif`.

## 7 · Ethical and usage risks

- **Who could be harmed if the system fails?** Someone who gets a recommendation poorly matched to their level, or with the technique poorly described, could get injured — the team's own evaluation already found that the model sometimes invents an adaptation instead of recognizing that no safe option exists (refusal recall of only 4.3%, see results report).
- **Are there groups of people the system is more likely to fail for?** Anyone asking for something outside the 4 equipment classes covered (cables, machines, and barbells have no defined household stand-in), or using an object the model never saw during training.
- **How would those risks be mitigated?** Make it clear in the interface that this doesn't replace a trainer or health professional. Since the model tends to invent rather than refuse, the application should validate on its own layer that the named exercise actually exists in the catalog before showing it to the user — reusing the same "constraint satisfaction" logic the evaluator already uses, but as a production filter rather than only an offline metric; if it doesn't pass the filter, show a message that no safe option was found instead of the model's answer. And keep the system's declared scope limited to the 4 equipment classes it actually covers, rather than implying it works for any situation.