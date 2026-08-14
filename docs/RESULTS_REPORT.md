# Results Report — M1

## Base model

`Qwen/Qwen3-1.7B`, a decoder. This family was chosen because the task is open-ended generation (a recommendation, an adaptation, and instructions — not a label from a closed set); an encoder classifier cannot emit prose, and an encoder-decoder buys nothing here since the input is a short question rather than a document to transform. This model also has to carry M2 (RAG on top of it) and M3 (a visual component), which an encoder classifier could not host.

Within decoders, Qwen3-1.7B was chosen because:
- It's Apache-2.0 and ungated — no Hugging Face login or licence click-through required (unlike Llama-3.2, which is gated).
- It fits the free tier with room: ~1.0% of parameters trainable under LoRA, ~5.8 GB peak against a T4's 15.6 GB.
- It has a forward path: `Qwen3-VL-2B` shares this tokenizer and chat template, so M3's visual component is a family swap rather than a rewrite — and the catalog already ships a thumbnail and animation GIF for all 1,324 exercises.

**Tokenization check (Week 3 Lab A):** comparing Qwen/Qwen3-1.7B against SmolLM2-1.7B-Instruct and GPT-2 on the domain vocabulary (exercise names and household-object phrases), Qwen3-1.7B used 1.43 tokens per word on exercise names and 1.06 on object phrases — the best of the three candidates.

## Baseline

The same model, same prompt, no adapter, scored on the same validation split used for the fine-tuned model — the baseline the assignment recommends, since it isolates exactly what LoRA contributed.

## LoRA configuration

| hyperparameter | value | why |
|---|---|---|
| `r` | 16 | 662 examples over ~358 exercises is a small, narrow target. The job is to index a closed catalog and recite it, not to install new knowledge about human movement. |
| `lora_alpha` | 32 | Holds `alpha = 2r`, so the effective scale `alpha/r` stays at 2.0 and changing `r` alone doesn't silently change the update magnitude too. |
| `target_modules` | all 7 linear projections (attention + MLP) | Mapping "two filled water bottles" → `dumbbell` is lexical-semantic, and that lives largely in the MLP blocks, not just attention. |
| `lora_dropout` | 0.05 | Light regularization on a small dataset. |
| `learning_rate` | 1e-4 | TRL's documented adapter rate, ~5× a full fine-tune's, since only the freshly-initialized low-rank matrices are learning. |
| epochs | 3 | 662 examples is small, and refusals are only 8% of train — two passes would underfit them. |
| effective batch | 16 | `2 × 8`, fixed rather than scaled to the available GPU. |

Training: 17,432,576 trainable parameters out of 1,738,007,552 total (1.00%), fp16 precision, gradient checkpointing enabled, ~2h04 of training on a free Colab T4. Final training loss: 0.613. Validation loss per epoch: 0.523 → 0.412 → 0.400.

## Results table

Validation set (n=154, 108 answerable + 46 expected refusals):

| metric | zero-shot (no adapter) | LoRA (fine-tuned) |
|---|---:|---:|
| format ok | 100.0% | 100.0% |
| **constraint satisfaction** (headline) | 0.0% | 13.0% |
| — exercise is real | 0.0% | 20.4% |
| — target matches | 0.0% | 20.4% |
| — equipment matches | 0.0% | 13.0% |
| step grounding (ROUGE-L) | n/a | 0.666 (n=33) |
| refusal precision / recall / F1 | 0.0% / 0.0% / 0.0% | 100.0% / 4.3% / 8.3% |
| seen objects — satisfaction | 0.0% (n=54) | 11.1% (n=54) |
| unseen objects — satisfaction | 0.0% (n=54) | 14.8% (n=54) |

## Honest reading

The zero-shot model answers in the required format 100% of the time, but always names a generic exercise ("Plank") that doesn't match the catalog's specific naming, so it never passes any of the "real exercise" checks. LoRA fine-tuning raises constraint satisfaction from 0% to 13.0%, and when it does name a real exercise, the steps it recites are mostly grounded in that exercise's real instructions (ROUGE-L 0.666).

Where it still fails: refusal recall is only 4.3% — it almost always invents an exercise instead of recognizing that no valid one exists, even though when it does refuse, it's correct 100% of the time (precision). The gap between seen objects (11.1%) and unseen objects (14.8%) isn't a real gap; with only 54 examples per group, that falls within noise (~0.6 points per example). Worth noting: a 40-line lookup table indexing the catalog by (target, equipment) would score 100% on this metric — what the fine-tune buys is doing this in natural language, on a model that M2 will put RAG on top of and M3 will give vision to, not that it's the best way to answer this one question today.

## Qualitative examples

**1 — answerable case** (*"I want to train my upper back but I have no gym equipment. All I have is an empty room."*)
- Zero-shot: recommends "Plank" (generic, not a real exercise from the catalog under that name).
- LoRA: recommends "dumbbell standing row" adapted as "empty weight on the dumbbell" — names a real exercise, but the adaptation to an "empty room" is weak.

**2 — expected refusal case** (*"I want to train my cardiovascular system but I have no gym equipment. All I have is a paint can with a wire handle."*)
- Zero-shot: invents a plank adaptation using the paint can.
- LoRA: instead of refusing (the correct answer), recommends "kettlebell squat" adapted to the paint can — still doesn't reliably recognize when it should refuse.

**3 — unseen object at training time** (*"I want to train my abs but I have no gym equipment. All I have is a bare hallway."*)
- Zero-shot: generic "Plank" again.
- LoRA: recommends "side plank with leg lift", a real body-weight exercise, with steps correctly taken from the catalog.