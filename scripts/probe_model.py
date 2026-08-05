"""Feasibility probe: does this model train and generate on this GPU, and how slow?

Standalone on purpose — no repo imports, no dataset needed. Paste it into Colab
and it answers the only question that matters before committing to a base model:
does a T4 hold it, and does a full run fit in a free session.

Dummy examples are shaped to the real MacGyver profile measured from the
catalog (prompt ~70 tokens, completion ~180, max 292), so the timings
extrapolate to the actual run instead of to a toy.

On Colab, uninstall torchao first: the image ships 0.10.0 pinned to its torch
build, peft refuses anything below 0.16, and upgrading it drags a different
torch in behind it.

    !pip uninstall -y -q torchao

Run:
    uv run python scripts/probe_model.py --model Qwen/Qwen3-1.7B
    uv run python scripts/probe_model.py --model Qwen/Qwen3-4B-Instruct-2507 --4bit
"""

import argparse
import time

import torch

# Measured from data/exercises-dataset: 924 reachable exercises, 1 object variant
# each plus ~15% refusals, 2 epochs. Change with --n-examples / --epochs.
N_EXAMPLES = 1800
EPOCHS = 2
N_EVAL_PROMPTS = 150
EVAL_NEW_TOKENS = 320

PROMPT = (
    "I want to train my {t} but I have no gym equipment. All I have is {o}.\n"
    "Recommend one safe exercise and explain how to do it.\n\n"
    "Answer in exactly this format:\n"
    "Exercise: <name>\nGym equivalent: <equipment>\nAdaptation: <how to substitute>\n"
    "Steps:\n1. <step>\nSafety: <note>"
)
COMPLETION = (
    "Exercise: dumbbell incline row\nGym equivalent: dumbbell\n"
    "Adaptation: Use your two filled water bottles in place of the dumbbells; "
    "match the weight on both sides so the movement stays symmetrical.\nSteps:\n"
    + "".join(
        f"{i}. Keep your back flat and your core braced while you move through "
        f"the full range of motion under control, step {i}.\n"
        for i in range(1, 7)
    )
    + "Safety: Keep the movement controlled and stop if you feel pain."
)


def vram(tag):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {tag:<22} {alloc:5.2f} GB now / {peak:5.2f} GB peak")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--4bit", dest="four_bit", action="store_true")
    p.add_argument("--steps", type=int, default=12, help="optimiser steps to time")
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    p.add_argument("--epochs", type=float, default=EPOCHS)
    p.add_argument("--no-generate", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU — this probe only answers questions about a GPU")

    props = torch.cuda.get_device_properties(0)
    major, _ = torch.cuda.get_device_capability()
    # Hardware bf16 arrived with Ampere (sm_80). torch.cuda.is_bf16_supported()
    # also counts software emulation and returns True on a T4, which runs
    # slowly and silently — compute capability is the unambiguous question.
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    print(f"gpu     {props.name} ({props.total_memory / 1e9:.1f} GB, sm_{major}{props.minor})")
    print(f"model   {args.model}  {'4-bit nf4' if args.four_bit else str(dtype)}")
    print(f"config  seq={args.seq} batch={args.batch} accum={args.grad_accum}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = {"dtype": dtype, "device_map": {"": 0}}
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    torch.cuda.synchronize()
    print(f"loaded in {time.perf_counter() - t0:.0f}s")
    vram("weights")

    if args.four_bit:
        # Casts the layers that stay unquantised — layernorms, embeddings, the
        # lm_head — up to fp32. Without it the fp16 GradScaler a T4 needs blows
        # up on gradients that inherited the checkpoint's native bfloat16:
        #   NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda
        #   not implemented for 'BFloat16'
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    n_tok = len(tok.encode(PROMPT.format(t="lats", o="a towel and a door") + COMPLETION))
    print(f"  {'dummy example':<22} {n_tok} tokens (real profile: ~250, max ~360)\n")

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = [
        {
            "prompt": [{"role": "user", "content": PROMPT.format(t="lats", o="a towel")}],
            "completion": [{"role": "assistant", "content": COMPLETION}],
        }
    ] * (args.steps * args.batch * args.grad_accum + args.batch * args.grad_accum)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir="/tmp/probe",
            max_steps=args.steps,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            max_length=args.seq,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=1e-4,
            logging_steps=1000,
            save_strategy="no",
            report_to="none",
            seed=42,
            bf16=dtype is torch.bfloat16,
            fp16=dtype is torch.float16,
        ),
        train_dataset=Dataset.from_list(rows),
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ),
    )
    # Belt and braces on the same problem: peft picks the LoRA dtype from the
    # layer it wraps, and for a 4-bit Linear the base weight is uint8, so the
    # choice falls through to the checkpoint's dtype rather than ours. fp32
    # adapters are what the scaler expects and cost ~130 MB at r=16.
    if args.four_bit:
        for param in trainer.model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    per_step = elapsed / args.steps
    eff_batch = args.batch * args.grad_accum
    total_steps = args.n_examples * args.epochs / eff_batch
    print(f"\ntrain   {per_step:.1f} s/step (effective batch {eff_batch})")
    vram("training peak")
    print(f"  -> {args.n_examples} examples x {args.epochs} epochs = "
          f"{total_steps:.0f} steps ~= {total_steps * per_step / 60:.0f} min")

    if args.no_generate:
        return

    # Generation is the half people forget to measure, and it is where 4-bit
    # hurts most: every matmul pays a dequantisation cost with no training
    # throughput to amortise it over.
    del trainer
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # peft injected the adapter into `model` in place, and it has just spent
    # `--steps` steps overfitting one repeated dummy example. Generating
    # through it would show that dummy echoed back, not what the base model
    # does with the prompt, so switch the adapter off for the sample below.
    from peft.tuners.lora import LoraLayer

    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.disable_adapters = True

    model.eval()
    tok.padding_side = "left"
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(t="lats", o="a towel")}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    ] * 8
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        model.generate(**enc, max_new_tokens=16, do_sample=False,
                       pad_token_id=tok.pad_token_id)  # warm up kernels
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**enc, max_new_tokens=EVAL_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
    gen = time.perf_counter() - t0

    per_prompt = gen / len(texts)
    print(f"\ngen     {per_prompt:.1f} s/prompt at {EVAL_NEW_TOKENS} new tokens, batch 8")
    vram("generation peak")
    print(f"  -> {N_EVAL_PROMPTS} val prompts x 2 models (zero-shot + tuned) "
          f"~= {2 * N_EVAL_PROMPTS * per_prompt / 60:.0f} min")

    print("\n--- one untrained sample (is the base model even close?) ---")
    print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)[:600])


if __name__ == "__main__":
    main()
