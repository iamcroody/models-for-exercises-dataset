"""LoRA fine-tuning of the base model on the MacGyver task.

The CLI mirror of the notebook's training section — same config, same seed,
same result — so the experiment can be rerun and ablated without a Jupyter
kernel.

Hyperparameters and why:

  r = 16          662 training examples over ~360 exercises is a small, narrow
                  target. The job is to index a closed catalog by (target,
                  equipment) and recite it, not to install new knowledge about
                  human movement, and that needs little capacity. The --r flag
                  exists so the claim is tested rather than asserted.

  alpha = 32      Holds the conventional alpha = 2r, so the effective LoRA
                  scale (alpha / r) stays at 2.0 and changing r alone does not
                  silently change the update magnitude too. Without this an r
                  ablation would confound two variables.

  target_modules  All linear projections, attention *and* MLP. The course
                  material suggests attention only (q_proj, v_proj); the
                  association from "two filled water bottles" to "dumbbell" is
                  lexical-semantic and that kind of knowledge lives largely in
                  the MLP blocks. --target-modules attention tests exactly
                  that rather than taking either side on faith.

  lr = 1e-4       TRL's documented adapter rate, ~5x a full fine-tune's,
                  because only the freshly-initialised low-rank matrices learn.

  epochs = 3      662 examples is small; two passes underfit the refusal class,
                  which is only 8% of train.

Batch size is fixed rather than scaled to the GPU on purpose: a local run and
the Colab T4 run then take identical optimisation steps, so the numbers in the
README are the numbers a grader reproduces.

Run:
    uv run python scripts/04_train_lora.py
    uv run python scripts/04_train_lora.py --model Qwen/Qwen3-4B-Instruct-2507 --4bit
    uv run python scripts/04_train_lora.py --r 8 --run-name r8
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import macgyver as mg

ATTENTION_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR = ATTENTION_ONLY + ["gate_proj", "up_proj", "down_proj"]

# The longest gold example renders to ~400 tokens under Qwen3's template, so
# 512 truncates nothing while costing far less memory and time than the 1024
# default. Verified by --check-lengths before every run.
MAX_LENGTH = 512


def build_dataset(split, tok):
    """Only the two columns TRL needs. The gold labels carried alongside in the
    JSONL for evaluation would otherwise be templated into the training text."""
    from datasets import Dataset

    rows = mg.load_split(split)
    return Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    )


def report_lengths(tok):
    """Truncation is silent and would quietly delete the Safety line from every
    long example, so the budget is checked rather than assumed."""
    import statistics

    lengths = []
    for r in mg.load_split("train"):
        text = tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=False)
        lengths.append(len(tok.encode(text, add_special_tokens=False)))
    longest = max(lengths)
    print(f"tokens     mean {statistics.mean(lengths):.0f}, max {longest}, "
          f"budget {MAX_LENGTH}")
    if longest > MAX_LENGTH:
        print(f"WARNING: {sum(l > MAX_LENGTH for l in lengths)} examples truncate")
    return longest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=mg.BASE_MODEL)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=None,
                        help="default 2*r, keeping the effective scale at 2.0")
    parser.add_argument("--target-modules", choices=["all-linear", "attention"],
                        default="all-linear")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--4bit", dest="four_bit", action="store_true")
    parser.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else 2 * args.r
    modules = ALL_LINEAR if args.target_modules == "all-linear" else ATTENTION_ONLY
    run_name = args.run_name or f"r{args.r}-{args.target_modules}"

    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    device, dtype = mg.pick_device_dtype()
    bf16 = dtype is torch.bfloat16

    print(f"model      {args.model}")
    print(f"device     {device} / {'4-bit nf4' if args.four_bit else dtype}")
    print(f"lora       r={args.r} alpha={alpha} dropout=0.05")
    print(f"modules    {', '.join(modules)}")
    print(f"batch      {args.batch_size} x {args.grad_accum} "
          f"= {args.batch_size * args.grad_accum} effective")
    print(f"run        {run_name}")

    tok = mg.load_tokenizer(args.model)
    report_lengths(tok)

    train_ds, val_ds = build_dataset("train", tok), build_dataset("val", tok)
    print(f"data       train {len(train_ds)}  val {len(val_ds)}\n")

    model = mg.load_model(args.model, four_bit=args.four_bit)
    torch.set_grad_enabled(True)  # load_model disables it for inference

    if args.four_bit:
        # Casts the layers that stay unquantised up to fp32. Without it the
        # fp16 GradScaler a T4 needs blows up on gradients that inherited the
        # checkpoint's bfloat16.
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_ckpt
        )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(mg.ROOT / "outputs" / run_name),
            run_name=run_name,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=args.grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=MAX_LENGTH,
            seed=mg.SEED,
            data_seed=mg.SEED,
            eval_strategy="epoch",
            logging_steps=10,
            save_strategy="no",  # the adapter is saved once, at the end
            report_to="none",  # never prompt for a wandb login inside Colab
            bf16=bf16,
            fp16=not bf16,
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=LoraConfig(
            r=args.r,
            lora_alpha=alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules,
        ),
    )

    if args.four_bit:
        # peft picks the adapter dtype from the layer it wraps; for a 4-bit
        # Linear the base weight is uint8, so the choice falls through to the
        # checkpoint's dtype rather than ours. fp32 adapters are what the
        # scaler expects and cost ~130 MB at r=16.
        for param in trainer.model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    trainer.model.print_trainable_parameters()
    result = trainer.train()

    out_dir = mg.ROOT / "models" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # LoRA parameters are created in fp32 regardless of the base model's dtype,
    # which makes the saved adapter twice the size it needs to be. Inference
    # runs the base in bf16 or fp16 anyway, so fp16 costs no usable precision
    # and halves what the repo carries. The tokenizer is deliberately not
    # copied: it is identical to the base model's and load_tokenizer fetches it
    # from there.
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)
    trainer.model.save_pretrained(str(out_dir))

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    (out_dir / "training_summary.json").write_text(json.dumps({
        "base_model": args.model,
        "run_name": run_name,
        "lora": {"r": args.r, "alpha": alpha, "dropout": 0.05,
                 "target_modules": modules},
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": MAX_LENGTH,
        "seed": mg.SEED,
        "precision": "4-bit nf4" if args.four_bit else ("bf16" if bf16 else "fp16"),
        "gradient_checkpointing": args.grad_ckpt,
        "trainable_params": trainable,
        "train_examples": len(train_ds),
        "train_runtime_s": round(result.metrics["train_runtime"], 1),
        "final_train_loss": round(result.metrics["train_loss"], 4),
        "eval_loss_per_epoch": [round(h["eval_loss"], 4) for h in history],
    }, indent=2) + "\n")

    print(f"\ntrain loss {result.metrics['train_loss']:.4f} "
          f"in {result.metrics['train_runtime']:.0f}s")
    print("eval loss  " + " -> ".join(f"{h['eval_loss']:.4f}" for h in history))
    print(f"adapter    {out_dir}")
    print(f"\nnext: python scripts/03_eval_zeroshot.py --adapter {out_dir} "
          f"--report-name finetuned")


if __name__ == "__main__":
    main()
