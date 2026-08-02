"""LoRA fine-tuning of the base model on the extraction task.

This is the CLI mirror of the notebook's training section — same config, same
seed, same result — so the experiment can be rerun and ablated without a
Jupyter kernel.

Hyperparameters and why:

  r = 16          1081 training examples over a 19 + 28 value label space is
                  a small, narrow target. The job is to constrain the output
                  to a closed vocabulary, not to install new knowledge, and
                  that needs little capacity. Higher rank mostly buys
                  overfitting here; the --r flag exists so that claim is
                  tested rather than asserted.

  alpha = 32      Holds the conventional alpha = 2r, so the effective LoRA
                  scale (alpha / r) stays at 2.0 and changing r alone does
                  not silently change the update magnitude too. Without this
                  the r ablation would confound two variables.

  target_modules  All linear projections, attention *and* MLP. The mapping
                  from "cable incline pushdown" to {lats, cable} is lexical-
                  semantic, and that association lives largely in the MLP
                  blocks; attention-only adapters can reweight what the model
                  attends to but not what it knows a term means. --target-
                  modules attention tests exactly that.

  lr = 1e-4       TRL's documented adapter learning rate, ~5x a full
                  fine-tune's, because only the freshly-initialised low-rank
                  matrices are learning.

Batch size is fixed rather than scaled to the GPU on purpose: local runs and
the Colab T4 run then take identical optimisation steps, so the numbers in
the README are the numbers a grader reproduces.

Run:
    uv run python scripts/04_train_lora.py
    uv run python scripts/04_train_lora.py --r 8 --run-name r8
    uv run python scripts/04_train_lora.py --target-modules attention --run-name attn
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exlib

ATTENTION_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR = ATTENTION_ONLY + ["gate_proj", "up_proj", "down_proj"]

# Measured with 00_tokenizer_check: the longest train example renders to 386
# tokens under Qwen3's template, so 512 truncates nothing while costing far
# less memory and time than the 1024 default.
MAX_LENGTH = 512


def build_config(args, run_name, dtype, bf16):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=str(exlib.ROOT / "outputs" / run_name),
        run_name=run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # effective batch 16
        max_length=MAX_LENGTH,
        seed=exlib.SEED,
        data_seed=exlib.SEED,
        eval_strategy="epoch",
        logging_steps=10,
        save_strategy="no",  # the adapter is saved once, at the end
        report_to="none",  # never prompt for a wandb login inside Colab
        model_init_kwargs={"dtype": dtype},
        bf16=bf16,
        fp16=not bf16,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=exlib.BASE_MODEL)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: 2 * r, keeping the effective scale at 2.0)",
    )
    parser.add_argument(
        "--target-modules", choices=["all-linear", "attention"], default="all-linear"
    )
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    alpha = args.alpha if args.alpha is not None else 2 * args.r
    modules = ALL_LINEAR if args.target_modules == "all-linear" else ATTENTION_ONLY
    run_name = args.run_name or f"r{args.r}-{args.target_modules}"

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer

    device, dtype = exlib.pick_device_dtype()
    bf16 = str(dtype).endswith("bfloat16")

    print(f"model      {args.model}")
    print(f"device     {device} / {dtype}")
    print(f"lora       r={args.r} alpha={alpha} dropout=0.05")
    print(f"modules    {', '.join(modules)}")
    print(f"run        {run_name}\n")

    # Only the columns TRL needs; the gold-label columns carried in the JSONL
    # for evaluation would otherwise be templated into the training text.
    def to_dataset(split):
        rows = exlib.load_split(split)
        return Dataset.from_list(
            [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
        )

    train_ds, val_ds = to_dataset("train"), to_dataset("val")
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=build_config(args, run_name, dtype, bf16),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    result = trainer.train()

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    out_dir = exlib.ROOT / "models" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # LoRA parameters are created in fp32 regardless of the base model's dtype,
    # which makes the saved adapter twice the size it needs to be. Inference
    # runs the base model in bf16 or fp16 anyway, so storing the adapter at
    # fp16 costs no usable precision and halves what the repo has to carry.
    # The tokenizer is deliberately not copied here either — it is identical
    # to the base model's and exlib.load_tokenizer fetches it from there.
    import torch

    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)
    trainer.model.save_pretrained(str(out_dir))

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    (out_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "base_model": args.model,
                "run_name": run_name,
                "lora": {
                    "r": args.r,
                    "alpha": alpha,
                    "dropout": 0.05,
                    "target_modules": modules,
                },
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "effective_batch_size": 16,
                "max_length": MAX_LENGTH,
                "seed": exlib.SEED,
                "precision": "bf16" if bf16 else "fp16",
                "trainable_params": trainable,
                "train_runtime_s": round(result.metrics["train_runtime"], 1),
                "final_train_loss": round(result.metrics["train_loss"], 4),
                "eval_loss_per_epoch": [round(h["eval_loss"], 4) for h in history],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\ntrain loss {result.metrics['train_loss']:.4f} "
          f"in {result.metrics['train_runtime']:.0f}s")
    print("eval loss  " + " -> ".join(f"{h['eval_loss']:.4f}" for h in history))
    print(f"adapter    {out_dir}")


if __name__ == "__main__":
    main()
