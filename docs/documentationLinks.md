# Documentation links

## Assignment

- [`STAI_M1_Asignacion_Fine-tuning_baseline.pdf`](STAI_M1_Asignacion_Fine-tuning_baseline.pdf) — the M1 brief and rubric

## Model

- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) — the base model (Apache-2.0, ungated)
- [Qwen3 technical report](https://arxiv.org/abs/2505.09388)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — the M3 upgrade path, same tokenizer family

## Fine-tuning stack

- [TRL `SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) — training loop; note `completion_only_loss` defaults to `True` for prompt-completion datasets
- [TRL dataset formats](https://huggingface.co/docs/trl/en/dataset_formats) — the conversational prompt/completion shape `02_prepare_data.py` emits
- [PEFT `LoraConfig`](https://huggingface.co/docs/peft/en/package_reference/lora)
- [PEFT conceptual guide to LoRA](https://huggingface.co/docs/peft/en/conceptual_guides/lora)
- [LoRA paper](https://arxiv.org/abs/2106.09685) — where `r` and `alpha` come from
- [QLoRA paper](https://arxiv.org/abs/2305.14314) — the "adapt all linear layers" finding behind our `target_modules`
- [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating) — and Qwen3's `enable_thinking` flag

## Tooling

- [uv docs](https://docs.astral.sh/uv/)
- [Installation](https://docs.astral.sh/uv/getting-started/installation/)
- [Working on projects](https://docs.astral.sh/uv/guides/projects/)
- [Running scripts](https://docs.astral.sh/uv/guides/scripts/)
- [Managing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [Locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [CLI reference](https://docs.astral.sh/uv/reference/cli/)
- [PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [GitHub repo](https://github.com/astral-sh/uv)

## Hardware notes

- [CUDA compute capabilities](https://developer.nvidia.com/cuda-gpus) — Colab's free T4 is Turing (sm_75): no bf16, no FlashAttention-2
- [Colab FAQ](https://research.google.com/colaboratory/faq.html) — free-tier GPU allocation and session limits
