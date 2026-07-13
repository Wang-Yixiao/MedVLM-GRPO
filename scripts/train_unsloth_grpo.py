"""Train medical VLM LoRA for exactly 200 GRPO steps with Unsloth + vLLM.

Unsloth's official GRPO integration patches TRL's trainer at import time, so
the two TRL symbols below are the required orchestration shell; model loading,
quantization, LoRA, kernels, checkpointing, and fast generation use Unsloth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medvlm_grpo.unsloth_pipeline.privacy import disable_optional_telemetry

disable_optional_telemetry()  # Must happen before importing Unsloth/TRL.

from unsloth import FastVisionModel, is_bf16_supported
from trl import GRPOConfig, GRPOTrainer

from medvlm_grpo.data import load_medical_vqa
from medvlm_grpo.unsloth_pipeline.monitoring import GRPOMetricsCallback
from medvlm_grpo.unsloth_pipeline.rewards import (
    RewardConfig,
    combine_reward_func,
    configure_reward_engine,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
        help="Use an *-unsloth-bnb-4bit repo for true Unsloth Dynamic 4-bit.",
    )
    parser.add_argument("--dataset", default="SLAKE_VQA_EN", choices=["Vqa_rad", "Vqa_Agupte", "SLAKE_VQA_EN", "Path_VQA"])
    parser.add_argument("--output_dir", default="output/unsloth-grpo")
    parser.add_argument("--semantic_model", default="cross-encoder/stsb-TinyBERT-L4")
    parser.add_argument("--fluency_model", default="microsoft/biogpt")
    parser.add_argument("--reward_device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.55)
    parser.add_argument("--strict_image_split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast_inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )

    train_dataset, _, _ = load_medical_vqa(
        args.dataset, strict_image_split=args.strict_image_split
    )
    train_dataset = train_dataset.select_columns(["prompt", "image", "solution"])

    configure_reward_engine(
        RewardConfig(
            semantic_model=args.semantic_model,
            fluency_model=args.fluency_model,
            device=args.reward_device,
            diagnostics_path=str(output_dir / "reward_components.jsonl"),
            print_every=1,
        )
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        logging_first_step=True,
        log_completions=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=4,
        max_grad_norm=0.1,
        report_to="none",
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        seed=args.seed,
        beta=0.04,  # Enables reference-policy KL logging and regularization.
        loss_type="dr_grpo",
        mask_truncated_completions=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[combine_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[GRPOMetricsCallback(str(output_dir / "training_metrics.jsonl"))],
    )
    trainer.train()

    # Save both a Trainer checkpoint representation and the vLLM-loadable LoRA.
    trainer.save_model(str(output_dir / "trained_adapter"))
    processor.save_pretrained(str(output_dir / "trained_adapter"))
    model.save_lora(str(output_dir / "grpo_lora"))
    print(f"Saved trained adapter to {output_dir / 'trained_adapter'}")
    print(f"Saved vLLM LoRA to {output_dir / 'grpo_lora'}")


if __name__ == "__main__":
    main()
