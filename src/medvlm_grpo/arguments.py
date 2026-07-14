import argparse
import os


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cold-start and GRPO/DAPO training for Qwen2.5-VL medical VQA.")
    parser.add_argument("--model_id", required=True, help="Local path or Hugging Face model id.")
    parser.add_argument(
        "--stage",
        choices=["sft", "grpo", "dapo", "pipeline"],
        default="pipeline",
        help="pipeline runs cold-start SFT followed by DAPO.",
    )
    # Backwards-compatible alias used by the original README.
    parser.add_argument("--grpo", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", default="./output/qwen-vl-med")
    parser.add_argument("--dataset", choices=["Vqa_rad", "Vqa_Agupte", "SLAKE_VQA_EN", "Path_VQA"], default="Vqa_rad", help=argparse.SUPPRESS)
    parser.add_argument("--cold_start_data", default=None, help="Optional JSON/JSONL override for OpenMedReason cold-start SFT.")
    parser.add_argument("--openmedreason_path", default="data/neginb/OpenMedReason")
    parser.add_argument("--cold_start_size", type=int, default=10_000, help="Target OpenMedReason SFT rows; image groups are never split.")
    parser.add_argument(
        "--openmedreason_rl_size",
        type=int,
        default=30_000,
        help="Target disjoint OpenMedReason RL rows; use 0 to include every remaining eligible row.",
    )
    parser.add_argument("--recipe_validation_size", type=int, default=1_000, help="Held-out OpenMedReason train rows used for SFT/RL validation.")
    parser.add_argument("--cold_start_max_reasoning_chars", type=int, default=1_600, help="Drop overly long OpenMedReason cold-start targets.")
    parser.add_argument(
        "--rl_datasets",
        nargs="*",
        choices=["Vqa_rad", "Vqa_Agupte", "SLAKE_VQA_EN", "Path_VQA"],
        default=["SLAKE_VQA_EN", "Vqa_rad", "Vqa_Agupte", "Path_VQA"],
        help="Answer-only VQA datasets mixed into RL, never SFT.",
    )
    parser.add_argument("--rl_per_dataset_cap", type=int, default=5_000, help="Maximum rows contributed by each answer-only VQA dataset; 0 keeps all.")
    parser.add_argument(
        "--strict_image_split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove VQA training images duplicated in test/validation (enabled by default).",
    )
    parser.add_argument("--deepspeed", default=None, help="Optional DeepSpeed JSON config.")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16"],
        default="auto",
        help="Training precision. auto prefers BF16 when the CUDA device supports it.",
    )
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=False, help="Enable SwanLab experiment tracking.")
    parser.add_argument("--swanlab_project", default="medvlm-grpo")
    parser.add_argument("--swanlab_experiment_name", default=None)
    parser.add_argument("--swanlab_workspace", default=None)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    args = parser.parse_args(argv)
    if args.grpo:
        args.stage = "grpo"
    return args
