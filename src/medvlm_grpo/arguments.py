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
    parser.add_argument("--cold_start_data", default=None, help="Optional JSON/JSONL with image, question, answer and reasoning fields.")
    parser.add_argument("--gemex_path", default="data/BoKelvin/GEMeX-VQA")
    parser.add_argument("--gemex_image_root", default=None, help="Optional MIMIC-CXR root containing GEMeX image_path files.")
    parser.add_argument("--slake_path", default="data/mdwiratathya/SLAKE-vqa-english")
    parser.add_argument("--agupte_path", default="data/agupte/MedVQA")
    parser.add_argument("--test_size", type=int, default=10, help="Number of agupte official-test examples used for unseen evaluation.")
    parser.add_argument("--strict_image_split", action="store_true", help="Remove training images duplicated in test/validation.")
    parser.add_argument("--deepspeed", default=None, help="Optional DeepSpeed JSON config.")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    args = parser.parse_args(argv)
    if args.grpo:
        args.stage = "grpo"
    return args
