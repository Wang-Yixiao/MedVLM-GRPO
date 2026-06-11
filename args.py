import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL with GRPO using a specified model path.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Path or Hugging Face model ID to the Qwen2.5-VL model (e.g., './model/Qwen/Qwen2.5-VL-3B-Instruct')"
    )
    parser.add_argument(
    "--grpo",
    action="store_true",
    help="Use GRPO training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/Qwevl-Instruct-GRPO",
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local rank for distributed training (used by DeepSpeed)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["Vqa_rad", "Vqa_Agupte","SLAKE_VQA_EN", "Path_VQA"],
        default="BUAADreamer",
        help="Dataset Name"
    )
    return parser.parse_args()
