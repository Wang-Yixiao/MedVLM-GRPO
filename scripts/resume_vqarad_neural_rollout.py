"""Resume the neural reward rollout from an already saved bridge-SFT LoRA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from local_vqarad_bridge_smoke import build_reward_engine, run_rollout_simulation
from medvlm_grpo.data import load_medical_vqa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=str(ROOT / "models" / "Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--adapter_dir", default=str(ROOT / "output" / "vqarad-smoke" / "bridge_sft_lora"))
    parser.add_argument("--output_dir", default=str(ROOT / "output" / "vqarad-smoke"))
    parser.add_argument("--num_rows", type=int, default=100)
    parser.add_argument("--rollout_samples", type=int, default=5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--semantic_model", default=str(ROOT / "reward_model" / "cross-encoder" / "stsb-roberta-base"))
    parser.add_argument("--fluency_model", default=str(ROOT / "reward_model" / "microsoft" / "biogpt"))
    parser.add_argument("--reward_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.reward_mode = "neural"
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28, use_fast=True
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=False)
    model.eval()
    train, _, _ = load_medical_vqa("Vqa_rad", strict_image_split=False)
    dataset = train.select(range(min(args.num_rows, len(train))))
    reward_engine = build_reward_engine(args, output_dir)
    run_rollout_simulation(model, processor, dataset, args, output_dir, reward_engine)


if __name__ == "__main__":
    main()
