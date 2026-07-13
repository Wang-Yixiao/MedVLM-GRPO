"""Local VQA-RAD bridge-SFT + grouped rollout simulation.

Scope is deliberately small:
1. Load only the first 100 VQA-RAD training rows.
2. LoRA bridge-SFT Qwen2.5-VL-3B on image/question/answer triples.
3. Generate a group of candidates and compute proxy rewards/advantages.

Step 3 simulates GRPO data flow and diagnostics; it does not apply a policy-
gradient update. Use train_unsloth_grpo.py for a real GRPO experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from medvlm_grpo.data import load_medical_vqa
from medvlm_grpo.smoke_rewards import proxy_combined_reward
from medvlm_grpo.unsloth_pipeline.rewards import RewardConfig, RewardEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default=str(ROOT / "models" / "Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--output_dir", default=str(ROOT / "output" / "vqarad-smoke"))
    parser.add_argument("--num_rows", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--rollout_samples", type=int, default=5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--reward_mode", choices=["neural", "proxy"], default="neural")
    parser.add_argument(
        "--semantic_model",
        default=str(ROOT / "reward_model" / "cross-encoder" / "stsb-roberta-base"),
    )
    parser.add_argument(
        "--fluency_model",
        default=str(ROOT / "reward_model" / "microsoft" / "biogpt"),
    )
    parser.add_argument("--reward_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def move_to_device(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def encode_training_example(processor, example, max_length):
    image = example["image"].convert("RGB")
    messages = example["messages"]
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = processor.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full = processor(
        text=[full_text], images=[image], return_tensors="pt", padding=False,
        truncation=True, max_length=max_length,
    )
    prompt = processor(
        text=[prompt_text], images=[image], return_tensors="pt", padding=False,
        truncation=True, max_length=max_length,
    )
    labels = full["input_ids"].clone()
    labels[:, : min(prompt["input_ids"].shape[1], labels.shape[1])] = -100
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    full["labels"] = labels
    return full


def generate_group(model, processor, example, args):
    prompt_text = processor.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_text], images=[example["image"].convert("RGB")], return_tensors="pt"
    )
    inputs = move_to_device(inputs, model.device)
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    return processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )


def build_reward_engine(args, output_dir):
    if args.reward_mode == "proxy":
        return None
    semantic_path = Path(args.semantic_model)
    fluency_path = Path(args.fluency_model)
    required = [
        semantic_path / "config.json",
        semantic_path / "model.safetensors",
        fluency_path / "config.json",
        fluency_path / "pytorch_model.bin",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Neural reward mode requires complete local CrossEncoder and BioGPT files. "
            f"Missing: {missing}. Use --reward_mode proxy only for a lightweight fallback."
        )
    print(f"Loading CrossEncoder reward model from: {semantic_path}")
    print(f"Loading BioGPT reward model from: {fluency_path}")
    print(f"Reward models device: {args.reward_device}")
    return RewardEngine(
        RewardConfig(
            semantic_model=str(semantic_path),
            fluency_model=str(fluency_path),
            device=args.reward_device,
            semantic_batch_size=max(args.num_generations, 1),
            perplexity_batch_size=max(args.num_generations, 1),
            max_reward_tokens=args.max_new_tokens,
            diagnostics_path=str(output_dir / "neural_reward_components.jsonl"),
            print_every=1,
        )
    )


def score_candidates(candidates, reference, reward_engine):
    if reward_engine is None:
        return [
            {
                "semantic_correctness": score["semantic_proxy"],
                "perplexity": None,
                "perplexity_score": score["fluency_proxy"],
                "tag_presence": score["tag_presence"],
                "combined_reward": score["combined_reward"],
                "reward_backend": "proxy",
            }
            for score in (proxy_combined_reward(text, reference) for text in candidates)
        ]
    return [
        asdict(result) | {"reward_backend": "cross-encoder+biogpt"}
        for result in reward_engine.score(candidates, [reference] * len(candidates))
    ]


def run_rollout_simulation(model, processor, dataset, args, output_dir, reward_engine):
    output_path = output_dir / "simulated_grpo_rollouts.jsonl"
    indices = list(range(min(args.rollout_samples, len(dataset))))
    all_rewards = []
    all_lengths = []
    with output_path.open("w", encoding="utf-8") as handle:
        for row_index in indices:
            example = dataset[row_index]
            candidates = generate_group(model, processor, example, args)
            components = score_candidates(candidates, example["solution"], reward_engine)
            rewards = [item["combined_reward"] for item in components]
            mean_reward = statistics.fmean(rewards)
            std_reward = statistics.pstdev(rewards)
            advantages = [0.0 if std_reward < 1e-8 else (value - mean_reward) / std_reward for value in rewards]
            all_rewards.extend(rewards)
            all_lengths.extend(len(text.split()) for text in candidates)
            record = {
                "row": row_index,
                "question": example["prompt"][-1]["content"][-1]["text"],
                "reference": example["solution"],
                "reward_mean": mean_reward,
                "reward_variance": statistics.pvariance(rewards),
                "candidates": [
                    {"output": text, **score, "relative_advantage": advantage, "words": len(text.split())}
                    for text, score, advantage in zip(candidates, components, advantages)
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            best = max(record["candidates"], key=lambda item: item["combined_reward"])
            print(
                f"[simulated GRPO] row={row_index} mean={mean_reward:.4f} "
                f"variance={record['reward_variance']:.6f} best={best['combined_reward']:.4f}"
            )
            print(f"  reference={example['solution']!r}")
            print(f"  best_output={best['output']!r}")
            print(
                f"  components: semantic={best['semantic_correctness']:.4f} "
                f"perplexity={best['perplexity']} "
                f"perplexity_score={best['perplexity_score']:.4f} "
                f"tag={best['tag_presence']:.1f} backend={best['reward_backend']}"
            )
    summary = {
        "rollout_groups": len(indices),
        "num_generations": args.num_generations,
        "reward_mean": statistics.fmean(all_rewards) if all_rewards else 0.0,
        "reward_variance": statistics.pvariance(all_rewards) if len(all_rewards) > 1 else 0.0,
        "mean_output_words": statistics.fmean(all_lengths) if all_lengths else 0.0,
        "reward_mode": args.reward_mode,
        "note": "Diagnostics only: no policy-gradient update was applied.",
    }
    (output_dir / "simulated_grpo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[simulated GRPO summary] {summary}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This 3B smoke test requires an NVIDIA CUDA GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, _, _ = load_medical_vqa("Vqa_rad", strict_image_split=False)
    dataset = train.select(range(min(args.num_rows, len(train))))
    print(f"Loaded VQA-RAD rows: {len(dataset)} (only the first {args.num_rows} requested)")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=64 * 28 * 28,
        max_pixels=256 * 28 * 28,
        use_fast=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0
    for epoch in range(args.epochs):
        for row_index in range(len(dataset)):
            batch = encode_training_example(processor, dataset[row_index], args.max_length)
            batch = move_to_device(batch, model.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            micro_step += 1
            should_step = (
                micro_step % args.gradient_accumulation_steps == 0
                or row_index == len(dataset) - 1
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                print(
                    f"[bridge SFT] epoch={epoch + 1} optimizer_step={optimizer_step} "
                    f"row={row_index + 1}/{len(dataset)} loss={(loss.item() * args.gradient_accumulation_steps):.4f}"
                )

    adapter_dir = output_dir / "bridge_sft_lora"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"Saved bridge-SFT LoRA: {adapter_dir}")

    model.eval()
    model.config.use_cache = True
    reward_engine = build_reward_engine(args, output_dir)
    run_rollout_simulation(model, processor, dataset, args, output_dir, reward_engine)


if __name__ == "__main__":
    main()
