"""Run a 16 GB-friendly OpenMedReason cold-start SFT -> minimal GRPO experiment.

This entry point deliberately avoids TRL/vLLM/Unsloth so it can run on native
Windows with the dependencies already used by this repository.  Its RL stage
is a real on-policy update: sample a group, normalize rewards inside the group,
and optimize completion log-probabilities with the normalized advantages.
For a one-update-per-group on-policy experiment this is the unclipped GRPO
objective with beta=0 (no reference-policy KL term).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image as PILImage
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from medvlm_grpo.data import SYSTEM_PROMPT, TAGGED_REASONING_RE
from medvlm_grpo.smoke_rewards import proxy_combined_reward


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=str(ROOT / "models/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--data_file", default=str(ROOT / "data/neginb/OpenMedReason/data/train-00000-of-00031.parquet"))
    parser.add_argument("--output_dir", default=str(ROOT / "output/openmedreason-100-grpo"))
    parser.add_argument("--sft_rows", type=int, default=100)
    parser.add_argument("--grpo_rows", type=int, default=4)
    parser.add_argument("--validation_rows", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--sft_learning_rate", type=float, default=2e-4)
    parser.add_argument("--grpo_learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_digest(encoded):
    payload = encoded.get("bytes") if isinstance(encoded, dict) else None
    if payload is not None:
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(str(encoded.get("path")).encode("utf-8")).hexdigest()


def prepare_splits(args):
    # Reading one row group once is dramatically faster on native Windows than
    # many random Dataset.__getitem__ calls, each of which may revisit a large
    # compressed image chunk.  A row group contains far more than the 114 rows
    # needed by the default smoke experiment.
    table = pq.ParquetFile(args.data_file).read_row_group(
        0, columns=["image", "question", "reasoning", "answer"]
    )
    raw = table.to_pylist()
    order = list(range(len(raw)))
    random.Random(args.seed).shuffle(order)
    needed = args.sft_rows + args.grpo_rows + args.validation_rows
    chosen, seen_images = [], set()
    for index in order:
        row = raw[index]
        reasoning = str(row.get("reasoning", "")).strip()
        if not row.get("question") or not row.get("answer") or not TAGGED_REASONING_RE.match(reasoning):
            continue
        digest = image_digest(row["image"])
        if digest in seen_images:
            continue
        seen_images.add(digest)
        chosen.append(index)
        if len(chosen) == needed:
            break
    if len(chosen) < needed:
        raise RuntimeError(f"Only {len(chosen)} eligible image-disjoint rows found; need {needed}")
    selected = [raw[index] for index in chosen]

    def convert(row):
        content = [{"type": "image"}, {"type": "text", "text": str(row["question"]).strip()}]
        prompt = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return {
            "image": PILImage.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"),
            "question": str(row["question"]).strip(),
            "solution": str(row["answer"]).strip(),
            "target": str(row["reasoning"]).strip(),
            "prompt": prompt,
            "messages": prompt + [{"role": "assistant", "content": [{"type": "text", "text": str(row["reasoning"]).strip()}]}],
        }

    rows = [convert(row) for row in selected]
    a, b = args.sft_rows, args.sft_rows + args.grpo_rows
    return rows[:a], rows[a:b], rows[b:]


def move(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def encode_sft(processor, row, max_length):
    full_text = processor.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    full = processor(text=[full_text], images=[row["image"].convert("RGB")], return_tensors="pt", truncation=True, max_length=max_length)
    prompt = processor(text=[prompt_text], images=[row["image"].convert("RGB")], return_tensors="pt", truncation=True, max_length=max_length)
    labels = full["input_ids"].clone()
    labels[:, : min(prompt["input_ids"].shape[1], labels.shape[1])] = -100
    full["labels"] = labels
    return full


def generate_group(model, processor, row, args):
    prompt_text = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt_text], images=[row["image"].convert("RGB")], return_tensors="pt")
    inputs = move(inputs, model.device)
    prompt_tokens = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        ids = model.generate(
            **inputs, do_sample=True, temperature=1.0, top_p=0.95,
            num_return_sequences=args.num_generations,
            max_new_tokens=args.max_new_tokens, use_cache=True,
        )
    texts = processor.batch_decode(ids[:, prompt_tokens:], skip_special_tokens=True)
    return texts


def completion_logprob(model, processor, row, completion, max_length):
    prompt_text = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    prompt = processor(text=[prompt_text], images=[row["image"].convert("RGB")], return_tensors="pt", truncation=True, max_length=max_length)
    full = processor(text=[prompt_text + completion], images=[row["image"].convert("RGB")], return_tensors="pt", truncation=True, max_length=max_length)
    prompt_length = min(prompt["input_ids"].shape[1], full["input_ids"].shape[1] - 1)
    full = move(full, model.device)
    logits = model(**full).logits[:, :-1].float()
    targets = full["input_ids"][:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    completion_logps = token_logps[:, max(prompt_length - 1, 0):]
    if completion_logps.numel() == 0:
        raise RuntimeError("Completion was fully truncated; reduce prompt/image tokens or increase max_length")
    return completion_logps.mean(), int(completion_logps.numel())


def adapter_fingerprint(model):
    values = [p.detach().float().sum().item() for n, p in model.named_parameters() if p.requires_grad]
    return statistics.fmean(values), sum(abs(value) for value in values)


def evaluate(model, processor, rows, args):
    records = []
    model.eval(); model.config.use_cache = True
    for row in rows:
        one_args = argparse.Namespace(**vars(args)); one_args.num_generations = 1
        output = generate_group(model, processor, row, one_args)[0]
        score = proxy_combined_reward(output, row["solution"])
        records.append({"question": row["question"], "reference": row["solution"], "output": output, **score})
    return records


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an NVIDIA CUDA GPU")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sft_rows, grpo_rows, validation_rows = prepare_splits(args)
    manifest = {
        "seed": args.seed, "source_shard": args.data_file,
        "sft_rows": len(sft_rows), "grpo_rows": len(grpo_rows), "validation_rows": len(validation_rows),
        "split_policy": "seeded shuffle; one row per encoded-image SHA256; disjoint sequential partitions",
        "grpo_objective": "one on-policy update/group, normalized group reward, beta=0, no PPO clipping needed",
    }
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=64*28*28, max_pixels=256*28*28, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.sft_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=True); optimizer.zero_grad(set_to_none=True)
    sft_log = []
    for epoch in range(args.epochs):
        for index, row in enumerate(sft_rows):
            batch = move(encode_sft(processor, row, args.max_length), model.device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            raw_loss = loss.item() * args.gradient_accumulation_steps
            if (index + 1) % args.gradient_accumulation_steps == 0 or index + 1 == len(sft_rows):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                item = {"epoch": epoch+1, "row": index+1, "loss": raw_loss, "max_cuda_gib": torch.cuda.max_memory_allocated()/2**30}
                sft_log.append(item); print(f"[SFT] {item}", flush=True)
    sft_dir = output_dir / "cold_start_lora"; model.save_pretrained(sft_dir); processor.save_pretrained(sft_dir)

    before = evaluate(model, processor, validation_rows, args)
    before_fp = adapter_fingerprint(model)
    for group in optimizer.param_groups: group["lr"] = args.grpo_learning_rate
    grpo_log = []
    for row_index, row in enumerate(grpo_rows):
        model.eval(); model.config.use_cache = True
        candidates = generate_group(model, processor, row, args)
        components = [proxy_combined_reward(text, row["solution"]) for text in candidates]
        rewards = [x["combined_reward"] for x in components]
        mean, std = statistics.fmean(rewards), statistics.pstdev(rewards)
        advantages = [0.0 if std < 1e-8 else (reward-mean)/(std+1e-8) for reward in rewards]
        optimizer.zero_grad(set_to_none=True); model.train(); model.config.use_cache = False
        policy_losses, token_counts = [], []
        for candidate, advantage in zip(candidates, advantages):
            logp, count = completion_logprob(model, processor, row, candidate, args.max_length)
            policy_losses.append(-float(advantage) * logp); token_counts.append(count)
        loss = torch.stack(policy_losses).mean()
        if std >= 1e-8:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        item = {
            "row": row_index, "question": row["question"], "reference": row["solution"],
            "reward_mean": mean, "reward_std": std, "loss": loss.detach().item(),
            "updated": std >= 1e-8, "candidates": [
                {"output": text, **score, "advantage": advantage, "tokens": tokens}
                for text, score, advantage, tokens in zip(candidates, components, advantages, token_counts)
            ],
        }
        grpo_log.append(item); print(f"[GRPO] row={row_index} reward={mean:.4f}+/-{std:.4f} loss={item['loss']:.4f} updated={item['updated']}", flush=True)
    after_fp = adapter_fingerprint(model)
    grpo_dir = output_dir / "grpo_lora"; model.save_pretrained(grpo_dir); processor.save_pretrained(grpo_dir)
    after = evaluate(model, processor, validation_rows, args)

    with (output_dir / "sft_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in sft_log: f.write(json.dumps(item, ensure_ascii=False)+"\n")
    with (output_dir / "grpo_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in grpo_log: f.write(json.dumps(item, ensure_ascii=False)+"\n")
    with (output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for phase, rows in (("after_sft", before), ("after_grpo", after)):
            for item in rows: f.write(json.dumps({"phase": phase, **item}, ensure_ascii=False)+"\n")
    summary = {
        **manifest, "trainable_parameters": trainable,
        "sft_loss_first": sft_log[0]["loss"], "sft_loss_last": sft_log[-1]["loss"],
        "grpo_groups_updated": sum(x["updated"] for x in grpo_log),
        "grpo_reward_mean": statistics.fmean(x["reward_mean"] for x in grpo_log),
        "validation_reward_after_sft": statistics.fmean(x["combined_reward"] for x in before),
        "validation_reward_after_grpo": statistics.fmean(x["combined_reward"] for x in after),
        "adapter_fingerprint_before_grpo": before_fp, "adapter_fingerprint_after_grpo": after_fp,
        "adapter_fingerprint_changed": before_fp != after_fp,
        "max_cuda_gib": torch.cuda.max_memory_allocated()/2**30,
        "elapsed_minutes": (time.time()-started)/60,
        "limitations": ["single OpenMedReason shard", "proxy reward", "tiny GRPO/validation sets", "beta=0; no reference KL"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
