"""Run a model/adapter over an entire Medical-VQA split and write JSONL."""

import argparse
import json
from pathlib import Path
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medvlm_grpo.data import SYSTEM_PROMPT, load_medical_vqa
from medvlm_grpo.metrics import classify_question


def question_from_prompt(prompt):
    for item in reversed(prompt[-1]["content"]):
        if item.get("type") == "text":
            return item["text"]
    return ""


def messages_for(image, question):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", default="Vqa_rad", choices=["Vqa_rad", "Vqa_Agupte", "SLAKE_VQA_EN", "Path_VQA"])
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--strict_image_split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, test, validation = load_medical_vqa(args.dataset, strict_image_split=args.strict_image_split)
    del train
    dataset = test if args.split == "test" else validation
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if args.load_in_4bit else None
    processor = AutoProcessor.from_pretrained(args.model_id)
    # Decoder-only generation requires left padding so every row starts
    # generation immediately after its own prompt rather than after pad tokens.
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16, quantization_config=quant
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and output.exists():
        completed = {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        pending = [i for i in range(len(dataset)) if i not in completed]
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start:start + args.batch_size]
            rows = [dataset[i] for i in indices]
            questions = [question_from_prompt(row["prompt"]) for row in rows]
            messages = [messages_for(row["image"], question) for row, question in zip(rows, questions)]
            texts = [processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in messages]
            inputs = processor(text=texts, images=[row["image"] for row in rows], padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            generated = generated[:, inputs.input_ids.shape[1]:]
            predictions = processor.batch_decode(generated, skip_special_tokens=True)
            for index, row, question, prediction in zip(indices, rows, questions, predictions):
                reference = row["solution"]
                record = {
                    "id": index,
                    "dataset": args.dataset,
                    "split": args.split,
                    "seed": args.seed,
                    "question": question,
                    "question_type": classify_question(question, reference),
                    "prediction": prediction.strip(),
                    "reference": reference,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"generated {min(start + len(indices), len(pending))}/{len(pending)}")


if __name__ == "__main__":
    main()
