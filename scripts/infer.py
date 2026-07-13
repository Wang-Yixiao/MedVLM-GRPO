"""Single-image inference for a base model or a saved LoRA adapter."""

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medvlm_grpo.data import SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16) if args.load_in_4bit else None
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, device_map="auto", torch_dtype=torch.float16, quantization_config=quant)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}, {"role": "user", "content": [{"type": "image", "image": Image.open(args.image).convert("RGB")}, {"type": "text", "text": args.question}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
