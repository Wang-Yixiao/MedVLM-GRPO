import os
import warnings

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from trl import GRPOTrainer, SFTTrainer

from .arguments import parse_args
from .data import load_cold_start_data, load_experiment_datasets
from .rewards import answer_reward_func, clinical_consistency_reward_func, format_reward_func, make_overlong_reward
from .training import FourDigitLogger, TrainingHealthCallback, make_lora_config, make_rl_config, make_sft_config


def load_model(model_id, local_rank, load_in_4bit=True):
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.float16,
        quantization_config=quantization if load_in_4bit else None,
    )
    return model, processor


def run_sft(model, processor, train, validation, args, output_dir, add_adapter=True):
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=make_sft_config(args, output_dir),
        train_dataset=train.select_columns(["messages", "image"]),
        eval_dataset=validation.select_columns(["messages", "image"]),
        peft_config=make_lora_config() if add_adapter else None,
        callbacks=[FourDigitLogger(), TrainingHealthCallback()],
    )
    trainer.train()
    trainer.save_model(output_dir)
    return trainer.model


def run_rl(model, processor, train, validation, args, output_dir, dapo, add_adapter=True):
    rewards = [format_reward_func, answer_reward_func, clinical_consistency_reward_func]
    if dapo:
        rewards.append(make_overlong_reward(args.max_completion_length))
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=rewards,
        args=make_rl_config(args, output_dir, dapo=dapo),
        train_dataset=train.select_columns(["prompt", "image", "solution"]),
        eval_dataset=validation.select_columns(["prompt", "image", "solution"]),
        peft_config=make_lora_config() if add_adapter else None,
        callbacks=[FourDigitLogger(), TrainingHealthCallback()],
    )
    trainer.train()
    trainer.save_model(output_dir)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    cold_train, train, validation, unseen_test = load_experiment_datasets(
        gemex_path=args.gemex_path,
        slake_path=args.slake_path,
        agupte_path=args.agupte_path,
        gemex_image_root=args.gemex_image_root,
        test_size=args.test_size,
    )
    print(f"Loaded cold_start={len(cold_train)}, train={len(train)}, validation={len(validation)}, unseen_test={len(unseen_test)}")
    model, processor = load_model(args.model_id, args.local_rank, args.load_in_4bit)

    if args.stage == "sft":
        run_sft(model, processor, train, validation, args, args.output_dir)
    elif args.stage in {"grpo", "dapo"}:
        run_rl(model, processor, train, validation, args, args.output_dir, dapo=args.stage == "dapo")
    else:
        cold_start_dir = os.path.join(args.output_dir, "cold-start")
        if args.cold_start_data:
            cold_train, cold_validation = load_cold_start_data(args.cold_start_data)
        else:
            cold_validation = validation
        model = run_sft(model, processor, cold_train, cold_validation, args, cold_start_dir)
        # Continue training the same LoRA adapter so DAPO starts from the SFT
        # policy rather than accidentally attaching a fresh adapter.
        run_rl(model, processor, train, validation, args, os.path.join(args.output_dir, "dapo"), dapo=True, add_adapter=False)


if __name__ == "__main__":
    warnings.filterwarnings("once")
    main()
