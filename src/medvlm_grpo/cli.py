import os
import warnings

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from trl import GRPOTrainer, SFTTrainer

from .arguments import parse_args
from .data import load_cold_start_data, load_experiment_datasets
from .precision import resolve_precision
from .rewards import answer_reward_func, clinical_consistency_reward_func, format_reward_func, make_overlong_reward
from .training import FourDigitLogger, TrainingHealthCallback, make_lora_config, make_rl_config, make_sft_config
from .tracking import make_swanlab_callback


def make_callbacks(args, stage):
    callbacks = [FourDigitLogger(), TrainingHealthCallback()]
    swanlab_callback = make_swanlab_callback(
        enabled=args.swanlab,
        project=args.swanlab_project,
        experiment_name=(
            f"{args.swanlab_experiment_name}-{stage}"
            if args.swanlab_experiment_name and args.stage == "pipeline"
            else args.swanlab_experiment_name
        ),
        workspace=args.swanlab_workspace,
        config={
            "stage": stage,
            "model_id": args.model_id,
            "seed": args.seed,
            "num_generations": args.num_generations,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
    )
    if swanlab_callback is not None:
        callbacks.append(swanlab_callback)
    return callbacks


def load_model(model_id, local_rank, load_in_4bit=True, precision="fp16"):
    device_map = {"": local_rank} if local_rank >= 0 else "auto"
    compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=compute_dtype,
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
        callbacks=make_callbacks(args, "sft"),
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
        callbacks=make_callbacks(args, "dapo" if dapo else "grpo"),
    )
    trainer.train()
    trainer.save_model(output_dir)


def main(argv=None):
    args = parse_args(argv)
    args.precision = resolve_precision(args.precision)
    print(f"Using {args.precision.upper()} mixed precision")
    os.makedirs(args.output_dir, exist_ok=True)
    cold_train, train, validation, unseen_test = load_experiment_datasets(
        openmedreason_path=args.openmedreason_path,
        cold_start_size=args.cold_start_size,
        openmedreason_rl_size=args.openmedreason_rl_size,
        validation_size=args.recipe_validation_size,
        max_reasoning_chars=args.cold_start_max_reasoning_chars,
        rl_dataset_names=args.rl_datasets,
        rl_per_dataset_cap=args.rl_per_dataset_cap,
        strict_image_split=args.strict_image_split,
        seed=args.seed,
    )
    print(
        f"Loaded cold_start={len(cold_train)}, rl_train={len(train)}, "
        f"validation={len(validation)}, openmedreason_official_test={len(unseen_test)}"
    )
    cold_validation = validation
    if args.cold_start_data:
        cold_train, cold_validation = load_cold_start_data(args.cold_start_data)
    model, processor = load_model(
        args.model_id,
        args.local_rank,
        args.load_in_4bit,
        precision=args.precision,
    )

    if args.stage == "sft":
        run_sft(model, processor, cold_train, cold_validation, args, args.output_dir)
    elif args.stage in {"grpo", "dapo"}:
        run_rl(model, processor, train, validation, args, args.output_dir, dapo=args.stage == "dapo")
    else:
        cold_start_dir = os.path.join(args.output_dir, "cold-start")
        model = run_sft(model, processor, cold_train, cold_validation, args, cold_start_dir)
        # Continue training the same LoRA adapter so DAPO starts from the SFT
        # policy rather than accidentally attaching a fresh adapter.
        run_rl(model, processor, train, validation, args, os.path.join(args.output_dir, "dapo"), dapo=True, add_adapter=False)


if __name__ == "__main__":
    warnings.filterwarnings("once")
    main()
