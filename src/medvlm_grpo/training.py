from inspect import signature

from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, SFTConfig

from .precision import precision_kwargs


def make_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )


def make_sft_config(args, output_dir):
    mixed_precision = precision_kwargs(args.precision)
    return SFTConfig(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1,
        warmup_ratio=0.1,
        weight_decay=0.1,
        logging_steps=1,
        **mixed_precision,
        lr_scheduler_type="cosine",
        save_steps=200,
        report_to="none",
        deepspeed=args.deepspeed,
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def make_rl_config(args, output_dir, dapo=False):
    mixed_precision = precision_kwargs(args.precision)
    common = dict(
        output_dir=output_dir,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        **mixed_precision,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=1.0,
        report_to="none",
        deepspeed=args.deepspeed,
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    # TRL <=0.28 exposed max_prompt_length on GRPOConfig; TRL 0.29 removed it.
    # Keep compatibility with both without passing an unknown dataclass field.
    if "max_prompt_length" in signature(GRPOConfig).parameters:
        common["max_prompt_length"] = args.max_prompt_length
    if dapo:
        common.update(
            # DAPO: token-level normalization and decoupled asymmetric clipping.
            loss_type="dapo",
            epsilon=0.2,
            epsilon_high=0.28,
            beta=0.0,
            # Exclude hard-truncated generations; a separate reward provides a
            # smooth penalty as a completion approaches the limit.
            mask_truncated_completions=True,
        )
    return GRPOConfig(**common)


class FourDigitLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print({k: (f"{v:.4e}" if isinstance(v, float) and abs(v) < 1e-4 else f"{v:.4f}" if isinstance(v, float) else v) for k, v in logs.items()})


class TrainingHealthCallback(TrainerCallback):
    """Surface DAPO collapse/length signals already emitted by modern TRL."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        zero_std = logs.get("frac_reward_zero_std")
        clipped = logs.get("completions/clipped_ratio")
        if isinstance(zero_std, float) and zero_std > 0.5:
            print(f"WARNING: {zero_std:.1%} groups have no reward variance; dynamic resampling is recommended.")
        if isinstance(clipped, float) and clipped > 0.1:
            print(f"WARNING: {clipped:.1%} completions are truncated; lower max length or strengthen length reward.")
