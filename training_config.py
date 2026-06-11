from trl import GRPOConfig, GRPOTrainer
from trl import SFTConfig, SFTTrainer
from args import parse_args
from peft import LoraConfig, get_peft_model # prepare_model_for_kbit_training, PeftModel
DS_CONFIG = "./ds_z2_offload_config.json"
peft_config = LoraConfig(
    r=8, #Rank
    lora_alpha=16,
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        # "gate_proj", 
        # "up_proj", 
        # "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
)

training_args_GRPO = GRPOConfig(
        # use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        # optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = False,
        fp16 = True,
        per_device_train_batch_size = 2,# keep same with num_generations 1->2 723->362
        gradient_accumulation_steps = 16, #  # 16 - > 4 1446 ->5784 16 - > 32 723
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = 2048,
        max_completion_length = 2048,
        num_train_epochs = 1, # Set to 1 for a full training run
        # max_steps = 100,
        save_steps = 5,
        max_grad_norm = 1.0,
        report_to = "none", # Can use Weights & Biases
        output_dir = parse_args.output_dir,
        deepspeed=DS_CONFIG,
        disable_tqdm=False,  # 确保不禁用进度条
    )
    # peft_model = get_peft_model(model, peft_config)




training_args_SFT = SFTConfig(
    learning_rate=5e-5,                 # SFT 通常比 RL 需要更大学习率
    per_device_train_batch_size=3,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    warmup_ratio=0.1,
    weight_decay=0.1,
    logging_steps=1,
    fp16=True,
    bf16=False,
    lr_scheduler_type="cosine",
    # optim="adamw_8bit",
    save_steps=200,
    output_dir=parse_args.output_dir,
    report_to="none",
    deepspeed=DS_CONFIG,               # 继续使用你的 ZeRO2 offload
    )

from transformers import TrainerCallback

class FourDigitLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            formatted = {}
            for k, v in logs.items():
                if isinstance(v, float):
                    if abs(v) < 1e-4:
                        formatted[k] = f"{v:.4e}"
                    else:
                        formatted[k] = f"{v:.4f}"
                else:
                    formatted[k] = v
            print(formatted)
