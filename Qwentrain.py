import os 
import warnings
import argparse
warnings.filterwarnings("ignore")  # 忽略所有 UserWarning / RuntimeWarning 等
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from qwen_vl_utils import process_vision_info
from transformers import (
Qwen2_5_VLForConditionalGeneration, 
AutoTokenizer, 
AutoProcessor,
BitsAndBytesConfig
)
import torch
import deepspeed
import json
from datasets import load_dataset,Dataset
from PIL import Image
from args import parse_args

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)
from training_config import FourDigitLogger
from metrics import compute_metrics_test
arguments = parse_args()
Model_id = arguments.model_id
# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
local_rank = arguments.local_rank if arguments.local_rank != -1 else 0
device_map = {"": local_rank}
tokenizer = AutoProcessor.from_pretrained(Model_id, use_fast=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(Model_id, 
                                            device_map=device_map, 
                                            torch_dtype=compute_dtype,
                                            quantization_config=bnb_config)

if arguments.dataset == "Vqa_rad": # OK
    from data_loader import load_Medical_Vqa_rad
    ds_train, ds_test, ds_val = load_Medical_Vqa_rad(grpo=arguments.grpo)
elif arguments.dataset == "Vqa_Agupte":  # OK
    from data_loader import load_Medical_Vqa_Agupte
    ds_train, ds_test, ds_val = load_Medical_Vqa_Agupte(grpo=arguments.grpo)
elif arguments.dataset == "SLAKE_VQA_EN": # OK
    from data_loader import load_SLAKE_VQA_EN
    ds_train, ds_test, ds_val = load_SLAKE_VQA_EN(grpo=arguments.grpo)
elif arguments.dataset == "Path_VQA":  # OK
    from data_loader import load_Path_Vqa
    ds_train, ds_test, ds_val = load_Path_Vqa(grpo=arguments.grpo)
output_dir=arguments.output_dir
print(ds_train,ds_test,ds_val)
print("arguments:\n",arguments,'\n')
if arguments.grpo:
    from training_config import training_args_GRPO,peft_config
    from trl import GRPOTrainer
    from reward_func import format_reward_func,levenshtein_reward_func
    # from training_config import GRPOTrainer
    training_args_GRPO.output_dir = output_dir
    trainer_GRPO = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func, # all reward functions
            levenshtein_reward_func],
        args=training_args_GRPO,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        # compute_metrics=compute_metrics_test,
        peft_config = peft_config,
        callbacks=[FourDigitLogger()],
    )
    trainer_GRPO.train()
    trainer_GRPO.save_model(output_dir)
else:
    from training_config import training_args_SFT,peft_config
    training_args_SFT.output_dir = output_dir
    from trl import SFTTrainer
    trainer_SFT = SFTTrainer(
    # processing_class=tokenizer,
    model=model,
    compute_metrics=compute_metrics_test,
    args=training_args_SFT,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    peft_config=peft_config,
    callbacks=[FourDigitLogger()],
    )
    trainer_SFT.train()
    trainer_SFT.save_model(output_dir)


