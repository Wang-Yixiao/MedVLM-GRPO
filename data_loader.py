import os
import warnings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


import torch
DS_CONFIG = "./ds_z2_offload_config.json"
from datasets import load_dataset,Dataset

from functools import partial

SYSTEM_PROMPT = r'''
            Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Before answering, think carefully about the question and create a step-by-step chain of
            thoughts to ensure a logical and accurate response.

            ### Instruction:
            You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
            Please answer the following medical question based on the input image. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.The output format should be as follows:
    <think> ...  </think> <answer>...</answer>'''
    

def format_conversation(example, grpo):
        if grpo:
            return {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": "SYSTEM_PROMPT"}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": example["prompt"]},
                        ],
                    },
                ]
            }
        else:
            return {
                'prompt': [
                    {'role': 'system', 'content': "SYSTEM_PROMPT"},
                    {'role': 'user', 'content': example["prompt"]},
                ],
                'images': [example["images"]],
                'completion':[{'content': example["completion"], 'role': 'user'}]
            }
            
def resize_image(example):
    if 'images' in example:
        img_field = 'images'
    elif 'image' in example:
        img_field = 'image'
    else:
        raise NotImplementError('not find images')
    example[img_field] = example[img_field].resize((140, 140))
    return example


def load_Medical_Vqa_Agupte(grpo):
    from datasets import Dataset, Image

    dataset_dict = load_dataset(path='./dataset/medical-vqa_agupte')
    dataset_dict = dataset_dict.map(resize_image)
    dataset_dict = dataset_dict.remove_columns(['ids', 'image_names'])
    dataset_dict = dataset_dict.rename_column('questions', 'prompt')
    if grpo:
        dataset_dict = dataset_dict.rename_column('answers', 'solution')
        dataset_dict = dataset_dict.rename_column('images', 'image')
    else:
        dataset_dict = dataset_dict.rename_column('answers', 'completion')

        # rename 后 image 列名是 "images"，保持一致即可
    
    dataset_dict = dataset_dict.map(partial(format_conversation, grpo=grpo))
    if not grpo:
        dataset_dict = dataset_dict.cast_column("images", [Image()])

    train_val = dataset_dict['train'].train_test_split(test_size=0.2)
    return train_val['train'], dataset_dict['test'], train_val['test']


def load_Path_Vqa(grpo):
    from datasets import Dataset, Image
    dataset_dict = load_dataset(path='./dataset/path-vqa')
    dataset_dict = dataset_dict.rename_column('question', 'prompt')
    dataset_dict = dataset_dict.map(resize_image)
    if grpo:
        dataset_dict = dataset_dict.rename_column('answer', 'solution')
        # dataset_dict = dataset_dict.rename_column('images', 'image')
    else:
        dataset_dict = dataset_dict.rename_column('answer', 'completion')
        dataset_dict = dataset_dict.rename_column('image', 'images')


    dataset_dict = dataset_dict.map(partial(format_conversation, grpo=grpo))
    if not grpo:
        dataset_dict = dataset_dict.cast_column("images", [Image()])
    # print("Train dataset type:", type(dataset_dict))
    # print("Train dataset length:", len(dataset_dict))
    return dataset_dict['train'],dataset_dict['test'],dataset_dict['validation']

    
def load_SLAKE_VQA_EN(grpo):
    from datasets import Dataset, Image
    dataset_dict = load_dataset(path='./dataset/SLAKE_VQA_EN')
    dataset_dict = dataset_dict.rename_column('question', 'prompt')
    if grpo:
        dataset_dict = dataset_dict.rename_column('answer', 'solution')
        # dataset_dict = dataset_dict.rename_column('images', 'image')
    else:
        dataset_dict = dataset_dict.rename_column('answer', 'completion')
        dataset_dict = dataset_dict.rename_column('image', 'images')

    dataset_dict = dataset_dict.map(partial(format_conversation, grpo=grpo))
    if not grpo:
        dataset_dict = dataset_dict.cast_column("images", [Image()])
    # print("Train dataset type:", type(dataset_dict))
    # print("Train dataset length:", len(dataset_dict))
    return dataset_dict['train'],dataset_dict['test'],dataset_dict['validation']

def load_Medical_Vqa_rad(grpo):
    from datasets import Dataset, Image
   
    dataset_dict = load_dataset(path='./dataset/vqa-rad')
    dataset_dict = dataset_dict.map(resize_image)
    dataset_dict = dataset_dict.rename_column('question', 'prompt')
    if grpo:
        dataset_dict = dataset_dict.rename_column('answer', 'solution')
    else:
        dataset_dict = dataset_dict.rename_column('answer', 'completion')
        dataset_dict = dataset_dict.rename_column('image', 'images')

        # rename 后 image 列名是 "images"，保持一致即可
  
    dataset_dict = dataset_dict.map(partial(format_conversation, grpo=grpo))
    if not grpo:
        dataset_dict = dataset_dict.cast_column("images", [Image()])

    train_val = dataset_dict['train'].train_test_split(test_size=0.2)
    return train_val['train'], dataset_dict['test'], train_val['test']
