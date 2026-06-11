# MedVLM-GRPO

A GRPO-based reinforcement learning framework for improving structured clinical reasoning in Medical Vision-Language Models.

This project explores how Group Relative Policy Optimization (GRPO), LoRA fine-tuning, and rule-based reward design can be used to improve medical visual question answering (Med-VQA). The framework trains Qwen2.5-VL models to generate responses with explicit reasoning and concise answers using a structured output format:

```text
<think> ... </think> <answer> ... </answer>
```

The goal is to improve reasoning stability, answer accuracy, and clinical preference alignment under limited medical annotations.

## Overview

Medical Vision-Language Models have shown strong potential for clinical decision support, but several challenges remain:

* Existing methods often rely heavily on manually annotated data.
* Generated answers may lack interpretable reasoning.
* Medical VQA systems can produce unstable or hallucinated responses.
* Open-ended and closed-ended clinical questions require different answer styles.

To address these issues, this project implements a lightweight reinforcement learning pipeline for medical VQA. The model is encouraged to produce well-formatted, clinically grounded, and concise responses through a multi-signal reward function.

## Main Features

* GRPO training for Qwen2.5-VL models.
* LoRA-based parameter-efficient fine-tuning.
* Rule-based reward functions for:

  * output format correctness,
  * answer similarity,
  * structured reasoning behavior.
* Support for both GRPO and SFT training modes.
* Dataset loaders for multiple medical VQA datasets.
* DeepSpeed ZeRO-2 offload support.
* Evaluation utilities for medical VQA generation metrics.

## Method

The training framework is based on three core components.

### 1. Structured Medical Reasoning Prompt

The model is prompted to first reason inside `<think>` tags and then provide the final answer inside `<answer>` tags.

Example:

```text
<think>The cardiac borders are clearly visible without obvious obscuration.</think>
<answer>yes</answer>
```

This format is designed to make model outputs more interpretable and easier to evaluate.

### 2. GRPO Training

GRPO is used as the reinforcement learning algorithm. Compared with PPO-style RLHF, GRPO avoids training a separate value model and estimates relative advantages from groups of sampled responses.

This makes it more suitable for resource-constrained medical VLM fine-tuning.

### 3. Rule-Based Reward Design

The current reward design includes two implemented reward functions:

| Reward             | Purpose                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| Format reward      | Checks whether the output follows the required `<think>` and `<answer>` format |
| Levenshtein reward | Measures similarity between the predicted answer and the ground-truth answer   |

The intended design can be extended to include additional rewards such as BERTScore-based semantic similarity and CoT quality scoring.

## Supported Models

The current implementation focuses on Qwen2.5-VL models, for example:

* `Qwen/Qwen2.5-VL-3B-Instruct`
* `Qwen/Qwen2.5-VL-7B-Instruct`

The project structure can also be extended to other medical VLMs such as LLaVA-based models.

## Supported Datasets

The code currently includes dataset loaders for:

* VQA-RAD
* SLAKE VQA EN
* PathVQA
* Medical VQA Agupte

Expected local dataset structure:

```text
dataset/
├── vqa-rad/
├── SLAKE_VQA_EN/
├── path-vqa/
└── medical-vqa_agupte/
```

## Project Structure

```text
MedVLM-GRPO/
├── args.py                 # Command-line arguments
├── data_loader.py          # Medical VQA dataset loading and preprocessing
├── metrics.py              # Evaluation metric utilities
├── Qwentrain.py            # Main training script for GRPO/SFT
├── reward_func.py          # Rule-based reward functions
├── training_config.py      # LoRA, GRPO, SFT, and DeepSpeed training configs
├── ds_z2_offload_config.json
└── README.md
```

## Installation

Create a Python environment:

```bash
conda create -n medvlm-grpo python=3.10
conda activate medvlm-grpo
```

Install dependencies:

```bash
pip install torch torchvision
pip install transformers datasets accelerate peft trl deepspeed
pip install qwen-vl-utils evaluate nltk rouge-score python-Levenshtein
```

Depending on your CUDA version and hardware, you may need to install a specific PyTorch build from the official PyTorch website.

## Training

### GRPO Training

```bash
python Qwentrain.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset Vqa_rad \
  --grpo \
  --output_dir ./output/Qwen2.5-VL-3B-MedVQA-GRPO
```

### SFT Training

```bash
python Qwentrain.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset Vqa_rad \
  --output_dir ./output/Qwen2.5-VL-3B-MedVQA-SFT
```

### Available Dataset Arguments

```text
Vqa_rad
Vqa_Agupte
SLAKE_VQA_EN
Path_VQA
```

## Training Configuration

The current configuration uses LoRA for parameter-efficient fine-tuning:

```text
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.05
Target modules: q_proj, k_proj, v_proj, o_proj
```

GRPO configuration:

```text
learning rate: 5e-6
per-device train batch size: 2
gradient accumulation steps: 16
num generations: 4
max prompt length: 2048
max completion length: 2048
epochs: 1
precision: fp16
DeepSpeed: ZeRO-2 offload
```

SFT configuration:

```text
learning rate: 5e-5
per-device train batch size: 3
gradient accumulation steps: 16
epochs: 1
precision: fp16
DeepSpeed: ZeRO-2 offload
```

## Evaluation

The project is designed for both closed-ended and open-ended medical VQA.

Closed-ended questions are usually evaluated with accuracy.

Open-ended questions can be evaluated with generation metrics such as:

* ROUGE-L
* METEOR
* BERTScore Precision
* BERTScore Recall

The project report compares GRPO and SFT across multiple medical VQA datasets and model families. In the reported experiments, GRPO consistently improves performance over SFT, especially on harder datasets and smaller models.

## Example Output

```text
<think>The image shows opacity in the left lower lung field, suggesting infiltration on the left side.</think>
<answer>Left</answer>
```

For yes/no questions:

```text
<think>The heart borders are clearly visible without major obscuration.</think>
<answer>yes</answer>
```

## Notes

This repository is a research prototype. Before using it as a fully reproducible benchmark, users should check dataset paths, model paths, DeepSpeed configuration, and hardware-specific settings.

Recommended future improvements:

* Add a complete inference script.
* Add a standalone evaluation script.
* Add BERTScore and METEOR evaluation implementation.
* Add clinician-in-the-loop reward modeling.
* Add visual grounding to connect textual answers with image regions.
* Extend the framework to interactive diagnostic dialogue.

## Disclaimer

This project is intended for research and educational purposes only. It is not a clinical diagnostic system and should not be used for medical decision-making.
