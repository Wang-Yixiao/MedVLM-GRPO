# MedVLM-GRPO

[中文](#中文说明) | [English](#english)

## 中文说明

面向医学视觉问答（Medical VQA）的多模态强化学习框架。项目基于 Qwen2.5-VL，覆盖数据治理、SFT 冷启动、GRPO/DAPO-style 训练、奖励监控、推理和离线评估。

> 当前状态：核心研究原型与测试已完成，实验结果仍在验证。本项目仅供科研和教学使用，不可用于临床诊断。

### 核心能力

- 支持 SFT、GRPO、DAPO-style 及 `SFT → DAPO` 两阶段训练；
- 支持 LoRA/QLoRA、4-bit 量化、梯度检查点、DeepSpeed、Unsloth 与 vLLM；
- 支持 VQA-RAD、SLAKE、PathVQA、MedVQA Agupte 和 GEMeX 数据配方；
- 支持按图像分组切分、跨 split 去重和数据集泄漏审计；
- 提供单图推理、JSONL 离线评估、训练监控及消融实验脚本；
- 当前指标包括 Exact Match 与 ROUGE-L。

模型统一输出：

```text
<think>基于图像证据的简短推理</think><answer>最终答案</answer>
```

### 奖励设计

通用训练管线包含格式奖励、答案相似度、临床一致性惩罚和软长度惩罚。Unsloth 管线采用：

```text
Reward = 0.5 × SemanticCorrectness
       + 0.4 × PerplexityScore
       + 0.1 × TagPresence
```

语义正确性由 CrossEncoder 计算，流畅度由 BioGPT 困惑度衡量，结构分检查 `<think>` 与 `<answer>` 的存在和顺序。系统额外识别 yes/no、否定词和 left/right 矛盾，并将奖励分量、方差与样例写入 JSONL。

### 项目结构

```text
MedVLM-GRPO/
├── src/medvlm_grpo/          # 数据、奖励、指标与训练代码
│   └── unsloth_pipeline/     # 组合奖励、监控与隐私设置
├── scripts/                  # 训练、审计、推理、评估与消融
├── tests/                    # 单元测试与模型集成测试
├── configs/                  # 环境配置
├── reports/                  # 数据审计报告
├── Qwentrain.py              # 兼容训练入口
└── pyproject.toml
```

### 安装

建议使用 Python 3.10 和支持 CUDA 的 NVIDIA GPU。

```bash
conda create -n medvlm-grpo python=3.10 -y
conda activate medvlm-grpo
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py
```

Unsloth 训练还需安装与本机 CUDA、PyTorch 和 GPU 兼容的 `unsloth` 与 `vllm`。

### 快速开始

最小本地 VQA-RAD 实验（只使用训练集前 100 条）：

```powershell
python scripts/local_vqarad_bridge_smoke.py
```

该入口先进行视觉桥接 LoRA SFT，再生成分组候选并打印模拟 GRPO 的
reward 均值、方差、相对优势及输出长度。模拟阶段不执行策略梯度更新。
详细说明见 [`docs/local_vqarad_smoke.md`](docs/local_vqarad_smoke.md)。

```bash
# 数据审计
python scripts/audit_datasets.py --root ./data --output reports/dataset_audit.json

# 冷启动 SFT + DAPO-style
python Qwentrain.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --stage pipeline \
  --cold_start_data ./dataset/cold_start.jsonl \
  --strict_image_split \
  --output_dir ./output/qwen-med-pipeline

# Unsloth GRPO
python scripts/train_unsloth_grpo.py \
  --model_id unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit \
  --dataset SLAKE_VQA_EN \
  --strict_image_split \
  --max_steps 200 \
  --output_dir ./output/unsloth-grpo
```

冷启动 JSONL 每行需包含 `image`、`question`、`reasoning` 和 `answer`，建议使用经临床审核的 reasoning。由于 `--strict_image_split` 会改变官方 benchmark 协议，官方划分与严格划分结果应分开报告。

### 测试与结果

批量推理并生成带题型标签的 JSONL：

```bash
python scripts/batch_infer.py --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter ./output/run-seed42/trained_adapter --dataset Vqa_rad \
  --seed 42 --batch_size 2 --output reports/predictions-seed42.jsonl

python scripts/evaluate_predictions.py reports/predictions-seed42.jsonl \
  --bertscore --output reports/metrics-seed42.json
```

评估结果包含 Exact Match、ROUGE-L、METEOR、可选 BERTScore、closed/open 分层指标，以及极性和左右侧临床矛盾率。多个 seed 完成后汇总均值与样本标准差：

```bash
python scripts/aggregate_seed_metrics.py reports/metrics-seed*.json \
  --output reports/metrics-multiseed.json
```

```bash
pytest -q
```

当前已打通数据审计、训练、奖励监控、推理和评估链路。量化结果将在统一硬件、随机种子和数据划分的对照实验完成后报告。

---

## English

MedVLM-GRPO is a multimodal reinforcement-learning framework for Medical Visual Question Answering. Built on Qwen2.5-VL, it covers dataset governance, supervised cold start, GRPO/DAPO-style training, inspectable rewards, inference, and offline evaluation.

> Status: the core research prototype and tests are implemented. Experimental results are still under validation. This project is for research and education only and must not be used for clinical diagnosis.

### Key Features

- SFT, GRPO, DAPO-style, and two-stage `SFT → DAPO` workflows;
- LoRA/QLoRA, 4-bit quantization, gradient checkpointing, DeepSpeed, Unsloth, and vLLM;
- VQA-RAD, SLAKE, PathVQA, MedVQA Agupte, and a GEMeX experiment recipe;
- image-grouped splits, cross-split deduplication, and leakage auditing;
- single-image inference, JSONL evaluation, training monitoring, and ablation scripts;
- Exact Match and ROUGE-L metrics.

The model is trained to produce:

```text
<think>brief rationale grounded in visible evidence</think><answer>final answer</answer>
```

### Reward Design

The general pipeline includes format compliance, answer similarity, clinical contradiction penalties, and a soft overlong penalty. The Unsloth pipeline uses:

```text
Reward = 0.5 × SemanticCorrectness
       + 0.4 × PerplexityScore
       + 0.1 × TagPresence
```

CrossEncoder measures semantic correctness, BioGPT perplexity estimates fluency, and the tag score checks `<think>`/`<answer>` presence and ordering. Explicit guards handle yes/no, negation, and left/right contradictions. Component scores, reward variance, and samples are logged to JSONL.

### Repository Layout

```text
MedVLM-GRPO/
├── src/medvlm_grpo/          # Data, rewards, metrics, and training
│   └── unsloth_pipeline/     # Combined reward and monitoring
├── scripts/                  # Training, audit, inference, evaluation
├── tests/                    # Unit and integration tests
├── configs/                  # Environment configuration
├── reports/                  # Dataset audit reports
├── Qwentrain.py              # Compatible training launcher
└── pyproject.toml
```

### Installation

Python 3.10 and a CUDA-capable NVIDIA GPU are recommended.

```bash
conda create -n medvlm-grpo python=3.10 -y
conda activate medvlm-grpo
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py
```

Install `unsloth` and `vllm` versions compatible with the local CUDA, PyTorch, and GPU setup when using the Unsloth pipeline.

### Quick Start

```bash
# Dataset audit
python scripts/audit_datasets.py --root ./data --output reports/dataset_audit.json

# Cold-start SFT + DAPO-style
python Qwentrain.py \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --stage pipeline \
  --cold_start_data ./dataset/cold_start.jsonl \
  --strict_image_split \
  --output_dir ./output/qwen-med-pipeline

# Unsloth GRPO
python scripts/train_unsloth_grpo.py \
  --model_id unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit \
  --dataset SLAKE_VQA_EN \
  --strict_image_split \
  --max_steps 200 \
  --output_dir ./output/unsloth-grpo
```

Each cold-start JSONL row must contain `image`, `question`, `reasoning`, and `answer`; clinician-reviewed reasoning is recommended. Since `--strict_image_split` changes the benchmark protocol, official-split and strict-split results should be reported separately.

### Tests and Results

Generate a full-split JSONL with automatic question-type labels, then evaluate it:

```bash
python scripts/batch_infer.py --model_id Qwen/Qwen2.5-VL-3B-Instruct \
  --adapter ./output/run-seed42/trained_adapter --dataset Vqa_rad \
  --seed 42 --batch_size 2 --output reports/predictions-seed42.jsonl

python scripts/evaluate_predictions.py reports/predictions-seed42.jsonl \
  --bertscore --output reports/metrics-seed42.json
```

Reports include Exact Match, ROUGE-L, METEOR, optional BERTScore, closed/open stratification, and polarity/laterality contradiction rates. Aggregate multiple seeds with:

```bash
python scripts/aggregate_seed_metrics.py reports/metrics-seed*.json \
  --output reports/metrics-multiseed.json
```

```bash
pytest -q
```

The end-to-end auditing, training, reward-monitoring, inference, and evaluation pipeline is operational. Quantitative comparisons will be reported after controlled experiments with consistent hardware, random seeds, and data splits.

### Disclaimer / 免责声明

This repository is intended solely for research and educational use. It is not a medical device and must not replace professional clinical judgment. Model outputs may be incorrect or unverified.

本项目仅供科研与教学使用，不是医疗器械，不应替代医生判断。模型输出可能包含错误或未经证实的内容。
