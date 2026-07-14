# AutoDL 环境安装

## 推荐实例

- 系统：Ubuntu 22.04；
- Python：3.10 或 3.11（推荐 3.11）；
- GPU：最低 24 GB 显存，建议 A100 40/80 GB、A800、L40S；
- 磁盘：数据盘至少保留 50 GB，模型、数据集和编译缓存都会放在 `/root/autodl-tmp`。

单张 24 GB GPU 可以从 Qwen2.5-VL-3B、4-bit、batch size 1 和较短上下文开始。GRPO 同时保留多条生成结果，显存需求明显高于普通 SFT；发生 OOM 时优先降低 `--max_seq_length`、`--max_prompt_length`、`--max_completion_length`、`--num_generations` 和 `--gpu_memory_utilization`。

## 一键安装

在 AutoDL 终端进入项目根目录后运行：

```bash
bash autodl/setup.sh
source autodl/activate.sh
```

安装器会创建隔离环境，按当前 NVIDIA 驱动自动选择 PyTorch CUDA wheel，并安装 Unsloth、vLLM、TRL 和项目本身。默认位置：

- 环境：`/root/autodl-tmp/envs/medvlm-grpo`
- Hugging Face 缓存：`/root/autodl-tmp/cache/huggingface`
- uv 缓存：`/root/autodl-tmp/cache/uv`

若想改位置：

```bash
ENV_DIR=/root/autodl-tmp/my-env \
CACHE_DIR=/root/autodl-tmp/my-cache \
bash autodl/setup.sh

ENV_DIR=/root/autodl-tmp/my-env \
CACHE_DIR=/root/autodl-tmp/my-cache \
source autodl/activate.sh
```

## 验证与首次训练

```bash
python scripts/check_environment.py --require-unsloth
pytest -q

# 先跑 2 step，确认数据下载、模型加载、vLLM 和保存流程正常
python scripts/train_unsloth_grpo.py \
  --max_steps 2 \
  --max_seq_length 4096 \
  --max_prompt_length 1024 \
  --max_completion_length 128 \
  --num_generations 2 \
  --gpu_memory_utilization 0.45 \
  --swanlab \
  --swanlab_project medvlm-grpo \
  --swanlab_experiment_name autodl-smoke \
  --output_dir output/autodl-smoke
```

首次使用云端可视化时先执行 `swanlab login`；无人值守任务也可以通过
`SWANLAB_API_KEY` 环境变量认证。省略 `--swanlab` 时训练只写本地 JSONL。

通过后再运行正式配置。每次新开终端都需要重新执行 `source autodl/activate.sh`。

## 常见问题

- `CUDA out of memory`：先使用上面的 smoke 参数，再逐项调大；也可以关闭快速推理 `--no-fast_inference`。
- `No space left on device`：确认环境变量中的 `HF_HOME` 指向 `/root/autodl-tmp`，并检查数据盘空间。
- `bitsandbytes` 报 CUDA 错误：运行 `python -m bitsandbytes` 和 `nvidia-smi`，不要另外安装系统 CUDA 覆盖镜像驱动。
- vLLM/torch 二进制不匹配：不要在安装后单独升级 torch。重新创建一个空的 `ENV_DIR`，再运行安装脚本，让解析器一次性选择整个训练栈。
- 无法下载 Hugging Face 模型：先执行 `hf auth login`；受限模型还需要在模型页面接受许可。
