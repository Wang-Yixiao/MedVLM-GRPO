"""GPU integration test using local reward models and real MedVQA rows."""

from pathlib import Path

import pytest
import torch
from datasets import load_dataset

from medvlm_grpo.unsloth_pipeline.rewards import RewardConfig, RewardEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIOGPT_PATH = PROJECT_ROOT / "reward_model" / "microsoft" / "biogpt"
CROSS_ENCODER_PATH = PROJECT_ROOT / "reward_model" / "cross-encoder" / "stsb-roberta-base"


@pytest.mark.integration
def test_local_reward_models_load_and_compute_reward(tmp_path):
    assert torch.cuda.is_available(), "CUDA is required for this GPU integration test"
    required_files = (
        BIOGPT_PATH / "config.json",
        BIOGPT_PATH / "pytorch_model.bin",
        BIOGPT_PATH / "vocab.json",
        CROSS_ENCODER_PATH / "config.json",
        CROSS_ENCODER_PATH / "model.safetensors",
        CROSS_ENCODER_PATH / "tokenizer.json",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    assert not missing, f"Missing local reward-model files: {missing}"

    config = RewardConfig(
        semantic_model=str(CROSS_ENCODER_PATH),
        fluency_model=str(BIOGPT_PATH),
        device="cuda:0",
        semantic_batch_size=2,
        perplexity_batch_size=2,
        max_reward_tokens=64,
        diagnostics_path=str(tmp_path / "reward_components.jsonl"),
        print_every=1,
    )
    engine = RewardEngine(config)

    print("\n=== loading local reward models ===")
    semantic_model = engine._load_semantic_model()
    fluency_model, tokenizer = engine._load_fluency_model()
    print(f"semantic_model={CROSS_ENCODER_PATH} | class={type(semantic_model).__name__}")
    print(f"fluency_model={BIOGPT_PATH} | class={type(fluency_model).__name__}")
    print(f"fluency_tokenizer_class={type(tokenizer).__name__}")
    semantic_device = next(semantic_model.model.parameters()).device
    fluency_device = next(fluency_model.parameters()).device
    print(f"semantic_device={semantic_device} | fluency_device={fluency_device}")
    assert semantic_device.type == "cuda"
    assert fluency_device.type == "cuda"

    medvqa = load_dataset(
        str(PROJECT_ROOT / "data" / "agupte" / "MedVQA"), split="test"
    ).select(range(2))
    references = medvqa["answers"]
    completions = [
        f"<think>The medical image was reviewed for the requested finding.</think><answer>{answer}</answer>"
        for answer in references
    ]

    # RewardEngine consumes text, not pixels. Check both the real MedVQA image
    # tensor and the text token batch explicitly before scoring.
    import numpy as np

    image_tensor = torch.from_numpy(np.asarray(medvqa[0]["images"]).copy()).to("cuda:0")
    token_batch = tokenizer(completions[0], return_tensors="pt").to("cuda:0")
    print("=== real MedVQA GPU data ===")
    print(
        f"id={medvqa[0]['ids']} | image={medvqa[0]['image_names']} "
        f"| question={medvqa[0]['questions']} | answer={medvqa[0]['answers']}"
    )
    print(
        f"image_device={image_tensor.device} | image_shape={tuple(image_tensor.shape)} "
        f"| token_device={token_batch.input_ids.device} | token_shape={tuple(token_batch.input_ids.shape)}"
    )
    assert image_tensor.device.type == "cuda"
    assert token_batch.input_ids.device.type == "cuda"

    memory_before = torch.cuda.memory_allocated()
    results = engine.score(completions, references)
    torch.cuda.synchronize()
    memory_after = torch.cuda.memory_allocated()

    print("=== combined reward results ===")
    for index, result in enumerate(results):
        print(
            f"[{index}] prediction={result.response!r} | reference={result.reference!r} "
            f"| semantic={result.semantic_correctness:.4f} "
            f"| perplexity={result.perplexity:.4f} "
            f"| perplexity_score={result.perplexity_score:.4f} "
            f"| tag={result.tag_presence:.1f} "
            f"| combined={result.combined_reward:.4f}"
        )

    assert len(results) == 2
    assert all(0.0 <= result.semantic_correctness <= 1.0 for result in results)
    assert all(result.perplexity > 0.0 for result in results)
    assert all(0.0 <= result.perplexity_score <= 1.0 for result in results)
    assert all(result.tag_presence == 1.0 for result in results)
    assert all(0.0 <= result.combined_reward <= 1.0 for result in results)
    assert all(result.semantic_correctness > 0.0 for result in results)
    print(f"cuda_memory_before_reward={memory_before / 2**20:.1f} MiB")
    print(f"cuda_memory_after_reward={memory_after / 2**20:.1f} MiB")
    assert Path(config.diagnostics_path).is_file()
