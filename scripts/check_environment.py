"""Print actionable compatibility checks before allocating a model."""

import importlib.metadata as metadata
import json

import torch


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


result = {
    "torch": version("torch"), "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpus": [{"name": torch.cuda.get_device_name(i), "vram_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)} for i in range(torch.cuda.device_count())],
    "packages": {name: version(name) for name in ["transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes", "qwen-vl-utils"]},
}
result["ready_for_dapo"] = bool(result["cuda_available"] and result["packages"]["bitsandbytes"] and result["packages"]["qwen-vl-utils"] and result["packages"]["trl"] and tuple(map(int, result["packages"]["trl"].split(".")[:2])) >= (0, 25))
print(json.dumps(result, indent=2))
