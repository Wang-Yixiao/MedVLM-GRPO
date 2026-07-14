"""Print actionable compatibility checks before allocating a model."""

import argparse
import importlib.metadata as metadata
import json
import platform
import sys

import torch


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-unsloth",
        action="store_true",
        help="Fail unless the Unsloth and vLLM distributions are installed.",
    )
    args = parser.parse_args()

    package_names = [
        "transformers", "trl", "peft", "datasets", "accelerate",
        "bitsandbytes", "qwen-vl-utils", "unsloth", "unsloth-zoo", "vllm", "swanlab",
    ]
    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append({
            "name": properties.name,
            "vram_gib": round(properties.total_memory / 2**30, 2),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
        })

    packages = {name: version(name) for name in package_names}
    trl_version = packages["trl"]
    try:
        trl_pair = tuple(map(int, trl_version.split(".")[:2])) if trl_version else None
        trl_ready = bool(trl_pair and (0, 25) <= trl_pair < (0, 30))
    except ValueError:
        trl_ready = False

    stack_import_error = None
    if args.require_unsloth and packages["unsloth"] and packages["vllm"]:
        try:
            # Unsloth must patch libraries before TRL is imported.
            import unsloth  # noqa: F401
            import vllm  # noqa: F401
            from unsloth import FastVisionModel  # noqa: F401
            from trl import GRPOConfig, GRPOTrainer  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on the GPU runtime
            stack_import_error = f"{type(exc).__name__}: {exc}"

    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": version("torch"),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpus,
        "packages": packages,
        "stack_import_error": stack_import_error,
    }
    result["ready_for_dapo"] = bool(
        result["cuda_available"]
        and packages["bitsandbytes"]
        and packages["qwen-vl-utils"]
        and trl_ready
    )
    result["ready_for_unsloth_grpo"] = bool(
        result["ready_for_dapo"]
        and packages["unsloth"]
        and packages["vllm"]
        and stack_import_error is None
    )
    print(json.dumps(result, indent=2))

    ready = result["ready_for_unsloth_grpo"] if args.require_unsloth else result["ready_for_dapo"]
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
