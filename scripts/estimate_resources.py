"""Estimate local MedVLM SFT/GRPO resources without loading model tensors.

The script reads safetensors and Parquet metadata only. It never constructs a
Transformers model, allocates CUDA tensors, decodes training images, or starts
training. Estimates are deliberately decomposed so assumptions can be changed
from the command line.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess


GIB = 2**30
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
VQA_TRAIN_DIRS = {
    "SLAKE_VQA_EN": "data/mdwiratathya/SLAKE-vqa-english/data",
    "Vqa_rad": "data/flaviagiammarino/vqa-rad/data",
    "Vqa_Agupte": "data/agupte/MedVQA/data",
    "Path_VQA": "data/flaviagiammarino/path-vqa/data",
}


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def read_safetensors_headers(model_dir):
    """Return tensor metadata without mapping or reading tensor payloads."""
    tensors = {}
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with path.open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
        for name, metadata in header.items():
            if name != "__metadata__":
                tensors[name] = metadata
    if not tensors:
        raise FileNotFoundError(f"No .safetensors files found under {model_dir}")
    return tensors


def model_statistics(model_dir, lora_rank, targets):
    tensors = read_safetensors_headers(model_dir)
    parameters = 0
    checkpoint_bytes = 0
    quantizable = 0
    lora_parameters = 0
    target_hits = []

    for name, metadata in tensors.items():
        shape = metadata["shape"]
        count = product(shape)
        parameters += count
        checkpoint_bytes += count * DTYPE_BYTES[metadata["dtype"]]

        is_matrix_weight = len(shape) == 2 and name.endswith(".weight")
        excluded_from_bnb = any(
            token in name for token in ("embed_tokens", "lm_head")
        )
        if is_matrix_weight and not excluded_from_bnb:
            quantizable += count

        module_name = name.removesuffix(".weight").rsplit(".", 1)[-1]
        if is_matrix_weight and module_name in targets:
            # For W[out, in], LoRA A and B contain r*in + out*r values.
            lora_parameters += lora_rank * (int(shape[0]) + int(shape[1]))
            target_hits.append(name)

    return {
        "parameters": parameters,
        "checkpoint_tensor_bytes": checkpoint_bytes,
        "quantizable_parameters": quantizable,
        "nonquantized_parameters": parameters - quantizable,
        "lora_parameters": lora_parameters,
        "lora_target_matrices": len(target_hits),
    }


def parquet_statistics(data_dir, pattern="*.parquet"):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"rows": None, "bytes": sum(p.stat().st_size for p in Path(data_dir).glob(pattern))}

    rows = 0
    size = 0
    for path in sorted(Path(data_dir).glob(pattern)):
        rows += pq.ParquetFile(path).metadata.num_rows
        size += path.stat().st_size
    return {"rows": rows, "bytes": size}


def physical_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total_physical, status.available_physical

    page_size = os.sysconf("SC_PAGE_SIZE")
    total = page_size * os.sysconf("SC_PHYS_PAGES")
    available = page_size * os.sysconf("SC_AVPHYS_PAGES")
    return total, available


def gpu_information():
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    result = []
    for line in output.splitlines():
        name, total, free = [part.strip() for part in line.rsplit(",", 2)]
        result.append({"name": name, "total_mib": int(total), "free_mib": int(free)})
    return result


def memory_components(args, config, stats, sequence_batch):
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden // heads
    vision = config.get("vision_config", {})

    quantized = stats["quantizable_parameters"] * args.quant_bits / 8
    nonquantized = stats["nonquantized_parameters"] * args.compute_bytes
    lora_training = stats["lora_parameters"] * args.lora_state_bytes
    sequence_tokens = args.text_tokens + args.image_tokens + args.completion_tokens

    # Checkpointed transformer activation model:
    # C_act * batch * layers * tokens * hidden * compute_bytes.
    language_activations = (
        args.activation_factor
        * sequence_batch
        * layers
        * sequence_tokens
        * hidden
        * args.compute_bytes
    )
    vision_activations = (
        args.vision_activation_factor
        * sequence_batch
        * int(vision.get("depth", 0))
        * args.image_tokens
        * int(vision.get("hidden_size", 0))
        * args.compute_bytes
    )
    # K and V cache for generation. SFT passes sequence_batch=1 and sets this
    # component to zero in the caller.
    kv_cache = (
        sequence_batch
        * sequence_tokens
        * layers
        * 2
        * kv_heads
        * head_dim
        * args.compute_bytes
    )
    components = {
        "4bit_quantized_matrices": quantized,
        "nonquantized_weights": nonquantized,
        "lora_parameters_gradients_adam": lora_training,
        "language_activations": language_activations,
        "vision_activations": vision_activations,
        "kv_cache": kv_cache,
        "cuda_workspace": args.cuda_workspace_gib * GIB,
    }
    return components


def gib(value):
    return value / GIB


def parser():
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path, default=root / "models/Qwen2.5-VL-3B-Instruct")
    result.add_argument("--openmedreason-dir", type=Path, default=root / "data/neginb/OpenMedReason/data")
    result.add_argument("--cold-start-size", type=int, default=10_000)
    result.add_argument("--validation-size", type=int, default=1_000)
    result.add_argument("--openmedreason-rl-size", type=int, default=30_000)
    result.add_argument("--rl-per-dataset-cap", type=int, default=5_000)
    result.add_argument("--micro-batch", type=int, default=1)
    result.add_argument("--gradient-accumulation", type=int, default=16)
    result.add_argument("--world-size", type=int, default=1)
    result.add_argument("--num-generations", type=int, default=4)
    result.add_argument("--epochs", type=float, default=1.0)
    result.add_argument("--text-tokens", type=int, default=512, help="Typical text prompt tokens, excluding image and completion.")
    result.add_argument("--image-tokens", type=int, default=256, help="Assumed visual tokens per image; current processor maximum is reported separately.")
    result.add_argument("--completion-tokens", type=int, default=512)
    result.add_argument("--lora-rank", type=int, default=8)
    result.add_argument("--lora-targets", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    result.add_argument("--quant-bits", type=float, default=4.1, help="NF4 plus double-quantization metadata, in effective bits/quantized parameter.")
    result.add_argument("--compute-bytes", type=int, default=2, help="FP16/BF16 bytes per activation and non-quantized parameter.")
    result.add_argument("--lora-state-bytes", type=int, default=16, help="Conservative parameter + gradient + two FP32 Adam moments bytes per LoRA parameter.")
    result.add_argument("--activation-factor", type=float, default=6.0, help="Checkpointed language activation multiplier.")
    result.add_argument("--vision-activation-factor", type=float, default=6.0)
    result.add_argument("--cuda-workspace-gib", type=float, default=2.0)
    result.add_argument("--safety-margin", type=float, default=1.15)
    result.add_argument("--json", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    processor = json.loads((args.model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    stats = model_statistics(args.model_dir, args.lora_rank, set(args.lora_targets))

    omr_train = parquet_statistics(args.openmedreason_dir, "train-*.parquet")
    omr_test = parquet_statistics(args.openmedreason_dir, "test-*.parquet")
    omr = {
        "rows": (omr_train["rows"] or 0) + (omr_test["rows"] or 0),
        "bytes": omr_train["bytes"] + omr_test["bytes"],
        "train": omr_train,
        "test": omr_test,
    }
    vqa = {}
    root = Path(__file__).resolve().parents[1]
    for name, relative in VQA_TRAIN_DIRS.items():
        directory = root / relative
        files = list(directory.glob("train-*.parquet"))
        vqa[name] = parquet_statistics(directory, "train-*.parquet") if files else {"rows": 0, "bytes": 0}

    rl_vqa_rows = sum(
        min(value["rows"], args.rl_per_dataset_cap)
        for value in vqa.values()
        if value["rows"] is not None
    )
    openmedreason_rl_rows = args.openmedreason_rl_size
    if openmedreason_rl_rows <= 0 and omr_train["rows"] is not None:
        openmedreason_rl_rows = max(
            omr_train["rows"] - args.cold_start_size - args.validation_size,
            0,
        )
    rl_rows_upper = openmedreason_rl_rows + rl_vqa_rows
    effective_batch = args.micro_batch * args.gradient_accumulation * args.world_size
    sft_steps = math.ceil(args.cold_start_size / effective_batch * args.epochs)
    rl_steps = math.ceil(rl_rows_upper / effective_batch * args.epochs)

    sft_components = memory_components(args, config, stats, args.micro_batch)
    sft_components["kv_cache"] = 0
    grpo_components = memory_components(
        args,
        config,
        stats,
        args.micro_batch * args.num_generations,
    )
    sft_peak = sum(sft_components.values())
    grpo_peak = sum(grpo_components.values())

    min_pixels = int(processor.get("min_pixels", 0))
    max_pixels = int(processor.get("max_pixels", 0))
    merge_patch_area = (
        int(processor.get("patch_size", 14))
        * int(processor.get("merge_size", 2))
    ) ** 2
    processor_max_tokens = math.ceil(max_pixels / merge_patch_area) if merge_patch_area else None

    total_ram, available_ram = physical_memory()
    disk = shutil.disk_usage(root)
    model_disk = sum(path.stat().st_size for path in args.model_dir.rglob("*") if path.is_file())
    selected_fraction = None
    selected_omr_disk = None
    if omr["rows"]:
        selected_rows = args.cold_start_size + args.validation_size + openmedreason_rl_rows
        selected_fraction = min(selected_rows / omr["rows"], 1.0)
        selected_omr_disk = omr["bytes"] * selected_fraction

    report = {
        "model": stats,
        "data": {
            "openmedreason": omr,
            "openmedreason_rl_rows": openmedreason_rl_rows,
            "vqa_train": vqa,
            "rl_vqa_rows_before_strict_split": rl_vqa_rows,
            "rl_rows_upper_bound": rl_rows_upper,
        },
        "tokens": {
            "assumed_text": args.text_tokens,
            "assumed_image": args.image_tokens,
            "completion": args.completion_tokens,
            "processor_min_image_tokens": math.ceil(min_pixels / merge_patch_area) if merge_patch_area else None,
            "processor_max_image_tokens": processor_max_tokens,
        },
        "training": {
            "effective_prompt_batch": effective_batch,
            "sft_optimizer_steps": sft_steps,
            "grpo_optimizer_steps_upper_bound": rl_steps,
            "grpo_generated_sequences_per_epoch_upper_bound": rl_rows_upper * args.num_generations,
        },
        "vram": {
            "sft_components": sft_components,
            "grpo_components": grpo_components,
            "sft_peak_estimate": sft_peak,
            "grpo_peak_estimate": grpo_peak,
            "sft_recommended_with_margin": sft_peak * args.safety_margin,
            "grpo_recommended_with_margin": grpo_peak * args.safety_margin,
        },
        "host": {
            "ram_total": total_ram,
            "ram_available": available_ram,
            "disk_free": disk.free,
            "model_disk": model_disk,
            "selected_openmedreason_disk_estimate": selected_omr_disk,
            "gpus": gpu_information(),
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=== Model metadata ===")
    print(f"Parameters: {stats['parameters'] / 1e9:.3f} B")
    print(f"Checkpoint tensor bytes: {gib(stats['checkpoint_tensor_bytes']):.2f} GiB")
    print(f"Quantizable / non-quantized: {stats['quantizable_parameters']/1e9:.3f} B / {stats['nonquantized_parameters']/1e9:.3f} B")
    print(f"LoRA: {stats['lora_parameters']/1e6:.3f} M parameters across {stats['lora_target_matrices']} matrices")

    print("\n=== Data and work ===")
    print(
        f"OpenMedReason: train={omr_train['rows']}, test={omr_test['rows']}, "
        f"disk={gib(omr['bytes']):.2f} GiB"
    )
    for name, value in vqa.items():
        print(f"{name}: train_rows={value['rows']}, disk={gib(value['bytes']):.2f} GiB")
    print(f"SFT optimizer steps: ceil({args.cold_start_size} / {effective_batch}) * {args.epochs:g} = {sft_steps}")
    print(f"GRPO rows upper bound: {rl_rows_upper}; optimizer steps upper bound: {rl_steps}")
    print(f"GRPO generations/epoch upper bound: {rl_rows_upper} * {args.num_generations} = {rl_rows_upper * args.num_generations}")

    print("\n=== VRAM formula components ===")
    print(f"Quantized matrices = N_quant * {args.quant_bits}/8")
    print(f"Non-quantized weights = N_nonquant * {args.compute_bytes}")
    print(f"LoRA train state = N_lora * {args.lora_state_bytes}")
    print("Language activations = C_act * B * L * S * H * bytes")
    print("Vision activations = C_vis * B * L_vis * V * H_vis * bytes")
    print("KV cache = B * S * L * 2(K,V) * n_kv * head_dim * bytes")
    for label, components in (("SFT", sft_components), ("GRPO", grpo_components)):
        print(f"\n{label}:")
        for name, value in components.items():
            print(f"  {name}: {gib(value):.2f} GiB")
        peak = sum(components.values())
        print(f"  peak estimate: {gib(peak):.2f} GiB")
        print(f"  recommended ({args.safety_margin:.0%}): {gib(peak * args.safety_margin):.2f} GiB")

    print("\n=== Host ===")
    print(f"RAM total/free: {gib(total_ram):.1f}/{gib(available_ram):.1f} GiB")
    print(f"Workspace disk free: {gib(disk.free):.1f} GiB")
    print(f"Model files: {gib(model_disk):.2f} GiB")
    if selected_omr_disk is not None:
        print(f"Selected OpenMedReason image payload rough share: {gib(selected_omr_disk):.2f} GiB ({selected_fraction:.1%})")
    for gpu in report["host"]["gpus"]:
        print(f"GPU: {gpu['name']} total/free={gpu['total_mib']/1024:.1f}/{gpu['free_mib']/1024:.1f} GiB")

    print("\n=== Important warning ===")
    print(f"Estimate assumes {args.image_tokens} visual tokens/image, but processor metadata allows up to {processor_max_tokens}.")
    if processor_max_tokens and processor_max_tokens > args.image_tokens:
        print("Training code currently does not cap processor max_pixels; large figures can exceed this estimate and OOM.")
    print("Gradient accumulation changes effective batch/optimizer steps, not micro-batch activation VRAM.")
    print("This is an analytical estimate, not a CUDA allocation test.")


if __name__ == "__main__":
    main()
