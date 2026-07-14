import torch


def resolve_precision(precision="auto"):
    """Choose one mixed-precision mode; FP16 GradScaler cannot unscale BF16 grads."""
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if precision == "auto":
        return "bf16" if bf16_supported else "fp16"
    if precision == "bf16" and not bf16_supported:
        raise RuntimeError(
            "--precision bf16 was requested, but the current CUDA device does not support BF16. "
            "Use --precision fp16 or --precision auto."
        )
    if precision not in {"bf16", "fp16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    return precision


def precision_kwargs(precision):
    """Return mutually exclusive Trainer flags for a resolved precision."""
    if precision not in {"bf16", "fp16"}:
        raise ValueError(f"Precision must be resolved before building Trainer config: {precision}")
    return {"bf16": precision == "bf16", "fp16": precision == "fp16"}
