"""Privacy switches that must be applied before importing Unsloth/Transformers."""

import os


PRIVACY_ENV = {
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "UNSLOTH_VLLM_STANDBY": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "disabled",
    "DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
}


def disable_optional_telemetry() -> dict[str, str]:
    """Disable anonymous statistics and trackers not explicitly requested."""
    os.environ.update(PRIVACY_ENV)
    return PRIVACY_ENV.copy()
