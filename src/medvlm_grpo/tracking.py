"""Optional experiment tracking integrations."""

from __future__ import annotations

from typing import Any


def make_swanlab_callback(
    *,
    enabled: bool,
    project: str,
    experiment_name: str | None = None,
    workspace: str | None = None,
    config: dict[str, Any] | None = None,
):
    """Build SwanLab's Transformers callback only when tracking is enabled."""
    if not enabled:
        return None
    try:
        from swanlab.integration.transformers import SwanLabCallback
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "SwanLab tracking was requested but swanlab is not installed. "
            "Run `pip install swanlab` or disable it with `--no-swanlab`."
        ) from exc

    kwargs: dict[str, Any] = {"project": project}
    if experiment_name:
        kwargs["experiment_name"] = experiment_name
    if workspace:
        kwargs["workspace"] = workspace
    if config:
        kwargs["config"] = config
    return SwanLabCallback(**kwargs)
