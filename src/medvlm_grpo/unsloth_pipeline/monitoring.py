"""Console and JSONL monitoring for GRPO reward/KL/length dynamics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from .rewards import get_reward_engine


class GRPOMetricsCallback(TrainerCallback):
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _first(logs: dict[str, Any], *keys: str):
        for key in keys:
            if key in logs:
                return logs[key]
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return
        components = get_reward_engine().latest_summary.copy()
        # Callback handlers receive the same log dictionary in order. Adding
        # reward components here makes them visible to the following SwanLab
        # callback while preserving Trainer's native GRPO metrics.
        logs.update(components)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": int(state.global_step),
            **{key: value for key, value in logs.items() if isinstance(value, (int, float))},
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        reward = self._first(logs, "reward", "rewards/mean")
        reward_std = self._first(logs, "reward_std", "rewards/std")
        kl = self._first(logs, "kl", "mean_kl")
        length = self._first(logs, "completion_length", "completions/mean_length")
        semantic = components.get("component/semantic_correctness_mean")
        perplexity = components.get("component/perplexity_score_mean")
        tag = components.get("component/tag_presence_mean")
        reasoning = components.get("component/reasoning_words_mean")
        print(
            "[train metrics] "
            f"step={state.global_step} reward={reward} reward_std={reward_std} "
            f"kl={kl} completion_length={length} reasoning_words={reasoning} "
            f"semantic={semantic} perplexity_score={perplexity} tag={tag}"
        )
