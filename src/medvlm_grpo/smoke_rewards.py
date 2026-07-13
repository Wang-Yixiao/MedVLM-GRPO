"""Dependency-free proxy rewards for the local GRPO rollout smoke test.

These scores verify the grouped generation/reward/advantage pipeline without
loading separate neural reward models. They are not a replacement for the
CrossEncoder + BioGPT reward used by a real experiment.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import math
import re


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<think>\s*.+?\s*</think>\s*<answer>\s*.+?\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def extract_answer(text: str) -> str:
    match = ANSWER_RE.search(text)
    return match.group(1).strip() if match else ""


def proxy_semantic_correctness(output: str, reference: str) -> float:
    prediction = " ".join(extract_answer(output).casefold().split())
    target = " ".join(str(reference).casefold().split())
    if not prediction:
        return 0.0
    if {prediction, target} == {"yes", "no"}:
        return 0.0
    return SequenceMatcher(None, prediction, target).ratio()


def proxy_fluency_score(output: str) -> float:
    """Cheap repetition/length proxy used only by the local smoke test."""
    plain = re.sub(r"</?(?:think|answer)>", " ", output, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9]+", plain.casefold())
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    length_factor = 1.0 - math.exp(-len(words) / 8.0)
    overlong_penalty = math.exp(-max(len(words) - 80, 0) / 40.0)
    return min(max(unique_ratio * length_factor * overlong_penalty, 0.0), 1.0)


def tag_presence(output: str) -> float:
    return 1.0 if FORMAT_RE.fullmatch(output) else 0.0


def proxy_combined_reward(output: str, reference: str) -> dict[str, float]:
    semantic = proxy_semantic_correctness(output, reference)
    fluency = proxy_fluency_score(output)
    tags = tag_presence(output)
    combined = 0.5 * semantic + 0.4 * fluency + 0.1 * tags
    return {
        "semantic_proxy": semantic,
        "fluency_proxy": fluency,
        "tag_presence": tags,
        "combined_reward": combined,
    }
