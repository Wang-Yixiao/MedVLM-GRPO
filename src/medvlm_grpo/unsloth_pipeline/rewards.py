"""Combined medical-VQA reward with inspectable component scores.

Reward = 0.5 * SemanticCorrectness
       + 0.4 * PerplexityScore
       + 0.1 * TagPresence

Models are lazy-loaded so importing this module never allocates GPU memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"</?(?:think|answer)>", re.IGNORECASE)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    if isinstance(completion, Sequence) and completion:
        return completion_text(completion[0])
    return ""


def extract_answer(text: str) -> str:
    match = ANSWER_RE.search(text)
    return match.group(1).strip() if match else ""


def extract_reasoning(text: str) -> str:
    match = THINK_RE.search(text)
    return match.group(1).strip() if match else ""


def natural_text(text: str) -> str:
    """Remove control tags before scoring language-model perplexity."""
    return " ".join(TAG_RE.sub(" ", text).split())


def tag_presence_score(text: str) -> float:
    """Score tag presence and enforce think-before-answer ordering."""
    think = THINK_RE.search(text)
    answer = ANSWER_RE.search(text)
    if think and answer and think.start() < think.end() <= answer.start():
        return 1.0
    if bool(think) ^ bool(answer):
        return 0.5
    return 0.0


def _medical_contradiction(response: str, reference: str) -> bool:
    """Catch short-answer contradictions that generic STS often over-scores."""
    tokens = lambda value: set(re.findall(r"[a-z0-9.]+", value.casefold()))
    predicted, expected = tokens(response), tokens(reference)
    polar = ({"yes", "present", "positive"}, {"no", "absent", "negative"})
    lateral = ({"left"}, {"right"})
    for first, second in (polar, lateral):
        if (predicted & first and expected & second) or (predicted & second and expected & first):
            return True
    return False


@dataclass(slots=True)
class RewardConfig:
    semantic_model: str = "cross-encoder/stsb-TinyBERT-L4"
    fluency_model: str = "microsoft/biogpt"
    device: str = "cpu"
    semantic_batch_size: int = 16
    perplexity_batch_size: int = 4
    max_reward_tokens: int = 256
    perplexity_temperature: float = 5.0
    semantic_weight: float = 0.5
    perplexity_weight: float = 0.4
    tag_weight: float = 0.1
    diagnostics_path: str = "output/unsloth-grpo/reward_components.jsonl"
    print_every: int = 1

    def __post_init__(self) -> None:
        total = self.semantic_weight + self.perplexity_weight + self.tag_weight
        if not math.isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError(f"Reward weights must sum to 1.0, got {total}")
        if self.perplexity_temperature <= 0:
            raise ValueError("perplexity_temperature must be positive")


@dataclass(slots=True)
class RewardResult:
    response: str
    reference: str
    semantic_correctness: float
    perplexity: float
    perplexity_score: float
    tag_presence: float
    combined_reward: float
    completion_words: int
    reasoning_words: int


class RewardEngine:
    """Lazy, batched component models plus JSONL/console diagnostics."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self._semantic_model = None
        self._fluency_model = None
        self._fluency_tokenizer = None
        self.calls = 0
        self.latest_summary: dict[str, float] = {}

    def _load_semantic_model(self):
        if self._semantic_model is None:
            import torch
            from sentence_transformers import CrossEncoder

            self._semantic_model = CrossEncoder(
                self.config.semantic_model,
                device=self.config.device,
                activation_fn=torch.nn.Sigmoid(),
            )
        return self._semantic_model

    def _load_fluency_model(self):
        if self._fluency_model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._fluency_tokenizer = AutoTokenizer.from_pretrained(self.config.fluency_model)
            if self._fluency_tokenizer.pad_token_id is None:
                self._fluency_tokenizer.pad_token = self._fluency_tokenizer.eos_token
            self._fluency_model = AutoModelForCausalLM.from_pretrained(self.config.fluency_model)
            self._fluency_model.to(self.config.device).eval()
        return self._fluency_model, self._fluency_tokenizer

    @staticmethod
    def _align(values: Sequence[Any], size: int) -> list[Any]:
        values = list(values)
        if not values:
            return [""] * size
        if len(values) == size:
            return values
        return [values[index % len(values)] for index in range(size)]

    def semantic_scores(self, responses: Sequence[str], references: Sequence[str]) -> list[float]:
        model = self._load_semantic_model()
        pairs = list(zip(responses, references))
        scores = model.predict(
            pairs,
            batch_size=self.config.semantic_batch_size,
            show_progress_bar=False,
        )
        return [
            0.0
            if not response or _medical_contradiction(response, reference)
            else min(max(float(score), 0.0), 1.0)
            for response, reference, score in zip(responses, references, scores)
        ]

    def perplexity_scores(self, texts: Sequence[str]) -> tuple[list[float], list[float]]:
        """Return per-sequence perplexity and a stable [0, 1] quality score."""
        import torch
        import torch.nn.functional as functional

        model, tokenizer = self._load_fluency_model()
        perplexities: list[float] = []
        quality_scores: list[float] = []
        for start in range(0, len(texts), self.config.perplexity_batch_size):
            batch = list(texts[start : start + self.config.perplexity_batch_size])
            encoded = tokenizer(
                [text or tokenizer.eos_token for text in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_reward_tokens,
            ).to(self.config.device)
            with torch.inference_mode():
                logits = model(**encoded).logits[:, :-1, :].float()
            labels = encoded.input_ids[:, 1:]
            mask = encoded.attention_mask[:, 1:].bool()
            token_nll = functional.cross_entropy(
                logits.transpose(1, 2), labels, reduction="none"
            )
            sequence_nll = (token_nll * mask).sum(1) / mask.sum(1).clamp_min(1)
            for original, nll in zip(batch, sequence_nll.detach().cpu().tolist()):
                if not original:
                    perplexities.append(math.exp(20.0))
                    quality_scores.append(0.0)
                    continue
                safe_nll = min(max(float(nll), 0.0), 20.0)
                perplexities.append(math.exp(safe_nll))
                # Fixed calibration preserves comparability between GRPO groups.
                quality_scores.append(math.exp(-safe_nll / self.config.perplexity_temperature))
        return perplexities, quality_scores

    def score(self, completions: Sequence[Any], references: Sequence[str]) -> list[RewardResult]:
        outputs = [completion_text(item) for item in completions]
        references = [str(item) for item in self._align(references, len(outputs))]
        answers = [extract_answer(item) for item in outputs]
        fluent_texts = [natural_text(item) for item in outputs]
        semantic = self.semantic_scores(answers, references)
        perplexities, perplexity_quality = self.perplexity_scores(fluent_texts)
        tags = [tag_presence_score(item) for item in outputs]

        results = []
        for output, answer, reference, sem, ppl, ppl_score, tag in zip(
            outputs, answers, references, semantic, perplexities, perplexity_quality, tags
        ):
            combined = (
                self.config.semantic_weight * sem
                + self.config.perplexity_weight * ppl_score
                + self.config.tag_weight * tag
            )
            results.append(
                RewardResult(
                    response=answer,
                    reference=reference,
                    semantic_correctness=sem,
                    perplexity=ppl,
                    perplexity_score=ppl_score,
                    tag_presence=tag,
                    combined_reward=min(max(combined, 0.0), 1.0),
                    completion_words=len(output.split()),
                    reasoning_words=len(extract_reasoning(output).split()),
                )
            )
        self._record(results, outputs)
        return results

    def _record(self, results: Sequence[RewardResult], outputs: Sequence[str]) -> None:
        self.calls += 1
        fields = (
            "semantic_correctness",
            "perplexity_score",
            "tag_presence",
            "combined_reward",
            "completion_words",
            "reasoning_words",
        )
        self.latest_summary = {
            f"component/{field}_mean": sum(getattr(item, field) for item in results) / max(len(results), 1)
            for field in fields
        }
        rewards = [item.combined_reward for item in results]
        mean = sum(rewards) / max(len(rewards), 1)
        self.latest_summary["component/reward_variance"] = sum((item - mean) ** 2 for item in rewards) / max(len(rewards), 1)
        path = Path(self.config.diagnostics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reward_call": self.calls,
            "config": asdict(self.config) if self.calls == 1 else None,
            "summary": self.latest_summary,
            "samples": [asdict(item) | {"full_output": output} for item, output in zip(results, outputs)],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        if self.calls % self.config.print_every == 0 and results:
            sample = results[0]
            print(
                "[reward] "
                f"call={self.calls} semantic={self.latest_summary['component/semantic_correctness_mean']:.4f} "
                f"ppl_score={self.latest_summary['component/perplexity_score_mean']:.4f} "
                f"tag={self.latest_summary['component/tag_presence_mean']:.4f} "
                f"combined={self.latest_summary['component/combined_reward_mean']:.4f} "
                f"variance={self.latest_summary['component/reward_variance']:.6f}"
            )
            print(
                "[reward sample] "
                f"prediction={sample.response!r} reference={sample.reference!r} "
                f"semantic={sample.semantic_correctness:.4f} perplexity={sample.perplexity:.2f} "
                f"perplexity_score={sample.perplexity_score:.4f} tag={sample.tag_presence:.2f}"
            )


_REWARD_ENGINE: RewardEngine | None = None


def configure_reward_engine(config: RewardConfig | None = None) -> RewardEngine:
    global _REWARD_ENGINE
    _REWARD_ENGINE = RewardEngine(config)
    return _REWARD_ENGINE


def get_reward_engine() -> RewardEngine:
    global _REWARD_ENGINE
    if _REWARD_ENGINE is None:
        _REWARD_ENGINE = RewardEngine()
    return _REWARD_ENGINE


def combine_reward_func(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    solution: Sequence[str] | None = None,
    answer: Sequence[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """The single reward function passed to GRPOTrainer."""
    del prompts, kwargs
    references = solution if solution is not None else answer
    results = get_reward_engine().score(completions, references or [])
    return [item.combined_reward for item in results]


# Compatibility with the supplied reference file's spelling.
combined_reward_func = combine_reward_func
