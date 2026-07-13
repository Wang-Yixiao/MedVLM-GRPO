"""Generation and safety metrics for medical visual question answering."""

from __future__ import annotations

from collections import Counter
import re
from typing import Sequence

import evaluate

from .rewards import extract_answer

CLOSED_PREFIXES = (
    "is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ",
    "could ", "has ", "have ", "had ", "which side", "what side",
    "where is", "where are", "how many", "what number", "what type of view",
    "what modality", "which modality", "what organ", "which organ",
)


def normalize_answer(value: str) -> str:
    return " ".join(extract_answer(str(value)).casefold().split())


def classify_question(question: str, reference: str = "") -> str:
    """Classify a VQA item as closed- or open-ended using reproducible rules."""
    question = " ".join(str(question).casefold().split())
    reference = normalize_answer(reference)
    return "closed" if reference in {"yes", "no"} or question.startswith(CLOSED_PREFIXES) else "open"


def clinical_contradictions(prediction: str, reference: str) -> list[str]:
    """Return high-risk polarity/laterality contradiction labels."""
    tokens = lambda text: set(re.findall(r"[a-z0-9.]+", normalize_answer(text)))
    predicted, expected = tokens(prediction), tokens(reference)
    groups = (
        ("polarity", {"yes", "present", "positive"}, {"no", "absent", "negative"}),
        ("laterality", {"left"}, {"right"}),
    )
    return [
        label for label, first, second in groups
        if (predicted & first and expected & second) or (predicted & second and expected & first)
    ]


def _meteor(predictions: Sequence[str], references: Sequence[str]) -> float:
    from nltk.translate.meteor_score import meteor_score
    scores = [meteor_score([r.split()], p.split()) for p, r in zip(predictions, references)]
    return sum(scores) / max(len(scores), 1)


def _bertscore(predictions: Sequence[str], references: Sequence[str], model_type: str | None, num_layers: int | None) -> dict[str, float]:
    kwargs = {"predictions": list(predictions), "references": list(references), "lang": "en"}
    if model_type:
        kwargs["model_type"] = model_type
    if num_layers is not None:
        kwargs["num_layers"] = num_layers
    scores = evaluate.load("bertscore").compute(**kwargs)
    return {
        f"bertscore_{key}": sum(map(float, scores[key])) / max(len(scores[key]), 1)
        for key in ("precision", "recall", "f1")
    }


def _basic_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    if not references:
        return {"count": 0, "exact_match": 0.0, "rougeL": 0.0}
    exact = sum(p == r for p, r in zip(predictions, references)) / len(references)
    rouge = evaluate.load("rouge").compute(predictions=list(predictions), references=list(references))
    return {"count": len(references), "exact_match": exact, "rougeL": float(rouge["rougeL"])}


def compute_generation_metrics(
    predictions: Sequence[str], references: Sequence[str], questions: Sequence[str] | None = None,
    *, include_meteor: bool = True, include_bertscore: bool = False,
    bertscore_model: str | None = None, bertscore_num_layers: int | None = None,
) -> dict:
    """Compute overall, question-stratified, semantic, and safety metrics."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if questions is not None and len(questions) != len(references):
        raise ValueError("questions and references must have the same length")
    predictions = [normalize_answer(text) for text in predictions]
    references = [normalize_answer(text) for text in references]
    questions = list(questions) if questions is not None else [""] * len(references)

    overall = _basic_metrics(predictions, references)
    if include_meteor:
        overall["meteor"] = _meteor(predictions, references)
    if include_bertscore:
        overall.update(_bertscore(predictions, references, bertscore_model, bertscore_num_layers))

    labels = [clinical_contradictions(p, r) for p, r in zip(predictions, references)]
    counts = Counter(label for item in labels for label in item)
    overall["clinical_contradiction_rate"] = sum(map(bool, labels)) / max(len(labels), 1)
    overall["polarity_contradictions"] = counts["polarity"]
    overall["laterality_contradictions"] = counts["laterality"]

    types = [classify_question(q, r) for q, r in zip(questions, references)]
    by_type = {}
    for group in ("closed", "open"):
        indices = [i for i, value in enumerate(types) if value == group]
        metrics = _basic_metrics([predictions[i] for i in indices], [references[i] for i in indices])
        metrics["clinical_contradiction_rate"] = sum(bool(labels[i]) for i in indices) / max(len(indices), 1)
        by_type[group] = metrics
    return {"overall": overall, "by_question_type": by_type}
