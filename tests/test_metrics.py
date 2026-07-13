import pytest

from medvlm_grpo.metrics import (
    classify_question,
    clinical_contradictions,
    compute_generation_metrics,
)


def test_question_type_and_clinical_contradictions():
    assert classify_question("Is there an effusion?", "yes") == "closed"
    assert classify_question("Describe the abnormality", "pleural effusion") == "open"
    assert clinical_contradictions("<answer>no</answer>", "yes") == ["polarity"]
    assert clinical_contradictions("right lung", "left lung") == ["laterality"]
    assert clinical_contradictions("left lung", "left lung") == []


def test_stratified_metrics_and_contradiction_rate():
    metrics = compute_generation_metrics(
        ["<answer>no</answer>", "left lower lung"],
        ["yes", "left lung"],
        ["Is a lesion present?", "Describe its location"],
        include_meteor=False,
        include_bertscore=False,
    )
    assert metrics["overall"]["count"] == 2
    assert metrics["overall"]["exact_match"] == 0.0
    assert metrics["overall"]["clinical_contradiction_rate"] == 0.5
    assert metrics["overall"]["polarity_contradictions"] == 1
    assert metrics["by_question_type"]["closed"]["count"] == 1
    assert metrics["by_question_type"]["open"]["count"] == 1


def test_metric_lengths_must_match():
    with pytest.raises(ValueError):
        compute_generation_metrics(["yes"], [], include_meteor=False)
