from datasets import Dataset

import pytest

from medvlm_grpo.data import (
    _drop_missing_encoded_images,
    _normalize,
    _take_image_groups,
)


def test_openmedreason_tagged_target_is_not_nested():
    row = {
        "image": None,
        "question": "Which option is correct?\nA. One\nB. Two",
        "answer": "B",
        "reasoning": "<think>The visible finding supports option B.</think><answer>B</answer>",
    }

    formatted = _normalize(row, "question", "answer")
    completion = formatted["messages"][-1]["content"][0]["text"]

    assert completion == row["reasoning"]
    assert completion.count("<think>") == 1
    assert completion.count("<answer>") == 1
    assert formatted["solution"] == "B"


def test_image_group_selection_never_splits_repeated_images():
    dataset = Dataset.from_dict(
        {
            "_image_key": ["image-a", "image-a", "image-b", "image-c", "image-c"],
            "question": ["q1", "q2", "q3", "q4", "q5"],
        }
    )

    selected, remaining = _take_image_groups(
        dataset,
        ["image-a", "image-b", "image-c"],
        target_rows=1,
    )

    # The target is one row, but the complete two-question image group is kept.
    assert len(selected) == 2
    assert set(selected["_image_key"]) == {"image-a"}
    assert remaining == ["image-b", "image-c"]


def test_nonpositive_group_target_selects_all_remaining_rows():
    dataset = Dataset.from_dict({"_image_key": ["a", "b", "b"]})
    selected, remaining = _take_image_groups(dataset, ["a", "b"], target_rows=0)

    assert len(selected) == 3
    assert remaining == []


def test_missing_external_image_is_removed_before_decode(tmp_path):
    existing = tmp_path / "existing.jpg"
    existing.write_bytes(b"not decoded by this test")
    dataset = Dataset.from_dict(
        {
            "image": [
                {"bytes": b"embedded", "path": "embedded.jpg"},
                {"bytes": None, "path": str(existing)},
                {"bytes": None, "path": "missing.jpg"},
            ],
            "question": ["q1", "q2", "q3"],
        }
    )

    with pytest.warns(RuntimeWarning, match="Dropping 1"):
        filtered = _drop_missing_encoded_images(dataset, "test")

    assert len(filtered) == 2
    assert filtered["question"] == ["q1", "q2"]


def test_image_lookup_does_not_touch_fallback_when_primary_exists():
    class PrimaryOnly(dict):
        def get(self, key, default=None):
            if key == "images":
                raise AssertionError("unused fallback was accessed")
            return super().get(key, default)

    formatted = _normalize(
        PrimaryOnly(image=None, question="q", answer="a"),
        "question",
        "answer",
    )

    assert formatted["image"] is None
