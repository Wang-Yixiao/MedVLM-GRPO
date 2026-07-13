from pathlib import Path

from datasets import load_dataset

from medvlm_grpo.data import load_experiment_datasets


def _normalize(value):
    return " ".join(str(value).lower().strip().split())


def test_local_experiment_recipe_loads_ten_unique_agupte_images():
    if not Path("data/agupte/MedVQA").exists():
        return
    cold, train, validation, unseen = load_experiment_datasets()
    raw_test = load_dataset("data/agupte/MedVQA", split="test")

    # Reproduce the loader's deterministic first-question-per-image selection.
    selected = []
    seen = set()
    for row in raw_test:
        if row["image_names"] not in seen:
            seen.add(row["image_names"])
            selected.append(row)
        if len(selected) == 10:
            break

    assert len(unseen) == 10
    assert len({row["image_names"] for row in selected}) == 10
    assert all(row["images"] is not None for row in selected)
    assert all(row["questions"].strip() and row["answers"].strip() for row in selected)
    assert len({row["ids"] for row in selected}) == 10

    # agupte is evaluation-only: none of its rows are appended to mixed train.
    assert len(cold) > 0
    assert len(train) == len(cold) + 4919
    assert len(validation) == 1053
    assert unseen.column_names == ["image", "solution", "prompt", "messages"]
    assert unseen[0]["image"] is not None
    assert [row["answers"] for row in selected] == unseen["solution"]

    mixed_train_qa = {
        (
            _normalize(prompt[-1]["content"][-1]["text"]),
            _normalize(solution),
        )
        for prompt, solution in zip(train["prompt"], train["solution"])
    }
    assert not any(
        (_normalize(row["questions"]), _normalize(row["answers"])) in mixed_train_qa
        for row in selected
    )

    print("\n=== agupte unseen test: 10 unique images ===")
    for index, row in enumerate(selected):
        image = row["images"]
        print(
            f"[{index:02d}] id={row['ids']} | image={row['image_names']} "
            f"| size={image.size} | question={row['questions']} | answer={row['answers']}"
        )
    print("=== reliability checks ===")
    print(f"rows={len(selected)} | unique_images={len(seen)}")
    print("empty_question_or_answer=0 | duplicate_ids=0 | image_decode=OK")
    print("exact_qa_overlap_with_mixed_train=0")

    # GEMeX must use its real `reason` field rather than the generic fallback.
    assert cold[0]["messages"][-1]["content"][0]["text"].startswith("<think>")
    assert "Review the relevant visible medical finding." not in cold[0]["messages"][-1]["content"][0]["text"]
