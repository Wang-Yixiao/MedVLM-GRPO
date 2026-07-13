from functools import partial
import hashlib

from pathlib import Path

from datasets import Dataset, DatasetDict, Image, concatenate_datasets, load_dataset


SYSTEM_PROMPT = """You are a medical visual question-answering assistant. Inspect the image and answer the question accurately and concisely. Return exactly: <think>brief clinically grounded rationale</think><answer>final answer only</answer>. Do not invent findings that are not visible in the image."""


def _image(example):
    return example.get("image", example.get("images"))


def _normalize(example, question_col, answer_col):
    image = _image(example)
    raw_answer = example[answer_col]
    answer = ", ".join(map(str, raw_answer)) if isinstance(raw_answer, list) else str(raw_answer)
    answer = answer.strip()
    reasoning = str(example.get("reasoning", example.get("reason", "Review the relevant visible medical finding."))).strip()
    completion = answer if "<answer>" in answer else f"<think>{reasoning}</think><answer>{answer}</answer>"
    # Keep the heavy image in one top-level column. The chat template only needs
    # an image placeholder; duplicating PIL bytes inside messages is extremely slow.
    user_content = []
    if image is not None:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": str(example[question_col]).strip()})
    return {
        "image": image,
        "solution": answer,
        "prompt": [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}, {"role": "user", "content": user_content}],
        "messages": [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}, {"role": "user", "content": user_content}, {"role": "assistant", "content": [{"type": "text", "text": completion}]}],
    }


DATASETS = {
    "Vqa_rad": ("./dataset/vqa-rad", "question", "answer"),
    "Vqa_Agupte": ("./dataset/medical-vqa_agupte", "questions", "answers"),
    "SLAKE_VQA_EN": ("./dataset/SLAKE_VQA_EN", "question", "answer"),
    "Path_VQA": ("./dataset/path-vqa", "question", "answer"),
}

RECIPE_PATHS = {
    "gemex": Path("data/BoKelvin/GEMeX-VQA"),
    "slake": Path("data/mdwiratathya/SLAKE-vqa-english"),
    "agupte": Path("data/agupte/MedVQA"),
}


def _format_dataset(dataset, question_col, answer_col, description):
    return dataset.map(
        partial(_normalize, question_col=question_col, answer_col=answer_col),
        remove_columns=dataset.column_names,
        desc=description,
    )


def load_gemex_vqa(path=RECIPE_PATHS["gemex"], image_root=None):
    """Load all GEMeX question types. Images are resolved from MIMIC-CXR when supplied."""
    path = Path(path)
    files = sorted(path.glob("*_question.jsonl"))
    if not files:
        raise FileNotFoundError(f"No GEMeX JSONL files found under {path}")
    datasets = []
    for file in files:
        data = load_dataset("json", data_files=str(file), split="train")
        if image_root is not None:
            image_root = Path(image_root)

            def resolve_image(example):
                image_path = image_root / example["image_path"]
                if not image_path.is_file():
                    raise FileNotFoundError(f"GEMeX image not found: {image_path}")
                return {"image": str(image_path)}

            data = data.map(resolve_image, desc=f"Resolving GEMeX images/{file.stem}")
        else:
            data = data.add_column("image", [None] * len(data))
        data = data.cast_column("image", Image())
        datasets.append(_format_dataset(data, "question", "answer", f"Formatting GEMeX-VQA/{file.stem}"))
    return concatenate_datasets(datasets)


def load_experiment_datasets(
    gemex_path=RECIPE_PATHS["gemex"],
    slake_path=RECIPE_PATHS["slake"],
    agupte_path=RECIPE_PATHS["agupte"],
    gemex_image_root=None,
    test_size=10,
):
    """Return (cold start, mixed train, train validation, image-unique unseen test)."""
    gemex = load_gemex_vqa(gemex_path, image_root=gemex_image_root)
    slake = load_dataset(str(slake_path))
    slake_train = _format_dataset(slake["train"], "question", "answer", "Formatting SLAKE/train")
    slake_validation = _format_dataset(slake["validation"], "question", "answer", "Formatting SLAKE/validation")
    train = concatenate_datasets([gemex, slake_train])

    agupte = load_dataset(str(agupte_path), split="test")
    # One question per image prevents paraphrases about the same image from
    # inflating an evaluation computed over a small test sample.
    selected_indices = []
    seen_images = set()
    for index, image_name in enumerate(agupte["image_names"]):
        if image_name in seen_images:
            continue
        seen_images.add(image_name)
        selected_indices.append(index)
        if len(selected_indices) == test_size:
            break
    unseen_test = _format_dataset(
        agupte.select(selected_indices),
        "questions",
        "answers",
        "Formatting agupte unseen test",
    )
    return gemex, train, slake_validation, unseen_test

CACHE_PATTERNS = {
    "Vqa_rad": "vqa-rad-*.arrow",
    "Vqa_Agupte": "med_vqa-*.arrow",
    "SLAKE_VQA_EN": "slake-vqa-english-*.arrow",
    "Path_VQA": "path-vqa-*.arrow",
}

LOCAL_PARQUET_REPOS = {
    "Vqa_rad": Path("data/flaviagiammarino/vqa-rad/data"),
}


def _load_available(name, path):
    if Path(path).exists():
        return load_dataset(path=path)
    parquet_root = LOCAL_PARQUET_REPOS.get(name)
    if parquet_root and parquet_root.exists():
        split_files = {
            split: str(files[0])
            for split in ("train", "validation", "test")
            if (files := sorted(parquet_root.glob(f"{split}-*.parquet")))
        }
        if split_files:
            return load_dataset("parquet", data_files=split_files)
    files = sorted(Path("data").rglob(CACHE_PATTERNS[name]))
    if not files:
        raise FileNotFoundError(f"Neither {path} nor a cached {CACHE_PATTERNS[name]} dataset was found")
    splits = {}
    for file in files:
        split = next((x for x in ("train", "validation", "test") if f"-{x}" in file.name), None)
        if split:
            splits[split] = Dataset.from_file(str(file))
    return DatasetDict(splits)


def _image_digest(image):
    return hashlib.sha256(image.mode.encode() + str(image.size).encode() + image.tobytes()).hexdigest()


def _remove_image_leakage(train, held_out):
    held_out_hashes = {_image_digest(image) for image in held_out["image"]}
    return train.filter(lambda image: _image_digest(image) not in held_out_hashes, input_columns=["image"], desc="Removing held-out image leakage")


def _group_train_validation(dataset, validation_ratio=0.2):
    """Split by image, keeping all questions for one image in the same split."""
    hashes = [_image_digest(image) for image in dataset["image"]]
    unique = sorted(set(hashes), key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    validation_hashes = set(unique[:max(1, int(len(unique) * validation_ratio))])
    validation = dataset.filter(lambda image: _image_digest(image) in validation_hashes, input_columns=["image"])
    train = dataset.filter(lambda image: _image_digest(image) not in validation_hashes, input_columns=["image"])
    return train, validation


def load_medical_vqa(name, strict_image_split=False):
    path, question_col, answer_col = DATASETS[name]
    raw = _load_available(name, path)
    data = DatasetDict({
        split: ds.map(
            partial(_normalize, question_col=question_col, answer_col=answer_col),
            remove_columns=ds.column_names,
            desc=f"Formatting {name}/{split}",
        )
        for split, ds in raw.items()
    })
    test = data.get("test")
    train = data["train"]
    if strict_image_split and test is not None:
        train = _remove_image_leakage(train, test)
    if "validation" not in data:
        train, validation = _group_train_validation(train)
    else:
        validation = data["validation"]
    if strict_image_split:
        train = _remove_image_leakage(train, validation)
    return train, test if test is not None else validation, validation


def load_cold_start_data(path):
    """Load clinician-reviewed reasoning data; no synthetic rationale is generated."""
    data = load_dataset("json", data_files=path, split="train")
    required = {"image", "question", "answer", "reasoning"}
    missing = required.difference(data.column_names)
    if missing:
        raise ValueError(f"Cold-start data is missing columns: {sorted(missing)}")
    data = data.cast_column("image", __import__("datasets").Image())
    data = data.map(partial(_normalize, question_col="question", answer_col="answer"))
    split = data.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


# Compatibility wrappers.
def load_Medical_Vqa_rad(grpo=False): return load_medical_vqa("Vqa_rad")
def load_Medical_Vqa_Agupte(grpo=False): return load_medical_vqa("Vqa_Agupte")
def load_SLAKE_VQA_EN(grpo=False): return load_medical_vqa("SLAKE_VQA_EN")
def load_Path_Vqa(grpo=False): return load_medical_vqa("Path_VQA")
