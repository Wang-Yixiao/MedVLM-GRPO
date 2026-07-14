from collections import Counter
from functools import partial
import hashlib
import re
import warnings

from pathlib import Path

from datasets import Dataset, DatasetDict, Image, concatenate_datasets, load_dataset


SYSTEM_PROMPT = """You are a medical visual question-answering assistant. Inspect the image and answer the question accurately and concisely. Return exactly: <think>brief clinically grounded rationale</think><answer>final answer only</answer>. Do not invent findings that are not visible in the image."""

TAGGED_REASONING_RE = re.compile(
    r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _image(example):
    # Do not use example.get("image", example.get("images")) here: Python
    # evaluates the default argument eagerly, which can make datasets decode an
    # unused fallback column (or fail while looking it up).
    if "image" in example:
        return example["image"]
    return example.get("images")


def _normalize(example, question_col, answer_col):
    image = _image(example)
    raw_answer = example[answer_col]
    answer = ", ".join(map(str, raw_answer)) if isinstance(raw_answer, list) else str(raw_answer)
    answer = answer.strip()
    reasoning = str(example.get("reasoning", example.get("reason", "Review the relevant visible medical finding."))).strip()
    # OpenMedReason already stores a complete tagged target. Re-wrapping it
    # would create nested <think>/<answer> tags and teach an invalid format.
    completion = reasoning if TAGGED_REASONING_RE.match(reasoning) else f"<think>{reasoning}</think><answer>{answer}</answer>"
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
    "openmedreason": Path("/root/autodl-tmp/datasets/neginb/OpenMedReason"),
    "gemex": Path("data/BoKelvin/GEMeX-VQA"),
    "slake": Path("/root/autodl-tmp/datasets/mdwiratathya/SLAKE-vqa-english"),
    "agupte": Path("/root/autodl-tmp/datasets/agupte/MedVQA"),
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


def _encoded_image_digest(image):
    """Hash an Image(decode=False) value without decoding it into pixels."""
    if isinstance(image, dict):
        payload = image.get("bytes")
        if payload is not None:
            return hashlib.sha256(payload).hexdigest()
        path = image.get("path")
        if path:
            return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    raise ValueError("OpenMedReason image has neither encoded bytes nor a path")


def _encoded_image_keys(dataset):
    """Read encoded images sequentially and retain only their small digests."""
    return [_encoded_image_digest(image) for image in dataset["image"]]


def _encoded_image_is_decodable(image):
    """Return whether an Image(decode=False) value can be decoded locally."""
    if not isinstance(image, dict):
        return False
    if image.get("bytes") is not None:
        return True
    image_path = image.get("path")
    return bool(image_path and Path(image_path).is_file())


def _drop_missing_encoded_images(dataset, split_name):
    """Drop rows whose image payload points at a nonexistent external file."""
    valid_indices = [
        index
        for index, image in enumerate(dataset["image"])
        if _encoded_image_is_decodable(image)
    ]
    dropped = len(dataset) - len(valid_indices)
    if dropped:
        warnings.warn(
            f"Dropping {dropped} {split_name} row(s) with missing image bytes/files; "
            "the source dataset contains unresolved external image paths.",
            RuntimeWarning,
            stacklevel=2,
        )
        return dataset.select(valid_indices)
    return dataset


def _valid_openmedreason(example, max_reasoning_chars):
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    reasoning = str(example.get("reasoning", "")).strip()
    return bool(
        question
        and answer
        and 40 <= len(reasoning) <= max_reasoning_chars
        and TAGGED_REASONING_RE.match(reasoning)
    )


def _take_image_groups(dataset, ordered_keys, target_rows):
    """Select whole image groups until target_rows is reached (or all for <=0)."""
    if target_rows <= 0:
        selected_keys = set(ordered_keys)
    else:
        counts = Counter(dataset["_image_key"])
        selected_keys = set()
        selected_rows = 0
        for key in ordered_keys:
            selected_keys.add(key)
            selected_rows += counts[key]
            if selected_rows >= target_rows:
                break
    selected_indices = [
        index for index, key in enumerate(dataset["_image_key"])
        if key in selected_keys
    ]
    selected = dataset.select(selected_indices)
    remaining_keys = [key for key in ordered_keys if key not in selected_keys]
    return selected, remaining_keys


def load_openmedreason_splits(
    path=RECIPE_PATHS["openmedreason"],
    cold_start_size=10_000,
    validation_size=1_000,
    rl_size=30_000,
    max_reasoning_chars=1_600,
    seed=42,
):
    """Build image-disjoint SFT, validation, RL, and official-test splits.

    The official test split is never used for training or split selection.
    Passing ``rl_size <= 0`` places every remaining eligible training image in
    the RL pool.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OpenMedReason repository not found: {path}")
    if cold_start_size <= 0 or validation_size <= 0:
        raise ValueError("cold_start_size and validation_size must be positive")
    raw = load_dataset(str(path))
    required = {"image", "question", "reasoning", "answer"}
    missing = required.difference(raw["train"].column_names)
    if missing:
        raise ValueError(f"OpenMedReason is missing columns: {sorted(missing)}")

    # Hash the encoded image bytes so repeated questions for one image cannot
    # cross SFT, validation, RL, or the official test boundary.
    train = raw["train"].cast_column("image", Image(decode=False))
    test = raw["test"].cast_column("image", Image(decode=False))
    # OpenMedReason's official test currently contains a small number of rows
    # with null image bytes and bare filenames that are not shipped alongside
    # the parquet file. Remove those corrupt rows before Image() invokes PIL.
    test = _drop_missing_encoded_images(test, "OpenMedReason official-test")
    train_keys = _encoded_image_keys(train)
    test_keys_list = _encoded_image_keys(test)
    test_keys = set(test_keys_list)

    # select() creates lightweight index views. Avoid map/filter over the full
    # 7.5 GB dataset because those operations can materialize large cache copies.
    text_view = train.select_columns(["question", "answer", "reasoning"])
    valid_indices = [
        index
        for index, (key, example) in enumerate(zip(train_keys, text_view))
        if key not in test_keys
        and _valid_openmedreason(example, max_reasoning_chars=max_reasoning_chars)
    ]
    train = train.add_column("_image_key", train_keys).select(valid_indices)
    test = test.add_column("_image_key", test_keys_list)

    unique_keys = sorted(
        set(train["_image_key"]),
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    cold_raw, unique_keys = _take_image_groups(train, unique_keys, cold_start_size)
    validation_raw, unique_keys = _take_image_groups(train, unique_keys, validation_size)
    rl_raw, _ = _take_image_groups(train, unique_keys, rl_size)

    def format_split(split, name):
        split = split.remove_columns(["_image_key"]).cast_column("image", Image())
        return _format_dataset(split, "question", "answer", f"Formatting OpenMedReason/{name}")

    official_test = test.remove_columns(["_image_key"]).cast_column("image", Image())
    return (
        format_split(cold_raw, "cold-start"),
        format_split(validation_raw, "validation"),
        format_split(rl_raw, "rl"),
        _format_dataset(official_test, "question", "answer", "Formatting OpenMedReason/official-test"),
    )

CACHE_PATTERNS = {
    "Vqa_rad": "vqa-rad-*.arrow",
    "Vqa_Agupte": "med_vqa-*.arrow",
    "SLAKE_VQA_EN": "slake-vqa-english-*.arrow",
    "Path_VQA": "path-vqa-*.arrow",
}

LOCAL_PARQUET_REPOS = {
    "Vqa_rad": Path("/root/autodl-tmp/datasets/flaviagiammarino/vqa-rad/data"),
    "Vqa_Agupte": Path("/root/autodl-tmp/datasets/agupte/MedVQA/data"),
    "SLAKE_VQA_EN": Path(
        "/root/autodl-tmp/datasets/mdwiratathya/SLAKE-vqa-english/data"
    ),
    "Path_VQA": Path(
        "/root/autodl-tmp/datasets/flaviagiammarino/path-vqa/data"
    ),
}



def _load_available(name, path):
    if Path(path).exists():
        return load_dataset(path=path)
    parquet_root = LOCAL_PARQUET_REPOS.get(name)
    if parquet_root and parquet_root.exists():
        split_files = {
            split: [str(file) for file in files]
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


def load_mixed_rl_data(
    openmedreason_rl,
    dataset_names=("SLAKE_VQA_EN", "Vqa_rad", "Vqa_Agupte", "Path_VQA"),
    per_dataset_cap=5_000,
    strict_image_split=True,
    seed=42,
):
    """Mix OpenMedReason with answer-only VQA prompts for RL.

    Answer-only datasets never enter SFT; their generated rationale is learned
    through reward while the reference answer remains the verifier target.
    """
    parts = [openmedreason_rl]
    for name in dataset_names:
        train, _, _ = load_medical_vqa(name, strict_image_split=strict_image_split)
        if per_dataset_cap > 0 and len(train) > per_dataset_cap:
            train = train.shuffle(seed=seed).select(range(per_dataset_cap))
        parts.append(train)
    return concatenate_datasets(parts).shuffle(seed=seed)


def load_experiment_datasets(
    openmedreason_path=RECIPE_PATHS["openmedreason"],
    cold_start_size=10_000,
    openmedreason_rl_size=30_000,
    validation_size=1_000,
    max_reasoning_chars=1_600,
    rl_dataset_names=("SLAKE_VQA_EN", "Vqa_rad", "Vqa_Agupte", "Path_VQA"),
    rl_per_dataset_cap=5_000,
    strict_image_split=True,
    seed=42,
):
    """Return OpenMedReason SFT, mixed RL, validation, and official test data."""
    cold, validation, openmedreason_rl, official_test = load_openmedreason_splits(
        path=openmedreason_path,
        cold_start_size=cold_start_size,
        validation_size=validation_size,
        rl_size=openmedreason_rl_size,
        max_reasoning_chars=max_reasoning_chars,
        seed=seed,
    )
    rl_train = load_mixed_rl_data(
        openmedreason_rl,
        dataset_names=rl_dataset_names,
        per_dataset_cap=rl_per_dataset_cap,
        strict_image_split=strict_image_split,
        seed=seed,
    )
    return cold, rl_train, validation, official_test


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
