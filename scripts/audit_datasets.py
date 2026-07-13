"""Audit cached Hugging Face medical-VQA Arrow files for leakage and quality."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset


PATTERNS = {
    "VQA-RAD": "vqa-rad-*.arrow",
    "SLAKE": "slake-vqa-english-*.arrow",
    "Agupte": "med_vqa-*.arrow",
}


def image_hash(image):
    return hashlib.sha256(image.mode.encode() + str(image.size).encode() + image.tobytes()).hexdigest()


def audit(root):
    report = {"root": str(root.resolve()), "datasets": {}}
    for name, pattern in PATTERNS.items():
        files = sorted(root.rglob(pattern))
        entry, split_sets = {"splits": {}, "overlap": {}}, {}
        for file in files:
            split = next((x for x in ("train", "validation", "test") if f"-{x}" in file.name), file.stem)
            ds = Dataset.from_file(str(file))
            qcol = "question" if "question" in ds.column_names else "questions"
            acol = "answer" if "answer" in ds.column_names else "answers"
            icol = "image" if "image" in ds.column_names else "images"
            questions = [str(x).strip().casefold() for x in ds[qcol]]
            answers = [str(x).strip().casefold() for x in ds[acol]]
            images = [image_hash(x) for x in ds[icol]]
            qa = list(zip(questions, answers))
            split_sets[split] = (set(images), set(qa))
            entry["splits"][split] = {
                "rows": len(ds), "unique_images": len(set(images)), "unique_questions": len(set(questions)),
                "duplicate_qa_rows": len(qa) - len(set(qa)),
                "yes_no_percent": round(100 * sum(x in {"yes", "no"} for x in answers) / max(len(ds), 1), 2),
                "top_answers": Counter(answers).most_common(10),
            }
        keys = list(split_sets)
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                entry["overlap"][f"{left}_vs_{right}"] = {
                    "exact_images": len(split_sets[left][0] & split_sets[right][0]),
                    "question_answers": len(split_sets[left][1] & split_sets[right][1]),
                }
        report["datasets"][name] = entry
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_audit.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
