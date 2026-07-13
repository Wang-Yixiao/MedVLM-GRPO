"""Aggregate per-seed metric JSON files into mean and sample standard deviation."""

import argparse
import json
from pathlib import Path
import statistics


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = float(item)
    return result


def unflatten(value):
    result = {}
    for path, item in value.items():
        cursor = result
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = item
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", help="Per-seed JSON files from evaluate_predictions.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = [flatten(json.loads(Path(path).read_text(encoding="utf-8"))) for path in args.metrics]
    common = sorted(set.intersection(*(set(run) for run in runs)))
    aggregated = {}
    for key in common:
        values = [run[key] for run in runs]
        aggregated[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    payload = {
        "num_seeds": len(runs),
        "source_files": [str(Path(path)) for path in args.metrics],
        "metrics": unflatten(aggregated),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
