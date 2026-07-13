"""Evaluate JSONL predictions without loading the training model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medvlm_grpo.metrics import compute_generation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="JSONL rows containing prediction and reference")
    parser.add_argument("--output", help="Optional metrics JSON output path")
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore (loads a neural model)")
    parser.add_argument("--bertscore_model", default=None, help="Optional local/Hugging Face BERTScore model")
    parser.add_argument("--bertscore_num_layers", type=int, default=None, help="Required for unregistered local BERTScore models")
    parser.add_argument("--no_meteor", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.predictions, encoding="utf-8") if line.strip()]
    metrics = compute_generation_metrics(
        [x["prediction"] for x in rows],
        [x["reference"] for x in rows],
        [x.get("question", "") for x in rows],
        include_meteor=not args.no_meteor,
        include_bertscore=args.bertscore,
        bertscore_model=args.bertscore_model,
        bertscore_num_layers=args.bertscore_num_layers,
    )
    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
