"""Evaluate a trained model with WER/CER reported per language."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import LANGUAGE_NAMES, generate_predictions, metric_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/common_voice"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("models/evaluation_report.json"))
    return parser.parse_args()


def load_model(model_dir: Path, base_model: str | None):
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. Run scripts\\train.py first."
        )
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True
        )
    except OSError:
        if not base_model:
            raise ValueError("LoRA adapter detected; pass --base-model used during training")
        model = PeftModel.from_pretrained(
            WhisperForConditionalGeneration.from_pretrained(base_model), str(model_dir)
        )
    return processor, model


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(str(args.dataset_dir))
    if not isinstance(dataset, DatasetDict):
        raise ValueError("Expected a DatasetDict on disk")
    processor, model = load_model(args.model_dir, args.base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    report = {"split": args.split, "languages": {}, "combined": {}}
    all_predictions: list[str] = []
    all_references: list[str] = []
    for language in LANGUAGE_NAMES:
        subset = dataset[args.split].filter(lambda example: example["language"] == language)
        predictions, references = generate_predictions(
            model, processor, subset, language=language, batch_size=args.batch_size, device=device
        )
        report["languages"][language] = metric_report(predictions, references)
        all_predictions.extend(predictions)
        all_references.extend(references)
    report["combined"] = metric_report(all_predictions, all_references)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
