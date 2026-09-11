"""Fine-tune one multilingual Whisper model on Yoruba and Hausa."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import LANGUAGE_NAMES, WhisperLanguageCollator, generate_predictions, metric_report


class LanguageMetricsCallback:
    """Log validation WER/CER independently for Yoruba and Hausa each epoch."""

    def __init__(self, processor, validation_dataset, batch_size: int):
        self.processor = processor
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.trainer = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        for language in LANGUAGE_NAMES:
            subset = self.validation_dataset.filter(
                lambda example: example["language"] == language
            )
            predictions, references = generate_predictions(
                model,
                self.processor,
                subset,
                language=language,
                batch_size=self.batch_size,
            )
            metrics = metric_report(predictions, references)
            metrics.pop("examples", None)
            for metric_name, value in metrics.items():
                if self.trainer is not None:
                    self.trainer.log({f"eval_{language}_{metric_name}": value})
        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed/common_voice"))
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", type=Path, default=Path("models/whisper-yoruba-hausa"))
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def language_subset(dataset: Dataset, language: str) -> Dataset:
    return dataset.filter(lambda example: example["language"] == language)


def latest_checkpoint(output_dir: Path) -> str | None:
    """Return the newest Trainer checkpoint in ``output_dir``, if one exists."""
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: path.stat().st_mtime))


def train_model(args: argparse.Namespace):
    """Train the multilingual Whisper LoRA model from a parsed configuration."""
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_dir}. "
            "Run 'python scripts\\prepare_data.py' before training."
        )
    dataset = load_from_disk(str(dataset_dir))
    if not isinstance(dataset, DatasetDict) or "train" not in dataset or "validation" not in dataset:
        raise ValueError("Expected a DatasetDict containing train and validation splits")

    processor = WhisperProcessor.from_pretrained(args.model_name)
    processor.tokenizer.set_prefix_tokens(language=None, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    strategy_argument = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_kwargs = {
        strategy_argument: "epoch",
    }
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        save_total_limit=2,
        predict_with_generate=False,
        load_best_model_at_end=False,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        **training_kwargs,
    )
    trainer_kwargs = {
        "processing_class"
        if "processing_class" in inspect.signature(Seq2SeqTrainer).parameters
        else "tokenizer": processor.feature_extractor,
    }
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=WhisperLanguageCollator(processor),
        **trainer_kwargs,
    )
    metrics_callback = LanguageMetricsCallback(
        processor, dataset["validation"], args.eval_batch_size
    )
    metrics_callback.trainer = trainer
    trainer.add_callback(metrics_callback)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    metadata = {
        "base_model": args.model_name,
        "languages": LANGUAGE_NAMES,
        "language_conditioning": "per-example label prefix and explicit generation prompt",
        "peft": "LoRA",
        "diacritics": "preserved by data preparation default",
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return trainer


def main() -> None:
    train_model(parse_args())


if __name__ == "__main__":
    main()
