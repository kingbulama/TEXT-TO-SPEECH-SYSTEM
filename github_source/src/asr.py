"""Shared Whisper preprocessing, collation, generation, and metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from jiwer import cer, wer
from transformers import WhisperProcessor

LANGUAGE_NAMES = {"yo": "yoruba", "ha": "hausa"}


@dataclass
class Batch:
    input_features: torch.Tensor
    labels: torch.Tensor
    languages: list[str]


class WhisperLanguageCollator:
    """Create mixed-language Whisper batches with per-example decoder prompts."""

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.feature_extractor = processor.feature_extractor
        self.language_token_ids = {
            code: self.tokenizer.convert_tokens_to_ids(f"<|{code}|>")
            for code in LANGUAGE_NAMES
        }
        self.transcribe_id = self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self.no_timestamps_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    def _label_ids(self, example: dict[str, Any]) -> list[int]:
        language = example["language"]
        if language not in self.language_token_ids:
            raise ValueError(f"Unsupported language: {language}")
        text_ids = self.tokenizer(example["text"], add_special_tokens=True).input_ids
        prefix = [
            self.language_token_ids[language],
            self.transcribe_id,
            self.no_timestamps_id,
        ]
        if text_ids and text_ids[-1] == self.tokenizer.eos_token_id:
            text_ids = text_ids[:-1]
        return prefix + text_ids + [self.tokenizer.eos_token_id]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        audio_arrays = [feature["audio"]["array"] for feature in features]
        sampling_rate = features[0]["audio"]["sampling_rate"]
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        label_lists = [self._label_ids(feature) for feature in features]
        max_length = max(len(labels) for labels in label_lists)
        labels = torch.full(
            (len(label_lists), max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for index, label_ids in enumerate(label_lists):
            labels[index, : len(label_ids)] = torch.tensor(label_ids, dtype=torch.long)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_features": inputs.input_features,
            "labels": labels,
            "languages": [feature["language"] for feature in features],
        }


def decoder_prompt(processor: WhisperProcessor, language: str) -> torch.Tensor:
    """Return the decoder prompt for one explicitly selected language."""
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"language must be one of {sorted(LANGUAGE_NAMES)}")
    ids = processor.tokenizer.convert_tokens_to_ids(
        [f"<|{language}|>", "<|transcribe|>", "<|notimestamps|>"]
    )
    decoder_start_token_id = processor.tokenizer.bos_token_id
    if decoder_start_token_id is None:
        raise ValueError("Whisper tokenizer is missing its decoder start token")
    return torch.tensor([decoder_start_token_id, *ids], dtype=torch.long)


def generate_predictions(
    model: Any,
    processor: WhisperProcessor,
    examples: Iterable[dict[str, Any]],
    *,
    language: str,
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> tuple[list[str], list[str]]:
    """Generate transcripts for examples belonging to one language."""
    model_device = torch.device(device or next(model.parameters()).device)
    examples = list(examples)
    predictions: list[str] = []
    references: list[str] = []
    prompt = decoder_prompt(processor, language).to(model_device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            arrays = [item["audio"]["array"] for item in batch]
            sampling_rate = batch[0]["audio"]["sampling_rate"]
            inputs = processor.feature_extractor(
                arrays,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(model_device)
            prompt_batch = prompt.unsqueeze(0).expand(len(batch), -1)
            generated = model.generate(
                input_features=input_features,
                decoder_input_ids=prompt_batch,
                max_new_tokens=225,
            )
            predictions.extend(processor.tokenizer.batch_decode(generated, skip_special_tokens=True))
            references.extend(item["text"] for item in batch)
    return predictions, references


def metric_report(predictions: list[str], references: list[str]) -> dict[str, float | int]:
    """Compute WER and CER for one language."""
    return {
        "examples": len(references),
        "wer": float(wer(references, predictions)) if references else 0.0,
        "cer": float(cer(references, predictions)) if references else 0.0,
    }
