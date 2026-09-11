"""Load, clean, label, and balance Yoruba/Hausa Common Voice data."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset


@dataclass(frozen=True)
class LanguageSpec:
    """A Common Voice language configuration."""

    code: str
    name: str


LANGUAGES = (LanguageSpec("yo", "yoruba"), LanguageSpec("ha", "hausa"))


def normalize_transcript(text: str, *, preserve_diacritics: bool = True) -> str:
    """Normalize whitespace and Unicode while optionally removing combining marks.

    Yoruba tone and under-dot marks can carry linguistic information. The default
    preserves them; stripping them is available for experiments and must be kept
    consistent across training, evaluation, and inference.
    """
    normalized = unicodedata.normalize("NFKC", text or "")
    if not preserve_diacritics:
        normalized = "".join(
            character
            for character in unicodedata.normalize("NFD", normalized)
            if unicodedata.category(character) != "Mn"
        )
        normalized = unicodedata.normalize("NFC", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _prepare_example(
    example: dict,
    language: LanguageSpec,
    preserve_yoruba_diacritics: bool,
    target_sampling_rate: int,
) -> dict:
    audio = example["audio"]
    samples = audio["array"]
    sampling_rate = audio["sampling_rate"]
    peak = max((abs(float(sample)) for sample in samples), default=0.0)
    if peak > 0:
        samples = [float(sample) / peak for sample in samples]

    example["audio"] = {"array": samples, "sampling_rate": sampling_rate}
    example["text"] = normalize_transcript(
        example.get("sentence", example.get("text", "")),
        preserve_diacritics=preserve_yoruba_diacritics or language.code != "yo",
    )
    example["language"] = language.code
    example["language_name"] = language.name
    example["duration"] = len(samples) / target_sampling_rate
    example["rms"] = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    return example


def _load_local_audio(example: dict, target_sampling_rate: int) -> dict:
    """Read one local Common Voice clip without the datasets audio decoder."""
    samples, sampling_rate = sf.read(example["audio_path"], dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    if sampling_rate != target_sampling_rate:
        samples = librosa.resample(
            samples,
            orig_sr=sampling_rate,
            target_sr=target_sampling_rate,
        )
    example["audio"] = {
        "array": samples.tolist(),
        "sampling_rate": target_sampling_rate,
    }
    return example


def _has_usable_audio(
    example: dict,
    min_duration: float,
    max_duration: float,
    min_rms: float,
) -> bool:
    duration = example.get("duration", 0.0)
    return (
        min_duration <= duration <= max_duration
        and example.get("rms", 0.0) >= min_rms
        and bool(example.get("text"))
    )


def load_language_split(
    dataset_repo: str,
    language: LanguageSpec,
    split: str,
    *,
    raw_dir: Path | None = None,
    revision: str | None = None,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
    min_rms: float = 1e-4,
    preserve_yoruba_diacritics: bool = True,
    target_sampling_rate: int = 16_000,
) -> Dataset:
    """Load and preprocess one Common Voice split from disk or Hugging Face."""
    if raw_dir is not None:
        language_dir = raw_dir / language.name
        split_candidates = {
            "train": ["train.tsv"],
            "validation": ["dev.tsv", "validation.tsv"],
            "test": ["test.tsv"],
        }
        metadata_path = next(
            (language_dir / name for name in split_candidates[split] if (language_dir / name).is_file()),
            None,
        )
        clips_dir = language_dir / "clips"
        if metadata_path is None or not clips_dir.is_dir():
            raise FileNotFoundError(
                f"Missing Common Voice files for {language.name}. Expected {language_dir} "
                "to contain train.tsv, dev.tsv, test.tsv, and clips\\. "
                "Do not reuse validated.tsv for every split because that causes evaluation leakage."
            )
        audio_paths: list[str] = []
        sentences: list[str] = []
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
            reader = csv.DictReader(metadata_file, delimiter="\t")
            if "path" not in (reader.fieldnames or {}) or "sentence" not in (reader.fieldnames or {}):
                raise ValueError(f"{metadata_path} must contain path and sentence columns")
            for row in reader:
                audio_path = clips_dir / row["path"]
                if audio_path.is_file():
                    audio_paths.append(str(audio_path))
                    sentences.append(row["sentence"])
        if not audio_paths:
            raise ValueError(f"No matching audio files found for {metadata_path}")
        dataset = Dataset.from_dict({"audio_path": audio_paths, "sentence": sentences})
        dataset = dataset.map(
            lambda example: _load_local_audio(example, target_sampling_rate),
            remove_columns=["audio_path"],
        )
    else:
        dataset = load_dataset(
            dataset_repo,
            language.code,
            split=split,
            revision=revision,
        )
    if "audio" not in dataset.column_names:
        raise ValueError(f"Expected an 'audio' column in {dataset_repo}/{language.code}")

    # Local audio was decoded and resampled with soundfile/librosa above.
    # Avoid datasets.Audio here because newer datasets releases invoke TorchCodec.
    if raw_dir is None:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
    dataset = dataset.map(
        lambda example: _prepare_example(
            example,
            language,
            preserve_yoruba_diacritics,
            target_sampling_rate,
        ),
    )
    columns_to_remove = [
        column
        for column in dataset.column_names
        if column not in {"audio", "text", "language", "language_name", "duration", "rms"}
    ]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.filter(
        lambda example: _has_usable_audio(example, min_duration, max_duration, min_rms)
    )
    return dataset


def combine_language_splits(
    language_datasets: Iterable[Dataset],
    *,
    balance: bool,
    seed: int,
) -> Dataset:
    """Combine datasets, optionally oversampling smaller languages equally.

    Equal interleaving uses the same probability for each language and
    ``all_exhausted`` so the smaller language is revisited instead of silently
    disappearing early. Validation and test sets should normally use
    ``balance=False`` so their natural distributions remain measurable.
    """
    datasets = list(language_datasets)
    if not datasets:
        raise ValueError("At least one language dataset is required")
    if len(datasets) == 1:
        return datasets[0].shuffle(seed=seed)
    if balance:
        probability = 1.0 / len(datasets)
        return interleave_datasets(
            datasets,
            probabilities=[probability] * len(datasets),
            seed=seed,
            stopping_strategy="all_exhausted",
        )
    return concatenate_datasets(datasets).shuffle(seed=seed)


def build_dataset_dict(
    dataset_repo: str,
    *,
    raw_dir: Path | None,
    revision: str | None,
    splits: Iterable[str],
    min_duration: float,
    max_duration: float,
    min_rms: float,
    preserve_yoruba_diacritics: bool,
    seed: int,
    balance_train: bool,
) -> DatasetDict:
    """Build a processed DatasetDict with balanced training data."""
    result = {}
    for split in splits:
        language_datasets = [
            load_language_split(
                dataset_repo,
                language,
                split,
                raw_dir=raw_dir,
                revision=revision,
                min_duration=min_duration,
                max_duration=max_duration,
                min_rms=min_rms,
                preserve_yoruba_diacritics=preserve_yoruba_diacritics,
            )
            for language in LANGUAGES
        ]
        result[split] = combine_language_splits(
            language_datasets,
            balance=balance_train and split == "train",
            seed=seed,
        )
        print(f"{split}: {len(result[split])} examples")
    return DatasetDict(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-repo",
        default="mozilla-foundation/common_voice_17_0",
        help="Common Voice dataset repository, including its release number.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Local Common Voice root containing yoruba/ and hausa/ folders.",
    )
    parser.add_argument(
        "--source",
        choices=["local", "huggingface"],
        default="local",
        help="Use extracted Mozilla Data Collective files or the legacy Hub source.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/common_voice"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--min-rms",
        type=float,
        default=1e-4,
        help="Reject near-silent clips below this normalized RMS threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Equalize language sampling in train; validation/test stay unbalanced.",
    )
    parser.add_argument(
        "--strip-yoruba-diacritics",
        action="store_true",
        help="Remove Yoruba combining marks; preserve them by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset_dict(
        args.dataset_repo,
        raw_dir=args.raw_dir if args.source == "local" else None,
        revision=args.revision,
        splits=args.splits,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_rms=args.min_rms,
        preserve_yoruba_diacritics=not args.strip_yoruba_diacritics,
        seed=args.seed,
        balance_train=args.balance_train,
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    print(f"Saved processed dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
