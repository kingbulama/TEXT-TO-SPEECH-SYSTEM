"""One-file Kaggle workflow for Yoruba/Hausa Whisper ASR.

Kaggle quick start:
1. Add your Common Voice dataset as a Kaggle Dataset.
2. Set DATA_ROOT below to the mounted dataset path.
3. Run this file with MODE='smoke', then MODE='prepare', then MODE='train'.
4. For OpenAI transcription, add OPENAI_API_KEY as a Kaggle Secret and use MODE='openai'.

This file is self-contained and does not require the rest of the repository.
"""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ============================ Kaggle settings ============================
MODE = os.environ.get("MODE", "smoke")  # smoke | prepare | train | local | openai
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/kaggle/input/yoruba-hausa-common-voice"))
WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/kaggle/working/yoruba-hausa-asr"))
RAW_ROOT = DATA_ROOT / "data" / "raw" if (DATA_ROOT / "data" / "raw").is_dir() else DATA_ROOT
PROCESSED_ROOT = WORK_ROOT / "processed"
MODEL_ROOT = WORK_ROOT / "model"
AUDIO_FILE = os.environ.get("AUDIO_FILE", "")
LANGUAGE = os.environ.get("LANGUAGE", "yo")
BASE_MODEL = os.environ.get("BASE_MODEL", "openai/whisper-small")
TRAIN_LIMIT = int(os.environ.get("TRAIN_LIMIT", "0"))  # 0 means all examples
EPOCHS = float(os.environ.get("EPOCHS", "1"))

LANGUAGE_NAMES = {"yo": "yoruba", "ha": "hausa"}


def install_dependencies() -> None:
    """Install packages when this file is run in a fresh Kaggle notebook."""
    import subprocess
    import sys

    packages = [
        "datasets[audio]>=2.19.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "peft>=0.10.0",
        "jiwer>=3.0.3",
        "librosa>=0.10.2",
        "soundfile>=0.12.1",
        "openai>=1.40.0",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def normalize_text(text: str, preserve_diacritics: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    if not preserve_diacritics:
        text = "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )
        text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def find_split_file(language_dir: Path, split: str) -> Path:
    names = {"train": ["train.tsv"], "validation": ["dev.tsv", "validation.tsv"], "test": ["test.tsv"]}
    for name in names[split]:
        candidate = language_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {split} TSV in {language_dir}")


def prepare_one_language(language: str, split: str):
    from datasets import Dataset
    import librosa

    language_dir = RAW_ROOT / LANGUAGE_NAMES[language]
    metadata = find_split_file(language_dir, split)
    clips_dir = language_dir / "clips"
    rows: list[dict[str, Any]] = []
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            audio_path = clips_dir / row["path"]
            if not audio_path.is_file():
                continue
            try:
                samples, _ = librosa.load(audio_path, sr=16_000, mono=True)
            except Exception:
                continue
            duration = len(samples) / 16_000
            if not 0.5 <= duration <= 30.0 or len(samples) == 0:
                continue
            rms = float((samples**2).mean() ** 0.5)
            text = normalize_text(row.get("sentence", ""), language == "ha" or True)
            if rms < 1e-4 or not text:
                continue
            rows.append({
                "audio": {"array": samples.tolist(), "sampling_rate": 16_000},
                "text": text,
                "language": language,
                "language_name": LANGUAGE_NAMES[language],
            })
            if TRAIN_LIMIT and split == "train" and len(rows) >= TRAIN_LIMIT:
                break
    if not rows:
        raise ValueError(f"No usable {language}/{split} examples found under {language_dir}")
    return Dataset.from_list(rows)


def prepare_dataset():
    from datasets import DatasetDict, concatenate_datasets, interleave_datasets

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    train_parts = [prepare_one_language(code, "train") for code in LANGUAGE_NAMES]
    validation_parts = [prepare_one_language(code, "validation") for code in LANGUAGE_NAMES]
    test_parts = [prepare_one_language(code, "test") for code in LANGUAGE_NAMES]
    train = interleave_datasets(train_parts, stopping_strategy="all_exhausted", seed=42)
    dataset = DatasetDict({
        "train": train,
        "validation": concatenate_datasets(validation_parts),
        "test": concatenate_datasets(test_parts),
    })
    dataset.save_to_disk(str(PROCESSED_ROOT))
    print(dataset)
    return dataset


def decoder_prompt(processor, language: str):
    import torch

    if language not in LANGUAGE_NAMES:
        raise ValueError("LANGUAGE must be yo or ha")
    ids = processor.tokenizer.convert_tokens_to_ids(
        [f"<|{language}|>", "<|transcribe|>", "<|notimestamps|>"]
    )
    start = processor.tokenizer.bos_token_id
    if start is None:
        raise ValueError("Tokenizer has no decoder start token")
    return torch.tensor([start, *ids], dtype=torch.long)


class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        arrays = [item["audio"]["array"] for item in features]
        rate = features[0]["audio"]["sampling_rate"]
        inputs = self.processor.feature_extractor(arrays, sampling_rate=rate, return_tensors="pt")
        labels = []
        for item in features:
            language_id = self.tokenizer.convert_tokens_to_ids(f"<|{item['language']}|>")
            ids = self.tokenizer(item["text"], add_special_tokens=True).input_ids
            if ids and ids[-1] == self.tokenizer.eos_token_id:
                ids = ids[:-1]
            labels.append([language_id, self.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
                           self.tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
                           *ids, self.tokenizer.eos_token_id])
        max_len = max(map(len, labels))
        output = torch.full((len(labels), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        for index, ids in enumerate(labels):
            output[index, :len(ids)] = torch.tensor(ids)
        output[output == self.tokenizer.pad_token_id] = -100
        return {"input_features": inputs.input_features, "labels": output}


def train():
    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    if not PROCESSED_ROOT.exists():
        prepare_dataset()
    dataset = load_from_disk(str(PROCESSED_ROOT))
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.set_prefix_tokens(language=None, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none", task_type=TaskType.SEQ_2_SEQ_LM,
    ))
    model.print_trainable_parameters()
    args = Seq2SeqTrainingArguments(
        output_dir=str(MODEL_ROOT), per_device_train_batch_size=4,
        per_device_eval_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=1e-4, num_train_epochs=EPOCHS, warmup_ratio=0.05,
        fp16=torch.cuda.is_available(), gradient_checkpointing=True,
        save_strategy="epoch", evaluation_strategy="epoch", logging_steps=10,
        save_total_limit=2, report_to="none", remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=Collator(processor), tokenizer=processor.feature_extractor,
    )
    trainer.train()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_ROOT))
    processor.save_pretrained(str(MODEL_ROOT))
    print(f"Saved LoRA model to {MODEL_ROOT}")


def local_transcribe(audio_file: Path, language: str = LANGUAGE) -> str:
    import librosa
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(str(MODEL_ROOT), local_files_only=True)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_ROOT), local_files_only=True)
    except OSError:
        model = PeftModel.from_pretrained(
            WhisperForConditionalGeneration.from_pretrained(BASE_MODEL), str(MODEL_ROOT)
        )
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    samples, _ = librosa.load(audio_file, sr=16_000, mono=True)
    inputs = processor.feature_extractor(samples, sampling_rate=16_000, return_tensors="pt")
    device = next(model.parameters()).device
    with torch.inference_mode():
        tokens = model.generate(
            input_features=inputs.input_features.to(device),
            decoder_input_ids=decoder_prompt(processor, language).to(device).unsqueeze(0),
            max_new_tokens=225,
        )
    return processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def openai_transcribe(audio_file: Path, language: str = LANGUAGE) -> str:
    """Transcribe through OpenAI. Store OPENAI_API_KEY in Kaggle Secrets, never in this file."""
    try:
        from kaggle_secrets import UserSecretsClient
        api_key = UserSecretsClient().get_secret("OPENAI_API_KEY")
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Add OPENAI_API_KEY to Kaggle Secrets or the environment")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    with audio_file.open("rb") as handle:
        result = client.audio.transcriptions.create(
            model=os.environ.get("OPENAI_AUDIO_MODEL", "gpt-4o-mini-transcribe"),
            file=handle,
            language="yo" if language == "yo" else "ha",
            response_format="text",
        )
    return str(result)


def smoke_test():
    import torch
    print({"python": os.sys.version, "torch": torch.__version__, "cuda": torch.cuda.is_available()})
    print("Smoke test passed. Set MODE=prepare to process data or MODE=train to fine-tune.")


def main():
    if MODE == "smoke":
        smoke_test()
    elif MODE == "prepare":
        install_dependencies()
        prepare_dataset()
    elif MODE == "train":
        install_dependencies()
        train()
    elif MODE in {"local", "openai"}:
        if not AUDIO_FILE:
            raise ValueError("Set AUDIO_FILE=/kaggle/input/.../audio.mp3")
        path = Path(AUDIO_FILE)
        text = local_transcribe(path) if MODE == "local" else openai_transcribe(path)
        print(json.dumps({"language": LANGUAGE, "text": text}, ensure_ascii=False))
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
