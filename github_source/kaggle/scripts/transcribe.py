"""Transcribe one WAV file or all WAV files in a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import decoder_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A .wav file or folder containing .wav files")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--language", choices=["yo", "ha"], required=True)
    parser.add_argument("--base-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            "Run scripts\\train.py first, or pass --model-dir to a completed checkpoint."
        )
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True
        )
    except OSError as error:
        if not args.base_model:
            raise ValueError(
                "Only a LoRA adapter was found; pass --base-model used during training."
            ) from error
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            WhisperForConditionalGeneration.from_pretrained(args.base_model), str(model_dir)
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    files = [args.input] if args.input.is_file() else sorted(args.input.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found under {args.input}")
    prompt = decoder_prompt(processor, args.language).to(device).unsqueeze(0)
    for path in files:
        samples, sampling_rate = librosa.load(path, sr=16_000, mono=True)
        audio = processor.feature_extractor(samples, sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(
                input_features=audio.input_features.to(device),
                decoder_input_ids=prompt,
                max_new_tokens=225,
            )
        transcript = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        print(f"{path.name}\t{args.language}\t{transcript}")


if __name__ == "__main__":
    main()
