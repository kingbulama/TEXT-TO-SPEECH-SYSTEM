"""Reusable Whisper inference for the local transcription API and CLI."""

from __future__ import annotations

from pathlib import Path


class WhisperTranscriber:
    """Load a trained Whisper or LoRA adapter and transcribe local audio files."""

    def __init__(self, model_dir: Path, base_model: str | None = None):
        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        from src.asr import decoder_prompt

        model_dir = model_dir.resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self.processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                str(model_dir), local_files_only=True
            )
        except OSError as error:
            if not base_model:
                raise ValueError(
                    "The model directory contains a LoRA adapter. Set ASR_BASE_MODEL."
                ) from error
            model = PeftModel.from_pretrained(
                WhisperForConditionalGeneration.from_pretrained(base_model),
                str(model_dir),
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self._torch = torch
        self._decoder_prompt = decoder_prompt

    def transcribe(self, audio_path: Path, language: str) -> str:
        """Transcribe one audio file with an explicit Yoruba or Hausa prompt."""
        import librosa

        if language not in {"yo", "ha"}:
            raise ValueError("language must be 'yo' or 'ha'")
        samples, sampling_rate = librosa.load(audio_path, sr=16_000, mono=True)
        inputs = self.processor.feature_extractor(
            samples, sampling_rate=16_000, return_tensors="pt"
        )
        prompt = self._decoder_prompt(self.processor, language).to(self.device).unsqueeze(0)
        with self._torch.inference_mode():
            tokens = self.model.generate(
                input_features=inputs.input_features.to(self.device),
                decoder_input_ids=prompt,
                max_new_tokens=225,
            )
        return self.processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()