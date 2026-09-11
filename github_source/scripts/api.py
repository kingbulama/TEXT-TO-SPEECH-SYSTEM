"""Local HTTP API for the Voxara ASR frontend."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if TYPE_CHECKING:
    from src.inference import WhisperTranscriber

app = FastAPI(title="Voxara ASR API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_transcriber: "WhisperTranscriber | None" = None


def get_transcriber() -> "WhisperTranscriber":
    global _transcriber
    if _transcriber is None:
        from src.inference import WhisperTranscriber

        model_dir = Path(os.getenv("ASR_MODEL_DIR", "models/whisper-tiny"))
        _transcriber = WhisperTranscriber(model_dir, os.getenv("ASR_BASE_MODEL"))
    return _transcriber


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form(...)) -> dict[str, str]:
    if language not in {"yo", "ha"}:
        raise HTTPException(status_code=400, detail="language must be 'yo' or 'ha'")
    if not file.filename:
        raise HTTPException(status_code=400, detail="An audio file is required")
    suffix = Path(file.filename).suffix.lower() or ".wav"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(await file.read())
            temporary_path = Path(temporary_file.name)
        text = get_transcriber().transcribe(temporary_path, language)
        return {"text": text, "language": language, "filename": file.filename}
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)