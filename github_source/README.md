# Yoruba and Hausa Multilingual ASR

This project fine-tunes one multilingual speech-recognition model on Mozilla Common Voice Yoruba (`yo`) and Hausa (`ha`). The initial recommendation is Whisper with parameter-efficient LoRA fine-tuning: Whisper already supports both languages, is more practical than training a new acoustic/language-conditioning stack on small datasets, and works well on a Colab T4. No local GPU is required: preparation, tests, evaluation, and CPU inference run locally; training runs in Colab.

## Project layout

- `data/`: local processed datasets and manifests
- `models/`: model and checkpoint artifacts
- `notebooks/`: exploratory experiments
- `scripts/`: command-line entry points
- `src/`: reusable data and training code
- `tests/`: focused unit tests
- `speech-to-text-frontend/`: React/Vite browser frontend
- `kaggle/`: complete Kaggle training package

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare Common Voice data

Mozilla Common Voice audio is now distributed through Mozilla Data Collective rather than the old Hugging Face dataset repository. After downloading the Yoruba and Hausa archives, extract them into `data/raw`:

```powershell
data/raw/yoruba/train.tsv
data/raw/yoruba/dev.tsv
data/raw/yoruba/test.tsv
data/raw/yoruba/clips/

data/raw/hausa/train.tsv
data/raw/hausa/dev.tsv
data/raw/hausa/test.tsv
data/raw/hausa/clips/
```

Then run:

```powershell
python scripts/prepare_data.py
```

The script reads the local Common Voice TSV files, casts audio to 16 kHz, peak-normalizes samples, removes clips outside 0.5-30 seconds or below the configurable near-silence RMS threshold, normalizes transcripts, adds `language` and `language_name`, and saves a Hugging Face `DatasetDict` under `data/processed/common_voice`. Keep the official train, dev, and test files separate; using one `validated.tsv` file for all three would invalidate evaluation. Adjust the quality threshold with `--min-rms` when inspecting a release with different recording levels.

Training is balanced by equal-probability interleaving. The smaller language is revisited when necessary; validation and test remain unbalanced so per-language evaluation reflects the source splits. Yoruba diacritics are preserved by default. To run an ablation that strips tone/under-dot marks, use `--strip-yoruba-diacritics` and use the same choice later during evaluation and inference.

## Current model decisions

Use explicit language selection at inference first (`yo` or `ha`) because the datasets are small and language identification can make errors before transcription. Automatic detection can be added as a separate mode after the language-conditioned baseline is working. The evaluation pipeline will always report WER and CER separately for each language, alongside the combined score.

## Train in Google Colab

Do not run training on the local CPU unless you are deliberately running a very small smoke test. The default uses `openai/whisper-small` with LoRA adapters on the free Colab T4. Use `openai/whisper-base` and/or reduce the batch size if memory is tight. The Colab notebook will mount Drive, install requirements, import the training function, and write resumable checkpoints to Drive.

```powershell
python scripts/train.py --model-name openai/whisper-small --output-dir /content/drive/MyDrive/yoruba-hausa-asr
```

The command above is shown for the Colab terminal or a notebook cell, not for local CPU training.

Training downloads the base Whisper model and creates the local directory used by inference. Do not run inference with the default model path until training has completed and that directory contains the saved processor and checkpoint files.

Training uses the language token embedded in every target sequence, so mixed-language batches remain correctly conditioned. Checkpoints are saved each epoch and training can resume with:

```powershell
python scripts/train.py --resume-from-checkpoint models/whisper-yoruba-hausa/checkpoint-XXXX
```

## Evaluate

Evaluation keeps Yoruba and Hausa separate and writes a JSON report containing WER, CER, and example counts for each language plus a combined score:

```powershell
python scripts/evaluate.py --model-dir models/whisper-yoruba-hausa --output models/evaluation_report.json
```

For a LoRA adapter directory that does not contain a merged base model, provide the same base model used for training:

```powershell
python scripts/evaluate.py --model-dir models/whisper-yoruba-hausa --base-model openai/whisper-small
```

## Inference

Explicit language selection is required initially because it is more reliable than automatic language identification with small, imbalanced datasets:

```powershell
python scripts/transcribe.py audio/example.wav --model-dir models/whisper-yoruba-hausa --language yo
python scripts/transcribe.py audio/ --model-dir models/whisper-yoruba-hausa --language ha
```

The folder mode transcribes every `.wav` file and prints `filename`, language, and transcript. Automatic language detection can be added later as an opt-in mode, but should not replace the explicit baseline until it is evaluated independently.

## Run the local API and frontend

The frontend sends `POST /api/transcribe` to the FastAPI service. Start the API from the repository root after training a model:

```powershell
$env:ASR_MODEL_DIR = "models/whisper-yoruba-hausa"
$env:ASR_BASE_MODEL = "openai/whisper-small"
python -m uvicorn scripts.api:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```powershell
Set-Location speech-to-text-frontend
npm run dev
```

Open `http://localhost:3000`. Vite proxies `/api` to the API on port 8000. The API loads the model only when the first transcription request arrives and automatically uses CPU when CUDA is unavailable.

## Validation

```powershell
python -m pytest -q
python -m compileall -q src scripts tests
```

## Kaggle one-file workflow

The complete Kaggle package is in the [`kaggle/`](kaggle/) folder. It includes a
ready-to-open notebook, the standalone runner, dependencies, and Kaggle-specific
instructions.

For a self-contained Kaggle workflow, upload `kaggle_asr_one_file.py` together with a
dataset containing `yoruba/` and `hausa/` Common Voice folders. Set these variables in
the Kaggle notebook before running the file:

```python
import os
os.environ["DATA_ROOT"] = "/kaggle/input/your-dataset-name"
os.environ["MODE"] = "smoke"
%run /kaggle/input/your-code-dataset/kaggle_asr_one_file.py
```

Then use `MODE="prepare"` to create processed data and `MODE="train"` to fine-tune
Whisper with LoRA. The default base model is `openai/whisper-small`; use
`BASE_MODEL="openai/whisper-tiny"` for a faster smoke run.

The same file supports OpenAI transcription without exposing a key in source code. Add
an `OPENAI_API_KEY` Kaggle Secret, then run:

```python
os.environ["MODE"] = "openai"
os.environ["LANGUAGE"] = "yo"
os.environ["AUDIO_FILE"] = "/kaggle/input/your-audio/audio.mp3"
%run /kaggle/input/your-code-dataset/kaggle_asr_one_file.py
```

OpenAI can transcribe audio through its API, but it does not train your local Whisper
LoRA model. API usage requires internet access, a valid key, and is billed separately.
