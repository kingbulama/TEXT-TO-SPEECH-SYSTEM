# Yoruba and Hausa ASR for Kaggle

This folder is a complete Kaggle package for preparing Common Voice data, fine-tuning Whisper with LoRA, running local inference, and optionally using the OpenAI transcription API.

## Files

- `kaggle_asr_one_file.py`: self-contained workflow runner
- `kaggle_asr_workflow.ipynb`: ready-to-open Kaggle notebook
- `requirements-kaggle.txt`: notebook dependencies
- `src/`: reusable preprocessing, batching, metrics, and inference code
- `scripts/`: data preparation, LoRA training, evaluation, transcription, and API entry points
- `tests/`: project tests
- `notebooks/`: the original Colab training notebook
- `colab_upload/`: notebook upload copy
- `frontend/`: complete React/Vite frontend source and configuration

## Kaggle setup

1. Create a Kaggle Notebook with GPU enabled.
2. Attach a Kaggle Dataset containing this structure:

```text
/kaggle/input/your-dataset/
  yoruba/
    train.tsv
    dev.tsv
    test.tsv
    clips/
  hausa/
    train.tsv
    dev.tsv
    test.tsv
    clips/
```

The runner also accepts a dataset containing `data/raw/yoruba` and `data/raw/hausa`.

3. Upload this entire `kaggle` folder as a Kaggle Dataset, or add its files to a Kaggle Notebook.
4. Set `DATA_ROOT` in the first notebook cell.
5. Run smoke, prepare, and train in order.

## Train with the full source tree

From a Kaggle Notebook cell, after installing `requirements-kaggle.txt`:

```python
import os
os.environ["PYTHONPATH"] = "/kaggle/input/your-code-dataset"
```

Prepare Common Voice data:

```python
!python /kaggle/input/your-code-dataset/scripts/prepare_data.py \
  --raw-dir /kaggle/input/yoruba-hausa-common-voice \
  --output-dir /kaggle/working/yoruba-hausa-asr/processed
```

Train the LoRA model on a Kaggle GPU:

```python
!python /kaggle/input/your-code-dataset/scripts/train.py \
  --dataset-dir /kaggle/working/yoruba-hausa-asr/processed \
  --model-name openai/whisper-small \
  --output-dir /kaggle/working/yoruba-hausa-asr/model \
  --epochs 1 \
  --train-batch-size 4 \
  --eval-batch-size 4
```

Evaluate the trained model:

```python
!python /kaggle/input/your-code-dataset/scripts/evaluate.py \
  --dataset-dir /kaggle/working/yoruba-hausa-asr/processed \
  --model-dir /kaggle/working/yoruba-hausa-asr/model \
  --base-model openai/whisper-small \
  --output /kaggle/working/yoruba-hausa-asr/evaluation.json
```

The trained adapter and processor are saved under `/kaggle/working/yoruba-hausa-asr/model`.
Download that directory from the Kaggle notebook output. For later inference, pass the
same base model used during training with `--base-model openai/whisper-small` when the
saved directory contains LoRA adapter weights.

## Complete package contents

This folder is intentionally self-contained. The training code does not import files
from the original Windows workspace, and the frontend is included for deployment after
training. Do not upload `data/`, `models/`, `node_modules/`, or generated caches to
Kaggle; attach the audio dataset as a separate Kaggle Dataset instead.

## Modes

Set `MODE` before running the Python file:

- `smoke`: verify Python, PyTorch, and CUDA
- `prepare`: decode audio and create `/kaggle/working/yoruba-hausa-asr/processed`
- `train`: fine-tune `openai/whisper-small` with LoRA
- `local`: transcribe using the saved local model
- `openai`: transcribe through OpenAI's API

For a quick training test, use `BASE_MODEL="openai/whisper-tiny"`, `TRAIN_LIMIT="8"`, and `EPOCHS="0.01"`.

## OpenAI option

Yes, OpenAI can be connected for transcription. Add a Kaggle Secret named `OPENAI_API_KEY`; never paste the key into this file. Set:

```python
import os
os.environ["MODE"] = "openai"
os.environ["LANGUAGE"] = "yo"
os.environ["AUDIO_FILE"] = "/kaggle/input/your-audio/audio.mp3"
%run /kaggle/input/your-code-dataset/kaggle_asr_one_file.py
```

OpenAI transcription is an API service and does not train your local Yoruba/Hausa LoRA model. It requires Kaggle internet access and may incur API charges.

## Outputs

Training outputs are written to `/kaggle/working/yoruba-hausa-asr/model`. Download that directory as a Kaggle output or copy it to a Kaggle Dataset for later inference.
