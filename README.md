# DeepVoice Detection Web

Web UI and FastAPI backend for uploading an audio file and running DeepVoice
fake voice detection.

The detection model source is not copied into this repository. It is linked as a
Git submodule:

```text
external/deepvoice-detection -> https://github.com/t-tanghuru/deepvoice-detection
```

## Project Structure

```text
.
|-- backend/                     # FastAPI API, DB, upload handling, model adapter
|-- frontend/                    # React + Vite + TypeScript UI
|-- external/deepvoice-detection # DeepVoice model source submodule
|-- checkpoints/                 # Model checkpoints tracked with Git LFS
|-- storage/                     # SQLite DB, uploads, and heatmaps at runtime
|-- run_server.py                # FastAPI entry point
|-- requirements.txt             # Python dependencies, including submodule deps
`-- package.json                 # Node/Vite dependencies and scripts
```

Do not add local copies of model code under `model/` or `scripts/`. Use the
submodule as the canonical model source.

Analysis results use the final Whisper encoder + LCNN model path. The web UI
does not show MFCC visualizations. For model evidence, the backend generates
log-Mel gradient heatmap artifacts under `storage/heatmaps/`:

```text
*_logmel_heatmap.png
*_logmel_heatmap.json
*_logmel_heatmap.heatmap.npy
*_logmel_heatmap.mel.npy
```

These heatmap artifacts are runtime outputs and are ignored by Git.

## Clone

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/<your-id>/<your-repo>.git
cd <your-repo>
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Model Checkpoint

The default checkpoint path is:

```text
checkpoints/best_model_tts_whisper_encoder_lcnn.pt
```

This repository is configured to track `*.pt` files with Git LFS. Before pushing
or pulling checkpoints:

```bash
git lfs install
git lfs pull
```

To use a checkpoint outside the repository, set:

```powershell
$env:DEEPVOICE_MODEL_PATH="C:\path\to\best_model_tts_whisper_encoder_lcnn.pt"
```

The model source path can also be overridden if needed:

```powershell
$env:DEEPVOICE_SOURCE_DIR="C:\path\to\deepvoice-detection"
```

## Install

Python 3.9 or 3.10 is recommended for the model stack.

```bash
python -m pip install -r requirements.txt
npm install
```

On Windows, if `python` points to a different version:

```powershell
py -3.9 -m pip install -r requirements.txt
```

## Run

Build the frontend and serve it through FastAPI:

```bash
npm run build
python run_server.py
```

Then open:

```text
http://127.0.0.1:8000
```

For frontend-only development:

```bash
npm run dev
```

Users cloning the repository should use `--recurse-submodules` and have Git LFS
installed if they need the checkpoint from the repository.
