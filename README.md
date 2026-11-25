# Facial Expression Recorder

A small webcam project that estimates facial expression scores, saves each recording for later, and generates a time-based heatmap and summary files.

## Setup

Use Python 3.10 or newer. From this folder:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The requirements pin a compatible setuptools version for FER. If you already installed dependencies, run `pip install -r requirements.txt` again. The first run of the expression detector may download its model weights. Allow camera access when prompted.

## Use

```bash
python main.py record --name "practice session"
python main.py list
python main.py analyze
```

Press **q** in the camera window to stop. Recordings are stored in `recordings/<session-id>/`; each includes `expressions.csv`, `session.json`, and an annotated MP4. To save only expression data, use `python main.py record --no-video --seconds 60`. Analyze a specific session with `python main.py analyze <session-id>`.

Analysis creates `recordings/<session-id>/analysis/expression_heatmap.png`, `summary.csv`, and `summary.json`. The CSV contains one row per captured frame, model scores for seven expression categories, detected face count, and face boxes.

## Notes

Expression scores are estimates from a general-purpose model, not reliable measures of a person's internal emotional state. Run locally and get informed consent before recording anyone. Sessions are kept on disk until you remove them.
