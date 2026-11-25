"""Record facial-expression scores from a webcam and analyze saved sessions."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECORDINGS = ROOT / "recordings"
EXPRESSIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
CSV_FIELDS = ("elapsed_seconds", "timestamp", "face_count", "dominant_expression",
              "boxes", *(f"{name}_score" for name in EXPRESSIONS))


def record(args: argparse.Namespace) -> None:
    # Import lazily so listing and analyzing saved recordings work without the camera stack.
    import cv2
    try:
        # FER 25.x stopped re-exporting the class from the package root.
        from fer.fer import FER
    except ImportError:
        # Keep compatibility with older FER releases.
        from fer import FER

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = RECORDINGS / session_id
    folder.mkdir(parents=True, exist_ok=False)
    csv_path = folder / "expressions.csv"
    detector = FER(mtcnn=False)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Check camera access and try again.")

    video = None
    started = time.monotonic()
    frame_count = 0
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            print("Recording. Press q in the camera window to stop.")
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera frame could not be read; stopping.")
                    break
                elapsed = time.monotonic() - started
                detections = detector.detect_emotions(frame)
                scores = {name: max((float(item["emotions"].get(name, 0.0)) for item in detections), default=0.0)
                          for name in EXPRESSIONS}
                dominant = max(scores, key=scores.get) if detections else "none"
                # FER may return boxes as NumPy arrays, which json.dumps cannot encode.
                boxes = [[int(value) for value in item["box"]] for item in detections]
                writer.writerow({
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "face_count": len(detections),
                    "dominant_expression": dominant,
                    "boxes": json.dumps(boxes),
                    **{f"{name}_score": f"{scores[name]:.5f}" for name in EXPRESSIONS},
                })
                frame_count += 1
                if not args.no_video:
                    if video is None:
                        height, width = frame.shape[:2]
                        video = cv2.VideoWriter(str(folder / "annotated.mp4"),
                                                cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
                    for item in detections:
                        x, y, w, h = item["box"]
                        label = max(item["emotions"], key=item["emotions"].get)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 220, 80), 2)
                        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (50, 220, 80), 2)
                    video.write(frame)
                    cv2.imshow("Expression recorder (q to stop)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                elif time.monotonic() - started >= args.seconds:
                    break
    finally:
        camera.release()
        if video is not None:
            video.release()
        cv2.destroyAllWindows()

    metadata = {"session_id": session_id, "name": args.name or session_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "duration_seconds": round(time.monotonic() - started, 2),
                "frames": frame_count, "camera": args.camera,
                "video_saved": not args.no_video, "data_file": "expressions.csv"}
    (folder / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {frame_count} frames to {folder}")


def sessions() -> list[Path]:
    return sorted((path for path in RECORDINGS.iterdir() if path.is_dir()), reverse=True) if RECORDINGS.exists() else []


def list_recordings(_: argparse.Namespace) -> None:
    found = sessions()
    if not found:
        print("No recordings yet. Start one with: python main.py record")
        return
    for folder in found:
        meta_path = folder / "session.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        print(f"{folder.name}  {meta.get('name', '')}  {meta.get('duration_seconds', '?')}s  {meta.get('frames', '?')} frames")


def analyze(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder = Path(args.session) if args.session else (sessions()[0] if sessions() else None)
    if folder is None:
        raise SystemExit("No saved sessions found. Record a session first.")
    if not folder.is_absolute():
        folder = RECORDINGS / folder
    source = folder / "expressions.csv" if folder.is_dir() else folder
    if not source.exists():
        raise SystemExit(f"Recording data not found: {source}")
    folder = source.parent
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("This recording has no frames to analyze.")

    scores = np.array([[float(row[f"{name}_score"]) for name in EXPRESSIONS] for row in rows])
    times = np.array([float(row["elapsed_seconds"]) for row in rows])
    duration = max(float(times[-1]), 0.01)
    bins = max(1, min(args.bins, len(rows)))
    edges = np.linspace(0, duration + 1e-6, bins + 1)
    heat = np.zeros((len(EXPRESSIONS), bins))
    counts = np.zeros(bins)
    indices = np.clip(np.digitize(times, edges) - 1, 0, bins - 1)
    for i, index in enumerate(indices):
        heat[:, index] += scores[i]
        counts[index] += 1
    heat /= np.maximum(counts, 1)

    output = folder / "analysis"
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, bins * 0.25), 4.5))
    image = ax.imshow(heat, aspect="auto", origin="lower", interpolation="nearest",
                      extent=[0, duration, -0.5, len(EXPRESSIONS) - 0.5], vmin=0, vmax=1, cmap="magma")
    ax.set_yticks(range(len(EXPRESSIONS)), EXPRESSIONS)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Facial expression intensity over time")
    fig.colorbar(image, ax=ax, label="Average model score")
    fig.tight_layout()
    fig.savefig(output / "expression_heatmap.png", dpi=160)
    plt.close(fig)

    summary = []
    for index, name in enumerate(EXPRESSIONS):
        summary.append({"expression": name, "mean_score": round(float(scores[:, index].mean()), 5),
                        "peak_score": round(float(scores[:, index].max()), 5),
                        "dominant_frames": sum(row["dominant_expression"] == name for row in rows)})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (output / "summary.json").write_text(json.dumps({"session": folder.name, "frames": len(rows),
        "duration_seconds": round(duration, 3), "expressions": summary}, indent=2), encoding="utf-8")
    print(f"Analysis saved to {output}")
    print("Created expression_heatmap.png, summary.csv, and summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record facial expression estimates and analyze sessions.")
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record", help="Capture and save a webcam session")
    rec.add_argument("--camera", type=int, default=0, help="Camera device number (default: 0)")
    rec.add_argument("--name", help="Optional session name")
    rec.add_argument("--no-video", action="store_true", help="Save expression data without annotated video")
    rec.add_argument("--seconds", type=int, default=60, help="Recording duration in data-only mode")
    rec.set_defaults(func=record)
    ls = commands.add_parser("list", help="List saved sessions")
    ls.set_defaults(func=list_recordings)
    report = commands.add_parser("analyze", help="Generate a heatmap and summary for a saved session")
    report.add_argument("session", nargs="?", help="Session folder name or path (default: newest session)")
    report.add_argument("--bins", type=int, default=40, help="Number of time intervals in the heatmap")
    report.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
