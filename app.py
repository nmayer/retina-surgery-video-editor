#!/usr/bin/env python3
"""Retina surgery video editor — web UI for reviewing and trimming footage."""

import argparse
import subprocess
import json
import sys
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_file

from analyzer import analyze_video, extract_thumbnails, classify_thumbnails

app = Flask(__name__)

VIDEO_PATH = None
ANALYSIS = None
ANALYSIS_DIR = None
THUMB_INTERVAL = 30
THUMB_COUNT = 0
THUMB_CLASSES = {}  # per-thumbnail classification {str(n): "microscope"|"external"|"blank"}


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analysis")
def api_analysis():
    return jsonify({
        **ANALYSIS,
        "thumb_interval": THUMB_INTERVAL,
        "thumb_count": THUMB_COUNT,
        "thumb_classes": THUMB_CLASSES,
    })


@app.route("/api/thumb/<int:n>")
def api_thumb(n):
    path = ANALYSIS_DIR / "thumbs" / f"thumb_{n:04d}.jpg"
    if path.exists():
        return send_file(path, mimetype="image/jpeg")
    return "", 404


@app.route("/api/export", methods=["POST"])
def api_export():
    segments = request.json.get("segments", [])
    if not segments:
        return jsonify({"error": "No segments to export"}), 400

    export_dir = ANALYSIS_DIR / "export"
    export_dir.mkdir(exist_ok=True)

    # Cut each segment with stream copy (fast, no re-encode)
    seg_files = []
    for i, seg in enumerate(segments):
        seg_file = export_dir / f"seg_{i:04d}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(seg["start"]),
            "-to", str(seg["end"]),
            "-i", str(VIDEO_PATH),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-v", "quiet",
            str(seg_file),
        ], check=True)
        seg_files.append(seg_file)

    # Concat
    concat_file = export_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for sf in seg_files:
            f.write(f"file '{sf}'\n")

    stem = Path(VIDEO_PATH).stem
    output_path = ANALYSIS_DIR / f"{stem}_edited.mp4"

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        "-v", "quiet",
        str(output_path),
    ], check=True)

    # Cleanup temp files
    for sf in seg_files:
        sf.unlink()
    concat_file.unlink()

    total_dur = sum(s["end"] - s["start"] for s in segments)
    print(f"\nExported {len(segments)} segments ({total_dur:.0f}s) to {output_path}")

    return jsonify({
        "output_path": str(output_path),
        "duration": round(total_dur, 1),
    })


# ── Labeling routes ──────────────────────────────────────────────────────────

@app.route("/label")
def label():
    return render_template("label.html")


@app.route("/api/labels", methods=["GET"])
def api_labels_get():
    labels_file = ANALYSIS_DIR / "labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/api/labels", methods=["POST"])
def api_labels_post():
    labels = request.json
    labels_file = ANALYSIS_DIR / "labels.json"
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)
    return jsonify({"ok": True, "count": len(labels)})


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global VIDEO_PATH, ANALYSIS, ANALYSIS_DIR, THUMB_INTERVAL, THUMB_COUNT, THUMB_CLASSES

    parser = argparse.ArgumentParser(description="Retina surgery video editor")
    parser.add_argument("video", help="Path to the surgery video file")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between review thumbnails (default: 30)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run analysis even if cached")
    args = parser.parse_args()

    VIDEO_PATH = Path(args.video).resolve()
    if not VIDEO_PATH.exists():
        print(f"Error: {VIDEO_PATH} not found")
        sys.exit(1)

    THUMB_INTERVAL = args.interval
    ANALYSIS_DIR = Path("analysis") / VIDEO_PATH.stem
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if args.force:
        cache = ANALYSIS_DIR / "analysis.json"
        if cache.exists():
            cache.unlink()

    # Analyze
    ANALYSIS = analyze_video(VIDEO_PATH, ANALYSIS_DIR)

    # Extract thumbnails
    thumbs_dir = ANALYSIS_DIR / "thumbs"
    THUMB_COUNT = extract_thumbnails(
        VIDEO_PATH, thumbs_dir, ANALYSIS["duration"], THUMB_INTERVAL
    )

    # Per-thumbnail classification (fast, uses edge brightness)
    print("  Classifying thumbnails individually...")
    THUMB_CLASSES = classify_thumbnails(thumbs_dir, THUMB_COUNT)

    print(f"\nStarting editor at http://localhost:{args.port}")
    print(f"  Review UI: http://localhost:{args.port}/")
    print(f"  Labeling:  http://localhost:{args.port}/label")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
