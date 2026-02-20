#!/usr/bin/env python3
"""Retina surgery video editor — web UI for reviewing and trimming footage."""

import argparse
import subprocess
import json
import sys
import threading
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_file, redirect

from analyzer import (
    analyze_videos, extract_thumbnails_multi, classify_thumbnails, get_video_info
)

app = Flask(__name__)

# ── State ────────────────────────────────────────────────────────────────────

STATE = {
    "status": "idle",       # idle | analyzing | ready | error
    "progress": 0.0,        # 0.0–1.0
    "phase": "",            # current pipeline phase label
    "message": "",          # human-readable progress detail
    "error": None,          # error message if status == "error"
    "video_paths": [],      # list of resolved video file paths
    "analysis": None,       # analysis result dict
    "analysis_dir": None,   # Path
    "thumb_interval": 30,
    "thumb_count": 0,
    "thumb_classes": {},     # {str(n): "microscope"|"external"|"blank"}
}

# CLI-configurable defaults
DEFAULTS = {
    "fps": 0.5,
    "absorb": 10,
    "interval": 30,
    "port": 5555,
}

VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv", ".MKV"}


# ── macOS native file pickers ────────────────────────────────────────────────

def pick_files():
    """Open native macOS file picker for video files. Returns list of paths."""
    script = '''
    set theFiles to choose file of type {"public.movie"} with multiple selections allowed
    set output to ""
    repeat with f in theFiles
        set output to output & POSIX path of f & linefeed
    end repeat
    return output
    '''
    result = subprocess.run(["osascript", "-e", script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [p for p in result.stdout.strip().split("\n") if p]


def pick_folder():
    """Open native macOS folder picker. Returns path or None."""
    script = 'return POSIX path of (choose folder)'
    result = subprocess.run(["osascript", "-e", script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def pick_save_location(default_name="edited.mp4"):
    """Open native macOS save dialog. Returns path or None."""
    script = f'''
    set savePath to choose file name with prompt "Export video as:" default name "{default_name}"
    return POSIX path of savePath
    '''
    result = subprocess.run(["osascript", "-e", script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    path = result.stdout.strip()
    if not path.endswith(".mp4"):
        path += ".mp4"
    return path


def find_videos_in_folder(folder_path):
    """Find all video files in a folder, sorted by name."""
    folder = Path(folder_path)
    videos = sorted(
        [f for f in folder.iterdir()
         if f.is_file() and f.suffix in VIDEO_EXTENSIONS],
        key=lambda f: f.name
    )
    return [str(f) for f in videos]


# ── Analysis helpers ─────────────────────────────────────────────────────────

def analysis_dir_for(video_paths):
    """Determine analysis directory name from video paths."""
    paths = [Path(p).resolve() for p in video_paths]
    if len(paths) == 1:
        return Path("analysis") / paths[0].stem
    else:
        parents = set(p.parent for p in paths)
        if len(parents) == 1:
            return Path("analysis") / next(iter(parents)).name
        else:
            return Path("analysis") / paths[0].stem


def run_analysis_thread(video_paths, interval, fps, absorb):
    """Run analysis in background thread, updating STATE as it goes."""
    try:
        STATE["status"] = "analyzing"
        STATE["progress"] = 0.0
        STATE["error"] = None

        resolved = [str(Path(p).resolve()) for p in video_paths]
        STATE["video_paths"] = resolved

        a_dir = analysis_dir_for(resolved)
        STATE["analysis_dir"] = a_dir
        STATE["thumb_interval"] = interval

        def progress_cb(frac, msg=""):
            STATE["progress"] = frac
            if msg:
                STATE["message"] = msg

        # Phase 1: Analyze
        STATE["phase"] = "Classifying frames"
        analysis = analyze_videos(resolved, a_dir, fps=fps,
                                  absorb_threshold=absorb,
                                  progress_cb=progress_cb)
        STATE["analysis"] = analysis

        # Phase 2: Extract thumbnails
        thumbs_dir = a_dir / "thumbs"
        videos = analysis["videos"]
        n_videos = len(videos)

        def thumb_progress(file_idx):
            STATE["progress"] = file_idx / n_videos
            STATE["message"] = f"File {file_idx}/{n_videos}"

        STATE["phase"] = "Extracting thumbnails"
        STATE["progress"] = 0.0
        STATE["message"] = ""
        thumb_count = extract_thumbnails_multi(videos, thumbs_dir, interval,
                                               progress_cb=thumb_progress)
        STATE["thumb_count"] = thumb_count

        # Phase 3: Classify thumbnails
        STATE["phase"] = "Classifying thumbnails"
        STATE["progress"] = 0.0
        STATE["message"] = ""
        STATE["thumb_classes"] = classify_thumbnails(thumbs_dir, thumb_count)

        STATE["status"] = "ready"
        STATE["progress"] = 1.0
        STATE["phase"] = ""
        STATE["message"] = "Ready"
        print(f"\nReady — open the editor in your browser.")

    except Exception as e:
        STATE["status"] = "error"
        STATE["error"] = str(e)
        STATE["message"] = f"Error: {e}"
        print(f"\nAnalysis error: {e}")
        import traceback
        traceback.print_exc()


def resolve_segments_to_sources(segments, videos):
    """Map global-time segments to per-source-file cuts.

    Returns list of {"path": str, "start": float, "end": float} dicts,
    where start/end are local times within that file.
    """
    cuts = []
    for seg in segments:
        for vid in videos:
            vid_start = vid["offset"]
            vid_end = vid["offset"] + vid["duration"]

            overlap_start = max(seg["start"], vid_start)
            overlap_end = min(seg["end"], vid_end)

            if overlap_start < overlap_end:
                cuts.append({
                    "path": vid["path"],
                    "start": round(overlap_start - vid["offset"], 2),
                    "end": round(overlap_end - vid["offset"], 2),
                })
    return cuts


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if STATE["status"] == "ready":
        return render_template("index.html")
    return render_template("open.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": STATE["status"],
        "progress": round(STATE["progress"], 3),
        "phase": STATE["phase"],
        "message": STATE["message"],
        "error": STATE["error"],
    })


@app.route("/api/pick", methods=["POST"])
def api_pick():
    mode = request.json.get("mode", "file")
    if mode == "folder":
        folder = pick_folder()
        if not folder:
            return jsonify({"paths": [], "cancelled": True})
        videos = find_videos_in_folder(folder)
        if not videos:
            return jsonify({"paths": [], "error": "No video files found in folder"})
        file_info = [{"path": vp, "name": Path(vp).name} for vp in videos]
        return jsonify({"paths": videos, "files": file_info})
    else:
        paths = pick_files()
        if not paths:
            return jsonify({"paths": [], "cancelled": True})
        paths.sort()
        file_info = [{"path": vp, "name": Path(vp).name} for vp in paths]
        return jsonify({"paths": paths, "files": file_info})


@app.route("/api/open", methods=["POST"])
def api_open():
    paths = request.json.get("paths", [])
    if not paths:
        return jsonify({"error": "No files specified"}), 400

    if STATE["status"] == "analyzing":
        return jsonify({"error": "Analysis already in progress"}), 409

    # Reset state
    STATE["status"] = "idle"
    STATE["analysis"] = None
    STATE["thumb_count"] = 0
    STATE["thumb_classes"] = {}

    interval = DEFAULTS["interval"]
    fps = DEFAULTS["fps"]
    absorb = DEFAULTS["absorb"]

    thread = threading.Thread(
        target=run_analysis_thread,
        args=(paths, interval, fps, absorb),
        daemon=True,
    )
    thread.start()

    return jsonify({"ok": True, "file_count": len(paths)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    STATE["status"] = "idle"
    STATE["progress"] = 0.0
    STATE["message"] = ""
    STATE["error"] = None
    STATE["analysis"] = None
    STATE["thumb_count"] = 0
    STATE["thumb_classes"] = {}
    return jsonify({"ok": True})


@app.route("/api/pick-save", methods=["POST"])
def api_pick_save():
    default_name = request.json.get("default_name", "edited.mp4")
    path = pick_save_location(default_name)
    if not path:
        return jsonify({"cancelled": True})
    return jsonify({"path": path})


@app.route("/api/analysis")
def api_analysis():
    if STATE["analysis"] is None:
        return jsonify({"error": "No analysis available"}), 404
    return jsonify({
        **STATE["analysis"],
        "thumb_interval": STATE["thumb_interval"],
        "thumb_count": STATE["thumb_count"],
        "thumb_classes": STATE["thumb_classes"],
    })


@app.route("/api/thumb/<int:n>")
def api_thumb(n):
    if STATE["analysis_dir"] is None:
        return "", 404
    path = STATE["analysis_dir"] / "thumbs" / f"thumb_{n:04d}.jpg"
    if path.exists():
        return send_file(path, mimetype="image/jpeg")
    return "", 404


@app.route("/api/export", methods=["POST"])
def api_export():
    segments = request.json.get("segments", [])
    if not segments:
        return jsonify({"error": "No segments to export"}), 400

    analysis = STATE["analysis"]
    if analysis is None:
        return jsonify({"error": "No analysis loaded"}), 400

    try:
        export_dir = STATE["analysis_dir"] / "export"
        export_dir.mkdir(exist_ok=True)

        videos = analysis["videos"]
        cuts = resolve_segments_to_sources(segments, videos)

        if not cuts:
            return jsonify({"error": "No valid cuts resolved"}), 400

        # Cut each segment with stream copy
        seg_files = []
        for i, cut in enumerate(cuts):
            seg_file = export_dir / f"seg_{i:04d}.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(cut["start"]),
                "-to", str(cut["end"]),
                "-i", str(cut["path"]),
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
                f.write(f"file '{sf.resolve()}'\n")

        display_name = analysis.get("video_path", "video")
        output_path = request.json.get("output_path")
        if output_path:
            output_path = Path(output_path)
        else:
            output_path = STATE["analysis_dir"] / f"{display_name}_edited.mp4"

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
        print(f"\nExported {len(cuts)} cuts ({total_dur:.0f}s) to {output_path}")

        return jsonify({
            "output_path": str(output_path),
            "duration": round(total_dur, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Labeling routes ──────────────────────────────────────────────────────────

@app.route("/label")
def label():
    return render_template("label.html")


@app.route("/api/labels", methods=["GET"])
def api_labels_get():
    if STATE["analysis_dir"] is None:
        return jsonify({})
    labels_file = STATE["analysis_dir"] / "labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/api/labels", methods=["POST"])
def api_labels_post():
    if STATE["analysis_dir"] is None:
        return jsonify({"error": "No analysis loaded"}), 400
    labels = request.json
    labels_file = STATE["analysis_dir"] / "labels.json"
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)
    return jsonify({"ok": True, "count": len(labels)})


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retina surgery video editor")
    parser.add_argument("video", nargs="?", default=None,
                        help="Path to video file (optional — can pick from UI)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between review thumbnails (default: 30)")
    parser.add_argument("--fps", type=float, default=0.5,
                        help="Analysis sampling rate in frames/sec (default: 0.5)")
    parser.add_argument("--absorb", type=float, default=10,
                        help="Absorb segments shorter than N seconds (default: 10)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run analysis even if cached")
    args = parser.parse_args()

    DEFAULTS["fps"] = args.fps
    DEFAULTS["absorb"] = args.absorb
    DEFAULTS["interval"] = args.interval

    if args.video:
        # CLI mode: start analysis immediately
        video_path = Path(args.video).resolve()
        if not video_path.exists():
            print(f"Error: {video_path} not found")
            sys.exit(1)

        if args.force:
            a_dir = analysis_dir_for([str(video_path)])
            cache = a_dir / "analysis.json"
            if cache.exists():
                cache.unlink()

        # Run analysis in background thread so server starts immediately
        thread = threading.Thread(
            target=run_analysis_thread,
            args=([str(video_path)], args.interval, args.fps, args.absorb),
            daemon=True,
        )
        thread.start()

    print(f"\nStarting editor at http://localhost:{args.port}")
    if args.video:
        print(f"  Analysis running in background...")
    else:
        print(f"  Open http://localhost:{args.port} to select a video")
    app.run(host="127.0.0.1", port=args.port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
