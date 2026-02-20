"""Video analysis for retina surgery footage.

Classifies video frames as:
- microscope: intraocular view through the surgical microscope (bright circular
  field against dark surround — dark edges from the scope vignette)
- external: external view of the eye / operating field (lit across full frame)
- blank: uniform color, near-black, or test pattern (camera capped, drape, idle)
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image


def get_video_info(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(video_path)],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)


def analyze_video(video_path, analysis_dir, fps=0.5):
    """Classify frames of a surgery video as microscope/external/blank.

    Streams downscaled grayscale frames from ffmpeg and classifies based on
    pixel statistics. Results are smoothed and merged into segments.

    Args:
        video_path: Path to the video file.
        analysis_dir: Directory to cache results.
        fps: Sampling rate for analysis (default 0.5 = one frame per 2 seconds).

    Returns:
        dict with video metadata and classified segments.
    """
    video_path = Path(video_path)
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_file = analysis_dir / "analysis.json"
    if analysis_file.exists():
        print(f"Loading cached analysis from {analysis_file}")
        with open(analysis_file) as f:
            return json.load(f)

    info = get_video_info(video_path)
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    duration = float(info["format"]["duration"])

    # Downscale for analysis
    aw = 320
    ah = int(height * (aw / width))

    print(f"Analyzing {duration/60:.1f} min video at {fps} fps "
          f"(~{int(duration * fps)} frames)...")

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps},scale={aw}:{ah}",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-v", "quiet", "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = aw * ah

    classifications = []
    idx = 0

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(ah, aw)
        std_val = float(np.std(frame))

        # Edge brightness: darkest edge of any opposing pair.
        # The scope circle always leaves at least one edge dark, even
        # when it drifts to the side of the frame.
        edge_w = max(1, aw // 10)
        edge_h = max(1, ah // 10)
        left   = float(np.mean(frame[:, :edge_w]))
        right  = float(np.mean(frame[:, -edge_w:]))
        top    = float(np.mean(frame[:edge_h, :]))
        bottom = float(np.mean(frame[-edge_h:, :]))
        min_pair = min(min(left, right), min(top, bottom))

        if std_val < 25:
            cls = "blank"
        elif min_pair < 10:
            cls = "microscope"
        else:
            cls = "external"

        classifications.append(cls)
        idx += 1
        if idx % 100 == 0:
            t = idx / fps
            print(f"\r  Classifying: {t/duration*100:.1f}% "
                  f"({t:.0f}s / {duration:.0f}s)", end="", flush=True)

    proc.wait()
    print(f"\r  Classifying: 100% ({idx} frames analyzed)        ")

    # Smooth with rolling mode (window = 5 frames = 10 sec at 0.5fps)
    window = 5
    smoothed = []
    for i in range(len(classifications)):
        lo = max(0, i - window // 2)
        hi = min(len(classifications), i + window // 2 + 1)
        mode = Counter(classifications[lo:hi]).most_common(1)[0][0]
        smoothed.append(mode)

    # Build segments
    segments = []
    if smoothed:
        seg_cls = smoothed[0]
        seg_start = 0.0

        for i in range(1, len(smoothed)):
            if smoothed[i] != seg_cls:
                segments.append({
                    "start": round(seg_start, 1),
                    "end": round(i / fps, 1),
                    "classification": seg_cls,
                })
                seg_cls = smoothed[i]
                seg_start = i / fps

        segments.append({
            "start": round(seg_start, 1),
            "end": round(duration, 1),
            "classification": seg_cls,
        })

    # Absorb tiny segments (< 10s) into their neighbors
    merged = []
    for seg in segments:
        if merged and (seg["end"] - seg["start"]) < 10:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    # Re-merge adjacent same-type segments
    final = [merged[0]] if merged else []
    for seg in merged[1:]:
        if seg["classification"] == final[-1]["classification"]:
            final[-1]["end"] = seg["end"]
        else:
            final.append(seg)

    result = {
        "video_path": str(video_path),
        "duration": round(duration, 1),
        "width": width,
        "height": height,
        "segments": final,
    }

    with open(analysis_file, "w") as f:
        json.dump(result, f, indent=2)

    _print_summary(result)
    return result


def extract_thumbnails(video_path, thumbs_dir, duration, interval=30):
    """Extract one thumbnail per `interval` seconds from the video."""
    thumbs_dir = Path(thumbs_dir)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    expected = int(duration / interval) + 1
    existing = len(list(thumbs_dir.glob("*.jpg")))
    if existing >= expected - 1:
        print(f"  Thumbnails already extracted ({existing} files)")
        return existing

    print(f"  Extracting ~{expected} thumbnails (1 every {interval}s)...")

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval},scale=640:-1",
        "-q:v", "3", "-v", "quiet",
        str(thumbs_dir / "thumb_%04d.jpg")
    ]
    subprocess.run(cmd, check=True)

    count = len(list(thumbs_dir.glob("*.jpg")))
    print(f"  Done: {count} thumbnails extracted")
    return count


def _print_summary(analysis):
    dur = analysis["duration"]
    by_type = {}
    for seg in analysis["segments"]:
        t = seg["classification"]
        by_type[t] = by_type.get(t, 0) + (seg["end"] - seg["start"])

    print(f"\nSummary for {Path(analysis['video_path']).name}:")
    print(f"  Duration:   {_fmt(dur)}")
    for t in ["microscope", "external", "blank"]:
        s = by_type.get(t, 0)
        print(f"  {t:11s}: {_fmt(s)} ({s/dur*100:.0f}%)")
    print(f"  Segments:   {len(analysis['segments'])}")


def classify_thumbnails(thumbs_dir, count):
    """Classify each thumbnail independently using edge brightness.

    This gives per-frame accuracy that doesn't suffer from the segment
    smoothing/merging that the video-level analysis uses.
    """
    thumbs_dir = Path(thumbs_dir)
    results = {}

    for i in range(1, count + 1):
        path = thumbs_dir / f"thumb_{i:04d}.jpg"
        if not path.exists():
            continue

        img = np.array(Image.open(path))
        gray = np.mean(img, axis=2)
        h, w = gray.shape

        std_val = float(np.std(gray))

        # Edge brightness: darkest edge of any opposing pair
        ew = max(1, w // 10)
        eh = max(1, h // 10)
        left   = float(np.mean(gray[:, :ew]))
        right  = float(np.mean(gray[:, -ew:]))
        top    = float(np.mean(gray[:eh, :]))
        bottom = float(np.mean(gray[-eh:, :]))
        min_pair = min(min(left, right), min(top, bottom))

        # Color saturation: detect SMPTE color bars / test patterns
        sat = float(np.mean(
            np.max(img, axis=2).astype(float) - np.min(img, axis=2).astype(float)
        ))

        if sat > 80:
            cls = "blank"
        elif std_val < 25:
            cls = "blank"
        elif min_pair < 10:
            cls = "microscope"
        else:
            cls = "external"

        results[str(i)] = cls

    print(f"  Classified {len(results)} thumbnails "
          f"(scope: {sum(1 for v in results.values() if v == 'microscope')}, "
          f"ext: {sum(1 for v in results.values() if v == 'external')}, "
          f"blank: {sum(1 for v in results.values() if v == 'blank')})")
    return results


def _fmt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
