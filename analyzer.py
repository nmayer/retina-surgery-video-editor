"""Video analysis for retina surgery footage.

Classifies video frames as:
- microscope: intraocular view through the surgical microscope (bright circular
  field against dark surround — dark edges from the scope vignette)
- external: external view of the eye / operating field (lit across full frame)
- blank: uniform color, near-black, or test pattern (camera capped, drape, idle)
"""

import subprocess
import json
import shutil
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


def _classify_single_video(video_path, fps=0.5, absorb_threshold=10,
                           progress_cb=None):
    """Classify frames of a single video file.

    Returns dict with duration, width, height, and raw segments (no offset).
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    duration = float(info["format"]["duration"])

    # Downscale for analysis
    aw = 320
    ah = int(height * (aw / width))

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
    total_frames = int(duration * fps)

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(ah, aw)
        std_val = float(np.std(frame))

        edge_w = max(1, aw // 10)
        edge_h = max(1, ah // 10)
        left   = float(np.mean(frame[:, :edge_w]))
        right  = float(np.mean(frame[:, -edge_w:]))
        top    = float(np.mean(frame[:edge_h, :]))
        bottom = float(np.mean(frame[-edge_h:, :]))
        min_pair = min(min(left, right), min(top, bottom))

        bright = float(np.percentile(frame, 99))

        if min_pair < 10 and bright > 50:
            cls = "microscope"
        elif std_val < 25:
            cls = "blank"
        else:
            cls = "external"

        classifications.append(cls)
        idx += 1

        if progress_cb and idx % 50 == 0:
            progress_cb(idx / max(total_frames, 1))

    proc.wait()

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

    # Absorb tiny segments into their neighbors
    merged = []
    for seg in segments:
        if merged and (seg["end"] - seg["start"]) < absorb_threshold:
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

    return {
        "duration": round(duration, 1),
        "width": width,
        "height": height,
        "segments": final,
    }


def analyze_videos(video_paths, analysis_dir, fps=0.5, absorb_threshold=10,
                   progress_cb=None):
    """Analyze one or more video files as a stitched timeline.

    Args:
        video_paths: List of Path objects to video files.
        analysis_dir: Directory to cache results.
        fps: Sampling rate for analysis.
        absorb_threshold: Absorb segments shorter than this (seconds).
        progress_cb: Optional callback(fraction, message) for progress updates.

    Returns:
        dict with video metadata and classified segments using global timestamps.
    """
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_file = analysis_dir / "analysis.json"
    if analysis_file.exists():
        with open(analysis_file) as f:
            cached = json.load(f)

        # Migrate old single-video format: add videos array if missing
        if "videos" not in cached and "video_path" in cached:
            old_path = cached["video_path"]
            if Path(old_path).exists():
                cached["videos"] = [{
                    "path": str(Path(old_path).resolve()),
                    "duration": cached["duration"],
                    "offset": 0,
                }]
                cached["video_path"] = Path(old_path).stem
                # Re-save with new format
                with open(analysis_file, "w") as f:
                    json.dump(cached, f, indent=2)

        # Validate cache: check that video paths match
        cached_paths = [v["path"] for v in cached.get("videos", [])]
        current_paths = [str(Path(p).resolve()) for p in video_paths]
        if cached_paths == current_paths:
            print(f"Loading cached analysis from {analysis_file}")
            if progress_cb:
                progress_cb(1.0, "Loaded from cache")
            return cached

    total_files = len(video_paths)
    videos = []
    all_segments = []
    offset = 0.0
    ref_width = ref_height = None

    for file_idx, vp in enumerate(video_paths):
        vp = Path(vp)
        name = vp.name
        if progress_cb:
            progress_cb(file_idx / total_files,
                        f"Analyzing {name} ({file_idx + 1}/{total_files})")

        print(f"\nAnalyzing {name} ({file_idx + 1}/{total_files})...")

        def file_progress(frac, _idx=file_idx, _name=name):
            if progress_cb:
                progress_cb(frac, f"Analyzing {_name} ({_idx + 1}/{total_files})")

        result = _classify_single_video(vp, fps=fps,
                                        absorb_threshold=absorb_threshold,
                                        progress_cb=file_progress)

        # Validate resolution consistency
        if ref_width is None:
            ref_width = result["width"]
            ref_height = result["height"]
        elif result["width"] != ref_width or result["height"] != ref_height:
            raise ValueError(
                f"{name} is {result['width']}x{result['height']} but "
                f"expected {ref_width}x{ref_height}. "
                f"All videos must have the same resolution."
            )

        videos.append({
            "path": str(vp.resolve()),
            "duration": result["duration"],
            "offset": round(offset, 1),
        })

        # Offset segments to global timeline
        for seg in result["segments"]:
            all_segments.append({
                "start": round(seg["start"] + offset, 1),
                "end": round(seg["end"] + offset, 1),
                "classification": seg["classification"],
            })

        offset += result["duration"]

    # Merge adjacent same-type segments across file boundaries
    final = [all_segments[0]] if all_segments else []
    for seg in all_segments[1:]:
        if seg["classification"] == final[-1]["classification"]:
            final[-1]["end"] = seg["end"]
        else:
            final.append(seg)

    total_duration = round(offset, 1)

    # Display name: common parent folder if multi-video, else file stem
    if len(video_paths) > 1:
        parents = set(Path(p).resolve().parent for p in video_paths)
        if len(parents) == 1:
            display_name = next(iter(parents)).name
        else:
            display_name = Path(video_paths[0]).stem
    else:
        display_name = Path(video_paths[0]).stem

    result = {
        "videos": videos,
        "video_path": display_name,
        "duration": total_duration,
        "width": ref_width,
        "height": ref_height,
        "segments": final,
    }

    with open(analysis_file, "w") as f:
        json.dump(result, f, indent=2)

    _print_summary(result)

    if progress_cb:
        progress_cb(1.0, "Analysis complete")

    return result


def extract_thumbnails_multi(videos, thumbs_dir, interval=30, progress_cb=None):
    """Extract thumbnails from multiple videos with global numbering.

    Args:
        videos: List of {"path", "duration", "offset"} dicts.
        thumbs_dir: Directory to write thumbnails.
        interval: Seconds between thumbnails.
        progress_cb: Optional callback(file_idx) called after each file.

    Returns:
        Total thumbnail count.
    """
    thumbs_dir = Path(thumbs_dir)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Calculate expected total
    expected = sum(int(v["duration"] / interval) + 1 for v in videos)
    existing = len(list(thumbs_dir.glob("*.jpg")))
    if existing >= expected - 1:
        print(f"  Thumbnails already extracted ({existing} files)")
        return existing

    # Clear any partial extraction
    for f in thumbs_dir.glob("*.jpg"):
        f.unlink()

    global_offset = 0
    total = 0

    for vid_idx, vid in enumerate(videos):
        vp = vid["path"]
        dur = vid["duration"]
        file_count = int(dur / interval) + 1

        print(f"  Extracting ~{file_count} thumbnails from {Path(vp).name}...")

        # Extract to temp dir
        tmp_dir = thumbs_dir / "_tmp"
        tmp_dir.mkdir(exist_ok=True)

        cmd = [
            "ffmpeg", "-i", str(vp),
            "-vf", f"fps=1/{interval},scale=640:-1",
            "-q:v", "3", "-v", "quiet",
            str(tmp_dir / "thumb_%04d.jpg")
        ]
        subprocess.run(cmd, check=True)

        # Rename with global offset
        extracted = sorted(tmp_dir.glob("*.jpg"))
        for i, src in enumerate(extracted):
            global_n = global_offset + i + 1
            dst = thumbs_dir / f"thumb_{global_n:04d}.jpg"
            src.rename(dst)
            total += 1

        global_offset += len(extracted)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if progress_cb:
            progress_cb(vid_idx + 1)

    print(f"  Done: {total} thumbnails extracted")
    return total


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

        bright = float(np.percentile(gray, 99))

        if sat > 80:
            cls = "blank"
        elif min_pair < 10 and bright > 50:
            cls = "microscope"
        elif std_val < 25:
            cls = "blank"
        else:
            cls = "external"

        results[str(i)] = cls

    print(f"  Classified {len(results)} thumbnails "
          f"(scope: {sum(1 for v in results.values() if v == 'microscope')}, "
          f"ext: {sum(1 for v in results.values() if v == 'external')}, "
          f"blank: {sum(1 for v in results.values() if v == 'blank')})")
    return results


def _print_summary(analysis):
    dur = analysis["duration"]
    by_type = {}
    for seg in analysis["segments"]:
        t = seg["classification"]
        by_type[t] = by_type.get(t, 0) + (seg["end"] - seg["start"])

    name = analysis.get("video_path", "video")
    n_files = len(analysis.get("videos", []))
    file_info = f" ({n_files} files)" if n_files > 1 else ""

    print(f"\nSummary for {name}{file_info}:")
    print(f"  Duration:   {_fmt(dur)}")
    for t in ["microscope", "external", "blank"]:
        s = by_type.get(t, 0)
        print(f"  {t:11s}: {_fmt(s)} ({s/dur*100:.0f}%)")
    print(f"  Segments:   {len(analysis['segments'])}")


def _fmt(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
