# Retina Surgery Video Editor

A tool for quickly trimming retina surgery recordings. Automatically classifies footage as microscope, external, or blank, then presents a web UI for reviewing and exporting the interesting parts.

## Quick Start

Double-click **Surgery Video Editor.command**. It will install any missing dependencies (Python, ffmpeg, pip packages) on first run, then open your browser.

Or from the terminal:

```bash
pip install -r requirements.txt
python3 app.py
```

## How It Works

1. **Open** a video file or folder of video files from the landing page
2. **Auto-classification** analyzes each frame using edge brightness and pixel statistics to identify:
   - **Microscope** — intraocular view through the surgical scope (dark edges, bright center)
   - **External** — external view of the eye/operating field
   - **Blank** — camera capped, drape, idle, or test pattern
3. **Review** the results in the editor — arrow keys to navigate, space to toggle keep/cut
4. **Export** the selected segments as a single video file (fast stream copy, no re-encoding)

## Editor Controls

| Key | Action |
|-----|--------|
| `←` `→` | Navigate frames |
| `Shift` + arrows | Jump 10 frames |
| `Space` | Toggle keep/cut on current segment |
| `I` | Set in-point |
| `O` | Set out-point |
| `X` | Delete selection under cursor |
| `F` | Toggle between keep-only and all frames |

Click and drag on the timeline to create selections. Selections act as sub-selections — only the "keep" segments within your selection are exported.

## Multi-Video Support

Surgeries split across multiple files (common with ZEISS and other recording systems) are automatically stitched into a single timeline. Select a folder containing the clips and they'll be sorted by filename and concatenated.

## Requirements

- macOS (uses native file picker dialogs)
- Python 3
- ffmpeg
- Flask, NumPy, Pillow (installed automatically by the launcher)
