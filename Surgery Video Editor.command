#!/bin/bash
cd "$(dirname "$0")"

# ── Check dependencies ───────────────────────────────────────────────────────

missing=0

# Python 3
if ! command -v python3 &>/dev/null; then
    echo "Python 3 not found."
    echo "macOS will prompt you to install it — click 'Install' when asked."
    echo ""
    xcode-select --install 2>/dev/null
    echo "After installation finishes, double-click this file again."
    read -p "Press Enter to close..."
    exit 1
fi

# ffmpeg & ffprobe
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    echo "ffmpeg not found — it's needed for video processing."
    echo ""
    if command -v brew &>/dev/null; then
        echo "Installing via Homebrew..."
        brew install ffmpeg
        if [ $? -ne 0 ]; then
            echo "Install failed. Please run 'brew install ffmpeg' manually."
            read -p "Press Enter to close..."
            exit 1
        fi
    else
        echo "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Homebrew on Apple Silicon installs to /opt/homebrew
        eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)"
        echo ""
        echo "Installing ffmpeg..."
        brew install ffmpeg
        if [ $? -ne 0 ]; then
            echo "Install failed. Please run 'brew install ffmpeg' manually."
            read -p "Press Enter to close..."
            exit 1
        fi
    fi
    echo ""
fi

# Python packages
if ! python3 -c "import flask" &>/dev/null; then
    echo "Installing Python packages..."
    python3 -m pip install -q -r requirements.txt
    echo ""
fi

# ── Launch ───────────────────────────────────────────────────────────────────

echo "Starting Surgery Video Editor..."
sleep 1 && open http://localhost:5555 &
python3 app.py
