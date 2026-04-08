"""
Web UI for Vidi1.5-9B MLX — upload video, enter query, see temporal grounding results.

Usage:
    python web_app.py [--model-path ./] [--port 7860]
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

UPLOAD_DIR = Path("/tmp/vidi_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global model references (loaded once at startup)
engine = None
preprocessor = None


def load_model_global(model_path: str):
    """Load Vidi model once at startup."""
    global engine, preprocessor

    print("Loading config ...")
    from mlx_vidi.run import load_config, load_model
    from mlx_vidi.preprocessing import VidiPreprocessor

    config = load_config(model_path)
    print("Building model ...")
    engine = load_model(model_path, config)
    print("Loading preprocessor ...")
    preprocessor = VidiPreprocessor(model_path)
    print("Model ready!")


def parse_timestamps(text: str, duration: float) -> list:
    """Parse model output like '0.21-0.22, 0.46-0.47' into absolute timestamps.

    Returns list of dicts: [{"start": 12.6, "end": 13.2, "raw": "0.21-0.22"}, ...]
    """
    segments = []
    # Match patterns like 0.21-0.22 or 0.210-0.220
    pattern = r"(\d+\.\d+)\s*-\s*(\d+\.\d+)"
    for m in re.finditer(pattern, text):
        start_norm = float(m.group(1))
        end_norm = float(m.group(2))
        segments.append({
            "start": round(start_norm * duration, 2),
            "end": round(end_norm * duration, 2),
            "start_norm": start_norm,
            "end_norm": end_norm,
            "raw": m.group(0),
        })
    return segments


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/upload", methods=["POST"])
def upload_video():
    """Handle video file upload."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save with timestamp to avoid collisions
    ext = Path(f.filename).suffix or ".mp4"
    saved_name = f"video_{int(time.time())}{ext}"
    save_path = UPLOAD_DIR / saved_name
    f.save(str(save_path))

    duration = get_video_duration(str(save_path))

    return jsonify({
        "filename": saved_name,
        "duration": duration,
        "size_mb": round(save_path.stat().st_size / 1024 / 1024, 1),
    })


@app.route("/infer", methods=["POST"])
def infer():
    """Run temporal grounding inference."""
    data = request.get_json()
    filename = data.get("filename")
    query = data.get("query", "").strip()
    fps = data.get("fps", 1.0)
    max_frames = data.get("max_frames", 128)

    if not filename or not query:
        return jsonify({"error": "Missing filename or query"}), 400

    video_path = str(UPLOAD_DIR / filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    duration = get_video_duration(video_path)

    # Wrap query in Vidi's expected format
    full_query = f"During which time segments in the video can we see {query}?"

    try:
        t0 = time.time()

        # Preprocess
        inputs = preprocessor.prepare_video(
            video_path, full_query, fps=fps, max_frames=max_frames
        )

        t_preprocess = time.time() - t0

        # Generate
        t1 = time.time()
        token_ids = engine.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            mel_features=inputs["mel_features"],
            audio_sizes=inputs.get("audio_sizes", []),
            max_tokens=512,
            temperature=0.0,
        )
        t_generate = time.time() - t1

        # Decode
        text = preprocessor.tokenizer.decode(token_ids, skip_special_tokens=True)
        segments = parse_timestamps(text, duration)

        return jsonify({
            "raw_output": text,
            "query": query,
            "full_query": full_query,
            "duration": duration,
            "segments": segments,
            "num_frames": inputs["num_frames"],
            "num_tokens": len(token_ids),
            "time_preprocess": round(t_preprocess, 2),
            "time_generate": round(t_generate, 2),
            "tokens_per_sec": round(len(token_ids) / t_generate, 1) if t_generate > 0 else 0,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=".", help="Path to MLX model directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    load_model_global(args.model_path)
    app.run(host=args.host, port=args.port, debug=False)
