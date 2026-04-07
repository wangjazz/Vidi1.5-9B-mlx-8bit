"""
Preprocessing for Vidi1.5-9B MLX port.

Handles:
- Video frames extraction (decord, 1 FPS)
- Image preprocessing (CLIP processor)
- Audio extraction (ffmpeg → mel spectrogram via WhisperFeatureExtractor)
- Text tokenization (Gemma2 tokenizer with chat template)

Reuses HuggingFace transformers processors, converting to mx.array at the end.
"""

import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_video_frames(
    video_path: str,
    fps: float = 1.0,
    max_frames: int = 128,
) -> np.ndarray:
    """Extract frames from video at specified FPS.

    Uses decord if available, falls back to OpenCV.

    Returns:
        np.ndarray of shape (N, H, W, 3) in uint8 RGB.
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        return _extract_frames_decord(video_path, fps, max_frames)
    except ImportError:
        pass

    try:
        import cv2
        return _extract_frames_cv2(video_path, fps, max_frames)
    except ImportError:
        raise ImportError("Either decord or opencv-python is required for video processing")


def _extract_frames_decord(video_path, fps, max_frames):
    import decord
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    duration = total_frames / video_fps
    num_frames = min(int(duration * fps), max_frames)
    if num_frames == 0:
        num_frames = 1
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return frames


def _extract_frames_cv2(video_path, fps, max_frames):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    num_frames = min(max(int(duration * fps), 1), max_frames)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from {video_path}")
    return np.stack(frames)


def extract_audio(video_path: str, sr: int = 16000) -> np.ndarray:
    """Extract audio from video using ffmpeg → raw PCM float32.

    Returns:
        np.ndarray of shape (num_samples,) float32 at 16kHz.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sr), "-ac", "1",
            tmp_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # Read PCM s16le via wave
        import wave
        with wave.open(tmp_path, "rb") as wf:
            nframes = wf.getnframes()
            audio_bytes = wf.readframes(nframes)

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    except subprocess.CalledProcessError:
        # No audio track
        return np.zeros(0, dtype=np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Image / audio processing via HuggingFace processors
# ---------------------------------------------------------------------------

def process_images(
    frames: np.ndarray,
    image_processor,
    image_size: int = 384,
) -> mx.array:
    """Process video frames using SiglipImageProcessor.

    Args:
        frames: (N, H, W, 3) uint8 RGB
        image_processor: SiglipImageProcessor instance

    Returns:
        mx.array of shape (N, image_size, image_size, 3) float32 NHWC.
    """
    from PIL import Image

    pil_images = [Image.fromarray(f) for f in frames]
    processed = image_processor(pil_images, return_tensors="np")
    pixel_values = processed["pixel_values"]  # (N, 3, H, W) NCHW

    # Convert NCHW → NHWC for MLX
    pixel_values = np.transpose(pixel_values, (0, 2, 3, 1))
    return mx.array(pixel_values)


def process_audio(
    audio: np.ndarray,
    feature_extractor,
    chunk_length_s: float = 30.0,
    sr: int = 16000,
) -> Tuple[mx.array, List[int]]:
    """Process audio using WhisperFeatureExtractor.

    Splits audio into 30s chunks, extracts mel spectrogram for each.

    Args:
        audio: (num_samples,) float32 at 16kHz.
        feature_extractor: WhisperFeatureExtractor instance.
        chunk_length_s: Chunk length in seconds.

    Returns:
        (mel_features, audio_sizes):
            mel_features: mx.array (N_chunks, T, n_mels) NLC format.
            audio_sizes: list of actual frame counts per chunk.
    """
    if len(audio) == 0:
        return None, []

    chunk_length = int(chunk_length_s * sr)
    nb_max_frames = feature_extractor.nb_max_frames  # 3000

    chunks = []
    sizes = []
    for start in range(0, len(audio), chunk_length):
        chunk = audio[start: start + chunk_length]
        actual_duration = len(chunk) / sr
        # WhisperFeatureExtractor returns (1, n_mels, T) or (n_mels, T)
        features = feature_extractor(
            chunk, sampling_rate=sr, return_tensors="np"
        )
        mel = features["input_features"]  # (1, n_mels, T)
        if mel.ndim == 3:
            mel = mel[0]  # (n_mels, T)

        mel = mel.T  # (T, n_mels) — NLC format for MLX Conv1d
        chunks.append(mel)

        # Actual number of frames for this chunk
        actual_frames = min(
            int(actual_duration / chunk_length_s * nb_max_frames),
            nb_max_frames,
        )
        sizes.append(actual_frames)

    mel_features = np.stack(chunks, axis=0)  # (N_chunks, T, n_mels)
    return mx.array(mel_features), sizes


# ---------------------------------------------------------------------------
# Text tokenization
# ---------------------------------------------------------------------------

def tokenize_chat(
    query: str,
    tokenizer,
    system_prompt: str = "You are a helpful assistant.",
) -> mx.array:
    """Tokenize a chat query using Gemma2's chat template.

    Returns:
        mx.array of shape (1, L) token IDs.
    """
    messages = [
        {"role": "user", "content": query},
    ]

    # Try using chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple formatting
        text = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"

    input_ids = tokenizer.encode(text, return_tensors="np")
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids.input_ids
    if isinstance(input_ids, list):
        input_ids = np.array([input_ids])
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    return mx.array(input_ids)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

class VidiPreprocessor:
    """Complete preprocessing pipeline for Vidi1.5-9B."""

    def __init__(self, model_path: str):
        """Initialize preprocessor with HuggingFace processors.

        Args:
            model_path: Path to model directory (with tokenizer files).
        """
        from transformers import AutoTokenizer, AutoImageProcessor, WhisperFeatureExtractor

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "right"

        # SigLip2 image processor (use_fast=False to support numpy output)
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_path, use_fast=False
            )
        except Exception:
            self.image_processor = AutoImageProcessor.from_pretrained(
                "google/siglip2-so400m-patch14-384", use_fast=False
            )

        # Whisper feature extractor
        try:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
        except Exception:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-large-v3"
            )

    def prepare_video(
        self,
        video_path: str,
        query: str,
        fps: float = 1.0,
        max_frames: int = 128,
    ) -> dict:
        """Prepare complete inputs for video understanding.

        Returns dict with:
            input_ids: (1, L) token IDs
            pixel_values: (N, H, W, C) image tensors
            mel_features: (N_chunks, T, n_mels) or None
            audio_sizes: list of actual audio frame counts
        """
        # Extract frames
        frames = extract_video_frames(video_path, fps=fps, max_frames=max_frames)

        # Process images
        pixel_values = process_images(frames, self.image_processor)

        # Extract and process audio
        audio = extract_audio(video_path)
        mel_features, audio_sizes = process_audio(audio, self.feature_extractor)

        # Tokenize
        input_ids = tokenize_chat(query, self.tokenizer)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mel_features": mel_features,
            "audio_sizes": audio_sizes,
            "num_frames": len(frames),
        }

    def prepare_image(
        self,
        image_path: str,
        query: str,
    ) -> dict:
        """Prepare inputs for single image understanding."""
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        frame = np.array(img)  # (H, W, 3)

        pixel_values = process_images(frame[None], self.image_processor)

        input_ids = tokenize_chat(query, self.tokenizer)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "mel_features": None,
            "audio_sizes": [],
            "num_frames": 1,
        }
