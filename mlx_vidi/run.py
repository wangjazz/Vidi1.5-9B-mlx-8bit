"""
CLI entry point for Vidi1.5-9B MLX inference.

Usage:
    python -m mlx_vidi.run \
        --model-path ./weights/Vidi1.5-9B-mlx \
        --video-path ./test.mp4 \
        --query "When does the person wave their hand?"

    python -m mlx_vidi.run \
        --model-path ./weights/Vidi1.5-9B-mlx \
        --image-path ./test.jpg \
        --query "Describe this image."
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .generate import VidiEngine


def load_config(model_path: str) -> ModelConfig:
    """Load config from model directory."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")

    with open(config_path) as f:
        raw = json.load(f)

    # Map HuggingFace config keys to our ModelConfig
    params = {}

    # Gemma2 LLM params
    for key in [
        "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "head_dim", "num_key_value_heads",
        "rms_norm_eps", "vocab_size", "rope_theta", "rope_traditional",
        "attn_logit_softcapping", "final_logit_softcapping",
        "query_pre_attn_scalar", "sliding_window",
    ]:
        if key in raw:
            params[key] = raw[key]

    # Multimodal params
    for key in [
        "mm_vision_tower", "mm_vision_select_layer",
        "mm_image_pool_size", "mm_image_aspect_ratio",
        "mm_input_type", "mm_image_grid_points",
        "mm_audio_tower", "mm_audio_pool_size",
        "mm_projector_type", "mm_splits", "mm_std", "mm_time_interval",
    ]:
        if key in raw:
            params[key] = raw[key]

    # Vision encoder params (from CLIP config if nested)
    if "mm_vision_tower_config" in raw:
        vc = raw["mm_vision_tower_config"]
        params["vision_num_hidden_layers"] = vc.get("num_hidden_layers", 24)
        params["vision_hidden_size"] = vc.get("hidden_size", 1024)
        params["vision_intermediate_size"] = vc.get("intermediate_size", 4096)
        params["vision_num_attention_heads"] = vc.get("num_attention_heads", 16)
        params["vision_image_size"] = vc.get("image_size", 336)
        params["vision_patch_size"] = vc.get("patch_size", 14)

    # Audio encoder params (from Whisper config if nested)
    if "mm_audio_tower_config" in raw:
        ac = raw["mm_audio_tower_config"]
        params["audio_n_mels"] = ac.get("num_mel_bins", 128)
        params["audio_n_ctx"] = ac.get("max_source_positions", 1500)
        params["audio_n_state"] = ac.get("d_model", 1280)
        params["audio_n_head"] = ac.get("encoder_attention_heads", 20)
        params["audio_n_layer"] = ac.get("encoder_layers", 32)

    return ModelConfig.from_dict(params)


def load_model(model_path: str, config: ModelConfig) -> VidiEngine:
    """Load Vidi model with MLX weights."""
    engine = VidiEngine(config)

    # Check if model is quantized and apply quantization structure first
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)

    if "quantization" in raw_config:
        quant = raw_config["quantization"]
        print(f"Applying {quant['bits']}-bit quantization structure (group_size={quant['group_size']})...")
        from .quantize import quantize_engine
        quantize_engine(engine, bits=quant["bits"], group_size=quant["group_size"])

    # Load weights
    weight_files = sorted(Path(model_path).glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    weights = {}
    for wf in weight_files:
        if wf.name.endswith(".index.json"):
            continue
        print(f"Loading {wf.name} ...")
        shard = mx.load(str(wf))
        weights.update(shard)

    print(f"Loaded {len(weights)} weight tensors")

    # Apply weights to model
    # The engine has: model, vision_encoder, audio_encoder, img_projector, etc.
    # We need to distribute weights to the correct sub-modules.
    engine_weights = _distribute_weights(weights, config)

    # Load into the engine's modules
    _load_weights_into_module(engine, engine_weights)

    mx.eval(engine.model.parameters())

    return engine


def _distribute_weights(weights: dict, config: ModelConfig) -> dict:
    """Re-key weight dict to match VidiEngine's module hierarchy."""
    # Weights are already mapped by convert_weights.py.
    # Keys like:
    #   model.model.layers.0.self_attn.q_proj.weight → model.model.layers.0...
    #   vision_encoder.encoder_layers.0... → vision_encoder...
    #   audio_encoder.blocks.0... → audio_encoder...
    #   img_projector.layers.0... → img_projector...
    #   etc.
    return weights  # Already correctly keyed


def _load_weights_into_module(engine: VidiEngine, weights: dict):
    """Load weight dict into VidiEngine using nn.Module.load_weights."""
    # Collect all weight pairs
    weight_list = list(weights.items())
    # nn.Module.load_weights expects list of (key, mx.array) pairs
    # where keys use dot notation matching the module tree
    try:
        engine.load_weights(weight_list, strict=False)
    except Exception as e:
        print(f"Warning during weight loading: {e}")
        # Try manual assignment for any remaining
        _manual_load(engine, weights)


def _manual_load(module, weights, prefix=""):
    """Manually walk module tree and assign weights."""
    for name, child in module.__dict__.items():
        full_key = f"{prefix}{name}" if prefix else name
        if isinstance(child, mx.array):
            if full_key + ".weight" in weights:
                setattr(module, name, weights[full_key + ".weight"])
        elif isinstance(child, nn.Module):
            _manual_load(child, weights, full_key + ".")
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    _manual_load(item, weights, f"{full_key}.{i}.")


def main():
    parser = argparse.ArgumentParser(description="Vidi1.5-9B MLX Inference")
    parser.add_argument("--model-path", required=True, help="Path to MLX weights directory")
    parser.add_argument("--video-path", default=None, help="Path to video file")
    parser.add_argument("--image-path", default=None, help="Path to image file")
    parser.add_argument("--query", required=True, help="Question to ask about the video/image")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction FPS")
    parser.add_argument("--max-frames", type=int, default=128, help="Max frames to extract")
    args = parser.parse_args()

    if args.video_path is None and args.image_path is None:
        parser.error("Must provide either --video-path or --image-path")

    # Load
    print("Loading config ...")
    config = load_config(args.model_path)

    print("Building model ...")
    engine = load_model(args.model_path, config)

    print("Loading preprocessor ...")
    from .preprocessing import VidiPreprocessor
    preprocessor = VidiPreprocessor(args.model_path)

    # Prepare inputs
    if args.video_path:
        print(f"Processing video: {args.video_path}")
        inputs = preprocessor.prepare_video(
            args.video_path, args.query,
            fps=args.fps, max_frames=args.max_frames,
        )
    else:
        print(f"Processing image: {args.image_path}")
        inputs = preprocessor.prepare_image(args.image_path, args.query)

    # Generate
    print("Generating ...")
    t0 = time.time()

    token_ids = engine.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        mel_features=inputs["mel_features"],
        audio_sizes=inputs.get("audio_sizes", []),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    t1 = time.time()

    # Decode
    text = preprocessor.tokenizer.decode(token_ids, skip_special_tokens=True)

    # Stats
    num_tokens = len(token_ids)
    elapsed = t1 - t0
    tps = num_tokens / elapsed if elapsed > 0 else 0

    print(f"\n{'='*60}")
    print(f"Response: {text}")
    print(f"{'='*60}")
    print(f"Tokens: {num_tokens} | Time: {elapsed:.2f}s | Speed: {tps:.1f} tok/s")


if __name__ == "__main__":
    main()
