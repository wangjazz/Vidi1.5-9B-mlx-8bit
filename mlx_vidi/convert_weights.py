"""
Convert Vidi1.5-9B PyTorch safetensors → MLX safetensors.

Key mappings:
- LLM layers: mostly direct (Gemma2 part)
- Vision: model.mm_vis.* → vision_encoder.*
- Audio: model.mm_aud.encoder.* → audio_encoder.*
- Projectors: model.mm_rand_* → {img,aud}_projector/norm/pool/pos
- Conv2d: (O,I,H,W) → (O,H,W,I)
- Conv1d: (O,I,K) → (O,K,I)
- VidiRMSNorm weights: kept as-is (direct multiply, not 1+w)

Usage:
    python -m mlx_vidi.convert_weights \
        --input-dir ./weights/Vidi1.5-9B \
        --output-dir ./weights/Vidi1.5-9B-mlx
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np


def load_pytorch_safetensors(input_dir: str) -> Dict[str, mx.array]:
    """Load all safetensors shards from a PyTorch model directory.

    Uses mx.load() which natively handles bfloat16 safetensors.
    """
    weights = {}
    shard_files = sorted(Path(input_dir).glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {input_dir}")

    print(f"Loading {len(shard_files)} shard(s) from {input_dir}")
    for shard_file in shard_files:
        print(f"  Loading {shard_file.name} ...")
        shard = mx.load(str(shard_file))
        weights.update(shard)

    print(f"Loaded {len(weights)} tensors total")
    return weights


def map_key(pt_key: str) -> str:
    """Map a PyTorch weight key to the MLX model key."""

    # === LLM core (Gemma2) — mostly direct mapping ===
    # model.embed_tokens.* → model.model.embed_tokens.*
    # model.layers.{i}.* → model.model.layers.{i}.*
    # model.norm.* → model.model.norm.*
    # lm_head.* → usually tied with embed_tokens

    # === Vision encoder ===
    # model.mm_vis.vision_model.embeddings.* → vision_encoder.embeddings.*
    # model.mm_vis.vision_model.encoder.layers.{i}.* → vision_encoder.encoder_layers.{i}.*
    # model.mm_vis.vision_model.pre_layrnorm.* → vision_encoder.pre_layrnorm.*
    # model.mm_vis.vision_model.post_layernorm.* → vision_encoder.post_layernorm.*
    m = re.match(r"model\.mm_vis\.vision_model\.encoder\.layers\.(\d+)\.(.*)", pt_key)
    if m:
        layer_idx, rest = m.group(1), m.group(2)
        rest = _map_clip_layer_key(rest)
        return f"vision_encoder.encoder_layers.{layer_idx}.{rest}"

    m = re.match(r"model\.mm_vis\.vision_model\.embeddings\.(.*)", pt_key)
    if m:
        rest = m.group(1)
        return f"vision_encoder.embeddings.{rest}"

    if pt_key.startswith("model.mm_vis.vision_model.pre_layrnorm."):
        rest = pt_key.split("model.mm_vis.vision_model.pre_layrnorm.")[-1]
        return f"vision_encoder.pre_layrnorm.{rest}"

    if pt_key.startswith("model.mm_vis.vision_model.post_layernorm."):
        rest = pt_key.split("model.mm_vis.vision_model.post_layernorm.")[-1]
        return f"vision_encoder.post_layernorm.{rest}"

    # SigLip2 head (pooler): head.attention.*, head.layernorm.*, head.mlp.*, head.probe
    m = re.match(r"model\.mm_vis\.vision_model\.head\.(.*)", pt_key)
    if m:
        rest = m.group(1)
        # head.attention.in_proj_weight/bias → split into q/k/v later in sanitize
        # head.attention.out_proj.* → head.attention.out_proj.*
        # head.layernorm.* → head.layernorm.*
        # head.mlp.fc1/fc2.* → head.mlp.fc1/fc2.*
        # head.probe → head.probe
        return f"vision_encoder.head.{rest}"

    # === Audio encoder ===
    # model.mm_aud.encoder.* → audio_encoder.*
    m = re.match(r"model\.mm_aud\.encoder\.(.*)", pt_key)
    if m:
        rest = m.group(1)
        rest = _map_whisper_key(rest)
        if rest is None:
            return None  # Skip (e.g. embed_positions)
        return f"audio_encoder.{rest}"

    # === Image projector ===
    # model.mm_rand_img_projector.model.{0,2}.* → img_projector.layers.{0,2}.*
    m = re.match(r"model\.mm_rand_img_projector\.model\.(\d+)\.(.*)", pt_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        return f"img_projector.layers.{idx}.{rest}"

    # === Audio projector ===
    m = re.match(r"model\.mm_rand_aud_projector\.model\.(\d+)\.(.*)", pt_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        return f"aud_projector.layers.{idx}.{rest}"

    # === Audio pool (Conv1d) ===
    if pt_key.startswith("model.mm_rand_aud_pool."):
        rest = pt_key.split("model.mm_rand_aud_pool.")[-1]
        return f"aud_pool.{rest}"

    # === Norms (VidiRMSNorm) ===
    if pt_key.startswith("model.mm_rand_img_norm."):
        rest = pt_key.split("model.mm_rand_img_norm.")[-1]
        return f"img_norm.{rest}"
    if pt_key.startswith("model.mm_rand_aud_norm."):
        rest = pt_key.split("model.mm_rand_aud_norm.")[-1]
        return f"aud_norm.{rest}"
    if pt_key.startswith("model.mm_rand_llm_norm."):
        rest = pt_key.split("model.mm_rand_llm_norm.")[-1]
        return f"llm_norm.{rest}"

    # === Conv2DPool (no trainable params) — skip ===
    if "mm_rand_img_pool" in pt_key:
        return None  # Conv2DPool has no learnable weights

    # === Position embeddings ===
    # model.mm_rand_pos_{h,w,t}.mlp.{0,2}.* → pos_{h,w,t}.mlp.layers.{0,2}.*
    m = re.match(r"model\.mm_rand_pos_(\w+)\.mlp\.(\d+)\.(.*)", pt_key)
    if m:
        dim, idx, rest = m.group(1), m.group(2), m.group(3)
        return f"pos_{dim}.mlp.layers.{idx}.{rest}"

    # === Image-only mode projector/norm ===
    m = re.match(r"model\.mm_rand_projector\.model\.(\d+)\.(.*)", pt_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        return f"img_projector.layers.{idx}.{rest}"
    if pt_key.startswith("model.mm_rand_norm."):
        rest = pt_key.split("model.mm_rand_norm.")[-1]
        return f"img_norm.{rest}"

    # === LLM keys — direct mapping under model.model.* ===
    if pt_key.startswith("model.embed_tokens."):
        rest = pt_key.split("model.embed_tokens.")[-1]
        return f"model.model.embed_tokens.{rest}"

    m = re.match(r"model\.layers\.(\d+)\.(.*)", pt_key)
    if m:
        layer_idx, rest = m.group(1), m.group(2)
        rest = _map_gemma2_layer_key(rest)
        return f"model.model.layers.{layer_idx}.{rest}"

    if pt_key.startswith("model.norm."):
        rest = pt_key.split("model.norm.")[-1]
        return f"model.model.norm.{rest}"

    if pt_key.startswith("lm_head."):
        # Usually tied with embed_tokens in Gemma2
        return None

    # Fallback: warn and skip
    return None


def _map_clip_layer_key(rest: str) -> str:
    """Map CLIP encoder layer subkeys."""
    # self_attn.q_proj → self_attn.q_proj (direct)
    # self_attn.k_proj → self_attn.k_proj
    # self_attn.v_proj → self_attn.v_proj
    # self_attn.out_proj → self_attn.out_proj
    # layer_norm1 → layer_norm1
    # layer_norm2 → layer_norm2
    # mlp.fc1 → mlp.fc1
    # mlp.fc2 → mlp.fc2
    return rest  # All keys map directly


def _map_whisper_key(rest: str) -> str:
    """Map Whisper encoder subkeys.

    HF transformers keys → our mlx_whisper-compatible layout:
    - layers.{i}.self_attn.q_proj → blocks.{i}.attn.query
    - layers.{i}.self_attn.k_proj → blocks.{i}.attn.key
    - layers.{i}.self_attn.v_proj → blocks.{i}.attn.value
    - layers.{i}.self_attn.out_proj → blocks.{i}.attn.out
    - layers.{i}.self_attn_layer_norm → blocks.{i}.attn_ln
    - layers.{i}.fc1 → blocks.{i}.mlp1
    - layers.{i}.fc2 → blocks.{i}.mlp2
    - layers.{i}.final_layer_norm → blocks.{i}.mlp_ln
    - layer_norm → ln_post
    - embed_positions → (skip, we use sinusoidal)
    """
    # Layer mapping
    m = re.match(r"layers\.(\d+)\.(.*)", rest)
    if m:
        idx, subkey = m.group(1), m.group(2)
        mapped = _map_whisper_block_key(subkey)
        if mapped is None:
            return None
        return f"blocks.{idx}.{mapped}"

    if rest.startswith("layer_norm."):
        return "ln_post." + rest.split("layer_norm.")[-1]

    if rest.startswith("embed_positions"):
        return None  # We use sinusoidal

    # conv1, conv2
    return rest


def _map_whisper_block_key(subkey: str) -> str:
    """Map a single Whisper block subkey."""
    mappings = {
        "self_attn.q_proj": "attn.query",
        "self_attn.k_proj": "attn.key",
        "self_attn.v_proj": "attn.value",
        "self_attn.out_proj": "attn.out",
        "self_attn_layer_norm": "attn_ln",
        "fc1": "mlp1",
        "fc2": "mlp2",
        "final_layer_norm": "mlp_ln",
    }
    for prefix, mapped in mappings.items():
        if subkey.startswith(prefix):
            suffix = subkey[len(prefix):]
            return mapped + suffix
    return subkey


def _map_gemma2_layer_key(rest: str) -> str:
    """Map Gemma2 decoder layer subkeys."""
    # self_attn.q_proj → self_attn.q_proj (direct)
    # self_attn.k_proj → self_attn.k_proj
    # self_attn.v_proj → self_attn.v_proj
    # self_attn.o_proj → self_attn.o_proj
    # mlp.gate_proj → mlp.gate_proj
    # mlp.up_proj → mlp.up_proj
    # mlp.down_proj → mlp.down_proj
    # input_layernorm → input_layernorm
    # post_attention_layernorm → post_attention_layernorm
    # pre_feedforward_layernorm → pre_feedforward_layernorm
    # post_feedforward_layernorm → post_feedforward_layernorm
    return rest  # All direct


def sanitize_weight(key: str, value: mx.array) -> dict:
    """Apply necessary transformations to weight tensors.

    Returns dict of {key: value} — usually 1 entry, but in_proj splits into 3.
    - Conv2d: (O, I, H, W) → (O, H, W, I)
    - Conv1d: (O, I, K) → (O, K, I)
    - SigLip2 in_proj_weight: (3*D, D) → split into q/k/v
    """
    # SigLip2 head attention in_proj_weight → split into q/k/v
    if "head.attention.in_proj_weight" in key:
        D = value.shape[1]
        q, k, v = value[:D], value[D:2*D], value[2*D:]
        base = key.replace("attention.in_proj_weight", "attention.")
        return {
            base + "q_proj.weight": q,
            base + "k_proj.weight": k,
            base + "v_proj.weight": v,
        }
    if "head.attention.in_proj_bias" in key:
        D = value.shape[0] // 3
        q, k, v = value[:D], value[D:2*D], value[2*D:]
        base = key.replace("attention.in_proj_bias", "attention.")
        return {
            base + "q_proj.bias": q,
            base + "k_proj.bias": k,
            base + "v_proj.bias": v,
        }

    # Vision encoder Conv2d (patch_embedding)
    if "patch_embedding.weight" in key and value.ndim == 4:
        O, dim1, dim2, dim3 = value.shape
        if dim2 == dim3 and dim1 != dim2:
            # PyTorch format (O, I, H, W) → MLX (O, H, W, I)
            value = value.transpose(0, 2, 3, 1)

    # Audio encoder Conv1d
    if ("conv1.weight" in key or "conv2.weight" in key) and "audio_encoder" in key:
        if value.ndim == 3:
            value = value.transpose(0, 2, 1)

    # Audio pool Conv1d
    if "aud_pool.weight" in key and value.ndim == 3:
        value = value.transpose(0, 2, 1)

    return {key: value}


def convert(input_dir: str, output_dir: str, dtype: str = "float16"):
    """Convert Vidi1.5-9B weights from PyTorch to MLX format."""
    os.makedirs(output_dir, exist_ok=True)

    # Load PyTorch weights
    pt_weights = load_pytorch_safetensors(input_dir)

    # Map and sanitize
    mlx_weights = {}
    skipped = []
    for pt_key, value in pt_weights.items():
        mlx_key = map_key(pt_key)
        if mlx_key is None:
            skipped.append(pt_key)
            continue

        # sanitize_weight returns a dict (may split 1 key → multiple)
        sanitized = sanitize_weight(mlx_key, value)

        for out_key, out_value in sanitized.items():
            # Cast dtype (keep position embedding MLP in float32)
            if dtype == "float16":
                if "pos_" in out_key and "mlp" in out_key:
                    out_value = out_value.astype(mx.float32)
                else:
                    out_value = out_value.astype(mx.float16)
            elif dtype == "bfloat16":
                if "pos_" in out_key and "mlp" in out_key:
                    out_value = out_value.astype(mx.float32)
                else:
                    out_value = out_value.astype(mx.bfloat16)

            mlx_weights[out_key] = out_value

    print(f"\nConverted {len(mlx_weights)} tensors, skipped {len(skipped)}:")
    for k in skipped:
        print(f"  SKIP: {k}")

    # Save as sharded safetensors (4GB per shard)
    _save_sharded(mlx_weights, output_dir, max_shard_size=4 * 1024**3)

    # Save config
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        import shutil
        shutil.copy(config_path, os.path.join(output_dir, "config.json"))
        print(f"Copied config.json")

    # Save tokenizer files
    for fname in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json"]:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(output_dir, fname))
            print(f"Copied {fname}")

    print(f"\nDone! MLX weights saved to {output_dir}")


def _save_sharded(weights: Dict[str, mx.array], output_dir: str,
                   max_shard_size: int = 4 * 1024**3):
    """Save weights as sharded safetensors files."""
    # Calculate total size and plan shards
    total_size = sum(w.nbytes for w in weights.values())
    print(f"Total weight size: {total_size / 1024**3:.2f} GB")

    if total_size <= max_shard_size:
        # Single file
        out_path = os.path.join(output_dir, "model.safetensors")
        mx.save_safetensors(out_path, weights)
        print(f"Saved single file: {out_path}")

        # Write index
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": {k: "model.safetensors" for k in weights},
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    else:
        # Shard
        shards = {}
        current_shard = {}
        current_size = 0
        shard_idx = 0
        weight_map = {}

        for key in sorted(weights.keys()):
            w = weights[key]
            if current_size + w.nbytes > max_shard_size and current_shard:
                shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
                shards[shard_name] = current_shard
                current_shard = {}
                current_size = 0
                shard_idx += 1

            current_shard[key] = w
            current_size += w.nbytes

        if current_shard:
            shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            shards[shard_name] = current_shard

        # Fix shard names
        n_shards = len(shards)
        final_shards = {}
        for old_name, shard_weights in shards.items():
            new_name = old_name.replace("XXXXX", f"{n_shards:05d}")
            final_shards[new_name] = shard_weights
            for key in shard_weights:
                weight_map[key] = new_name

        # Save each shard
        for shard_name, shard_weights in final_shards.items():
            out_path = os.path.join(output_dir, shard_name)
            mx.save_safetensors(out_path, shard_weights)
            print(f"Saved shard: {shard_name} ({sum(w.nbytes for w in shard_weights.values()) / 1024**3:.2f} GB)")

        # Write index
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    print(f"Weight index saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Vidi1.5-9B PyTorch → MLX")
    parser.add_argument("--input-dir", required=True, help="Path to PyTorch model directory")
    parser.add_argument("--output-dir", required=True, help="Path to save MLX weights")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    convert(args.input_dir, args.output_dir, args.dtype)
