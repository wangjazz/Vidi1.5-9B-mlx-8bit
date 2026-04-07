"""
Quantize Vidi1.5-9B MLX weights to 4-bit.

Quantizes Linear layers in:
- LLM decoder layers (biggest savings)
- Vision encoder layers
- Audio encoder layers
- Projectors

Skips:
- Embedding layers (embed_tokens, position_embedding, patch_embedding)
- Norm layers (RMSNorm, LayerNorm)
- Very small layers (< group_size)

Usage:
    python -m mlx_vidi.quantize \
        --input-dir ./weights/Vidi1.5-9B-mlx \
        --output-dir ./weights/Vidi1.5-9B-mlx-4bit
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


def quantize_engine(
    engine: VidiEngine,
    bits: int = 4,
    group_size: int = 64,
) -> None:
    """Quantize VidiEngine in-place using nn.quantize.

    Applies quantization to all Linear layers that meet size requirements,
    skipping embeddings, norms, and small layers.
    """

    def should_quantize(path: str, module: nn.Module) -> bool:
        """Predicate: which layers to quantize."""
        if not hasattr(module, "to_quantized"):
            return False

        # Skip if weight dimension not divisible by group_size
        if module.weight.shape[-1] % group_size != 0:
            return False

        # Skip embedding layers
        skip_patterns = [
            "embed_tokens",
            "position_embedding",
            "patch_embedding",
            "class_embedding",
            "probe",
        ]
        for pattern in skip_patterns:
            if pattern in path:
                return False

        # Skip norm layers (they don't have to_quantized anyway, but just in case)
        if "norm" in path.split(".")[-1]:
            return False

        # Skip Conv layers (Conv1d, Conv2d)
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            return False

        return True

    nn.quantize(
        engine,
        group_size=group_size,
        bits=bits,
        class_predicate=should_quantize,
    )


def compute_bits_per_weight(model: nn.Module) -> float:
    """Compute average bits per weight after quantization."""
    total_bits = 0
    total_params = 0
    for name, param in model.parameters().items():
        if isinstance(param, dict):
            # Quantized layer: has 'weight', 'scales', 'biases'
            for k, v in param.items():
                total_bits += v.nbytes * 8
        elif isinstance(param, mx.array):
            total_bits += param.nbytes * 8
            total_params += param.size

    # For quantized layers, count the original param count
    leaf_modules = model.leaf_modules()
    for path, mod in leaf_modules.items():
        if hasattr(mod, "weight") and isinstance(mod.weight, mx.array):
            total_params += mod.weight.size

    # Rough estimate
    total_bytes = sum(
        v.nbytes for _, v in nn.utils.tree_flatten(model.parameters())
    )
    total_elements = sum(
        v.size for _, v in nn.utils.tree_flatten(model.parameters())
    )
    if total_elements > 0:
        return (total_bytes * 8) / total_elements
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Quantize Vidi1.5-9B MLX")
    parser.add_argument("--input-dir", required=True, help="Path to fp16 MLX weights")
    parser.add_argument("--output-dir", required=True, help="Path to save quantized weights")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config_path = os.path.join(args.input_dir, "config.json")
    with open(config_path) as f:
        raw = json.load(f)
    config = ModelConfig.from_dict(raw)
    print(f"Config loaded: {config.hidden_size}d, {config.num_hidden_layers}L")

    # Build engine
    print("Building model...")
    engine = VidiEngine(config)

    # Load weights
    print("Loading fp16 weights...")
    all_weights = {}
    for wf in sorted(Path(args.input_dir).glob("model-*.safetensors")):
        if "index" in wf.name:
            continue
        print(f"  {wf.name}")
        all_weights.update(mx.load(str(wf)))

    engine.load_weights(list(all_weights.items()), strict=False)
    mx.eval(engine.parameters())
    del all_weights

    # Compute pre-quantization size
    pre_bytes = sum(v.nbytes for _, v in nn.utils.tree_flatten(engine.parameters()))
    print(f"Pre-quantization size: {pre_bytes / 1024**3:.2f} GB")

    # Quantize
    print(f"Quantizing to {args.bits}-bit (group_size={args.group_size})...")
    t0 = time.time()
    quantize_engine(engine, bits=args.bits, group_size=args.group_size)
    mx.eval(engine.parameters())
    t1 = time.time()
    print(f"Quantization done in {t1-t0:.1f}s")

    # Compute post-quantization size
    post_bytes = sum(v.nbytes for _, v in nn.utils.tree_flatten(engine.parameters()))
    print(f"Post-quantization size: {post_bytes / 1024**3:.2f} GB")
    print(f"Compression ratio: {pre_bytes / post_bytes:.2f}x")

    # Save quantized weights
    print("Saving quantized weights...")
    weights = dict(nn.utils.tree_flatten(engine.parameters()))
    _save_sharded(weights, args.output_dir)

    # Save config with quantization info
    quant_config = raw.copy()
    quant_config["quantization"] = {
        "group_size": args.group_size,
        "bits": args.bits,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    # Copy tokenizer files
    import shutil
    for fname in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json"]:
        src = os.path.join(args.input_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_dir, fname))

    print(f"\nDone! Quantized weights saved to {args.output_dir}")


def _save_sharded(weights: dict, output_dir: str, max_shard_size: int = 4 * 1024**3):
    """Save weights as sharded safetensors."""
    total_size = sum(w.nbytes for w in weights.values())
    print(f"Total size: {total_size / 1024**3:.2f} GB")

    if total_size <= max_shard_size:
        out_path = os.path.join(output_dir, "model.safetensors")
        mx.save_safetensors(out_path, weights)
        print(f"Saved: {out_path}")
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": {k: "model.safetensors" for k in weights},
        }
    else:
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
            shards[f"model-{shard_idx:05d}-of-XXXXX.safetensors"] = current_shard

        n_shards = len(shards)
        final_shards = {}
        for old_name, shard_weights in shards.items():
            new_name = old_name.replace("XXXXX", f"{n_shards:05d}")
            final_shards[new_name] = shard_weights
            for key in shard_weights:
                weight_map[key] = new_name

        for shard_name, shard_weights in final_shards.items():
            out_path = os.path.join(output_dir, shard_name)
            mx.save_safetensors(out_path, shard_weights)
            sz = sum(w.nbytes for w in shard_weights.values())
            print(f"Saved: {shard_name} ({sz / 1024**3:.2f} GB)")

        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
