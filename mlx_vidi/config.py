"""
ModelConfig for Vidi1.5-9B MLX port.
Combines Gemma2 LLM config with multimodal extensions.
"""

import inspect
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Unified config for Vidi1.5-9B (Dual Attention Gemma2 + SigLip2 + Whisper)."""

    # --- Gemma2 LLM ---
    model_type: str = "dattn_gemma2"
    hidden_size: int = 3584
    num_hidden_layers: int = 42
    intermediate_size: int = 14336
    num_attention_heads: int = 16
    head_dim: int = 256
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    vocab_size: int = 256000
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0
    query_pre_attn_scalar: float = 256.0  # head_dim for Gemma2-9B
    sliding_window: int = 4096

    # --- Vision ---
    mm_vision_tower: str = "google/siglip2-so400m-patch14-384"
    mm_vision_select_layer: int = -2
    mm_image_pool_size: Optional[int] = 2
    mm_image_aspect_ratio: str = "resize"
    mm_input_type: str = "video"
    mm_image_grid_points: Optional[List[List[int]]] = None

    # --- Audio ---
    mm_audio_tower: str = "openai/whisper-large-v3"
    mm_audio_pool_size: Optional[int] = 5

    # --- Multimodal misc ---
    mm_projector_type: str = "mlp2x_gelu"
    mm_splits: int = 4
    mm_std: Optional[float] = 0.028976401314139366
    mm_time_interval: Optional[int] = 10000

    # --- Vision encoder (SigLip2-so400m-patch14-384) ---
    vision_model_type: str = "siglip_vision_model"  # "clip_vision_model" or "siglip_vision_model"
    vision_num_hidden_layers: int = 27
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_attention_heads: int = 16
    vision_image_size: int = 384
    vision_patch_size: int = 14
    vision_num_channels: int = 3
    vision_layer_norm_eps: float = 1e-6

    # --- Audio encoder (Whisper-large-v3) ---
    audio_n_mels: int = 128
    audio_n_ctx: int = 1500
    audio_n_state: int = 1280
    audio_n_head: int = 20
    audio_n_layer: int = 32
    audio_max_source_positions: int = 1500

    # --- EOS ---
    eos_token_id: int = 107

    @classmethod
    def from_dict(cls, params: dict):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    @property
    def vision_num_patches_per_side(self) -> int:
        return self.vision_image_size // self.vision_patch_size

    @property
    def vision_num_patches(self) -> int:
        return self.vision_num_patches_per_side ** 2
