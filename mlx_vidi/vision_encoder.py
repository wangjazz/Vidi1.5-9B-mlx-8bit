"""
SigLip2 Vision Tower for Vidi1.5-9B MLX port.

SigLip2-so400m-patch14-384:
- hidden_size: 1152, intermediate_size: 4304
- 27 encoder layers, 16 attention heads
- image_size: 384, patch_size: 14 → 27×27 = 729 patches
- No CLS token (unlike CLIP)
- Has bias on patch embedding and attention projections
- Has a pooler head (attention + MLP) — not used for Vidi features

Extracts patch features from select_layer (default -2 = layer 25).
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class VisionAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dims // num_heads) ** -0.5
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, D = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        num_heads = self.num_heads
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU(approx="fast")

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 layer_norm_eps: float):
        super().__init__()
        self.self_attn = VisionAttention(hidden_size, num_heads, bias=True)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = VisionMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        y = self.layer_norm1(x)
        y = self.self_attn(y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class VisionEmbeddings(nn.Module):
    """Patch embeddings for SigLip2 (no CLS token, has bias)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.vision_hidden_size
        self.image_size = config.vision_image_size
        self.patch_size = config.vision_patch_size
        self.model_type = config.vision_model_type

        has_bias = (config.vision_model_type == "siglip_vision_model")

        self.patch_embedding = nn.Conv2d(
            in_channels=config.vision_num_channels,
            out_channels=config.vision_hidden_size,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
            bias=has_bias,
        )

        num_patches = (config.vision_image_size // config.vision_patch_size) ** 2

        if config.vision_model_type == "clip_vision_model":
            # CLIP: CLS token + position embedding
            self.class_embedding = mx.zeros((config.vision_hidden_size,))
            num_positions = num_patches + 1
        else:
            # SigLip: no CLS token
            self.class_embedding = None
            num_positions = num_patches

        self.position_embedding = nn.Embedding(num_positions, config.vision_hidden_size)
        self.num_positions = num_positions

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        # x: (B, H, W, C) in MLX NHWC convention
        patch_embeddings = self.patch_embedding(x)  # (B, H', W', D)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)  # (B, N, D)

        if self.class_embedding is not None:
            # CLIP: prepend CLS
            cls_embeddings = mx.broadcast_to(
                self.class_embedding, (batch_size, 1, self.hidden_size)
            )
            embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        else:
            # SigLip: no CLS
            embeddings = patch_embeddings

        position_ids = mx.array(np.arange(self.num_positions)[None, :])
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


# ---------------------------------------------------------------------------
# SigLip2 Pooler Head (for completeness — not used for Vidi patch features)
# ---------------------------------------------------------------------------

class SigLipHead(nn.Module):
    """SigLip2 multi-head attention pooler head.

    Uses a learnable probe as query, attends over patch features,
    then MLP. Only needed if using pooler_output.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.probe = mx.zeros((1, 1, config.vision_hidden_size))
        self.attention = SigLipMultiheadAttentionPoolingHead(config)
        self.layernorm = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_layer_norm_eps)
        self.mlp = VisionMLP(config.vision_hidden_size, config.vision_intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, N, D)
        B = x.shape[0]
        probe = mx.broadcast_to(self.probe, (B, 1, x.shape[-1]))
        x = self.attention(probe, x)
        residual = x
        x = self.layernorm(x)
        x = residual + self.mlp(x)
        return x[:, 0]  # (B, D)


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """In-proj style multihead attention (q=probe, k/v=features)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.vision_hidden_size
        # SigLip2 uses in_proj_weight/in_proj_bias for q/k/v combined
        # We split into separate projections for MLX
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.num_heads = config.vision_num_attention_heads
        self.scale = (dim // config.vision_num_attention_heads) ** -0.5

    def __call__(self, query: mx.array, kv: mx.array) -> mx.array:
        B, Lq, D = query.shape
        _, Lkv, _ = kv.shape

        q = self.q_proj(query).reshape(B, Lq, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(kv).reshape(B, Lkv, self.num_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(kv).reshape(B, Lkv, self.num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(B, Lq, -1)
        return self.out_proj(output)


# ---------------------------------------------------------------------------
# SigLip2VisionEncoder — full encoder
# ---------------------------------------------------------------------------

class SigLip2VisionEncoder(nn.Module):
    """SigLip2-so400m-patch14-384 vision encoder for Vidi.

    Returns ``(pooler_output, patch_features)`` from select_layer (default -2).

    SigLip2 has no CLS token; pooler_output comes from the attention head.
    For Vidi's pipeline, only ``patch_features`` is used.

    Attributes:
        num_patches_per_side: 384//14 = 27
        hidden_size: 1152
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.select_layer = config.mm_vision_select_layer  # -2

        self.embeddings = VisionEmbeddings(config)

        # SigLip has no pre_layrnorm
        if config.vision_model_type == "clip_vision_model":
            self.pre_layrnorm = nn.LayerNorm(
                config.vision_hidden_size, eps=config.vision_layer_norm_eps
            )
        else:
            self.pre_layrnorm = None

        self.encoder_layers = [
            VisionEncoderLayer(
                config.vision_hidden_size,
                config.vision_num_attention_heads,
                config.vision_intermediate_size,
                config.vision_layer_norm_eps,
            )
            for _ in range(config.vision_num_hidden_layers)
        ]

        self.post_layernorm = nn.LayerNorm(
            config.vision_hidden_size, eps=config.vision_layer_norm_eps
        )

        # SigLip2 has a pooler head
        if config.vision_model_type == "siglip_vision_model":
            self.head = SigLipHead(config)
        else:
            self.head = None

    @property
    def num_patches_per_side(self) -> int:
        return self.config.vision_num_patches_per_side

    @property
    def hidden_size(self) -> int:
        return self.config.vision_hidden_size

    def __call__(self, x: mx.array) -> tuple:
        """Forward pass.

        Args:
            x: (B, H, W, C) pixel values in NHWC (MLX convention).

        Returns:
            (pooler_output, patch_features):
                pooler_output: (B, D) — from head (SigLip) or CLS post-norm (CLIP)
                patch_features: (B, num_patches, D) — from select_layer
        """
        x = self.embeddings(x)     # (B, N, D) — N=729 for SigLip, 730 for CLIP

        if self.pre_layrnorm is not None:
            x = self.pre_layrnorm(x)

        # Run encoder layers, grab hidden state at select_layer
        target_layer = len(self.encoder_layers) + self.select_layer  # e.g. 27 + (-2) = 25

        selected = None
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if i == target_layer:
                selected = x

        if selected is None:
            selected = x  # Fallback

        # Extract patch features
        if self.config.vision_model_type == "clip_vision_model":
            # CLIP: CLS at index 0, patches at 1:
            pooler_output = self.post_layernorm(selected[:, 0])
            patch_features = selected[:, 1:]
        else:
            # SigLip: all tokens are patches, no CLS
            patch_features = selected
            if self.head is not None:
                pooler_output = self.head(self.post_layernorm(x))
            else:
                pooler_output = self.post_layernorm(x[:, 0])

        return pooler_output, patch_features
