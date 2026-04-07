"""
Autoregressive generation for Vidi1.5-9B MLX port.

Supports:
- Greedy decoding (do_sample=False)
- Temperature sampling
- Three-way KV cache: text (mlx_lm KVCache), image (list), audio (list)
- EOS token = 107
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .model import Model
from .vision_encoder import SigLip2VisionEncoder
from .audio_encoder import WhisperAudioEncoder
from .projectors import (
    VidiRMSNorm, MMProjector, LearnablePosEmbd, Conv2DPool,
    rms_norm, space_to_depth, splitted_call,
)


# ---------------------------------------------------------------------------
# KV Cache — simple concat-based (matches mlx-lm's ConcatenateKVCache)
# ---------------------------------------------------------------------------

class KVCache:
    """Simple KV cache that concatenates new keys/values."""

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

    @property
    def offset(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[2]

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values

    def make_mask(self, N: int, return_array: bool = False, window_size=None):
        """Create causal mask."""
        offset = self.offset - N  # We already added N tokens
        total = self.offset
        rinds = mx.arange(total)
        linds = mx.arange(offset, total)
        mask = linds[:, None] >= rinds[None]
        if window_size is not None:
            mask = mask & (linds[:, None] < rinds[None] + window_size)
        return mask


# ---------------------------------------------------------------------------
# Vidi inference engine
# ---------------------------------------------------------------------------

class VidiEngine(nn.Module):
    """End-to-end inference engine for Vidi1.5-9B.

    Manages:
    - Model loading and weight assignment
    - Multimodal encoding (vision + audio)
    - Autoregressive generation with tri-modal KV cache
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build LLM
        self.model = Model(config)

        # Build vision encoder
        self.vision_encoder = SigLip2VisionEncoder(config)

        # Build audio encoder
        self.audio_encoder = WhisperAudioEncoder(config)

        # Build multimodal projectors and norms
        if config.mm_input_type == "video":
            pool_size = config.mm_image_pool_size or 2
            self.img_pool = Conv2DPool(
                d_in=config.vision_hidden_size,
                d_out=config.vision_hidden_size,
                s_in=config.vision_num_patches_per_side,
                s_out=pool_size,
                mm_splits=config.mm_splits,
                mm_image_pool_size=pool_size,
            )
            self.img_projector = MMProjector(
                config.mm_projector_type,
                config.vision_hidden_size * (pool_size ** 2),
                config.hidden_size,
            )
            self.img_norm = VidiRMSNorm(config.hidden_size)

            self.aud_pool = nn.Conv1d(
                config.audio_n_state,
                config.hidden_size,
                kernel_size=config.mm_audio_pool_size or 5,
                stride=config.mm_audio_pool_size or 5,
                bias=False,
            )
            self.aud_projector = MMProjector(
                config.mm_projector_type,
                config.hidden_size,
                config.hidden_size,
            )
            self.aud_norm = VidiRMSNorm(config.hidden_size)

            self.pos_h = LearnablePosEmbd(config.hidden_size, pool_size)
            self.pos_w = LearnablePosEmbd(config.hidden_size, pool_size)
            self.pos_t = LearnablePosEmbd(
                config.hidden_size, config.mm_time_interval or 10000
            )
        elif config.mm_input_type == "image":
            self.img_projector = MMProjector(
                config.mm_projector_type,
                config.vision_hidden_size,
                config.hidden_size,
            )
            self.img_norm = VidiRMSNorm(config.hidden_size)
            self.pos_h = LearnablePosEmbd(
                config.hidden_size, config.vision_num_patches_per_side
            )
            self.pos_w = LearnablePosEmbd(
                config.hidden_size, config.vision_num_patches_per_side
            )

        self.llm_norm = VidiRMSNorm(config.hidden_size)

    def encode_video_images(self, pixel_values: mx.array) -> tuple:
        """Encode video frames through vision pipeline.

        Args:
            pixel_values: (N_frames, H, W, C) NHWC images.

        Returns:
            (image_features, image_mask):
                image_features: (1, total_tokens, D)
                image_mask: (1, total_tokens) bool
        """
        config = self.config
        pool_size = config.mm_image_pool_size or 2
        num_patches_per_side = config.vision_num_patches_per_side

        # Vision encoder: (N, H, W, C) → (N, num_patches, D)
        _, patch_features = self.vision_encoder(pixel_values)

        N = patch_features.shape[0]
        D = patch_features.shape[-1]

        # Reshape to spatial: (N, h, w, D) → (N, D, h, w) NCHW
        h = w = num_patches_per_side
        feat = patch_features.reshape(N, h, w, D)
        feat = feat.transpose(0, 3, 1, 2)  # (N, D, h, w)

        # Dynamic token limiting
        n_tokens = N * (h + 1) * (w + 1)
        max_tokens = 60000 * pool_size * pool_size
        if n_tokens > max_tokens:
            hw = _resize_by_tokens(N, h + 1, w + 1, max_tokens)
        else:
            hw = (h + 1, w + 1)

        # Conv2DPool: pad + interpolate + space_to_depth
        feat = self.img_pool(feat, hw)  # (N, D*pool², h', w')

        # Back to NHWC: (N, D', h', w') → (N, h', w', D')
        feat = feat.transpose(0, 2, 3, 1)

        # MLP projection + norm
        feat = self.img_projector(feat)
        feat = self.img_norm(feat)

        # Add spatial position embeddings
        feat = feat + rms_norm(self.pos_h(feat, dim=1))
        feat = feat + rms_norm(self.pos_w(feat, dim=2))

        # Add temporal position embeddings
        # feat is (N, h', w', D) — add per-frame temporal PE
        feat = feat + rms_norm(self.pos_t(feat, dim=0))

        # Flatten spatial dims: (N, h', w', D) → (N*h'*w', D)
        feat = feat.reshape(-1, feat.shape[-1])  # (total_tokens, D)

        # Add batch dim and create mask
        feat = feat[None, :, :]  # (1, total_tokens, D)
        mask = mx.ones((1, feat.shape[1]), dtype=mx.bool_)

        # LLM norm
        feat = self.llm_norm(feat)
        feat = feat * mask[:, :, None]

        return feat, mask

    def encode_video_audios(
        self,
        mel_features: mx.array,
        audio_sizes: list,
    ) -> tuple:
        """Encode audio through Whisper + pool + project.

        Args:
            mel_features: (N_chunks, T, n_mels)
            audio_sizes: list of actual frame counts per chunk

        Returns:
            (audio_features, audio_mask):
                audio_features: (1, total_tokens, D)
                audio_mask: (1, total_tokens) bool
        """
        config = self.config
        pool_size = config.mm_audio_pool_size or 5

        # Whisper encoder: (N, T, n_mels) → (N, T', n_state)
        audio_feat = self.audio_encoder(mel_features)

        # Pool ratio
        pool_ratio = config.audio_max_source_positions / 3000  # nb_max_frames
        scaled_sizes = [int(s * pool_ratio) for s in audio_sizes]

        # Flatten: (N, T', D) → flatten and truncate per chunk
        chunks = []
        for i, size in enumerate(scaled_sizes):
            chunk = audio_feat[i, :size]  # (size, D)
            chunks.append(chunk)

        if not chunks:
            return None, None

        # Pad to max length
        max_len = max(c.shape[0] for c in chunks)
        D = chunks[0].shape[-1]
        padded = mx.zeros((len(chunks), max_len, D), dtype=chunks[0].dtype)
        for i, c in enumerate(chunks):
            padded = padded.at[i, :c.shape[0]].add(c)

        # Conv1d pool: (B, T, D) → transpose → conv → transpose
        audio_feat = padded.transpose(0, 2, 1)  # (B, D, T)
        audio_feat = self.aud_pool(audio_feat.transpose(0, 2, 1))  # Conv1d expects (B, T, C_in) in MLX

        pooled_sizes = [int(s / pool_size) for s in scaled_sizes]

        # Truncate and flatten
        audio_chunks = []
        for i, s in enumerate(pooled_sizes):
            if s > 0:
                audio_chunks.append(audio_feat[i, :s])

        if not audio_chunks:
            return None, None

        # Project + norm
        all_audio = mx.concatenate(audio_chunks, axis=0)
        all_audio = self.aud_projector(all_audio)
        all_audio = self.aud_norm(all_audio)

        # Split back and add temporal PE
        parts = []
        offset = 0
        for s in pooled_sizes:
            if s > 0:
                part = all_audio[offset:offset + s]
                part = part + rms_norm(self.pos_t(part, dim=0))
                parts.append(part)
                offset += s

        # Pad to max length
        max_len = max(p.shape[0] for p in parts)
        D = parts[0].shape[-1]
        padded = mx.zeros((len(parts), max_len, D), dtype=parts[0].dtype)
        for i, p in enumerate(parts):
            padded = padded.at[i, :p.shape[0]].add(p)

        # Reduce to single batch (batch=1 for inference)
        # Flatten all chunks: (N_chunks, max_len, D) → (1, total, D)
        total_feat = padded.reshape(1, -1, D)

        # Mask
        mask = mx.zeros((1, total_feat.shape[1]), dtype=mx.bool_)
        offset = 0
        for s in pooled_sizes:
            mask = mask.at[0, offset:offset + s].add(mx.ones((s,), dtype=mx.bool_))
            offset += max_len  # Each chunk is padded to max_len

        # LLM norm
        total_feat = self.llm_norm(total_feat)
        total_feat = total_feat * mask[:, :, None]

        return total_feat, mask

    def generate(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mel_features: Optional[mx.array] = None,
        audio_sizes: Optional[list] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
    ) -> str:
        """Generate text given multimodal inputs.

        Args:
            input_ids: (1, L) token IDs
            pixel_values: (N, H, W, C) video frames
            mel_features: (N_chunks, T, n_mels) audio mel spectrograms
            audio_sizes: actual audio frame counts per chunk
            max_tokens: maximum tokens to generate
            temperature: sampling temperature (0 = greedy)
            eos_token_id: end of sequence token ID

        Returns:
            Generated text string.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        # === Encode multimodal inputs ===
        image_embeds = None
        image_mask = None
        audio_embeds = None
        audio_mask = None

        if pixel_values is not None:
            image_embeds, image_mask = self.encode_video_images(pixel_values)
            mx.eval(image_embeds, image_mask)

        if mel_features is not None and audio_sizes:
            audio_embeds, audio_mask = self.encode_video_audios(mel_features, audio_sizes)
            if audio_embeds is not None:
                mx.eval(audio_embeds, audio_mask)

        # === Initialise KV caches ===
        text_cache = [KVCache() for _ in range(self.config.num_hidden_layers)]
        image_cache = [] if image_embeds is not None else None
        audio_cache = [] if audio_embeds is not None else None

        # === Prefill ===
        logits = self.model(
            input_ids,
            cache=text_cache,
            image_embeds=image_embeds,
            image_mask=image_mask,
            image_cache=image_cache,
            audio_embeds=audio_embeds,
            audio_mask=audio_mask,
            audio_cache=audio_cache,
        )
        mx.eval(logits)

        # After prefill, image/audio caches are populated.
        # For subsequent decode steps, set embeds to None to use cache.
        # No — actually in decode steps the cross-attention will use the cache
        # because layer_idx < len(cache) will be True.

        # Get first token
        token = _sample(logits[:, -1, :], temperature)
        mx.eval(token)
        tokens = [token.item()]

        # === Decode loop ===
        for _ in range(max_tokens - 1):
            if tokens[-1] == eos_token_id:
                break

            token_input = mx.array([[tokens[-1]]])
            logits = self.model(
                token_input,
                cache=text_cache,
                image_embeds=image_embeds,
                image_mask=image_mask,
                image_cache=image_cache,
                audio_embeds=audio_embeds,
                audio_mask=audio_mask,
                audio_cache=audio_cache,
            )
            mx.eval(logits)

            token = _sample(logits[:, -1, :], temperature)
            mx.eval(token)
            tokens.append(token.item())

        return tokens


def _sample(logits: mx.array, temperature: float) -> mx.array:
    """Sample next token from logits."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    probs = mx.softmax(logits / temperature, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-10))


def _resize_by_tokens(num_frames: int, h: int, w: int, max_tokens: int) -> tuple:
    """Compute target h, w to stay within token budget."""
    import math
    ratio = math.sqrt(max_tokens / (num_frames * h * w))
    new_h = max(10, int(h * ratio) - int(h * ratio) % 2)
    new_w = max(10, int(w * ratio) - int(w * ratio) % 2)
    return new_h, new_w
