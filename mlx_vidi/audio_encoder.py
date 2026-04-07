"""
Whisper Audio Encoder for Vidi1.5-9B MLX port.

Based on mlx-whisper's AudioEncoder. Only the encoder part is needed
(no text decoder). Uses Whisper-large-v3 architecture:
128 mel bins → Conv1d×2 → 32 transformer blocks → LayerNorm.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> mx.array:
    """Sinusoidal positional embedding (Whisper-style)."""
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = mx.exp(-log_timescale_increment * mx.arange(channels // 2))
    scaled_time = mx.arange(length)[:, None] * inv_timescales[None, :]
    return mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], axis=1)


class WhisperMultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.reshape(n_batch, n_ctx, self.n_head, -1).transpose(0, 2, 1, 3) * scale
        k = k.reshape(n_batch, n_ctx, self.n_head, -1).transpose(0, 2, 3, 1) * scale
        v = v.reshape(n_batch, n_ctx, self.n_head, -1).transpose(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = mx.softmax(qk, axis=-1, precise=True)
        out = (w @ v).transpose(0, 2, 1, 3)
        out = out.reshape(n_batch, n_ctx, n_state)
        return self.out(out)


class WhisperResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = WhisperMultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        n_mlp = n_state * 4
        self.mlp1 = nn.Linear(n_state, n_mlp)
        self.mlp2 = nn.Linear(n_mlp, n_state)
        self.mlp_ln = nn.LayerNorm(n_state)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.attn(self.attn_ln(x))
        x = x + y
        y = self.mlp2(nn.gelu(self.mlp1(self.mlp_ln(x))))
        x = x + y
        return x


class WhisperAudioEncoder(nn.Module):
    """Whisper-large-v3 Audio Encoder (encoder-only, no decoder).

    Architecture: Conv1d(128→1280, k=3, p=1) → GELU →
                  Conv1d(1280→1280, k=3, s=2, p=1) → GELU →
                  + positional_embedding →
                  32× ResidualAttentionBlock →
                  LayerNorm

    Attributes:
        hidden_size: 1280 (whisper-large-v3)
        max_source_positions: 1500
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        n_mels = config.audio_n_mels
        n_ctx = config.audio_n_ctx
        n_state = config.audio_n_state
        n_head = config.audio_n_head
        n_layer = config.audio_n_layer

        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self._positional_embedding = sinusoids(n_ctx, n_state)
        self.blocks = [
            WhisperResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)
        ]
        self.ln_post = nn.LayerNorm(n_state)

        self._hidden_size = n_state
        self._max_source_positions = config.audio_max_source_positions

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def max_source_positions(self) -> int:
        return self._max_source_positions

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (B, T, n_mels) mel spectrogram in NLC format.

        Returns:
            (B, T', n_state) encoder hidden states where T' = T // 2.
        """
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        # x: (B, T', n_state)
        pe = self._positional_embedding[: x.shape[1]]
        x = x + pe
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x
