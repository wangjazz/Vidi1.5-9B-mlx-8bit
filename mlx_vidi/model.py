"""
Dual Attention Gemma2 model for Vidi1.5-9B — MLX port.

Architecture:
- 42-layer Gemma2 decoder with Dual Attention
- Each layer: T2T self-attn + T2V cross-attn + T2A cross-attn
- V2V and A2A diagonal attention for modality-internal updates
- Cross-attention reuses q/k/v/o_proj weights from self-attention

Three-way KV cache: text (KVCache), image (list), audio (list).
"""

from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .projectors import splitted_call


# ---------------------------------------------------------------------------
# Gemma2 RMSNorm (1 + weight) * rms_norm(x) — for LLM layers
# ---------------------------------------------------------------------------

class Gemma2RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


# ---------------------------------------------------------------------------
# Gemma2 MLP (gate_proj * up_proj → down_proj with gelu)
# ---------------------------------------------------------------------------

class Gemma2MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# DattnAttention — self-attn + cross-attn sharing q/k/v/o_proj
# ---------------------------------------------------------------------------

class DattnAttention(nn.Module):
    """Dual Attention module: T2T self-attention + T2V/T2A cross-attention.

    Cross-attention **reuses** the same q/k/v/o_proj as self-attention.
    Cross-attention does NOT apply RoPE and is NOT causal.
    """

    def __init__(self, args: ModelConfig):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = 1.0 / (args.query_pre_attn_scalar ** 0.5)
        self.attn_logit_softcapping = args.attn_logit_softcapping

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def _self_attention(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
    ) -> mx.array:
        """Standard T2T self-attention (with RoPE, causal mask, softcapping)."""
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries * self.scale

        # GQA: group queries
        if self.repeats > 1:
            queries = queries.reshape(B, self.n_kv_heads, self.repeats, -1, self.head_dim)
            keys = mx.expand_dims(keys, 2)
            values = mx.expand_dims(values, 2)

        scores = queries @ keys.swapaxes(-1, -2)

        # Softcapping
        scores = mx.tanh(scores / self.attn_logit_softcapping)
        scores = scores * self.attn_logit_softcapping

        # Apply causal mask
        if mask is not None:
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask

        scores = mx.softmax(scores, precise=True, axis=-1)
        output = scores @ values

        if self.repeats > 1:
            output = output.reshape(B, self.n_heads, -1, self.head_dim)

        output = output.transpose(0, 2, 1, 3).reshape(B, -1, self.n_heads * self.head_dim)
        return self.o_proj(output)

    def _cross_attention(
        self,
        x_q: mx.array,
        x_kv: mx.array,
        kv_mask: Optional[mx.array],
        cross_cache: Optional[list],
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        """T2V or T2A cross-attention.

        No RoPE. Non-causal. Uses padding mask from kv_mask.
        Returns (attn_output, value_states_flat).
        """
        B, L_q, _ = x_q.shape

        queries = self.q_proj(x_q)  # reuse self-attn weights

        # KV: use cache if available, otherwise compute and cache
        if cross_cache is not None and layer_idx < len(cross_cache):
            keys, values = cross_cache[layer_idx]
        else:
            keys = self.k_proj(x_kv)
            values = self.v_proj(x_kv)
            if cross_cache is not None:
                cross_cache.append((keys, values))

        _, L_kv, _ = keys.shape

        queries = queries.reshape(B, L_q, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L_kv, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L_kv, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # GQA expand
        if self.repeats > 1:
            keys = mx.repeat(keys, self.repeats, axis=1)
            values = mx.repeat(values, self.repeats, axis=1)

        queries = queries * self.scale

        scores = queries @ keys.swapaxes(-1, -2)

        # Softcapping
        scores = mx.tanh(scores / self.attn_logit_softcapping)
        scores = scores * self.attn_logit_softcapping

        # Apply padding mask (non-causal): kv_mask is (B, L_kv) bool
        if kv_mask is not None:
            # Expand to (B, 1, 1, L_kv)
            mask_expanded = kv_mask[:, None, None, :].astype(mx.bool_)
            scores = mx.where(mask_expanded, scores, mx.finfo(scores.dtype).min)

        scores = mx.softmax(scores, precise=True, axis=-1)
        output = scores @ values  # (B, n_heads, L_q, head_dim)

        # Also extract value_states for V2V/A2A diagonal attention
        # value_states shape: (B, n_heads, L_kv, head_dim) → (B, L_kv, n_heads*head_dim)
        value_states = values.transpose(0, 2, 1, 3).reshape(B, L_kv, -1)

        output = output.transpose(0, 2, 1, 3).reshape(B, L_q, -1)
        attn_output = self.o_proj(output)

        return attn_output, value_states

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """T2T self-attention only (standard Gemma2 path)."""
        return self._self_attention(x, mask, cache)


# ---------------------------------------------------------------------------
# DattnTransformerBlock — one decoder layer with tri-modal attention
# ---------------------------------------------------------------------------

class DattnTransformerBlock(nn.Module):
    """Dual Attention decoder layer.

    Flow per layer:
      1. T2T Self-Attention
      2. T2V Cross-Attention + V2V diagonal update (if image_embeds present)
      3. T2A Cross-Attention + A2A diagonal update (if audio_embeds present)
      4. Merge: h_text + h_image + h_audio → residual + post_attn_norm → FFN
    """

    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = args.hidden_size
        self.is_sliding = not bool(layer_idx % 2)  # even layers: sliding window
        self.sliding_window = args.sliding_window

        self.self_attn = DattnAttention(args)
        self.mlp = Gemma2MLP(args)

        self.input_layernorm = Gemma2RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.args = args

    def _feed_forward(self, x: mx.array) -> mx.array:
        """Pre-norm → MLP → post-norm + residual."""
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return residual + x

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        image_embeds: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        image_cache: Optional[list] = None,
        audio_embeds: Optional[mx.array] = None,
        audio_mask: Optional[mx.array] = None,
        audio_cache: Optional[list] = None,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[mx.array]]:

        # Fast path: no multimodal inputs → pure Gemma2 layer
        if image_embeds is None and audio_embeds is None:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + self.post_attention_layernorm(r)
            out = self._feed_forward(h)
            return out, None, None

        residual = x
        normed_x = self.input_layernorm(x)

        # --- 1. T2T Self-Attention ---
        h_text = self.self_attn(normed_x, mask, cache)

        # --- 2. T2V Cross-Attention ---
        h_image = 0.0
        if image_embeds is not None:
            use_image_cache = (image_cache is not None and self.layer_idx < len(image_cache))
            # Check if any images have content
            image_has_content = mx.sum(mx.abs(mx.sum(image_mask, axis=-1))) > 0 if image_mask is not None else True

            if image_has_content:
                # Prepare KV mask: set all-zero rows to True to avoid NaN in softmax
                _image_mask = image_mask
                if _image_mask is not None:
                    row_sum = mx.sum(_image_mask.astype(mx.float32), axis=-1)
                    all_zero_mask = (row_sum == 0)
                    # For rows with no content, set mask to all True
                    _image_mask = mx.where(
                        all_zero_mask[:, None],
                        mx.ones_like(_image_mask),
                        _image_mask,
                    )

                # Normalise image KV (skip if using cache — already normalised)
                if not use_image_cache:
                    normed_image = self.input_layernorm(image_embeds)
                else:
                    normed_image = image_embeds  # won't be used, cache has k/v

                h_img, v_states = self.self_attn._cross_attention(
                    normed_x, normed_image, _image_mask, image_cache, self.layer_idx,
                )

                # Zero out cross-attn output for batch items with no images
                if image_mask is not None:
                    row_sum = mx.sum(image_mask.astype(mx.float32), axis=-1)
                    has_image = (row_sum != 0)[:, None, None]  # (B, 1, 1)
                    h_img = h_img * has_image

                h_image = h_img

                # V2V diagonal update (only during prefill, not when using cache)
                if not use_image_cache:
                    v_update = self.self_attn.o_proj(v_states)
                    v_update = self.post_attention_layernorm(v_update)
                    image_embeds = image_embeds + v_update
                    image_embeds = self._feed_forward(image_embeds)

        # --- 3. T2A Cross-Attention ---
        h_audio = 0.0
        if audio_embeds is not None:
            use_audio_cache = (audio_cache is not None and self.layer_idx < len(audio_cache))
            audio_has_content = mx.sum(mx.abs(mx.sum(audio_mask, axis=-1))) > 0 if audio_mask is not None else True

            if audio_has_content:
                _audio_mask = audio_mask
                if _audio_mask is not None:
                    row_sum = mx.sum(_audio_mask.astype(mx.float32), axis=-1)
                    all_zero_mask = (row_sum == 0)
                    _audio_mask = mx.where(
                        all_zero_mask[:, None],
                        mx.ones_like(_audio_mask),
                        _audio_mask,
                    )

                if not use_audio_cache:
                    normed_audio = self.input_layernorm(audio_embeds)
                else:
                    normed_audio = audio_embeds

                h_aud, a_states = self.self_attn._cross_attention(
                    normed_x, normed_audio, _audio_mask, audio_cache, self.layer_idx,
                )

                if audio_mask is not None:
                    row_sum = mx.sum(audio_mask.astype(mx.float32), axis=-1)
                    has_audio = (row_sum != 0)[:, None, None]
                    h_aud = h_aud * has_audio

                h_audio = h_aud

                # A2A diagonal update
                if not use_audio_cache:
                    a_update = self.self_attn.o_proj(a_states)
                    a_update = self.post_attention_layernorm(a_update)
                    audio_embeds = audio_embeds + a_update
                    audio_embeds = self._feed_forward(audio_embeds)

        # --- 4. Merge ---
        hidden = h_text + h_image + h_audio
        hidden = residual + self.post_attention_layernorm(hidden)
        hidden = self._feed_forward(hidden)

        return hidden, image_embeds, audio_embeds


# ---------------------------------------------------------------------------
# DattnGemma2Model — full model assembly
# ---------------------------------------------------------------------------

class DattnGemma2Model(nn.Module):
    """Complete Dual Attention Gemma2 model.

    Components:
    - embed_tokens + hidden_size**0.5 scaling (Gemma2 convention)
    - 42 × DattnTransformerBlock
    - final_norm + lm_head + logit_softcapping
    """

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.hidden_scale = args.hidden_size ** 0.5
        self.final_logit_softcapping = args.final_logit_softcapping

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DattnTransformerBlock(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = Gemma2RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list] = None,
        image_embeds: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        image_cache: Optional[list] = None,
        audio_embeds: Optional[mx.array] = None,
        audio_mask: Optional[mx.array] = None,
        audio_cache: Optional[list] = None,
    ) -> mx.array:
        """Forward pass through the LLM backbone.

        Args:
            inputs: (B, L) token IDs.
            cache: list of KVCache for T2T self-attention.
            image_embeds: (B, N_img, D) encoded image features.
            image_mask: (B, N_img) bool mask.
            image_cache: list of (K, V) tuples per layer for cross-attn.
            audio_embeds: (B, N_aud, D) encoded audio features.
            audio_mask: (B, N_aud) bool mask.
            audio_cache: list of (K, V) tuples per layer for cross-attn.

        Returns:
            (B, L, D) hidden states after final norm.
        """
        h = self.embed_tokens(inputs)
        h = h * self.hidden_scale  # Gemma2 embedding scaling

        # Scale multimodal embeddings too
        if image_embeds is not None:
            image_embeds = image_embeds * self.hidden_scale
        if audio_embeds is not None:
            audio_embeds = audio_embeds * self.hidden_scale

        if cache is None:
            cache = [None] * len(self.layers)

        # Create causal mask for T2T
        mask = _create_causal_mask(h, cache[0])

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            h, image_embeds, audio_embeds = layer(
                h, mask, c,
                image_embeds, image_mask, image_cache,
                audio_embeds, audio_mask, audio_cache,
            )

        return self.norm(h)


class Model(nn.Module):
    """Top-level Vidi model (LLM backbone + lm_head + logit softcapping)."""

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DattnGemma2Model(args)
        self.final_logit_softcapping = args.final_logit_softcapping

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list] = None,
        image_embeds: Optional[mx.array] = None,
        image_mask: Optional[mx.array] = None,
        image_cache: Optional[list] = None,
        audio_embeds: Optional[mx.array] = None,
        audio_mask: Optional[mx.array] = None,
        audio_cache: Optional[list] = None,
    ) -> mx.array:
        out = self.model(
            inputs, cache,
            image_embeds, image_mask, image_cache,
            audio_embeds, audio_mask, audio_cache,
        )
        # Tied embeddings: lm_head = embed_tokens.T
        out = self.model.embed_tokens.as_linear(out)
        # Logit softcapping
        out = mx.tanh(out / self.final_logit_softcapping)
        out = out * self.final_logit_softcapping
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_causal_mask(h: mx.array, cache: Optional[Any]) -> Optional[mx.array]:
    """Create causal attention mask for T2T self-attention."""
    N = h.shape[1]
    if N == 1:
        return None

    offset = 0
    if cache is not None and hasattr(cache, "offset"):
        offset = cache.offset

    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] >= rinds[None]
    return mask
