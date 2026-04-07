"""
Auxiliary modules for Vidi1.5-9B MLX port.

Includes:
- VidiRMSNorm: weight * rms_norm(x) — differs from Gemma2's (1+weight)*rms_norm(x)
- rms_norm: standalone normalize without learnable weight
- MMProjector: Linear → GELU → Linear (mlp2x_gelu)
- FractionalSinusoidalEmbedding: non-parametric sinusoidal PE
- LearnablePosEmbd: sinusoidal base + float32 MLP
- space_to_depth: pixel-unshuffle reshape
- Conv2DPool: pad + interpolate + space_to_depth (no trainable params)
- splitted_call: chunked evaluation to bound compute-graph size
"""

import math
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# RMSNorm variants
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Parameterless RMS normalisation (used for position embeddings)."""
    orig_dtype = x.dtype
    x = x.astype(mx.float32)
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + eps)
    return x.astype(orig_dtype)


class VidiRMSNorm(nn.Module):
    """Vidi-style RMSNorm: ``weight * rms_norm(x)``.

    Note: Gemma2 uses ``(1 + weight) * rms_norm(x)`` (weight init 0).
    Vidi initialises weight to *std* and multiplies directly.
    """

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        # weight will be loaded from checkpoint (init value = std)
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight * rms_norm(x, self.eps)


# ---------------------------------------------------------------------------
# MLP projector
# ---------------------------------------------------------------------------

class MMProjector(nn.Module):
    """Multi-layer projector: ``mlp{depth}x_gelu`` or ``linear``.

    For ``mlp2x_gelu``: Linear(d_mm→d_llm) → GELU → Linear(d_llm→d_llm).
    """

    def __init__(self, arch: str, d_mm: int, d_llm: int):
        super().__init__()
        if arch == "linear":
            self.layers = [nn.Linear(d_mm, d_llm)]
        elif arch.startswith("mlp"):
            import re
            m = re.match(r"^mlp(\d+)x_gelu$", arch)
            if m is None:
                raise ValueError(f"Unknown projector arch: {arch}")
            depth = int(m.group(1))
            layers = [nn.Linear(d_mm, d_llm)]
            for _ in range(1, depth):
                layers.append(nn.GELU())
                layers.append(nn.Linear(d_llm, d_llm))
            self.layers = layers
        else:
            raise ValueError(f"Unknown projector arch: {arch}")

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Position embeddings
# ---------------------------------------------------------------------------

class FractionalSinusoidalEmbedding:
    """Non-parametric sinusoidal embedding (like classic Transformer PE)."""

    def __init__(self, d: int):
        assert d % 2 == 0
        self.d = d
        # div_term: shape (d//2,)
        self.div_term = mx.exp(
            mx.arange(0, d, 2, dtype=mx.float32) * -(math.log(10000.0) / d)
        )

    def __call__(self, position: mx.array) -> mx.array:
        """position: (N,) float → returns (N, d) float32."""
        position = position.astype(mx.float32)[:, None]  # (N, 1)
        pe_sin = mx.sin(position * self.div_term)  # (N, d//2)
        pe_cos = mx.cos(position * self.div_term)  # (N, d//2)
        pe = mx.zeros((position.shape[0], self.d), dtype=mx.float32)
        # Interleave sin/cos
        pe = pe.at[:, 0::2].add(pe_sin)
        pe = pe.at[:, 1::2].add(pe_cos)
        return pe


class LearnablePosEmbd(nn.Module):
    """Learnable position embedding: sinusoidal base + float32 MLP.

    Inference only (no noise).
    """

    def __init__(self, d: int, N: int):
        super().__init__()
        self.d = d
        self.N = N
        self.embd_weights = FractionalSinusoidalEmbedding(d)
        # float32 MLP
        self.mlp = MLPFloat32(d, d)

    def __call__(self, x: mx.array, dim: int, l: Optional[int] = None) -> mx.array:
        """Return position embedding broadcastable along *dim* of *x*.

        Args:
            x: reference tensor to determine shape.
            dim: spatial dimension (0-based among the leading dims, excluding last).
            l: effective length (≤ x.shape[dim]).  Defaults to x.shape[dim].
        """
        if l is None:
            l = x.shape[dim]

        # Generate normalised positions [0 .. N-1]
        p = mx.arange(l, dtype=mx.float32)
        if l > 1:
            p = p / (l - 1) * (self.N - 1)
        else:
            p = p * 0.0  # single element → position 0

        pe = self.embd_weights(p)        # (l, d) float32
        pe = self.mlp(pe)                # (l, d) float32
        pe = pe.astype(x.dtype)

        # Pad if l < x.shape[dim]
        if l < x.shape[dim]:
            pad_size = x.shape[dim] - l
            pe = mx.concatenate([pe, mx.zeros((pad_size, self.d), dtype=pe.dtype)])
            l = x.shape[dim]

        # Reshape to be broadcastable: insert 1s except at *dim* and last axis
        ndim = x.ndim - 1  # number of "spatial" dims (excluding feature)
        shape = [1 if d != dim else l for d in range(ndim)]
        shape.append(self.d)
        pe = pe.reshape(*shape)

        return pe


class MLPFloat32(nn.Module):
    """Two-layer MLP that operates in float32 (used for position embeddings)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.layers = [nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out)]

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# space_to_depth & Conv2DPool
# ---------------------------------------------------------------------------

def space_to_depth(x: mx.array, m_size: int = 2) -> mx.array:
    """Pixel-unshuffle: (B, C, H, W) → (B, C*m², H//m, W//m).

    MLX layout is NCHW-like via manual reshape/transpose.
    Input *x* is assumed to be (B, C, H, W).
    """
    B, C, H, W = x.shape
    assert H % m_size == 0 and W % m_size == 0
    x = x.reshape(B, C, H // m_size, m_size, W // m_size, m_size)
    x = x.transpose(0, 1, 3, 5, 2, 4)  # (B, C, m, m, H//m, W//m)
    x = x.reshape(B, C * m_size * m_size, H // m_size, W // m_size)
    return x


class Conv2DPool(nn.Module):
    """Pad → bilinear interpolate → space_to_depth.  No trainable parameters."""

    def __init__(self, d_in: int, d_out: int, s_in: int, s_out: int,
                 mm_splits: int, mm_image_pool_size: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.s_in = s_in
        self.s_out = s_out
        self.mm_splits = mm_splits
        self.merge_size = mm_image_pool_size

    def __call__(self, x: mx.array, hw: tuple) -> mx.array:
        """x: (B, C, H, W) in NCHW layout.

        hw: target (H, W) *after* interpolation (must be divisible by merge_size).
             The PyTorch reference pads +1, then interpolates to hw.
             Default hw = (H+1, W+1) rounded down to even.
        """
        B, C, H, W = x.shape
        # Zero-pad +1 on right and bottom (matching PyTorch F.pad)
        x = mx.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1)])
        # Now x is (B, C, H+1, W+1)

        target_h, target_w = hw
        cur_h, cur_w = H + 1, W + 1

        # Ensure target is divisible by merge_size
        target_h = target_h - (target_h % self.merge_size)
        target_w = target_w - (target_w % self.merge_size)
        if target_h < self.merge_size:
            target_h = self.merge_size
        if target_w < self.merge_size:
            target_w = self.merge_size

        if target_h != cur_h or target_w != cur_w:
            x_nhwc = x.transpose(0, 2, 3, 1)  # (B, H+1, W+1, C)
            x_nhwc = _bilinear_resize(x_nhwc, target_h, target_w)
            x = x_nhwc.transpose(0, 3, 1, 2)  # back to NCHW

        x = space_to_depth(x, m_size=self.merge_size)
        return x


def _bilinear_resize(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Bilinear interpolation for (B, H, W, C) tensors using grid sampling.

    This reimplements ``F.interpolate(..., mode='bilinear', align_corners=False)``
    semantics from PyTorch.
    """
    B, H, W, C = x.shape
    if H == target_h and W == target_w:
        return x

    # Compute source coordinates (align_corners=False semantics)
    # dst_coord -> src_coord = (dst + 0.5) * (src_size / dst_size) - 0.5
    ys = (mx.arange(target_h, dtype=mx.float32) + 0.5) * (H / target_h) - 0.5
    xs = (mx.arange(target_w, dtype=mx.float32) + 0.5) * (W / target_w) - 0.5

    ys = mx.clip(ys, 0, H - 1)
    xs = mx.clip(xs, 0, W - 1)

    y0 = mx.floor(ys).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, H - 1)
    x0 = mx.floor(xs).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, W - 1)

    wy = (ys - y0.astype(mx.float32))[:, None]     # (target_h, 1)
    wx = (xs - x0.astype(mx.float32))[None, :]      # (1, target_w)

    # Gather corners — shape (B, target_h, target_w, C)
    top_left = x[:, y0][:, :, x0]
    top_right = x[:, y0][:, :, x1]
    bottom_left = x[:, y1][:, :, x0]
    bottom_right = x[:, y1][:, :, x1]

    # Interpolate
    top = top_left * (1 - wx)[..., None] + top_right * wx[..., None]
    bottom = bottom_left * (1 - wx)[..., None] + bottom_right * wx[..., None]
    result = top * (1 - wy)[..., None] + bottom * wy[..., None]

    return result.astype(x.dtype)


# ---------------------------------------------------------------------------
# splitted_call
# ---------------------------------------------------------------------------

def splitted_call(
    func: Callable,
    inputs: mx.array,
    num_splits: int = 1,
    dim_split: int = 0,
    hw: Optional[tuple] = None,
) -> mx.array:
    """Split *inputs* along *dim_split*, call *func* per chunk, concatenate.

    Inserts ``mx.eval()`` between chunks to keep the compute graph bounded.
    """
    if num_splits <= 1:
        if hw is not None:
            return func(inputs, hw)
        return func(inputs)

    size = inputs.shape[dim_split]
    if size <= num_splits:
        # Tile so we have enough to split
        reps = [1] * inputs.ndim
        reps[dim_split] = math.ceil(num_splits / size)
        inputs = mx.tile(inputs, reps)

    original_size = size
    # Split into chunks
    chunk_size = inputs.shape[dim_split] // num_splits
    chunks = []
    for i in range(num_splits):
        start = i * chunk_size
        end = start + chunk_size if i < num_splits - 1 else inputs.shape[dim_split]
        slices = [slice(None)] * inputs.ndim
        slices[dim_split] = slice(start, end)
        chunks.append(inputs[tuple(slices)])

    outputs = []
    for chunk in chunks:
        if hw is not None:
            o = func(chunk, hw)
        else:
            o = func(chunk)
        mx.eval(o)  # Evaluate to free intermediate compute graph
        outputs.append(o)

    result = mx.concatenate(outputs, axis=dim_split)
    # Narrow back to original size
    slices = [slice(None)] * result.ndim
    slices[dim_split] = slice(0, original_size)
    return result[tuple(slices)]
