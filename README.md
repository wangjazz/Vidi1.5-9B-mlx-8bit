---
license: other
license_name: vidi-license
license_link: https://huggingface.co/bytedance-research/Vidi1.5-9B
base_model: bytedance-research/Vidi1.5-9B
base_model_relation: quantized
tags:
  - mlx
  - video-understanding
  - temporal-grounding
  - multimodal
  - gemma2
  - siglip2
  - whisper
  - 8-bit
  - quantized
library_name: mlx
pipeline_tag: video-text-to-text
---

# Vidi1.5-9B MLX (8-bit Quantized)

MLX port of [ByteDance Vidi1.5-9B](https://huggingface.co/bytedance-research/Vidi1.5-9B) for Apple Silicon.

Vidi is a multimodal video temporal grounding model — give it a video and a text query, and it tells you *when* things happen.

## Features

- Full Dual Attention architecture (Gemma2 + SigLip2 + Whisper)
- Video + audio multimodal understanding
- 8-bit quantized: **11.3 GB** (vs 19.6 GB fp16)
- Runs locally on Apple Silicon Macs via [MLX](https://github.com/ml-explore/mlx)

## Architecture

```
                    ┌──────────────┐
  Video Frames ───> │ SigLip2      │──> Vision Tokens ──┐
  (384x384)         │ 27L, 1152d   │                    │  Cross-Attention
                    └──────────────┘                    │  (every layer)
                                                        v
                    ┌──────────────┐              ┌──────────────┐
  Text Tokens ────> │              │──> T2T ────> │              │──> Output
                    │  Gemma2-9B   │              │  42 Decoder   │   Timestamps
                    │  3584d       │<── T2A <──── │  Layers       │
                    └──────────────┘              └──────────────┘
                                                        ^
                    ┌──────────────┐                    │  Cross-Attention
  Audio (16kHz) ──> │ Whisper-v3   │──> Audio Tokens ──┘
  (mel spectrogram) │ 32L, 1280d   │
                    └──────────────┘
```

Key: Cross-attention **reuses** self-attention q/k/v/o weights (no extra parameters).

## Quick Start

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~12 GB unified memory for 8-bit model

```bash
pip install mlx safetensors transformers opencv-python
```

### Download Weights

Download the 8-bit quantized weights from HuggingFace:

```bash
# Option 1: huggingface-cli
huggingface-cli download wangjazz/Vidi1.5-9B-mlx-8bit --local-dir ./Vidi1.5-9B-mlx-8bit

# Option 2: git lfs
git lfs install
git clone https://huggingface.co/wangjazz/Vidi1.5-9B-mlx-8bit
```

### Run Inference

```bash
# Video temporal grounding
python -m mlx_vidi.run \
    --model-path ./Vidi1.5-9B-mlx-8bit \
    --video-path ./your_video.mp4 \
    --query "a person talking"

# Image mode
python -m mlx_vidi.run \
    --model-path ./Vidi1.5-9B-mlx-8bit \
    --image-path ./your_image.jpg \
    --query "describe this image"
```

### Python API

```python
import mlx.core as mx
from pathlib import Path
from mlx_vidi.config import ModelConfig
from mlx_vidi.generate import VidiEngine
from mlx_vidi.quantize import quantize_engine
import json

# Load
with open("./Vidi1.5-9B-mlx-8bit/config.json") as f:
    raw = json.load(f)
config = ModelConfig.from_dict(raw)

engine = VidiEngine(config)
quant = raw["quantization"]
quantize_engine(engine, bits=quant["bits"], group_size=quant["group_size"])

weights = {}
for wf in sorted(Path("./Vidi1.5-9B-mlx-8bit").glob("model-*.safetensors")):
    weights.update(mx.load(str(wf)))
engine.load_weights(list(weights.items()), strict=False)
mx.eval(engine.parameters())

# Prepare inputs (see mlx_vidi/preprocessing.py)
from mlx_vidi.preprocessing import extract_video_frames, process_images, ...

# Generate
token_ids = engine.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    mel_features=mel_features,
    audio_sizes=audio_sizes,
    max_tokens=200,
    temperature=0.0,
)
```

## Output Format

Vidi is a **temporal grounding** model. It outputs normalized timestamps (0.0-1.0):

```
Query: "During which time segments in the video can we see a person singing?"
Output: 0.21-0.22, 0.46-0.47

# For a 60s video → actual time: 12.6s-13.2s, 27.6s-28.2s
```

## Convert Your Own Weights

If you want to convert from the original PyTorch weights:

```bash
# 1. Download original weights
huggingface-cli download bytedance-research/Vidi1.5-9B --local-dir ./Vidi1.5-9B

# 2. Convert to MLX fp16
python -m mlx_vidi.convert_weights \
    --input-dir ./Vidi1.5-9B \
    --output-dir ./Vidi1.5-9B-mlx \
    --dtype float16

# 3. Quantize to 8-bit
python -m mlx_vidi.quantize \
    --input-dir ./Vidi1.5-9B-mlx \
    --output-dir ./Vidi1.5-9B-mlx-8bit \
    --bits 8 --group-size 64
```

## Project Structure

```
mlx_vidi/
├── config.py              # ModelConfig dataclass
├── model.py               # Dual Attention Gemma2 (T2T + T2V + T2A cross-attention)
├── vision_encoder.py      # SigLip2-so400m-patch14-384
├── audio_encoder.py       # Whisper-large-v3 encoder
├── projectors.py          # VidiRMSNorm, MMProjector, LearnablePosEmbd, Conv2DPool, etc.
├── generate.py            # VidiEngine + KVCache + autoregressive generation
├── preprocessing.py       # Video/audio/text preprocessing (OpenCV + transformers)
├── convert_weights.py     # PyTorch safetensors → MLX safetensors
├── quantize.py            # 4-bit / 8-bit quantization
└── run.py                 # CLI entry point
```

## Quantization Comparison

Tested on a 60-second music video:

| Version | Size | "guitar played" | "audience/crowd" | "face close-up" |
|---------|------|-----------------|------------------|-----------------|
| fp16    | 19.6 GB | 0:11-0:12 | 7 segments | 0:49-0:51 |
| **8-bit** | **11.3 GB** | **0:11-0:12** | **7 segments** | **0:49-0:51** |
| 4-bit   | 6.9 GB | 0:00-0:01 | 2 segments | 0:49-0:51 |

8-bit maintains near-identical quality to fp16 with 42% less memory.

## Acknowledgements

- [ByteDance Research](https://github.com/bytedance/vidi) — original Vidi model and paper
- [Apple MLX](https://github.com/ml-explore/mlx) — ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) / [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — reference MLX implementations

## Citation

If you use this project, please cite the original Vidi paper:

```bibtex
@article{Vidi2026vidi2.5,
    title={Vidi2.5: Large Multimodal Models for Video Understanding and Creation},
    author={Vidi Team, Chia-Wen Kuo, Chuang Huang, Dawei Du, Fan Chen,
            Fanding Lei, Feng Gao, Guang Chen, Haoji Zhang, Haojun Zhao,
            Jin Liu, Jingjing Zhuge, Lili Fang, Lingxi Zhang, Longyin Wen,
            Lu Guo, Lu Xu, Lusha Li, Qihang Fan, Rachel Deng, Shaobo Fang,
            Shu Zhang, Sijie Zhu, Stuart Siew, Weiyan Tao, Wen Zhong,
            Xiaohui Shen, Xin Gu, Ye Yuan, Yicheng He, Yiming Cui,
            Zhenfang Chen, Zhihua Wu, Zuhua Lin},
    journal={arXiv preprint arXiv:2511.19529},
    year={2026}
}
```

## License

Code: [Apache 2.0](LICENSE)

Model weights: Subject to the [original Vidi model license](https://huggingface.co/bytedance-research/Vidi1.5-9B).
