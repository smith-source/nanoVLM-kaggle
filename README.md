# nanoVLM — Kaggle dual-T4 low-VRAM adaptation

An **unofficial** adaptation of [Hugging Face nanoVLM](https://github.com/huggingface/nanoVLM) for Kaggle's two NVIDIA T4 GPUs (about 15GB each). Adapted from upstream commit `4e0c0961846135c2217f95e54cb4c2d66eb55e42`. Original README: [README_UPSTREAM.md](README_UPSTREAM.md). Licensed under [Apache 2.0](LICENSE).

**Not GPU-tested end to end.** These conservative defaults reduce VRAM needs but may still run out of memory on some batches. Two 15GB GPUs do not make a pooled 30GB device.

## Launch on Kaggle

Turn on **GPU T4 ×2** and **Internet** in your Notebook. Then run:

```python
!git clone https://github.com/smith-source/nanoVLM-kaggle.git
%cd /kaggle/working/nanoVLM-kaggle
!pip install -q "torch<3" "numpy<3" torchvision pillow datasets huggingface-hub transformers wandb einops safetensors
!python train.py --no_log_wandb
```

Run **single GPU first**. After it succeeds, use DDP:

```python
!torchrun --standalone --nproc_per_node=2 train.py --no_log_wandb
```

## Modified settings

| Option | Original | Kaggle |
|---|---:|---:|
| Per-GPU batch size | 2 | 1 |
| Gradient accumulation | 8 | 16 |
| LM max length | 4096 | 1024 |
| Max sample length | 4096 | 1024 |
| Max image size | 2048 | 512 |
| Images per example | 4 | 1 |
| Images per packed sample | 18 | 2 |
| Vision LR | 5e-5 | 0 (frozen) |
| Validation examples | 50000 | 200 |
| WandB and external eval | on | off |

T4 training uses FP16 autocast with GradScaler (rather than unconditional BF16). Some source examples are skipped by packing limits, so this is not a reproduction of the upstream training recipe.

See [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) for troubleshooting.
