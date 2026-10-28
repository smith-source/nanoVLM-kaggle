# Kaggle training guide

1. Enable **GPU T4 ×2** and **Internet** in your Kaggle Notebook.
2. Run the commands from [README](README.md); start with a single GPU.
3. The default model uses SigLIP2 and SmolLM2 pretrained backbones. Downloading model weights and streaming dataset requires Internet and suitable credentials/permissions.
4. If you hit CUDA OOM, reduce `lm_max_length` and `max_sample_length` together to 512, leave per-GPU batch size 1 and inspect the full error trace.
5. After single GPU succeeds, try `torchrun --standalone --nproc_per_node=2 train.py --no_log_wandb`.
6. Save checkpoints from `checkpoints/` before a Kaggle session expires.

This adaptation is not tested on a Kaggle T4 yet. It retains model architecture and trains the decoder/projector while freezing the vision backbone. DDP replicates the model rather than sharding it. License and attribution: [upstream nanoVLM](https://github.com/huggingface/nanoVLM).
