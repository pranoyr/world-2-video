# World-2-Video

### Cosmos Backend
Uses Nvidia's **Cosmos VAE** (`nvidia/Cosmos3-Nano`) to compress video frames into smaller 16-channel images (latents). This makes it faster and uses less memory.

### Lapflow Multiscale Joint Attention
Learns how pixels move by looking at **space (width/height)** and **time (frames)** together. "Multiscale" means it processes at different image sizes (coarse to fine) to capture both large movements and small details.

### How to Train
Just open your terminal and run:
```bash
python train_w2v.py
```
It will start training a `LapFlowDiT` model on a dummy video dataset for 100,000 steps using a batch size of 8.
