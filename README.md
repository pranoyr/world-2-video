# World-2-Video 🚧 (WIP)

> **Note**: This repository is currently a Work In Progress.

## Overview

World-2-Video is an experimental generative video model that leverages the **Cosmos 3 VAE** and **T5** as its robust backbone, paired with **LapFlow** (Laplacian Pyramid Flow Matching) for the actual video generation process.

### Architecture Idea
- **Backbone**: We utilize the `nvidia/Cosmos3-Nano` Video VAE to heavily compress raw 5D video frames into rich continuous latent representations (producing 16-channel video latents). The text embeddings are provided continuously by a T5-v1.1-small encoder. 
- **Generation Strategy**: Instead of standard diffusion, we use `LapFlowDiT`, which has been heavily extended to support 5D spatial-temporal inputs. LapFlow dynamically matches velocities across Laplacian pyramid scales, resolving global attention across both Time and Space natively!

## Sample Input / Output

Below is an example of what passing dummy inputs through our unified backbone looks like:

### Input
- **Video Input**: A batch of RGB video clips. 
- **Text Input**: Continuous text embeddings.

```python
# Dummy video (Batch, Channels, Time, Height, Width)
videos = torch.randn(2, 3, 17, 256, 256).to(device)

# Dummy continuous text embeddings from T5
text_embeds = torch.randn(2, 512).to(device)
```

### Forward Pass (Trainer & LapFlow)

```python
from lapflow import LapFlow, LapFlowDiT

# ... Initialize Cosmos VAE and model ...

loss = lap_flow((videos, text_embeds))
print(f"Forward pass successful! Loss: {loss.item()}")
```

### Expected Output Print
When executing the forward pass to train the flow model, you should see the successfully reshaped latent generations being optimized smoothly:

```text
Loading ACTUAL Cosmos VAE (Forcing past the shape mismatch)...
Loading T5 Text Encoder...
Video shape: torch.Size([2, 3, 17, 256, 256])
Text embed shape: torch.Size([2, 512])
Running forward pass...
[1] loss: 0.252
[2] loss: 1.002
[3] loss: 0.251
[4] loss: 0.251
...
```
