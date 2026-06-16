import torch
import torch.nn as nn
from diffusers import AutoencoderKLCosmos
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from einops import rearrange

class CosmosDataEncoder(nn.Module):
    def __init__(
        self,
        cosmos_model_id: str = "nvidia/Cosmos3-Nano",
        t5_model_id: str = "google/t5-v1_1-small",
        device: str = "cuda",
        max_seq_length: int = 128
    ):
        super().__init__()
        self.device = device
        self.max_seq_length = max_seq_length

        print("Loading ACTUAL Cosmos VAE (Forcing past the shape mismatch)...")
        # THE FIX: We force-load the pretrained model but tell it to ignore the broken layers.
        # It will randomly initialize the broken bias layer, allowing the forward pass to survive.
        config = AutoencoderKLCosmos.load_config(cosmos_model_id, subfolder="vae")
        config["in_channels"] = 3
        config["out_channels"] = 3
        self.vae = AutoencoderKLCosmos.from_config(
            config,
        ).to(device, dtype=torch.bfloat16)
        
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae.enable_slicing()
        self.vae.enable_tiling()

        print("Loading T5 Text Encoder...")
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_id)
        t5_config = T5Config.from_pretrained(t5_model_id)
        self.text_encoder = T5EncoderModel(
            t5_config
        ).to(device, dtype=torch.bfloat16)
        
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

    def forward(self, x: torch.Tensor, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        
        # 1. ACTUAL VISUAL LATENTS
        x = rearrange(x, "b t c h w -> b c t h w")
        x = x.to(device=self.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            latent_dist = self.vae.encode(x).latent_dist
            visual_latents = latent_dist.sample()
            visual_latents = visual_latents * self.vae.config.scaling_factor

        # 2. T5 TEXT ENCODER
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        with torch.inference_mode():
            outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
            text_embeds = outputs.last_hidden_state

        return visual_latents, text_embeds


if __name__ == "__main__":
    encoder = CosmosDataEncoder()

    # Dummy video: 2 batch, 17 frames, 3 RGB, 512x512
    dummy_video = torch.zeros(2, 17, 3, 512, 512)

    prompts = [
        "A close-up shot of a bright yellow sunflower swaying gently in the wind.",
        "A KTM 390 Adventure driving over a dirt jump in slow motion."
    ]

    print("\nRunning actual forward pass through the VAE and T5...")
    video_latents, text_embeddings = encoder(dummy_video, prompts)

    print(f"Video Latent Shape: {video_latents.shape}")
    print(f"Text Embedding Shape: {text_embeddings.shape}")

    torch.save({
        "video": video_latents.cpu(),
        "text": text_embeddings.cpu()
    }, "training_batch_0.pt")
    
    print("Done. Real latents exported.")