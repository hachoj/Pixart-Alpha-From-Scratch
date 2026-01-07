import functools
import os
import sys
from typing import cast

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchdiffeq
from einops import rearrange
from torch.amp import autocast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from models_fp8.mmDiT.dit import DiT
from models.vae import create_vae


@functools.lru_cache(maxsize=1)
def _load_model(model_path):
    mmDiT_config: dict[str, int] = {
        "in_dim": 16,
        "dim": 1152,
        "cond_dim": 256,
        "text_dim": 2048,
        "num_heads": 16,
        "mlp_ratio": 4,
        "num_blocks": 28,
        "patch_size": 2,
        "base_image_size": 32,
    }
    model = DiT(**mmDiT_config)

    if os.path.exists(model_path):
        state_dict = torch.load(
            os.path.abspath(model_path), map_location="cpu"
        )
    else:
        raise ValueError("Invalid model path")

    model.load_state_dict(state_dict)
    return model


@functools.lru_cache(maxsize=1)
def _load_vae():
    vae = create_vae()
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


@functools.lru_cache(maxsize=1)
def _load_gemma(device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-xl-xl-ul2")
    gemma = AutoModelForSeq2SeqLM.from_pretrained(
        "google/t5gemma-xl-xl-ul2",
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    gemma = gemma.model
    gemma.eval()
    for param in gemma.parameters():
        param.requires_grad = False
    return gemma, tokenizer


@torch.inference_mode()
def generate_samples(model, noise, text_tokens, text_mask, num_steps=24, num_save_steps=24):
    text_tokens = text_tokens.to(device=noise.device)
    text_mask = text_mask.to(device=noise.device)
    if text_mask.dtype is not torch.bool:
        text_mask = text_mask.bool()
    save_times = torch.linspace(0.0, 1.0, num_save_steps, device=noise.device)

    def vector_field(t, y):
        t_batch = torch.full((y.shape[0],), t, device=y.device, dtype=y.dtype)
        with autocast(
            device_type=y.device.type,
            dtype=torch.bfloat16,
            enabled=(y.device.type == "cuda"),
        ):
            out = model(y, t_batch, text_tokens, text_mask)
        return out.to(y.dtype)

    sol = torchdiffeq.odeint(
        vector_field,
        noise,
        save_times,
        method="euler",
        options={"step_size": 1.0 / num_steps},
    )

    if isinstance(sol, tuple):
        if len(sol) != 1:
            raise TypeError(
                f"Expected single-state solution tensor, got tuple of length {len(sol)}."
            )
        sol = sol[0]

    sol = cast(torch.Tensor, sol)
    return sol.permute(1, 0, 2, 3, 4)


def generate(
    model_path: str,
    prompt: str,
    seed: int | None = None,
    device: str = "cuda",
    max_input_length: int = 512,
):
    """
    Generate an image from a text prompt.
    
    Args:
        model_path: Path to the model checkpoint
        prompt: Text prompt describing the image to generate
        seed: Random seed for noise generation (optional)
        device: Device to run generation on (default: "cuda")
        max_input_length: Maximum length for tokenized input (default: 512)
    
    Returns:
        tuple: (timesteps, final_image)
            - timesteps: torch.Tensor of shape [24, 3, 256, 256] with images at each timestep
            - final_image: torch.Tensor of shape [3, 256, 256] with final generated image
    """
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Load model, VAE, and text encoder
    model = _load_model(model_path)
    model = model.to(device=device)
    model.eval()
    
    vae = _load_vae()
    vae = vae.to(device=device)
    
    gemma, tokenizer = _load_gemma(device=device)
    
    # Tokenize and encode text
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    text_inputs = {
        "input_ids": text_inputs["input_ids"].to(device=device),
        "attention_mask": text_inputs["attention_mask"].to(device=device),
    }
    
    with torch.inference_mode():
        with autocast(
            device_type=device,
            dtype=torch.bfloat16,
            enabled=(device == "cuda"),
        ):
            text_tokens = gemma.encoder(**text_inputs).last_hidden_state
    
    text_mask = text_inputs["attention_mask"].bool()
    
    # Generate noise
    noise = torch.randn((1, 16, 32, 32), dtype=torch.float32, device=device)
    
    # Generate latents at 24 timesteps
    generated_latents = generate_samples(
        model,
        noise,
        text_tokens,
        text_mask,
        num_steps=24,
        num_save_steps=24,
    )  # Shape: [1, 24, 16, 32, 32]
    
    # Denormalize latents
    LATENT_MEAN = 0.177456
    scaling_factor = vae.config.scaling_factor  # pyrefly:ignore
    generated_latents = (generated_latents + LATENT_MEAN) / scaling_factor
    
    # Prepare for VAE decode
    vae_param = next(vae.parameters())
    vae_device = vae_param.device
    vae_dtype = vae_param.dtype
    generated_latents = generated_latents.to(device=vae_device, dtype=vae_dtype)
    
    # Decode all timesteps: [1, 24, 16, 32, 32] -> [24, 16, 32, 32]
    generated_latents = generated_latents.squeeze(0)
    decoded_images = vae.decode(generated_latents)[0]  # Shape: [24, 3, 256, 256]
    
    # Clamp to [0, 1]
    decoded_images = decoded_images.clamp(0, 1)
    
    # Return all timesteps and final image
    timesteps = decoded_images  # [24, 3, 256, 256]
    final_image = decoded_images[-1]  # [3, 256, 256]
    
    return timesteps, final_image