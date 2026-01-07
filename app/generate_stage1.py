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

from models_fp8.ccDiT.dit import DiT
from models.vae import create_vae


@functools.lru_cache(maxsize=1)
def _load_model(model_path):
    ccDiT_config: dict[str, int] = {
        "in_dim": 16,
        "dim": 1152,
        "cond_dim": 256,
        "num_heads": 16,
        "mlp_ratio": 4,
        "num_blocks": 28,
        "patch_size": 2,
        "num_classes": 1001,
        "base_image_size": 32,
    }
    model = DiT(**ccDiT_config)

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


@torch.inference_mode()
def generate_samples(model, noise, labels, num_steps=24, num_save_steps=24):
    labels = labels.to(device=noise.device)
    save_times = torch.linspace(0.0, 1.0, num_save_steps, device=noise.device)

    def vector_field(t, y):
        t_batch = torch.full((y.shape[0],), t, device=y.device, dtype=y.dtype)
        with autocast(
            device_type=y.device.type,
            dtype=torch.bfloat16,
            enabled=(y.device.type == "cuda"),
        ):
            out = model(y, t_batch, labels)
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
    class_label: int,
    seed: int | None = None,
    device: str = "cuda",
):
    """
    Generate an image from a class label.
    
    Args:
        model_path: Path to the model checkpoint
        class_label: ImageNet-1K class label (0-1000 inclusive)
        seed: Random seed for noise generation (optional)
        device: Device to run generation on (default: "cuda")
    
    Returns:
        tuple: (timesteps, final_image)
            - timesteps: torch.Tensor of shape [24, 3, 256, 256] with images at each timestep
            - final_image: torch.Tensor of shape [3, 256, 256] with final generated image
    """
    if not 0 <= class_label <= 1000:
        raise ValueError(f"class_label must be in range [0, 1000], got {class_label}")
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Load model and VAE
    model = _load_model(model_path)
    model = model.to(device=device)
    model.eval()
    
    vae = _load_vae()
    vae = vae.to(device=device)
    
    # Generate noise
    noise = torch.randn((1, 16, 32, 32), dtype=torch.float32, device=device)
    label = torch.tensor([class_label], device=device, dtype=torch.long)
    
    # Generate latents at 24 timesteps
    generated_latents = generate_samples(
        model,
        noise,
        label,
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