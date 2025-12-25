import torch
from diffusers.models import AutoencoderKL


def create_vae():
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    return vae
