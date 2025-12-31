import copy
import gc
import os
import re
import time
from typing import cast

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
import wandb
from einops import rearrange, repeat
from omegaconf import DictConfig
from torch import Tensor
from torch.amp import autocast
from torch.distributed import destroy_process_group, init_process_group

# DDP Code
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Dataloader
from data.data import LatentShardDatasetStage2

# Gemma
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def param_groups_weight_decay(
    model: nn.Module,
    weight_decay: float,
):
    decay = []
    no_decay = []

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding)):
                no_decay.append(param)
            elif param_name == "bias":
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def ema_scheduler(decay, init_decay, step, warmup_steps):
    if step >= warmup_steps:
        return decay
    else:
        return ((decay - init_decay) / warmup_steps) * step + init_decay


@torch.no_grad()
def update_ema(model_ema, model, decay):
    # #####################
    # CHAT GPT ADDED
    # #################
    if isinstance(model, DDP):
        sd = model.module.state_dict()
    else:
        sd = model.state_dict()
    ema_sd = model_ema.state_dict()

    for k, v in sd.items():
        if v.dtype.is_floating_point:
            ema_sd[k].mul_(decay).add_(v * (1 - decay))

    return model_ema


def single_sample_fn(
    model, noise, text_tokens, text_mask, num_steps=24, num_save_steps=6
):
    if noise.dim() == 3:
        noise = noise.unsqueeze(0)
    if text_tokens.dim() == 2:
        text_tokens = text_tokens.unsqueeze(0)
    if text_mask.dim() == 1:
        text_mask = text_mask.unsqueeze(0)

    samples = generate_samples(
        model,
        noise,
        text_tokens,
        text_mask,
        num_steps=num_steps,
        num_save_steps=num_save_steps,
    )
    return samples[0]


@torch.inference_mode()
def generate_samples(model, noise, text_tokens, text_mask, num_steps=24, num_save_steps=6):
    text_tokens = text_tokens.to(device=noise.device)
    text_mask = text_mask.to(device=noise.device)
    if text_mask.dtype is not torch.bool:
        text_mask = text_mask.bool()
    save_times = torch.linspace(0.0, 1.0, num_save_steps, device=noise.device)

    def vector_field(t, y):
        t_batch = torch.full((y.shape[0],), t, device=y.device, dtype=y.dtype)
        # Sampling is called outside the training autocast context.
        # If `noise`/`y` is BF16 but model params are FP32, Conv2d will error
        # (input BF16 vs bias FP32). Run forward under autocast on CUDA and
        # cast back to the ODE state's dtype.
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

    # torchdiffeq returns a tuple when y0 is a tuple; unwrap single-state cases.
    if isinstance(sol, tuple):
        if len(sol) != 1:
            raise TypeError(
                f"Expected single-state solution tensor, got tuple of length {len(sol)}."
            )
        sol = sol[0]

    sol = cast(torch.Tensor, sol)
    return sol.permute(1, 0, 2, 3, 4)


def train(
    model,
    model_ema,
    # #####################
    # CHAT GPT ADDED
    # #################
    model_for_ema,
    gemma,
    tokenizer,
    optimizer,
    dataiter,
    vae,
    cfg,
    is_main,
    device,
    step_start=0,
):
    if cfg.wandb.enabled and is_main:
        wandb.init(
            project=cfg.wandb.project,
            config={  # optional; store hyperparameters
                "lr": cfg.train.lr,
                "batch_size": cfg.train.batch_size,
            },
            name=cfg.wandb.name,
            group="torch",
        )
        wandb.define_metric("train_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("image/*", step_metric="train_step")

    model_ema.eval()
    gemma.eval()
    vae.eval()

    scaling_factor = vae.config.scaling_factor

    validation_noise = None
    validation_text_tokens = None
    validation_text_mask = None
    validation_text = []
    if is_main:
        validation_text = [
            "A fractured concrete stairwell spirals downward inside a cylindrical shaft, its chipped edges exposing layered aggregate and rust-stained rebar. Cool directional light from a circular skylight creates sharp radial shadows, emphasizing rough textures and high-frequency surface noise. The palette is desaturated gray and ochre, with dark voids between steps fading into low-contrast black.",
            "A transparent glass cube rests on a matte black plane, containing suspended metallic spheres of varying diameters arranged in a precise lattice. Hard studio lighting produces crisp caustics, mirrored reflections, and specular highlights that refract through the cube’s beveled edges. The color scheme is minimal, dominated by clear glass, chrome silver, and deep neutral blacks.",
            "A high-speed macro view of a water droplet impacts a shallow liquid surface, forming a crown-shaped splash with thin upward jets and micro-beaded rims. Backlighting creates bright rim highlights and translucent gradients within the fluid structures, freezing fine surface tension details. The background is uniformly dark, contrasting with the pale blue-gray liquid and sharp white highlights.",
            "A dense urban intersection is captured from above, with parallel lanes of vehicles rendered as elongated streaks due to long-exposure motion blur. Sodium-vapor streetlights cast warm orange bands across asphalt textured with painted markings and oil-slick reflections. Cool blue shadows from surrounding buildings intersect the warm tones, creating high-contrast color separation across the frame.",
        ]
        validation_noise = torch.randn(
            (1, 16, 32, 32), dtype=torch.float32, device=device
        )
        validation_noise = repeat(
            validation_noise, "1 ... -> b ...", b=len(validation_text)
        )

        validation_inputs = tokenizer(
            validation_text,
            padding="max_length",
            truncation=True,
            max_length=cfg.train.max_input_length,
            return_tensors="pt",
        )
        validation_inputs = {
            "input_ids": validation_inputs["input_ids"].to(device=device),
            "attention_mask": validation_inputs["attention_mask"].to(device=device),
        }
        with torch.inference_mode():
            with autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                validation_text_tokens = gemma.encoder(
                    **validation_inputs
                ).last_hidden_state
        validation_text_mask = validation_inputs["attention_mask"].bool()

    start_time = time.time()

    for step in range(step_start, cfg.train.total_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for micro_step in range(cfg.train.grad_accum):
            try:
                batch = next(dataiter)
            except StopIteration:
                break

            # the non blocking means that the CPU can keep working while
            # loading the data onto the GPU, basically just a speed up that
            # required pin memory also to be True
            latents = batch[0].to(device=device, non_blocking=True)
            short = batch[1].to(device=device, non_blocking=True, dtype=torch.long)
            short_mask = batch[2].to(device=device, non_blocking=True, dtype=torch.long)
            long = batch[3].to(device=device, non_blocking=True, dtype=torch.long)
            long_mask = batch[4].to(device=device, non_blocking=True, dtype=torch.long)

            B = latents.shape[0]

            # short cpation ratio
            num_drop = int(B * cfg.train.cfg_p)
            if num_drop > 0:
                indx = torch.randint(
                    0,
                    B,
                    (num_drop,),
                    device=short.device,
                    dtype=torch.long,
                )
                long[indx] = short[indx]
                long_mask[indx] = short_mask[indx]

            X1 = latents.to(dtype=torch.bfloat16) * scaling_factor
            # Mean shift to 0
            X1 = X1 - cfg.train.latent_mean

            # Noise tensor
            X0 = torch.randn(X1.shape, dtype=X1.dtype, device=device)

            eps = torch.randn((B,), device=device, dtype=torch.float32)
            t = torch.sigmoid(eps).to(dtype=X1.dtype)
            t_mult = t[:, None, None, None]

            Xt = t_mult * X1 + (1 - t_mult) * X0
            V = X1 - X0

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # pyrefly:ignore
                input_ids = {
                    "input_ids": long,
                    "attention_mask": long_mask,
                }
                with torch.no_grad():
                    text_embed = gemma.encoder(**input_ids).last_hidden_state
                text_mask = long_mask.bool()
                pred = model(Xt, t, text_embed, text_mask)

            loss = F.mse_loss(V, pred) / cfg.train.grad_accum
            loss_sum += loss.detach()
            if micro_step >= cfg.train.grad_accum - 1:
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()
        optimizer.step()


        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_steps == 0:
            loss_detached = loss_sum
            dist.all_reduce(loss_detached, op=dist.ReduceOp.AVG)
            if is_main:
                wandb.log(
                    {
                        "train_step": step + 1,
                        "train/loss": loss_detached.item(),
                    }
                )
        if (step + 1) % cfg.train.every_n_ema == 0:
            decay = ema_scheduler(
                cfg.train.ema_decay, cfg.train.ema_init, step, cfg.train.ema_warmup
            )
            # #####################
            # CHAT GPT ADDED
            # #################
            model_ema = update_ema(model_ema, model_for_ema, decay=decay)
        if (step + 1) % cfg.train.every_n_checkpoint == 0 and is_main:
            save_dir = os.path.abspath(cfg.train.checkpoint_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"model_{step + 1}.pt")
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "model_ema": model_ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                save_path,
            )
            # Remove anything more than 5 checkpoints (keep highest step numbers).
            # Only considers files matching model_<int>.pt in the checkpoint dir.
            checkpoints = []
            for fname in os.listdir(save_dir):
                m = re.fullmatch(r"model_(\d+)\.pt", fname)
                if m is None:
                    continue
                checkpoints.append((int(m.group(1)), os.path.join(save_dir, fname)))

            if len(checkpoints) > 5:
                checkpoints.sort(key=lambda x: x[0])  # ascending by step
                to_delete = checkpoints[: max(0, len(checkpoints) - 5)]
                for _, path in to_delete:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass

        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_image == 0 and is_main:
            print("------------------------------")
            print(f" VAE device: {next(vae.parameters()).device}")
            print("------------------------------")

            model.eval()
            model_ema.eval()
            gemma.eval()
            vae.eval()
            assert (
                validation_noise is not None
                and validation_text_tokens is not None
                and validation_text_mask is not None
            )
            with torch.inference_mode():
                generated_latents_ema = generate_samples(
                    model_ema,
                    validation_noise,
                    validation_text_tokens,
                    validation_text_mask,
                )
                generated_latents_model = generate_samples(
                    model,
                    validation_noise,
                    validation_text_tokens,
                    validation_text_mask,
                )
            generated_latents_ema = (
                generated_latents_ema + cfg.train.latent_mean
            ) / vae.config.scaling_factor
            generated_latents_model = (
                generated_latents_model + cfg.train.latent_mean
            ) / vae.config.scaling_factor

            # Ensure VAE decode inputs match the VAE's device/dtype.
            vae_param = next(vae.parameters())
            vae_device = vae_param.device
            vae_dtype = vae_param.dtype
            generated_latents_ema = generated_latents_ema.to(
                device=vae_device, dtype=vae_dtype
            )
            generated_latents_model = generated_latents_model.to(
                device=vae_device, dtype=vae_dtype
            )
            decode_start_time = time.time()

            # [B,T,C,H,W]
            generated_latents_ema = generated_latents_ema.reshape(-1, 16, 32, 32)
            generated_latents_model = generated_latents_model.reshape(-1, 16, 32, 32)
            with torch.inference_mode():
                decoded_images_ema = vae.decode(generated_latents_ema)[0]
                decoded_images_model = vae.decode(generated_latents_model)[0]
            decoded_images_ema = rearrange(
                decoded_images_ema,
                "(b t) c h w -> c (b h) (t w)",
                b=validation_noise.shape[0],
            )
            decoded_images_model = rearrange(
                decoded_images_model,
                "(b t) c h w -> c (b h) (t w)",
                b=validation_noise.shape[0],
            )

            decoded_images_ema = (
                decoded_images_ema.permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
                * 255.0
            )
            decoded_images_model = (
                decoded_images_model.permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
                * 255.0
            )
            decoded_images = np.concatenate(
                [decoded_images_ema, decoded_images_model], axis=1
            ).astype(np.uint8)

            print(f"Time to decode latents: {(time.time() - decode_start_time):.4f} s.")
            end_time = time.time()
            if start_time is not None:
                wandb.log(
                    {f"train/{cfg.train.every_n_image} time": end_time - start_time}
                )

            caption = "Left: EMA | Right: Regular.\n" + "\n".join(
                [f"Row {i}: {text}" for i, text in enumerate(validation_text)]
            )
            wandb.log({"image/examples": wandb.Image(decoded_images, caption=caption)})
            start_time = time.time()

            model.train()

        if step >= cfg.train.total_steps:
            break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # setup DDP
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # world_size and rank are redundant on one node but good practice
    is_main: bool = True if rank == 0 else False

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl", init_method="env://"
    )  # nvidia collective communications library

    device = torch.device("cuda", local_rank)

    data_dir = cfg.data.data_dir
    shard_names = sorted(os.listdir(data_dir))  # For determinism accross nodes
    shard_paths = [os.path.join(data_dir, shard_name) for shard_name in shard_names]
    dataset = LatentShardDatasetStage2(shard_paths=shard_paths, seed=cfg.data.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.data.num_workers > 0),  # Since infinite dataloader
        drop_last=True,
    )
    dataiter = iter(dataloader)

    model = hydra.utils.instantiate(cfg.model)

    params = param_groups_weight_decay(model, cfg.train.weight_decay)
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, eps=1e-15)

    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-xl-xl-ul2")
    gemma = AutoModelForSeq2SeqLM.from_pretrained(
        "google/t5gemma-xl-xl-ul2",
        device_map={"": device},
    )
    gemma = gemma.model
    gemma = gemma.to(dtype=torch.bfloat16)

    step_start = 0

    if os.path.exists(cfg.train.checkpoint_init_dir):
        resume_dict = torch.load(
            os.path.abspath(cfg.train.checkpoint_init_dir), map_location="cpu"
        )
        model.load_state_dict(resume_dict)

        model_ema = copy.deepcopy(model)
    else:
        raise ValueError("No init model given.")

    # #####################
    # CHAT GPT ADDED
    # #################
    model_for_ema = model

    vae = hydra.utils.instantiate(cfg.vae)
    vae = vae.to(device="cpu")
    if is_main:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        vae = vae.to(device=device)
        print(f"VAE number of parameters: {sum(p.numel() for p in vae.parameters())}")
        print(f"Gemma number of parameters: {sum(p.numel() for p in gemma.parameters())}")

    # move to devices
    model = model.to(device=device)
    model = torch.compile(model)
    model_ema = model_ema.to(device=device)
    model = DDP(model, device_ids=[local_rank])

    for param in model_ema.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in gemma.parameters():
        param.requires_grad = False
    model_ema.eval()
    gemma.eval()
    vae.eval()

    train(
        model,
        model_ema,
        # #####################
        # CHAT GPT ADDED
        # #################
        model_for_ema,
        gemma,
        tokenizer,
        optimizer,
        dataiter,
        vae,
        cfg,
        is_main,
        device,
        step_start,
    )

    dist.destroy_process_group()  # DDP cleanup


if __name__ == "__main__":
    main()
