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
from data.data import LatentShardDatasetStage1

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
    sd = model.module.state_dict()
    ema_sd = model_ema.state_dict()

    for k, v in sd.items():
        if v.dtype.is_floating_point:
            ema_sd[k].mul_(decay).add_(v * (1 - decay))

    return model_ema


def single_sample_fn(model, noise, label, num_steps=24, num_save_steps=6):
    if noise.dim() == 3:
        noise = noise.unsqueeze(0)
    if label.dim() == 0:
        label = label.unsqueeze(0)

    samples = generate_samples(
        model,
        noise,
        label,
        num_steps=num_steps,
        num_save_steps=num_save_steps,
    )
    return samples[0]


@torch.inference_mode()
def generate_samples(model, noise, labels, num_steps=24, num_save_steps=6):
    labels = labels.to(device=noise.device)
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
            out = model(y, t_batch, labels)
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

    scaling_factor = vae.config.scaling_factor

    validation_noise = None
    validation_text = ["", ""]# TODO
    if is_main:
        validation_noise = torch.randn(
            (1, 16, 32, 32), dtype=torch.float32, device=device
        )
        validation_noise = repeat(validation_noise, "1 ... -> b ...", b=8)

        # ImageNet-1K labels:
        # 2   = great white shark
        # 316 = praying mantis
        # 418 = hot air balloon
        # 551 = espresso maker
        # 804 = snowplow
        # 981 = volcano
        # 984 = scuba diver
        validation_labels = torch.tensor(
            [2, 316, 418, 551, 804, 981, 984, 1000], device=device
        )

    start_time = time.time()

    for step in range(step_start, cfg.train.total_steps):
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

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # pyrefly:ignore
            input_ids = {
                "input_ids": long,
                "attention_mask": long_mask,
            }
            text_embed = gemma.encoder(**input_ids).last_hidden_state 
            pred = model(Xt, t, text_embed)

        loss = F.mse_loss(V, pred)
        loss.backward()
        optimizer.step()

        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_steps == 0:
            loss_detached = loss.detach()
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
            model_ema = update_ema(model_ema, model, decay=decay)
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
            assert validation_noise is not None and validation_labels is not None
            with torch.inference_mode():
                generated_latents_ema = generate_samples(
                    model_ema,
                    validation_noise,
                    validation_labels,
                )
                generated_latents_model = generate_samples(
                    model,
                    validation_noise,
                    validation_labels,
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

            caption = (
                "Left: EMA | Right: Regular. Rows (top→bottom): "
                "2 great white shark; 316 praying mantis; 418 hot air balloon; "
                "551 espresso maker; 804 snowplow; 981 volcano; 984 scuba diver; "
                "1000 null"
            )
            wandb.log({"image/examples": wandb.Image(decoded_images, caption=caption)})
            start_time = time.time()

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
    dataset = LatentShardDatasetStage1(shard_paths=shard_paths, seed=cfg.data.seed)
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

    step_start = 0

    if os.path.exists(cfg.train.resume_path) and cfg.train.is_restore:
        model_ema = copy.deepcopy(model)

        resume_dict = torch.load(
            os.path.abspath(cfg.train.resume_path), map_location="cpu"
        )

        model.load_state_dict(resume_dict["model"])
        model_ema.load_state_dict(resume_dict["model_ema"])
        optimizer.load_state_dict(resume_dict["optimizer"])
        step_start = resume_dict["step"]
    else:
        model_ema = copy.deepcopy(model)

    vae = hydra.utils.instantiate(cfg.vae)
    # Keep VAE on CPU (used only for logging/decoding).
    vae = vae.to(device="cpu")
    if is_main:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        vae = vae.to(device=device)
        print(f"VAE number of parameters: {sum(p.numel() for p in vae.parameters())}")
        print(f"Gemma number of parameters: {sum(p.numel() for p in gemma.parameters())}")

    # move to devices
    model = model.to(device=device)
    # model = torch.compile(model)
    model_ema = model_ema.to(device=device)
    model = DDP(model, device_ids=[local_rank])

    for param in model_ema.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in gemma.parameters():
        param.required_grad = False
    model_ema.eval()
    gemma.eval()
    vae.eval()

    train(
        model,
        model_ema,
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
