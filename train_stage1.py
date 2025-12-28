import copy
import gc
import os
import time

import hydra
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

from data.data import LatentShardDatasetStage1

# Dataloader



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


def update_ema(model_ema, model, decay):
    sd = model.module.state_dict()
    ema_sd = model_ema.state_dict()

    for k, v in sd.items():
        if v.dtype.is_floating_point:
            ema_sd[k].mul_(decay).add_(v * (1 - decay))


# def single_sample_fn(model, noise, label):
#     def vector_field(t, y, args):
#         model, label = args
#         return model(y, t, label)

#     term = diffrax.ODETerm(vector_field)

#     solver = diffrax.Euler()

#     num_steps = 24
#     stepsize_controller = diffrax.ConstantStepSize()
#     save_times = jnp.linspace(0.0, 1.0, 6)

#     sol = diffrax.diffeqsolve(
#         term,
#         solver,
#         t0=0.0,
#         t1=1.0,
#         dt0=1.0 / num_steps,
#         y0=noise,
#         args=(model, label),
#         stepsize_controller=stepsize_controller,
#         saveat=diffrax.SaveAt(ts=save_times),
#         max_steps=num_steps + 2,
#     )
#     return sol.ys


# @eqx.filter_jit
# def generate_samples(model, noise, labels, model_sharding, data_sharding):
#     return jax.vmap(single_sample_fn, in_axes=(None, 0, 0))(model, noise, labels)


def train(
    model,
    model_ema,
    optimizer,
    dataiter,
    vae,
    cfg,
    is_main,
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
        )
        wandb.define_metric("train_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("image/*", step_metric="train_step")

    scaling_factor = vae.config.scaling_factor

    if is_main:
        validation_noise = torch.randn((8, 16, 32, 32), dtype=torch.bfloat16)
        # ImageNet-1K labels:
        # 2   = great white shark
        # 316 = praying mantis
        # 418 = hot air balloon
        # 551 = espresso maker
        # 804 = snowplow
        # 981 = volcano
        # 984 = scuba diver
        validation_labels = torch.tensor([2, 316, 418, 551, 804, 981, 984, 1000])

    start_time = time.time()

    for step in range(step_start, cfg.train.total_steps):
        try:
            batch = next(dataiter)
        except StopIteration:
            break

        latents = batch["latent"]
        labels = batch["label"]

        B = latents.shape[0]

        # classifier free guidience
        indx = torch.randint(B, (int(B * cfg.train.cfg_p),), dtype=torch.bfloat16)
        labels[indx] == 1000

        # Convert from bit viewed stored int16
        X1 = (
            torch.tensor(latents, dtype=torch.int16).view(torch.bfloat16)
            * scaling_factor
        )

        # Mean shift to 0
        X1 = X1 - cfg.train.latent_mean

        # Noise tensor
        X0 = torch.randn(X1.shape, dtype=X1.dtype)

        eps = torch.randn((B))
        t = F.sigmoid(eps)
        t_mult = t[:, None, None, None]

        Xt = t_mult * X1 + (1 - t_mult) * X0
        V = X1 - X0

        optimizer.zero_grad(set_t_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # pyrefly:ignore
            pred = model(Xt, t, labels)

        loss = F.mse_loss(V, pred)
        loss.backward()
        optimizer.step()

        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_steps == 0 and is_main:
            loss_detached = loss.detach()
            dist.all_reduce(loss_detached, op=dist.ReduceOp.AVG)
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
            save_path = os.path.abspath(cfg.train.checkpoint_dir)
            save_path = os.path.join(save_path, f"model_{step}")
            torch.save(
                {
                    model.module.state_dict(),
                    model_ema.state_dict(),
                    optimizer.state_dict(),
                    step,
                },
                save_path,
            )
        # if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_image == 0 and is_main:
        #     jax.block_until_ready(state)
        #     end_time = time.time()

        #     generated_latents_ema = generate_samples(
        #         model_ema,
        #         validation_noise,
        #         validation_labels,
        #     )

        #     generated_latents_ema = generated_latents_ema + cfg.train.latent_mean
        #     generated_latents_ema = generated_latents_ema / vae.config.scaling_factor
        #     jax.block_until_ready(generated_latents_ema)

        #     generated_latents_model = generate_samples(
        #         model,
        #         validation_noise,
        #         validation_labels,
        #     )
        #     generated_latents_model = generated_latents_model + cfg.train.latent_mean
        #     generated_latents_model = (
        #         generated_latents_model / vae.config.scaling_factor
        #     )

        #     generated_latents_ema = jax.device_get(generated_latents_ema)
        #     generated_latents_model = jax.device_get(generated_latents_model)

        #     decode_start_time = time.time()
        #     generated_latents_ema = np.array(generated_latents_ema, copy=True)
        #     generated_latents_model = np.array(generated_latents_model, copy=True)
        #     generated_latents_ema = (
        #         torch.from_numpy(generated_latents_ema)
        #         .to("cpu")
        #         .to(dtype=torch.bfloat16)
        #     )
        #     generated_latents_model = (
        #         torch.from_numpy(generated_latents_model)
        #         .to("cpu")
        #         .to(dtype=torch.bfloat16)
        #     )

        #     # [B,T,C,H,W]
        #     generated_latents_ema = generated_latents_ema.view(-1, 16, 32, 32)
        #     generated_latents_model = generated_latents_model.view(-1, 16, 32, 32)
        #     with torch.inference_mode():
        #         decoded_images_ema = vae.decode(generated_latents_ema)[0]
        #         decoded_images_model = vae.decode(generated_latents_model)[0]
        #     decoded_images_ema = rearrange(
        #         decoded_images_ema,
        #         "(b t) c h w -> c (b h) (t w)",
        #         b=validation_noise.shape[0],
        #     )
        #     decoded_images_model = rearrange(
        #         decoded_images_model,
        #         "(b t) c h w -> c (b h) (t w)",
        #         b=validation_noise.shape[0],
        #     )

        #     decoded_images_ema = (
        #         decoded_images_ema.permute(1, 2, 0).clip(0, 1).float().numpy() * 255.0
        #     )
        #     decoded_images_model = (
        #         decoded_images_model.permute(1, 2, 0).clip(0, 1).float().numpy() * 255.0
        #     )
        #     decoded_images = np.concatenate(
        #         [decoded_images_ema, decoded_images_model], axis=1
        #     ).astype(np.uint8)

        #     print(
        #         f"Time to decode latents fully on cpu: {(time.time() - decode_start_time):.4f} s."
        #     )
        #     if start_time is not None:
        #         wandb.log(
        #             {f"train/{cfg.train.every_n_image} time": end_time - start_time}
        #         )

        #     wandb.log(
        #         {"image/examples": wandb.Image(decoded_images, caption="EMA | Regular")}
        #     )
        #     start_time = time.time()

        if step >= cfg.train.total_steps:
            break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # setup DDP
    dist.init_process_group(backend="nccl")  # nvidia collective communications library
    local_rank = dist.get_node_local_rank()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # world_size and rank are redundant on one node but good practice
    is_main: bool = True if rank == 0 else False

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    data_dir = cfg.data.data_dir
    shard_names = os.listdir(data_dir)
    shard_paths = [os.path.join(data_dir,shard_name) for shard_name in shard_names]
    dataloader = LatentShardDatasetStage1(shard_paths=shard_paths)
    dataiter = iter(dataloader)

    model = hydra.utils.instantiate(cfg.model)

    params = param_groups_weight_decay(model, cfg.train.weight_decay)
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, eps=1e-15)

    step_start = 0

    if os.path.exists(cfg.train.resume_path) and cfg.train.is_restore:
        model_ema = copy.deepcopy(model)

        resume_dict = torch.load(os.path.abspath(cfg.train.resume_path))

        model.load_state_dict(resume_dict["model"])
        model_ema.load_state_dict(resume_dict["model_ema"])
        optimizer.load_state_dict(resume_dict["optimizer"])
        step_start = resume_dict["step"]
    else:
        model_ema = copy.deepcopy(model)

    vae = hydra.utils.instantiate(cfg.vae)

    # move to devices
    model = model.to(device=device)
    # model = torch.compile(model)
    model_ema = model_ema.to(device=device)
    model = DDP(model, device_ids=[local_rank])

    train(
        model,
        model_ema,
        optimizer,
        dataiter,
        vae,
        cfg,
        is_main,
        step_start,
    )

    dist.destroy_process_group()  # DDP cleanup


if __name__ == "__main__":
    main()
