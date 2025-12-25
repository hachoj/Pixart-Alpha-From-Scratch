import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
import orbax.checkpoint as ocp
import equinox as eqx
import optax
import numpy as np
import grain.python as grain

import hydra
from omegaconf import DictConfig
import wandb
import time
import gc

import diffrax
from einops import rearrange
import torch


def ema_scheduler(decay, init_decay, step, warmup_steps):
    if step >= warmup_steps:
        return decay
    else:
        return ((decay - init_decay) / warmup_steps) * step + init_decay


@eqx.filter_jit(donate="all")
def update_ema(model_ema, model_train, decay):
    mask = get_trainable_mask(model_ema)
    params_ema, static_ema = eqx.partition(model_ema, mask)
    params_train, _ = eqx.partition(model_train, mask)

    step_size = 1.0 - decay
    params_ema = optax.incremental_update(params_train, params_ema, step_size)

    return eqx.combine(params_ema, static_ema)


def get_trainable_mask(model):
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, model)

    mask = eqx.tree_at(
        lambda m: [m.pos_embed.emb, m.time_proj.emb], mask, replace=(False, False)
    )
    return mask


def get_weight_decay_mask(model):
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, model)

    mask = eqx.tree_at(lambda m: m.cond_proj.weight, mask, replace=False)
    return mask


def single_sample_fn(model, noise, label):
    def vector_field(t, y, args):
        model, label = args
        return model(y, t, label)

    term = diffrax.ODETerm(vector_field)

    solver = diffrax.Euler()

    num_steps = 24
    stepsize_controller = diffrax.ConstantStepSize()
    save_times = jnp.linspace(0.0, 1.0, 6)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=1.0,
        dt0=1.0 / num_steps,
        y0=noise,
        args=(model, label),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=save_times),
        max_steps=num_steps + 2,
    )
    return sol.ys


@eqx.filter_jit
def generate_samples(model, noise, labels, model_sharding, data_sharding):
    model = eqx.filter_shard(model, model_sharding)
    noise, labels = eqx.filter_shard((noise, labels), data_sharding)
    return jax.vmap(single_sample_fn, in_axes=(None, 0, 0))(model, noise, labels)


@eqx.filter_value_and_grad
def compute_grads(params, static, x_t, v, t, labels):
    model = eqx.combine(params, static)

    logits = jax.vmap(model)(x_t, t, labels)
    loss = optax.losses.l2_loss(logits, v)
    return jnp.mean(loss)


@eqx.filter_jit(donate="all")
def step_model(state, optimizer, x_t, v, t, labels, model_sharding, data_sharding):
    # you shard again here for XLA efficiency, it doesn't
    # actually divide the shards into smaller "sub-shards"

    # model, opt_state = state
    state = eqx.filter_shard(state, model_sharding)
    model_fp32, opt_state = state
    mask = get_trainable_mask(model_fp32)
    params_fp32, static_fp32 = eqx.partition(model_fp32, mask)

    # cast model weights to bf16
    params_bf16 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, params_fp32
    )
    static_bf16 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, static_fp32
    )

    x_t, v, t, labels = eqx.filter_shard((x_t, v, t, labels), data_sharding)
    x_t, v, t = (
        x_t.astype(jnp.bfloat16),
        v.astype(jnp.bfloat16),
        t.astype(jnp.bfloat16),
    )

    loss, grads = compute_grads(params_bf16, static_bf16, x_t, v, t, labels)

    # use bf16 grads to update fp32 weights since bf16 should be stable enough
    # without the use of grad scaling
    updates, opt_state = optimizer.update(grads, opt_state, params=params_fp32)
    params_fp32 = eqx.apply_updates(params_fp32, updates)

    model_fp32 = eqx.combine(params_fp32, static_fp32)

    return (model_fp32, opt_state), loss


def train(
    state,
    model_ema,
    optimizer,
    data_iterator,
    vae,
    cfg,
    model_sharding,
    data_sharding,
    key,
    checkpoint_manager,
    step_start=0,
):
    if cfg.wandb.enabled:
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

    key, sub_key = jr.split(key)

    validation_noise = jax.random.normal(key, shape=(1, 16, 32, 32), dtype=jnp.bfloat16)
    validation_noise = jnp.repeat(validation_noise, repeats=8, axis=0)
    validation_labels = jnp.array([0, 5, 10, 15, 200, 500, 750, 1000])

    # shard once before the loop since it's reused
    validation_noise = eqx.filter_shard(validation_noise, data_sharding)
    validation_labels = eqx.filter_shard(validation_labels, data_sharding)

    start_time = time.time()

    for step in range(step_start, cfg.train.total_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            break

        latents = batch["latent"]
        labels = batch["label"]

        latents = jax.device_put(latents, data_sharding)
        labels = jax.device_put(labels, data_sharding)

        # with jax sharding, this still actually prints the global batch size
        B = latents.shape[0]

        key, sub_key = jr.split(key)

        # classifier free guidience
        probs = jnp.array([1 - cfg.train.cfg_p, cfg.train.cfg_p])
        mask = jr.choice(key, 2, shape=(B,), p=probs)

        labels = jnp.where(mask == 1, 1000, labels)

        # X1 inherit the sharding from latents
        X1 = jnp.array(latents, dtype=jnp.int16).view(jnp.bfloat16) * scaling_factor

        if not jnp.all(jnp.isfinite(X1)):
            print(f"Skipping step {step}: NaN detected in latents")
            continue

        X1 = X1 - cfg.train.latent_mean

        key, sub_key = jr.split(key)
        # because jax sharding treats the shape as if it were on one GIGA-GPU
        # I have to reshard this
        X0 = jax.random.normal(key, shape=X1.shape, dtype=X1.dtype)
        X0 = jax.device_put(X0, data_sharding)

        # and the same idea for X0 is needed here
        key, sub_key = jr.split(key)
        eps = jax.random.normal(key, shape=[B])
        eps = jax.device_put(eps, data_sharding)

        # t inherits the sharding from eps
        t = jax.nn.sigmoid(eps)
        t_mult = t[:, None, None, None]

        Xt = t_mult * X1 + (1 - t_mult) * X0

        state, loss = step_model(
            state, optimizer, Xt, X1 - X0, t, labels, model_sharding, data_sharding
        )
        model, _opt_state = state

        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_steps == 0:
            wandb.log(
                {
                    "train_step": step + 1,
                    "train/loss": loss,
                }
            )
        if (step + 1) % cfg.train.every_n_ema == 0:
            decay = ema_scheduler(
                cfg.train.ema_decay, cfg.train.ema_init, step, cfg.train.ema_warmup
            )
            model_ema = update_ema(model_ema, model, decay=decay)
        if (step + 1) % cfg.train.every_n_checkpoint == 0:
            save_args = ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                model_ema=ocp.args.StandardSave(model_ema),
                dataset=grain.PyGrainCheckpointSave(data_iterator),
            )
            checkpoint_manager.save(step + 1, args=save_args)
        if cfg.wandb.enabled and (step + 1) % cfg.train.every_n_image == 0:
            jax.block_until_ready(state)
            end_time = time.time()

            generated_latents_ema = generate_samples(
                model_ema,
                validation_noise,
                validation_labels,
                model_sharding,
                data_sharding,
            )

            generated_latents_ema = generated_latents_ema + cfg.train.latent_mean
            generated_latents_ema = generated_latents_ema / vae.config.scaling_factor
            jax.block_until_ready(generated_latents_ema)

            generated_latents_model = generate_samples(
                model,
                validation_noise,
                validation_labels,
                model_sharding,
                data_sharding,
            )
            generated_latents_model = generated_latents_model + cfg.train.latent_mean
            generated_latents_model = (
                generated_latents_model / vae.config.scaling_factor
            )

            generated_latents_ema = jax.device_get(generated_latents_ema)
            generated_latents_model = jax.device_get(generated_latents_model)

            decode_start_time = time.time()
            generated_latents_ema = np.array(generated_latents_ema, copy=True)
            generated_latents_model = np.array(generated_latents_model, copy=True)
            generated_latents_ema = (
                torch.from_numpy(generated_latents_ema)
                .to("cpu")
                .to(dtype=torch.bfloat16)
            )
            generated_latents_model = (
                torch.from_numpy(generated_latents_model)
                .to("cpu")
                .to(dtype=torch.bfloat16)
            )

            # [B,T,C,H,W]
            generated_latents_ema = generated_latents_ema.view(-1, 16, 32, 32)
            generated_latents_model = generated_latents_model.view(-1, 16, 32, 32)
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
                decoded_images_ema.permute(1, 2, 0).clip(0, 1).float().numpy() * 255.0
            )
            decoded_images_model = (
                decoded_images_model.permute(1, 2, 0).clip(0, 1).float().numpy() * 255.0
            )
            decoded_images = np.concatenate(
                [decoded_images_ema, decoded_images_model], axis=1
            ).astype(np.uint8)

            print(
                f"Time to decode latents fully on cpu: {(time.time() - decode_start_time):.4f} s."
            )
            if start_time is not None:
                wandb.log(
                    {f"train/{cfg.train.every_n_image} time": end_time - start_time}
                )

            wandb.log(
                {"image/examples": wandb.Image(decoded_images, caption="EMA | Regular")}
            )
            start_time = time.time()

        if step >= cfg.train.total_steps:
            break

    return state


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    devices = jax.devices()
    mesh = jshard.Mesh(devices, axis_names=("data",))

    key = jr.PRNGKey(cfg.train.seed)

    ckpt_dir = os.path.abspath(cfg.train.checkpoint_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir, options=options, item_names=("state", "model_ema", "dataset")
    )

    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("data"))
    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])

    dataloader = hydra.utils.instantiate(cfg.data)
    data_iterator = iter(dataloader)

    model = hydra.utils.instantiate(cfg.model, key=key)
    trainable_mask = get_trainable_mask(model)
    params, _ = eqx.partition(model, trainable_mask)

    wd_mask = get_weight_decay_mask(params)
    schedule = hydra.utils.instantiate(cfg.optim)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad),
        optax.adamw(
            learning_rate=schedule,
            eps=1e-15,
            weight_decay=cfg.train.weight_decay,
            mask=wd_mask,
        ),
    )
    opt_state = optimizer.init(params)

    # restoration logic
    step_start = 0
    if checkpoint_manager.latest_step() is not None and cfg.train.is_restore:

        state = (model, opt_state)

        abstract_state = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=cpu_sharding),
            state,
        )

        abstract_model = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=cpu_sharding),
            model,
        )

        del state
        del model
        del opt_state

        print(
            f"Restoring previous trianing at step {checkpoint_manager.latest_step()}/{cfg.train.total_steps}"
        )
        restore_args = ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            model_ema=ocp.args.StandardRestore(abstract_model),
            dataset=grain.PyGrainCheckpointRestore(data_iterator),
        )
        restored = checkpoint_manager.restore(
            checkpoint_manager.latest_step(), args=restore_args
        )

        state = restored["state"]
        model_ema = restored["model_ema"]

        state = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, model_sharding) if eqx.is_array(x) else x, state
        )
        model_ema = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, model_sharding) if eqx.is_array(x) else x,
            model_ema,
        )

        step_start = checkpoint_manager.latest_step()

        del restored
    else:
        model_ema = jax.tree_util.tree_map(
            lambda x: jnp.copy(x) if eqx.is_inexact_array(x) else x, model
        )
        state = (model, opt_state)
        state = eqx.filter_shard(state, model_sharding)
        model_ema = eqx.filter_shard(model_ema, model_sharding)

    vae = hydra.utils.instantiate(cfg.vae).to("cpu")

    train(
        state,
        model_ema,
        optimizer,
        data_iterator,
        vae,
        cfg,
        model_sharding,
        data_sharding,
        key,
        checkpoint_manager,
        step_start,
    )


if __name__ == "__main__":
    main()
