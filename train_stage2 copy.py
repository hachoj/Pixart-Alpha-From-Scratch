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

import diffrax
from einops import rearrange
import torch

from typing import Union, Sequence
from jaxtyping import Array, Float, Bool, Int


def encode_with_t5gemma_encoder(
    texts: Union[str, Sequence[str]],
    *,
    model=None,
    params,
    tokenizer,
    max_input_length: int = 256,
    model_sharding: jshard.NamedSharding,
    data_sharding: jshard.NamedSharding,
    return_on_host: bool = True,
    forward_fn=None,
) -> tuple[Float[Array, "batch max_length d_model"], Bool[Array, "batch max_length"]]:
    if isinstance(texts, str):
        texts = [texts]

    if hasattr(tokenizer, "special_tokens"):
        pad_id = tokenizer.special_tokens.PAD
    else:
        raise ValueError("Expected PAD token to exist")

    padded_batch = []
    for text in texts:
        token_ids = tokenizer.encode(text)[:max_input_length]
        padded_batch.append(token_ids + [pad_id] * (max_input_length - len(token_ids)))

    input_tokens: Int[Array, "batch max_length"] = jnp.asarray(
        padded_batch, dtype=jnp.int32
    )

    # Padding-based mask.
    inputs_mask: Bool[Array, "batch max_length"] = input_tokens != pad_id

    params_s = jax.device_put(params, model_sharding)
    tokens_s = jax.device_put(input_tokens, data_sharding)
    mask_s = jax.device_put(inputs_mask, data_sharding)

    if forward_fn is not None:
        encoder_last_hidden = forward_fn(params_s, tokens_s, mask_s)
    else:
        if model is None:
            raise ValueError("model must be provided if forward_fn is not provided")

        def _encoder_last_hidden(params, tokens, mask):
            encoder_acts = model.apply(
                {"params": params},
                tokens=tokens,
                inputs_mask=mask,
                method=model.compute_encoder_activations,
            )
            return encoder_acts.activations[-1]  # [B, L, d_model]

        forward = jax.jit(
            _encoder_last_hidden,
            in_shardings=(model_sharding, data_sharding, data_sharding),
            out_shardings=data_sharding,
        )
        encoder_last_hidden = forward(params_s, tokens_s, mask_s)

    if return_on_host:
        encoder_last_hidden = jax.device_get(encoder_last_hidden)
        inputs_mask = jax.device_get(inputs_mask)
    return encoder_last_hidden, inputs_mask


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


def single_sample_fn(model, noise, text_tokens, token_mask):
    def vector_field(t, y, args):
        model, text_tokens, token_mask = args
        return model(y, t, text_tokens, token_mask)

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
        args=(model, text_tokens, token_mask),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=save_times),
        max_steps=num_steps + 2,
    )
    return sol.ys


@eqx.filter_jit
def generate_samples(
    model, noise, text_tokens, token_mask, model_sharding, data_sharding
):
    model = eqx.filter_shard(model, model_sharding)
    noise, text_tokens, token_mask = eqx.filter_shard(
        (noise, text_tokens, token_mask), data_sharding
    )
    return jax.vmap(single_sample_fn, in_axes=(None, 0, 0, 0))(
        model, noise, text_tokens, token_mask
    )


@eqx.filter_value_and_grad
def compute_grads(params, static, x_t, v, t, text_tokens, token_mask):
    model = eqx.combine(params, static)

    logits = jax.vmap(model)(x_t, t, text_tokens, token_mask)
    loss = optax.losses.l2_loss(logits, v)
    return jnp.mean(loss)


@eqx.filter_jit(
    donate="all",
)
def step_model(
    state,
    optimizer,
    x_t,
    v,
    t,
    text_tokens,
    token_mask,
    model_sharding,
    data_sharding,
    micro_step,
    grad_accum_steps,
    gradients,
):
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

    x_t, v, t, text_tokens, token_mask = eqx.filter_shard(
        (x_t, v, t, text_tokens, token_mask), data_sharding
    )
    x_t, v, t, text_tokens = (
        x_t.astype(jnp.bfloat16),
        v.astype(jnp.bfloat16),
        t.astype(jnp.bfloat16),
        text_tokens.astype(jnp.bfloat16),
    )

    loss, grads = compute_grads(
        params_bf16, static_bf16, x_t, v, t, text_tokens, token_mask
    )

    # Accumulate gradients
    gradients = eqx.apply_updates(gradients, grads)

    # use bf16 grads to update fp32 weights since bf16 should be stable enough
    # without the use of grad scaling
    def update_step(operands):
        params_fp32, opt_state, gradients = operands
        grads = jax.tree_util.tree_map(lambda g: g / grad_accum_steps, gradients)
        updates, opt_state = optimizer.update(grads, opt_state, params=params_fp32)
        params_fp32 = eqx.apply_updates(params_fp32, updates)

        model_fp32 = eqx.combine(params_fp32, static_fp32)

        zeros = jax.tree_util.tree_map(jnp.zeros_like, gradients)
        return (model_fp32, opt_state), loss, zeros

    def skip_step(operands):
        params_fp32, opt_state, gradients = operands
        model_fp32 = eqx.combine(params_fp32, static_fp32)
        return (model_fp32, opt_state), loss, gradients

    condition: bool = micro_step >= grad_accum_steps - 1
    operands = (params_fp32, opt_state, gradients)
    return jax.lax.cond(condition, update_step, skip_step, operands)


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
    t5gemma_model,
    t5gemma_params,
    preset,
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

    validation_captions = [
        "A neon-lit vending machine in a dark alley.",
        "A chrome skull resting on a velvet pillow.",
        "The rain-slicked pavement of the neon metropolis reflects the holographic advertisements towering above, casting a kaleidoscope of electric blues and violent magentas onto the street. A lone figure in a transparent raincoat stands under a flickering awning, their face obscured by the glow of a datapad. Steam rises from a street vendor’s stall nearby, mingling with the dense fog that clings to the base of the skyscrapers. Drones buzz overhead like angry insects, their red navigation lights cutting through the gloom.",
        "An ancient oak tree, its bark twisted into spirals of silver and grey, dominates the center of a twilight glade. Bioluminescent mushrooms in shades of teal and violet cling to its massive roots, casting an ethereal upward glow. The air is filled with floating pollen that glimmers like gold dust in the fading light. In the background, jagged crystal spires rise from the mist, piercing the purple sky where three pale moons hang low. A stream of crystal-clear water winds through the foreground, slipping over smooth, moss-covered stones.",
        "The workshop is a cluttered labyrinth of brass gears, ticking clockwork mechanisms, and scattered blueprints stained with grease. A magnifying glass, mounted on an articulated brass arm, distorts the view of a tiny, mechanical beetle resting on the workbench. Dust motes dance in the shaft of warm, amber light streaming through a circular window, illuminating the particles of sawdust suspended in the air. Shelves line the walls, packed with glass jars containing preserved insects and spare springs.",
        "A colossal space station shaped like a ring world creates a silhouette against the burning orange curve of a gas giant. Tiny shuttles leave trails of white exhaust as they approach the docking bays, looking like specks of dust against the massive scale of the station. The station’s lights twinkle like diamond dust, contrasting with the deep, velvet black of the surrounding void. A nebula in the distance swirls with hues of deep crimson and indigo, providing a dramatic backdrop to the industrial rigidity of the metal structure.",
        "A warrior clad in ceremonial armor stands in a snowy courtyard, the metal plates etched with intricate floral patterns that catch the cold winter light. A fur cloak, heavy and textured with frost, drapes over their shoulders. Their eyes are sharp and piercing, reflecting the gray sky, while a scar runs faintly across their cheek. The background is a blur of falling snowflakes and dark pine trees, creating a soft bokeh effect that isolates the figure. The breath of the warrior is visible as a wisp of white vapor.",
        "A floating island made of melting clocks and marble staircases drifts through a sky of liquid clouds. The staircases lead nowhere, twisting into loops and spirals that defy gravity. A grand piano sits on the edge of a precipice, its keys pouring out water instead of sound, creating a waterfall that cascades into the abyss below. The lighting is surreal, with two light sources casting shadows in opposing directions—one warm and golden, the other cool and teal. Giant butterflies with wings made of stained glass flutter around the piano.",
    ]

    # Shard T5 params across devices (Data Parallel / FSDP-style) to save memory
    t5_sharding = model_sharding
    t5gemma_params = jax.device_put(t5gemma_params, t5_sharding)

    def _t5_forward(params, tokens, mask):
        encoder_acts = t5gemma_model.apply(
            {"params": params},
            tokens=tokens,
            inputs_mask=mask,
            method=t5gemma_model.compute_encoder_activations,
        )
        return encoder_acts.activations[-1]

    t5_forward_jit = jax.jit(
        _t5_forward,
        in_shardings=(t5_sharding, data_sharding, data_sharding),
        out_shardings=data_sharding,
    )

    validation_tokens, validation_masks = encode_with_t5gemma_encoder(
        validation_captions,
        model=None,
        params=t5gemma_params,
        tokenizer=preset.tokenizer,
        max_input_length=cfg.train.max_input_length,
        model_sharding=t5_sharding,
        data_sharding=data_sharding,
        return_on_host=False,  # Keep them on device/sharded
        forward_fn=t5_forward_jit,
    )

    validation_noise = jax.random.normal(key, shape=(1, 16, 32, 32), dtype=jnp.bfloat16)
    validation_noise = jnp.repeat(validation_noise, repeats=8, axis=0)
    validation_noise = eqx.filter_shard(validation_noise, data_sharding)

    start_time = time.time()

    model = state[0]
    mask = get_trainable_mask(model)
    params, _ = eqx.partition(model, mask)
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    loss = 0

    for step in range(step_start, cfg.train.total_steps):
        for micro_step in range(cfg.train.gradient_accum):
            batch = next(data_iterator)

            latents = batch["latent"]
            short_captions = batch["short_caption"]
            long_captions = batch["long_caption"]

            key, sub_key = jr.split(key)

            indices = jax.random.randint(
                key,
                shape=(int(len(short_captions) * cfg.train.short_caption_percent),),
                minval=0,
                maxval=len(short_captions),
            )

            captions = long_captions
            captions[indices] = short_captions[indices]

            text_tokens, masks = encode_with_t5gemma_encoder(
                captions,
                model=None,
                params=t5gemma_params,
                tokenizer=preset.tokenizer,
                max_input_length=cfg.train.max_input_length,
                model_sharding=t5_sharding,
                data_sharding=data_sharding,
                return_on_host=False,
                forward_fn=t5_forward_jit,
            )

            latents = jax.device_put(latents, data_sharding)
            text_tokens = jax.device_put(text_tokens, data_sharding)
            masks = jax.device_put(masks, data_sharding)

            # with jax sharding, this still actually prints the global batch size
            B = latents.shape[0]

            key, sub_key = jr.split(key)

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

            state, loss, grads = step_model(
                state,
                optimizer,
                Xt,
                X1 - X0,
                t,
                text_tokens,
                masks,
                model_sharding,
                data_sharding,
                jnp.asarray(micro_step),
                cfg.train.gradient_accum,
                grads,
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
                validation_tokens,
                validation_masks,
                model_sharding,
                data_sharding,
            )

            generated_latents_ema = generated_latents_ema + cfg.train.latent_mean
            generated_latents_ema = generated_latents_ema / vae.config.scaling_factor
            jax.block_until_ready(generated_latents_ema)

            generated_latents_model = generate_samples(
                model,
                validation_noise,
                validation_tokens,
                validation_masks,
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

            caption_text = "EMA | Regular\n" + "\n".join(
                [f"Row {i}: {c}" for i, c in enumerate(validation_captions)]
            )
            wandb.log(
                {"image/examples": wandb.Image(decoded_images, caption=caption_text)}
            )
            start_time = time.time()

        if step >= cfg.train.total_steps:
            break

    return state


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    from jax._src.interpreters.partial_eval import Val
    import os
    import jax
    import jax.numpy as jnp
    from pathlib import Path
    from typing import Any, Sequence, Union
    from jaxtyping import Array, Bool, Float, Int
    import jax.sharding as jshard

    # This was some HPC issue that I had no idea what it was
    # Thanks Gemini 3.0 Pro for fixing it lol
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        ca_bundle = Path(conda_prefix) / "ssl" / "cacert.pem"
        if ca_bundle.exists():
            os.environ.setdefault("SSL_CERT_FILE", str(ca_bundle))
            os.environ.setdefault("CURL_CA_BUNDLE", str(ca_bundle))
            os.environ.setdefault("REQUESTS_CA_BUNDLE", str(ca_bundle))
            print("Using CA bundle:", ca_bundle)
        else:
            print("Conda CA bundle not found at:", ca_bundle)
    else:
        print("CONDA_PREFIX not set; leaving SSL cert settings unchanged.")

    CKPT_DIR = Path("/home/chojnowski.h/weishao/chojnowski.h/JaxFM/t5gemma")
    assert CKPT_DIR.exists(), f"Checkpoint folder not found: {CKPT_DIR}"
    print("Using checkpoint:", CKPT_DIR)

    from gemma import gm
    from gemma.research import t5gemma

    preset = t5gemma.T5GemmaPreset.GEMMA2_XL_XL
    t5gemma_model = preset.config.make("transformer")

    t5gemma_params = gm.ckpts.load_params(CKPT_DIR)

    if "decoder" in t5gemma_params:
        del t5gemma_params["decoder"]  # pyrefly:ignore

    # Cast T5 params to bfloat16 for faster inference
    t5gemma_params = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x,
        t5gemma_params,
    )

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

    schedule = hydra.utils.instantiate(cfg.optim)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad),
        optax.adamw(
            learning_rate=schedule,
            eps=1e-15,
            weight_decay=cfg.train.weight_decay,
        ),
    )
    opt_state = optimizer.init(params)

    # restoration logic
    step_start = 0
    if not cfg.train.is_restore:
        ckpt_dir_new = os.path.abspath(cfg.train.checkpoint_init_dir)
        options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
        initial_mngr = ocp.CheckpointManager(
            ckpt_dir_new, options=options, item_names=("model",)
        )

        abstract_model = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=cpu_sharding),
            model,
        )

        del model

        print("Loading reparameterized model...")
        restore_args = ocp.args.Composite(
            model=ocp.args.StandardRestore(abstract_model),
        )
        restored = initial_mngr.restore(0, args=restore_args)

        model = restored["model"]

        model = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, model_sharding) if eqx.is_array(x) else x,
            model,
        )

        # Ensures optimizer state matches the restored parameters.
        trainable_mask = get_trainable_mask(model)
        params, _ = eqx.partition(model, trainable_mask)
        opt_state = optimizer.init(params)

        state = (model, opt_state)
        model_ema = jax.tree_util.tree_map(
            lambda x: jnp.copy(x) if eqx.is_inexact_array(x) else x, model
        )

        state = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, model_sharding) if eqx.is_array(x) else x,
            state,
        )
        model_ema = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, model_sharding) if eqx.is_array(x) else x,
            model_ema,
        )

        step_start = 0

        del restored
    elif checkpoint_manager.latest_step() is not None and cfg.train.is_restore:
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
            model_ema=ocp.args.StandardRestore(abstract_model),
            dataset=grain.PyGrainCheckpointRestore(data_iterator),
        )
        restored = checkpoint_manager.restore(
            checkpoint_manager.latest_step(), args=restore_args
        )

        state = restored["state"]
        model_ema = restored["model_ema"]

        state = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, model_sharding) if eqx.is_array(x) else x,
            state,
        )
        model_ema = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, model_sharding) if eqx.is_array(x) else x,
            model_ema,
        )

        step_start = checkpoint_manager.latest_step()

        del restored
    else:
        return ValueError("Some initial point should be given.")

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
        t5gemma_model,
        t5gemma_params,
        preset,
        step_start,
    )


if __name__ == "__main__":
    main()
