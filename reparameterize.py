import os
import jax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as ocp
import equinox as eqx
import optax

import hydra
from omegaconf import DictConfig, OmegaConf

from einops import rearrange

from models.mmDiT.dit import DiT


def get_trainable_mask(model):
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, model)

    mask = eqx.tree_at(
        lambda m: [m.pos_embed.emb, m.time_proj.emb], mask, replace=(False, False)
    )
    return mask


def reparameterize(model, E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b):
    move_adaln = lambda m: [
        m.adaLN1_single.weight,
        m.adaLN1_single.bias,
        m.adaLN2_single.weight,
        m.adaLN2_single.bias,
    ]
    final_adaln = lambda m: [m.adaLN1.bias]
    new_embeds = lambda m: [block.adaLN for block in m.dit_blocks]
    E_list = [E[i] for i in range(len(model.dit_blocks))]

    class_baked_ada1b = ada1w @ c + ada1b
    new_model = eqx.tree_at(
        move_adaln, model, replace=(ada1w, class_baked_ada1b, ada2w, ada2b)
    )

    new_adaln1_b = adaln1w @ c + adaln1b
    new_model = eqx.tree_at(final_adaln, new_model, replace=[new_adaln1_b])
    new_model = eqx.tree_at(new_embeds, new_model, E_list)

    return new_model


def get_weight_decay_mask(model):
    mask = jax.tree_util.tree_map(eqx.is_inexact_array, model)

    mask = eqx.tree_at(lambda m: m.cond_proj.weight, mask, replace=False)
    return mask


def create_embeddings(model, cfg, key):
    key, sub_key = jr.split(key)

    noise = jax.random.normal(key, shape=(16, 32, 32), dtype=jnp.bfloat16)
    # label = jax.Array([1000], dtype=jnp.int32)

    key, sub_key = jr.split(key)

    t = 0.5
    label = 1000

    # print("printing")
    # eqx.tree_pprint(model)

    model = model

    t_embed = model.time_proj(t)
    c_embed = model.cond_proj(label)

    S = dict()

    for i, dit_block in enumerate(model.dit_blocks):
        logits1 = dit_block.adaLN1(t_embed + c_embed)
        S[i] = dit_block.adaLN2(logits1)

    Sbar = S[0]

    E = dict()

    for i in range(len(model.dit_blocks) - 1):
        E[i + 1] = Sbar - S[i + 1]
        if i == 0:
            E[0] = jnp.zeros_like(Sbar)

    return (
        E,
        c_embed,
        model.adaLN1.weight,
        model.adaLN1.bias,
        model.dit_blocks[0].adaLN1.weight,
        model.dit_blocks[0].adaLN1.bias,
        model.dit_blocks[0].adaLN2.weight,
        model.dit_blocks[0].adaLN2.bias,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    key = jr.PRNGKey(cfg.train.seed)

    ckpt_dir = os.path.abspath(cfg.train.checkpoint_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir, options=options, item_names=("state", "model_ema", "dataset")
    )

    new_ckpt_dir = os.path.abspath("0new_model")
    new_checkpoint_manager = ocp.CheckpointManager(
        new_ckpt_dir,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, create=True, enable_async_checkpointing=False
        ),
        item_names=("model",),
    )

    # this is for model loading
    cpu_sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])

    # note:
    # make sure you use class conditioned config, not the new config.
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
    if checkpoint_manager.latest_step() is not None:

        state = (model, opt_state)

        abstract_state = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=cpu_sharding),
            state,
        )

        del state
        del model
        del opt_state

        print(
            f"Restoring previous trianing at step {checkpoint_manager.latest_step()}/{cfg.train.total_steps}"
        )
        restore_args = ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
        )
        restored = checkpoint_manager.restore(
            checkpoint_manager.latest_step(), args=restore_args
        )

        state = restored["state"]
        model = state[0]

        del restored
        new_config: dict = OmegaConf.to_container(  # pyrefly:ignore
            cfg.model, resolve=True
        )
        new_config.pop("_target_", None)
        new_config.pop("num_classes", None)
        new_config["text_dim"] = 2048
        new_model = DiT(**new_config, key=key)
        E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b = create_embeddings(
            model, cfg, key
        )
        new_model = reparameterize(
            new_model, E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b
        )

        args = ocp.args.Composite(model=ocp.args.StandardSave(new_model))
        new_checkpoint_manager.save(0, args=args)
        new_checkpoint_manager.wait_until_finished()

    else:
        print("Error: No model files found.")


if __name__ == "__main__":
    main()
