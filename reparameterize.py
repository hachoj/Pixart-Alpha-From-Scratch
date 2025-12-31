import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

from models.mmDiT.dit import DiT


@torch.no_grad()
def reparameterize(model, E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b):
    class_baked_ada1b = c @ ada1w.T + ada1b
    class_baked_ada1b = class_baked_ada1b.squeeze(0)

    # --- AdaLN that feeds into other dit block ---
    model.adaLN1_single.weight.copy_(ada1w)
    model.adaLN1_single.bias.copy_(class_baked_ada1b)
    model.adaLN2_single.weight.copy_(ada2w)
    model.adaLN2_single.bias.copy_(ada2b)

    # --- AdaLN for final output bake in class ---
    new_adaln1_b = c @ adaln1w.T + adaln1b
    new_adaln1_b = new_adaln1_b.squeeze(0)

    model.adaLN1.bias.copy_(new_adaln1_b)

    # --- AdaLN embedding inside each DiT block ---
    for i, block in enumerate(model.dit_blocks):
        block.adaLNembed.copy_(E[i].squeeze(0))

    return model


@torch.no_grad()
def create_embeddings(model):
    t = torch.tensor(0.5).float()
    label = torch.tensor(1000).long()
    t = t.unsqueeze(0)
    label = label.unsqueeze(0)

    t_embed = model.time_proj(t)
    c_embed = model.cond_proj(label)

    S = dict()

    for i, dit_block in enumerate(model.dit_blocks):
        logits1 = dit_block.adaLN1(t_embed + c_embed)
        logits1 = F.silu(logits1)
        S[i] = dit_block.adaLN2(logits1)

    Sbar = S[0]

    E = dict()

    for i in range(len(model.dit_blocks) - 1):
        E[i + 1] = S[i + 1] - Sbar
        if i == 0:
            E[0] = torch.zeros_like(Sbar)

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
    """
    Make sure you are using the hydra confg
    for the stage 1 training not stage 2
    """
    print("Creating model...")
    model = hydra.utils.instantiate(cfg.model)

    print("Loading pretrained weights...")
    resume_dict = torch.load(
        os.path.abspath("checkpoints/model_300000.pt"), map_location="cpu"
    )

    model.load_state_dict(resume_dict["model_ema"])

    print("Creating reparameterized model...")
    new_config: dict = OmegaConf.to_container(cfg.model, resolve=True)  # pyrefly:ignore
    new_config.pop("_target_", None)
    new_config.pop("num_classes", None)
    new_config["text_dim"] = 2048
    new_model = DiT(**new_config)
    E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b = create_embeddings(model)

    print("Reparameterizing model...")
    new_model = reparameterize(
        new_model, E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b
    )

    print("Saving reparameterized model...")
    save_path = os.path.abspath("checkpoints/new_reparameterized_model.pt")
    torch.save(new_model.state_dict(), save_path)
    print(f"Saved reparameterized model to {save_path}")


if __name__ == "__main__":
    main()
