import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

from models_fp8.mmDiT.dit import DiT as mmDiT
from models.ccDiT.dit import DiT as ccDiT
from models_fp8.ccDiT.dit import DiT as ccDiT_fp8


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
    t = torch.tensor(0.5).float().to(device='cuda')
    label = torch.tensor(1000).long().to(device='cuda')
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


def main():
    """
    Make sure you are using the hydra confg
    for the stage 1 training not stage 2
    """
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
    # cc_model = ccDiT(**ccDiT_config)
    mm_fp8_model = mmDiT(**mmDiT_config)
    cc_fp8_model = ccDiT_fp8(**ccDiT_config)

    cc_resume_dict = torch.load(
        os.path.abspath("checkpoints_stage1/transformer_engine_model.pt"), map_location="cpu"
    )

    cc_fp8_model.load_state_dict(cc_resume_dict)

    mm_fp8_model.to(device='cuda')
    cc_fp8_model.to(device='cuda')

    # save_path = os.path.abspath("checkpoints_stage1/transformer_engine_model.pt")
    # torch.save(cc_fp8_model.state_dict(), save_path)

    print("Creating reparameterized model...")
    E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b = create_embeddings(cc_fp8_model)

    print("Reparameterizing model...")
    mm_fp8_model = reparameterize(
        mm_fp8_model, E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b
    )

    print("Saving reparameterized model...")
    save_path = os.path.abspath("checkpoints_stage1/reparameterized_transformer_engine_model.pt")
    torch.save(mm_fp8_model.state_dict(), save_path)
    print(f"Saved reparameterized model to {save_path}")


if __name__ == "__main__":
    main()
    # THE NEW PLAN IS TO HAVE MULTIPLE FUNCCTIONS HERE, ONE FOR REGULAR TO TE
    # THEN THE REPARAMETERIZE, AND THE CODE SHOULD JUST BE BETTER.
