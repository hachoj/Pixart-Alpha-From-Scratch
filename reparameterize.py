import os
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from models.mmDiT.dit import DiT

def reparameterize(model, E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b):
    class_baked_ada1b = ada1w @ c + ada1b

    # --- AdaLN that feeds into other dit block ---
    model.adaLN1_single.weight = ada1w
    model.adaLN1_single.bias = class_baked_ada1b
    model.adaLN2_single.weight = ada2w
    model.adaLN2_single.bias = ada2b

    # --- AdaLN for final output bake in class ---
    new_adaln1_b = adaln1w @ c + adaln1b

    model.adaLN1.bias = new_adaln1_b

    # --- AdaLN embedding inside each DiT block ---
    for i, block in enumerate(model.dit_blocks):
        block.adaLNembed = E[i]

    return model


def create_embeddings(model):
    t = 0.5
    label = 1000

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
    model = hydra.utils.instantiate(cfg.model)

    resume_dict = torch.load(
        os.path.abspath("final_checkpint_stage_1.pt"), map_location="cpu"
    )

    model.load_state_dict(resume_dict["model"])

    new_config: dict = OmegaConf.to_container(  # pyrefly:ignore
        cfg.model, resolve=True
    )
    new_config.pop("_target_", None)
    new_config.pop("num_classes", None)
    new_config["text_dim"] = 2048
    new_model = DiT(**new_config)
    E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b = create_embeddings(
        model
    )
    new_model = reparameterize(
        new_model, E, c, adaln1w, adaln1b, ada1w, ada1b, ada2w, ada2b
    )

    save_path = os.path.abspath("new_reparameterized_model.pt")
    torch.save(new_model.state_dict(), save_path)
    print(f"Saved reparameterized model to {save_path}")


if __name__ == "__main__":
    main()
