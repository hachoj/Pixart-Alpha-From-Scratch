import copy
import inspect
import os
import sys
import time
from typing import List, Optional

import gradio as gr
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from train_stage1 import generate_samples

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
NUM_STEPS = 24


def _status_html(message: str, kind: str = "info") -> str:
    return f"<div class='status {kind}'>{message}</div>"


def _resolve_checkpoint_path(cfg: DictConfig) -> str:
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        ckpt_path = OmegaConf.select(cfg, "train.resume_path")
    if not ckpt_path:
        return ""
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    return os.path.abspath(ckpt_path)


def _load_models(cfg: DictConfig, device: torch.device):
    model = hydra.utils.instantiate(cfg.model)
    model_ema = copy.deepcopy(model)

    ckpt_path = _resolve_checkpoint_path(cfg)
    if not ckpt_path:
        raise ValueError(
            "Checkpoint path missing. Pass ckpt_path=/path/to/model.pt "
            "or set train.resume_path in the config."
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    if "model_ema" in checkpoint:
        model_ema.load_state_dict(checkpoint["model_ema"], strict=True)

    model = model.to(device=device).eval()
    model_ema = model_ema.to(device=device).eval()

    for param in model.parameters():
        param.requires_grad = False
    for param in model_ema.parameters():
        param.requires_grad = False

    sampling_model = model_ema if "model_ema" in checkpoint else model
    return model, model_ema, sampling_model, ckpt_path


def _load_vae(cfg: DictConfig, device: torch.device):
    vae = hydra.utils.instantiate(cfg.vae)
    vae = vae.to(device=device).eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae


def _validate_integer(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    try:
        if not float(value).is_integer():
            return None
    except (TypeError, ValueError):
        return None
    return int(value)


def _decode_latents_to_frames(
    vae: torch.nn.Module, latents: torch.Tensor, cfg: DictConfig
) -> List[Image.Image]:
    latents = (latents + cfg.train.latent_mean) / vae.config.scaling_factor

    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)

    steps = latents.shape[1]
    latents = latents.reshape(-1, latents.shape[2], latents.shape[3], latents.shape[4])
    with torch.inference_mode():
        decoded = vae.decode(latents)[0]
    decoded = decoded.reshape(1, steps, decoded.shape[1], decoded.shape[2], decoded.shape[3])

    frames = []
    for step_idx in range(steps):
        frame = (
            decoded[0, step_idx]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .float()
            .cpu()
            .numpy()
            * 255.0
        )
        frames.append(Image.fromarray(frame.astype(np.uint8)))
    return frames


def _save_gif(frames: List[Image.Image], output_dir: str) -> str:
    if not frames:
        raise ValueError("No frames available to build GIF.")
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"stage1_sample_{stamp}_{os.getpid()}.gif")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
    )
    return out_path


CSS = """
:root {
  --m3-primary: #0b6bcb;
  --m3-primary-variant: #0856a6;
  --m3-surface: #f7f7fb;
  --m3-card: #ffffff;
  --m3-outline: #d6dae2;
  --m3-text: #1c1b1f;
  --m3-muted: #5f6368;
}
body, .gradio-container {
  font-family: "Manrope", "DM Sans", "Source Sans 3", sans-serif;
  background:
    radial-gradient(1200px 600px at 10% -10%, #e9f2ff 0%, rgba(233,242,255,0) 60%),
    radial-gradient(900px 500px at 90% 0%, #f3f9f5 0%, rgba(243,249,245,0) 55%),
    #f5f6fb;
  color: var(--m3-text);
}
#app-shell {
  max-width: 1200px;
  margin: 0 auto;
}
#header {
  background: linear-gradient(135deg, #0b6bcb, #2bb673);
  color: white;
  border-radius: 18px;
  padding: 20px 24px;
  margin-bottom: 18px;
  box-shadow: 0 12px 24px rgba(11, 107, 203, 0.2);
}
.section-card {
  background: var(--m3-card);
  border: 1px solid var(--m3-outline);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
}
.section-title {
  font-weight: 600;
  font-size: 1.05rem;
  color: var(--m3-text);
  margin-bottom: 8px;
}
#generate-btn {
  background: var(--m3-primary);
  border: none;
  color: white;
  font-weight: 600;
  padding: 12px 18px;
  border-radius: 12px;
  box-shadow: 0 10px 20px rgba(11, 107, 203, 0.25);
}
#generate-btn:hover {
  background: var(--m3-primary-variant);
}
#status .status {
  background: #eef2ff;
  border: 1px solid #d6dcff;
  color: #1e3a8a;
  padding: 10px 12px;
  border-radius: 10px;
  font-size: 0.95rem;
}
#status .status.error {
  background: #fff1f2;
  border-color: #fecdd3;
  color: #9f1239;
}
#status .status.success {
  background: #ecfdf3;
  border-color: #bbf7d0;
  color: #166534;
}
"""


def _strip_defaults(cfg: DictConfig) -> DictConfig:
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(cfg_dict, dict) and "defaults" in cfg_dict:
        cfg_dict.pop("defaults")
    return OmegaConf.create(cfg_dict)


def _load_group_config(config_dir: str, group: str, name: str) -> DictConfig:
    path = os.path.join(config_dir, group, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def _parse_default_item(item):
    if isinstance(item, str):
        if item == "_self_":
            return None, None
        if "/" in item:
            group, name = item.split("/", 1)
            return group, name
        return None, None
    if isinstance(item, DictConfig) and len(item) == 1:
        group = next(iter(item.keys()))
        name = item[group]
        return group, name
    if isinstance(item, dict) and len(item) == 1:
        group, name = next(iter(item.items()))
        return group, name
    return None, None


def _apply_overrides(
    cfg: DictConfig, config_dir: str, groups: List[str], overrides: List[str]
) -> DictConfig:
    group_overrides = {}
    dotlist = []
    for raw in overrides:
        item = raw[1:] if raw.startswith("+") else raw
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key in groups:
            group_overrides[key] = value
        else:
            dotlist.append(item)

    for group, name in group_overrides.items():
        cfg[group] = _load_group_config(config_dir, group, name)

    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def _compose_config(
    config_dir: str, config_name: str, overrides: List[str]
) -> DictConfig:
    base_path = os.path.join(config_dir, f"{config_name}.yaml")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Config not found: {base_path}")
    base_cfg = OmegaConf.load(base_path)
    defaults = base_cfg.get("defaults", [])
    cfg = OmegaConf.create()
    base_no_defaults = _strip_defaults(base_cfg)

    groups = []
    inserted_self = False
    for item in defaults:
        if item == "_self_":
            cfg = OmegaConf.merge(cfg, base_no_defaults)
            inserted_self = True
            continue
        group, name = _parse_default_item(item)
        if group is None or name is None:
            continue
        groups.append(group)
        cfg[group] = _load_group_config(config_dir, group, name)

    if not inserted_self:
        cfg = OmegaConf.merge(cfg, base_no_defaults)

    cfg = _apply_overrides(cfg, config_dir, groups, overrides)
    return cfg


def _parse_cli(argv: List[str]):
    config_dir = os.path.join(REPO_ROOT, "configs")
    config_name = "config"
    overrides = []
    config_path = None

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg in ("--config-path", "--config_path"):
            if idx + 1 < len(argv):
                config_path = argv[idx + 1]
                idx += 2
                continue
        if arg in ("--config-dir", "--config_dir"):
            if idx + 1 < len(argv):
                config_dir = argv[idx + 1]
                idx += 2
                continue
        if arg in ("--config-name", "--config_name"):
            if idx + 1 < len(argv):
                config_name = argv[idx + 1]
                idx += 2
                continue

        if arg.startswith("config_path="):
            config_path = arg.split("=", 1)[1]
        elif arg.startswith("config_dir="):
            config_dir = arg.split("=", 1)[1]
        elif arg.startswith("config_name="):
            config_name = arg.split("=", 1)[1]
        else:
            overrides.append(arg)
        idx += 1

    if config_path:
        config_path = os.path.expanduser(config_path)
        if not os.path.isabs(config_path):
            config_path = os.path.join(REPO_ROOT, config_path)
        config_dir = os.path.dirname(config_path)
        config_name = os.path.splitext(os.path.basename(config_path))[0]
    else:
        if not os.path.isabs(config_dir):
            config_dir = os.path.join(REPO_ROOT, config_dir)

    return config_dir, config_name, overrides


def _resolve_gradio_kwargs():
    blocks_kwargs = {"title": "Stage 1 Sampler", "elem_id": "app-shell"}
    launch_kwargs = {}

    launch_params = inspect.signature(gr.Blocks.launch).parameters
    if "css" in launch_params:
        launch_kwargs["css"] = CSS
    else:
        blocks_params = inspect.signature(gr.Blocks.__init__).parameters
        if "css" in blocks_params:
            blocks_kwargs["css"] = CSS

    return blocks_kwargs, launch_kwargs


def _queue_demo(demo: gr.Blocks) -> None:
    queue_params = inspect.signature(demo.queue).parameters
    if "concurrency_count" in queue_params:
        demo.queue(concurrency_count=1)
    elif "default_concurrency_limit" in queue_params:
        demo.queue(default_concurrency_limit=1)
    else:
        demo.queue()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config_dir, config_name, overrides = _parse_cli(sys.argv[1:])
    cfg = _compose_config(config_dir, config_name, overrides)

    model = model_ema = sampling_model = vae = None
    load_error = None
    try:
        model, model_ema, sampling_model, ckpt_path = _load_models(cfg, device)
        vae = _load_vae(cfg, device)
    except Exception as exc:
        ckpt_path = _resolve_checkpoint_path(cfg)
        load_error = str(exc)
    ckpt_display = ckpt_path if ckpt_path else "unset"

    blocks_kwargs, launch_kwargs = _resolve_gradio_kwargs()
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            "# PixArt-Alpha Stage 1 Sampler\n"
            "Generate samples with the Stage 1 model using the same validation sampling path.",
            elem_id="header",
        )

        with gr.Row():
            with gr.Column(scale=4, min_width=320):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("Inputs", elem_classes="section-title")
                    class_id = gr.Number(
                        label="ImageNet class id (0-1000)",
                        precision=0,
                        value=2,
                    )
                    seed = gr.Number(
                        label="Seed (optional)",
                        precision=0,
                        value=None,
                    )
                    gr.Markdown(
                        f"Checkpoint: `{ckpt_display}`",
                        elem_classes="meta",
                    )
                    initial_status = (
                        _status_html("Ready.")
                        if load_error is None
                        else _status_html(f"Error: {load_error}", kind="error")
                    )
                    status = gr.HTML(initial_status, elem_id="status")
                    generate_btn = gr.Button(
                        "Generate",
                        variant="primary",
                        elem_id="generate-btn",
                        interactive=(load_error is None),
                    )

            with gr.Column(scale=8, min_width=420):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("Outputs", elem_classes="section-title")
                    gif_output = gr.Image(
                        label="Denoising animation (24 steps)",
                        type="filepath",
                    )
                    final_output = gr.Image(
                        label="Final image",
                        type="pil",
                    )

        def disable_button():
            return gr.update(interactive=False), _status_html("Generating...")

        def enable_button():
            return gr.update(interactive=True)

        def generate(class_id_value, seed_value, progress=gr.Progress()):
            if load_error is not None or sampling_model is None or vae is None:
                return None, None, _status_html(
                    "Error: model or VAE failed to load. Check the checkpoint path.",
                    kind="error",
                )
            start_time = time.time()
            class_id_int = _validate_integer(class_id_value)
            if class_id_int is None:
                return None, None, _status_html(
                    "Error: class id must be an integer between 0 and 1000.",
                    kind="error",
                )
            if not (0 <= class_id_int <= 1000):
                return None, None, _status_html(
                    "Error: class id must be in the range [0, 1000].",
                    kind="error",
                )

            seed_int = _validate_integer(seed_value)
            if seed_value is not None and seed_int is None:
                return None, None, _status_html(
                    "Error: seed must be an integer or left blank.",
                    kind="error",
                )

            try:
                progress(0, desc="Preparing noise")
                generator = None
                if seed_int is not None:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(seed_int)

                noise = torch.randn(
                    (1, 16, 32, 32),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                labels = torch.tensor(
                    [class_id_int],
                    device=device,
                    dtype=torch.long,
                )

                progress(0.15, desc="Sampling 24 steps")
                with torch.inference_mode():
                    latents = generate_samples(
                        sampling_model,
                        noise,
                        labels,
                        num_steps=NUM_STEPS,
                        num_save_steps=NUM_STEPS,
                    )

                progress(0.65, desc="Decoding frames")
                frames = _decode_latents_to_frames(vae, latents, cfg)

                progress(0.85, desc="Building GIF")
                gif_path = _save_gif(
                    frames,
                    output_dir=os.path.join(REPO_ROOT, "outputs", "gradio"),
                )
                final_image = frames[-1]

            except Exception as exc:
                return None, None, _status_html(
                    f"Error: {exc}",
                    kind="error",
                )

            elapsed = time.time() - start_time
            seed_msg = f"{seed_int}" if seed_int is not None else "random"
            return (
                gif_path,
                final_image,
                _status_html(
                    f"Done in {elapsed:.2f}s. Seed: {seed_msg}.",
                    kind="success",
                ),
            )

        generate_btn.click(
            fn=disable_button,
            outputs=[generate_btn, status],
            queue=False,
        ).then(
            fn=generate,
            inputs=[class_id, seed],
            outputs=[gif_output, final_output, status],
        ).then(
            fn=enable_button,
            outputs=[generate_btn],
            queue=False,
        )

    _queue_demo(demo)
    demo.launch(**launch_kwargs, share=True)


if __name__ == "__main__":
    main()
