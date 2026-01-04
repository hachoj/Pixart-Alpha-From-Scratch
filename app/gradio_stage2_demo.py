import copy
import inspect
import os
import sys
import time
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gradio as gr
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.reprompt import reprompt as reprompt_prompt
from train_stage2 import generate_samples

NUM_STEPS = 24
TEXT_ENCODER_NAME = "google/t5gemma-xl-xl-ul2"
DEFAULT_INIT_CKPT = "/home/chojnowski.h/weishao/chojnowski.h/Pixart-Alpha/checkpoints_stage1/new_reparameterized_model.pt"


def _status_html(message: str, kind: str = "info") -> str:
    return f"<div class='status {kind}'>{message}</div>"


def _resolve_checkpoint_path(cfg: DictConfig) -> str:
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        ckpt_path = OmegaConf.select(cfg, "train.resume_path")
    if not ckpt_path and os.path.exists(DEFAULT_INIT_CKPT):
        ckpt_path = DEFAULT_INIT_CKPT
    if not ckpt_path:
        return ""
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    return os.path.abspath(ckpt_path)


def _build_model_from_cfg(cfg: DictConfig) -> torch.nn.Module:
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(model_cfg, dict):
        raise TypeError("Model config must resolve to a dictionary.")
    target = model_cfg.pop("_target_", None)
    if not target:
        raise ValueError("Model config is missing _target_.")
    model_cls = hydra.utils.get_class(target)
    return model_cls(**model_cfg)


def _load_models(cfg: DictConfig, device: torch.device):
    model = _build_model_from_cfg(cfg)
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


def _load_text_encoder(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
    text_encoder = AutoModelForSeq2SeqLM.from_pretrained(TEXT_ENCODER_NAME)
    text_encoder = text_encoder.model
    text_encoder = text_encoder.to(device=device, dtype=torch.bfloat16).eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    return tokenizer, text_encoder


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
    out_path = os.path.join(output_dir, f"stage2_sample_{stamp}_{os.getpid()}.gif")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
    )
    return out_path


CSS = """
@import url("https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600&family=Space+Grotesk:wght@400;600;700&display=swap");

:root {
  --bg: #fbf3e6;
  --text: #111111;
  --muted: #5b5b5b;
  --card: rgba(255, 255, 255, 0.84);
  --border: rgba(15, 15, 15, 0.07);
  --accent-1: #ff3a2e;
  --accent-2: #ff6fb1;
  --accent-3: #6c65ff;
  --panel: rgba(255, 255, 255, 0.92);
}
body, .gradio-container {
  font-family: "Space Grotesk", "Instrument Sans", sans-serif;
  background-color: var(--bg);
  background-image:
    radial-gradient(1200px 600px at 8% 8%, rgba(255, 72, 68, 0.18), transparent 60%),
    radial-gradient(900px 520px at 92% 12%, rgba(114, 109, 255, 0.2), transparent 60%),
    radial-gradient(800px 520px at 70% 80%, rgba(255, 111, 181, 0.25), transparent 65%),
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/></filter><rect width='160' height='160' filter='url(%23n)' opacity='0.22'/></svg>");
  background-size: auto, auto, auto, 180px 180px;
  background-blend-mode: normal, normal, normal, soft-light;
  color: var(--text);
}
body::before,
body::after {
  content: "";
  position: fixed;
  width: 520px;
  height: 520px;
  border-radius: 999px;
  filter: blur(70px);
  opacity: 0.45;
  z-index: 0;
  pointer-events: none;
}
body::before {
  top: -120px;
  right: 5%;
  background: radial-gradient(circle, rgba(255, 111, 181, 0.55), rgba(255, 58, 46, 0));
}
body::after {
  bottom: -160px;
  left: 6%;
  background: radial-gradient(circle, rgba(92, 104, 255, 0.5), rgba(92, 104, 255, 0));
}
#app-shell {
  max-width: 1200px;
  margin: 0 auto;
  padding: 28px 24px 36px;
  position: relative;
  z-index: 1;
}
#header {
  background: transparent;
  color: var(--text);
  border-radius: 0;
  padding: 0 4px 12px;
  margin-bottom: 24px;
  box-shadow: none;
}
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 16px;
  align-items: center;
}
.hero-title {
  font-size: clamp(2.6rem, 4vw, 4.4rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 0.95;
}
.hero-subtitle {
  font-size: clamp(2.2rem, 3.4vw, 3.6rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1;
  margin-top: 8px;
}
.hero-caption {
  font-family: "Instrument Sans", sans-serif;
  font-size: 1rem;
  color: var(--muted);
  margin-top: 12px;
  max-width: 520px;
}
.gradient-word {
  background: linear-gradient(90deg, #ff3a2e 0%, #ff6fb1 45%, #6c65ff 75%, #3a4aff 100%);
  background-size: 200% 200%;
  color: transparent;
  -webkit-background-clip: text;
  background-clip: text;
  animation: gradientShift 6s ease infinite;
}
.section-card {
  position: relative;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 16px 34px rgba(12, 12, 12, 0.12);
  backdrop-filter: blur(14px);
  animation: rise 0.6s ease both;
  overflow: hidden;
}
.section-card::after {
  content: "";
  position: absolute;
  inset: 40% -20% -40% 45%;
  background: radial-gradient(circle, rgba(255, 111, 181, 0.18), transparent 65%);
  filter: blur(30px);
  opacity: 0.7;
}
.section-card::before {
  content: "";
  position: absolute;
  inset: 0 0 auto 0;
  height: 46px;
  background: linear-gradient(
    90deg,
    rgba(255, 58, 46, 0.18),
    rgba(255, 111, 181, 0.18),
    rgba(92, 104, 255, 0.18)
  );
  opacity: 0.5;
}
.section-card > * {
  position: relative;
}
.section-title {
  font-weight: 600;
  font-size: 1.05rem;
  color: var(--text);
  margin-bottom: 8px;
}
.gradio-container label span {
  font-family: "Instrument Sans", sans-serif;
  font-weight: 600;
  color: #383838;
}
.gradio-container input,
.gradio-container textarea {
  background: var(--panel);
  border: 1px solid rgba(15, 15, 15, 0.08);
  border-radius: 14px;
  padding: 12px 14px;
  font-family: "Instrument Sans", sans-serif;
  font-size: 0.98rem;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: rgba(255, 58, 46, 0.35);
  box-shadow: 0 0 0 3px rgba(255, 58, 46, 0.12);
}
#generate-btn {
  background: linear-gradient(90deg, #ff3a2e 0%, #ff6fb1 50%, #6c65ff 100%);
  border: none;
  color: white;
  font-weight: 600;
  padding: 12px 22px;
  border-radius: 999px;
  box-shadow: 0 14px 30px rgba(255, 79, 79, 0.3);
  letter-spacing: 0.02em;
}
#generate-btn:hover {
  filter: brightness(1.03);
}
#generate-btn:active {
  transform: translateY(1px);
}
.gradio-container .gr-input,
.gradio-container .gr-number,
.gradio-container .gr-textbox {
  background: transparent;
  border: none;
  box-shadow: none;
}
.gradio-container .gr-form {
  gap: 12px;
}
.gradio-container .gr-box {
  background: var(--panel);
  border: 1px solid rgba(15, 15, 15, 0.08);
  border-radius: 16px;
  box-shadow: 0 12px 30px rgba(16, 24, 40, 0.08);
}
.gradio-container .gr-image {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(15, 15, 15, 0.06);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
}
#status .status {
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid rgba(15, 15, 15, 0.1);
  color: var(--text);
  padding: 12px 14px;
  border-radius: 14px;
  font-size: 0.95rem;
}
#status .status.error {
  background: rgba(255, 225, 225, 0.9);
  border-color: rgba(255, 58, 46, 0.35);
  color: #9b1c1c;
}
#status .status.success {
  background: rgba(227, 255, 236, 0.9);
  border-color: rgba(22, 163, 74, 0.3);
  color: #166534;
}
.meta {
  font-family: "Instrument Sans", sans-serif;
  color: var(--muted);
}
@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes rise {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 900px) {
  .hero {
    grid-template-columns: 1fr;
  }
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
    blocks_kwargs = {"title": "Stage 2 Sampler", "elem_id": "app-shell"}
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
    tokenizer = text_encoder = None
    load_error = None
    try:
        model, model_ema, sampling_model, ckpt_path = _load_models(cfg, device)
        vae = _load_vae(cfg, device)
        tokenizer, text_encoder = _load_text_encoder(device)
    except Exception as exc:
        ckpt_path = _resolve_checkpoint_path(cfg)
        load_error = str(exc)
    ckpt_display = ckpt_path if ckpt_path else "unset"

    blocks_kwargs, launch_kwargs = _resolve_gradio_kwargs()
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            """
            <div class="hero">
              <div class="hero-text">
                <div class="hero-title">PixArt-Alpha</div>
                <div class="hero-subtitle"><span class="gradient-word">Stage 2</span> sampler</div>
                <div class="hero-caption">
                  Prompt -> reprompt -> text-conditioned sampling with optional seeding.
                </div>
              </div>
            </div>
            """,
            elem_id="header",
        )

        with gr.Row():
            with gr.Column(scale=4, min_width=320):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("Inputs", elem_classes="section-title")
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        placeholder="Describe the scene you want to generate.",
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

        def generate(prompt_value, seed_value, progress=gr.Progress()):
            if (
                load_error is not None
                or sampling_model is None
                or vae is None
                or tokenizer is None
                or text_encoder is None
            ):
                return None, None, _status_html(
                    "Error: model components failed to load. Check the checkpoint path.",
                    kind="error",
                )
            start_time = time.time()
            prompt_text = (prompt_value or "").strip()
            if not prompt_text:
                return None, None, _status_html(
                    "Error: prompt is required.",
                    kind="error",
                )

            seed_int = _validate_integer(seed_value)
            if seed_value is not None and seed_int is None:
                return None, None, _status_html(
                    "Error: seed must be an integer or left blank.",
                    kind="error",
                )

            try:
                progress(0, desc="Reprompting")
                reprompted = reprompt_prompt(prompt_text)

                progress(0.1, desc="Tokenizing")
                max_length = int(OmegaConf.select(cfg, "train.max_input_length") or 100)
                tokenized = tokenizer(
                    reprompted,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokenized = {
                    "input_ids": tokenized["input_ids"].to(device=device),
                    "attention_mask": tokenized["attention_mask"].to(device=device),
                }
                with torch.inference_mode():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"),
                    ):
                        text_tokens = text_encoder.encoder(**tokenized).last_hidden_state
                text_mask = tokenized["attention_mask"].bool()

                progress(0.2, desc="Preparing noise")
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

                progress(0.35, desc="Sampling 24 steps")
                with torch.inference_mode():
                    latents = generate_samples(
                        sampling_model,
                        noise,
                        text_tokens,
                        text_mask,
                        num_steps=NUM_STEPS,
                        num_save_steps=NUM_STEPS,
                    )

                progress(0.7, desc="Decoding frames")
                frames = _decode_latents_to_frames(vae, latents, cfg)

                progress(0.88, desc="Building GIF")
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
            inputs=[prompt, seed],
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
