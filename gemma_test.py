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


def encode_with_t5gemma_encoder(
    texts: Union[str, Sequence[str]],
    *,
    model,
    params,
    tokenizer,
    max_input_length: int = 256,
    model_sharding: jshard.NamedSharding,
    data_sharding: jshard.NamedSharding,
    return_on_host: bool = True,
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

    def _encoder_last_hidden(params, tokens, mask):
        encoder_acts = model.apply(
            {"params": params},
            tokens=tokens,
            inputs_mask=mask,
            method=model.compute_encoder_activations,
        )
        return encoder_acts.activations[-1]  # [B, L, d_model]

    params_s = jax.device_put(params, model_sharding)
    tokens_s = jax.device_put(input_tokens, data_sharding)
    mask_s = jax.device_put(inputs_mask, data_sharding)

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


texts = ["A picture of a sunlit sky.", "A watercolor painting of a mountain."]

devices = jax.devices()
mesh = jshard.Mesh(devices, axis_names=("data",))
model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())
data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("data"))

encs_padded, masks = encode_with_t5gemma_encoder(
    texts,
    model=t5gemma_model,
    params=t5gemma_params,
    tokenizer=preset.tokenizer,
    max_input_length=256,
    model_sharding=model_sharding,
    data_sharding=data_sharding,
)
for enc, mask in zip(encs_padded, masks):
    print("Encoded shape:", enc.shape)  # [L, d_model]
    print("Mask: ", mask[:40])
    print(f"Type of enc: {type(enc)}")
    print(f"Dtype of enc: {enc.dtype}")
    # Cast to float32 for stable statistics calculation and printing
    enc_f32 = enc.astype(jnp.float32)
    print(
        f"Output stats: Mean={jnp.mean(enc_f32):.4f}, Std={jnp.std(enc_f32):.4f}, Min={jnp.min(enc_f32):.4f}, Max={jnp.max(enc_f32):.4f}"
    )
    print("-------------------------------------------------------------")
