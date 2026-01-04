BATCH_SIZE = 1

from prompts import caption_prompt  # pyrefly:ignore

from diffusers.models import AutoencoderKL
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
import torch
import torchvision.transforms.v2 as v2
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader

vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
).to(device="cuda")

model_name = "Qwen/Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
qwen = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

for param in vae.parameters():
    param.requires_grad = False

vae.eval()

vae = vae.to(device="cuda")

ds: IterableDataset = load_dataset(  # pyrefly:ignore
    "gmongaras/Imagenet21K", streaming=True, split="train"
)

from datasets.distributed import split_dataset_by_node

ds = split_dataset_by_node(ds, rank=0, world_size=7760)


transform_latent = v2.Compose(
    [
        v2.RGB(),
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=False),
    ]  # pyrefly:ignore
)


transform_qwen = v2.Compose(
    [
        v2.RGB(),
        v2.Resize(512),
        v2.CenterCrop(512),
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=False),
    ]  # pyrefly:ignore
)


def preprocess(batch):
    latent_tensor = [
        transform_latent(img)
        for img in batch["image"]
        if img.size[0] > 256 and img.size[1] > 256
    ]

    qwen_tensor = [
        transform_qwen(img)
        for img in batch["image"]
        if img.size[0] > 256 and img.size[1] > 256
    ]

    return {"latent_tensor": latent_tensor, "qwen_tensor": qwen_tensor}


cols_to_keep = ["latent_tensor", "qwen_tensor"]
cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]  # pyrefly:ignore

ds = ds.map(
    preprocess,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=cols_to_remove,
)

dataloader = DataLoader(
    ds,  # pyrefly:ignore
    batch_size=BATCH_SIZE,
    num_workers=0,
    # prefetch_factor=2,
)

for i, data in enumerate(dataloader):
    latent = data["latent_tensor"]
    qwen_tensor = data["qwen_tensor"]
    latent = latent.to(device="cuda")
    qwen_tensor = qwen_tensor.to(device="cuda")

    latent = latent / 255.0

    latent = latent.to(dtype=torch.bfloat16)
    qwen_img = [img.to(dtype=torch.bfloat16) for img in qwen_tensor]

    # --- Qwen Template ---
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": caption_prompt},
                ],
            }
        ]
        for img in qwen_img
    ]
    inputs = processor.apply_chat_template(  # pyrefly:ignore
        messages,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        # truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(qwen.device)

    with torch.inference_mode():
        meanlogvar = vae._encode(latent)
        mean = meanlogvar[:, :16, :, :]

        generated_ids = qwen.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            top_k=30,
            max_new_tokens=120,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(  # pyrefly:ignore
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    for mean, caption in zip(mean, output_text):

