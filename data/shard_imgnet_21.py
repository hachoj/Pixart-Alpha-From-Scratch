BATCH_SIZE = 32
MAX_TOKENS = 120
RECORDS_PER_SHARD = 50000
OUT_DIR = "data/stage2/"

from prompts import caption_prompt  # pyrefly:ignore

import os
from diffusers.models import AutoencoderKL
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
import torch
from transformers import AutoTokenizer
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
    device_map="cuda",
)

gemma_tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-xl-xl-ul2")

for param in vae.parameters():
    param.requires_grad = False

vae.eval()

vae = vae.to(device="cuda")

ds: IterableDataset = load_dataset(  # pyrefly:ignore
    "gmongaras/Imagenet21K", streaming=True, split="train"
)

from datasets.distributed import split_dataset_by_node

# 7760
ds = split_dataset_by_node(ds, rank=0, world_size=8)


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
    images = [
        img
        for img in batch["image"]
        if img.size[0] > 256 and img.size[1] > 256
    ]

    latent_tensor = [transform_latent(img) for img in images]
    qwen_tensor = [transform_qwen(img) for img in images]

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
    num_workers=2,
    pin_memory=True,
)

latent_buf = torch.empty((RECORDS_PER_SHARD, 16, 32, 32), dtype=torch.bfloat16)
token_buf = torch.empty((RECORDS_PER_SHARD, MAX_TOKENS), dtype=torch.int32)
mask_buf = torch.empty((RECORDS_PER_SHARD, MAX_TOKENS), dtype=torch.int8)

i = 0
shard_number = 0

import time

start_time = time.perf_counter()
for i, data in enumerate(dataloader):
    data_load_time = time.perf_counter()
    t0 = time.perf_counter()
    latent = data["latent_tensor"]
    qwen_tensor = data["qwen_tensor"]
    t1 = time.perf_counter()
    latent = latent.to(device="cuda", non_blocking=True)
    qwen_tensor = qwen_tensor.to(device="cuda", non_blocking=True)
    t2 = time.perf_counter()

    latent = latent / 255.0

    latent = latent.to(dtype=torch.bfloat16)
    qwen_img = [img.to(dtype=torch.bfloat16) for img in qwen_tensor]
    t3 = time.perf_counter()

    # Build chat template to ensure image tokens are inserted, then batch process.
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
    prompts = processor.apply_chat_template(  # pyrefly:ignore
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(  # pyrefly:ignore
        images=qwen_img,
        text=prompts,
        padding=True,
        return_tensors="pt",
    ).to(qwen.device)
    t4 = time.perf_counter()

    with torch.inference_mode():
        t5 = time.perf_counter()
        meanlogvar = vae._encode(latent)
        mean = meanlogvar[:, :16, :, :]
        t6 = time.perf_counter()

        generated_ids = qwen.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            top_p=0.85,
            top_k=30,
            max_new_tokens=120,
            use_cache=True,
        )
        t7 = time.perf_counter()

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(  # pyrefly:ignore
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"Output_text: {output_text[0]}")
    t8 = time.perf_counter()
    input_ids = gemma_tokenizer(
        output_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    )
    t9 = time.perf_counter()

    tokens = input_ids["input_ids"]
    attn_mask = input_ids["attention_mask"]

    t10 = t9
    t11 = t9
    if i + BATCH_SIZE >= RECORDS_PER_SHARD:
        t10 = time.perf_counter()
        save_path = os.path.join(OUT_DIR, f"shard_{shard_number:0{6}d}.pt")
        torch.save(
            {
                "latents": latent_buf,
                "text_tokens": token_buf,
                "attn_mask": mask_buf,
            },
            save_path,
        )
        shard_number += 1
        i = 0
        latent_buf = torch.empty((RECORDS_PER_SHARD, 16, 32, 32), dtype=torch.bfloat16)
        token_buf = torch.empty((RECORDS_PER_SHARD, MAX_TOKENS), dtype=torch.int32)
        mask_buf = torch.empty((RECORDS_PER_SHARD, MAX_TOKENS), dtype=torch.int8)
        t11 = time.perf_counter()

    latent_buf[i : i + BATCH_SIZE] = mean
    token_buf[i : i + BATCH_SIZE] = tokens
    mask_buf[i : i + BATCH_SIZE] = attn_mask
    i += BATCH_SIZE
    save_record_time = time.perf_counter()
    t12 = time.perf_counter()
    print(
        "To save "
        f"{BATCH_SIZE} records it took {(data_load_time - start_time):.2f}s to load "
        f"and {(save_record_time - start_time):.2f}s total. "
        f"batch_unwrap={(t1 - t0):.3f}s, "
        f"to_cuda={(t2 - t1):.3f}s, "
        f"preprocess={(t3 - t2):.3f}s, "
        f"qwen_inputs={(t4 - t3):.3f}s, "
        f"vae_encode={(t6 - t5):.3f}s, "
        f"qwen_generate={(t7 - t6):.3f}s, "
        f"decode={(t8 - t7):.3f}s, "
        f"tokenize={(t9 - t8):.3f}s, "
        f"save={(t11 - t10):.3f}s, "
        f"buffer_copy={(t12 - t9):.3f}s"
    )
    start_time = time.time()
