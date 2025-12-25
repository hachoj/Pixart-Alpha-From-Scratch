BATCH_SIZE = 1024

from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch
from torch import Tensor

from diffusers.models import AutoencoderKL

import numpy as np


def serialize(mean, label):
    record = {
        "latent": mean,
        "label": label,
    }

    return record


def preprocess(batch) -> dict[str, list[Tensor] | list[int]]:
    tensor: list[Tensor] = [transform(img) for img in batch["image"]]
    label: list[int] = batch.get("label")
    return {"img": tensor, "label": label}


vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="vae",
    torch_dtype=torch.bfloat16,
).to(device="cuda")

transform = v2.Compose(
    [
        v2.RGB(),
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToImage(),
        v2.ToDtype(torch.bfloat16, scale=True),
    ]  # pyrefly:ignore
)

ds: IterableDataset = load_dataset(
    "ILSVRC/imagenet-1k", streaming=True, split="train"
)  # pyrefly:ignore

cols_to_keep = ["img", "label"]
cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]  # pyrefly:ignore

ds = ds.map(preprocess, batched=True, remove_columns=cols_to_remove)

dataloader = DataLoader(
    ds,  # pyrefly:ignore
    batch_size=BATCH_SIZE,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
)

import time
from datetime import timedelta
from array_record.python import array_record_module  # pyrefly:ignore
import pickle

path = "image_net.array_record"
writer = array_record_module.ArrayRecordWriter(path, "group_size:1")

total = 1280000 // BATCH_SIZE
record_count = 0
last_10_times = []
for i, data in enumerate(dataloader):
    start_time = time.time()
    img_tensor = data["img"]
    labels = data["label"]  # B
    img_tensor = img_tensor.to(vae.device, non_blocking=True)

    with torch.inference_mode():
        meanlogvar = vae._encode(img_tensor)  # B,32,32,32
        mean = meanlogvar[:, :16, :, :]  # B,16,32,32
        mean = mean.view(torch.int16)

    mean_np = mean.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    for b in range(mean_np.shape[0]):
        record = serialize(mean_np[b], labels_np[b])
        writer.write(pickle.dumps(record))
        record_count += 1

    percentage = 1 - ((i + 1) / total)
    last_10_times.append(time.time() - start_time)
    if len(last_10_times) > 10:
        last_10_times.pop(0)
    seconds = int(np.mean(last_10_times) * total * percentage)
    print(f"{i+1}/{total} | Estimated time remaining: {timedelta(seconds=seconds)}")


writer.close()
print(f"Number of records written to array_record file {path} :" f" {record_count}")
