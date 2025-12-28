import os
import pickle

import grain
import torch
from array_record.python import array_record_module  # pyrefly:ignore
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

base_dir = os.path.abspath("data/datasets")

records_per_shard: int = 50000
shard_number: int = 0

latents: list[Tensor] = []
labels: list[int] = []

print(base_dir)
print(os.listdir(base_dir))
# for path in tqdm(os.listdir(base_dir)):
    # new_path = os.path.join(base_dir, path)
    # buf = torch.empty((records_per_shard, 16, 32, 32), dtype=torch.bfloat16)
    # lbl = torch.empty((records_per_shard,))
    # try:
    #     array_record_data_source = grain.sources.ArrayRecordDataSource(new_path)
    #     i = 0
    #     for data in tqdm(array_record_data_source):
    #         if i >= records_per_shard:
    #             save_path = os.path.join(base_dir, f"shard_{shard_number:0{6}d}.pt")
    #             torch.save({"latents": buf, "labels": lbl}, save_path)
    #             shard_number += 1
    #             i = 0
    #             buf = torch.empty((records_per_shard, 16, 32, 32), dtype=torch.bfloat16)
    #             lbl = torch.empty((records_per_shard,))
    #         element = pickle.loads(data)
    #         latent = element["latent"]
    #         label: int = element["label"]
    #         latent: Float[Tensor, "C H W"] = torch.from_numpy(latent).view(torch.bfloat16)
    #         buf[i] = latent 
    #         lbl[i] = label
    #         i += 1

    # except:
    #     raise ValueError("invalid record")
