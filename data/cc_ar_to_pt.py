import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import os
import pickle

import grain
import torch
from array_record.python import array_record_module  # pyrefly:ignore
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
import random

base_dir = os.path.abspath("data/common")
out_dir = os.path.abspath("data/stage2")

records_per_shard: int = 50000
shard_number: int = 0

buf = torch.empty((records_per_shard, 16, 32, 32), dtype=torch.bfloat16)
shrt_buf = torch.empty((records_per_shard, 100), dtype=torch.int32)
shrt_atn = torch.empty((records_per_shard, 100), dtype=torch.int8)
lng_buf = torch.empty((records_per_shard, 100), dtype=torch.int32)
lng_atn = torch.empty((records_per_shard, 100), dtype=torch.int8)
tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-xl-xl-ul2")
for path in tqdm(os.listdir(base_dir)):
    new_path = os.path.join(base_dir, path)
    try:
        array_record_data_source = grain.sources.ArrayRecordDataSource(new_path)
        i = 0
        # for data in tqdm(array_record_data_source):
        for data in tqdm(array_record_data_source):
            if i >= records_per_shard:
                save_path = os.path.join(out_dir, f"shard_{shard_number:0{6}d}.pt")
                torch.save(
                    {
                        "latents": buf,
                        "short_caption": shrt_buf,
                        "short_caption_mask": shrt_atn,
                        "long_caption": lng_buf,
                        "long_caption_mask": lng_atn,
                    },
                    save_path,
                )
                shard_number += 1
                i = 0
                buf = torch.empty((records_per_shard, 16, 32, 32), dtype=torch.bfloat16)
                shrt_buf = torch.empty((records_per_shard, 100), dtype=torch.int32)
                shrt_atn = torch.empty((records_per_shard, 100), dtype=torch.int8)
                lng_buf = torch.empty((records_per_shard, 100), dtype=torch.int32)
                lng_atn = torch.empty((records_per_shard, 100), dtype=torch.int8)

            element = pickle.loads(data)
            latent = element["latent"]
            short_caption: str = element["short_caption"]
            long_caption: str = element["long_caption"]
            latent: Float[Tensor, "C H W"] = torch.from_numpy(latent).view(
                torch.bfloat16
            )

            # --- Tokenization ---
            input_ids = tokenizer(
                short_caption,
                padding="max_length",
                truncation=True,
                max_length=100,
                return_tensors="pt",
            )
            shrt_buf[i] = input_ids["input_ids"]
            shrt_atn[i] = input_ids["attention_mask"]
            input_ids = tokenizer(
                long_caption,
                padding="max_length",
                truncation=True,
                max_length=100,
                return_tensors="pt",
            )
            lng_buf[i] = input_ids["input_ids"]
            lng_atn[i] = input_ids["attention_mask"]
            buf[i] = latent
            i += 1

    except:
        raise ValueError("invalid record")
