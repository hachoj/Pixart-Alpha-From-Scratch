import argparse
import os
import pickle

import grain
import torch
from array_record.python import array_record_module  # pyrefly:ignore
from transformers import AutoTokenizer
from tqdm import tqdm


def _save_shard(
    *,
    out_dir: str,
    shard_number: int,
    latents: torch.Tensor,
    short_ids: torch.Tensor,
    short_mask: torch.Tensor,
    long_ids: torch.Tensor,
    long_mask: torch.Tensor,
    count: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"shard_{shard_number:0{6}d}.pt")
    torch.save(
        {
            "latents": latents[:count],
            "short_caption": short_ids[:count],
            "short_caption_mask": short_mask[:count],
            "long_caption": long_ids[:count],
            "long_caption_mask": long_mask[:count],
        },
        save_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=os.path.abspath("data/common"))
    parser.add_argument("--out-dir", default=os.path.abspath("data/stage2"))
    parser.add_argument("--records-per-shard", type=int, default=50000)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--tokenizer",
        default="google/t5gemma-xl-xl-ul2",
        help="HF tokenizer name or path",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    max_length: int = int(args.max_length)

    records_per_shard: int = int(args.records_per_shard)
    shard_number: int = 0
    i: int = 0

    latents_buf = torch.empty((records_per_shard, 16, 32, 32), dtype=torch.bfloat16)
    shrt_buf = torch.empty((records_per_shard, max_length), dtype=torch.int32)
    shrt_atn = torch.empty((records_per_shard, max_length), dtype=torch.int8)
    lng_buf = torch.empty((records_per_shard, max_length), dtype=torch.int32)
    lng_atn = torch.empty((records_per_shard, max_length), dtype=torch.int8)

    batch_latents: list[torch.Tensor] = []
    batch_short: list[str] = []
    batch_long: list[str] = []

    def flush_batch() -> None:
        nonlocal i, shard_number
        if not batch_short:
            return

        n = len(batch_short)

        enc_short = tokenizer(
            batch_short,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc_long = tokenizer(
            batch_long,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        short_ids = enc_short["input_ids"].to(dtype=torch.int32)
        short_mask = enc_short["attention_mask"].to(dtype=torch.int8)
        long_ids = enc_long["input_ids"].to(dtype=torch.int32)
        long_mask = enc_long["attention_mask"].to(dtype=torch.int8)

        # Latents are stored as bf16 bitpatterns in numpy (typically int16/uint16).
        # We keep the original semantics: reinterpret bits when int16/uint16.
        latent_stack = torch.stack(batch_latents, dim=0)
        if latent_stack.dtype in (torch.int16, torch.uint16):
            latent_stack = latent_stack.view(torch.bfloat16)
        else:
            latent_stack = latent_stack.to(dtype=torch.bfloat16)

        shrt_buf[i : i + n] = short_ids
        shrt_atn[i : i + n] = short_mask
        lng_buf[i : i + n] = long_ids
        lng_atn[i : i + n] = long_mask
        latents_buf[i : i + n] = latent_stack

        i += n
        batch_latents.clear()
        batch_short.clear()
        batch_long.clear()

        if i >= records_per_shard:
            _save_shard(
                out_dir=args.out_dir,
                shard_number=shard_number,
                latents=latents_buf,
                short_ids=shrt_buf,
                short_mask=shrt_atn,
                long_ids=lng_buf,
                long_mask=lng_atn,
                count=records_per_shard,
            )
            shard_number += 1
            i = 0

    for filename in tqdm(sorted(os.listdir(args.base_dir)), desc="array_record files"):
        path = os.path.join(args.base_dir, filename)
        try:
            array_record_data_source = grain.sources.ArrayRecordDataSource(path)
            for data in tqdm(array_record_data_source, desc=filename, leave=False):
                element = pickle.loads(data)
                latent_np = element["latent"]
                short_caption: str = element["short_caption"]
                long_caption: str = element["long_caption"]

                latent_t = torch.from_numpy(latent_np).reshape(16, 32, 32)

                batch_latents.append(latent_t)
                batch_short.append(short_caption)
                batch_long.append(long_caption)

                if (
                    len(batch_short) >= args.batch_size
                    or i + len(batch_short) >= records_per_shard
                ):
                    flush_batch()
        except Exception as exc:
            raise ValueError(f"invalid record in {path}") from exc

    flush_batch()
    if i > 0:
        _save_shard(
            out_dir=args.out_dir,
            shard_number=shard_number,
            latents=latents_buf,
            short_ids=shrt_buf,
            short_mask=shrt_atn,
            long_ids=lng_buf,
            long_mask=lng_atn,
            count=i,
        )


if __name__ == "__main__":
    main()
