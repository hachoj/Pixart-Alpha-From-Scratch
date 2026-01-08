BATCH_SIZE = 512
MAX_TOKENS = 120
RECORDS_PER_SHARD = 50000

OUT_DIR = "data/stage2/"
TOTAL_SHARDS = 7760

from prompts import caption_prompt  # pyrefly:ignore

import argparse
from io import BytesIO
import os
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Guard against stale SLURM scratch paths in TorchInductor/Triton caches.
cache_root = os.path.join(os.path.expanduser("~"), ".cache", "pixart_vllm")
cache_dirs = {
    "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache_root, "torchinductor"),
    "TORCH_COMPILE_CACHE_DIR": os.path.join(cache_root, "torch_compile"),
    "TRITON_CACHE_DIR": os.path.join(cache_root, "triton"),
    "TMPDIR": os.path.join(cache_root, "tmp"),
    "TMP": os.path.join(cache_root, "tmp"),
    "TEMP": os.path.join(cache_root, "tmp"),
}

for key, path in cache_dirs.items():
    current = os.environ.get(key)
    if not current or current.startswith("/scratch/local/"):
        os.makedirs(path, exist_ok=True)
        os.environ[key] = path

from PIL import Image
from diffusers.models import AutoencoderKL
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import torchvision.transforms.v2 as v2
from datasets import Image as DatasetsImage
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader

MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "768"))
MM_ENCODER_ATTN_BACKEND = os.environ.get("VLLM_MM_ENCODER_ATTN_BACKEND", "TORCH_SDPA")
ATTENTION_BACKEND = os.environ.get("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")


def decode_pil_image_safely(image_value):
    """Decode a HF datasets image into a PIL Image, skipping known-bad samples.

    We avoid letting `datasets` decode images in DataLoader workers because Pillow
    may raise on PNGs with huge iCCP/text chunks (a safety limit).
    """

    # If already decoded by some upstream formatting, pass through.
    if isinstance(image_value, Image.Image):
        return image_value

    # When using datasets.Image(decode=False), values are dict-like with bytes.
    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        if image_bytes is None:
            return None
        try:
            with Image.open(BytesIO(image_bytes)) as im:
                im.load()
                return im.copy()
        except (OSError, ValueError):
            return None

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shard Imagenet-21K with vLLM captions."
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def prepare_inputs_for_vllm(messages, processor):
    prompt = processor.apply_chat_template(  # pyrefly:ignore
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


def main(args):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        subfolder="vae",
        dtype=torch.bfloat16,
    ).to(device="cuda")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, padding_side="left")
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        mm_encoder_attn_backend=MM_ENCODER_ATTN_BACKEND,
        attention_backend=ATTENTION_BACKEND,
        trust_remote_code=True,
        gpu_memory_utilization=0.70,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.85,
        top_k=30,
        max_tokens=MAX_TOKENS,
    )

    gemma_tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-xl-xl-ul2")

    for param in vae.parameters():
        param.requires_grad = False

    vae.eval()

    vae = vae.to(device="cuda")

    ds: IterableDataset = load_dataset(  # pyrefly:ignore
        "gmongaras/Imagenet21K", streaming=True, split="train"
    )

    # Important: keep images as raw bytes so PIL decode errors don't crash
    # DataLoader workers. We'll decode inside `preprocess` and drop failures.
    ds = ds.cast_column("image", DatasetsImage(decode=False))

    from datasets.distributed import split_dataset_by_node

    # 7760
    ds = split_dataset_by_node(ds, rank=args.rank, world_size=args.world_size)

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
        # Some samples contain PNG metadata that triggers Pillow safety limits
        # (e.g. huge iCCP chunk). Decode safely and skip problematic images.
        images = []
        for img_val in batch["image"]:
            img = decode_pil_image_safely(img_val)
            if img is None:
                continue
            if img.size[0] <= 256 or img.size[1] <= 256:
                continue
            images.append(img)

        latent_tensor = [transform_latent(img) for img in images]
        qwen_tensor = [transform_qwen(img) for img in images]

        return {"latent_tensor": latent_tensor, "qwen_tensor": qwen_tensor}

    cols_to_keep = ["latent_tensor", "qwen_tensor"]
    cols_to_remove = [
        c for c in ds.column_names if c not in cols_to_keep
    ]  # pyrefly:ignore

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

    record_idx = 0

    shard_number = args.rank

    for data in dataloader:
        iter_start = time.perf_counter()
        data_load_time = time.perf_counter()
        t0 = time.perf_counter()
        latent = data["latent_tensor"]
        qwen_tensor = data["qwen_tensor"]
        t1 = time.perf_counter()
        latent = latent.to(device="cuda", non_blocking=True)
        t2 = time.perf_counter()

        latent = latent / 255.0
        latent = latent.to(dtype=torch.bfloat16)
        qwen_images = [
            Image.fromarray(img.permute(1, 2, 0).contiguous().numpy())
            for img in qwen_tensor
        ]
        t3 = time.perf_counter()

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
            for img in qwen_images
        ]
        vllm_inputs = [prepare_inputs_for_vllm(msg, processor) for msg in messages]
        t4 = time.perf_counter()

        with torch.inference_mode():
            t5 = time.perf_counter()
            meanlogvar = vae._encode(latent)
            mean = meanlogvar[:, :16, :, :]
            t6 = time.perf_counter()

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        t7 = time.perf_counter()

        output_text = [out.outputs[0].text for out in outputs]
        if args.debug:
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
        batch_size = tokens.shape[0]
        if args.debug:
            print(f"Images in batch: {batch_size}")

        t10 = t9
        t11 = t9
        if record_idx + batch_size >= RECORDS_PER_SHARD:
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
            shard_number += args.world_size
            record_idx = 0
            latent_buf = torch.empty((RECORDS_PER_SHARD, 16, 32, 32), dtype=torch.bfloat16)
            token_buf = torch.empty((RECORDS_PER_SHARD, MAX_TOKENS), dtype=torch.int32)
            mask_buf = torch.empty((RECORDS_PER_SHARD, MAX_TOKENS), dtype=torch.int8)
            t11 = time.perf_counter()

        latent_buf[record_idx : record_idx + batch_size] = mean
        token_buf[record_idx : record_idx + batch_size] = tokens
        mask_buf[record_idx : record_idx + batch_size] = attn_mask
        record_idx += batch_size
        save_record_time = time.perf_counter()
        t12 = time.perf_counter()
        if args.debug:
            print(
                "To save "
                f"{batch_size} records it took {(data_load_time - iter_start):.2f}s to load "
                f"and {(save_record_time - iter_start):.2f}s total. "
                f"batch_unwrap={(t1 - t0):.3f}s, "
                f"to_cuda={(t2 - t1):.3f}s, "
                f"preprocess={(t3 - t2):.3f}s, "
                f"vllm_inputs={(t4 - t3):.3f}s, "
                f"vae_encode={(t6 - t5):.3f}s, "
                f"vllm_generate={(t7 - t6):.3f}s, "
                f"tokenize={(t9 - t8):.3f}s, "
                f"save={(t11 - t10):.3f}s, "
                f"buffer_copy={(t12 - t9):.3f}s"
            )


if __name__ == "__main__":
    main(parse_args())
