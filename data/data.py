import os
import random
import warnings

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


_WARNED_FALLBACK_RANK_WS = False


def _get_rank_world_size() -> tuple[int, int, str]:
    """Return (rank, world_size, source).

    DataLoader workers may not have torch.distributed initialized; torchrun
    provides RANK/WORLD_SIZE env vars which are inherited by workers.
    """

    env_rank = os.environ.get("RANK")
    env_world_size = os.environ.get("WORLD_SIZE")
    if env_rank is not None and env_world_size is not None:
        return int(env_rank), int(env_world_size), "env"

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), "dist"

    return 0, 1, "fallback"


class LatentShardDatasetStage1(IterableDataset):
    def __init__(self, shard_paths, seed=0):
        self.shard_paths = shard_paths
        self.seed = seed

    def __iter__(self):
        # --- Workers Setup ---
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        # --- DDP Setup ---
        rank, world_size, source = _get_rank_world_size()
        global _WARNED_FALLBACK_RANK_WS
        if source == "fallback" and worker_id == 0 and not _WARNED_FALLBACK_RANK_WS:
            warnings.warn(
                "LatentShardDatasetStage1: torch.distributed not initialized and "
                "RANK/WORLD_SIZE not set; falling back to rank=0/world_size=1. "
                "In multi-GPU runs with DataLoader workers, this can duplicate data.",
                RuntimeWarning,
            )
            _WARNED_FALLBACK_RANK_WS = True

        # --- Distriute Shards ---

        rank_shards = self.shard_paths[rank::world_size]
        worker_shards = rank_shards[worker_id::num_workers]

        if len(worker_shards) == 0:
            raise RuntimeError("No shards assigned to this worker")

        epoch = 0

        # infinite dataset
        while True:
            rng = random.Random(self.seed + epoch * 1000 + rank * 100 + worker_id)

            rng.shuffle(worker_shards)

            for shard_path in worker_shards:
                data = torch.load(shard_path, map_location="cpu")

                latents = data['latents']
                labels = data['labels']

                idx = list(range(len(latents)))
                rng.shuffle(idx)

                for i in idx:
                    yield (latents[i], labels[i])
                
            epoch += 1
            

class LatentShardDatasetStage2(IterableDataset):
    def __init__(self, shard_paths, seed=0):
        self.shard_paths = shard_paths
        self.seed = seed

    def __iter__(self):
        # --- Workers Setup ---
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        # --- DDP Setup ---
        rank, world_size, source = _get_rank_world_size()
        global _WARNED_FALLBACK_RANK_WS
        if source == "fallback" and worker_id == 0 and not _WARNED_FALLBACK_RANK_WS:
            warnings.warn(
                "LatentShardDatasetStage2: torch.distributed not initialized and "
                "RANK/WORLD_SIZE not set; falling back to rank=0/world_size=1. "
                "In multi-GPU runs with DataLoader workers, this can duplicate data.",
                RuntimeWarning,
            )
            _WARNED_FALLBACK_RANK_WS = True

        # --- Distriute Shards ---

        rank_shards = self.shard_paths[rank::world_size]
        worker_shards = rank_shards[worker_id::num_workers]

        if len(worker_shards) == 0:
            raise RuntimeError("No shards assigned to this worker")

        epoch = 0

        # infinite dataset
        while True:
            rng = random.Random(self.seed + epoch * 1000 + rank * 100 + worker_id)

            rng.shuffle(worker_shards)

            for shard_path in worker_shards:
                data = torch.load(shard_path, map_location="cpu")

                latents = data['latents']
                text_tokens = data['text_tokens']
                attention_mask = data['attn_mask']

                # Filter out invalid/uninitialized records
                # Valid token IDs should be in range [0, vocab_size)
                # Using 256000 as the Gemma vocab size upper bound
                valid_mask = (text_tokens.min(dim=1).values >= 0) & (text_tokens.max(dim=1).values < 256000)
                valid_indices = valid_mask.nonzero(as_tuple=True)[0].tolist()

                rng.shuffle(valid_indices)

                for i in valid_indices:
                    yield (latents[i], text_tokens[i], attention_mask[i])
                
            epoch += 1
            
