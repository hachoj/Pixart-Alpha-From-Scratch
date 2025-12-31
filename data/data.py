import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


class LatentShardDatasetStage1(IterableDataset):
    def __init__(self, shard_paths, seed=0):
        self.shard_paths = shard_paths
        self.seed = seed

    def __iter__(self):
        # --- DDP Setup ---
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank: int = 0
            world_size: int = 1

        # --- Workers Setup ---
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

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
        # --- DDP Setup ---
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank: int = 0
            world_size: int = 1

        # --- Workers Setup ---
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

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
                short_caption = data['short_caption']
                short_caption_mask = data['short_caption_mask']
                long_caption = data['long_caption']
                long_caption_mask = data['long_caption_mask']

                idx = list(range(len(latents)))
                rng.shuffle(idx)

                for i in idx:
                    yield (latents[i], short_caption[i], short_caption_mask[i], long_caption[i], long_caption_mask[i])
                
            epoch += 1
            
