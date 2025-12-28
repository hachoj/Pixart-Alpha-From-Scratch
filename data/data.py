import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


class LatentShardDatasetStage1(IterableDataset):
    def __init__(self, shard_paths, base_seed=0):
        self.shard_paths = list(shard_paths)
        self.base_seed = base_seed

    def __iter__(self):
        # ----- DDP info -----
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # ----- Worker info -----
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        # ----- Shard ownership -----

        # First split the shards into groups based on the number of workers
        rank_shards = self.shard_paths[rank::world_size]

        # Then split the shards within a rank by the number of workers
        my_shards = rank_shards[worker_id::num_workers]

        if len(my_shards) == 0:
            raise RuntimeError("No shards assigned to this worker")

        epoch = 0

        while True:
            rng = random.Random(
                self.base_seed + epoch * 1000 + rank * 100 + worker_id
            )

            rng.shuffle(my_shards)

            for shard_path in my_shards:
                data = torch.load(shard_path, map_location="cpu")
                latents = data["latents"]   # bf16 tensor [N, 16, 32, 32]
                labels = data["labels"]     # int tensor [N]

                indices = list(range(len(latents)))
                rng.shuffle(indices)

                for i in indices:
                    yield latents[i], labels[i]

            epoch += 1

class LatentShardDatasetStage2(IterableDataset):
    def __init__(self, shard_paths, base_seed=0):
        self.shard_paths = list(shard_paths)
        self.base_seed = base_seed

    def __iter__(self):
        # ----- DDP info -----
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # ----- Worker info -----
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        # ----- Shard ownership -----

        # First split the shards into groups based on the number of workers
        rank_shards = self.shard_paths[rank::world_size]

        # Then split the shards within a rank by the number of workers
        my_shards = rank_shards[worker_id::num_workers]

        if len(my_shards) == 0:
            raise RuntimeError("No shards assigned to this worker")

        epoch = 0

        while True:
            rng = random.Random(
                self.base_seed + epoch * 1000 + rank * 100 + worker_id
            )

            rng.shuffle(my_shards)

            for shard_path in my_shards:
                data = torch.load(shard_path, map_location="cpu")
                latents = data["latents"]   # bf16 tensor [N, 16, 32, 32]
                labels = data["labels"]     # int tensor [N]

                indices = list(range(len(latents)))
                rng.shuffle(indices)

                for i in indices:
                    yield latents[i], labels[i]

            epoch += 1
