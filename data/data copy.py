from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import torch
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.types import DALIImageType
from nvidia.dali.plugin.pytorch import DALIGenericIterator


# 1. Define the Pipeline (The "Blueprint")
@pipeline_def(
    batch_size=32, num_threads=8, device_id=0, prefetch_queue_depth=4
)  # pyrefly: ignore
def cat_pipeline(tar_path, idx_path, shard_id, num_shards):
    # READER: Reads the tar file directly
    # 'ext' tells it which files to grab. Since you only have pngs, we ask for that.
    # It returns a tuple, so we grab the first element (the image bytes).
    images, labels = fn.readers.webdataset(
        paths=[tar_path],
        index_paths=[idx_path],
        ext=["png", "cls"],
        random_shuffle=True,  # Shuffle the buffer
        initial_fill=8192,  # Size of shuffle buffer
        name="Reader",  # Important for tracking epoch size later
        shard_id=shard_id,
        num_shards=num_shards,
    )

    # DECODER: The Magic Step
    # 'mixed' = Input is on CPU (bytes), Output goes to GPU (pixels)
    images = fn.decoders.image(
        images, output_type=DALIImageType.RGB, device="mixed"
    )  # pyrefly: ignore
    images = images / 127.5 - 1

    # RESIZE / AUGMENT (Happens on GPU)
    # Let's resize to 224x224 to match standard models
    images = fn.resize(images, size=[512, 512])

    should_flip = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=should_flip)

    labels = labels.gpu()
    labels = fn.cast(labels, dtype=types.INT64)

    # THIS SOLUTION IS A HACK FIX BETTER LATER
    labels = labels - 48

    return images, labels


# 2. Build the Iterator (The "Switch")
def get_dali_loader(
    tar_path, idx_path, batch_size, global_rank, world_size, local_rank
):
    # Instantiate the pipeline
    pipe = cat_pipeline(
        tar_path,
        idx_path,
        global_rank,
        world_size,
        batch_size=batch_size,
        device_id=local_rank,
    )
    pipe.build()

    # Wrap it for PyTorch
    # output_map tells DALI what keys to use in the dictionary it yields
    loader = DALIGenericIterator(
        [pipe],
        output_map=["image", "labels"],
        reader_name="Reader",  # Syncs with the reader inside to handle epochs correctly
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.DROP,
    )

    return loader
