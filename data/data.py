import grain.python as grain
import pickle
import numpy as np


class ParseRecordStage1(grain.MapTransform):
    def map(self, record: bytes):  # pyrefly:ignore
        data = pickle.loads(record)
        return {"latent": data["latent"], "label": data["label"]}


class ParseRecordStage2(grain.MapTransform):
    def map(self, record: bytes):  # pyrefly:ignore
        data = pickle.loads(record)
        return {
            "latent": data["latent"],
            "short_caption": data["short_caption"],
            "long_caption": data["long_caption"],
        }


def create_dataloader_stage_1(file_path, num_workers, batch_size, seed):
    source = grain.ArrayRecordDataSource(file_path)

    index_sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=None,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=True,
        seed=seed,
    )

    operations = [
        ParseRecordStage1(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=index_sampler,
        worker_count=num_workers,
        operations=operations,
    )

    return dataloader


def create_dataloader_stage_2(file_paths: list[str], num_workers, batch_size, seed):
    source = grain.ArrayRecordDataSource(file_paths)

    index_sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=None,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=True,
        seed=seed,
    )

    operations = [
        ParseRecordStage2(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=index_sampler,
        worker_count=num_workers,
        operations=operations,
    )

    return dataloader
