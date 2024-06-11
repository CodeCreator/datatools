import json
import os
import logging
from typing import Any, Dict, Union, List
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from streaming.base.array import Array
from streaming.base.format import get_index_basename, reader_from_json
from streaming.base.spanner import Spanner


from filelock import FileLock


class Subset(Array):
    def __init__(self, dataset: Array, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    @classmethod
    def shard(cls, dataset: Array, shard_id: int, num_shards: int):
        N = len(dataset)
        shard_indices = np.linspace(0, N, num_shards + 1)

        return cls(dataset, range(int(shard_indices[shard_id]), int(shard_indices[shard_id + 1])))

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def size(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[self.indices[idx]]


class LocalDatasets(Array):
    def __init__(self, paths: List[Union[Path, str]]):
        self.shards = []
        for path in paths:
            filename = os.path.join(path, get_index_basename())  # pyright: ignore
            obj = json.load(open(filename))

            for info in obj['shards']:
                shard = reader_from_json(path, "", info)
                self.shards.append(shard)
        self.num_samples = sum([shard.samples for shard in self.shards])

        shard_sizes = np.array([x.samples for x in self.shards])
        self.spanner = Spanner(shard_sizes)

    def __len__(self) -> int:
        return self.num_samples

    @property
    def size(self) -> int:
        return self.num_samples

    def get_item(self, sample_id: int) -> Dict[str, Any]:
        shard_id, index_in_shard = self.spanner[sample_id]
        shard = self.shards[shard_id]
        return shard[index_in_shard]


class JsonlDataset(Array):
    def __init__(self, paths: List[Union[str, Path]]):
        self.paths = paths

        lines = []
        for path in paths:
            with Path(path).open() as f:
                lines.extend(f.readlines())

    def __len__(self) -> int:
        return len(self.lines)

    @property
    def size(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return json.loads(self.lines[idx])


def merge_index(directory : Path):
    print(f"Merge the index for {directory}")
    index_filename = get_index_basename()

    # sorting should ensure that the index order of the original datasets is retained
    subfolders = sorted([
        d for d in directory.iterdir()
        if d.is_dir() and (d / index_filename).is_file()
    ])

    if len(subfolders) > 0:
        shards = []
        for d in subfolders:
            with (d / index_filename).open() as fp:
                subindex = json.load(fp)
            for shard in subindex["shards"]:
                shard['raw_data']['basename'] = str(d.relative_to(d.parent) / shard['raw_data']['basename'])
                if 'zip_data' in shard and shard['zip_data'] is not None and 'basename' in shard['zip_data']:
                    shard['zip_data']['basename'] = str(d.relative_to(d.parent) / shard['zip_data']['basename'])
                shards.append(shard)
        new_index = {
            'version': 2,
            'shards': shards,
        }
        with (directory / index_filename).open("w") as fp:
            json.dump(new_index, fp, indent=4)


def merge_index_recursively(directory : Path):
    assert directory.is_dir(), f"{directory} is not a directory"
    index_filename = get_index_basename()

    subfolders = sorted([d for d in directory.iterdir() if d.is_dir()])

    for subfolder in subfolders:
        merge_index_recursively(subfolder)

    if subfolders and all((subfolder / index_filename).is_file() for subfolder in subfolders):
        # Add a lock to avoid multiple jobs writing to the same index.json
        with FileLock(directory / (index_filename + ".lock")):
            index_file = directory / index_filename
            if index_file.exists():
                assert index_file.is_file(), f"{index_file} must be a file"
                index_file.unlink()

            merge_index(directory)

