import json
import os
import io

from typing import Any, Dict, Union, List
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from datetime import datetime

from streaming.base.array import Array
from streaming.base.format import get_index_basename, reader_from_json
from streaming.base.spanner import Spanner

import zstandard
from contextlib import contextmanager


class ZstdUtf8WriteFile:
    def __init__(self, filename, level=3):
        self.filename = filename
        self.level = level
        self.file = None

    def open(self):
        self.file = open(self.filename, "wb")
        self.compressor = zstandard.ZstdCompressor(level=self.level)
        self.writer = self.compressor.stream_writer(self.file)
        self.text_writer = io.TextIOWrapper(self.writer, encoding='utf-8')
        return self.text_writer

    def close(self):
        self.text_writer.flush()
        self.writer.flush(zstandard.FLUSH_FRAME)
        self.file.close()


@contextmanager
def zstd_utf8_read_open(filename, level=3):
    with open(filename, "rb") as f:
        decompressor = zstandard.ZstdDecompressor(max_window_size=2147483648)
        with decompressor.stream_reader(f) as stream_reader:
            yield io.TextIOWrapper(stream_reader, encoding='utf-8')


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
        return self.dataset[int(self.indices[int(idx)])]


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

        self.lines = []
        for path in paths:
            path = Path(path)
            if path.suffixes[-1] in [".zstd", ".zst"]:
                with zstd_utf8_read_open(path) as f:
                    self.lines.extend(f.readlines())
            else:
                with open(path) as f:
                    self.lines.extend(f.readlines())

    def __len__(self) -> int:
        return len(self.lines)

    @property
    def size(self) -> int:
        return len(self.lines)

    def get_item(self, idx: int) -> Dict[str, Any]:
        return json.loads(self.lines[idx])



class NDArrayWriter:
    def __init__(self, columns, out, compression=None):
        self.columns = columns
        self.out = out

        self.buffers = {}

    def write(self, item):
        for column in self.columns:
            if column not in self.buffers:
                self.buffers[column] = []
            self.buffers[column].append(item[column])

    def finish(self):
        if self.buffers:
            os.makedirs(os.path.dirname(self.out), exist_ok=True)
            for column, buffer in self.buffers.items():
                if column:
                    np.save(f"{self.out}__{column}.npy", np.array(buffer))
                else:
                    np.save(f"{self.out}.npy", np.array(buffer))




class DatetimeJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return str(obj)
        if isinstance(obj, np.number):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


class JsonlWriter:
    def __init__(self, columns, out, compression=None):
        self.columns = set(columns)
        self.out = out

        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        if compression is not None and compression.startswith("zst"):
            ext = compression.split(":")[0]
            level = int(compression.split(":")[1]) if ":" in compression else 3
            self.file_handle = ZstdUtf8WriteFile(f"{out}.jsonl.{ext}", level)
            self.file = self.file_handle.open()
        else:
            self.file_handle = open(f"{out}.jsonl", "w")
            self.file = self.file_handle

    def write(self, item):
        if not self.columns.issubset(item.keys()):
            print(f"Warning: Item {item} does not contain all columns: {self.columns - item.keys()}")
        self.file.write(json.dumps(item, cls=DatetimeJsonEncoder) + "\n")

    def finish(self):
        self.file_handle.close()