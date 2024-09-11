import json
import os
import io

from typing import Any, Dict, Union, List
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from streaming.base.array import Array
from streaming.base.format import get_index_basename, reader_from_json
from streaming.base.spanner import Spanner


import zstandard
from filelock import FileLock



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

