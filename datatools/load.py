import os

from typing import Optional, List, Union
from collections.abc import Sequence
from dataclasses import dataclass

from pathlib import Path

from datatools.io_utils import LocalDatasets, JsonlDataset

@dataclass
class LoadOptions:
    """Options for load function"""

    # Type of data {mds, hf, jsonl, mds, parquet, arrow, hub}. Default: Infer from files
    input_type: Optional[str] = None


def load_from_hub(path: str):
    from datasets import load_dataset
    # split a path like path/to/dataset>name#split into path, name, split
    rest, *split = path.rsplit("#", 1)
    path, *name = rest.split(">", 1)
    return load_dataset(path, name=(name[0] if name else None), split=(split[0] if split else None))


def load_hf_dataset(path: Union[Path, str], input_type: str):
    from datasets import load_from_disk, Dataset
    path = str(path)

    return {
        "hf": load_from_disk,
        "arrow": Dataset.from_file,
        "parquet": Dataset.from_parquet,
        "hub": load_from_hub,
    }[input_type](path)


def load(*input_paths: List[Union[Path, str]], options: Optional[LoadOptions] = None) -> Sequence:
    assert len(input_paths) > 0, "Empty input list"

    options = options or LoadOptions()
    input_type = options.input_type

    if input_type is None:
        path = Path(input_paths[0])
        if path.is_dir() and (path / "index.json").is_file():
            input_type = "mosaic"
        elif path.is_dir() and (path / "state.json").is_file():
            input_type = "hf"
        for suffix in [".jsonl", ".arrow", ".parquet"]:
            if path.is_file() and suffix in path.suffixes:
                input_type = suffix[1:]

    if input_type == "mosaic":
        return LocalDatasets(input_paths)
    elif input_type == "jsonl":
        return JsonlDataset(input_paths)
    elif input_type in {"hf", "arrow", "parquet", "hub"}:
        from datasets import concatenate_datasets
        return concatenate_datasets([load_hf_dataset(path, input_type) for path in input_paths])
    else:
        if input_type is None and not Path(input_paths[0]).exists():
            raise ValueError(f"Could not infer input type from non-existent local path: {input_paths[0]}")
        raise ValueError(f"Unknown input type: {input_type}")

