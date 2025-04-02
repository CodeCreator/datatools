import pandas as pd

import numpy as np
from typing import Optional, List, Union
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Pool

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


def load_csv(path: Union[Path, str]):
    from datasets import load_dataset
    if isinstance(path, Path):
        # HY: load_dataset expects a string path
        path = str(path)

    if "tsv" in path:
        return load_dataset("csv", data_files=path, delimiter="\t")['train']
    else:
        return load_dataset("csv", data_files=path)['train']


def load_hf_dataset(path: Union[Path, str], input_type: str):
    from datasets import load_from_disk, Dataset
    path = str(path)

    return {
        "hf": load_from_disk,
        "arrow": Dataset.from_file,
        "parquet": Dataset.from_parquet,
        "hub": load_from_hub,
        "csv": load_csv,
        "tsv": load_csv,
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
        elif path.is_file():
            # Best guess from file extension
            # Iterate over suffixes in reverse order to handle cases like .jsonl.zst
            for suffix in path.suffixes[::-1]:
                if suffix in [".arrow", ".parquet", ".npy", ".jsonl", ".tsv", ".csv"]:
                    input_type = suffix[1:]
                    break
        elif not path.exists():
            # HY: if the path does not exist (not a file or directory), we assume it should be loaded from hub
            input_type = "hub"

    if input_type == "mosaic":
        return LocalDatasets(input_paths)
    elif input_type == "jsonl":
        return JsonlDataset(input_paths)
    elif input_type == "npy":
        return np.concatenate([np.load(path) for path in input_paths])
    elif input_type in {"hf", "arrow", "parquet", "hub", "csv", "tsv"}:
        from datasets import concatenate_datasets
        return concatenate_datasets([load_hf_dataset(path, input_type) for path in input_paths])
    else:
        if input_type is None and not Path(input_paths[0]).exists():
            raise ValueError(f"Could not infer input type from non-existent local path: {input_paths[0]}")
        raise ValueError(f"Unknown input type: {input_type}")

def load_pandas(*input_paths: List[Union[Path, str]], n_rows: int = -1, num_proc: int = 1, options: Optional[LoadOptions] = None) -> pd.DataFrame:
    dataset = load(*input_paths, options=options)
    n_rows = n_rows if n_rows > 0 else len(dataset)

    if num_proc > 1:
        with Pool(num_proc) as pool:
            return pd.DataFrame(pool.map(dataset.__getitem__, range(n_rows)))
    else:
        return pd.DataFrame([dataset[i] for i in range(n_rows)])

