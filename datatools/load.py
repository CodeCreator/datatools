import pandas as pd
import numpy as np
from typing import Optional, List, Union
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from upath import UPath

from datatools.io_utils import LocalDatasets, JsonlDataset, is_remote_path, has_compressed_mds_files, RemoteDatasets


@dataclass
class LoadOptions:
    """Options for load function"""

    # Type of data {mds, hf, jsonl, mds, parquet, arrow, hub}. Default: Infer from files
    input_type: Optional[str] = None


def load_from_hub(path: str):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for loading from Hugging Face Hub. "
            "Install it with: pip install datatools-py[datasets] or pip install datasets"
        )
    # split a path like path/to/dataset>name#split into path, name, split
    rest, *split = path.rsplit("#", 1)
    path, *name = rest.split(">", 1)
    return load_dataset(path, name=(name[0] if name else None), split=(split[0] if split else None))


def load_hf_dataset(path: UPath, input_type: str):
    """Load Hugging Face dataset from UPath."""
    try:
        from datasets import load_from_disk, Dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for loading Hugging Face datasets. "
            "Install it with: pip install datatools-py[datasets] or pip install datasets"
        )

    return {
        "hf": load_from_disk,
        "arrow": Dataset.from_file,
        "parquet": Dataset.from_parquet,
        "hub": load_from_hub,
    }[input_type](str(path))


def load(*input_paths: List[Union[UPath, str]], options: Optional[LoadOptions] = None) -> Sequence:
    assert len(input_paths) > 0, "Empty input list"

    input_paths = [UPath(path) for path in input_paths]
    options = options or LoadOptions()
    input_type = options.input_type
    
    if input_type is None:
        path = input_paths[0]
        if path.is_dir() and (path / "index.json").is_file():
            input_type = "mosaic"
        elif path.is_dir() and (path / "state.json").is_file():
            input_type = "hf"
        elif path.is_file():
            # Best guess from file extension
            # Iterate over suffixes in reverse order to handle cases like .jsonl.zst
            for suffix in path.suffixes[::-1]:
                if suffix in [".arrow", ".parquet", ".npy", ".jsonl"]:
                    input_type = suffix[1:]
                    break

    if input_type == "mosaic":
        if any(is_remote_path(path) or has_compressed_mds_files(path) for path in input_paths):
            return RemoteDatasets(input_paths)
        else:
            return LocalDatasets(input_paths)
    elif input_type == "jsonl":
        return JsonlDataset(input_paths)
    elif input_type == "npy":
        return np.concatenate([np.load(str(path)) for path in input_paths])
    elif input_type in {"hf", "arrow", "parquet", "hub"}:
        try:
            from datasets import concatenate_datasets
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for loading Hugging Face datasets. "
                "Install it with: pip install datatools-py[datasets] or pip install datasets"
            )
        return concatenate_datasets([load_hf_dataset(path, input_type) for path in input_paths])
    else:
        if input_type is None and not input_paths[0].exists():
            raise ValueError(f"Could not infer input type from non-existent local path: {input_paths[0]}")
        raise ValueError(f"Unknown input type: {input_type}")


def load_pandas(*input_paths: List[Union[UPath, str]], n_rows: int = -1, num_proc: int = 1, options: Optional[LoadOptions] = None) -> pd.DataFrame:
    dataset = load(*input_paths, options=options)
    n_rows = n_rows if n_rows > 0 else len(dataset)

    if num_proc > 1:
        with Pool(num_proc) as pool:
            return pd.DataFrame(pool.map(dataset.__getitem__, range(n_rows)))
    else:
        return pd.DataFrame([dataset[i] for i in range(n_rows)])

