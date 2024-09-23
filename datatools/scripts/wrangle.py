from tqdm import tqdm

from typing import Dict, Optional, List
from copy import copy

import numpy as np
from pathlib import Path

from functools import partial

from simple_parsing import ArgumentParser, field

from datatools import load, LoadOptions, process, ProcessOptions, load_indices, merge_index_recursively
from streaming.base.array import Array

from dataclasses import dataclass


@dataclass
class WrangleOptions:
    """Options for data wrangling"""

    splits: Optional[List[str]] = field(alias="-s", default=None)
    proportions: List[float] = field(alias="-p", default_factory=lambda: [1.0])
    examples: Optional[List[int]] = field(alias="-n", default=None)

    randomize: bool = field(alias="-r", default=True)
    seed: int = 42

    join: List[List[Path]] = field(alias="-J", required=False, default_factory=lambda: [], action="append", type=Path, nargs="+") # Other input dataset paths that will be joined along the index
    prefix: List[str] = field(alias="-P", required=False, default_factory=lambda: [], action="append", type=str, nargs=None) # Prefix for the fields of the dataset
    suffix: List[str] = field(alias="-S", required=False, default_factory=lambda: [], action="append", type=str, nargs=None) # Suffix for the fields of the dataset

    def __post_init__(self):
        assert len(self.prefix) == 0 or len(self.prefix) == len(self.join), "Number of prefixes must match number of join datasets or be empty"
        assert len(self.suffix) == 0 or len(self.suffix) == len(self.join), "Number of suffixes must match number of join datasets or be empty"


def join_fn(data: Array,
            indices: Array,
            process_id: int,
            join_datasets: List[Array],
            prefixes: List[str],
            suffixes: List[str]):

    for i in tqdm(range(len(data)), disable=process_id != 0):
        item = data[i]
        for j, join_dataset in enumerate(join_datasets):
            join_item = join_dataset[indices[i]]
            for key, value in join_item.items():
                item[prefixes[j] + key + suffixes[j]] = value
        yield item


def split_fn(data: Array,
             indices: Array,
             process_id: int,
             partitions: Dict[str, int],
             seed: int):

    # On each worker: split the entire dataset into train, validation, and test sets
    partition_id = np.repeat(np.arange(3), list(partitions.values()))
    partition_names = list(partitions.keys())
    np.random.seed(seed)
    np.random.shuffle(partition_id)

    for i in tqdm(range(len(data)), disable=process_id!=0):
        yield partition_names[partition_id[indices[i]]], data[i]


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(WrangleOptions, dest="wrangle_options")
    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()
    options = args.wrangle_options

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    print(f"Loaded dataset with {len(dataset)} samples")

    prefixes = options.prefix if options.prefix else [""]*len(options.join)
    suffixes = options.suffix if options.suffix else [""]*len(options.join)

    indices = load_indices(args.process_options)
    if indices is None:
        indices = np.arange(len(dataset))
    N = len(indices)

    join_datasets = [
        load(*join_paths, options=args.load_options)
        for join_paths in options.join
    ]

    if options.splits is not None:
        if options.examples is not None:
            assert len(options.examples) == len(options.splits), "Number of examples must match number of splits"
            index_ranges = [0] + np.cumsum(options.examples).tolist()
        else:
            assert len(options.proportions) == len(options.splits)
            proportions = np.array(options.proportions)
            index_ranges = [0] + np.cumsum(np.round(proportions * N).astype(int)).tolist()
        splits = options.splits
    else:
        if options.examples is not None:
            assert len(options.examples) == 1, "Only one number of examples can be specified"
            index_ranges = [0, options.examples[0]]
        else:
            assert len(options.proportions) == 1
            index_ranges = [0, round(options.proportions[0] * N)]
        splits = [""]

    if options.randomize:
        np.random.seed(options.seed)
        indices = indices[np.random.permutation(len(indices))]

    for split, (start, end) in zip(splits, zip(index_ranges[:-1], index_ranges[1:])):
        print(f"Processing split {split} with {end - start} examples")

        process_options = copy(args.process_options)
        process_options.indices = indices[start:end]
        process_options.index_path = None
        process_options.index_range = None

        process(dataset,
                partial(join_fn,
                        join_datasets=join_datasets,
                        prefixes=prefixes,
                        suffixes=suffixes),
                args.output / split,
                process_options)

    merge_index_recursively(args.output)


if __name__ == "__main__":
    main()
