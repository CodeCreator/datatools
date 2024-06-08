from tqdm import tqdm

from typing import Dict

import numpy as np
from pathlib import Path

from functools import partial

from simple_parsing import ArgumentParser

from datatools.load import load, LoadOptions
from datatools.process import process, ProcessOptions
from datatools.utils import Subset


def split_fn(subset: Subset,
             indices: Subset,
             process_id: int,
             partitions: Dict[str, int],
             seed: int):

    # On each worker: split the entire dataset into train, validation, and test sets
    partition_id = np.repeat(np.arange(3), list(partitions.values()))
    partition_names = list(partitions.keys())
    np.random.seed(seed)
    np.random.shuffle(partition_id)

    for i in tqdm(range(len(subset)), disable=process_id!=0):
        yield partition_names[partition_id[indices[i]]], subset[i]


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_argument("--validation", type=float, default=0.00, help="Proportion of validation set")
    parser.add_argument("--test", type=float, default=0.05, help="Proportion of test set")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()

    assert args.test + args.validation <= 1, "Test and validation set proportions must not exceed 1"

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    num_validation = int(args.validation * N)
    num_test = int(args.test * N)
    partitions = {
        "train": N - num_test - num_validation,
        "validation": num_validation,
        "test": num_test
    }

    process(dataset,
            partial(split_fn, partitions=partitions, seed=args.seed),
            args.output,
            args.process_options)


if __name__ == "__main__":
    main()