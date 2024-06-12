from pathlib import Path

from tqdm import tqdm

from typing import List

from functools import partial, reduce

from simple_parsing import ArgumentParser

from datatools.load import load, LoadOptions
from datatools.process import process, ProcessOptions
from streaming.base.array import Array

def join_fn(data: Array,
            indices: Array,
            process_id: int,
            join_datasets: List[Array],
            prefixes: List[str],
            suffixes: List[str]):

    for i in tqdm(range(len(data)), desc=f"Process {process_id}"):
        item = data[i]
        for j, join_dataset in enumerate(join_datasets):
            join_item = join_dataset[indices[i]]
            for key, value in join_item.items():
                item[prefixes[j] + key + suffixes[j]] = value
        yield None, item


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_argument("-d", "--dataset", type=Path, nargs="+", action="append", help="Input dataset paths that will be added")
    parser.add_argument("-p", "--prefix", type=str, action="append", help="Prefix for the fields of the dataset")
    parser.add_argument("-s", "--suffix", type=str, action="append", help="Suffix for the fields of the dataset")

    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    assert args.join, "Must specify datasets to join"
    prefixes = args.prefix if args.prefix else []
    suffixes = args.suffix if args.suffix else []

    assert max(len(prefixes or []), len(suffixes or [])) <= len(args.join)


    prefixes = prefixes + [""] * (len(args.join) - len(prefixes))
    suffixes = suffixes + [""] * (len(args.join) - len(suffixes))

    join_datasets = [
        load(*join_paths, options=args.load_options) for join_paths in args.join
    ]

    process(dataset, partial(join_fn, join_datasets=join_datasets, prefixes=prefixes, suffixes=suffixes), args.output, args.process_options)


if __name__ == "__main__":
    main()