from contextlib import contextmanager

from datasets import Dataset
from pathlib import Path

import pprint

import random
import json
import subprocess
import sys

from streaming import LocalDataset

from datatools.io_utils import LocalDatasets, DatetimeJsonEncoder
from datatools.load import load, LoadOptions

from simple_parsing import ArgumentParser


@contextmanager
def redirect_stdout_to_stderr():
    old_stdout = sys.stdout
    old_stdout.flush()
    sys.stdout = sys.stderr
    yield
    sys.stdout = old_stdout


@contextmanager
def jq_printer(compact=False):
    if compact:
        jq = subprocess.Popen(["jq", "-c", "."], stdin=subprocess.PIPE)
    else:
        jq = subprocess.Popen(["jq", "."], stdin=subprocess.PIPE)
    yield jq.stdin
    jq.stdin.close()
    jq.wait()


def dataset_summary(dataset):
    if isinstance(dataset, Dataset):
        features_dict = dataset._info.features.to_dict()
        dataset_type = "HuggingfaceDataset"
    elif isinstance(dataset, (LocalDataset, LocalDatasets)):
        features_dicts = [dict(zip(shard.column_names, shard.column_encodings)) for shard in dataset.shards]
        assert all(features_dicts[0] == features for features in features_dicts), "All shards must have the same features"
        features_dict = features_dicts[0]
        dataset_type = "MosaicDataset"
    else:
        features_dict = None
        dataset_type = "UnknownDataset"

    if features_dict is not None:
        features = pprint.pformat(features_dict, sort_dicts=False).replace("\n", "\n" + " "*14)
    else:
        features = "unknown"

    return f"{dataset_type}({{\n    length: {len(dataset)},\n    features: {features}\n}})"


def head(dataset, head):
    if head is not None:
        if head > 0:
            indices = range(0, min(head, len(dataset)))
        elif head < 0:
            indices = range(max(0, len(dataset) + head), len(dataset))
        else:
            indices = range(0, 0)
        return dataset.select(indices)
    else:
        return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("datasets",
                        type=Path,
                        nargs="+",
                        help="Dataset name or path to dataset. If multiple, will be concatenated.")

    parser.add_argument("-n",
                        type=int, default=None,
                        help="Keep top n examples (after shuffling), negative for bottom n")
    parser.add_argument("-r", "--raw",
                        action="store_true",
                        help="Print raw data instead of json")
    parser.add_argument("-c", "--compact",
                        action="store_true",
                        help="Print compact entries (jsonl)")

    parser.add_argument("-x", "--shuffle", "--randomize",
                        action="store_true",
                        help="Shuffle data")
    parser.add_argument("-s", "--seed",
                        default=42,
                        type=int,
                        help="Seed for shuffling data")
    parser.add_argument("--sort",
                        type=str,
                        default=None,
                        help="Sort the dataset by the given columns, prefix a column with ^ to sort in descending order.")

    parser.add_arguments(LoadOptions, dest="load_options")

    with redirect_stdout_to_stderr():
        args = parser.parse_args()

        dataset = load(*args.datasets, options=args.load_options)

        # Print dataset summary
        dataset_desc = dataset_summary(dataset)
        dataset_desc = dataset_desc.replace("\n", "\n| ")
        print(f"\033[0;31m| {dataset_desc}\033[0m")

        # Sort by field
        if args.sort is not None or args.shuffle:
            indices = list(range(len(dataset)))

            if args.shuffle:
                assert args.sort is None, "Cannot both shuffle and sort"
                random.seed(args.seed)
                random.shuffle(indices)
            else:
                column = args.sort.startswith("^") and args.sort[1:] or args.sort
                descending = args.sort.startswith("^")
                values = [dataset[i][column] for i in indices]
                indices.sort(key=lambda i: values[i], reverse=descending)
        else:
            indices = range(len(dataset))

        # Select head/tail
        if args.n is not None:
            if args.n > 0:
                indices = indices[:min(args.n, len(indices))]
            elif args.n < 0:
                indices = indices[max(0, len(indices) + args.n):]
            else:
                indices = indices[:0]

    if args.raw:
        for i in indices:
            print(dataset[i])
    else:
        with jq_printer(args.compact) as jq:
            for i in indices:
                jq.write(json.dumps(dataset[i], cls=DatetimeJsonEncoder).encode("utf-8"))


if __name__ == "__main__":
    main()
