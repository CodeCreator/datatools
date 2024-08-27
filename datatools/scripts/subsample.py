from tqdm import tqdm

from typing import Dict

import numpy as np
from pathlib import Path

from functools import partial

from simple_parsing import ArgumentParser

from datatools.load import load, LoadOptions
from datatools.process import process, ProcessOptions, identity_fn
from streaming.base.array import Array


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_argument("-p", type=float, help="Proportion of test set")
    parser.add_argument("-n", type=int, help="Proportion of test set")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()
    
    assert args.p or args.n, "Must either specify -n or -p"

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")
    
    n = args.n if args.n is not None else int(args.p * N)

    np.random.seed(args.seed)
    indices = np.random.permutation(N)[:n]
    args.process_options.indices = indices

    process(dataset,
            identity_fn,
            args.output,
            args.process_options)


if __name__ == "__main__":
    main()
