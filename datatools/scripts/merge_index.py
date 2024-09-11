from pathlib import Path

from functools import partial, reduce

from simple_parsing import ArgumentParser

from datatools.merge_index import merge_index_recursively


def main():
    parser = ArgumentParser()

    parser.add_argument("paths", type=Path, nargs="+", help="Input dataset paths")

    args = parser.parse_args()

    print("Arguments:", args)
    for path in args.paths:
        print(f"Merging {path}")
        merge_index_recursively(path)


if __name__ == "__main__":
    main()