from tqdm import tqdm

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from functools import partial, reduce

from dataclasses import dataclass
from simple_parsing import ArgumentParser, field

from collections import defaultdict

from datatools.load import load, LoadOptions
from datatools.process import process, ProcessOptions
from streaming.base.array import Array


@dataclass
class PackOptions:
    """Options for packing"""

    max_length: int = field(alias=["-l"], default=8192)
    min_length: int = field(alias=["-s"], default=1)
    overlap: int = field(alias=["-o"], default=0)

    # Pack sequences into separate subset of fixed lengths
    # This will always pack to the longest available lengths,
    # Note that it still requires setting max_length
    split_by_lengths: List[int] = field(alias=["-f"], default=None)


    # Only include a single sequence per document
    single: bool = False

    bfd: bool = False
    bfd_num_bins: int = 50

    add_special_tokens: bool = True
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None
    mos_id: Optional[int] = None  # middle-of-sentence token (<bos> for mid-document)
    use_for_mos: Optional[str] = field(choices=["bos", "eos"], default=None)

    # Use preset tokenizer config to set bos_id and eos_id
    tokenizer: Optional[str] = field(alias=["-T"], default=None)


    sort_by_length: bool = False

    split_by_column: str = None

    token_field: str = "input_ids"
    length_field: str = "length"
    indices_field: str = "indices"
    domain_field: str = "domain"

    def special_tokens_from_tokenizer(self, tokenizer):
        if tokenizer == "llama3":
            return 128000, 128001
        elif tokenizer == "llama2":
            return 1, 2
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            return tokenizer.bos_token_id, tokenizer.eos_token_id


    def __post_init__(self):
        if self.tokenizer is not None:
            bos_id, eos_id = self.special_tokens_from_tokenizer(self.tokenizer)
            if self.bos_id is None:
                self.bos_id = bos_id
            if self.eos_id is None:
                self.eos_id = eos_id

        if self.use_for_mos is not None:
            if self.use_for_mos == "bos":
                self.mos_id = self.bos_id
            elif self.use_for_mos == "eos":
                self.mos_id = self.eos_id
            else:
                raise ValueError(f"Invalid value for use_for_mos: {self.use_for_mos}")


def add_special_tokens(tokens: NDArray[np.uint32], options: PackOptions, bos=False, eos=False, mos=False):
    if not options.add_special_tokens:
        return tokens

    tokens_list = []

    # Add bos_id if not already present and if sequence is not empty
    if bos and options.bos_id is not None and len(tokens) > 0 and tokens[0] != options.bos_id:
        tokens_list.append(np.array([options.bos_id], dtype=tokens.dtype))
    elif mos and options.mos_id is not None and len(tokens) > 0 and tokens[0] != options.mos_id:
        tokens_list.append(np.array([options.mos_id], dtype=tokens.dtype))

    tokens_list.append(tokens)

    if eos and options.eos_id is not None and len(tokens) > 0 and tokens[-1] != options.eos_id:
        tokens_list.append(np.array([options.eos_id], dtype=tokens.dtype))

    return np.concatenate(tokens_list, axis=0)


class SingleBuffer:
    def __init__(self, options: PackOptions):
        self.options = options
        self.max_length = self.options.max_length
        self.min_length = self.options.min_length

        self.token_buffer = []

        self.num_tokens = 0

    def process(self, tokens: NDArray):
        self.token_buffer.append(tokens)
        self.num_tokens += len(tokens)

        while self.num_tokens >= self.max_length:
            tokens = np.concatenate(self.token_buffer, 0)
            item = {
                self.options.token_field: tokens[:self.max_length]
            }
            if self.options.length_field:
                item[self.options.length_field] = self.max_length
            if self.options.indices_field:
                item[self.options.indices_field] = self.compute_indices(self.token_buffer)

            yield item

            self.token_buffer = []
            self.num_tokens = 0
            if not self.options.single:
                tokens = add_special_tokens(tokens[self.max_length - self.options.overlap:], self.options, mos=True)

                if len(tokens) >= self.min_length:
                    self.token_buffer.append(tokens)
                    self.num_tokens += len(tokens)

    def compute_indices(self, token_buffer: List[NDArray]):
        indices = []
        start = 0
        for seq in token_buffer:
            indices.append((start, min(start+len(seq), self.max_length)))
            start += len(seq)
            if start >= self.max_length:
                break
        return np.array(indices, dtype=np.uint32)

    @property
    def available(self):
        return self.max_length - self.num_tokens


# Best-fit-decreasing algorithm from "Fewer Truncations Improve Language Modeling"
class BFDBuffer:
    def __init__(self, options: PackOptions):
        self.buffers = [SingleBuffer(options) for _ in range(options.bfd_num_bins)]

    def add(self, tokens: NDArray):
        available_buffers = [buffer for buffer in self.buffers if buffer.available >= len(tokens)]

        if available_buffers:
            # Identify the smallest available buffer
            buffer = min(available_buffers, key=lambda x: x.available)
            yield from buffer.process(tokens)
        else:
            # Or if no buffer is available,
            # identify the buffer with the largest available space
            buffer = max(self.buffers, key=lambda x: x.available)
            yield from buffer.process(tokens)


def pack_fn(data: Array,
            indices: Array,
            process_id: int,
            options: PackOptions):
    buffer_cls = BFDBuffer if options.bfd else SingleBuffer
    buffers = defaultdict(partial(buffer_cls, options=options))

    indices = list(range(len(data)))
    if options.sort_by_length:
        indices.sort(key=lambda x: len(data[x]['input_ids']), reverse=True)

    if options.split_by_lengths:
        sorted_lengths = sorted(options.split_by_lengths, reverse=True)

    for i in tqdm(indices, disable=process_id != 0):
        item = data[i]

        # Then identify the mapped subset
        subset = Path()
        if options.split_by_column:
            if "." in options.split_by_column:
                dict_fields = options.split_by_column.split(".")
                subset = subset / reduce(dict.get, [item, *dict_fields])
            else:
                subset = subset / item[options.split_by_column]

        # Concatenate bos/eos
        input_ids = add_special_tokens(np.array(item['input_ids'], dtype=np.uint32), options, bos=True, eos=True)

        if options.split_by_lengths:
            while len(input_ids) >= sorted_lengths[-1]:
                # From longest to shortest
                target_len = next(target_len
                                  for target_len in sorted_lengths
                                  if len(input_ids) >= target_len)

                target_subset = subset / f"{target_len}-{options.max_length}"

                for item in buffers[target_len].process(input_ids[:target_len]):
                    if options.domain_field:
                        item.update({options.domain_field: str(target_subset)})
                    yield target_subset, item

                if options.single:
                    break

                input_ids = add_special_tokens(input_ids[target_len - options.overlap:], options, mos=True)
        else:
            for item in buffers[subset].process(input_ids):
                if options.domain_field:
                    item.update({options.domain_field: str(subset)})
                yield subset, item


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(PackOptions, dest="pack_options")
    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

    args = parser.parse_args()

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    process(dataset,
            partial(pack_fn, options=args.pack_options),
            args.output,
            args.process_options)


if __name__ == "__main__":
    main()
