from functools import partial

from tqdm import tqdm

from typing import Optional

import numpy as np
from pathlib import Path

from simple_parsing import ArgumentParser, field
from dataclasses import dataclass

from datatools.load import load, LoadOptions
from datatools.process import process, ProcessOptions
from streaming.base.array import Array


@dataclass
class TokenizeOptions:
    """Options for tokenizing"""

    # Can use "llama2" or "llama3" for Llama models
    # Otherwise HF tokenizer name
    tokenizer: str = field(alias=["-T"], default="llama3")

    truncate_bytes: int = 10_000_000_000

    domain: str = ""
    domain_by: Optional[str] = None

    text_field: str = "text"
    token_field: str = "input_ids"
    length_field: str = "length"
    domain_field: str = "domain"


def load_tokenizer_encoder(options: TokenizeOptions):
    if options.tokenizer == "llama2":
        from datatools.scripts.tokenizers.llama2_tokenizer import Tokenizer
        tokenizer = Tokenizer(str(Path(__file__).parent / "tokenizers" / "llama2_tokenizer.model"))
        return partial(tokenizer.encode, bos=False, eos=False)
    elif options.tokenizer == "llama3":
        from datatools.scripts.tokenizers.llama3_tokenizer import Tokenizer
        tokenizer = Tokenizer(str(Path(__file__).parent / "tokenizers" / "llama3_tokenizer.model"))
        return partial(tokenizer.encode, bos=False, eos=False)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(options.tokenizer)
        return partial(tokenizer.encode, add_special_tokens=False)


def tokenize_fn(data: Array,
                indices: Array,
                process_id: int,
                options: TokenizeOptions):

    encode_fn = load_tokenizer_encoder(options)

    for i in tqdm(range(len(data)), disable=(process_id != 0)):
        item = data[i]
        tokens = encode_fn(item[options.text_field][:options.truncate_bytes])
        domain = item[options.domain_by] if options.domain_by is not None else options.domain

        output_item = {
            options.token_field: np.array(tokens, dtype=np.uint32),
        }
        if options.length_field:
            output_item[options.length_field] = len(tokens)
        if options.domain_field:
            output_item[options.domain_field] = domain

        yield Path(), output_item


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")
    parser.add_arguments(TokenizeOptions, dest="tokenize_options")

    args = parser.parse_args()

    print("Arguments:", args)
    dataset = load(*args.inputs, options=args.load_options)
    N = len(dataset)
    print(f"Loaded dataset with {N} samples")

    process(dataset,
            partial(tokenize_fn, options=args.tokenize_options),
            args.output,
            args.process_options)


if __name__ == "__main__":
    main()