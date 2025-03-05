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

    domain: str = ""
    domain_by: Optional[str] = None

    # Either specify a chat template or a template file
    chat_template: bool = False
    chat_messages_field: Optional[str] = "messages"
    chat_assistant_masking: bool = True

    template_file: Optional[Path] = None
    template: str = "{text}"
    truncate_bytes: int = 10_000_000_000    #  Only usedwithout chat template

    token_field: str = "input_ids"
    length_field: str = "length"
    domain_field: str = "domain"

    def __post_init__(self):
        if self.template_file is not None:
            with Path(self.template_file).open() as f:
                self.template = f.read()


def load_tokenizer_encoder(options: TokenizeOptions):
    if options.tokenizer in ["llama2", "llama3"]:
        if options.tokenizer == "llama2":
            from datatools.scripts.tokenizers.llama2_tokenizer import Tokenizer
            tokenizer = Tokenizer(str(Path(__file__).parent / "tokenizers" / "llama2_tokenizer.model"))
        else:
            from datatools.scripts.tokenizers.llama3_tokenizer import Tokenizer
            tokenizer = Tokenizer(str(Path(__file__).parent / "tokenizers" / "llama3_tokenizer.model"))
        from datatools.scripts.tokenizers.llama3_tokenizer import ChatFormat
        
        if options.chat_template:
            chat_format = ChatFormat(tokenizer)
            def encode_fn(item):
                dialog = item[options.chat_messages_field]
                return chat_format.encode_dialog_prompt(dialog, return_assistant_masks=options.chat_assistant_masking)
            return encode_fn
        else:
            def encode_fn(item):
                text = options.template.format(**item)
                tokens = tokenizer.encode(text[:options.truncate_bytes], bos=False, eos=False)
                return tokens
            return encode_fn
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(options.tokenizer)
        assert not options.chat_assistant_masking, "Chat masking is not supported for HF tokenizers (yet)"
        if options.chat_template:
            def encode_fn(item):
                dialog = item[options.chat_messages_field]
                tokens = chat_format.apply_chat_template(dialog, truncation=False, max_length=None)
                return tokens
            return encode_fn
        else:
            def encode_fn(item):
                text = options.template.format(**item)
                tokens = tokenizer.encode(text[:options.truncate_bytes], add_special_tokens=False, truncation=False, max_length=None)
                return tokens
    return encode_fn

        


def tokenize_fn(data: Array,
                process_id: int,
                options: TokenizeOptions):

    encode_fn = load_tokenizer_encoder(options)

    for i in tqdm(range(len(data)), desc=f"Process {process_id}"):
        item = data[i]
        domain = item[options.domain_by] if options.domain_by is not None else options.domain
        
        if options.chat_template and options.chat_assistant_masking:
            tokens, masks = encode_fn(item)

            output_item = {
                options.token_field: np.array(tokens, dtype=np.uint32),
                "mask": np.array(masks, dtype=np.uint8)
            }
        else:
            tokens = encode_fn(item)
            output_item = {
                options.token_field: np.array(tokens, dtype=np.uint32),
            }
        
        if options.length_field:
            output_item[options.length_field] = len(tokens)
        if options.domain_field:
            output_item[options.domain_field] = domain

        yield output_item


def main():
    parser = ArgumentParser()

    parser.add_argument("inputs", type=Path, nargs="+", help="Input dataset paths")
    parser.add_argument("output", type=Path, help="Output dataset path")

    parser.add_arguments(TokenizeOptions, dest="tokenize_options")
    parser.add_arguments(LoadOptions, dest="load_options")
    parser.add_arguments(ProcessOptions, dest="process_options")

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
