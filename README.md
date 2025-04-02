## *datatools*: simple utilities for common data operations

Scripts and reusable functions for implementing common data operations (tokenization, splitting, subsampling, packing, ...)

Built with special support for [Mosaic Streaming Datasets (MDS)](https://docs.mosaicml.com/projects/streaming/en/stable/index.html).

#### Installation

Clone this repo and install via `pip install -e .` or install from pypi via `pip install datatools-py`.

#### Library

*datatools* contributes some core libraries that can be used to easily build custom data pipelines, specifically `from datatools import load, process`.

###### `load(path, load_options)` 
Loads the dataset at the path _**tries to infer what format it is in**_ (e.g., compressed json, pyarrow, MDS, ...) based on clues from the file format and directory structure

For loading from datasets hosted on huggingface hub, you can use the `hub` input type and specify the dataset name as `path/to/dataset>name#split`. For example, `load("tatsu-lab/alpaca_eval>alpaca_eval#eval")` is equivalent to `datasets.load_dataset("tatsu-lab/alpaca_eval", split="eval")`.

###### `process(input_dataset, process_fn, output_path, process_options)`

Processes an input dataset and writes the results to disk. It supports:
1. Multi-processing with many CPUs, e.g. `ProcessOptions(num_proc=16)` (or as flag `-w 16`)
2. Slurm array parallelization, e.g. `ProcessOptions(slurm_array=True)` (or `--slurm_array`) automatically sets up `job_id` and `num_jobs` using slurm environment variables
3. Custom indexing, e.g. only working on a subset `--index_range 0 30` or using a custom index file `--index_path path/to/index.npy`
See [ProcessOptions](https://github.com/CodeCreator/datatools/blob/main/datatools/process.py#L30) for details.
4. By default we write the output file as mosaic-streaming MDS shards, which we merge into a single MDS dataset when the job finishes. However, the code also supports writing to JSONL files (`--jsonl`) and ndarray files for each column (`--ndarray`). The shards for these output formats are not automatically merged.

The `process_fn` should be a function takes one to three arguments:
1. A subset of the data with `len(...)` and `.[...]` access
2. The global indices corresponding to the subset (optionally)
3. The `process_id` for logging or sharding purposes (optionally)

##### Example

```python
from datatools import load, process, ProcessOptions
from transformers import AutoTokenizer

# Load dataset (can be json, parquet, MDS, etc.)
dataset = load("path/to/dataset")

# Setup tokenizer and processing function
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
def tokenize_docs(data_subset):
    for item in data_subset:
        # Tokenize text and return dict with tokens and length
        tokens = tokenizer.encode(item["text"], add_special_tokens=False)
        
        # Chunk the text into 1024 token chunks
        for i in range(0, len(tokens), 1024):
            yield {
                "input_ids": tokens[i:i+1024],
                "length": len(tokens[i:i+1024])
            }

# Process dataset with 4 workers and write to disk
process(dataset, tokenize_docs, "path/to/output", process_options=ProcessOptions(num_proc=4))
```

#### Scripts

*datatools* comes with the following default script:

* `tokenize`: tokenize datasets per document
* `pack`: pack tokenized documents into fixed sequences
* `peek`: print datasets as json to stdout
* `wrangle`: subsample, merge datasets, make random splits (e.g., train/test/validation), etc...
* `merge_index`: merge mosaic streaming datasets in subfolders to a big dataset

Run `<script> --help` for detailed arguments! Many scripts automatically add all arguments from `ProcessOptions` (e.g. number of processes `-w <processes>`) and `LoadOptions`.
