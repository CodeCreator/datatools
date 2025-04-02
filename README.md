# 🛠️ *datatools*: Simple utilities for common data actions

Minimal scripts and reusable functions for implementing common data operations (tokenization, splitting, subsampling, packing, and more).

Built with special support for [Mosaic Streaming Datasets (MDS)](https://docs.mosaicml.com/projects/streaming/en/stable/index.html).

## Table of contents
- [Installation](#installation)
- [Library](#library)
  - [Core Functions](#core-functions)
  - [Example](#example)
- [Scripts](#scripts)

## Installation

Clone this repo and install via `pip install -e .` or install from PyPI via `pip install datatools-py`.

## Library

*datatools* provides core libraries that can be used to easily build custom data pipelines, specifically through `from datatools import load, process`.

### Core functions

```python
load(path, load_options)
```
Loads the dataset at the path and **automatically infers its format** (e.g., compressed JSON, PyArrow, MDS, etc.) based on clues from the file format and directory structure.

---

```python
process(input_dataset, process_fn, output_path, process_options)
```
Processes an input dataset and writes the results to disk. It supports:

1. **Multi-processing** with many CPUs, e.g. `ProcessOptions(num_proc=16)` (or as flag `-w 16`)
2. **Slurm array parallelization**, e.g. `ProcessOptions(slurm_array=True)` (or `--slurm_array`) automatically sets up `job_id` and `num_jobs` using Slurm environment variables
3. **Custom indexing**, e.g. only working on a subset `--index_range 0 30` or using a custom index file `--index_path path/to/index.npy`
   See [ProcessOptions](https://github.com/CodeCreator/datatools/blob/main/datatools/process.py#L30) for details.
4. By default, output is written as mosaic-streaming MDS shards, which are merged into a single MDS dataset when the job finishes. The code also supports writing to JSONL files (`--jsonl`) and ndarray files for each column (`--ndarray`). The shards for these output formats are not automatically merged.

The `process_fn` should be a function that takes one to three arguments:
1. A subset of the data with `len(...)` and `.[...]` access
2. The global indices corresponding to the subset (optional)
3. The `process_id` for logging or sharding purposes (optional)

### Example

```python
from datatools import load, process, ProcessOptions
from transformers import AutoTokenizer

# Load dataset (can be JSON, Parquet, MDS, etc.)
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

## Scripts

*datatools* comes with the following default scripts:

* `tokenize`: Tokenize datasets per document
* `pack`: Pack tokenized documents into fixed sequences
* `peek`: Print datasets as JSON to stdout
* `wrangle`: Subsample, merge datasets, make random splits (e.g., train/test/validation), etc.
* `merge_index`: Merge Mosaic streaming datasets in subfolders into a larger dataset

Run `<script> --help` for detailed arguments. Many scripts automatically include all arguments from `ProcessOptions` (e.g., number of processes `-w <processes>`) and `LoadOptions`.
