## *datatools*

Common NLP data utilities (tokenizing, splitting, packing, ...), with special support for [Mosaic Streaming Datasets](https://docs.mosaicml.com/projects/streaming/en/stable/index.html).

#### Installation

Clone this repo and install via `pip install -e .`

#### Library

*datatools* contributes some core libraries that can be used to easily build custom data pipelines, specifically `from datatools import load, process`.

* `load(path, load_options)`: loads the dataset at the path, while trying to infer what format it is in (e.g., compressed json, pyarrow, MDS, ...) based on clues from the file format and directory structure.
* `process(input_dataset, process_fn, output_path, process_options)`: processes an input dataset and writes the results to disk. It supports *multi-processing* and *slurm array parallelization* and *custom indexing*, see [ProcessOptions](https://github.com/CodeCreator/datatools/blob/main/datatools/process.py#L30) for details. `process_fn` is a function that should take up to three arguments: (1) a subset of the data with `len(...)` and `.[...]` access, (2) the global indices corresponding to the subset, and (3) the `process_id` for logging purposes.

#### Scripts

*datatools* comes with the following default script:

* `tokenize`: tokenize datasets per document
* `pack`: pack tokenized documents into fixed sequences
* `peek`: print datasets as json to stdout
* `wrangle`: subsample, merge datasets, make random splits (e.g., train/test/validation), etc...
* `merge_index`: merge mosaic streaming datasets in subfolders to a big dataset

Run `<script> --help` for detailed arguments! Many scripts automatically add all arguments from `ProcessOptions` (e.g. number of processes `-w <processes>`) and `LoadOptions`.
