## *datatools*

Common NLP data utilities (tokenizing, splitting, packing, ...) designed for [Mosaic Streaming Datasets](https://docs.mosaicml.com/projects/streaming/en/stable/index.html).

#### Installation

Clone this repo and install via `pip install -e .`

#### Library

*datatools* contributes some core libraries that can be used to easily build custom data pipelines, specifically `from datatools import load, process`.


#### Scripts

*datatools* comes with the following default script:

* `tokenize`: tokenize datasets per document
* `pack`: pack tokenized documents into fixed sequences
* `peek`: print datasets as json to stdout
* `splits`: make random splits, e.g., train/test/validation
* `merge_data`: merge datasets of same length with different features
* `merge_index`: merge mosaic streaming datasets in subfolders to a big dataset


