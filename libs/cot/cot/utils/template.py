# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

[bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""

import os
from typing import Dict, List, Tuple

import datasets

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig

_LOCAL = False

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

# TODO: create a module level variable with your dataset name (should match script name)
#  E.g. Hallmarks of Cancer: [dataset_name] --> hallmarks_of_cancer
_DATASETNAME = "[dataset_name]"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset is designed for XXX NLP task.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = ""

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: "url or list of urls or ... ",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = (
    []
)  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
#  Append "Dataset" to the class name: BioASQ --> BioasqDataset
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    # TODO: For each dataset, implement Config for Source and BigBio;
    #  If dataset contains more than one subset (see examples/bioasq.py) implement for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. bioasq10b_(source|bigbio)_[bigbio_schema_name]
    #   - version: option = (SOURCE_VERSION|BIGBIO_VERSION)
    #   - description: one line description for the dataset
    #   - schema: options = (source|bigbio_[bigbio_schema_name])
    #   - subset_id: subset id is the canonical name for the dataset (eg. bioasq10b)
    #  where [bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="[dataset_name]_source",
            version=SOURCE_VERSION,
            description="[dataset_name] source schema",
            schema="source",
            subset_id="[dataset_name]",
        ),
        ThoughtSourceConfig(
            name="[dataset_name]_thoughtsource",
            version=BIGBIO_VERSION,
            description="[dataset_name] thoughtsource schema",
            schema="thoughtsource",
            subset_id="[dataset_name]",
        ),
    ]

    DEFAULT_CONFIG_NAME = "[dataset_name]_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            # TODO: Create your source schema here
            raise NotImplementedError()

            # EX: Arbitrary NER type dataset
            # features = datasets.Features(
            #    {
            #        "doc_id": datasets.Value("string"),
            #        "text": datasets.Value("string"),
            #        "entities": [
            #            {
            #                "offsets": [datasets.Value("int64")],
            #                "text": datasets.Value("string"),
            #                "type": datasets.Value("string"),
            #                "entity_id": datasets.Value("string"),
            #            }
            #        ],
            #    }
            # )

        # Choose the appropriate bigbio schema for your task and copy it here.
        # You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas.
        # In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "thoughtsource":
            features = schemas.cot_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "bigbio" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument.
        # Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs;
        # many examples use the download_and_extract method; see the DownloadManager docs here:
        # https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and
        # will give back the same structure with the url replaced with the path to local files.

        # TODO: KEEP if your dataset is PUBLIC; remove if not
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        # TODO: KEEP if your dataset is LOCAL; remove if NOT
        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files

        data = None

        if self.config.schema == "source":
            # TODO: yield (key, example) tuples in the original dataset schema
            for key, example in data:
                yield key, example

        elif self.config.schema == "thoughtsource":
            # TODO: yield (key, example) tuples in the bigbio schema
            for key, example in data:
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
