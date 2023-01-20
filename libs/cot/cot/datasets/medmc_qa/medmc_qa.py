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
import json
from tqdm import tqdm
import glob

from cot.utils import schemas, map_example_to_lievin_cot
from cot.utils.configs import ThoughtSourceConfig
from cot.utils.constants import Licenses
from collections import defaultdict


_LOCAL = False

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{pmlr-v174-pal22a,
  title = 	 {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author =       {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = 	 {Proceedings of the Conference on Health, Inference, and Learning},
  pages = 	 {248--260},
  year = 	 {2022},
  editor = 	 {Flores, Gerardo and Chen, George H and Pollard, Tom and Ho, Joyce C and Naumann, Tristan},
  volume = 	 {174},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {07--08 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v174/pal22a/pal22a.pdf},
  url = 	 {https://proceedings.mlr.press/v174/pal22a.html},
}
"""

_DATASETNAME = "medmc_qa"

_DESCRIPTION = """\
A large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address realworld medical entrance exam questions.
The MedMCQA task can be formulated as X = {Q, O} where Q represents the questions in the text, O represents the candidate options, 
multiple candidate answers are given for each question O = {O1, O2, ..., On}. The goal is to select the single or multiple answers 
from the option set.
"""

_HOMEPAGE = "https://medmcqa.github.io/"

_LICENSE = Licenses.MIT

_URLS = {
    "medmcqa": "https://samwald.info/res/thoughtsource/data/medmc_qa.zip",
    "cots": "https://samwald.info/res/thoughtsource/data/lievin-cots.zip"
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class MedMCQADataset(datasets.GeneratorBasedBuilder):
    """A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="source",
            version=SOURCE_VERSION,
            description="MedMCQA source schema",
            schema="source",
            subset_id="medmc_qa",
        ),
        ThoughtSourceConfig(
            name="thoughtsource",
            version=BIGBIO_VERSION,
            description="MedMCQA thoughtsource schema",
            schema="thoughtsource",
            subset_id="medmc_qa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "exp": datasets.Value("string"),
                    "cop": datasets.Value("int32"),
                    "opa": datasets.Value("string"),
                    "opb": datasets.Value("string"),
                    "opc": datasets.Value("string"),
                    "opd": datasets.Value("string"),
                    "subject_name": datasets.Value("string"),
                    "topic_name": datasets.Value("string"),
                    "choice_type": datasets.Value("string"),
                }
            )
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

        data_dir = dl_manager.download_and_extract(_URLS)
        cotspath = os.path.join(data_dir["cots"], "thought-source-med")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir["medmcqa"], "train.json"),
                    "cotspath": None
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["medmcqa"], "test.json"),
                    "cotspath": None
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["medmcqa"], "dev.json"),
                    "cotspath": cotspath
                },
            ),
        ]

    def _generate_examples(self, filepath, cotspath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        data = []
        with open(filepath, "r") as infile:
            for line in infile.readlines():
                data.append(json.loads(line))

        if self.config.schema == "source":
            for example in data:
                example.setdefault("exp")
                example.setdefault("cop")
                yield example["id"], example

        elif self.config.schema == "thoughtsource":

            cots = defaultdict(list)
            if cotspath is not None:
                for file in tqdm(glob.glob(os.path.join(cotspath, "[0-4]-medmcqa*", "*.json")), desc="Preparing Lievin CoTs"):
                    filename = os.path.basename(file)[:-len(".json")]
                    id = filename.split("_")[1].split("-")[1]
                    with open(file, "r") as infile:
                        example = json.load(infile)
                    cots[id].append(example)

            for example in data:
                example.setdefault("exp")
                example.setdefault("cop")
                key = example["id"]

                generated_cots = []
                for item_idx, item in enumerate(cots[key]):
                    assert (example["question"] == item["question"]), f"Question mismatch {example['question']} {item['question']}"
                    cot_item = map_example_to_lievin_cot(f"{key}_{item_idx}", item, "medmc_qa")
                    generated_cots.append(cot_item)

                choices = [example["opa"], example["opb"], example["opc"], example["opd"]]
                answer = choices[example["cop"]-1] if example["cop"] is not None else ""
                example_ = {
                    "id": key,
                    "ref_id": "",
                    "question": example["question"],
                    "type": "multiplechoice",
                    "choices": choices,
                    "context": "",
                    "cot": [example["exp"]] if example["exp"] is not None else "",
                    "answer": [answer],
                    "feedback": [],
                    "generated_cot": generated_cots,
                }

                yield key, example_


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    a = datasets.load_dataset(__file__)
    from pprint import pprint
    pprint(a["validation"][0])
