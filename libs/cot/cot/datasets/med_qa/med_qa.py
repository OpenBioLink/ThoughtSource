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

import os
from typing import Dict, List, Tuple

import datasets
import glob
import json
from collections import defaultdict
from cot.utils import (schemas, map_example_to_lievin_cot)
from cot.utils.configs import ThoughtSourceConfig
from cot.utils.constants import Licenses
from tqdm import tqdm

import pandas as pd

_LOCAL = False

# TODO: Add BibTeX citation
_CITATION = """\
@article{jin2021disease,
  title={What disease does this patient have? a large-scale open domain question answering dataset from medical exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
"""

_DATASETNAME = "med_qa"

_DESCRIPTION = """\
In this work, we present the first free-form multiple-choice OpenQA dataset for solving medical problems, MedQA,
collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and
traditional Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively. Together
with the question data, we also collect and release a large-scale corpus from medical textbooks from which the reading
comprehension models can obtain necessary knowledge for answering the questions.
"""

_HOMEPAGE = "https://github.com/jind11/MedQA"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://drive.google.com/u/0/uc?export=download&confirm=t&id=1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw",
    "lievin_cot": "https://drive.google.com/u/0/uc?export=download&confirm=t&id=1l0y35SO0mRhc81Asrvo5WU7Mhi1mm2a0"
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class MedQADataset(datasets.GeneratorBasedBuilder):
    """Free-form multiple-choice OpenQA dataset covering three languages."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="med_qa_source",
            version=SOURCE_VERSION,
            description="MedQA source schema",
            schema="source",
            subset_id="med_qa",
        ),
        ThoughtSourceConfig(
            name="med_qa_thoughtsource",
            version=BIGBIO_VERSION,
            description="MedQA thoughtsource schema",
            schema="thoughtsource",
            subset_id="med_qa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "med_qa_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "meta_info": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer_idx": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "options": [
                        {
                            "key": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
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
        cots_path = os.path.join(data_dir["lievin_cot"], "thought-source-med")

        base_dir = os.path.join(data_dir[_DATASETNAME], "data_clean", "questions", "US")
        paths = {
            "train": os.path.join(base_dir, "train.jsonl"),
            "test": os.path.join(base_dir, "test.jsonl"),
            "valid": os.path.join(base_dir, "dev.jsonl"),
        }
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": paths["train"],
                    "cotspath": None
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": paths["test"],
                    "cotspath": cots_path
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": paths["valid"],
                    "cotspath": None
                },
            ),
        ]

    def _generate_examples(self, filepath, cotspath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = pd.read_json(filepath, lines=True)

        if self.config.schema == "source":
            for key, example in data.iterrows():
                example = example.to_dict()
                example["options"] = [
                    {"key": key, "value": value}
                    for key, value in example["options"].items()
                ]
                yield key, example

        elif self.config.schema == "thoughtsource":

            cots = defaultdict(list)
            if cotspath is not None:
                for file in tqdm(glob.glob(cotspath + r"\[0-4]-medqa*\*.json"), desc="Preparing Lievin CoTs"):
                    filename = os.path.basename(file)[:-len(".json")]
                    id = int(filename.split("_")[2].split("-")[1])
                    assert (0 <= id < 1273), f"Oh no {id}"
                    with open(file, "r") as infile:
                        example = json.load(infile)
                    cots[id].append(example)

            for key, example in data.iterrows():

                generated_cots = []
                for item in cots[key]:
                    assert (example["question"] == item["question"]), f"Question mismatch {example['question']} {item['question']}"
                    cot_item = map_example_to_lievin_cot(item, "med_qa")
                    generated_cots.append(cot_item)


                example_ = {
                    "id": key,
                    "question_id": key,
                    "document_id": key,
                    "question": example["question"],
                    "type": "multiplechoice",
                    "cot_type": "list",
                    "choices": example["options"].values(),
                    "context": "",
                    "cot": "",
                    "answer": [example["answer"]],
                    "feedback": [],
                    "generated_cot": generated_cots,
                }

                yield key, example_

if __name__ == "__main__":
    a = datasets.load_dataset(__file__)
    from pprint import pprint
    pprint(a["test"][0])
