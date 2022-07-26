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

import json
import os
from typing import List, Tuple, Dict

import datasets
from dataloader.utils import schemas
from dataloader.utils.configs import ThoughtSourceConfig

_LOCAL = False

_CITATION = """\
@inproceedings{ling-etal-2017-program,
    title = "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems",
    author = "Ling, Wang  and
        Yogatama, Dani  and
        Dyer, Chris  and
        Blunsom, Phil",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
        Papers)",
    month = jul,
    year = "2017",
    address = "Vancouver, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P17-1015",
    doi = "10.18653/v1/P17-1015",
    pages = "158--167",
    abstract = "Solving algebraic word problems requires executing a series of arithmetic operations{---}a program{---}to
        obtain a final answer. However, since programs can be arbitrarily complicated, inducing them directly from
        question-answer pairs is a formidable challenge. To make this task more feasible, we solve these problems by generating
        answer rationales, sequences of natural language and human-readable mathematical expressions that derive the final answer
        through a series of small steps. Although rationales do not explicitly specify programs, they provide a scaffolding for
        their structure via intermediate milestones. To evaluate our approach, we have created a new 100,000-sample dataset of
        questions, answers and rationales. Experimental results show that indirect supervision of program learning via answer
        rationales is a promising strategy for inducing arithmetic programs.",
    }
"""

_DATASETNAME = "aqua"

_DESCRIPTION = """\
This dataset contains the algebraic word problems with rationales described in our paper:

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. (2017) Program Induction by Rationale Generation: Learning to Solve and
Explain Algebraic Word Problems. In Proc. ACL.

The dataset consists of about 100,000 algebraic word problems with natural language rationales. Each problem is a json object
consisting of four parts:

question - A natural language definition of the problem to solve
options - 5 possible options (A, B, C, D and E), among which one is correct
rationale - A natural language description of the solution to the problem
correct - The correct option
"""

_HOMEPAGE = "https://github.com/deepmind/AQuA"

_LICENSE = "Apache License, Version 2.0"

_URLS = {
    "train": "https://github.com/deepmind/AQuA/raw/master/train.json",
    "test": "https://github.com/deepmind/AQuA/raw/master/test.json",
    "valid": "https://github.com/deepmind/AQuA/raw/master/dev.json",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class AquaDataset(datasets.GeneratorBasedBuilder):
    """AQUA-RAT (Algebra Question Answering with Rationales) Dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="aqua_source",
            version=SOURCE_VERSION,
            description="AQuA source schema",
            schema="source",
            subset_id="aqua",
        ),
        ThoughtSourceConfig(
            name="aqua_thoughtsource",
            version=BIGBIO_VERSION,
            description="AQuA thoughtsource schema",
            schema="thoughtsource",
            subset_id="aqua",
        ),
    ]

    DEFAULT_CONFIG_NAME = "aqua_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "choices": [datasets.Value("string")],
                    "rationale": datasets.Value("string"),
                    "answer": datasets.Value("string"),
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["valid"],
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        with open(filepath, "r", encoding="utf8") as infile:
            data = [json.loads(line) for line in infile]

        if self.config.schema == "source":
            for key, example in enumerate(data):
                choices = [x.split(")") for x in example["options"]]
                choices = {x[0]: x[1] for x in choices}
                example['choices'] = choices.values()
                example.pop('options')
                example['answer'] = choices[example.pop('correct')]

                yield key, example

        elif self.config.schema == "thoughtsource":
            for key, example in enumerate(data):

                choices = [x.split(")") for x in example["options"]]
                choices = {x[0]: x[1] for x in choices}

                example_ = {
                    "id": key,
                    "question_id": key,
                    "document_id": key,
                    "question": example["question"],
                    "type": "multiplechoice",
                    "cot_type": "list",
                    "choices": choices.values(),
                    "context": "",
                    "cot": example["rationale"].split("\n"),
                    "answer": [choices[example["correct"]]],
                    "feedback": [],
                    "cot_after_feedback": [],
                    "answer_after_feedback": [],
                }
                yield key, example_


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
