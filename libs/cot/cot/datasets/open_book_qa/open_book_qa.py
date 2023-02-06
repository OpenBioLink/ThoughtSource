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
from typing import Dict, List, Tuple

import datasets

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig
from cot.utils.constants import Licenses

_LOCAL = False

_CITATION = """\
@inproceedings{OpenBookQA2018,
 title={Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering},
 author={Todor Mihaylov and Peter Clark and Tushar Khot and Ashish Sabharwal},
 booktitle={EMNLP},
 year={2018}
}
"""

_DATASETNAME = "open_book_qa"

_DESCRIPTION = """\
OpenBookQA aims to promote research in advanced question-answering, probing a deeper understanding of both the topic (with
salient facts summarized as an open book, also provided with the dataset) and the language it is expressed in. In particular, it
contains questions that require multi-step reasoning, use of additional common and commonsense knowledge, and rich text
comprehension.

OpenBookQA is a new kind of question-answering dataset modeled after open book exams for assessing human understanding of a
subject. It consists of 5,957 multiple-choice elementary-level science questions (4,957 train, 500 dev, 500 test), which probe
the understanding of a small “book” of 1,326 core science facts and the application of these facts to novel situations. For
training, the dataset includes a mapping from each question to the core science fact it was designed to probe. Answering
OpenBookQA questions requires additional broad common knowledge, not contained in the book. The questions, by design, are
answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. Strong neural baselines achieve
around 50% on OpenBookQA, leaving a large gap to the 92% accuracy of crowd-workers.

Additionally, we provide 5,167 crowd-sourced common knowledge facts, and an expanded version of the train/dev/test questions
where each question is associated with its originating core fact, a human accuracy score, a clarity score, and an anonymized
crowd-worker ID (in the “Additional” folder).
"""

_HOMEPAGE = "https://allenai.org/data/open-book-qa"

_LICENSE = Licenses.APACHE_2p0

_URLS = {
    _DATASETNAME: "https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class OpenBookQADataset(datasets.GeneratorBasedBuilder):
    """Question-answering dataset modeled after open book exams for assessing human understanding of a subject."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="source",
            version=SOURCE_VERSION,
            description="OpenBookQA source schema",
            schema="source",
            subset_id="open_book_qa",
        ),
        ThoughtSourceConfig(
            name="thoughtsource",
            version=BIGBIO_VERSION,
            description="OpenBookQA BigBio schema",
            schema="thoughtsource",
            subset_id="open_book_qa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thoughtsource"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": {
                        "stem": datasets.Value("string"),
                        "choices": [
                            {
                                "text": datasets.Value("string"),
                                "label": datasets.Value("string"),
                            }
                        ],
                    },
                    "fact1": datasets.Value("string"),
                    "humanScore": datasets.Value("float"),
                    "clarity": datasets.Value("float"),
                    "turkIdAnonymized": datasets.Value("string"),
                    "answerKey": datasets.Value("string"),
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

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "OpenBookQA-V1-Sep2018",
                        "Data",
                        "Additional",
                        "train_complete.jsonl",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "OpenBookQA-V1-Sep2018",
                        "Data",
                        "Additional",
                        "test_complete.jsonl",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "OpenBookQA-V1-Sep2018",
                        "Data",
                        "Additional",
                        "dev_complete.jsonl",
                    ),
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as json_file:
            data = [json.loads(line) for line in json_file]

        if self.config.schema == "source":
            for key, example in enumerate(data):
                yield key, example

        elif self.config.schema == "thoughtsource":
            for key, example in enumerate(data):
                choices_ = {x["label"]: x["text"] for x in example["question"]["choices"]}

                answer = choices_[example["answerKey"]]
                choices = choices_.values()

                example_ = {
                    "id": key,
                    "ref_id": "",
                    "question": example["question"]["stem"],
                    "type": "multiplechoice",
                    "choices": choices,
                    "context": "",
                    "cot": [example["fact1"]],
                    "answer": [answer],
                    "feedback": [],
                    "generated_cot": [],
                }

                yield key, example_


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
