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
import re
from typing import Dict, List, Tuple

import datasets

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig
from cot.utils.constants import Licenses

_LOCAL = False

_CITATION = """\
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser,
  Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
"""

_DATASETNAME = "gsm8k"

_DESCRIPTION = """\
State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform
multi-step mathematical reasoning. To diagnose the failures of current models and support research, we're releasing GSM8K, a
dataset of 8.5K high quality linguistically diverse grade school math word problems. We find that even the largest transformer
models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution.
"""

_HOMEPAGE = "https://github.com/openai/grade-school-math"

_LICENSE = Licenses.MIT

_URLS = {
    "gsm8k": {
        "train": "https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/train.jsonl",
        "test": "https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl",
    },
    "gsm8k_socratic": {
        "train": "https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/train_socratic.jsonl",
        "test": "https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test_socratic.jsonl",
    },
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class Gsm8kDataset(datasets.GeneratorBasedBuilder):
    """High quality linguistically diverse grade school math word problem."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="source",
            version=SOURCE_VERSION,
            description="GSM8K source schema",
            schema="source",
            subset_id="gsm8k",
        ),
        ThoughtSourceConfig(
            name="thoughtsource",
            version=BIGBIO_VERSION,
            description="GSM8K thoughtsource schema",
            schema="thoughtsource",
            subset_id="gsm8k",
        ),
        ThoughtSourceConfig(
            name="socratic_source",
            version=SOURCE_VERSION,
            description="GSM8K Socratic source schema",
            schema="source",
            subset_id="gsm8k_socratic",
        ),
        ThoughtSourceConfig(
            name="socratic_thoughtsource",
            version=BIGBIO_VERSION,
            description="GSM8K Socratic thoughtsource schema",
            schema="thoughtsource",
            subset_id="gsm8k_socratic",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thoughtsource"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question": datasets.Value("string"),
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

        data_dir = dl_manager.download_and_extract(_URLS[self.config.subset_id])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["train"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir["test"]),
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        with open(filepath, "r") as json_file:
            data = [json.loads(line) for line in json_file]

        if self.config.schema == "source":
            for key, example in enumerate(data):
                yield key, example

        elif self.config.schema == "thoughtsource":
            for key, example in enumerate(data):
                chain = re.sub(r"<<[0-9\.\(\)+\-/*=]+>>", "", example["answer"])
                assert "<<" not in chain, chain
                assert ">>" not in chain, chain
                chain = chain.split("\n")
                chain_of_thought = chain[:-1]
                answer = chain[-1].replace("#### ", "")

                example_ = {
                    "id": key,
                    "ref_id": "",
                    "question": example["question"],
                    "type": "number",
                    "choices": [],
                    "context": "",
                    "cot": chain_of_thought,
                    "answer": [answer],
                    "feedback": [],
                    "generated_cot": [],
                }
                yield key, example_


if __name__ == "__main__":
    datasets.load_dataset(__file__)
