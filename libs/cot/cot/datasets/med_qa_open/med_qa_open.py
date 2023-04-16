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

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import datasets
import pandas as pd
from tqdm import tqdm

from cot.utils import (map_example_to_lievin_cot, map_json_to_lievin_cots_2,
                       schemas)
from cot.utils.configs import ThoughtSourceConfig
from cot.utils.constants import Licenses

_LOCAL = False

_CITATION = """\
@misc{nair2023dera,
      title={DERA: Enhancing Large Language Model Completions with Dialog-Enabled Resolving Agents}, 
      author={Varun Nair and Elliot Schumacher and Geoffrey Tso and Anitha Kannan},
      year={2023},
      eprint={2303.17071},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "med_qa_open"

_DESCRIPTION = """\
The following repository contains the open-ended question-answering version of MedQA. These consists of questions that were rewritten using a GPT-4 prompt, using the approach described in the paper. These were not manually rewritten by human annotators, so there may be some inconsistencies.

Notes

In each file, the open-ended question is included for each question-answer pair in the "question_open" field.
The file format is the same as the original except for that additional field. Also, note that the training file was converted from the 4-option format, while the others are from the 5-option format. This does not matter for open-ended evaluation, but be aware that they have different amounts of unused options.
Please see LICENSE.txt and LICENSE_MEDQA.txt (original MedQA license).
"""

_HOMEPAGE = "https://github.com/curai/curai-research/tree/main/DERA"

_LICENSE = Licenses.MIT

_URLS = {
    "valid": "https://raw.githubusercontent.com/curai/curai-research/main/DERA/medqa_open/dev.jsonl",
    "test": "https://raw.githubusercontent.com/curai/curai-research/main/DERA/medqa_open/test.jsonl",
    "train": "https://raw.githubusercontent.com/curai/curai-research/main/DERA/medqa_open/train.jsonl",
}


_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MedQAOpenDataset(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="source",
            version=SOURCE_VERSION,
            description="MedQAOpen source schema",
            schema="source",
            subset_id="med_qa_open",
        ),
        ThoughtSourceConfig(
            name="thoughtsource",
            version=BIGBIO_VERSION,
            description="MedQAOpen thoughtsource schema",
            schema="thoughtsource",
            subset_id="med_qa_open",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thoughtsource"

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
                    "question_open": datasets.Value("string"),
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

        files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "filepath": files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepath": files["test"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "filepath": files["valid"],
                },
            ),
        ]

    def _generate_examples(self, filepath, split) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data = pd.read_json(filepath, lines=True)

        if self.config.schema == "source":
            for key, example in data.iterrows():
                example = example.to_dict()
                #example["options"] = [{"key": key, "value": value} for key, value in example["options"].items()]
                yield key, example

        elif self.config.schema == "thoughtsource":
            for key, example in data.iterrows():
                example_ = {
                    "id": "med_qa_open_" + split + "_" + str(key),
                    "ref_id": "",
                    "question": example["question_open"],
                    "type": "text",
                    "choices": [],
                    "context": "",
                    "cot": "",
                    "answer": [example["answer"]],
                    "feedback": [],
                    "generated_cot": [],
                }

                yield key, example_


if __name__ == "__main__":
    a = datasets.load_dataset(__file__)
    from pprint import pprint

    pprint(a["test"][4])
