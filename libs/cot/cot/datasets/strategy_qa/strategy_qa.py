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

import copy
import json
import os
from typing import Dict, List, Tuple

import datasets

from cot.utils import (map_example_to_kojima_cot, map_example_to_wei_cot,
                       parse_kojima_log, parse_wei_log, schemas)
from cot.utils.configs import ThoughtSourceConfig
from cot.utils.constants import Licenses

_LOCAL = False

_CITATION = """\
@article{geva2021strategyqa,
  title = {{Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies}},
  author = {Geva, Mor and Khashabi, Daniel and Segal, Elad and Khot, Tushar and Roth, Dan and Berant, Jonathan},
  journal = {Transactions of the Association for Computational Linguistics (TACL)},
  year = {2021},
}
"""

_DATASETNAME = "strategy_qa"

_DESCRIPTION = """\
StrategyQA is a question-answering benchmark focusing on open-domain questions where the required reasoning steps are implicit in
the question and should be inferred using a strategy. StrategyQA includes 2,780 examples, each consisting of a strategy question,
its decomposition, and evidence paragraphs.
"""

_HOMEPAGE = "https://allenai.org/data/strategyqa"

_LICENSE = Licenses.MIT

_URLS = {
    _DATASETNAME: "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip",
    "kojimalogs": "https://github.com/kojima-takeshi188/zero_shot_cot/raw/main/log/strategyqa_zero_shot_cot.log",
    "weilogs": "https://github.com/jasonwei20/chain-of-thought-prompting/raw/main/chain-of-thought-zip.zip",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class StrategyQADataset(datasets.GeneratorBasedBuilder):
    """StrategyQA is a question-answering benchmark focusing on open-domain questions."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="strategy_qa_source",
            version=SOURCE_VERSION,
            description="StrategyQA source schema",
            schema="source",
            subset_id="strategy_qa",
        ),
        ThoughtSourceConfig(
            name="strategy_qa_thoughtsource",
            version=BIGBIO_VERSION,
            description="StrategyQA thoughtsource schema",
            schema="thoughtsource",
            subset_id="strategy_qa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "strategy_qa_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "qid": datasets.Value("string"),
                    "term": datasets.Value("string"),
                    "description": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("bool"),
                    "facts": datasets.Sequence(datasets.Value("string")),
                    "decomposition": datasets.Sequence(datasets.Value("string")),
                    "evidence": datasets.Sequence(
                        datasets.Sequence(
                            {
                                "paragraphs": [
                                    {
                                        "title": datasets.Value("string"),
                                        "section": datasets.Value("string"),
                                        "headers": [datasets.Value("string")],
                                        "para_index": datasets.Value("int64"),
                                        "content": datasets.Value("string"),
                                    }
                                ],
                                "no_evidence": datasets.Value("string"),
                                "operation": datasets.Value("string"),
                            }
                        )
                    ),
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

        with open(
            os.path.join(data_dir[_DATASETNAME], "strategyqa_train_paragraphs.json"),
            "r",
            encoding="utf8",
        ) as jsonfile:
            paragraphs = json.load(jsonfile)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir[_DATASETNAME], "strategyqa_train.json"),
                    "paragraphs": paragraphs,
                    "split": "train",
                    "kojimalogs": data_dir["kojimalogs"],
                    "weilogs": data_dir["weilogs"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir[_DATASETNAME], "strategyqa_test.json"),
                    "paragraphs": paragraphs,
                    "split": "test",
                    "kojimalogs": None,
                    "weilogs": None,
                },
            ),
        ]

    def _generate_examples(self, filepath, paragraphs, split, kojimalogs, weilogs) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r", encoding="utf8") as jsonfile:
            data = json.load(jsonfile)

        if split == "train":
            if self.config.schema == "source":
                for key, example in enumerate(data):
                    example_ = copy.deepcopy(example)
                    example_["evidence"] = []
                    for annotation in example["evidence"]:
                        decomp_ = []
                        for decomp in annotation:
                            for element in decomp:
                                foo = {
                                    "no_evidence": False,
                                    "operation": False,
                                    "paragraphs": [],
                                }
                                if element == "no_evidence":
                                    foo["no_evidence"] = True
                                elif element == "operation":
                                    foo["operation"] = True
                                elif type(element) == list:
                                    for el in element:
                                        foo["paragraphs"].append(paragraphs[el])
                            decomp_.append(foo)
                        example_["evidence"].append(decomp_)
                    yield key, example_

            elif self.config.schema == "thoughtsource":

                kojima_cots = []
                if kojimalogs is not None:
                    kojima_cots = parse_kojima_log(kojimalogs, "strategyqa")
                wei_cots = []
                if weilogs is not None:
                    wei_cots = parse_wei_log(
                        os.path.join(weilogs, "chain-of-thought-zip", "gpt-3-text-davinci-002", "strategyqa_stream"), "strategyqa"
                    )

                kojima_cot_mapped = 0
                wei_cot_mapped = 0

                for key, example in enumerate(data):

                    generated_cot = []
                    kojima_cot = map_example_to_kojima_cot(example["question"], kojima_cots)
                    if kojima_cot is not None:
                        generated_cot.append(kojima_cot)
                        kojima_cot_mapped += 1
                    wei_cot = map_example_to_wei_cot(example["question"], wei_cots)
                    if wei_cot is not None:
                        generated_cot.append(wei_cot)
                        wei_cot_mapped += 1

                    example_ = {
                        "id": example["qid"],
                        "ref_id": "",
                        "question": example["question"],
                        "type": "bool",
                        "choices": [],
                        "context": "",
                        "cot": example["facts"],
                        "answer": [example["answer"]],
                        "feedback": None,
                        "generated_cot": generated_cot,
                    }
                    yield key, example_

                print(f"{kojima_cot_mapped} kojima cots mapped.")
                print(f"{wei_cot_mapped} wei cots mapped.")

        elif split == "test":
            if self.config.schema == "source":
                for key, example in enumerate(data):
                    example_ = {
                        "qid": example["qid"],
                        "term": None,
                        "description": None,
                        "question": example["question"],
                        "answer": None,
                        "facts": None,
                        "decomposition": None,
                        "evidence": None,
                    }
                    yield key, example_

            elif self.config.schema == "thoughtsource":
                for key, example in enumerate(data):
                    example_ = {
                        "id": example["qid"],
                        "ref_id": "",
                        "question": example["question"],
                        "type": "bool",
                        "choices": [],
                        "context": "",
                        "cot": [],
                        "answer": "",
                        "feedback": None,
                        "generated_cot": [],
                    }
                    yield key, example_


if __name__ == "__main__":
    datasets.load_dataset(__file__)
