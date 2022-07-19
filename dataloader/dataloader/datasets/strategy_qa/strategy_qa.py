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
import os
import json
from typing import List, Tuple, Dict

import datasets
from dataloader.utils import schemas
from dataloader.utils.configs import ThoughtSourceConfig

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

_LICENSE = "MIT"

_URLS = {
    _DATASETNAME: "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip",
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
                                "paragraphs": [{
                                    "title": datasets.Value("string"),
                                    "section": datasets.Value("string"),
                                    "headers": [
                                        datasets.Value("string")
                                    ],
                                    "para_index": datasets.Value("int64"),
                                    "content": datasets.Value("string"),
                                }],
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
        
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        with open(os.path.join(data_dir, "strategyqa_train_paragraphs.json"), "r", encoding="utf8") as jsonfile:
            paragraphs = json.load(jsonfile)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "strategyqa_train.json"),
                    "paragraphs": paragraphs,
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "strategyqa_test.json"),
                    "paragraphs": paragraphs,
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, filepath, paragraphs, split) -> Tuple[int, Dict]:
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
                                    "paragraphs": []
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
                for key, example in enumerate(data):
                    example_ = {
                        "id": example["qid"],
                        "question_id": example["qid"],
                        "document_id": example["qid"],
                        "question": example["question"],
                        "type": "bool",
                        "cot_type": "set",
                        "choices": None,
                        "context": None,
                        "cot": example["facts"],
                        "answer": [example["answer"]],
                        "feedback": None,
                        "cot_after_feedback": None,
                        "answer_after_feedback": None,
                    }
                    yield key, example_

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
                        "question_id": example["qid"],
                        "document_id": example["qid"],
                        "question": example["question"],
                        "type": "bool",
                        "cot_type": None,
                        "choices": None,
                        "context": None,
                        "cot": None,
                        "answer": None,
                        "feedback": None,
                        "cot_after_feedback": None,
                        "answer_after_feedback": None,
                    }
                    yield key, example_

if __name__ == "__main__":
    datasets.load_dataset(__file__)
