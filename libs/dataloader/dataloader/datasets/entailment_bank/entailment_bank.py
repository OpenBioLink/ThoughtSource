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

from dataloader.utils import schemas
from dataloader.utils.configs import ThoughtSourceConfig

_LOCAL = False

_CITATION = """\
@inproceedings{Dalvi2021ExplainingAW,
  title={Explaining Answers with Entailment Trees},
  author={Bhavana Dalvi and Peter Alexander Jansen and Oyvind Tafjord and Zhengnan Xie and Hannah Smith and Leighanna
  Pipatanangkura and Peter Clark},
  booktitle={EMNLP},
  year={2021}
}
"""

_DATASETNAME = "entailment_bank"

_DESCRIPTION = """\
Our goal, in the context of open-domain textual question-answering (QA), is to explain answers by not just listing supporting
textual evidence (“rationales”), but also showing how such evidence leads to the answer in a systematic way. If this could be
done, new opportunities for understanding and debugging the system’s reasoning would become possible. Our approach is to generate
explanations in the form of entailment trees, namely a tree of entailment steps from facts that are known, through intermediate
conclusions, to the final answer. To train a model with this skill, we created EntailmentBank, the first dataset to contain
multistep entailment trees. At each node in the tree (typically) two or more facts compose together to produce a new conclusion.
Given a hypothesis (question + answer), we define three increasingly difficult explanation tasks: generate a valid entailment
tree given (a) all relevant sentences (the leaves of the gold entailment tree) (b) all relevant and some irrelevant sentences (c)
a corpus. We provide baseline results for this task, and analyze the problems involved.
"""

_HOMEPAGE = "https://allenai.org/data/entailmentbank"

_LICENSE = "CC BY 4.0"

_URLS = {
    _DATASETNAME: "https://drive.google.com/uc?export=download&id=1kVr-YsUVFisceiIklvpWEe0kHNSIFtNh",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = (
    []
)  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class EntailmentBankDataset(datasets.GeneratorBasedBuilder):
    """2k multi-step entailment trees, explaining the answers to ARC science questions"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="entailment_bank_source",
            version=SOURCE_VERSION,
            description="EntailmentBank source schema",
            schema="source",
            subset_id="entailment_bank",
        ),
        ThoughtSourceConfig(
            name="entailment_bank_thoughtsource",
            version=BIGBIO_VERSION,
            description="EntailmentBank thoughtsource schema",
            schema="thoughtsource",
            subset_id="entailment_bank",
        ),
    ]

    DEFAULT_CONFIG_NAME = "entailment_bank_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "hypothesis": datasets.Value("string"),
                    "proof": datasets.Value("string"),
                    "full_text_proof": datasets.Value("string"),
                    "depth_of_proof": datasets.Value("int32"),
                    "length_of_proof": datasets.Value("int32"),
                    "meta": {
                        "question_text": datasets.Value("string"),
                        "answer_text": datasets.Value("string"),
                        "hypothesis_id": datasets.Value("string"),
                        "triples": [
                            {
                                "sent_id": datasets.Value("string"),
                                "value": datasets.Value("string"),
                            }
                        ],
                        "distractors": [
                            datasets.Value("string"),
                        ],
                        "distractors_relevance": [
                            datasets.Value("float32"),
                        ],
                        "intermediate_conclusions": [
                            {
                                "int_id": datasets.Value("string"),
                                "value": datasets.Value("string"),
                            }
                        ],
                        "core_concepts": [
                            datasets.Value("string"),
                        ],
                        "step_proof": datasets.Value("string"),
                        "lisp_proof": datasets.Value("string"),
                        "polish_proof": datasets.Value("string"),
                        "worldtree_provenance": [
                            {
                                "sent_id": datasets.Value("string"),
                                "uuid": datasets.Value("string"),
                                "original_text": datasets.Value("string"),
                            },
                        ],
                        "add_list": [
                            {
                                "sid": datasets.Value("string"),
                                "fact": datasets.Value("string"),
                            }
                        ],
                        "delete_list": [
                            {
                                "uuid": datasets.Value("string"),
                                "fact": datasets.Value("string"),
                            },
                        ],
                    },
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
        dataset_path = os.path.join(
            data_dir, "entailment_trees_emnlp2021_data_v3", "dataset", "task_1"
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(dataset_path, "train.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(dataset_path, "test.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(dataset_path, "dev.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as json_file:
            data = [json.loads(line) for line in json_file]

        if self.config.schema == "source":
            for key, example in enumerate(data):
                example["meta"]["triples"] = [
                    {"sent_id": key, "value": value}
                    for key, value in example["meta"]["triples"].items()
                ]
                example["meta"]["intermediate_conclusions"] = [
                    {"int_id": key, "value": value}
                    for key, value in example["meta"][
                        "intermediate_conclusions"
                    ].items()
                ]
                example["meta"]["worldtree_provenance"] = [
                    {"sent_id": key, **value}
                    for key, value in example["meta"]["worldtree_provenance"].items()
                ]
                yield key, example

        elif self.config.schema == "thoughtsource":

            for key, example in enumerate(data):

                cot = example["proof"]

                pattern = r"int[0-9]: "
                cot = re.sub(pattern, "", cot)

                assert cot[-2:] == "; ", cot

                cot_ = []
                cot = cot.split(";")[:-1]
                for inferral in cot:
                    inferral = inferral.replace("hypothesis", example["hypothesis"])

                    for sent_id, value in example["meta"]["triples"].items():
                        inferral = inferral.replace(sent_id, value)

                    for int_id, value in example["meta"][
                        "intermediate_conclusions"
                    ].items():
                        inferral = inferral.replace(int_id, value)

                    for stmt in inferral.split("&"):
                        stmt = self._untokenize(stmt)

                        if "->" in stmt:
                            stmt, therefore = stmt.split("->")
                            cot_.append(self._untokenize(stmt).capitalize() + ".")
                            cot_.append(
                                "Therefore, " + self._untokenize(therefore) + "."
                            )
                        else:
                            cot_.append(self._untokenize(stmt).capitalize() + ".")

                example_ = {
                    "id": example["id"],
                    "question_id": example["id"],
                    "document_id": example["id"],
                    "question": example["question"],
                    "type": "text",
                    "cot_type": "list",
                    "choices": [],
                    "context": "",
                    "cot": cot_,
                    "answer": [example["answer"]],
                    "feedback": [],
                    "generated_cot": [],
                }

                yield key, example_

    def _untokenize(self, text):
        """
        Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.
        """
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
        step5 = (
            step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
        )
        step6 = step5.replace(" ` ", " '")
        return step6.strip()


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
