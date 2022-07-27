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
import re
import pandas as pd
from typing import List, Tuple, Dict

import datasets
from dataloader.utils import schemas
from dataloader.utils.configs import ThoughtSourceConfig

_LOCAL = False

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{miao-etal-2020-diverse,
  title={A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers},
  author={Miao, Shen-yun and Liang, Chao-Chun and Su, Keh-Yih},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={975--984},
  year={2020}
}
"""

_DATASETNAME = "asdiv"

_DESCRIPTION = """\
This repository provides ASDiv (a new diverse dataset in terms of both language patterns and problem types) for evaluating and
developing MWP Solvers. It contains 2305 english Math Word Problems (MWPs), and is published in this paper "A Diverse Corpus for
Evaluating and Developing English Math Word Problem Solvers".
"""

_HOMEPAGE = "https://github.com/chaochun/nlu-asdiv-dataset"

_LICENSE = "MIT"

_URLS = {
    "train": "https://github.com/arkilpatel/SVAMP/raw/main/data/cv_asdiv-a/fold0/train.csv",
    "dev": "https://github.com/arkilpatel/SVAMP/raw/main/data/cv_asdiv-a/fold0/dev.csv",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class AsdivDataset(datasets.GeneratorBasedBuilder):
    """Dataset containing 2305 english Math Word Problems (MWPs)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="asdiv_source",
            version=SOURCE_VERSION,
            description="ASDiv source schema",
            schema="source",
            subset_id="asdiv",
        ),
        ThoughtSourceConfig(
            name="asdiv_thoughtsource",
            version=BIGBIO_VERSION,
            description="ASDiv thoughtsource schema",
            schema="thoughtsource",
            subset_id="asdiv",
        ),
    ]

    DEFAULT_CONFIG_NAME = "asdiv_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "numbers": [datasets.Value("int32")],
                    "equation": datasets.Value("string"),
                    "answer": datasets.Value("float"),
                    "group_nums": [datasets.Value("int32")],
                    "grade": datasets.Value("int32"),
                    "type": datasets.Value("string"),
                    "body": datasets.Value("string"),
                    "ques": datasets.Value("string"),
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
                gen_kwargs={
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(self, data_dir) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        train = pd.read_csv(data_dir["train"])
        dev = pd.read_csv(data_dir["dev"])
        data = pd.concat([train, dev], ignore_index=True)

        if self.config.schema == "source":
            for key, example in data.iterrows():
                
                all_numbers = {f"number{i}": x for i,x in enumerate([float(x) for x in example["Numbers"].split(" ")])}
                for number_id, number in all_numbers.items():
                    example["Question"] = example["Question"].replace(number_id, str(number))
                example_ = {
                    "question": example["Question"],
                    "numbers": [float(x) for x in example["Numbers"].split(" ")],
                    "equation": example["Equation"],
                    "answer": float(example["Answer"]),
                    "group_nums": [int(x.strip()) for x in example["group_nums"][1:-1].split(",")],
                    "grade": example["Grade"],
                    "type": example["Type"],
                    "body": example["Body"],
                    "ques": example["Ques_Statement"],
                }
                yield key, example_

        elif self.config.schema == "thoughtsource":

            operator_to_result = {
                "+": "sum",
                "-": "difference",
                "*": "product",
                "/": "quotient"
            }
            operator_to_nomen = {
                "+": "addition",
                "-": "subtraction",
                "*": "multiplication",
                "/": "division"
            }
            operator_to_verb = {
                "+": "add",
                "-": "subtract",
                "*": "multiply",
                "/": "divide"
            }

            for key, example in data.iterrows():

                all_numbers = {f"number{i}": x for i,x in enumerate([float(x) for x in example["Numbers"].split(" ")])}
                for number_id, number in all_numbers.items():
                    example["Question"] = example["Question"].replace(number_id, str(number))

                steps = self._decompose_equation(example["Equation"])

                int_ = {}
                chain_of_thought = []
                for idx, (operator, num1, num2) in enumerate(steps):
                    num1 = all_numbers.get(num1, num1)
                    num2 = all_numbers.get(num2, num2)
                    num1 = str(int_[num1]) if str(num1).startswith("int") else str(num1)
                    num2 = str(int_[num2]) if str(num2).startswith("int") else str(num2)
                    int_[f"int{idx}"] = eval(num1 + operator + num2)

                    cot = f"{'First' if idx == 0 else 'Then'} we {operator_to_verb[operator]} "

                    if operator == "+":
                        cot += f"{num1} to {num2} "
                    elif operator == "-":
                        cot += f"{num2} from {num1} "
                    elif operator == "*":
                        cot += f"{num1} by {num2} "
                    elif operator == "/":
                        cot += f"{num1} by {num2} "
                    cot += f"and get {int_[f'int{idx}']}."

                    chain_of_thought.append(cot)
                        
                example_ = {
                    "id": key,
                    "question_id": key,
                    "document_id": key,
                    "question": example["Question"],
                    "type": "number",
                    "cot_type": "list",
                    "choices": [],
                    "context": "",
                    "cot": chain_of_thought,
                    "answer": [example["Answer"]],
                    "feedback": [],
                    "cot_after_feedback": [],
                    "answer_after_feedback": [],
                }
                yield key, example_
    
    def _decompose_equation(self, equation, idx=0):
        # special case equation single number no operator
        if equation == "number0":
            return []

        pattern = "[+\-/*] (number[0-9]|int[0-9]|[0-9]+(\.[0-9]+)?) (number[0-9]|int[0-9]|[0-9]+(\.[0-9]+)?)"
        if equation == f"int{idx-1}":
            return []
        else:
            result = re.search(pattern, equation)
            assert (result), equation
            # assert (len(re.findall(pattern, equation)) == 1), equation
            equation = equation[:result.span()[0]] + "int" + str(idx) + equation[result.span()[1]:]
            return [result.group().split(" ")] + self._decompose_equation(equation, idx+1)


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)