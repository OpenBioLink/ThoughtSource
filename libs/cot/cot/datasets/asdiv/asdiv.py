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

import re
import xml.etree.ElementTree as et
from typing import Dict, List, Tuple

import datasets

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig

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
    "corpus": "https://github.com/chaochun/nlu-asdiv-dataset/raw/master/dataset/ASDiv.xml",
    "fold0": "https://github.com/chaochun/nlu-asdiv-dataset/raw/master/dataset/nfolds/asdiv-a/fold0.txt",
    "fold1": "https://github.com/chaochun/nlu-asdiv-dataset/raw/master/dataset/nfolds/asdiv-a/fold1.txt",
    "fold2": "https://github.com/chaochun/nlu-asdiv-dataset/raw/master/dataset/nfolds/asdiv-a/fold2.txt",
    "fold3": "https://github.com/chaochun/nlu-asdiv-dataset/raw/master/dataset/nfolds/asdiv-a/fold3.txt",
    "fold4": "https://github.com/chaochun/nlu-asdiv-dataset/raw/master/dataset/nfolds/asdiv-a/fold4.txt",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = (
    []
)  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

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
                    "ID": datasets.Value("string"),
                    "Body": datasets.Value("string"),
                    "Question": datasets.Value("string"),
                    "Solution-Type": datasets.Value("string"),
                    "Answer": datasets.Value("string"),
                    "Formula": datasets.Value("string"),
                    "Grade": datasets.Value("int32"),
                    "Class": datasets.Value("string"),
                    "Source": datasets.Value("string"),
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
                    "corpuspath": files["corpus"],
                    "folds": [files[f"fold{x}"] for x in [0, 1, 2, 3, 4]],
                },
            ),
        ]

    def _generate_examples(self, corpuspath, folds) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        ids = set()
        for fold in folds:
            with open(fold) as infile:
                content = infile.readlines()
                for id in content:
                    ids.add(id.strip())

        tree = et.parse(corpuspath)
        problems = tree.findall("ProblemSet/Problem")
        problems = [x for x in problems if x.attrib["ID"] in ids]

        if self.config.schema == "source":

            for key, example in enumerate(problems):
                example_ = example.attrib
                if "Class" not in example_:
                    example_["Class"] = None
                for child in example:
                    example_[child.tag] = child.text
                yield key, example_

        elif self.config.schema == "thoughtsource":

            operator_to_verb = {
                "+": "add",
                "-": "subtract",
                "*": "multiply",
                "/": "divide",
            }

            for key, example in enumerate(problems):

                formula = example.find("Formula").text.replace(" ", "")
                equation, ans = formula.split("=")
                assert "r" in ans or float(ans), f"Answer is not number {ans}"

                steps = self._decompose_equation(equation)

                int_ = {}
                chain_of_thought = []
                for idx, (num1, operator, num2) in enumerate(steps):
                    num1 = str(int_[num1]) if str(num1).startswith("int") else str(num1)
                    num2 = str(int_[num2]) if str(num2).startswith("int") else str(num2)
                    int_[f"int{idx}"] = eval(num1 + operator + num2)

                    if idx == 0 and len(steps) > 1:
                        "First we"
                    elif idx > 0 and len(steps) > 1:
                        cot = "Then we"
                    else:
                        cot = "We"
                    cot += f" {operator_to_verb[operator]} "

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
                    "id": example.attrib["ID"],
                    "question_id": example.attrib["ID"],
                    "document_id": example.attrib["ID"],
                    "question": " ".join(
                        [example.find("Body").text, example.find("Question").text]
                    ),
                    "type": "number",
                    "cot_type": "list",
                    "choices": [],
                    "context": "",
                    "cot": chain_of_thought,
                    "answer": [example.find("Answer").text],
                    "feedback": [],
                    "generated_cot": [],
                }
                yield key, example_

    def _decompose_equation(self, equation, idx=0):

        equation = equation.replace(" ", "")

        # special case equation single number no operator
        if equation.replace(".", "", 1).isdigit():
            return []

        if equation == f"int{idx-1}":
            return []
        else:
            pattern = (
                r"\((int[0-9]|[0-9]+(\.[0-9]+)?)([+\-*/])(int[0-9]|[0-9]+(\.[0-9]+)?)\)"
            )
            result = re.search(pattern, equation)
            if not result:
                pattern = (
                    r"(int[0-9]|[0-9]+(\.[0-9]+)?)([+\-*/])(int[0-9]|[0-9]+(\.[0-9]+)?)"
                )
                result = re.search(pattern, equation)
            assert result, equation
            equation = (
                equation[: result.span()[0]]
                + "int"
                + str(idx)
                + equation[result.span()[1] :]
            )
            return [
                [result.group(1), result.group(3), result.group(4)]
            ] + self._decompose_equation(equation, idx + 1)


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
