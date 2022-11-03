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
import re
from typing import Dict, List, Tuple

import datasets

from dataloader.utils import schemas
from dataloader.utils.configs import ThoughtSourceConfig

_LOCAL = False

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{patel-etal-2021-nlp,
    title = "Are {NLP} Models really able to Solve Simple Math Word Problems?",
    author = "Patel, Arkil  and
        Bhattamishra, Satwik  and
        Goyal, Navin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics:
    Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.168",
    doi = "10.18653/v1/2021.naacl-main.168",
    pages = "2080--2094",
    abstract = "The problem of designing NLP solvers for math word problems (MWP) has seen sustained research activity and steady
    gains in the test accuracy. Since existing solvers achieve high performance on the benchmark datasets for elementary level MWPs
    containing one-unknown arithmetic word problems, such problems are often considered {``}solved{''} with the bulk of research
    attention moving to more complex MWPs. In this paper, we restrict our attention to English MWPs taught in grades four and lower.
    We provide strong evidence that the existing MWP solvers rely on shallow heuristics to achieve high performance on the benchmark
    datasets. To this end, we show that MWP solvers that do not have access to the question asked in the MWP can still solve a large
    fraction of MWPs. Similarly, models that treat MWPs as bag-of-words can also achieve surprisingly high accuracy. Further, we
    introduce a challenge dataset, SVAMP, created by applying carefully chosen variations over examples sampled from existing
    datasets. The best accuracy achieved by state-of-the-art models is substantially lower on SVAMP, thus showing that much remains
    to be done even for the simplest of the MWPs.",
}
"""

_DATASETNAME = "svamp"

_DESCRIPTION = """\
The task of solving Math Word Problems (MWPs) has received significant research attention in the past years. An MWP consists of a
short Natural Language narrative that describes a state of the world and poses a question about some unknown quantities (see
Table 1 for examples). In this work, we show deficiencies in two benchmark datasets - ASDiv-A and MAWPS. We first show that
existing models achieve reasonably high accuracies on these datasets even after removing the "question" part of the MWP at test
time. We further show that a simple model without any word-order information can also solve a majority of MWPs in these datasets.
Our experiments indicate that existing models rely on shallow heuristics in benchmark MWP datasets for achieving high
performance. Our experiments render the benchmark datasets unreliable to measure model performance. To enable more robust
evaluation of automatic MWP solvers, we created a challenge set called "SVAMP". The examples in SVAMP test a model across
different aspects of solving MWPs. Table 1 provides three examples from SVAMP that test whether a model is Question-sensitive,
has robust reasoning ability or is invariant to structural alterations respectively. 
"""

_HOMEPAGE = "https://github.com/arkilpatel/SVAMP"

_LICENSE = "MIT"

_URLS = {
    _DATASETNAME: "https://github.com/arkilpatel/SVAMP/raw/main/SVAMP.json",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = (
    []
)  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class SvampDataset(datasets.GeneratorBasedBuilder):
    """Challenging Math Word Problems (MWPs) dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="svamp_source",
            version=SOURCE_VERSION,
            description="SVAMP source schema",
            schema="source",
            subset_id="svamp",
        ),
        ThoughtSourceConfig(
            name="svamp_thoughtsource",
            version=BIGBIO_VERSION,
            description="SVAMP thoughtsource schema",
            schema="thoughtsource",
            subset_id="svamp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "svamp_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "ID": datasets.Value("string"),
                    "Body": [datasets.Value("string")],
                    "Question": datasets.Value("string"),
                    "Equation": datasets.Value("string"),
                    "Answer": [datasets.Value("float")],
                    "Type": datasets.Value("string"),
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
        filepath = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as jsonfile:
            data = json.load(jsonfile)

        if self.config.schema == "source":
            for key, example in enumerate(data):
                yield key, example

        elif self.config.schema == "thoughtsource":

            operator_to_result = {
                "+": "sum",
                "-": "difference",
                "*": "product",
                "/": "quotient",
            }
            operator_to_nomen = {
                "+": "addition",
                "-": "subtraction",
                "*": "multiplication",
                "/": "division",
            }
            operator_to_verb = {
                "+": "add",
                "-": "subtract",
                "*": "multiply",
                "/": "divide",
            }

            for key, example in enumerate(data):

                steps = self._decompose_equation(example["Equation"])

                int_ = {}
                chain_of_thought = [
                    f"To get to the correct answer we have to perform {example['Type']}."
                ]
                for idx, (num1, operator, num2) in enumerate(steps):
                    num1 = str(int_[num1]) if str(num1).startswith("int") else str(num1)
                    num2 = str(int_[num2]) if str(num2).startswith("int") else str(num2)
                    int_[f"int{idx}"] = eval(num1 + operator + num2)

                    cot = f"{'First we' if (idx == 0 and len(steps) > 1) else 'Then we' if (idx > 0 and len(steps) > 1) else 'We'} {operator_to_verb[operator]} "

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

                question = example["Body"]
                if example["Body"][-1] != ".":
                    question += ","
                    example["Question"] = (
                        example["Question"][0].lower() + example["Question"][1:]
                    )
                question += " " + example["Question"]

                example_ = {
                    "id": key,
                    "question_id": key,
                    "document_id": key,
                    "question": question,
                    "type": "number",
                    "cot_type": "list",
                    "choices": [],
                    "context": "",
                    "cot": chain_of_thought,
                    "answer": [example["Answer"]],
                    "feedback": [],
                    "generated_cot": [],
                }
                yield key, example_

    def _decompose_equation(self, equation, idx=0):
        # special case equation single number no operator
        if equation.replace(".", "", 1).isdigit():
            return []

        pattern = (
            r"\( (int[0-9]|[0-9]+(\.[0-9]+)?) ([+\-*/]) (int[0-9]|[0-9]+(\.[0-9]+)?) \)"
        )
        if equation == f"int{idx-1}":
            return []
        else:
            result = re.search(pattern, equation)
            assert result, equation
            # assert (len(re.findall(pattern, equation)) == 1), equation
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
