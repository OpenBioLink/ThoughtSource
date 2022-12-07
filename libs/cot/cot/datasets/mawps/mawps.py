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
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig
from cot.utils.constants import Licenses

_LOCAL = False

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{koncel-kedziorski-etal-2016-mawps,
    title = "{MAWPS}: A Math Word Problem Repository",
    author = "Koncel-Kedziorski, Rik  and
      Roy, Subhro  and
      Amini, Aida  and
      Kushman, Nate  and
      Hajishirzi, Hannaneh",
    booktitle = "Proceedings of the 2016 Conference of the North {A}merican Chapter of the Association for Computational
        Linguistics: Human Language Technologies",
    month = jun,
    year = "2016",
    address = "San Diego, California",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N16-1136",
    doi = "10.18653/v1/N16-1136",
    pages = "1152--1157",
}
"""

_DATASETNAME = "mawps"

_DESCRIPTION = """\
Recent work across several AI subdisciplines has focused on automatically solving math word problems. In this paper we introduce
MAWPS, an online repository of Math Word Problems, to provide a unified testbed to evaluate different algorithms. MAWPS allows
for the automatic construction of datasets with particular characteristics, providing tools for tuning the lexical and template
overlap of a dataset as well as for filtering ungrammatical problems from web-sourced corpora. The online nature of this
repository facilitates easy community contribution. At present, we have amassed 3,320 problems, including the full datasets used
in several prominent works.
"""

_HOMEPAGE = "https://github.com/sroy9/mawps"

_LICENSE = Licenses.MIT

_URLS = {
    "train": "https://github.com/arkilpatel/SVAMP/raw/main/data/cv_mawps/fold0/train.csv",
    "dev": "https://github.com/arkilpatel/SVAMP/raw/main/data/cv_mawps/fold0/dev.csv",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MawpsDataset(datasets.GeneratorBasedBuilder):
    """Dataset containing 3,320 english Math Word Problems (MWPs)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="mawps_source",
            version=SOURCE_VERSION,
            description="MAWPS source schema",
            schema="source",
            subset_id="mawps",
        ),
        ThoughtSourceConfig(
            name="mawps_thoughtsource",
            version=BIGBIO_VERSION,
            description="MAWPS thoughtsource schema",
            schema="thoughtsource",
            subset_id="mawps",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mawps_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question": datasets.Value("string"),
                    "numbers": [datasets.Value("float")],
                    "equation": datasets.Value("string"),
                    "answer": datasets.Value("float"),
                    "group_nums": [datasets.Value("int32")],
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
                example_ = {
                    "question": example["Question"],
                    "numbers": [float(x) for x in example["Numbers"].split(" ")],
                    "equation": example["Equation"],
                    "answer": float(example["Answer"]),
                    "group_nums": [int(x.strip()) for x in example["group_nums"][1:-1].split(",")],
                    "body": example["Body"],
                    "ques": example["Ques_Statement"],
                }
                yield key, example_

        elif self.config.schema == "thoughtsource":

            operator_to_verb = {
                "+": "add",
                "-": "subtract",
                "*": "multiply",
                "/": "divide",
            }

            for key, example in data.iterrows():

                example["Question"] = self._untokenize(example["Question"])
                all_numbers = {f"number{i}": x for i, x in enumerate([float(x) for x in example["Numbers"].split(" ")])}
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
                    "id": key,
                    "ref_id": "",
                    "question": example["Question"],
                    "type": "number",
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
        if equation == "number0":
            return []

        pattern = r"[+\-/*] (number[0-9]|int[0-9]|[0-9]+(\.[0-9]+)?) (number[0-9]|int[0-9]|[0-9]+(\.[0-9]+)?)"
        if equation == f"int{idx-1}":
            return []
        else:
            result = re.search(pattern, equation)
            assert result, equation
            # assert (len(re.findall(pattern, equation)) == 1), equation
            equation = equation[: result.span()[0]] + "int" + str(idx) + equation[result.span()[1] :]
            return [result.group().split(" ")] + self._decompose_equation(equation, idx + 1)

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
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        return step6.strip()


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
