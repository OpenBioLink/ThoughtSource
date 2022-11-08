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

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig

_LOCAL = False

_CITATION = """\
@misc{lamm2020qed,
    title={QED: A Framework and Dataset for Explanations in Question Answering},
    author={Matthew Lamm and Jennimaria Palomaki and Chris Alberti and Daniel Andor
    and Eunsol Choi and Livio Baldini Soares and Michael Collins},
    year={2020},
    eprint={2009.06354},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DATASETNAME = "qed"

_DESCRIPTION = """\
QED is a linguistically principled framework for explanations in question answering. As presented in the paper, given a question
and a passage, QED represents an explanation of the answer as a combination of discrete, human-interpretable steps.
"""

_HOMEPAGE = "https://github.com/google-research-datasets/QED"

_LICENSE = "Unknown"

_URLS = {
    "train": "https://github.com/google-research-datasets/QED/raw/master/qed-train.jsonlines",
    "dev": "https://github.com/google-research-datasets/QED/raw/master/qed-dev.jsonlines",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class QedDataset(datasets.GeneratorBasedBuilder):
    """QED is a linguistically principled framework for explanations in question answering."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="qed_source",
            version=SOURCE_VERSION,
            description="QED source schema",
            schema="source",
            subset_id="qed",
        ),
        ThoughtSourceConfig(
            name="qed_thoughtsource",
            version=BIGBIO_VERSION,
            description="QED thoughtsource schema",
            schema="thoughtsource",
            subset_id="qed",
        ),
    ]

    DEFAULT_CONFIG_NAME = "qed_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "example_id": datasets.Value("string"),
                    "title_text": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "question_text": datasets.Value("string"),
                    "paragraph_text": datasets.Value("string"),
                    "sentence_starts": [datasets.Value("int64")],
                    "original_nq_answers": [
                        [
                            {
                                "start": datasets.Value("int64"),
                                "end": datasets.Value("int64"),
                                "string": datasets.Value("string"),
                            }
                        ],
                    ],
                    "annotation": {
                        "referential_equalities": [
                            {
                                "question_reference": {
                                    "start": datasets.Value("int64"),
                                    "end": datasets.Value("int64"),
                                    "string": datasets.Value("string"),
                                },
                                "sentence_reference": {
                                    "start": datasets.Value("int64"),
                                    "end": datasets.Value("int64"),
                                    "bridge": datasets.Value("string"),
                                    "string": datasets.Value("string"),
                                },
                            },
                        ],
                        "answer": [
                            {
                                "sentence_reference": {
                                    "start": datasets.Value("int64"),
                                    "end": datasets.Value("int64"),
                                    "bridge": datasets.Value("string"),
                                    "string": datasets.Value("string"),
                                },
                                "paragraph_reference": {
                                    "start": datasets.Value("int64"),
                                    "end": datasets.Value("int64"),
                                    "string": datasets.Value("string"),
                                },
                            }
                        ],
                        "explanation_type": datasets.Value("string"),
                        "selected_sentence": {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "string": datasets.Value("string"),
                        },
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

        data_dir = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["dev"],
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as json_file:
            data = [json.loads(line) for line in json_file]

        if self.config.schema == "source":
            for key, example in enumerate(data):
                if "referential_equalities" not in example["annotation"]:
                    example["annotation"]["referential_equalities"] = []
                if "answer" not in example["annotation"]:
                    example["annotation"]["answer"] = []
                if "selected_sentence" not in example["annotation"]:
                    example["annotation"]["selected_sentence"] = {
                        "start": None,
                        "end": None,
                        "string": None,
                    }
                for x in example["annotation"]["answer"]:
                    x["sentence_reference"]["bridge"] = str(x["sentence_reference"]["bridge"])
                for x in example["annotation"]["referential_equalities"]:
                    x["sentence_reference"]["bridge"] = str(x["sentence_reference"]["bridge"])
                yield key, example

        elif self.config.schema == "thoughtsource":
            for key, example in enumerate(data):

                annotation = example["annotation"]

                # skip examples without explanation
                if annotation["explanation_type"] == "none" or annotation["explanation_type"] == "multi_sentence":
                    continue

                cot = []
                cot.append(f"The answer is contained in the following sentence: {annotation['selected_sentence']['string']}")
                for x in annotation["referential_equalities"]:

                    if x["sentence_reference"]["bridge"] is not False:
                        if x["sentence_reference"]["string"] != "":
                            cot.append(
                                f"The noun phrase {x['sentence_reference']['string']} "
                                + f"in the sentence refers to {x['sentence_reference']['string']} {x['sentence_reference']['bridge']} "
                                + f"the noun phrase {x['question_reference']['string']} in the question."
                            )
                    else:
                        cot.append(
                            f"The noun phrase {x['sentence_reference']['string']} "
                            + f"in the sentence and the noun phrase {x['question_reference']['string']} "
                            + "in the question refer to the same thing."
                        )
                for x in annotation["answer"]:
                    if x["sentence_reference"]["bridge"] is not False:
                        if x["sentence_reference"]["string"] != "":
                            cot.append(
                                f"The noun phrase {x['sentence_reference']['string']} "
                                + f"in the sentence and the noun phrase {x['paragraph_reference']['string']} "
                                + "in the context refer to the same thing."
                            )
                    else:
                        assert x["sentence_reference"]["string"] == x["paragraph_reference"]["string"], f"Ohno {x}"

                example["question_text"] = example["question_text"].capitalize()
                example["question_text"] = (
                    example["question_text"] + "?" if example["question_text"][-1] != "?" else example["question_text"]
                )

                # Detokenization
                cot = [self._untokenize(x.strip()) for x in cot]
                example["question_text"] = self._untokenize(example["question_text"])
                example["title_text"] = self._untokenize(example["title_text"])
                example["paragraph_text"] = self._untokenize(example["paragraph_text"])

                example_ = {
                    "id": example["example_id"],
                    "question_id": example["example_id"],
                    "document_id": example["example_id"],
                    "question": example["question_text"],
                    "type": "collection",
                    "cot_type": "list",
                    "choices": [],
                    "context": f"Title: {example['title_text']} Text: {example['paragraph_text']}",
                    "cot": cot,
                    "answer": [x[0]["string"] for x in example["original_nq_answers"]],
                    "feedback": None,
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
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        return step6.strip()


if __name__ == "__main__":
    datasets.load_dataset(__file__)
