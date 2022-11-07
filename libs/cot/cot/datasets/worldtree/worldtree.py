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
from typing import Dict, List, Tuple

import datasets

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig

_LOCAL = False

_CITATION = """\
@inproceedings{xie-etal-2020-worldtree,
    title = "{W}orld{T}ree V2: A Corpus of Science-Domain Structured Explanations and Inference Patterns supporting Multi-Hop Inference",
    author = "Xie, Zhengnan  and
      Thiem, Sebastian  and
      Martin, Jaycie  and
      Wainwright, Elizabeth  and
      Marmorstein, Steven  and
      Jansen, Peter",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.671",
    pages = "5456--5473",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_DATASETNAME = "worldtree"

_DESCRIPTION = """\
This is the February 2020 snapshot of the Worldtree corpus of explanation graphs, explanatory role ratings, and associated
tablestore, from the paper WorldTree V2: A Corpus of Science-Domain Structured Explanations and Inference Patterns supporting
Multi-Hop Inference (LREC 2020). WorldTree is one of the most detailed multi-hop question answering/explanation datasets, where
questions require combining between 1 and 16 facts (average 6) to generate detailed explanations for question answering
inference. Explanation graphs for approximately 4,400 questions, and 9,000 tablestore rows across 81 semi-structured tables are
provided.
"""

_HOMEPAGE = "http://cognitiveai.org/explanationbank/"

_LICENSE = "EULA AI2 Mercury Dataset"

_URLS = {
    _DATASETNAME: "http://www.cognitiveai.org/dist/WorldtreeExplanationCorpusV2.1_Feb2020.zip",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = (
    []
)  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class WorldtreeDataset(datasets.GeneratorBasedBuilder):
    """Worldtree is of the most detailed multi-hop question answering/explanation datasets"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="worldtree_source",
            version=SOURCE_VERSION,
            description="Worldtree source schema",
            schema="source",
            subset_id="worldtree",
        ),
        ThoughtSourceConfig(
            name="worldtree_thoughtsource",
            version=BIGBIO_VERSION,
            description="Worldtree thoughtsource schema",
            schema="thoughtsource",
            subset_id="worldtree",
        ),
    ]

    DEFAULT_CONFIG_NAME = "worldtree_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "choices": datasets.Value("string"),
                    "explanation": [datasets.Value("string")],
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "WorldtreeExplanationCorpusV2.1_Feb2020",
                        "explanations-plaintext",
                        "explanations.plaintext.train.txt",
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "WorldtreeExplanationCorpusV2.1_Feb2020",
                        "explanations-plaintext",
                        "explanations.plaintext.test.txt",
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "WorldtreeExplanationCorpusV2.1_Feb2020",
                        "explanations-plaintext",
                        "explanations.plaintext.dev.txt",
                    )
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, "r") as infile:
                for key, example in enumerate(self._generate_parsed_documents(infile)):
                    yield key, example

        elif self.config.schema == "thoughtsource":
            with open(filepath, "r") as infile:
                for key, example in enumerate(self._generate_parsed_documents(infile)):
                    yield key, self._source_to_thoughtsource(example)

    def _generate_parsed_documents(self, fstream):
        for raw_document in self._generate_raw_documents(fstream):

            question_id = int(raw_document[0][10:])
            field1 = raw_document[1][10:].split("\t")
            question, *field1 = field1
            choices = [x.split(": ")[1] for x in field1]
            answer = int(raw_document[2][16:])
            answer = choices[answer]
            explanations = [
                re.search(r".*(?= \(.*\) \(.*\))", x).group()
                for x in raw_document[4:]
                if "No UUID specified" not in x
            ]

            yield {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "choices": choices,
                "explanation": explanations,
            }

    def _generate_raw_documents(self, fstream):
        raw_document = []
        for line in fstream:
            if line.strip():
                raw_document.append(line.strip())
            elif raw_document:
                yield raw_document
                raw_document = []
        if raw_document:
            yield raw_document

    def _source_to_thoughtsource(self, example):
        cot = example["explanation"]

        # resolve ( tree ; plant ) synsets
        pattern = r"\((.*?) ; (.*?)\)"
        for idx in range(len(cot)):
            match = re.search(pattern, cot[idx])
            while match:
                cot[idx] = (
                    cot[idx][: match.span()[0]]
                    + match.group(1)
                    + cot[idx][match.span()[1] :]
                )
                match = re.search(pattern, cot[idx])

        cot = [x.capitalize() for x in cot]
        cot = [x + "." if x[-1] not in [".", "!", "?"] else x for x in cot]

        example_ = {
            "id": example["question_id"],
            "question_id": example["question_id"],
            "document_id": example["question_id"],
            "question": example["question"],
            "type": "multiplechoice",
            "cot_type": "list",
            "choices": example["choices"],
            "context": "",
            "cot": cot,
            "answer": [example["answer"]],
            "feedback": [],
            "generated_cot": [],
        }
        return example_


if __name__ == "__main__":
    datasets.load_dataset(__file__)
