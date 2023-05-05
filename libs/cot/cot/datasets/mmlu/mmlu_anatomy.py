# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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


import csv

import os
import re
from typing import Dict, List, Tuple

import datasets

from cot.utils import schemas
from cot.utils.configs import ThoughtSourceConfig

_CITATION = """\
@article{hendryckstest2021,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }
"""
_DATASETNAME = "mmlu_anatomy"

_DESCRIPTION = """\
anatomy subset of tasksource/mmlu
"""

_HOMEPAGE = "https://github.com/hendrycks/test"

_LICENSE = ""
_URLS = {
    _DATASETNAME: "https://www.dropbox.com/s/nv4z13trkpq80bj/mmlu.tar?dl=1"
}


_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class mmluDataset(datasets.GeneratorBasedBuilder):
    """MC test consisting of anatomy questions"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="source", 
            version=SOURCE_VERSION,
            description="mmlu_anatomy source schema",
            schema="source",
            subset_id="mmlu_anatomy",
            # description=f"Hendrycks Test Subject {sub}"
        ),
        # for sub in _SUBJECTS
         ThoughtSourceConfig(
            name="thoughtsource",
            version=BIGBIO_VERSION,
            description="mmlu_anatomy thoughtsource schema",
            schema="thoughtsource",
            subset_id="mmlu_anatomy",
        ),
    ]

    DEFAULT_CONFIG_NAME = "thoughtsource"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "choices": datasets.features.Sequence(datasets.Value("string")),
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
                gen_kwargs={
                    "split": "train",
                    "filepath": os.path.join(
                        data_dir,
                        "data", 
                        "dev",
                        "anatomy_dev.csv",
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepath": os.path.join(
                        data_dir,
                        "data", 
                        "test",
                        "anatomy_test.csv",
                    )
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation",
                    "filepath": os.path.join(
                        data_dir,
                        "data", 
                        "val",
                        "anatomy_val.csv",
                    )
                },
            ),
            
        ]
   
    def _generate_examples(self, filepath, split) -> Tuple[int, Dict]:
        f = open(filepath, encoding='UTF8')
        data = csv.reader(f)

        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":
            for key, example in enumerate(data):
                yield key, example

        elif self.config.schema == "thoughtsource":
            id_counter = 0
            for key, example in enumerate(data):
                yield key, self._source_to_thoughtsource(example, split, id_counter)
                id_counter += 1

    def _source_to_thoughtsource(self, example, split, id):
        cot = []

        # resolve ( tree ; plant ) synsets
        pattern = r"\((.*?) ; (.*?)\)"
        for idx in range(len(cot)):
            match = re.search(pattern, cot[idx])
            while match:
                cot[idx] = cot[idx][: match.span()[0]] + match.group(1) + cot[idx][match.span()[1] :]
                match = re.search(pattern, cot[idx])

        cot = [x.capitalize() for x in cot]
        cot = [x + "." if x[-1] not in [".", "!", "?"] else x for x in cot]

        example_ = {
            "id": "mmlu_anatomy_" + split + "_" + str(id),
            "ref_id": "",
            "question": example[0],
            "type": "multiplechoice",
            "choices": example[1:5],
            "context": "",
            "cot": cot,
            "answer": [example[self._answer_mapper(example[5])]],
            "feedback": [],
            "generated_cot": [],
        }
        return example_
    
    def _answer_mapper(self, answer):
        if answer == 'A':
            return(1)
        elif answer == 'B':
            return(2)
        elif answer == 'C':
            return(3)
        elif answer == 'D':
            return(4)