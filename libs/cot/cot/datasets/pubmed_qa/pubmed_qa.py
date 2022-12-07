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
from typing import Dict, List, Tuple

import datasets
import json
from cot.utils import schemas, map_example_to_lievin_cot
from cot.utils.configs import ThoughtSourceConfig
from collections import defaultdict
from tqdm import tqdm
import glob

_LOCAL = False

_CITATION = """\
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}
"""

_DATASETNAME = "pubmed_qa"

_DESCRIPTION = """\
PubMedQA is a novel biomedical question answering (QA) dataset collected from PubMed abstracts.
The task of PubMedQA is to answer research biomedical questions with yes/no/maybe using the corresponding abstracts.
PubMedQA has 1k expert-annotated (PQA-L), 61.2k unlabeled (PQA-U) and 211.3k artificially generated QA instances (PQA-A).
Each PubMedQA instance is composed of:
  (1) a question which is either an existing research article title or derived from one,
  (2) a context which is the corresponding PubMed abstract without its conclusion,
  (3) a long answer, which is the conclusion of the abstract and, presumably, answers the research question, and
  (4) a yes/no/maybe answer which summarizes the conclusion.
PubMedQA is the first QA dataset where reasoning over biomedical research texts,
especially their quantitative contents, is required to answer the questions.
PubMedQA datasets comprise of 3 different subsets:
  (1) PubMedQA Labeled (PQA-L): A labeled PubMedQA subset comprises of 1k manually annotated yes/no/maybe QA data collected from PubMed articles.
  (2) PubMedQA Artificial (PQA-A): An artificially labelled PubMedQA subset comprises of 211.3k PubMed articles with automatically generated questions from the statement titles and yes/no answer labels generated using a simple heuristic.
  (3) PubMedQA Unlabeled (PQA-U): An unlabeled PubMedQA subset comprises of 61.2k context-question pairs data collected from PubMed articles.

This dataset only supports PQA-L.
"""

_HOMEPAGE = "https://github.com/pubmedqa/pubmedqa"

_LICENSE = "MIT"

_URLS = {
    "pubmed": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
    "cots": "https://drive.google.com/u/0/uc?export=download&confirm=t&id=1l0y35SO0mRhc81Asrvo5WU7Mhi1mm2a0"
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class PubmedQADataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="pubmed_qa_source",
            version=SOURCE_VERSION,
            description="PubmedQA source schema",
            schema="source",
            subset_id="pubmed_qa",
        ),
        ThoughtSourceConfig(
            name="pubmed_qa_thoughtsource",
            version=BIGBIO_VERSION,
            description="PubmedQA thoughtsource schema",
            schema="thoughtsource",
            subset_id="pubmed_qa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pubmed_qa_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "PUBMED_ID": datasets.Value("string"),
                    "QUESTION": datasets.Value("string"),
                    "CONTEXTS": [datasets.Value("string")],
                    "LABELS": [datasets.Value("string")],
                    "MESHES": [datasets.Value("string")],
                    "YEAR": datasets.Value("string"),
                    "reasoning_required_pred": datasets.Value("string"),
                    "reasoning_free_pred": datasets.Value("string"),
                    "final_decision": datasets.Value("string"),
                    "LONG_ANSWER": datasets.Value("string"),
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
        cotspath = os.path.join(data_dir["cots"], "thought-source-med")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["pubmed"],
                    "cotspath": cotspath,
                },
            )
        ]

    def _generate_examples(self, filepath, cotspath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as infile:
            data = json.load(infile)

        if self.config.schema == "source":
            for key, example in data.items():
                example["PUBMED_ID"] = key
                yield key, example
        elif self.config.schema == "thoughtsource":

            cots = defaultdict(list)
            if cotspath is not None:
                for file in tqdm(glob.glob(cotspath + r"\[0-4]-pubmedqa*\*.json"), desc="Preparing Lievin CoTs"):
                    filename = os.path.basename(file)[:-len(".json")]
                    id = int(filename.split("_")[1].split("-")[1])
                    with open(file, "r") as infile:
                        example = json.load(infile)
                    cots[id].append(example)

            for key, example in data.items():

                generated_cots = []
                for item_idx, item in enumerate(cots[int(key)]):
                    assert (example["QUESTION"] == item["question"]), f"Question mismatch {example['QUESTION']} {item['question']}"
                    cot_item = map_example_to_lievin_cot(f"{key}_{item_idx}", item, "pubmed_qa")
                    generated_cots.append(cot_item)


                example_ = {
                    "id": key,
                    "ref_id": key,
                    "question": example["QUESTION"],
                    "type": "multiplechoice",
                    "choices": ["yes", "no", "maybe"],
                    "context": "\n".join(example["CONTEXTS"]),
                    "cot": "",
                    "answer": [example["final_decision"]],
                    "feedback": [],
                    "generated_cot": generated_cots,
                }

                yield key, example_


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    a = datasets.load_dataset(__file__)
    from pprint import pprint
    pprint(a["train"][0])