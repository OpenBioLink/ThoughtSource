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
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import os
from typing import List, Tuple, Dict

import datasets
from dataloader.utils import schemas
from dataloader.utils.configs import ThoughtSourceConfig

_LOCAL = False

_CITATION = """\
@inproceedings{talmor-etal-2019-commonsenseqa,
    title = "{C}ommonsense{QA}: A Question Answering Challenge Targeting Commonsense Knowledge",
    author = "Talmor, Alon  and
        Herzig, Jonathan  and
        Lourie, Nicholas  and
        Berant, Jonathan",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational
        Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1421",
    doi = "10.18653/v1/N19-1421",
    pages = "4149--4158",
    abstract = "When answering a question, people often draw upon their rich world knowledge in addition to the particular
        context. Recent work has focused primarily on answering questions given some relevant document or context, and required
        very little general background. To investigate question answering with prior knowledge, we present CommonsenseQA: a
        challenging new dataset for commonsense question answering. To capture common sense beyond associations, we extract from
        ConceptNet (Speer et al., 2017) multiple target concepts that have the same semantic relation to a single source concept.
        Crowd-workers are asked to author multiple-choice questions that mention the source concept and discriminate in turn
        between each of the target concepts. This encourages workers to create questions with complex semantics that often
        require prior knowledge. We create 12,247 questions through this procedure and demonstrate the difficulty of our task
        with a large number of strong baselines. Our best baseline is based on BERT-large (Devlin et al., 2018) and obtains
        56{\%} accuracy, well below human performance, which is 89{\%}.",
}
"""

_DATASETNAME = "commonsense_qa"

_DESCRIPTION = """\
CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to
predict the correct answers . It contains 12,102 questions with one correct answer and four distractor answers.  The dataset is
provided in two major training/validation/testing set splits: "Random split" which is the main evaluation split, and "Question
token split", see paper for details.

CommonsenseQA does not come with explanations per default. We use explanations from Aggarwal et al, 2021, which can be found at 
https://github.com/dair-iitd/ECQA-Dataset.
"""

_HOMEPAGE = "https://www.tau-nlp.sites.tau.ac.il/commonsenseqa"

_LICENSE = "Unknown"

_URLS = {
    "commonsense": {
        "train": "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl",
        "dev": "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl",
        "test": "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl",
    },
    "ecqa": "https://github.com/dair-iitd/ECQA-Dataset/raw/main/ecqa.jsonl"
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = []  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class CommonsenseQADataset(datasets.GeneratorBasedBuilder):
    """CommonsenseQA is a new multiple-choice commonsense knowledge question answering dataset containing 12,102 questions."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        ThoughtSourceConfig(
            name="commonsense_qa_source",
            version=SOURCE_VERSION,
            description="CommonsenseQA source schema",
            schema="source",
            subset_id="commonsense_qa",
        ),
        ThoughtSourceConfig(
            name="commonsense_qa_thoughtsource",
            version=BIGBIO_VERSION,
            description="CommonsenseQA thoughtsource schema",
            schema="thoughtsource",
            subset_id="commonsense_qa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "commonsense_qa_thoughtsource"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "answerKey": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "question": {
                        "question_concept": datasets.Value("string"),
                        "choices": [
                            {
                                "label": datasets.Value("string"),
                                "text": datasets.Value("string"),
                            },
                        ],
                        "stem": datasets.Value("string"),
                    },
                    "explanation": datasets.Value("string"),
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

        with open(data_dir["ecqa"], 'r') as json_file:
            ecqa = [json.loads(line) for line in json_file]
        ecqa = {x["id"]: x["explanation"] for x in ecqa}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["commonsense"]["train"],
                    "ecqa": ecqa,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["commonsense"]["test"],
                    "ecqa": ecqa,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["commonsense"]["dev"],
                    "ecqa": ecqa,
                },
            ),
        ]

    def _generate_examples(self, filepath, ecqa) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        with open(filepath, 'r') as json_file:
            data = [json.loads(line) for line in json_file]

        if self.config.schema == "source":
            for key, example in enumerate(data):
                if "answerKey" not in example:
                    example["answerKey"] = None
                example["explanation"] = ecqa.get(example["id"], "")
                yield key, example

        elif self.config.schema == "thoughtsource":
            for key, example in enumerate(data):
                
                choices = {x["label"]: x["text"] for x in example["question"]["choices"]}
                example_ = {
                    "id": example["id"],
                    "question_id": example["id"],
                    "document_id": example["id"],
                    "question": example["question"]["stem"],
                    "type": "multiplechoice",
                    "cot_type": "list",
                    "choices": choices.values(),
                    "context": "",
                    "cot": [x.capitalize() for x in sent_tokenize(ecqa[example["id"]])] if example["id"] in ecqa else [],
                    "answer": [choices[example["answerKey"]]] if "answerKey" in example else [],
                    "feedback": [],
                    "cot_after_feedback": [],
                    "answer_after_feedback": [],
                }
                yield key, example_


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
