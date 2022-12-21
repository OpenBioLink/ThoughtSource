import os
from pathlib import Path
from typing import Iterator

import datasets
import pytest
from cot import Collection
from cot.config import Config

from .utils import chdir, get_test_collection, simple_config

# this file contains code from the following source:
# https://github.com/hwchase17/langchain/blob/780ef84cf0ca95aa752ae79b6749e2b39b5b7343/tests/unit_tests/prompts/test_loading.py#L14


def test_correct_example_load() -> None:
    """Test that the correct data is loaded."""
    collection = get_test_collection("worldtree_100_dataset")
    assert (
        collection._cache["worldtree"]["train"][0]["question"]
        == "A parent and a child share several characteristics. Both individuals are tall, have curly hair, are good cooks, and have freckles. Which of these characteristics is a learned behavior?"
    )


def test_similarity_loading_methods():
    """Test that the json file is up to date and loaded correctly."""
    collection_json = get_test_collection("worldtree_100_dataset")
    collection_loaded = Collection(["worldtree"], verbose=False)
    collection_loaded = collection_loaded.select(split="train", number_samples=100)
    assert (
        collection_loaded["worldtree"]["train"][:]
        == collection_json["worldtree"]["train"][:]
    )


def test_find_all_datasets() -> None:
    """Test if all datasets listed in data/dataset_names.txt are found"""
    with chdir("tests/unit_tests/data"):
        with open("dataset_names.txt", "r") as dataset_names:
            dataset_names = dataset_names.read().splitlines()
    dataset_list = Collection._find_datasets()
    dataset_list = [i[0] for i in dataset_list]
    assert dataset_list == dataset_names


def test_load_data_type() -> None:
    """Test that the data type is correct."""
    collection = get_test_collection("worldtree_100_dataset")
    assert isinstance(collection._cache["worldtree"], datasets.dataset_dict.DatasetDict)


def test_load_data_length() -> None:
    """Test that the data length is correct."""
    collection = get_test_collection("worldtree_100_dataset")
    assert len(collection._cache["worldtree"]["train"]) == 100


def test_basic_load_generate_evalute() -> None:
    # using same code as in 0_overview.ipynb

    # 1) Dataset load and selecting random sample
    with chdir("tests/unit_tests/data"):
        collection = Collection.from_json("worldtree_100_dataset.json")
    collection = collection.select(split="train", number_samples=5)
    # 2) Language Model generates chains of thought and then extracts answers
    config = {
        "multiple_choice_answer_format": "Letters",
        "instruction_keys": ["qa-01"],
        "cot_trigger_keys": ["kojima-01"],
        "answer_extraction_keys": ["kojima-A-D"],
        "warn": False,
        "verbose": False,
    }
    collection.generate(config=config)
    # 3) Performance evaluation
    collection.evaluate()


def test_keys_all_plus_None() -> None:
    # test for automatic loading of keys
    config = {
        "instruction_keys": "all",
        "cot_trigger_keys": "all",
        "answer_extraction_keys": "all",
        "warn": False,
        "verbose": False,
    }
    config = Config(**config)
    instruction_keys = ["qa-01", "qa-02", "qa-03", "qa-04"]
    cot_trigger_keys = [
        "kojima-01",
        "kojima-02",
        "kojima-03",
        "kojima-04",
        "kojima-05",
        "kojima-06",
        "kojima-07",
        "kojima-08",
        "kojima-09",
        "kojima-10",
        "kojima-11",
        "kojima-12",
        "kojima-13",
        "kojima-14",
        "lievin-01",
        "lievin-02",
        "lievin-03",
        "lievin-04",
        "lievin-05",
        "lievin-06",
        "lievin-07",
        "lievin-08",
        "lievin-09",
        "lievin-10",
        "lievin-11",
        "lievin-12",
        "lievin-13",
        "lievin-14",
        "lievin-15",
        "lievin-16",
        "lievin-17",
        "lievin-18",
        "lievin-19",
        "lievin-20",
        "lievin-21",
        "lievin-22",
        "lievin-23",
        "lievin-24",
        "lievin-25",
        "lievin-26",
        "lievin-27",
        "lievin-28",
    ]
    answer_extraction_keys = [
        "kojima-01",
        "kojima-02",
        "kojima-03",
        "kojima-numerals",
        "kojima-yes-no",
        "kojima-A-C",
        "kojima-A-D",
        "kojima-A-E",
        "kojima-A-F",
    ]
    assert config.instruction_keys == [None] + instruction_keys
    assert config.cot_trigger_keys == [None] + cot_trigger_keys
    assert config.answer_extraction_keys == [None] + answer_extraction_keys

    # test for selecting "all" in keys, which should be the same
    config = {
        "instruction_keys": "all",
        "cot_trigger_keys": "all",
        "answer_extraction_keys": "all",
        "warn": False,
        "verbose": False,
    }
    config = Config(**config)
    assert config.instruction_keys == [None] + instruction_keys
    assert config.cot_trigger_keys == [None] + cot_trigger_keys
    assert config.answer_extraction_keys == [None] + answer_extraction_keys


def test_template_default_f_strings() -> None:
    collection = get_test_collection("test_1_dataset")
    config = simple_config()
    collection.generate(config=config)
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["prompt_text"]
        == """Answer the following question through step-by-step reasoning.

Question
A) choice A
B) choice B
C) choice C
D) choice D

Answer: Let's think step by step."""
    )
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["answers"][0][
            "answer_extraction_text"
        ]
        == """Answer the following question through step-by-step reasoning.

Question
A) choice A
B) choice B
C) choice C
D) choice D

Answer: Let's think step by step. Test mock chain of thought.
Therefore, the answer is"""
    )


def test_template_instruction_is_none() -> None:
    collection = get_test_collection("test_1_dataset")
    config = simple_config()
    config["instruction_keys"] = [None]
    collection.generate(config=config)
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["prompt_text"]
        == """Question
A) choice A
B) choice B
C) choice C
D) choice D

Answer: Let's think step by step."""
    )
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["answers"][0][
            "answer_extraction_text"
        ]
        == """Question
A) choice A
B) choice B
C) choice C
D) choice D

Answer: Let's think step by step. Test mock chain of thought.
Therefore, the answer is"""
    )


# these tests take a very long time
# either change them or do not run them every time


def test_thougthsource() -> None:
    collection = Collection("all", generate_mode="recache")
    """Test that id is unique within a dataset"""
    for name, dataset in collection:
        for split in dataset:
            pd_ = dataset[split].to_pandas()
            assert (
                len(pd_["id"]) == pd_["id"].nunique()
            ), f"IDs are not unique in {name} {split}"


def test_source() -> None:
    collection = Collection("all", generate_mode="recache", source=True)
    assert collection


"""
# TBD rework takes too long for tests
def test_merging() -> None:
    collection1 = Collection(["worldtree"])
    collection2 = Collection(["entailment_bank"])
    collection_all = collection1.merge(collection2)
    assert len(collection_all) == 2
"""
