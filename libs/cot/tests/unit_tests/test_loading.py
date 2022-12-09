import pytest
import datasets
from contextlib import contextmanager
from typing import Iterator
from cot import Collection
import os
from pathlib import Path

# this file contains code from the following source:
# https://github.com/hwchase17/langchain/blob/780ef84cf0ca95aa752ae79b6749e2b39b5b7343/tests/unit_tests/prompts/test_loading.py#L14

@contextmanager
def chdir(path):
    """Switch working directory to path and back to base directory"""
    base_dir = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(base_dir)

def test_correct_example_load():
    """Test that the correct data is loaded."""
    with chdir("tests/unit_tests/data"):
        collection = Collection.from_json("worldtree_100_dataset.json")
        assert collection._cache["worldtree"]["train"][0]["question"] == "A parent and a child share several characteristics. Both individuals are tall, have curly hair, are good cooks, and have freckles. Which of these characteristics is a learned behavior?"

def test_similarity_loading_methods():
    """Test that the json file is up to date and loaded correctly."""
    with chdir("tests/unit_tests/data"):
        collection_json = Collection.from_json("worldtree_100_dataset.json")
        collection_loaded = Collection(["worldtree"], verbose=False)
        collection_loaded = collection_loaded.select(split="train", number_samples=100)
        assert collection_loaded["worldtree"]["train"][:] == collection_json["worldtree"]["train"][:]

def test_find_all_datasets() -> None:
    """Test if all datasets listed in data/dataset_names.txt are found"""
    with chdir("tests/unit_tests/data"):
        with open("dataset_names.txt", "r") as dataset_names:
            dataset_names = dataset_names.read().splitlines()
    dataset_list = Collection._find_datasets()
    dataset_list = [i[0] for i in dataset_list]
    assert dataset_list == dataset_names

def test_load_data_type():
    """Test that the data type is correct."""
    with chdir("tests/unit_tests/data"):
        collection = Collection.from_json("worldtree_100_dataset.json")
        assert isinstance(collection._cache["worldtree"], datasets.dataset_dict.DatasetDict)

# same test using pathlib, but it doesn't work
# def test_load_data_type():
#     """Test that the data type is correct."""
#     subdir = Path().absolute() / Path("tests/unit_tests")
#     with subdir:
#         collection = Collection.from_json("data/worldtree_100_dataset.json")
#         assert isinstance(collection._cache["worldtree"], datasets.dataset_dict.DatasetDict)

def test_load_data_length():
    """Test that the data length is correct."""
    with chdir("tests/unit_tests/data"):
        collection = Collection.from_json("worldtree_100_dataset.json")
        assert len(collection._cache["worldtree"]["train"]) == 100

def test_thougthsource() -> None:
    collection = Collection("all", generate_mode="recache")
    """Test that id is unique within a dataset"""
    for name, dataset in collection:
        for split in dataset:
            pd_ = dataset[split].to_pandas()
            assert (len(pd_["id"]) == pd_["id"].nunique()), f"IDs are not unique in {name} {split}"

def test_basic_load_generate_evalute() -> None:
    # 1) Dataset load and selecting random sample
    with chdir("tests/unit_tests/data"):
        collection = Collection.from_json("worldtree_100_dataset.json")
    collection = collection.select(split="train", number_samples=5)
    # 2) Language Model generates chains of thought and then extracts answers
    config={
        "debug": True,
        "multiple_choice_answer_format": "Letters",
        "instruction_keys": ['qa-01'],
        "cot_trigger_keys": ['kojima-01'],
        "answer_extraction_keys": ['kojima-A-D'],
        "warn": False,
        "verbose": False,
    }
    collection.generate(config=config)
    # 3) Performance evaluation
    collection.evaluate()

def test_source() -> None:
    collection = Collection("all", generate_mode="recache", source=True)
    assert (collection)

"""
# TBD rework takes too long for tests
def test_merging() -> None:
    collection1 = Collection(["worldtree"])
    collection2 = Collection(["entailment_bank"])
    collection_all = collection1.merge(collection2)
    assert len(collection_all) == 2
"""

