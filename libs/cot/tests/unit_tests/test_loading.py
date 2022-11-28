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

def test_load_data_type():
    """Test that the data type is correct."""
    with chdir("tests/unit_tests"):
        collection = Collection.from_json("data/worldtree_100_dataset.json")
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
    with chdir("tests/unit_tests"):
        collection = Collection.from_json("data/worldtree_100_dataset.json")
        assert len(collection._cache["worldtree"]["train"]) == 100

def test_unique_id() -> None:
    """Test that id is unique within a dataset"""
    collection = Collection("all")
    for name, dataset in collection:
        for split in dataset:
            pd_ = dataset[split].to_pandas()
            assert (len(pd_["id"]) == pd_["id"].nunique()), f"IDs are not unique in {name} {split}"


def test_merging() -> None:
    collection1 = Collection(["worldtree"])
    collection2 = Collection(["entailment_bank"])
    collection_all = collection1.merge(collection2)
    assert len(collection_all) == 2

