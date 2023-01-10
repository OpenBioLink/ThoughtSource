import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from cot import Collection
from cot.config import Config


@contextmanager
def chdir(path):
    """Switch working directory to path and back to base directory"""
    base_dir = Path().absolute()
    try:
        os.chdir(os.path.join(base_dir, "libs/cot/tests", path))
        # os.chdir(path)
        yield
    finally:
        os.chdir(base_dir)


def simple_config():
    """Simple config for testing"""
    config = {
        "api_service": "mock_api",
        "multiple_choice_answer_format": "Letters",
        "instruction_keys": ["qa-01"],
        "cot_trigger_keys": ["kojima-01"],
        "answer_extraction_keys": ["kojima-01"],
        "warn": False,
        "verbose": False,
    }
    config = Config(**config)
    config = asdict(config)
    return config


def get_test_collection(name: str) -> Collection:
    """Load a test collection from the data folder"""
    with chdir("unit_tests/data"):
        collection = Collection.from_json(f"{name}.json")
    return collection
