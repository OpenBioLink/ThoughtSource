from contextlib import contextmanager
from pathlib import Path
import os
from cot import Collection
from cot.config import Config
from dataclasses import asdict

@contextmanager
def chdir(path):
    """Switch working directory to path and back to base directory"""
    base_dir = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(base_dir)

def simple_config():
    """Simple config for testing"""
    config = {
        "debug": True,
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
    with chdir("tests/unit_tests/data"):
        collection = Collection.from_json(f"{name}.json")
    return collection
