import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from cot import Collection
from cot.config import Config

import json


@contextmanager
def chdir(path):
    """Switch working directory to path and back to base directory"""
    base_dir = Path().absolute()
    try:
        os.chdir(os.path.join(base_dir, "tests", path))
        yield
    finally:
        os.chdir(base_dir)


def simple_config():
    """Simple config for testing"""
    config = {
        "api_service": "mock_api",
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


def compare_nested_dict_float_values(dict1, dict2, precision):
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            if not compare_nested_dict_float_values(dict1[key], dict2[key], precision):
                return False
        elif not isinstance(dict1[key], float) or not isinstance(dict2[key], float):
            return False
        elif abs(dict1[key] - dict2[key]) > precision:
            return False
    return True

"""Reads every line of child json and checks whether that line exists in the parent json - sensitive lines such as "}\n" 
- make sure last line is non-empty in both files"""

# Fix for dictionary
def json_file_contains(child_file, parent_file):
     
    path_or_json =f'{parent_file}.json'
    parent_lines = []

    with chdir("unit_tests/data"):
        if isinstance(path_or_json, str):
            with open(path_or_json, "r") as infile:
                for line in infile:
                    parent_lines.append(line)
    
    i = 0 #counter where faulty line is
    non_lines = [] #list of lines not found in parent json
    
    path_or_json =f'{child_file}.json'
    with chdir("unit_tests/data"):
        if isinstance(path_or_json, str):
            with open(path_or_json, "r") as infile:
                for line in infile:
                    #log.debug(line)
                    i += 1
                    if line not in parent_lines:
                        non_lines.append(line) #may be returned if desired
                        return False
    return True
     
        # elif isinstance(path_or_json, dict):
        #     content = path_or_json
        

"""Reads every line of child json and checks whether that line exists in the parent json - sensitive lines such as "}\n" 
- make sure last line is non-empty in both files"""

## Fix for dictionary
# def json_file_contains(child_file, parent_file):

#     parent_json = get_test_collection(parent_file)
#     child_json = get_test_collection(child_file)

#     parent_lines = []
#     parent_json = parent_json.to_json()
#     for line in parent_json:
#         parent_lines.append(line)

#     i = 0 #counter where faulty line is
#     non_lines = [] #list of lines not found in parent json
    
#     child_json = child_json.to_json()
#     for line in child_json:
#         i += 1
#         if line not in parent_lines:
#             print(i)
#             non_lines.append(line) #may be returned if desired
#             return False
#     return True