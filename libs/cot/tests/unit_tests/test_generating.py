from typing import Iterator

import datasets
import pytest
from cot import Collection

# import os
# from pathlib import Path
from cot.generate import Correct_output

from .utils import chdir, get_test_collection, simple_config


def test_correct_output() -> None:
    """Test the Correct_output class"""
    simple_dict = {"a": 1, "b": None}
    # leave number as is
    simple_sentence_a = "{a}"
    # if key is None return empty string
    simple_sentence_b = "{b}"
    # if key is missing return empty string
    simple_sentence_c = "{c}"
    correct_a = "1"
    correct_b = ""
    correct_c = ""
    answer_a = simple_sentence_a.format_map(Correct_output(simple_dict))
    answer_b = simple_sentence_b.format_map(Correct_output(simple_dict))
    answer_c = simple_sentence_c.format_map(Correct_output(simple_dict))
    assert answer_a == correct_a
    assert answer_b == correct_b
    assert answer_c == correct_c

    # without correct output mapping:
    correct_normal_b = "None"
    normal_b = simple_sentence_b.format_map(simple_dict)
    assert normal_b == correct_normal_b


def test_template_input_only_string():
    collection = get_test_collection("test_1_dataset")
    config = simple_config()
    text1 = "abc"
    text2 = "123"
    config["template_cot_generation"] = text1
    config["template_answer_extraction"] = text2
    collection.generate(config=config)

    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["cot_trigger_template"]
        == text1
    )
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["answers"][0][
            "answer_extraction_template"
        ]
        == text2
    )

    assert collection["worldtree"]["train"][0]["generated_cot"][0]["prompt_text"] == ""
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["answers"][0][
            "answer_extraction_text"
        ]
        == ""
    )

    collection.full_text_prompts()

    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["prompt_text"] == text1
    )
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["answers"][0][
            "answer_extraction_text"
        ]
        == text2
    )


def test_template_cot_wrong_variables():
    """Test if error is raised when wrong variables are used in templates"""
    collection = get_test_collection("test_1_dataset")
    config = simple_config()
    text1 = "abc {wrong_variable}"
    config["template_cot_generation"] = text1
    with pytest.raises(ValueError) as error:
        collection.generate(config=config)


def test_template_answer_extraction_wrong_variables():
    """Test if error is raised when wrong variables are used in templates"""
    collection = get_test_collection("test_1_dataset")
    config = simple_config()
    text1 = "abc {wrong_variable}"
    config["template_answer_extraction"] = text1
    with pytest.raises(ValueError) as error:
        collection.generate(config=config)


def test_template_default_f_strings() -> None:
    collection = get_test_collection("test_1_dataset")
    config = simple_config()
    collection.generate(config=config)
    collection.full_text_prompts()
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
    collection.full_text_prompts()
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


def test_generate_change_config() -> None:
    # Dataset loading and selecting a random sample
    collection = Collection(["worldtree"], verbose=False)
    collection = collection.select(split="train", number_samples=1)

    config = simple_config()
    config["instruction_keys"] = ["qa-01"]

    collection.generate(config=config)

    # Again Dataset loading and selecting a random sample
    collection = Collection(["worldtree"], verbose=False)
    collection = collection.select(split="train", number_samples=1)

    config = simple_config()
    # Set a new the instruction key
    config["instruction_keys"] = ["qa-02"]

    # Generate with a new config
    collection.generate(config=config)

    # Check if the instruction is the one from the new config
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["instruction"]
        == "qa-02"
    )


def test_generate_twice() -> None:
    # Dataset loading and selecting a random sample
    collection = Collection(["worldtree"], verbose=False)
    collection = collection.select(split="train", number_samples=1)

    config = simple_config()
    config["instruction_keys"] = ["qa-01"]

    collection.generate(config=config)

    config = simple_config()
    # Set a new the instruction key
    config["instruction_keys"] = ["qa-02"]

    # Generate with a new config
    collection.generate(config=config)

    # Check if both generated_cot are saved and have the correct instructions
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][0]["instruction"]
        == "qa-01"
    )
    assert (
        collection["worldtree"]["train"][0]["generated_cot"][1]["instruction"]
        == "qa-02"
    )


def test_keep_generated_cot() -> None:
    ## keep all authors
    # load comparison file
    collection = get_test_collection("test_1_delete_cots")
    # load file with nothing deleted
    collection_delete = get_test_collection("test_1_delete_cots")
    # delete nothing
    collection_delete.keep_generated_cots(["1", "2"])
    # check if they are the same
    assert collection.to_json() == collection_delete.to_json()

    ## keep only author "2"
    collection_1 = get_test_collection("test_1_delete_cots_1")
    collection_delete = get_test_collection("test_1_delete_cots")
    # delete author "1"
    collection_delete.keep_generated_cots(["2"])
    assert collection_1.to_json() == collection_delete.to_json()

    ## delete all authors
    collection_all = get_test_collection("test_1_delete_cots_all")
    collection_delete = get_test_collection("test_1_delete_cots")
    # delete all
    collection_delete.keep_generated_cots()
    assert collection_all.to_json() == collection_delete.to_json()


# def test_multiple_choice_formatting() -> None:

# def test_check_templates() -> None:
