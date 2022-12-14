import pytest
import datasets
from typing import Iterator
from cot import Collection
# import os
# from pathlib import Path
from cot.generate import Correct_output

from .utils import chdir, simple_config, get_test_collection

def test_correct_output()-> None:
    """Test the Correct_output class"""
    simple_dict = {'a':1, 'b':None}
    # leave number as is
    simple_sentence_a = "{a}"
    # if key is None return empty string
    simple_sentence_b = "{b}"
    # if key is missing return empty string
    simple_sentence_c = "{c}"
    correct_a = '1'
    correct_b = ''
    correct_c = ''
    answer_a = simple_sentence_a.format_map(Correct_output(simple_dict))
    answer_b = simple_sentence_b.format_map(Correct_output(simple_dict))
    answer_c = simple_sentence_c.format_map(Correct_output(simple_dict))
    assert answer_a == correct_a
    assert answer_b == correct_b
    assert answer_c == correct_c

    # without correct output mapping:
    correct_normal_b = 'None'
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

    assert collection["worldtree"]["train"][0]["generated_cot"][0]["prompt_text"] == text1
    assert collection["worldtree"]["train"][0]["generated_cot"][0]["answers"][0]["answer_extraction_text"] == text2

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


# def test_multiple_choice_formatting() -> None:

# def test_check_templates() -> None:
    



