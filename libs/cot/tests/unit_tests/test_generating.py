import pytest
import datasets
# from contextlib import contextmanager
from typing import Iterator
from cot import Collection
# import os
# from pathlib import Path
from cot.generate import Correct_output

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

# def test_multiple_choice_formatting() -> None:

# def test_check_templates() -> None:
    



