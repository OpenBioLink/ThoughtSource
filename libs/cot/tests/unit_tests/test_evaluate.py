from typing import Iterator

import datasets
import pytest
from cot import Collection

# import os
# from pathlib import Path
from cot.generate import Correct_output

from cot.evaluate import clean

from .utils import chdir, get_test_collection, simple_config

def test_clean():
    type_ = "multiplechoice"

    assert clean(type_, "E", 7) == "E"
    assert clean(type_, "e", 7) == "E"
    assert clean(type_, "E.", 7) == "E"
    assert clean(type_, "E ", 7) == "E"
    assert clean(type_, "(E)", 7) == "E"
    assert clean(type_, "[E]", 7) == "E"
    assert clean(type_, r"{e}", 7) == "E"

    assert clean(type_, "So the answer is (b)", 7) == "B"
    assert clean(type_, "So the answer is B", 7) == "B"
    assert clean(type_, "So the answer is B.", 7) == "B"
    assert clean(type_, "So the answer is b", 7) == "B"
    assert clean(type_, "So the answer isB", 7) == "B"
    assert clean(type_, "So the answer isb", 7) == "B"
    assert clean(type_, "Therefore, the answer is B", 7) == "B"
    assert clean(type_, "The answer is B", 7) == "B"
    assert clean(type_, "Answer is B", 7) == "B"
    assert clean(type_, "Answer B", 7) == "B"
    assert clean(type_, "The correct answer is B", 7) == "B"
    assert clean(type_, "The correct answer B", 7) == "B"
    assert clean(type_, "Correct answer is B", 7) == "B"
    assert clean(type_, "Correct answer B", 7) == "B"
    assert clean(type_, "Among A through F, the answer is B", 7) == "B"
    assert clean(type_, "Among A through F, the correct answer is B", 7) == "B"
    assert clean(type_, "Therefore, among A through F, the answer is B", 7) == "B"

    assert clean(type_, "B is the answer.", 7) == "B"
    assert clean(type_, "b is the answer", 7) == "B"
    assert clean(type_, "(b) is the answer", 7) == "B"
    assert clean(type_, "B is the answer", 7) == "B"
    assert clean(type_, "B is the correct answer", 7) == "B"
    assert clean(type_, "B is the correct answer.", 7) == "B"
    assert clean(type_, "B is the right answer", 7) == "B"
    assert clean(type_, "B is the right answer.", 7) == "B"

    assert clean(type_, "B is correct", 7) == "B"
    assert clean(type_, "B is correct.", 7) == "B"
    assert clean(type_, "B is right", 7) == "B"
    assert clean(type_, "B is right.", 7) == "B"