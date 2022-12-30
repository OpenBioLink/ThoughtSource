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
    sent1 = "Therefore, among A through E, the answer is D."
    answ1 = "D"
    sent2 = "Therefore, among A through E, the answer is E."
    answ2 = "E"
    sent3 = "So the answer is (b)."
    answ3 = "B"
    assert clean(type_, sent1) == answ1
    assert clean(type_, sent2) == answ2
    assert clean(type_, sent3) == answ3