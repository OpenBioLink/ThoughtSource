from typing import Iterator

import datasets
import pytest
from cot import Collection

# import os
# from pathlib import Path
from cot.generate import Correct_output

from cot.evaluate import is_correct, search_regex, answer_to_multiplechoice

from .utils import chdir, get_test_collection, simple_config


def test_is_correct():
    type_ = "multiplechoice"
    number_of_choices = 7

    gold = "E"
    for pred in ["E", "E.", "E ", "(E)", "[E]", r"{E}", "E)"]:
        assert is_correct(type_, pred, gold, number_of_choices)

    gold = "B"
    for pred in [
        "So the answer is B",
        "So the answer is B.",
        # "So the answer isB",
        "Therefore, the answer is B",
        "The answer is B",
        "Answer is B",
        "Answer B",
        "The correct answer is B",
        "The correct answer B",
        "Correct answer is B",
        "Correct answer B",
        "Among A through F, the answer is B",
        "Among A through F, the correct answer is B",
        "Therefore, among A through F, the answer is B",
        "Therefore, among A through F, the correct answer is B",
    ]:
        assert is_correct(type_, pred, gold, number_of_choices)
    

    gold = "B"
    for pred in [
        "B is the answer.",
        "B is the answer",
        "B is the correct answer",
        "B is the correct answer.",
        "B is the right answer",
        "B is the right answer.",
        "B is correct",
        "B is correct.",
        "B is right",
        "B is right.",
    ]:
        assert is_correct(type_, pred, gold, number_of_choices)


    gold = "B"
    for pred in [
        "So the answer is (b)",
        "b is the answer",
        "(b) is the answer",
        "So the answer is b",
        # "So the answer isb",
    ]:
        assert is_correct(type_, pred, gold, number_of_choices)

# def test_search_regex():

def test_predefined_correct_value(): 
    # med_qa
    collection = Collection(["med_qa"], verbose=False)
    collection = collection.select(split="test", number_samples=10, random_samples=False)

    collection2 = Collection(["med_qa"], verbose=False)
    collection2 = collection2.select(split="test", number_samples=10, random_samples=False)

    # only do evaluation on one of them, nothing should change
    collection.evaluate(warn=False)

    collection_json = collection.to_json()
    collection2_json = collection2.to_json()

    assert collection_json == collection2_json


    # pubmed_qa
    collection = Collection(["pubmed_qa"], verbose=False)
    collection = collection.select(split="train", number_samples=10, random_samples=False)
    collection2 = Collection(["pubmed_qa"], verbose=False)
    collection2 = collection2.select(split="train", number_samples=10, random_samples=False)

    # only do evaluation on one of them, nothing should change
    collection.evaluate()

    collection_json = collection.to_json()
    collection2_json = collection2.to_json()

    assert collection_json == collection2_json


def test_answer_to_multiplechoice():
    assert answer_to_multiplechoice(answer="x", choices=["x", "y", "z"], warn=True) == (3, "A")
    assert answer_to_multiplechoice(answer="c", choices=["a", "b", "c", "d", "e"], warn=True) == (5, "C")
    assert answer_to_multiplechoice(answer="C", choices=["a", "b", "c", "d", "e"], warn=True) == (5, "C")
    assert answer_to_multiplechoice(answer="c", choices=["A", "B", "C", "D", "E"], warn=True) == (5, "C")
    assert answer_to_multiplechoice(answer="ax", choices=["Ax", "Bx", "Cx", "Dx", "Ex"], warn=True) == (5, "A")

    # assert answer_to_multiplechoice(answer=1, choices=["1", "2", "3"], warn=True) == (3, "A")
    # assert answer_to_multiplechoice(answer="1", choices=[1, 2, 3], warn=True) == (3, "A")
    # assert answer_to_multiplechoice(answer=1, choices=[1, 2, 3], warn=True) == (3, "A")
    # assert answer_to_multiplechoice(answer="1.00", choices=["1.0", "2", "3"], warn=True) == (3, "A")
