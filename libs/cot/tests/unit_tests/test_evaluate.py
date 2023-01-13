from typing import Iterator

import datasets
import pytest
from cot import Collection

# import os
# from pathlib import Path
from cot.generate import Correct_output

from cot.evaluate import clean, is_correct, search_regex, answer_to_multiplechoice

from .utils import chdir, get_test_collection, simple_config


def test_clean():
    type_ = "multiplechoice"
    number_of_choices = 7

    assert clean(type_, "E", number_of_choices) == "E"
    assert clean(type_, "E.", number_of_choices) == "E"
    assert clean(type_, "E ", number_of_choices) == "E"
    assert clean(type_, "(E)", number_of_choices) == "E"
    assert clean(type_, "[E]", number_of_choices) == "E"

    assert clean(type_, "So the answer is B", number_of_choices) == "B"
    assert clean(type_, "So the answer is B.", number_of_choices) == "B"
    # assert clean(type_, "So the answer isB", number_of_choices) == "B"
    assert clean(type_, "Therefore, the answer is B", number_of_choices) == "B"
    assert clean(type_, "The answer is B", number_of_choices) == "B"
    assert clean(type_, "Answer is B", number_of_choices) == "B"
    assert clean(type_, "Answer B", number_of_choices) == "B"
    assert clean(type_, "The correct answer is B", number_of_choices) == "B"
    assert clean(type_, "The correct answer B", number_of_choices) == "B"
    assert clean(type_, "Correct answer is B", number_of_choices) == "B"
    assert clean(type_, "Correct answer B", number_of_choices) == "B"
    assert clean(type_, "Among A through F, the answer is B", number_of_choices) == "B"
    assert (
        clean(type_, "Among A through F, the correct answer is B", number_of_choices)
        == "B"
    )
    assert (
        clean(type_, "Therefore, among A through F, the answer is B", number_of_choices)
        == "B"
    )

    assert clean(type_, "B is the answer.", number_of_choices) == "B"
    assert clean(type_, "B is the answer", number_of_choices) == "B"
    assert clean(type_, "B is the correct answer", number_of_choices) == "B"
    assert clean(type_, "B is the correct answer.", number_of_choices) == "B"
    assert clean(type_, "B is the right answer", number_of_choices) == "B"
    assert clean(type_, "B is the right answer.", number_of_choices) == "B"

    assert clean(type_, "B is correct", number_of_choices) == "B"
    assert clean(type_, "B is correct.", number_of_choices) == "B"
    assert clean(type_, "B is right", number_of_choices) == "B"
    assert clean(type_, "B is right.", number_of_choices) == "B"


def test_clean_and_is_correct():
    # test upper and lower case

    type_ = "multiplechoice"
    number_of_choices = 7

    pred = clean(type_, r"{e}", number_of_choices)
    assert is_correct(type_, pred, "E")

    pred = clean(type_, "e", number_of_choices)
    assert is_correct(type_, pred, "E")

    pred = clean(type_, "So the answer is (b)", number_of_choices)
    assert is_correct(type_, pred, "B")

    pred = clean(type_, "b is the answer", number_of_choices)
    assert is_correct(type_, pred, "B")

    pred = clean(type_, "(b) is the answer", number_of_choices)
    assert is_correct(type_, pred, "B")

    pred = clean(type_, "So the answer is b", number_of_choices)
    assert is_correct(type_, pred, "B")

    # pred = clean(type_, "So the answer isb", number_of_choices)
    # assert is_correct(type_, pred, "B")

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
