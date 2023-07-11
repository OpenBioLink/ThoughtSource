from cot import Collection
from cot.evaluate import is_correct


def test_is_correct_multiplechoice():
    type_ = "multiplechoice"
    choices = ["1", "2", "3", "4", "5", "6", "7"]
    # choices = ["A","B","C","D","E","F","G"]

    gold = "E"
    for pred in ["E", "E.", "(E)", "[E]", r"{E}", "E)"]:
        assert is_correct(type_, pred, gold, choices)

    gold = "5"
    for pred in ["E", "E.", "(E)", "[E]", r"{E}", "E)"]:
        assert is_correct(type_, pred, gold, choices)

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
        # "Among A through F, the answer is B",
        # "Among A through F, the correct answer is B",
        "Therefore, among A through F, the answer is B",
        # "Therefore, among A through F, the correct answer is B",
    ]:
        assert is_correct(type_, pred, gold, choices)

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
        assert is_correct(type_, pred, gold, choices)

    gold = "B"
    for pred in [
        "So the answer is (b)",
        "b is the answer",
        "(b) is the answer",
        "So the answer is b",
    ]:
        assert is_correct(type_, pred, gold, choices)

    choices = ["sadness", "anxiety", "inspiration", "discomfort", "insights"]
    pred = "Therefore, among A through E, the answer is most likely C, inspiration."
    gold = "C"
    assert is_correct(type_, pred, gold, choices)

    choices = ["sadness", "anxiety", "inspiration", "discomfort", "insights"]
    pred = "Therefore, among A through E, the answer is most likely (C), inspiration."
    gold = "C"
    assert is_correct(type_, pred, gold, choices)

    choices = ["facade", "front door", "doorway", "entrance porch", "hallway"]
    pred = "Therefore, among A through E, the answer is (B) front door."
    gold = "B"
    assert is_correct(type_, pred, gold, choices)

    choices = ["midwest", "countryside", "estate", "farming areas", "illinois"]
    pred = "Therefore, among A through E, the answer is A, the midwest."
    gold = "A"
    assert is_correct(type_, pred, gold, choices)


def test_is_correct_bool():
    type_ = "bool"

    pred = "Therefore, the answer (Yes or No) is: No"
    gold = "False"
    assert is_correct(type_, pred, gold)

    pred = "Therefore, the answer (Yes or No) is No."
    gold = "No"
    assert is_correct(type_, pred, gold)

    pred = "Therefore, the answer (Yes or No) is No."
    gold = "False"
    assert is_correct(type_, pred, gold)

    pred = "Therefore, the answer (Yes or No) is Yes."
    gold = "Yes"
    assert is_correct(type_, pred, gold)

    pred = "Therefore, the answer (Yes or No) is Yes."
    gold = "True"
    assert is_correct(type_, pred, gold)

    # pred = "Therefore, the answer (Yes or No) is uncertain."
    # gold = "True"
    # assert not is_correct(type_, pred, gold)


# def test_is_correct_multiple_answers():
#     type_ = "multiplechoice"
#     # if multiple answers are given take the first one (can be changed of course)

#     choices = ["1", "2", "3", "4", "5", "6", "7"]

#     pred = "So the answer is (a), (b), or (e)."
#     gold = "A"
#     assert not is_correct(type_, pred, gold, choices)

#     pred = "Therefore, among A through E, the answer is A, B, C, or D."
#     gold = "A"
#     assert not is_correct(type_, pred, gold, choices)

#     pred = "Therefore, among A through E, the answer is most likely B, D, or E."
#     gold = "B"
#     assert not is_correct(type_, pred, gold, choices)

#     pred = "Therefore, among A through E, the answer is probably A or C."
#     gold = "A"
#     assert not is_correct(type_, pred, gold, choices)

#     pred = "Therefore, among A through E, the answer is most likely C, airport, but it could also be A, car."
#     gold = "C"
#     assert not is_correct(type_, pred, gold, choices)

#     pred = "Therefore, among A through E, the answer is probably (A), (B), (C), or (D)."
#     gold = "A"
#     assert not is_correct(type_, pred, gold, choices)


def test_predefined_correct_value():
    # med_qa
    # collection = Collection(["med_qa"], verbose=False)
    # collection = collection.select(
    #     split="test", number_samples=10, random_samples=False
    # )

    # collection2 = Collection(["med_qa"], verbose=False)
    # collection2 = collection2.select(
    #     split="test", number_samples=10, random_samples=False
    # )

    # # only do evaluation on one of them, nothing should change
    # collection.evaluate(warn=False)

    # collection_json = collection.to_json()
    # collection2_json = collection2.to_json()

    # assert collection_json == collection2_json

    # pubmed_qa
    collection = Collection(["pubmed_qa"], verbose=False)
    collection = collection.select(split="train", number_samples=10, random_samples=False)
    # collection2 = Collection(["pubmed_qa"], verbose=False)
    # collection2 = collection2.select(
    #     split="train", number_samples=10, random_samples=False
    # )

    collection_json = collection.to_json()

    # only do evaluation on one of them, nothing should change
    collection.evaluate(overwrite=False)

    collection2_json = collection.to_json()

    assert collection_json == collection2_json
