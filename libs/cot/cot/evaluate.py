import re
from collections import defaultdict

import datasets as ds


# ver 0.2
def clean(type_, pred):
    if type_ == "multiplechoice":
        pred = re.findall(r"A|B|C|D", pred)
        if len(pred):
            pred = pred[0]
        else:
            pred = ""
    else:
        raise ValueError("type is not supported ...")

    return pred


def evaluate_example(type_, pred, gold):
    if type_ == "multiplechoice":
        return pred == gold


def answer_to_multiplechoice(answer, choices):
    for ix, choice in enumerate(choices):
        if choice == answer:
            return chr(65 + ix)
    raise ValueError("Thats weird, gold-answer not found in choices")


def evaluate(dataset, config=None):

    # implemented for single dataset right now collection["worldtree"]["train"]
    # TODO implement for ds.dataset_dict.DatasetDict collection["worldtree"]
    assert isinstance(
        dataset, ds.arrow_dataset.Dataset
    ), "Only implemented for single datasets right now e.g. collection['worldtree']['train']"

    keys = set()
    predictions = defaultdict(int)

    # support only one type per dataset
    # TODO support datasets contining different example types (mulichoice, number, ...), if needed?
    type_ = dataset[0]["type"]
    for example in dataset:
        assert (
            type_ == example["type"]
        ), "Datasets contains examples with multiple different types"

        gold_answer = example["answer"][0]

        if type_ == "multiplechoice":
            gold_answer = answer_to_multiplechoice(gold_answer, example["choices"])

        for cot in example["generated_cot"]:
            for answer in cot["answers"]:
                key = f"{cot['instruction']}_{cot['cot-trigger']}_{answer['answer-extraction']}"
                keys.add(key)
                answer_str = answer["answer"]
                answer_str = clean(type_, answer_str)
                if evaluate_example(type_, answer_str, gold_answer):
                    predictions[key] += 1

    evaluations = defaultdict(dict)
    for key in keys:
        for metric in ["accuracy"]:
            evaluations[metric][key] = predictions[key] / len(dataset)
    return dict(evaluations)
