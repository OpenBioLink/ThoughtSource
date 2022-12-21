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


def is_correct(type_, pred, gold):
    if type_ == "multiplechoice":
        return pred == gold


def answer_to_multiplechoice(answer, choices):
    for ix, choice in enumerate(choices):
        if choice == answer:
            return chr(65 + ix)
    raise ValueError("Thats weird, gold-answer not found in choices")


def evaluate_sample(example, type_):
    assert type_ == example["type"], "Datasets contains examples with multiple different types"

    gold_answer = example["answer"][0]

    if type_ == "multiplechoice":
        gold_answer = answer_to_multiplechoice(gold_answer, example["choices"])

    for cot in example["generated_cot"]:
        for answer in cot["answers"]:
            answer_str = answer["answer"]
            answer_str = clean(type_, answer_str)
            if is_correct(type_, answer_str, gold_answer):
                answer["correct_answer"] = True
            else:
                answer["correct_answer"] = False
    return example


def evaluate(dataset, config=None):

    # implemented for single dataset right now collection["worldtree"]["train"]
    # TODO implement for ds.dataset_dict.DatasetDict collection["worldtree"]
    assert isinstance(
        dataset, ds.arrow_dataset.Dataset
    ), "Only implemented for single datasets right now e.g. collection['worldtree']['train']"

    # support only one type per dataset
    # TODO support datasets contining different example types (mulichoice, number, ...), if needed?
    type_ = dataset[0]["type"]

    dataset = dataset.map(evaluate_sample, fn_kwargs={"type_": type_}, features=dataset.info.features)

    keys = set()
    predictions = defaultdict(int)
    for example in dataset:
        for cot in example["generated_cot"]:
            for answer in cot["answers"]:
                key = f"{cot['instruction']}_{cot['cot_trigger']}_{answer['answer_extraction']}"
                keys.add(key)
                if answer["correct_answer"]:
                    predictions[key] += 1

    evaluations = defaultdict(dict)
    for key in keys:
        for metric in ["accuracy"]:
            evaluations[metric][key] = predictions[key] / len(dataset)
    print(dict(evaluations))
    return dataset
