import re
from collections import defaultdict

import datasets as ds


# ver 0.2
def clean(type_, pred, num_choices):
    if type_ == "multiplechoice":
        # TODO: BIG change necessary. Important to correct, add free text answers
        # Will make big mistakes, the code is I think from another paper
        choices = [chr(65 + i) for i in range(num_choices)]

        # multiple choice
        "Therefore, among A through E, the answer is (C)"
        "So the answer is (a)"
        "Therefore, among A through E, the answer is A."

        # boolean
        "Therefore, the answer (Yes or No) is No."
        "So the answer is yes."
        "Therefore, the answer (Yes or No) is NO."
        

        possible_answers_sequences = [
            r"So the answer is",
            r"Therefore, the answer is",
            r"The answer is",
            r"Answer is",
            r"Answer",
            r"The correct answer is",
            r"The correct answer",
            r"Correct answer is",
            r"Correct answer",
            r"Among . through ., the answer is",
            r"Among . through ., the correct answer is",]
        
        for seq in possible_answers_sequences:
            pred = re.sub(seq, "", pred)

        pred_found = re.findall(r'|'.join(choices), pred)
        if len(pred_found):
            # if multiple choices are found, take the last one
            pred_selected = pred_found[-1]
            # return warning if multiple choices are found
            import warnings
            warnings.warn(f"Multiple choices found in prediction: {pred_found}")
        else:
            pred_selected = ""
    else:
        raise ValueError("type is not supported ...")

    return pred_selected


def is_correct(type_, pred, gold):
    if type_ == "multiplechoice":
        return pred == gold


def answer_to_multiplechoice(answer, choices):
    num_choices = len(choices)
    for ix, choice in enumerate(choices):
        if choice == answer:
            return (num_choices, chr(65 + ix))
    raise ValueError("Thats weird, gold-answer not found in choices")


def evaluate_sample(example, type_):
    assert type_ == example["type"], "Datasets contains examples with multiple different types"

    gold_answer = example["answer"][0]

    if type_ == "multiplechoice":
        num_choices, gold_answer = answer_to_multiplechoice(gold_answer, example["choices"])

    for cot in example["generated_cot"]:
        for answer in cot["answers"]:
            answer_str = answer["answer"]
            answer_str_cleaned = clean(type_, answer_str, num_choices)
            if is_correct(type_, answer_str_cleaned, gold_answer):
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
