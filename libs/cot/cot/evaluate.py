import re
from collections import defaultdict


import datasets as ds


# ver 0.2
def clean(type_, pred, num_choices):
    if type_ == "multiplechoice":
        # TODO: Add boolean answers

        # define choices A,B,C,...
        choices = [chr(65 + i) for i in range(num_choices)]

        # multiple choice examples
        # "Therefore, among A through E, the answer is (C)"
        # "So the answer is (a)"
        # "Therefore, among A through E, the answer is A."

        # boolean examples
        # "Therefore, the answer (Yes or No) is No."
        # "So the answer is yes."
        # "Therefore, the answer (Yes or No) is NO."


        
        expected_answer = r'|'.join([c.upper() for c in choices]) + r'|'.join([c.lower() for c in choices])

        # match pattern. matches A or (A) or {A} or [A] to A.
        # also matches A. or (A). or {A}. or [A]. to A. To correct if the dot at the end is not given
        # also matches "isA" or "answerA" to A. To correct if the whitespace is not given
        expected_answer_location = r"\s?[\(\{\[]?(" + expected_answer + r")[\)\}\]]?\.?"

        starting_sequence = r"[Aa]nswer:?\s(?:is)?\s?" + expected_answer_location
        ending_sequence = expected_answer_location + r"\s?(?:is)?\s?(?:the)?\s?(?:correct|right|true)?\s?(?:[Aa]nswer)?\.?"

        # personalized sequences
        # please add your sequences here, if it is not covered with the regex above
        possible_answers_sequences = [
            #e.g.
            "So the answer is " + expected_answer_location,
            ]

        if len(pred) < 6:
            match = re.search(expected_answer_location, pred, re.MULTILINE)
            pred_match = match.group(0)

        elif re.search(starting_sequence, pred, re.MULTILINE):
            match = re.search(starting_sequence, pred, re.MULTILINE)
            pred_match = match.group(1)

        elif re.search(ending_sequence, pred, re.MULTILINE):
            match = re.search(ending_sequence, pred, re.MULTILINE)
            pred_match = match.group(1)

        # elif True:
        #     for seq in possible_answers_sequences:
        #         if pred.find(seq):
        #             match = pred.find(seq)
        #             pred = match.group(0)

        else:
            import warnings
            warnings.warn(
                """Your answer could not be extracted, please add your sequence to the list of possible answers sequences.
                In the file: libs/cot/cot/evaluate.py under the function clean()""")
        
        # for seq in possible_answers_sequences:
        #     pred = re.sub(seq, "", pred)

        pred_match = pred_match.upper()

        pred_found = re.findall(r'|'.join(choices), pred_match)
        if len(pred_found) == 1:
            pred_selected = pred_found[0]
        elif len(pred_found) > 1:
            # if multiple choices are found, take the last one
            pred_selected = pred_found[-1]
            # return warning if multiple choices are found
            import warnings
            warnings.warn(f"Multiple choices found in prediction '{pred}' :\n Found: {pred_found} \n Selected by default is the last one.")
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

    # take full text answer if not multiple choice
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
