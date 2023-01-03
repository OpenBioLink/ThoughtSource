import re
from collections import defaultdict

import datasets as ds


def search_regex(s: str, patterns: list) -> str:
    for pattern in patterns:
        # Compile the regular expression
        regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        # Search the string for the regex pattern
        match = regex.search(s)
        if match:
            # If more than one group is defined in the regex, print a warning return the last group
            if len(match.groups()) > 1:
                import warnings

                warnings.warn(
                    f"""Found more than one possible answer to compute the evaluation score. By default returning the first found answer.
                                 In the answer sentence '{s}' these possible answers were found: '{match.groups()}'
                                 If you want to return a specific answer, please define a regex pattern with only one group.
                """
                )

            # If the regex pattern is found, return the group you defined in the regex
            return match.group(1)
    # If none of the regex patterns are found, return an empty string
    return ""


# ver 0.2
def clean(type_, pred, num_choices):
    """Cleans the prediction string to be able to compare it to the gold answer."""
    # TODO: Add boolean type answers
    # "Therefore, the answer (Yes or No) is NO."

    if type_ == "multiplechoice":
        # "Therefore, among A through E, the answer is (c)"

        # TODO: If models report multiple possible answers, this is not covered yet.
        # e.g. from gpt-3: "Therefore, among A through E, the answer is (A) or (B)."
        # here the code will just select A without a warning.

        # define choices A,B,C,...
        choices = [chr(65 + i) for i in range(num_choices)]

        # options of expected answers, e.g. A,B,C, or a,b,c or Yes,No ...
        expected_answer = r"|".join(
            choices
        )  # join([c.upper() for c in choices]) + r'|'.join([c.lower() for c in choices])

        # match pattern.
        # Matches A or A. or (A). or {A}. or [A]. to  just "A".
        expected_answer_location = (
            r"\s?[\(\{\[]?(" + expected_answer + r")[\)\}\]]?\.?\s?"
        )

        # match only single answer without sentence
        only_answer_sequence = r"^" + expected_answer_location + r"$"

        # If the answer is at the end of the sentence. e.g. "The answer is A."
        # also matches "isA" or "answerA" to A. To correct if the whitespace is not given
        starting_sequence = (
            r"answer:?\s(?:is)?(?:\smost\slikely)?\s?" + expected_answer_location
        )

        # If the answer is at the beginning of the sentence. e.g. "A is the answer"
        ending_sequence = (
            expected_answer_location
            + r"\s?(?:is)?\s?(?:the)?\s?(?:correct|right|true)?\s?(?:answer)?\.?"
        )

        # TODO: personalized sequences
        # please add your sequences here, if it is not covered with the regex above
        # possible_answers_sequences = [
        #     #e.g.
        #     "So the answer is " + expected_answer_location,
        #     ]

        pred_match = search_regex(
            pred, [only_answer_sequence, starting_sequence, ending_sequence]
        )  # + possible_answers_sequences)

        if pred_match == "":
            import warnings

            warnings.warn(
                """Your answer could not be extracted, please add your sequence to the list of personalized answers sequences.
                sequence: {pred}
                In the file: libs/cot/cot/evaluate.py under the function clean()"""
            )

    else:
        raise ValueError("type is not supported ...")

    return pred_match


def is_correct(type_, pred, gold):
    if type_ == "multiplechoice":
        return pred.lower() == gold.lower()


def answer_to_multiplechoice(answer, choices):
    num_choices = len(choices)
    for ix, choice in enumerate(choices):
        if choice == answer:
            return (num_choices, chr(65 + ix))
    raise ValueError("Thats weird, gold-answer not found in choices")


def evaluate_sample(example, type_):
    assert (
        type_ == example["type"]
    ), "Datasets contains examples with multiple different types"

    # take full text answer if not multiple choice
    gold_answer = example["answer"][0]

    if type_ == "multiplechoice":
        num_choices, gold_answer = answer_to_multiplechoice(
            gold_answer, example["choices"]
        )

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

    dataset = dataset.map(
        evaluate_sample, fn_kwargs={"type_": type_}, features=dataset.info.features
    )

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
