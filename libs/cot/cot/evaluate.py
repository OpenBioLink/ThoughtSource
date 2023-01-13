import re
from collections import defaultdict
from pprint import pprint

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
def clean(type_: str, pred: str, num_choices: int) -> str:
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
        expected_answer_location = r"\s?[\(\{\[]?(" + expected_answer + r")[\)\}\]]?\s?"

        # match only single answer without sentence
        only_answer_sequence = r"^" + expected_answer_location + r"$"

        # If the answer is at the end of the sentence. e.g. "The answer is A."
        # At the moment does NOT match "isA" or "answerA" to A. As this leads to false positives...
        starting_sequence = (
            r"answer:?(?: is)?(?: most likely)?\s" + expected_answer_location
        )

        # If the answer is at the beginning of the sentence. e.g. "A is the answer"
        ending_sequence = (
            expected_answer_location
            + r"\s?(?: is)?(?: the)?(?: correct| right| true)?(?: answer)?\.?"
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
                f"""Your answer could not be extracted, please add your sequence to the list of personalized answers sequences.
                sequence: {pred}
                In the file: libs/cot/cot/evaluate.py under the function clean()"""
            )

    else:
        raise ValueError("type is not supported ...")

    return pred_match


def is_correct(type_, pred, gold):
    if type_ == "multiplechoice":
        return pred.lower() == gold.lower()


def answer_to_multiplechoice(answer, choices, warn):
    # assert type(answer) == str, "answer must be a string"
    # assert type(choices) == list, "choices must be a list"
    # assert type(choices[0]) == str, "choices must be a list of strings"
    num_choices = len(choices)
    for ix, choice in enumerate(choices):
        if choice.lower() == answer.lower():
            return (num_choices, chr(65 + ix))
        # for which are numbers to correct for float/int differences
        # choice_float = None
        # choice_answer = None
        # try: choice_float = float(choice)
        # except: pass
        # try: choice_answer = float(answer)
        # except: pass
        # if choice_float and choice_answer and choice_float == choice_answer:
        #     return (num_choices, chr(65 + ix))

        if answer == None and warn:
            import warnings
            warnings.warn(
                f"""The right answer is not given in the given example.
                This can be intentionally, but running an evaluation is not possible.
                To turn off warnings, set warn=False in the evaluate() function.
                """
            )
        if answer == None and not warn:
            return (num_choices, None)
    if answer != None:
        raise ValueError(
            f"""f"Thats weird, gold-answer '{answer}' not found in choices '{choices}'"
            Evaluation is not possible.
            """
        )


def evaluate_sample(example, type_, overwrite, warn):
    assert (
        type_ == example["type"]
    ), "Datasets contains examples with multiple different types"

    # take full text answer if not multiple choice
    gold_answer = example["answer"][0]

    if type_ == "multiplechoice":
        num_choices, gold_answer = answer_to_multiplechoice(
            gold_answer, example["choices"], warn
        )
        if gold_answer == None:
            return example

    for cot in example["generated_cot"]:
        for answer in cot["answers"]:
            if answer["correct_answer"] is not None and not overwrite:
                continue
            answer_str = answer["answer"]
            answer_str_cleaned = clean(type_, answer_str, num_choices)
            if is_correct(type_, answer_str_cleaned, gold_answer):
                answer["correct_answer"] = True
            else:
                answer["correct_answer"] = False
    return example


def evaluate(dataset, overwrite=False, warn=True, config=None):

    # implemented for single dataset right now collection["worldtree"]["train"]
    # TODO implement for ds.dataset_dict.DatasetDict collection["worldtree"]
    assert isinstance(
        dataset, ds.arrow_dataset.Dataset
    ), "Only implemented for single datasets right now e.g. collection['worldtree']['train']"

    # support only one type per dataset
    # TODO support datasets contining different example types (mulichoice, number, ...), if needed?
    type_ = dataset[0]["type"]

    dataset = dataset.map(
        evaluate_sample, fn_kwargs={"type_": type_, "overwrite": overwrite, "warn": warn}, features=dataset.info.features
    )

    keys = set()
    predictions = defaultdict(int)
    counter = defaultdict(int)

    for example in dataset:
        for cot in example["generated_cot"]:
            for answer in cot["answers"]:
                # make a key for each combination of triggers, e.g. "None_lievin-02_kojima-A-C"
                key = f"{cot['instruction']}_{cot['cot_trigger']}_{answer['answer_extraction']}"
                keys.add(key)
                counter[key] += 1
                if answer["correct_answer"]:
                    predictions[key] += 1

    evaluations = defaultdict(dict)

    if warn:
        for count in counter.values():
            if count != len(dataset):
                    import warnings

                    warnings.warn(
                        f"""It seems that not all examples of the dataset include an answer to be evaluated.
                    Counter of examples:
                    {counter.items()}
                    Length of dataset:
                    {len(dataset)}
                    The evaluation score was only calculated based on the examples that include an answer.
                    To turn this warning off, set warn=False in the evaluate function."""
                    )

    for key in keys:
        for metric in ["accuracy"]:
            evaluations[metric][key] = predictions[key] / counter[key]

    pprint(dict(evaluations))
    return dataset
