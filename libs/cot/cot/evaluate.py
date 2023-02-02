import json
import re
import string
import warnings
from ast import literal_eval
from collections import defaultdict
from pprint import pprint

import datasets as ds


def search_regex(s: str, patterns: list, warn: bool) -> str:
    """Searches a string for a list of regex patterns and returns the first found match."""
    # strip the string from whitespaces
    s = s.strip()
    for pattern in patterns:
        # Compile the regular expression
        regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        # Search the string for the regex pattern
        match = regex.search(s)
        if match:
            # If more than one group is defined in the regex, print a warning return the last group
            if len(match.groups()) > 1 and warn:
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


def escape_special_characters(string):
    result = r""
    # everything but | because it is used in the regex
    special_characters = r"\^$.?*+()["
    for c in string:
        if c in special_characters:
            result += "\\"
        result += c
    return result


def is_correct(type_: str, pred: str, gold: str, choices=None, warn=False) -> bool:
    """Compares prediction with gold answer."""

    if type_ not in ["bool", "multiplechoice"]:
        warnings.warn(f"Answer type {type_} not supported yet.")
        return None

    if type_ == "multiplechoice":
        # E.g.: "Therefore, among A through E, the answer is (c)"

        # make dict of choices with uppercase letters A,B,C,...
        choices_dict = dict(zip(string.ascii_uppercase, choices))
        choices_keys = list(choices_dict.keys())
        choices_values_raw = list(choices_dict.values())

        # if values have special characters, we need to escape them
        choices_values = [re.escape(item) for item in choices_values_raw]

        # false positive in rare cases: choices_dict is {'A': 'B', 'B': 'D' ...} and gold is 'B'
        # Check if there are any common elements between the keys and values:
        keys_lower = [i.lower() for i in choices_dict.keys()]
        values_lower = [j.lower() for j in choices_dict.values()]
        common_elements = set(keys_lower).intersection(values_lower)

        # warning if common elements, except if the keys and values are the same, e.g. {'A': 'A', 'B': 'B', 'C': 'C', ...}
        if common_elements and not keys_lower == values_lower:
            if warn:
                warnings.warn(
                    f"Choices: {choices_dict} contain common elements: {common_elements}. This might lead to false positives."
                )

    if type_ == "bool":
        # E.g.: "Therefore, the answer (Yes or No) is NO."
        choices_dict = {"Yes": "True", "No": "False"}
        choices_keys = list(choices_dict.keys())
        choices_values = list(choices_dict.values())
        choices_values_raw = (
            choices_values  # in bool case, we need the raw values for the quick check
        )
        keys_lower = [i.lower() for i in choices_dict.keys()]
        values_lower = [j.lower() for j in choices_dict.values()]

    # quick check if pred is in choices_dict
    if (
        # We need to take the raw values here, as this is not regex
        pred in choices_values_raw
        or pred in choices_keys
        or pred in keys_lower
        or pred in values_lower
    ):
        # raise ValueError("not in choices_dict")
        is_correct = compare_pred_with_gold(pred, gold, choices_dict)

        return is_correct

    # check if only one of the choices are part of the pred and report this as answer
    # therefor search choice_value in pred and return if only one hit
    hits = []
    for value in choices_values:
        # only check if length of value is smaller or same than pred
        if len(value) <= len(pred):
            # make value a group for regex
            match = search_regex(
                # "(" +  escape_special_characters(value) + ")", [escape_special_characters(pred)], warn
                escape_special_characters(pred),
                ["(" + value + ")"],
                warn,
            )
            if match:
                hits.append(match)
            if len(hits) == 1:
                pred = hits[0]
                is_correct = compare_pred_with_gold(pred, gold, choices_dict)
                return is_correct

    # if pred is not in choices_dict, we need to use regex

    # uppercase and lowercase is not important, as we will match the pattern case insensitive.
    expected_answer = r"|".join(choices_values + choices_keys)

    # Matches A or A. or (A). or {A}. or [A]. to  just "A".
    # Matches word or word. or (word). or {word}. or [word]. to  just "word".
    expected_answer_location = r"[\(\{\[\'\"]?(" + expected_answer + r")[\)\}\]\'\"]?"

    # match only answer directly or the index of the answer in the choices_dict without sentence
    only_answer_sequence = r"^\s?" + expected_answer_location + r"\.?\s?$"

    # If the answer is at the end of the sentence. e.g. "The answer is A."
    # At the moment does NOT match "isA" or "answerA" to A. As this leads to false positives...
    starting_sequence = (
        # e.g. '..., the answer is A, apple.' # answer A is apple
        # answer
        r"answer:?"
        +
        # is or most likely or probably
        r"(?: \(Yes or No\))?(?: is)?:?(?: most likely)?(?: probably)?\s?"
        +
        # capturing group "answer" or "answer as string" # possibly inside brackets, etc
        expected_answer_location
        +
        # , the
        r"(?:,)?(?: the)?\s?(?:"
        +
        # non-capturing group "answer_as_string" # optional
        expected_answer
        + r")?"
        +
        # . end of sentence
        r"\.?\s?$"
    )

    # If the answer is at the beginning of the sentence. e.g. "A is the answer"
    ending_sequence = (
        r"^\s?"
        + expected_answer_location
        + r"(?: is)?(?: the)?(?: correct| right| true)?(?: answer)?\.?"
        + r"\.?\s?$"
    )

    # individual sequences at the moment only for multiplechoice
    if type_ == "multiplechoice":
        # the following part of the individual sequences needs some simplification....
        expected_answer_raw_as_group = (
            r"(" + r"|".join(choices_values_raw + choices_keys) + r")"
        )

        individual_sequences = [
            # insert your individual answer sequences here
            # replace both places of the answer (A,B,C,...) and the full text answer with the expected_answer_raw
            # the rest part of the sequence put with raw strings in between.
            # e.g. for "The answer is: A) answer_as_text."
            # rewrite as: r"The answer is: " + expected_answer_raw + r") " + expected_answer_raw + r"."
            # expected_answer_raw + re.escape(r") ") + expected_answer_raw + re.escape(r".")
            expected_answer_raw_as_group
            + escape_special_characters(") ")
            + expected_answer_raw_as_group
            + escape_special_characters(".")
        ]
        # idea to generalize the individual sequences:
        # make file in codebase where people can add their individual sequences
        # let people replace the answer with the placeholder "expected_answer_location"
        # then read the file
        # split the sentences by the placeholder
        # for every split that is not the placeholder, apply the escape_special_characters function
        # then join the sentences again with the placeholder in between

        # make individual sequences have start and end of sentence
        individual_sequences = [
            r"^" + sequence + r"$" for sequence in individual_sequences
        ]

    if type_ == "multiplechoice":
        sequences_for_search = (
            [only_answer_sequence]
            + individual_sequences
            + [starting_sequence, ending_sequence]
        )
    else:
        sequences_for_search = [
            only_answer_sequence,
            starting_sequence,
            ending_sequence,
        ]

    # search for the sequences in the prediction
    pred_match = search_regex(pred, sequences_for_search, warn=warn)

    # if not one specific value is found, search if multiple are found and return the first one
    if pred_match == "":
        # select the string after the last word "answer"
        if "answer" in pred:
            str_after_word = pred.rsplit("answer", 1)[1]
            if str_after_word:
                if type_ == "bool":
                    # remove "(Yes or No)" from the string
                    str_after_word = str_after_word.replace("(Yes or No)", "")
                    multiple_findings = (
                        r"[\s|\,|\.|\:]" + expected_answer_location + r"[\s|\,|\.]"
                    )

                if type_ == "multiplechoice":
                    multiple_findings = " " + expected_answer_location + r"[\s|\,|\.]"

                pred_match = search_regex(
                    str_after_word, [multiple_findings], warn=warn
                )

    if pred_match == "" and warn:
        warnings.warn(
            f"""Your answer could not be extracted from this sequence.
            sequence: {pred}
            possible answers: {choices_dict}"""
        )

    # match for: "Answer is A" and for "Answer is 'word'", using keys and values of choices_dict
    is_correct = compare_pred_with_gold(pred_match, gold, choices_dict)

    return is_correct


def compare_pred_with_gold(pred: str, gold: str, choices_dict: dict) -> bool:
    """Compares the predicted answer with all the gold answer. It matches with the key (e.g: 'A')
     and the value, which is a word (e.g. "apple") of the dictionary of multiple choice answers.
    Returns True if prediction is equal to gold, False otherwise."""
    for key, value in choices_dict.items():
        if gold.lower() == key.lower() or gold.lower() == value.lower():
            gold_key = key
            gold_value = value

    if not gold_key or not gold_value:
        raise ValueError(
            f"""f"Thats weird, gold-answer '{gold}' not found in choices '{choices_dict}'"
            Evaluation is not possible.
            """
        )

    comparison = pred.lower() == gold_key.lower() or pred.lower() == gold_value.lower()

    return comparison


def evaluate_sample(example, type_, overwrite, warn):
    assert (
        type_ == example["type"]
    ), "Datasets contains examples with multiple different types"

    # only run evaluation if answer is given
    if example["answer"][0] == None:
        if warn:
            warnings.warn(
                f"""
                The right answer is not given in the given example.
                No evaluation is possible for this example.
                {example}"""
            )
        return example

    # take full text answer if not multiple choice
    dataset_correct_answer = example["answer"][0]
    dataset_choices = example["choices"]

    # if no choices are given, set to None
    if dataset_choices == []:
        dataset_choices = None

    for cot in example["generated_cot"]:
        for answer in cot["answers"]:
            if answer["correct_answer"] is not None and not overwrite:
                continue
            prediction = answer["answer"]
            if is_correct(
                type_, prediction, dataset_correct_answer, dataset_choices, warn
            ):
                answer["correct_answer"] = True
            else:
                answer["correct_answer"] = False
    return example


def evaluate(dataset, overwrite=False, warn=True, config=None):  # config can be deleted
    assert isinstance(
        dataset, ds.arrow_dataset.Dataset
    ), "dataset must be an arrow dataset"

    # get dataset type, e.g. multiplechoice
    type_ = dataset[0]["type"]

    # evaluate each sample
    dataset = dataset.map(
        evaluate_sample,
        fn_kwargs={"type_": type_, "overwrite": overwrite, "warn": warn},
        features=dataset.info.features,
        # deleting the cache is necessary in generate if you call it multiple times
        # not clear if it is needed here, but it doesn't hurt
        load_from_cache_file = False,
    )

    keys = set()
    model_names = set()
    predictions = defaultdict(lambda: defaultdict(int))
    counter = defaultdict(lambda: defaultdict(int))
    evaluations = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for example in dataset:
        for cot in example["generated_cot"]:
            # if no gold answer is given, skip this example
            if example["answer"][0] == None:
                continue
            for answer in cot["answers"]:
                # when model is a dict, e.g. {'name': 'google/flan-t5-xl', 'temperature': 0, 'max_tokens': 512}
                if "{" in cot["model"]:
                    # extract model name from dict (which has to be read from a string)
                    model = literal_eval(cot["model"])
                    model_name = model["name"]
                else:
                    # when model is a string, e.g. "text_davinci_002", happens at preloaded generated cots (e.g. from lievin)
                    model_name = cot["model"]
                model_names.add(model_name)
                # make a key for each combination of triggers, e.g. "None_lievin-02_kojima-A-C"
                key = f"{cot['instruction']}_{cot['cot_trigger']}_{answer['answer_extraction']}"
                keys.add(key)
                counter[model_name][key] += 1
                if answer["correct_answer"]:
                    predictions[model_name][key] += 1

    if warn:
        for count in counter.values():
            if count != len(dataset):
                warnings.warn(
                    f"""It seems that not all examples of the dataset include an answer to be evaluated.
                    Counter of examples:
                    {counter.items()}
                    Length of dataset:
                    {len(dataset)}
                    The evaluation score was only calculated based on the examples that include an answer.
                    To turn this warning off, set warn=False in the evaluate function."""
                )

    keys = sorted(keys)
    for model_name in model_names:
        for key in keys:
            for metric in ["accuracy"]:
                if counter[model_name][key] != 0:
                    value = predictions[model_name][key] / counter[model_name][key]
                    evaluations[metric][model_name][key] = round(value, 6)

    # pprint(dict(evaluations))
    return dataset, json.loads(json.dumps(evaluations))
