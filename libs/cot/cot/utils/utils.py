import os


def _read_file(path):
    with open(path, "r", encoding="utf8") as infile:
        content = infile.readlines()
    content = [x.strip() for x in content]
    return content


def parse_kojima_log(path, dataset):
    content = _read_file(path)

    assert content[9] == "*************************"

    iterator = iter(content[9:])

    cot_trigger = "Let's think step by step."

    if dataset == "commonsenseqa":
        answer_trigger = "Therefore, among A through E, the answer is"
    else:
        answer_trigger = "Therefore, the answer (Yes or No) is"

    def parse_elements(iterator):
        try:
            while True:
                element = {"question": "", "cot": "", "prediction": "", "correct_answer": ""}

                stars = next(iterator)
                if stars.startswith("accuracy"):
                    break
                assert stars == "*************************", "Stars begin"

                st_data = next(iterator)
                assert st_data.endswith("st data"), f"st data {st_data}"

                # skip headaches at all
                if dataset == "commonsenseqa" and st_data in ["56st data", "448st data", "709st data", "1058st data"]:
                    it_ = ""
                    while it_ != "*************************":
                        it_ = next(iterator)
                    continue
                elif dataset == "strategyqa" and st_data in [
                    "200st data",
                    "645st data",
                    "767st data",
                    "781st data",
                    "933st data",
                    "1744st data",
                    "1852st data",
                    "2116st data",
                ]:
                    it_ = ""
                    while it_ != "*************************":
                        it_ = next(iterator)
                    continue

                sampling = next(iterator)
                assert sampling == "1_th_sampling", f"_th_samp {sampling}"
                question = next(iterator)
                assert question.startswith("Q: "), f"Q {question}"
                element["question"] = question[len("Q: ") :]

                cot_multiline = ""
                cot = next(iterator)
                assert cot.startswith("A: " + cot_trigger), f"A {cot}"
                cot_multiline += cot

                next_line = next(iterator)
                while not next_line.startswith(answer_trigger):
                    cot_multiline += "\n" + next_line
                    next_line = next(iterator)

                element["cot"] = cot_multiline[len("A: " + cot_trigger + " ") :]

                answer_mulitiline = ""
                answer_mulitiline += next_line
                next_line = next(iterator)
                while not next_line.startswith("pred_before"):
                    answer_mulitiline += next_line
                    next_line = next(iterator)
                element["prediction"] = answer_mulitiline

                pred_before = next_line
                assert pred_before.startswith("pred_before :"), f"pred_before {pred_before}"

                pred_after = next(iterator)
                assert pred_after.startswith("pred_after :"), f"pred_after {pred_after}"
                pred_after = pred_after[len("pred_after : ") :]

                pred_list = next(iterator)
                assert pred_list.startswith("pred_list :"), f"pred_list {pred_list}"

                pred_mode = next(iterator)
                assert pred_mode.startswith("pred_mode :"), f"pred_mode {pred_mode}"

                GT = next(iterator)
                assert GT.startswith("GT :"), f"GT {GT}"
                GT = GT[len("GT : ") :]
                element["correct_answer"] = GT == pred_after

                stars = next(iterator)
                assert stars == "*************************", "Stars end"

                yield element
        except StopIteration:
            pass
        finally:
            pass

    elements = []
    for element in parse_elements(iterator):
        elements.append(element)
    return elements


def parse_wei_log(path_to_directory, dataset):
    inputs = _read_file(os.path.join(path_to_directory, dataset + "_stream_inputs"))
    targets = _read_file(os.path.join(path_to_directory, dataset + "_stream_targets"))
    predictions = _read_file(os.path.join(path_to_directory, dataset + "_stream_predictions"))

    if dataset == "commonsenseqa":
        spl = "Answer Choices"
        skip = 2149
    elif dataset == "strategyqa":
        spl = "A:"
        skip = 1467

    elements = []
    for (input, target, prediction) in zip(inputs, targets, predictions):

        # skip few shot examples
        question = input[skip:].split(spl)[0].strip()
        split = prediction.split(" So the answer is ")
        if len(split) == 1:
            cot = prediction
            answer = ""
        elif len(split) == 2:
            (cot, answer) = split
            answer = answer[:-1] if answer[-1] == "." else answer  # remove . at the end
        else:
            raise ValueError
        elements.append(
            {
                "id": "",
                "question": question,
                "cot": cot,
                "prediction": "So the answer is " + answer + ".",
                "correct_answer": (target == answer),
            }
        )
    return elements


def map_example_to_kojima_cot(question, cots):
    """
    Given a question from a dataset and list of Cots from the collection of Kojima (see function parse_kojima_log)
    it returns a populated CoT item for the given question if found in the list of Cots.
    If the question cannot be found in the cots it returns None.
    
    :param question: Original question from dataset
    :param cots: the question
    :return: The populated ThoughtSource CoT item
    """
    for id, cot in enumerate(cots):
        if question in cot["question"]:
            generated_cot = {
                "id": id,
                "fragments_version": "0.01",
                "instruction": None,
                "cot_trigger": "kojima-01",
                "prompt_text": "",
                "answers": [
                    {
                        "id": 0,
                        "answer_extraction": "kojima-A-E",
                        "answer_extraction_text": "",
                        "answer": cot["prediction"],
                        "correct_answer": cot["correct_answer"],
                    }
                ],
                "cot": cot["cot"],
                "author": "kojima",
                "date": None,
                "api_service": "",
                "model": "gpt-3",
                "comment": "",
                "annotation": [],
            }
            return generated_cot
    else:
        return None


def map_example_to_wei_cot(question, cots):
    """
    Given a question from a dataset and list of Cots from the collection of Wei (see function parse_wei_log)
    it returns a populated CoT item for the given question if found in the list of Cots.
    If the question cannot be found in the cots it returns None.
    
    :param question: Original question from dataset
    :param cots: the question
    :return: The populated ThoughtSource CoT item
    """
    for id, cot in enumerate(cots):
        if question in cot["question"]:
            generated_cot = {
                "id": id,
                "fragments_version": "0.01",
                "instruction": None,
                "cot_trigger": None,
                "prompt_text": "",
                "answers": [
                    {
                        "id": 0,
                        "answer_extraction": None,
                        "answer_extraction_text": "",
                        "answer": cot["prediction"],
                        "correct_answer": cot["correct_answer"],
                    }
                ],
                "cot": cot["cot"],
                "author": "wei",
                "date": None,
                "api_service": "",
                "model": "gpt-3",
                "comment": "",
                "annotation": [],
            }
            return generated_cot
    else:
        return None


def map_example_to_lievin_cot(id, item, dataset):
    """
    Given an CoT item from the collection of Lievin, returns a populated CoT item.
    
    :param item: the CoT item loaded from Lievin
    :return: The populated ThoughtSource CoT item
    """
    assert (__import__("cot").generate.FRAGMENTS["version"] == "0.01"), "New version"
    if dataset in ["med_qa", "medmc_qa"]:
        assert (item["extractive_prompt"] == "\n\nTherefore, among A through D, the answer is"), f"Different extractive prompt than expected {item['extractive_prompt']}"
        answer_extraction = "kojima-A-D"
    elif dataset == "pubmed_qa":
        assert (item["extractive_prompt"] == "\n\nTherefore, among A through C, the answer is"), f"Different extractive prompt than expected {item['extractive_prompt']}"
        answer_extraction = "kojima-A-C"
    else:
        raise ValueError

    assert (item["cot"].startswith(item["strategy"])), f"Different CoT than expected {item['cot']}"

    # TODO 
    # Investigate lievin cot prefix (They start differently and connot be easily removed)
    # medqa_us_test-0_incorrect_B_A.json: Let's think step by step about what the resident should do. \n\nThe first step
    # medqa_us_test-1_correct_D_D.json: Let's think step by step. This patient has a ring

    cot_triggers = {
        "Let's think step by step": "kojima-01",
        "Let's think step by step like a medical expert": "lievin-10",
        "Let's derive the differential diagnosis step by step": "lievin-01",
        "Let's use step by step inductive reasoning, given the medical nature of the question": "lievin-02",
        "Let's differentiate using step by step reasoning like a medical expert": "lievin-03",
    }

    return {
        "id": id,
        "fragments_version": "0.01",
        "instruction": None,
        "cot_trigger": cot_triggers[item["strategy"]],
        "prompt_text": "",
        "answers": [
            {
                "id": 0,
                "answer_extraction": answer_extraction,
                "answer_extraction_text": "",
                "answer": item["options"][item["prediction_idx"]],
                "correct_answer": (item["prediction_symbol"] == "correct"),
            }
        ],
        "cot": item["cot"],
        "author": "lievin",
        "date": None,
        "api_service": "",
        "model": "davinci-002",
        "comment": "",
        "annotation": [],
    }
