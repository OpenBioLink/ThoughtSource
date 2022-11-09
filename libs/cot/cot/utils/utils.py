import os

def parse_kojima_log(path, dataset):
    with open(path, "r", encoding="utf8") as infile:
        content = infile.readlines()
    content = [x.strip() for x in content]

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
                element = {"question": "", "cot": "", "prediction": ""}

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

                pred_before = next(iterator)
                if not pred_before.startswith("pred_before :"):
                    while not pred_before.startswith("pred_before"):
                        pred_before = next(iterator)
                assert pred_before.startswith("pred_before :"), f"pred_before {pred_before}"

                pred_after = next(iterator)
                assert pred_after.startswith("pred_after :"), f"pred_after {pred_after}"
                element["prediction"] = pred_after[len("pred_after : ") :]
                pred_list = next(iterator)
                assert pred_list.startswith("pred_list :"), f"pred_list {pred_list}"
                pred_mode = next(iterator)
                assert pred_mode.startswith("pred_mode :"), f"pred_mode {pred_mode}"
                GT = next(iterator)
                assert GT.startswith("GT :"), f"GT {GT}"

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


def _read_file(path):
    with open(path, "r") as infile:
        content = infile.readlines()
    content = [x.strip() for x in content]
    return content


def parse_wei_log(path_to_directory, dataset):
    inputs = _read_file(os.path.join(path_to_directory, dataset + "_stream_inputs"))
    targets = _read_file(os.path.join(path_to_directory, dataset + "_stream_targets"))
    predictions = _read_file(os.path.join(path_to_directory, dataset + "_stream_predictions"))

    elements = []
    for (input, target, prediction) in zip(inputs, targets, predictions):
        # skip few shot examples
        question = input[2149:].split("Answer Choices")[0].strip()
        target = True if target == "yes" else False

        elements.append({"id": "", "question": question, "cot": prediction, "prediction": prediction})
    return elements


def map_example_to_kojima_cot(example, cots):
    for cot in cots:
        if example["question"]["stem"] in cot["question"]:
            generated_cot = {
                "templates_version": "0.01",
                "instruction": None,
                "cot-trigger": "kojima-01",
                "answers": [
                    {
                        "answer-extraction": "kojima-A-E",
                        "answer": cot["prediction"],
                        "correct_answer": None,
                    }
                ],
                "cot": cot["cot"],
                "author": "kojima",
                "date": None,
                "model": "gpt-3",
                "comment": "",
                "annotation": [],
            }
            return generated_cot
    else:
        return None

def map_example_to_wei_cot(example, cots):
    for cot in cots:
        if example["question"]["stem"] in cot["question"]:
            generated_cot = {
                "templates_version": "0.01",
                "instruction": None,
                "cot-trigger": None,
                "answers": [
                    {
                        "answer-extraction": None,
                        "answer": cot["prediction"],
                        "correct_answer": None,
                    }
                ],
                "cot": cot["cot"],
                "author": "wei",
                "date": None,
                "model": "gpt-3",
                "comment": "",
                "annotation": [],
            }
            return generated_cot
    else:
        return None