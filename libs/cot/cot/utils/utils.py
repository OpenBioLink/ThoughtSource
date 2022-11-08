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
                element = {"id": "", "question": "", "cot": "", "prediction": ""}

                stars = next(iterator)
                if stars.startswith("accuracy"):
                    break
                assert stars == "*************************", "Stars begin"

                st_data = next(iterator)
                element["id"] = st_data
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
