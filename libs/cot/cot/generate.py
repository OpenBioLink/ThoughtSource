import datetime
import json
import os
import pkgutil
import time
import uuid

import datasets as ds

from cot.config import Config
from dataclasses import asdict

# import pydantic
# from langchain.prompts import BasePromptTemplate
# from pydantic import BaseModel

# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate


# disable transformation (e.g. map) caching
# https://huggingface.co/docs/datasets/v2.6.1/en/package_reference/main_classes#datasets.disable_caching
ds.disable_caching()

FRAGMENTS = json.loads(pkgutil.get_data(__name__, "fragments.json"))

def generate_and_extract(data, config):
    """
    It takes a dataset and a config and generates cots for each example and extract answers.

    :param data: Dataset/DatasetDict - the dataset you want to generate CoTs for and extract answers
    :param config: Dictionary - the configurations of the input and model
    :return: the dataset with generated cots and extracted answers
    """

    ds.disable_caching()

    # # Creating configurations for the options 'all' or 'None':
    # keys = ["instruction_keys", "cot_trigger_keys", "answer_extraction_keys"]
    # names_in_template = ["instructions", "cot_triggers", "answer_extractions"]
    # for key, name in zip(keys, names_in_template):
    #     if key not in config or config[key] == "all":
    #         config[key] = [None] + list(FRAGMENTS[name].keys())
    #     elif not config[key]:
    #         config[key] = [None]

    # Inserts None at index 0 of instruction_keys to query without an explicit instruction
    # TODO rethink this, maybe add option to disable this
    # for key in ["instruction_keys","cot_trigger_keys"]:
    #     if None not in config[key]:
    #         config[key].insert(0, None)

    if isinstance(data, ds.arrow_dataset.Dataset):
        features = data.info.features
        if "idx_range" in config and config["idx_range"] != "all":
            n_samples = config["idx_range"][1] - config["idx_range"][0]
        else:
            n_samples = len(data)
    elif isinstance(data, ds.dataset_dict.DatasetDict):
        features = data["train"].info.features
        if "idx_range" in config and config["idx_range"] != "all":
            n_samples = (config["idx_range"][1] - config["idx_range"][0]) * len(data)
        else:
            n_samples = sum([len(data[x]) for x in data])
    else:
        raise ValueError("Not recognized data")
    
    if config["warn"]:
        print_warning(config, n_samples)

    # The config is transformed into a dataclass object, where all testing is done
    # But it will be transformed back to a dictionary for the function 'map'
    config = Config(**config)

    return data.map(
        _generate_and_extract, with_indices=True, fn_kwargs=asdict(config), features=features
    )


def _generate_and_extract(
    item,
    idx,
    # CONTINUE HERE
    # **fn_kwargs,
    
    idx_range="all",
    author="",
    api_service="huggingface_hub",
    engine="google/flan-t5-xl",
    temperature=0,
    max_tokens=128,
    api_time_interval=1.0,
    multiple_choice_answer_format="Letters",
    instruction_keys="all",
    cot_trigger_keys="all",
    template_cot_generation="\n{instruction}\n\n{question}\n{answer_choices}\n\n{cot_trigger}\n",
    answer_extraction_keys="all",
    template_answer_extraction="\n{instruction}\n\n{question}\n{answer_choices}\n\n{cot_trigger}\n{cot}\n{answer_extraction}\n",
    debug=True,
    warn=True,
    verbose=False,
):
    """
    The function takes in a JSON object (item) and generates a CoT (Chain-of-Thought) for each combination of
    of instructions and CoT triggers. For each generated CoT and for each of the given answer extractions it extracts an answer

    :param item: the item (example) of a dataset to be processed
    :param idx: the index of the item in the dataset
    :param idx_range: the range of indices to generate and extract for, if idx not within idx_range do nothing and return item
    :param author: the name of the person who generated the CoT
    :param engine: the GPT-3 engine to use, defaults to text-davinci-002 (optional)
    :param temperature: 0.0 means the model will output the most likely answer, 1.0 means the model will
        output the most random answer, defaults to 0 (optional)
    :param max_tokens: The maximum number of tokens to generate, defaults to 128 (optional)
    :param api_time_interval: The time interval between API calls
    :param instruction_keys: the instructions to generate the CoT
    :param cot_trigger_keys: the trigger to generate the CoT
    :param answer_extraction_keys: the trigger to extract answers given a generated CoT
    :param debug: If True, will print out the prompts and generated text, defaults to True (optional)
    :return: item populated with various fields
    """

    if idx_range == "all" or (idx >= idx_range[0] and idx < idx_range[1]):
        pass
    else:
        return item

    # predefine values in template dictionary that stay same over all runs
    template_dict = {
        "instruction": None,
        "question": item["question"],
        "answer_choices": multiple_choice_answer_formatting(
            multiple_choice_answer_format, item["choices"]
        ),
        "cot_trigger": None,
        "cot": None,
        "answer_extraction": None,
    }

    # CONTINUE HERE
    # Not clear how to access the variables. Cannot access config here, but I think if I use
    # instruction_keys from here it will just always be "all"
    # check_templates(config, template_dict, template_cot_generation, template_answer_extraction)

    # generate CoT and extract answers
    for instruction_key in instruction_keys:
        template_dict["instruction"] = FRAGMENTS["instructions"][instruction_key]

        for cot_trigger_key in cot_trigger_keys:
            generated_cot = {
                "id": str(uuid.uuid4()),
                "fragments_version": FRAGMENTS["version"],
                "instruction": instruction_key,
                "cot_trigger": cot_trigger_key,
                "prompt_text": "",
                "cot": "",
                "answers": [],
                "author": author,
                "date": "",
                "api_service": api_service,
                "model": str(
                    {
                        "name": engine,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                ),
                "comment": "",
                "annotation": [],
            }

            template_dict["cot_trigger"] = FRAGMENTS["cot_triggers"][cot_trigger_key]
            generate_cot_prompt = format_prompt(template_cot_generation, template_dict)

            if verbose:
                print("\n-------------------COT TRIGGER-------------------")
                print(generate_cot_prompt)

            cot = query_model(
                generate_cot_prompt,
                api_service,
                engine,
                temperature,
                max_tokens,
                api_time_interval,
                debug,
            )
            if verbose:
                print("\n------------------GENERATED COT-------------------")
                print(cot)

            template_dict["cot"] = cot

            generated_cot["cot"] = cot
            generated_cot["prompt_text"] = generate_cot_prompt
            generated_cot["date"] = print_now(1)

            for answer_extraction_key in answer_extraction_keys:
                
                if answer_extraction_key is None:
                    pass

                else:
                    answer = {
                        "id": str(uuid.uuid4()),
                        "answer_extraction": answer_extraction_key,
                        "answer_extraction_text": "",
                        "answer": "",
                        "correct_answer": None,
                    }

                    template_dict["answer_extraction"] = FRAGMENTS["answer_extractions"][answer_extraction_key]
                    answer_extraction_prompt = format_prompt(template_answer_extraction,template_dict)

                    if verbose:
                        print("\n------------------ANSWER EXTRACTION------------------")
                        print(answer_extraction_prompt)

                    predicted_answer = query_model(
                        answer_extraction_prompt,
                        api_service,
                        engine,
                        temperature,
                        max_tokens,
                        api_time_interval,
                        debug,
                    )
                    if verbose:
                        print("\n------------------EXTRACTED ANSWER-------------------")
                        print(predicted_answer)

                    answer["answer"] = predicted_answer
                    answer["answer_extraction_text"] = answer_extraction_prompt
                    generated_cot["answers"].append(answer)

            item["generated_cot"].append(generated_cot)

    return item

def print_now(return_flag=0):
    """
    It takes a flag as an argument and prints the current time in a specific format

    :param return_flag: 0 = print, 1 = return, defaults to 0 (optional)
    :return: the current time in the format of 'YYYY/MM/DD HH:MM:SS'
    """
    now = datetime.datetime.now()
    now = now.strftime("%Y/%m/%d %H:%M:%S")
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

def print_warning(config, n_samples):
    n_instruction_keys = len(config["instruction_keys"])
    n_cot_trigger_keys = len(config["cot_trigger_keys"])
    n_answer_extraction_keys = len(config["answer_extraction_keys"])

    n_total = (
        n_samples * n_instruction_keys * n_cot_trigger_keys
        + n_samples * n_instruction_keys * n_cot_trigger_keys * n_answer_extraction_keys
    )
    warning = f"""
        You are about to \033[1m call an external API \033[0m in total {n_total} times, which \033[1m may produce costs \033[0m.
        Number API calls for CoT generation: n_samples {n_samples} * n_instruction_keys {n_instruction_keys} * n_cot_trigger_keys {n_cot_trigger_keys}
        Number API calls for answer extraction: n_samples {n_samples} * n_instruction_keys {n_instruction_keys} * n_cot_trigger_keys {n_cot_trigger_keys} * n_answer_extraction_keys {n_answer_extraction_keys}
        Do you want to continue? y/n
        """
    if config["debug"]:
        warning += "\033[1m Note: You are in debug mode. When entering 'y', a test run without API calls is made. \033[0m"
    print(warning)
    ans = input()
    if ans.lower() == "y":
        pass
    else:
        return

def multiple_choice_answer_formatting(format, answer_choices):
    if format == None:
        return answer_choices
    elif format == "Letters":
        # Adding Letters (A,B,C,...) for the given multiple choice answers.
        return "\n".join(
            [
                f"{chr(65+i)}) {example}" for i, example in enumerate(answer_choices)
            ]  # 65 is the ASCII code for A
        )

# MOVED to config.py
# def check_templates(config, template_dict, template_cot_generation, template_answer_extraction):
#     ''' checks if the templates contain only allowed keys and if template matches the given config '''
#     import re
#     input_variables = re.findall(
#         "{(.*?)}", template_cot_generation + template_answer_extraction
#     )
#     assert all(elem in template_dict.keys() for elem in input_variables), (
#         f"Not all input variables {input_variables} are part of the specified {template_dict.keys()}"
#     )

#     # assert "instruction" in input_variables == (config["cot_trigger_keys"] != ["None"])

#     for (config_var, template_var) in zip(
#         [config["instruction_keys"], config["cot_trigger_keys"], config["answer_extraction_keys"]],
#         ["instruction", "cot_trigger", "answer_extraction"]
#     ):
#         assert (config_var != ["None"]) == template_var in input_variables, (
#             '''There seems to be a mismatch between the template and the provided keys in the config,
#             if {template_var} is in the template, {config_var} has to be specified and cannot be [None]'''
#             )

def format_prompt(template, dictionary):
    output = template.format_map(Correct_output(dictionary))
    output = delete_empty_curly_brackets(output)
    return output

class Correct_output(dict):
    # TODO: do I ever need this? I think there will never be missing keys
    # and None keys are handled by delete_empty_curly_brackets
    def __missing__(self, key):
        return ""

    def __getitem__(self, key):
        return dict.get(self, key) or ""

    # def get(self, key):
    #     return dict.get(self, key) or ""

def delete_empty_curly_brackets(string):
    string.replace("{None}\n", "")
    # string.replace("\n{None}", "") # TODO: do I need this?
    string.replace("{None}", "")
    return string

# # replace in dict None with empty string
# def dict_replace_none_with_empty_string(d):
#     for k, v in d.items():
#         if v is None:
#             d[k] = ""
#     return d


def query_model(
    input, api_service, engine, temperature, max_tokens, api_time_interval, debug
):
    if debug:
        return "test"

    # langchain package implementation
    else:
        from langchain import LLMChain, Prompt

        time.sleep(api_time_interval)
        template = "{prompt}"
        prompt = Prompt(template=template, input_variables=["prompt"])

        if api_service == "openai":
            from langchain import OpenAI

            llm_chain = LLMChain(
                prompt=prompt,
                llm=OpenAI(
                    model_name=engine,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
        if api_service == "huggingface_hub":
            from langchain import HuggingFaceHub

            llm_chain = LLMChain(
                prompt=prompt,
                llm=HuggingFaceHub(
                    repo_id=engine,
                    # parameter options: https://huggingface.co/docs/api-inference/detailed_parameters
                    model_kwargs={"temperature": temperature, "max_length": max_tokens},
                ),
            )
        response = llm_chain.predict(prompt=input, stop=None)
        return response
