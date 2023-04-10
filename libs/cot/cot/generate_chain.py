"""
Context:

The data loader contains the following functions:

generate_extract_flexibly
generate_flexible
extract_flexible
metareason_flexible

These functions call methods in the generate_chain.py file, namely:
- self_generate_extract
- self_generate
- self_extract
- self_reflect

All of them require an input_dictionary with variables and a langchain.
The different functions are there to put the right output in the right ts-variables of the schema.
If you want to save different variables, 
you can iterate on one of the existing functions 
and feed a new langchain with additions to the input dict

flexible_langchains.ipynb is a tutorial
And experiments are performed in langchain_experiments.ipynb and summary_experiments.ipynb (internal)

Issues:
*Use of fragments
*How to choose which existing CoT to use in self_reflect for instance?
"""


import datetime
import json
import pkgutil
import uuid
from dataclasses import asdict
import datasets as ds
# from cot.config import Config

# disable transformation (e.g. map) caching
# https://huggingface.co/docs/datasets/v2.6.1/en/package_reference/main_classes#datasets.disable_caching
ds.disable_caching()
FRAGMENTS = json.loads(pkgutil.get_data(__name__, "fragments.json"))

""" 
Input: item, langchains, triggers
Output: cot and answer
Generate a cot and extract an answer with helper function _self_generate_extract
"""
def self_generate_extract(data,chain,input_dict):

    #temp dataset to be filled
    new_dataset = []
    
    #For loop in dataset.map()
    for example in data:
        processed_example = _self_generate_extract(example,input_dict,chain)
        new_dataset.append(processed_example)
    return new_dataset

def _self_generate_extract(item,input_dict,chain):
    
    input_dict['question'] = item["question"]
    input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])
    
    #this is where the magic happens: get cot and predicted answer
    lang_chain = chain(input_dict) 

    generated_cot = {
                "id": str(uuid.uuid4()),
                "fragments_version": FRAGMENTS["version"],
                "instruction": input_dict['instruction'],
                "cot_trigger": input_dict['cot_trigger'],
                "cot_trigger_template": "",
                "prompt_text": "",
                "cot": lang_chain['cot'],
                "answers": [],
                "author": "",
                "date": "",
                "api_service": "",
                "model": str(
                    {
                        "name": input_dict['model_name'],
                        "temperature": 0,
                        "max_tokens": 800,
                    }
                ),
                "comment": "",
                "annotations": [],
            }
    generated_cot["date"] = print_now(1)

    answer = {
                        "id": str(uuid.uuid4()),
                        "answer_extraction": input_dict['answer_extraction'],
                        "answer_extraction_template": "",
                        "answer_extraction_text": "",
                        "answer": "",
                        "correct_answer": None,
                }
    answer["answer"] = lang_chain['predicted_answer']
    generated_cot["answers"].append(answer)

    item["generated_cot"].append(generated_cot)

    return item

"""Generate CoTs only"""
def self_generate(data,chain,input_dict):

    input_dict['chain'] = chain

    new_dataset = []
    for example in data:
        processed_example = _self_generate(example,input_dict,chain)
        new_dataset.append(processed_example)
    return new_dataset

def _self_generate(item,input_dict,chain):

    input_dict['question'] = item["question"]
    input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])
    
    lang_chain = chain(input_dict)

    """If conditions for input keys"""
    generated_cot = {
                "id": str(uuid.uuid4()),
                "fragments_version": FRAGMENTS["version"],
                "instruction": input_dict["instruction"],
                "cot_trigger": input_dict["cot_trigger"],
                "cot_trigger_template": "",
                "prompt_text": "",
                "cot": lang_chain['cot'],
                "answers": [],
                "author": "",
                "date": "",
                "api_service": input_dict["api_service"],
                "model": str(
                    {
                        "name": input_dict["model"],
                        "temperature": 0,
                        "max_tokens": 800,
                    }
                ),
                "comment": "",
                "annotations": [],
            }
    generated_cot["date"] = print_now(1)

    item["generated_cot"].append(generated_cot)

    return item

"""Extract answers based on CoTs only"""
def self_extract(data,chain,input_dict):

    new_dataset = []
    for example in data:
        processed_example = _self_extract(example,input_dict,chain)
        new_dataset.append(processed_example)
    return new_dataset

"""ToDo show which CoT to take"""
def _self_extract(item,input_dict,chain):

    input_dict['question'] = item["question"]
    input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])

    #extract based on the first cot in the dataset
    cot = item['generated_cot'][0]['cot'] 
    input_dict['cot'] = cot
    
    #this is where the magic happens
    lang_chain = chain(input_dict)
    #retrieve question and answer choices from item, add to input dict
    generated_cot = {
                "id": str(uuid.uuid4()),
                "fragments_version": FRAGMENTS["version"],
                "instruction": input_dict["instruction"],
                "cot_trigger": input_dict["cot_trigger"],
                "cot_trigger_template": "",
                "prompt_text": "",
                "cot": lang_chain['cot'],
                "answers": [],
                "author": "",
                "date": "",
                "api_service": input_dict["api_service"],
                "model": str(
                    {
                        "name": input_dict["model"],
                        "temperature": 0,
                        "max_tokens": 800,
                    }
                ),
                "comment": "answer_extraction cot",
                "annotations": [],
            }
    generated_cot["date"] = print_now(1)

    """If conditions for input keys"""
    answer = {
                        "id": str(uuid.uuid4()),
                        "answer_extraction": input_dict['answer_extraction'],
                        "answer_extraction_template": "",
                        "answer_extraction_text": "",
                        "answer": "",
                        "correct_answer": None,
                }
    answer["answer"] = lang_chain['predicted_answer']
    
    #we add a generated cot (with new ans) to be assessed in annotator
    generated_cot["answers"].append(answer) 
    item["generated_cot"].append(generated_cot)

    #could use line below to add the answer to existing cot
    #item["generated_cot"][0]["answers"].append(answer)

    return item

"""Reflect on CoT (or some other part) and generate new answer"""
def self_reflect(data,chain,input_dict):

    new_dataset = []
    for example in data:
        processed_example = _self_reflect(example,input_dict,chain)
        new_dataset.append(processed_example)
    return new_dataset


"""In this version the reflection is added to generated_cot"""
def _self_reflect(item,input_dict,chain):

    input_dict['question'] = item["question"]
    input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])
    input_dict['cot'] = item['generated_cot'][0]['cot']

    # here we take the first answer from the first cot
    input_dict['answer'] = item["generated_cot"][0]['answers'][0]['answer']
    
    #this is where the magic happens
    lang_chain = chain(input_dict)

    #retrieve question and answer choices from item, add to input dict
    generated_cot = {
                "id": str(uuid.uuid4()),
                "fragments_version": FRAGMENTS["version"],
                "instruction": "",
                "cot_trigger": input_dict["reflection_prompt"],
                "cot_trigger_template": "",
                "prompt_text": "",
                "cot": lang_chain['reflection'],
                "answers": [],
                "author": "",
                "date": "",
                "api_service": input_dict["api_service"],
                "model": str(
                    {
                        "name": input_dict["model"],
                        "temperature": 0,
                        "max_tokens": 800,
                    }
                ),
                "comment": "self_reflection cot",
                "annotations": [],
            }
    generated_cot["date"] = print_now(1)

    """If conditions for input keys"""
    answer = {
                        "id": str(uuid.uuid4()),
                        "answer_extraction": input_dict['reflect_answer_extraction'],
                        "answer_extraction_template": "",
                        "answer_extraction_text": "self_reflection",
                        "answer": "",
                        "correct_answer": None,
                }
    answer["answer"] = lang_chain['reflection_answer']


    generated_cot["answers"].append(answer) 

    item["generated_cot"].append(generated_cot)
    

    return item

def keep_generated_cots(dataset, authors=None):
    """This function handles which pregenerated COTS are deleted (after loading a collection).

    :param authors: A list of authors of the pregenerated COTS to delete. If None, all of the pregenerated COTS are kept.
    if "all", all of the pregenerated COTS are deleted.
    """
    # Unfortunately the loading function of the datasets does not let you specify which pregenerated COTS to load
    # So we load all of them and then delete the ones we don't want

    # remove all the pregenerated COTS that are not in the list
    dataset = dataset.map(
        _keep_generated_cots,
        fn_kwargs={"authors": authors},
        features=dataset.info.features,
        # deleting the cache is necessary in generate if you call it multiple times
        # not clear if it is needed here, but it doesn't hurt
        load_from_cache_file=False,
    )
    return dataset


def _keep_generated_cots(item, authors=None):
    if authors is None:
        item["generated_cot"] = []
    else:
        item["generated_cot"] = [cot for cot in item["generated_cot"] if cot["author"] in authors]
        # for deletion we could use "... not in authors" instead of "in authors"

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


def multiple_choice_answer_formatting(answer_choices):
    """Transforms a list of answer choices into a string with letters (A,B,C,...) for each answer choice."""
    # only supports uppercase letters at the moment, as this is current standard

    # Adding Letters (A,B,C,...) for the given multiple choice answers.
    return "\n".join([f"{chr(65+i)}) {example}" for i, example in enumerate(answer_choices)])  # 65 is the ASCII code for A


def get_fragments_value(str, key):
    if key is None:
        return None
    else:
        return FRAGMENTS[str][key]
