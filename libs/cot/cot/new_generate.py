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
input of chains can be reduced to one or a list for flexibility
"""
def self_generate_extract(data,chain,input_dict):

    ds.disable_caching()
    data.cleanup_cache_files()

    """Not sure if necessary"""
    if isinstance(data, ds.arrow_dataset.Dataset):
        features = data.info.features

    elif isinstance(data, ds.dataset_dict.DatasetDict):
        name_of_first_split = list(data.keys())[0]
        features = data[name_of_first_split].info.features

    else:
        raise ValueError("Not recognized data")
    

    input_dict['chain'] = chain

    return data.map(
        _self_generate_extract,
        with_indices=True,
        fn_kwargs=input_dict,
        features=features,
        load_from_cache_file=False,
    )

def _self_generate_extract(item,idx,instruction,cot_trigger,answer_extraction,chain):
   

    input_dict = {}
    input_dict['question'] = item["question"]
    input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])
    input_dict['instruction'] = instruction
    input_dict['cot_trigger'] = cot_trigger
    input_dict['answer_extraction'] = answer_extraction
    
    #this is where the magic happens
    lang_chain = chain(input_dict)
    #retrieve question and answer choices from item, add to input dict

    """If conditions for input keys"""
    generated_cot = {
                "id": str(uuid.uuid4()),
                "fragments_version": FRAGMENTS["version"],
                "instruction": instruction,
                "cot_trigger": cot_trigger,
                "cot_trigger_template": "",
                "prompt_text": "",
                "cot": lang_chain['cot'],
                "answers": [],
                "author": "",
                "date": "",
                "api_service": "",
                "model": str(
                    {
                        "name": "",
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
                        "answer_extraction": answer_extraction,
                        "answer_extraction_template": "",
                        "answer_extraction_text": "",
                        "answer": "",
                        "correct_answer": None,
                }
    answer["answer"] = lang_chain['predicted_answer']
    generated_cot["answers"].append(answer)

    item["generated_cot"].append(generated_cot)

    return item

# def _self_generate_extract(item,input_dict,chain):
    

#     input_dict['question'] = item["question"]
#     input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])
    
#     #this is where the magic happens
#     lang_chain = chain(input_dict)
#     #retrieve question and answer choices from item, add to input dict

#     """If conditions for input keys"""
#     generated_cot = {
#                 "id": str(uuid.uuid4()),
#                 "fragments_version": FRAGMENTS["version"],
#                 "instruction": input_dict["instruction"],
#                 "cot_trigger": input_dict["cot_trigger"],
#                 "cot_trigger_template": "",
#                 "prompt_text": "",
#                 "cot": lang_chain['cot'],
#                 "answers": [],
#                 "author": "",
#                 "date": "",
#                 "api_service": "",
#                 "model": str(
#                     {
#                         "name": "",
#                         "temperature": 0,
#                         "max_tokens": 800,
#                     }
#                 ),
#                 "comment": "",
#                 "annotations": [],
#             }
#     generated_cot["date"] = print_now(1)

#     answer = {
#                         "id": str(uuid.uuid4()),
#                         "answer_extraction": input_dict['answer_extraction_key'],
#                         "answer_extraction_template": "",
#                         "answer_extraction_text": "",
#                         "answer": "",
#                         "correct_answer": None,
#                 }
#     answer["answer"] = lang_chain['predicted_answer']
#     generated_cot["answers"].append(answer)

#     item["generated_cot"].append(generated_cot)

#     return item


def self_generate(data,chain,input_dict):

    ds.disable_caching()
    data.cleanup_cache_files()

    """Not sure if necessary"""
    if isinstance(data, ds.arrow_dataset.Dataset):
        features = data.info.features

    elif isinstance(data, ds.dataset_dict.DatasetDict):
        name_of_first_split = list(data.keys())[0]
        features = data[name_of_first_split].info.features

    else:
        raise ValueError("Not recognized data")
    
    input_dict['chain'] = chain

    # return data.map(
    #     _self_generate_extract,
    #     with_indices=True,
    #     fn_kwargs=input_dict,
    #     features=features,
    #     load_from_cache_file=False,
    # )


    
    new_dataset = []
    for example in data['train']: #hardcoded
        processed_example = _self_generate(example,input_dict,chain)
        print("processed_example:")
        print(processed_example)
        new_dataset.append(processed_example)
    return new_dataset

def _self_generate(item,input_dict,chain):


    input_dict['question'] = item["question"]
    input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])
    
    #this is where the magic happens
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
                "api_service": "",
                "model": str(
                    {
                        "name": "",
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

"""Can een item have multiple cots, if so how to choose one?"""  
def extract(data,chain,input_dict):

    ds.disable_caching()
    data.cleanup_cache_files()

    """Not sure if necessary"""
    if isinstance(data, ds.arrow_dataset.Dataset):
        features = data.info.features

    elif isinstance(data, ds.dataset_dict.DatasetDict):
        name_of_first_split = list(data.keys())[0]
        features = data[name_of_first_split].info.features

    else:
        raise ValueError("Not recognized data")
    
    new_dataset = []
    for example in data['train']:
        processed_example = _extract(example,input_dict,chain)
        print("processed_example:")
        print(processed_example)
        new_dataset.append(processed_example)
    return new_dataset


def _extract(item,input_dict,chain):

    input_dict['question'] = item["question"]
    input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])

    """Use this cot"""
    cot = item['generated_cot']['cot'] 
    
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
                "api_service": "",
                "model": str(
                    {
                        "name": "",
                        "temperature": 0,
                        "max_tokens": 800,
                    }
                ),
                "comment": "self_reflection cot",
                "annotations": [],
            }
    generated_cot["date"] = print_now(1)

    item["generated_cot"].append(generated_cot)

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
    item["generated_cot"]["answers"].append(answer)

    return item

def self_reason(data,chain,input_dict):

    """Can remove this?"""
    ds.disable_caching()
    data.cleanup_cache_files()

    """Not sure if necessary"""
    if isinstance(data, ds.arrow_dataset.Dataset):
        features = data.info.features

    elif isinstance(data, ds.dataset_dict.DatasetDict):
        name_of_first_split = list(data.keys())[0]
        features = data[name_of_first_split].info.features

    else:
        raise ValueError("Not recognized data")
    """Until here?"""

    
    new_dataset = []
    for example in data['train']: #hardcoded
        processed_example = _self_reason(example,input_dict,chain)
        print("processed_example:")
        print(processed_example)
        new_dataset.append(processed_example)
    return new_dataset

def _self_reason(item,input_dict,chain):

    input_dict['question'] = item["question"]
    input_dict['answer_choices'] = multiple_choice_answer_formatting(item["choices"])
    
    #this is where the magic happens
    lang_chain = chain(input_dict)
    #retrieve question and answer choices from item, add to input dict

    """If conditions for input keys"""
    answer = {
                        "id": str(uuid.uuid4()),
                        "answer_extraction": input_dict['answer_extraction'],
                        "answer_extraction_template": "",
                        "answer_extraction_text": "self_reflection",
                        "answer": "",
                        "correct_answer": None,
                }
    answer["answer"] = lang_chain['predicted_answer']
    item["generated_cot"]["answers"].append(answer)

    return item
    
# def generate_and_extract(
#     item,
#     idx,
#     author,
#     api_service,
#     engine,
#     temperature,
#     max_tokens,
#     api_time_interval,
#     instruction_keys,
#     cot_trigger_keys,
#     template_cot_generation,
#     answer_extraction_keys,
#     template_answer_extraction,
#     warn,
#     verbose,
# ):
#     return


# def generate_and_extract(data, config):
#     """
#     It takes a dataset and a config and generates cots for each example and extract answers.

#     :param data: Dataset/DatasetDict - the dataset you want to generate CoTs for and extract answers
#     :param config: Dictionary - the configurations of the input and model
#     :return: the dataset with generated cots and extracted answers
#     """

#     return data.map(
#         _generate_and_extract,
#         with_indices=True,
#         fn_kwargs=asdict(config_as_dataclass),
#         features=features,
#         load_from_cache_file=False,
#     )
    #return item





"""
"""
"""
"""

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
