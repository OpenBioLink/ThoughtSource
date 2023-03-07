import datetime
import json
import pkgutil
import uuid
from dataclasses import asdict
import datasets as ds
from cot.config import Config

# disable transformation (e.g. map) caching
# https://huggingface.co/docs/datasets/v2.6.1/en/package_reference/main_classes#datasets.disable_caching
ds.disable_caching()
FRAGMENTS = json.loads(pkgutil.get_data(__name__, "fragments.json"))

""" 
Input: item, langchains, triggers
Output: cot and answer
"""
def generate():

    # can also be input
    instruction = get_fragments_value("instructions", instruction_key)
    template_dict["cot_trigger"] = get_fragments_value("cot_triggers", cot_trigger_key)

    generated_cot = {
                "id": str(uuid.uuid4()),
                "fragments_version": FRAGMENTS["version"],
                "instruction": instruction_key,
                "cot_trigger": cot_trigger_key,
                "cot_trigger_template": template_cot_generation,
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
                "annotations": [],
            }
    generated_cot["cot"] = cot
    generated_cot["date"] = print_now(1)
    
def extract():
    answer = {
                        "id": str(uuid.uuid4()),
                        "answer_extraction": answer_extraction_key,
                        "answer_extraction_template": template_answer_extraction,
                        "answer_extraction_text": "",
                        "answer": "",
                        "correct_answer": None,
                }
    answer["answer"] = predicted_answer
    generated_cot["answers"].append(answer)
    item["generated_cot"].append(generated_cot)
    
def generate_and_extract(
    item,
    idx,
    author,
    api_service,
    engine,
    temperature,
    max_tokens,
    api_time_interval,
    instruction_keys,
    cot_trigger_keys,
    template_cot_generation,
    answer_extraction_keys,
    template_answer_extraction,
    warn,
    verbose,
):






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
