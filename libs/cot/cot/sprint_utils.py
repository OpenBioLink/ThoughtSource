import os
from cot import Collection
from cot.generate import FRAGMENTS
from rich.pretty import pprint
import json

def get_answer_extraction(type,choices): # collection[dataset_name][split][0]["choices"]
    if type == "bool": 
        return "kojima-yes-no"
    elif type == "multiplechoice":
        n_choices = len(choices)
        if n_choices == 3: answer_extraction_key = 'kojima-A-C'
        elif n_choices == 4: answer_extraction_key = 'kojima-A-D'
        elif n_choices == 5: answer_extraction_key = 'kojima-A-E'
        elif n_choices == 6: answer_extraction_key = 'kojima-A-F'
        return(answer_extraction_key)
    else: raise ValueError("type must be bool or multiplechoice")

# join strings from a list with underscore
def join_strings(list_of_strings):
    joined_string = ""
    for string in list_of_strings:
        joined_string += ("_" + str(string))
    return(joined_string)


def model_sprint(dataset_name_splits, api_service_model, cot_trigger_keys, number_examples):
    for dataset_name, split in dataset_name_splits:
        collection = Collection([dataset_name], verbose=False)
        collection = collection.select(split=split, number_samples=number_examples)
        answer_extraction_key = get_answer_extraction(collection[dataset_name][split][0]["type"], collection[dataset_name][split][0]["choices"])
        for api_service, model, api_time_interval in api_service_model:

            config={
                "cot_trigger_keys": cot_trigger_keys,
                "answer_extraction_keys": [answer_extraction_key], # Therefore, among A through C/D/E/F, the answer is'
                "author" : "thoughtsource",
                "api_service": api_service,
                "engine": model,
                "temperature": 0,
                "max_tokens": 512,
                "api_time_interval": api_time_interval,
                "verbose": False,
                "warn": False,
            }
        
            collection.generate(config=config)
            collection.evaluate()
            collection.dump(dataset_name + "_" + split + "_" + str(number_examples) + "_" + api_service + "_" + model.replace("/", "_") + join_strings(config["cot_trigger_keys"]) + ".json")