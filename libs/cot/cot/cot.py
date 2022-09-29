import os
import openai
import re
import datetime
import time
import json
import pkgutil
import datasets as ds

TEMPLATES = json.loads(pkgutil.get_data(__name__, "templates.json"))

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


def generate_and_extract(data, config):

    if "instruction_keys" not in config or not config["instruction_keys"]:
        config["instruction_keys"] = TEMPLATES["instructions"].keys()
    if "cot_trigger_keys" not in config or not config["cot_trigger_keys"]:
        config["cot_trigger_keys"] = TEMPLATES["cot-triggers"].keys()
    if "answer_extraction_keys" not in config or not config["answer_extraction_keys"]:
        config["answer_extraction_keys"] = TEMPLATES["answer-extractions"].keys()

    if isinstance(data, ds.arrow_dataset.Dataset):
        n_samples = config['idx_range'][1] - config['idx_range'][0] if (('idx_range' in config) or (config['idx_range'] is None)) else len(data)
    elif isinstance(data, ds.dataset_dict.DatasetDict):
        n_samples = config['idx_range'][1] - config['idx_range'][0] if (('idx_range' in config) or (config['idx_range'] is None)) else sum([len(x) for x in data])
    else:
        raise ValueError("Not recognized data")

    n_samples = config['idx_range'][1] - config['idx_range'][0] if (('idx_range' in config) or (config['idx_range'] is None)) else len(data)
    n_instruction_keys = len(config["instruction_keys"])
    n_cot_trigger_keys = len(config["cot_trigger_keys"])
    n_answer_extraction_keys = len(config["answer_extraction_keys"])

    n_total = n_samples * n_instruction_keys * n_cot_trigger_keys + n_samples * n_instruction_keys * n_cot_trigger_keys * n_answer_extraction_keys

    if "debug" in config and not config["debug"]:
        warning = f"You are about to call the openai API which produces costs." + "\n"
        warning += f"Due to your settings you are about to call the openai API in total {n_total} times." + "\n"
        warning += f"Do you want to continue? y/n" + "\n"
        print(warning)
        ans = input()
        if ans.lower() == "y":
            pass
        else:
            return

    return data.map(_generate_and_extract, with_indices=True, fn_kwargs=config)


def _generate_and_extract(
        item, 
        idx, 
        idx_range=None, 
        author="", 
        engine="text-davinci-002", 
        temperature=0, 
        max_tokens=128, 
        api_time_interval=1.0, 
        instruction_keys=None, 
        cot_trigger_keys=None, 
        answer_extraction_keys=None,
        debug=True
    ):

    if idx_range is None or (idx >= idx_range[0] and idx < idx_range[1]):
        pass
    else:
        return item

    for instruction_key in instruction_keys:
        for cot_trigger_key in cot_trigger_keys:
            generated_cot = {
                "templates_version": TEMPLATES["version"],
                "instruction": instruction_key,
                "cot-trigger": cot_trigger_key,
                "cot": "",
                "answers": [],
                "author": author,
                "date": "",
                "model": {
                    "name": engine,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                "comment": "",
                "annotation": [],
            }
            template_version, generate_cot_prompt = get_cot_generation_prompt(item, instruction_key, cot_trigger_key)
            print("\n-------------------COT TRIGGER-------------------")
            print(generate_cot_prompt)
            cot = query(generate_cot_prompt, engine, temperature, max_tokens, api_time_interval, debug)
            print("\n------------------GENERATED COT-------------------")
            print(cot)
            generated_cot["cot"] = cot
            generated_cot["date"] = print_now(1)

            for answer_extraction_key in answer_extraction_keys:
                answer = {
                    "answer-extraction": answer_extraction_key,
                    "answer": ""
                }
                _, answer_extraction_prompt = get_answer_extraction_prompt(item, cot, answer_extraction_key)
                print("\n------------------ANSWER EXTRACTION-------------------")
                print(TEMPLATES["answer-extractions"][answer_extraction_key])
                assert (_ == template_version), "Version mismatch cot trigger <-> answer extraction"
                predicted_answer = query(answer_extraction_prompt, engine, temperature, max_tokens, api_time_interval, debug)
                print("\n------------------EXTRACTED ANSWER-------------------")
                print(predicted_answer)
                answer["answer"] = predicted_answer
                generated_cot["answers"].append(answer)
            item["generated_cot"].append(generated_cot)
    return item


def get_cot_generation_prompt(item, instruction_key, cot_trigger_key):
    choices = '\n'.join([f'{chr(65+i)}) {example}' for i, example in enumerate(item['choices'])])
    prompt = TEMPLATES["instructions"][instruction_key] + "\n\n" + item['question'] + "\n" + choices + "\n\n" + TEMPLATES["cot-triggers"][cot_trigger_key]
    return TEMPLATES["version"], prompt

def get_answer_extraction_prompt(item, generated_cot, answer_extraction_key):
    choices = '\n'.join([f'{chr(65+i)}) {example}' for i, example in enumerate(item['choices'])])
    prompt = item['question'] + "\n" + choices + "\n\n" + generated_cot + "\n" + TEMPLATES["answer-extractions"][answer_extraction_key]
    return TEMPLATES["version"], prompt

# ver 0.2
def answer_cleansing(args, pred):

    print("pred_before : " + pred)
    
    if args.method in ("few_shot", "few_shot_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]

    if args.dataset in ("aqua", "commonsensqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    
    print("pred_after : " + pred)
    
    return pred

def query(input, engine, temperature, max_tokens, api_time_interval, debug):
    if debug:
        return "test"
    else:
        # GPT-3 API allows each users execute the API within 60 times in a minute ...
        # time.sleep(1)
        time.sleep(api_time_interval)
        response = openai.Completion.create(
            engine=engine,
            prompt=input,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None
        )
        return response["choices"][0]["text"]




