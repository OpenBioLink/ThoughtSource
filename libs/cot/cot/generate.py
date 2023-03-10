import datetime
import json
import pkgutil
import time
import uuid
import os
from dataclasses import asdict

import datasets as ds

from cot.config import Config
from cot.utils.schemas.cot import features as cot_features

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
    data.cleanup_cache_files()

    if isinstance(data, ds.arrow_dataset.Dataset):
        features = data.info.features

    elif isinstance(data, ds.dataset_dict.DatasetDict):
        name_of_first_split = list(data.keys())[0]
        features = data[name_of_first_split].info.features

    else:
        raise ValueError("Not recognized data")

    # The config is transformed into a dataclass object, where all testing is done
    # But it will be transformed back to a dictionary for the function 'map'
    config_as_dataclass = Config(**config)

    return data.map(
        _generate_and_extract,
        with_indices=True,
        fn_kwargs=asdict(config_as_dataclass),
        features=features,
        load_from_cache_file=False,
    )


def _generate_and_extract(
    item,
    idx,
    # all of the following variables will be defined by the config_as_dataclass object
    idx_range,
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
    """
    The function takes in a JSON object (item) and generates a CoT (Chain-of-Thought) for each combination of
    of instructions and CoT triggers. For each generated CoT and for each of the given answer extractions it extracts an answer.

    :param item: the item (example) of a dataset to be processed
    :param idx: the index of the item in the dataset
    other parameters are handed over from config and are described in config.py

    :return: item populated with various fields
    """

    if idx_range == "all" or (idx >= idx_range[0] and idx < idx_range[1]):
        pass
    else:
        return item

    # predefine values in template dictionary that stay same over all runs of the current item
    template_dict = {
        "instruction": None,
        "question": item["question"],
        "answer_choices": multiple_choice_answer_formatting(item["choices"]),
        "cot_trigger": None,
        "cot": None,
        "answer_extraction": None,
    }

    # try multiple times in case of API-Error
    additional_api_time = 0
    number_of_tries = 5
    for i in range(0, number_of_tries):  
        try:
            # add additional time to api_time_interval if there was an error
            api_time_interval = api_time_interval + additional_api_time

            # generate chain of thoughts and extract answers
            for instruction_key in instruction_keys:
                template_dict["instruction"] = get_fragments_value("instructions", instruction_key)

                for cot_trigger_key in cot_trigger_keys:
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

                    template_dict["cot_trigger"] = get_fragments_value("cot_triggers", cot_trigger_key)

                    # change template_cot_generation to generated_cot["cot_trigger_template"] to make it more logical
                    generate_cot_prompt = format_prompt(template_cot_generation, template_dict)

                    if verbose:
                        print("\n-----------------COT TRIGGER TEXT-----------------")
                        print(generate_cot_prompt)

                    cot = query_model(
                        generate_cot_prompt,
                        api_service,
                        engine,
                        temperature,
                        max_tokens,
                        api_time_interval,
                    )
                    if verbose:
                        print("\n------------------GENERATED COT-------------------")
                        print(cot)

                    template_dict["cot"] = cot

                    generated_cot["cot"] = cot

                    # deactivated automatic prompt text generation: (code line stays here for testing purposes)
                    # generated_cot["prompt_text"] = generate_cot_prompt

                    generated_cot["date"] = print_now(1)

                    # extract answers from generated chain of thoughts
                    for answer_extraction_key in answer_extraction_keys:
                        if answer_extraction_key is None:
                            pass

                        else:
                            answer = {
                                "id": str(uuid.uuid4()),
                                "answer_extraction": answer_extraction_key,
                                "answer_extraction_template": template_answer_extraction,
                                "answer_extraction_text": "",
                                "answer": "",
                                "correct_answer": None,
                            }

                            template_dict["answer_extraction"] = get_fragments_value("answer_extractions", answer_extraction_key)
                            answer_extraction_prompt = format_prompt(template_answer_extraction, template_dict)

                            if verbose:
                                print("\n----------------ANSWER EXTRACTION TEXT----------------")
                                print(answer_extraction_prompt)

                            predicted_answer = query_model(
                                answer_extraction_prompt,
                                api_service,
                                engine,
                                temperature,
                                max_tokens,
                                api_time_interval,
                            )
                            if verbose:
                                print("\n------------------EXTRACTED ANSWER-------------------")
                                print(predicted_answer)

                            answer["answer"] = predicted_answer

                            # deactivated automatic prompt text generation: (code line stays here for testing purposes)
                            # answer["answer_extraction_text"] = answer_extraction_prompt

                            generated_cot["answers"].append(answer)

                    item["generated_cot"].append(generated_cot)

        except Exception as ex:
            # if last try, raise error
            if i == number_of_tries - 1:
                raise ex

            # if not last try, add additional time to api_time_interval and try again
            additional_api_time += 10
            print("(API-)Error in item " + str(idx) + ": " + str(ex))
            print("Retrying with additional time of " + str(additional_api_time) + " seconds.")
            pass
            
        else:
            break
    
    return item


def full_text_prompts(dataset, prompt_text=True, answer_extraction_text=True):
    assert isinstance(dataset, ds.arrow_dataset.Dataset), "dataset must be an arrow dataset"

    dataset = dataset.map(
        _full_text_prompts,
        fn_kwargs={
            "prompt_text": prompt_text,
            "answer_extraction_text": answer_extraction_text,
        },
        features=dataset.info.features,
        # deleting the cache is necessary in generate if you call it multiple times
        # not clear if it is needed here, but it doesn't hurt
        load_from_cache_file=False,
    )

    return dataset


def _full_text_prompts(item, prompt_text, answer_extraction_text):
    # predefine values in template dictionary that stay same over all runs of the current item
    template_dict = {
        "instruction": None,
        "question": item["question"],
        "cot_trigger": None,
        "cot": None,
        "answer_extraction": None,
    }

    for generated_cot in item["generated_cot"]:
        answer_choices = (multiple_choice_answer_formatting(item["choices"]),)

        # function returns a tuple instead of a string
        # did not find out why it behaves differently here than in the _generate_and_extract function
        if type(answer_choices) == tuple:
            answer_choices = answer_choices[0]

        template_dict["answer_choices"] = answer_choices

        # generate chain of thoughts and extract answers
        # for instruction_key in instruction_keys:
        template_dict["instruction"] = get_fragments_value("instructions", generated_cot["instruction"])

        template_dict["cot_trigger"] = get_fragments_value("cot_triggers", generated_cot["cot_trigger"])

        generate_cot_prompt = format_prompt(generated_cot["cot_trigger_template"], template_dict)

        template_dict["cot"] = generated_cot["cot"]
        # Everything above could also be relevant for the answer extraction

        # now generating the full text for the chain of thoughts
        if prompt_text:
            generated_cot["prompt_text"] = generate_cot_prompt

        # if answer_extraction: ...
        if answer_extraction_text:
            # extract answers from generated chain of thoughts
            for answer in generated_cot["answers"]:
                if answer["answer_extraction"] is None:
                    # if no answer extraction key is given, return item, since cot_prompt text is already generated
                    return item

                else:
                    template_dict["answer_extraction"] = get_fragments_value("answer_extractions", answer["answer_extraction"])
                    answer_extraction_prompt = format_prompt(answer["answer_extraction_template"], template_dict)

                    answer["answer_extraction_text"] = answer_extraction_prompt

    return item


def select_generated_cots(dataset, **kwargs):
    """This function handles which pregenerated CoTs are deleted (can be used after loading a collection with "load_pregenerated_cots=True").
    :param dataset: The dataset to delete unwanted pregenerated CoTs from.
    :param kwargs: A dictionary of the form {"key": value}, where value has to be a string or list of strings.
    e.g. {"author": ["author1", "author2"]} or {"author": "author1"}.

    Overviews of current authors and their cot_triggers:
        "kojima": kojima-01 
        "wei": few-shot (as a prompt)
        "lievin": kojima-01, lievin-01, lievin-02, lievin-03, lievin-10
        "lievin_100": 100 times kojima-01 with high temperature
        "thoughtsource": None, kojima-01
    """
    # general info why this function is necessary:
    # Unfortunately the loading function of the datasets does not let you specify which pregenerated COTS to load
    # So we load all of them and then delete the ones we don't want

    # remove all the pregenerated COTS that are not in the list
    dataset = dataset.map(
        _select_generated_cots,
        fn_kwargs={**kwargs},
        features=dataset.info.features,
        load_from_cache_file=False,
    )
    return dataset

def _select_generated_cots(item, **kwargs):
    # load all allows keys from the cot_features
    allowed_keys = list(cot_features["generated_cot"][0].keys())
    for key, value in kwargs.items():
        # check if key is allowed
        if key not in allowed_keys:
            raise ValueError(f"Key '{key}' not in allowed keys {allowed_keys}")
        # if value is None or a string, convert it to a list
        if value is None or type(value) == str:
            value = [value]
        # loop over all generated CoTs in the item and delete the ones that don't match the given criteria
        item["generated_cot"] = [cot for cot in item["generated_cot"] if cot[str(key)] in value]
    return item

def delete_all_generated_cots(dataset):
    """This function deletes all pregenerated COTS from a dataset."""
    dataset = dataset.map(
        _delete_all_generated_cots,
        features=dataset.info.features,
        load_from_cache_file=False,
    )
    return dataset

def _delete_all_generated_cots(item):
    item["generated_cot"] = []
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


def format_prompt(template, dictionary):
    output = template.format_map(Correct_output(dictionary))
    # remove leading whitespaces
    output = output.lstrip()
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


# def delete_empty_curly_brackets(string):
#     string.replace("{None}\n", "")
#     # string.replace("\n{None}", "") # TODO: do I need this?
#     string.replace("{None}", "")
#     return string


def query_model(input, api_service, engine, temperature, max_tokens, api_time_interval):
    if api_service == "mock_api":
        # time.sleep(api_time_interval)
        return " Test mock chain of thought."
        # return ("This is a " + 20 * "long " + "Mock CoT.\n")*20

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
                    # parameter options: https://beta.openai.com/docs/api-reference/completions/create-completion
                    model_name=engine,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    # type: ignore (suppress pylance error)
                ),
            )

        if api_service == "huggingface_hub":
            from langchain import HuggingFaceHub

            llm_chain = LLMChain(
                prompt=prompt,
                llm=HuggingFaceHub(
                    # parameter options: https://huggingface.co/docs/api-inference/detailed_parameters
                    repo_id=engine,
                    model_kwargs={"temperature": temperature, "max_length": max_tokens},
                    # type: ignore (suppress pylance error)
                ),
            )

        if api_service == "huggingface_endpoint":
            # from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint

            llm_chain = LLMChain(
                prompt=prompt,
                llm=HuggingFaceEndpoint(
                # we just use the engine name as the endpoint url here
                endpoint_url=engine,
                # read API key from environment variable
                huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                model_kwargs={"temperature": temperature, "max_length": max_tokens},
                task="text2text-generation"
                ),
            )

        if api_service == "cohere":
            from langchain import Cohere

            llm_chain = LLMChain(
                prompt=prompt,
                llm=Cohere(
                    model=engine,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    # type: ignore (suppress pylance error)
                ),
            )

        response = llm_chain.predict(prompt=input, stop=None)
        return response
    
### this is code from the langchain package
# I needed to make a small adaptation to the HuggingFaceEndpoint class to catch an Error
# will be deleted in the future

"""Wrapper around HuggingFace APIs."""
from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

VALID_TASKS = ("text2text-generation", "text-generation")


class HuggingFaceEndpoint(LLM, BaseModel):
    """Wrapper around HuggingFaceHub Inference Endpoints.
    To use, you should have the ``huggingface_hub`` python package installed, and the
    environment variable ``HUGGINGFACEHUB_API_TOKEN`` set with your API token, or pass
    it as a named parameter to the constructor.
    Only supports `text-generation` and `text2text-generation` for now.
    Example:
        .. code-block:: python
            from langchain import HuggingFaceEndpoint
            endpoint_url = (
                "https://abcdefghijklmnop.us-east-1.aws.endpoints.huggingface.cloud"
            )
            hf = HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token="my-api-key"
            )
    """

    endpoint_url: str = ""
    """Endpoint URL to use."""
    task: Optional[str] = None
    """Task to call the model with. Should be a task that returns `generated_text`."""
    model_kwargs: Optional[dict] = None
    """Key word arguments to pass to the model."""

    huggingfacehub_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        huggingfacehub_api_token = get_from_dict_or_env(
            values, "huggingfacehub_api_token", "HUGGINGFACEHUB_API_TOKEN"
        )
        try:
            from huggingface_hub.hf_api import HfApi

            try:
                HfApi(
                    endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
                    token=huggingfacehub_api_token,
                ).whoami()
            except Exception as e:
                raise ValueError(
                    "Could not authenticate with huggingface_hub. "
                    "Please check your API token."
                ) from e

        except ImportError:
            raise ValueError(
                "Could not import huggingface_hub python package. "
                "Please it install it with `pip install huggingface_hub`."
            )
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url, "task": self.task},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "huggingface_endpoint"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call out to HuggingFace Hub's inference endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = hf("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}

        # payload samples
        parameter_payload = {"inputs": prompt, "parameters": _model_kwargs}

        # HTTP headers for authorization
        headers = {
            "Authorization": f"Bearer {self.huggingfacehub_api_token}",
            "Content-Type": "application/json",
        }

        # send request
        try:
            response = requests.post(
                self.endpoint_url, headers=headers, json=parameter_payload
            )
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise ValueError(f"Error raised by inference endpoint: {e}")
        generated_text = response.json()
        if "error" in generated_text:
            raise ValueError(f"Error raised by inference API: {generated_text['error']}")
        if self.task == "text-generation":
            # Text generation return includes the starter text.
            text = generated_text[0]["generated_text"][len(prompt) :]
        elif self.task == "text2text-generation":
            text = generated_text[0]["generated_text"]
        else:
            raise ValueError(
                f"Got invalid task {self.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text