import glob
import importlib
import io
import json
import os
import pathlib
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

import datasets as ds
import pandas as pd

from .evaluate import evaluate
from .generate import (full_text_prompts, generate_and_extract,
                       select_generated_cots, delete_all_generated_cots)
from .merge import merge
from .new_generate import *


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# Collection is a class that represents a collection of datasets.
class Collection:
    def __init__(self, names=None, verbose=True, generate_mode=None, source=False, load_pregenerated_cots=False):
        """
        The function takes in a list of names and a boolean value. If the boolean value is true, it will
        print out the progress of the function. If the boolean value is false, it will not print out the
        progress of the function. If the list of names is "all", it will load all the datasets. If the
        list of names is a list, it will load the datasets in the list.

        :param names: List of dataset names to load. (aqua, asdiv, commonsense_qa, entailment_bank, 
        gsm8k, mawps, med_qa, medmc_qa, open_book_qa, pubmed_qa, qed, strategy_qa, svamp, worldtree).
        If None, create empty Collection. If "all", load all datasets.
        If you want to load the collection thoughtsource_100, use the method Collection.load_thoughtsource_100().
        :param verbose: If True, prints out the name of the dataset as it is being loaded, defaults to
        True (optional)
        :param generate_mode:
        - if "redownload": deletes download and dataset caches, redownloads all sources and regenerates all datasets.
        Try this if datasets give unexplainable KeyErrors, ...
        - if "recache": deletes dataset caches and regenerates all datasets
        - if None: reuse cached dataset
        :param source: If true, loads all datasets in source view (their original form)
        :param load_pregenerated_cots: decides if generated CoTs are loaded. If False, load no generated CoTs. 
        If True, load all generated CoTs. Defaults to True. Parameter source must be False.
        Selection of specific generated CoTs can be done after loading by select_generated_cots().
        """
        self.verbose = verbose
        self.download_mode = None
        self.load_source = source

        if load_pregenerated_cots is True and source is True:
            raise ValueError(
                "load_pregenerated_cots only works if datasets are loaded in ThoughSource view. \
                Param source needs to be False for pregenerated CoTs to be loaded."
            )
        
        # if dataset name is a string, convert to list
        if isinstance(names, str) and names != "all":
            names = [names]
        # test if dataset name is valid
        if names is not None and names != "all":
            for name in names:
                available_datasets = Collection._all_available_datasets()
                if name not in available_datasets:
                    raise ValueError(
                        f"""Dataset '{name}' not found. Please check spelling.
                        Available datasets: {available_datasets}"""
                        )

        if generate_mode in ["redownload", "recache"]:
            # see https://huggingface.co/docs/datasets/v2.1.0/en/package_reference/builder_classes#datasets.DownloadMode
            self.download_mode = "reuse_cache_if_exists"
            if names == "all":
                # delete datasets cache
                for dataset_folder in glob.glob(
                    os.path.join(ds.config.HF_DATASETS_CACHE, "*_dataset", "source" if self.load_source else "thoughtsource")
                ):
                    shutil.rmtree(dataset_folder)
            else:
                for name in names:
                    path = os.path.join(ds.config.HF_DATASETS_CACHE, f"{name}_dataset", "source" if self.load_source else "thoughtsource")
                    if os.path.exists(path):
                        shutil.rmtree(path)
            if generate_mode == "redownload":
                shutil.rmtree(os.path.join(ds.config.HF_DATASETS_CACHE, "downloads"))
                self.download_mode = "force_redownload"
        if not verbose:
            ds.disable_progress_bar()
        else:
            ds.enable_progress_bar()
        self._cache = {}
        if names == "all":
            self.load_datasets()
        elif isinstance(names, list):
            self.load_datasets(names)

        # unfortunately all generated cots have to be loaded when loading datasets in ThoughtSource view
        # here: all or None, selection of specific generated cots can be done later with select_generated_cots
        if not load_pregenerated_cots and not source:
            self.delete_all_generated_cots()

    def __getitem__(self, key):
        """
        Returns a dataset. If the key is not in the cache, load the dataset.

        :param key: The name of the dataset to load
        :return: The dataset is being returned.
        """
        if key not in self._cache:
            self.load_datasets(names=[key])
        return self._cache[key]

    def __setitem__(self, key, dataset):
        """
        The function takes in a key and a dataset and sets the key to the dataset.

        :param key: The key to store the dataset under
        :param dataset: The dataset to be stored
        """
        self._cache[key] = dataset

    def __iter__(self):
        """
        The function is a generator that yields the loaded datasets as tuples (name, data).
        """
        yield from self._cache.items()

    def __len__(self):
        """
        The function returns the number of loaded datasets.
        :return: The number of loaded datasets.
        """
        return len(self._cache)

    def __repr__(self):
        data = [
            (
                name,
                self._cache[name]["train"].num_rows if "train" in self._cache[name] else "-",
                self._cache[name]["validation"].num_rows if "validation" in self._cache[name] else "-",
                self._cache[name]["test"].num_rows if "test" in self._cache[name] else "-",
            )
            for name in self._cache.keys()
        ]
        table = pd.DataFrame.from_records(data, columns=["Name", "Train", "Valid", "Test"])
        table = table.to_markdown(index=False, tablefmt="github")
        not_loaded = [name for name, _ in Collection._find_datasets() if name not in self._cache]
        return table + "\n\nNot loaded: " + str(not_loaded)

    @staticmethod
    def _find_datasets(names=None):
        path_to_biodatasets = (pathlib.Path(__file__).parent.absolute() / "datasets").resolve()
        if names is None:
            dataloader_scripts = sorted(path_to_biodatasets.glob(os.path.join("*", "*.py")))
            dataloader_scripts = [(el.name.replace(".py", ""), el) for el in dataloader_scripts if el.name != "__init__.py"]
        else:
            dataloader_scripts = [(name, path_to_biodatasets / name / (name + ".py")) for name in names]
        return dataloader_scripts
    
    @staticmethod
    def _all_available_datasets():
        return [name for name, _ in Collection._find_datasets()]

    def _get_metadata(self):
        for name, script_path in Collection._find_datasets():
            spec = importlib.util.spec_from_file_location("foo", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            break

    def load_datasets(self, names=None):
        """
        It takes a list of names, finds the corresponding scripts, and loads the datasets

        :param names: A list of dataset names to load. If None, all datasets are loaded
        """
        datasets = Collection._find_datasets(names)
        for name, script in datasets:
            print(f"Loading {name}...")
            if self.verbose:
                self._cache[name] = ds.load_dataset(
                    str(script), name="source" if self.load_source else "thoughtsource", download_mode=self.download_mode
                )
            else:
                with suppress_stdout_stderr():
                    self._cache[name] = ds.load_dataset(
                        str(script), name="source" if self.load_source else "thoughtsource", download_mode=self.download_mode
                    )

    def select_generated_cots(self, *args, **kwargs):
        """Decides which generated cots to keep after loading the datasets"""
        # just apply it to all of the datasets and splits, no specific name or split
        for name in self._cache:
            for split in self._cache[name]:
                self[name][split] = select_generated_cots(self[name][split],*args, **kwargs)

        # specific name or split could maybe solved by setting: if "name" and "split" in kwargs...
        # for now it is good enough, no need to specify the name and split

    def delete_all_generated_cots(self):
        """Deletes all generated cots from the datasets"""
        for name in self._cache:
            for split in self._cache[name]:
                self[name][split] = delete_all_generated_cots(self[name][split])

    def unload_datasets(self, names=None):
        """
        It takes a list of names and unloads the datasets

        :param names: A list of dataset names to load. If None, all datasets are unloaded
        """
        if names is None:
            self._cache.clear()
        else:
            for name in names:
                if name in self._cache:
                    del self._cache[name]

    def clear(self):
        self.unload_datasets()

    def dump(self, path_to_file_or_directory="./dump.json"):
        if not path_to_file_or_directory.endswith(".json"):
            path_to_file_or_directory = path_to_file_or_directory + ".json"
        with open(path_to_file_or_directory, "w") as outfile:
            # use json library to prettify output
            json.dump(self.to_json(), outfile, indent=4)

    def to_json(self):
        d_dict = defaultdict(dict)
        for name in self._cache:
            for split in self._cache[name]:
                d_dict[name][split] = self._dataset_to_json(self._cache[name][split])
        return d_dict

    def _dataset_to_json(self, data):
        data_stream = io.BytesIO()
        data.to_json(data_stream)
        data_stream.seek(0)
        return [json.loads(x.decode()) for x in data_stream.readlines()]

    def copy(self):
        import copy
        return copy.deepcopy(self)

    # should raise an error if it is called on an instance
    # make this a classmethod? (same for load_thoughtsource_100)
    @staticmethod
    def from_json(path_or_json, download_mode="reuse_dataset_if_exists", source=False):
        if isinstance(path_or_json, str):
            with open(path_or_json, "r") as infile:
                content = json.load(infile)
        elif isinstance(path_or_json, dict):
            content = path_or_json

        scripts = {x[0]: x[1] for x in Collection._find_datasets(names=list(content.keys()))}

        collection = Collection()
        for dataset_name in content.keys():
            info = ds.load_dataset_builder(
                str(scripts[dataset_name]), name="source" if source else "thoughtsource", download_mode=download_mode
            ).info
            dataset_dict = dict()
            for split_name in content[dataset_name].keys():
                split = None
                if split_name == "train":
                    split = ds.Split.TRAIN
                elif split_name == "validation":
                    split = ds.Split.VALIDATION
                elif split_name == "test":
                    split = ds.Split.TEST

                dic = pd.DataFrame.from_records(content[dataset_name][split]).to_dict("series")
                dic = {k: list(v) for (k, v) in dic.items()}
                # important: after annotation on the annotator website, the dataset is saved with the following keys:
                # 'lengths', 'sentences', 'subsetType', which have to be deleted before loading the dataset
                for key in ['lengths', 'sentences', 'subsetType']:
                    if key in dic:
                        del dic[key]

                dataset_dict[split_name] = ds.Dataset.from_dict(dic, info.features, info, split)
            collection[dataset_name] = ds.DatasetDict(dataset_dict)
        return collection

    @staticmethod
    def load_thoughtsource_100(names="all", load_pregenerated_cots=True) -> "Collection":
        """load the thoughtsource_100 dataset"""
        path_to_biodatasets = (pathlib.Path(__file__).parent.absolute() / "datasets").resolve()
        path_to_thoughtsource_100 = path_to_biodatasets / "thoughtsource" / "thoughtsource_100.json"
        collection = Collection.from_json(str(path_to_thoughtsource_100))
        # drop all names that are not in the list
        if names != "all":
            all_names = list(collection._cache.keys())
            names_to_remove = [name for name in all_names if name not in names]
            collection.unload_datasets(names_to_remove)
        # drop all generated cots if load_pregenerated_cots is False
        if not load_pregenerated_cots:
            collection.delete_all_generated_cots()
        return collection

    def number_examples(self, name=None, split=None):
        """
        The function returns the number of examples in the loaded datasets.
        :return: The number of examples in the loaded datasets.
        """
        # We moved the computing of the number of samples from the generate function, because we need it here
        # But did not consider the idx_range option there
        # this does not yet include if specific indices are selected, not sure if that is still needed
        # here is an example of how it could be done:
        
        # if "idx_range" in config and config["idx_range"] != "all":
        #     n_samples = config["idx_range"][1] - config["idx_range"][0]
        # else:
        #     n_samples = len(data)

        count = 0
        if name is None:
            for name in self._cache.keys():
                if split is None:
                    # need to use current_split because split is a reserved word
                    for current_split in self._cache[name].keys():
                        count += self._cache[name][current_split].num_rows
                else:
                    count += self._cache[name][split].num_rows
        else:
            if split is None:
                for current_split in self._cache[name].keys():
                    count += self._cache[name][current_split].num_rows
            else:
                count += self._cache[name][split].num_rows
        return count

    def generate(self, name=None, split=None, config={}):
        if ("warn" not in config or config["warn"]) and not config["api_service"] == "mock_api":
            # more detailed option, but not necessary:
            # if ("warn" not in config or config["warn"]) and (("api_service" in config and not config["api_service"]=="mock_api") ...
            # ... or "api_service" not in config):
            n_samples = self.number_examples(name, split)
            print_warning(config, n_samples)
        # always do it per split, so less data is lost in case of an API error causing a crash
        if name is None:
            for name in self._cache:
                print(f"Generating {name}...")
                for split in self._cache[name]:
                    self[name][split] = generate_and_extract(self[name][split], config=config)
        else:
            if split is None:
                print(f"Generating {name}...")
                for split in self._cache[name]:
                    self[name][split] = generate_and_extract(self[name][split], config=config)
            else:
                print(f"Generating {name}...")
                self[name][split] = generate_and_extract(self[name][split], config=config)

    def generate_flexible(self,chain, input_dict,name=None, split=None):
        if name is None:
            for name in self._cache:
                print(f"Generating {name}...")
                return self_generate(self[name], chain, input_dict)
        else:
            if split is None:
                print(f"Generating {name}...")
                return self_generate(self[name], chain, input_dict)
            else:
                print(f"Generating {name}...")
                return self_generate(self[name][split], chain, input_dict)

    
    def evaluate(self, name=None, split=None, overwrite=False, warn=False):
        evaluations_dict = defaultdict(dict)
        if name is None:
            for name in self._cache:
                for split in self._cache[name]:
                    # print(f"Evaluating {name}...")
                    self[name][split], evaluation = evaluate(self[name][split], overwrite=overwrite, warn=warn)
                    evaluations_dict[name][split] = evaluation
        else:
            if split is None:
                for split in self._cache[name]:
                    # print(f"Evaluating {name}...")
                    self[name][split], evaluations = evaluate(self[name][split], overwrite=overwrite, warn=warn)
                    evaluations_dict[name][split] = evaluations

            else:
                # print(f"Evaluating {name}...")
                self[name][split], evaluations = evaluate(self[name][split], overwrite=overwrite, warn=warn)
                evaluations_dict[name][split] = evaluations

        # return evaluation outcome
        return dict(evaluations_dict)

    def full_text_prompts(self, name=None, split=None, prompt_text=True, answer_extraction_text=True):
        if name is None:
            for name in self._cache:
                for split in self._cache[name]:
                    self[name][split] = full_text_prompts(
                        self[name][split], prompt_text=prompt_text, answer_extraction_text=answer_extraction_text
                    )
        else:
            if split is None:
                for split in self._cache[name]:
                    self[name][split] = full_text_prompts(
                        self[name][split], prompt_text=prompt_text, answer_extraction_text=answer_extraction_text
                    )
            else:
                self[name][split] = full_text_prompts(
                    self[name][split], prompt_text=prompt_text, answer_extraction_text=answer_extraction_text
                )

    def merge(self, collection_other):
        return merge(self, collection_other)

    def select(self, split="train", number_samples=None, random_samples=True, seed=0):
        """
        The function takes in a collection and returns a split (train,test,validation) of the collection.
        It can also give back a part of the split, random or first number of entries.
        :param collection: the collection (of datasets) to be processed
        :param split: the split (train,test,validation) to be selected. Defaults: "train".
        :param number_samples: how many samples to select from the split. Default: "None" (all samples of the split)
        :param random: if the number_samples are selected randomly or as the first entries of the dataset.
            Default: "True" (random selection)
        :param seed: when random selection is used, whether to use it with seed to make it reproducible.
            If None no seed. If integer: seed. Default: "0" (same random collection over multiple runs)
        """
        import copy
        import random

        # if split is a string, convert to list
        if type(split) is str:
            split = [split]

        sampled_collection = copy.deepcopy(self)
        for dataset in sampled_collection:
            _, dataset_dict = dataset
            # if "all", select all available splits
            if split == ["all"]:
                split_list = list(dataset_dict.keys())
            else:
            # else, just select the stated ones
                split_list = split
            for current_split in list(dataset_dict.keys()):
                # if the dataset does not include the current_split, no selection needed
                if current_split not in split_list:
                    dataset_dict.pop(current_split)
                    continue
                subset = copy.deepcopy(dataset_dict[current_split])
                # # select the whole split, without specified number of samples
                # if not number_samples:
                #     pass
                # select a certain number of samples
                if number_samples:
                    # get number of samples in subset
                    samples_count = subset.num_rows
                    # random sample
                    if random_samples:
                        # set seed for reproducibility
                        if type(seed) is int:
                            random.seed(seed)
                        elif seed is True:
                            # setting the same seed as the default
                            random.seed(0)
                        random_ids = random.sample(range(samples_count), number_samples)
                        # sort ids
                        random_ids = sorted(random_ids)
                        # random sample from subset
                        subset = subset.select(random_ids)
                    # first rows of dataset, not random
                    else:
                        subset = subset.select(range(0, number_samples))
                # reinsert selected samples
                dataset_dict[current_split] = subset

        return sampled_collection

    @property
    def loaded(self):
        return list(self._cache.keys())

    @property
    def all_train(self):
        """
        It takes the training sets all the datasets in the cache and concatenates them into one big dataset
        :return: A concatenated dataset of all the training data.
        """
        return ds.concatenate_datasets([self._cache[name]["train"] for name in self._cache])

    @property
    def all_validation(self):
        """
        It takes the validation sets all the datasets in the cache and concatenates them into one big dataset
        :return: A concatenated dataset of all the validation data.
        """
        return ds.concatenate_datasets([self._cache[name]["validation"] for name in self._cache if "validation" in self._cache[name]])

    @property
    def all_test(self):
        """
        It takes the testing sets all the datasets in the cache and concatenates them into one big dataset
        :return: A concatenated dataset of all the testing data.
        """
        return ds.concatenate_datasets([self._cache[name]["test"] for name in self._cache if "test" in self._cache[name]])

def print_warning(config, n_samples):
    n_instruction_keys = len(config["instruction_keys"]) if "instruction_keys" in config else 1
    n_cot_trigger_keys = len(config["cot_trigger_keys"]) if "cot_trigger_keys" in config else 1
    n_answer_extraction_keys = len(config["answer_extraction_keys"]) if "answer_extraction_keys" in config else 1

    n_total = (
        n_samples * n_instruction_keys * n_cot_trigger_keys + n_samples * n_instruction_keys * n_cot_trigger_keys * n_answer_extraction_keys
    )
    warning = f"""
        You are about to \033[1m call an external API \033[0m in total {n_total} times, which \033[1m may produce costs \033[0m.
        API calls for reasoning chain generation: {n_samples} samples  * {n_instruction_keys} instructions  * {n_cot_trigger_keys} reasoning chain triggers
        API calls for answer extraction: n_samples  {n_samples} samples  * {n_instruction_keys} instructions  * {n_cot_trigger_keys} reasoning chain triggers * {n_answer_extraction_keys} answer extraction triggers
        Do you want to continue? y/n
        """
    if config["api_service"] == "mock_api":
        warning += "\033[1m Note: You are using a mock api. When entering 'y', a test run without API calls is made. \033[0m"
    print(warning)
    time.sleep(1)
    ans = input()
    if ans.lower() == "y":
        pass
    else:
        # break the execution of the code if the user does not want to continue
        raise ValueError("Generation aborted by user.")
