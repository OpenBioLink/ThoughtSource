import importlib
import io
import json
import os
import pathlib
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

import datasets as ds
import pandas as pd


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# Collection is a class that represents a collection of datasets.
class Collection:
    def __init__(
        self, names=None, verbose=True, download_mode="reuse_dataset_if_exists"
    ):
        """
        The function takes in a list of names and a boolean value. If the boolean value is true, it will
        print out the progress of the function. If the boolean value is false, it will not print out the
        progress of the function. If the list of names is "all", it will load all the datasets. If the
        list of names is a list, it will load the datasets in the list.

        :param names: List of dataset names to load. If None, load no dataset. If "all", load all
        datasets
        :param verbose: If True, prints out the name of the dataset as it is being loaded, defaults to
        True (optional)
        :param download_mode: "reuse_dataset_if_exists" (default), "reuse_cache_if_exists", "force_redownload"
        see https://huggingface.co/docs/datasets/v2.1.0/en/package_reference/builder_classes#datasets.DownloadMode
        """
        self.verbose = verbose
        self.download_mode = download_mode
        if not verbose:
            ds.disable_progress_bar()
        self._cache = {}
        if names == "all":
            self.load_datasets()
        elif isinstance(names, list):
            self.load_datasets(names)

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
                self._cache[name]["train"].num_rows
                if "train" in self._cache[name]
                else "-",
                self._cache[name]["validation"].num_rows
                if "validation" in self._cache[name]
                else "-",
                self._cache[name]["test"].num_rows
                if "test" in self._cache[name]
                else "-",
            )
            for name in self._cache.keys()
        ]
        table = pd.DataFrame.from_records(
            data, columns=["Name", "Train", "Valid", "Test"]
        )
        table = table.to_markdown(index=False, tablefmt="github")
        not_loaded = [
            name for name, _ in Collection._find_datasets() if name not in self._cache
        ]
        return table + "\n\nNot loaded: " + str(not_loaded)

    @staticmethod
    def _find_datasets(names=None):
        path_to_biodatasets = (
            pathlib.Path(__file__).parent.absolute() / "datasets"
        ).resolve()
        if names is None:
            dataloader_scripts = sorted(
                path_to_biodatasets.glob(os.path.join("*", "*.py"))
            )
            dataloader_scripts = [
                (el.name.replace(".py", ""), el)
                for el in dataloader_scripts
                if el.name != "__init__.py"
            ]
        else:
            dataloader_scripts = [
                (name, path_to_biodatasets / name / (name + ".py")) for name in names
            ]
        return dataloader_scripts

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
                    str(script), download_mode=self.download_mode
                )
            else:
                with suppress_stdout_stderr():
                    self._cache[name] = ds.load_dataset(
                        str(script), download_mode=self.download_mode
                    )

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

    def dump(self, path_to_file_or_directory="./dump", single_file=False):
        if single_file:
            d_dict = defaultdict(dict)
            for name, dataset_dict in self._cache.items():
                for split, data in dataset_dict.items():
                    data_stream = io.BytesIO()
                    data.to_json(data_stream)
                    data_stream.seek(0)
                    d_dict[name][split] = [
                        json.loads(x.decode()) for x in data_stream.readlines()
                    ]

            if not path_to_file_or_directory.endswith(".json"):
                path_to_file_or_directory = path_to_file_or_directory + ".json"

            with open(path_to_file_or_directory, "w") as outfile:
                # use json library to prettify output
                json.dump(d_dict, outfile, indent=4)
        else:
            for name, dataset_dict in self._cache.items():
                for split, data in dataset_dict.items():
                    data.to_json(
                        pathlib.Path(path_to_file_or_directory) / name / f"{split}.json"
                    )

    def _replace(batch, data):
        return data

    # pretty dirty loading function which presevers metadata (load and replace data :/ )
    # metadata needed?
    @staticmethod
    def from_json(
        path_to_json, single_file=True, download_mode="reuse_dataset_if_exists"
    ):
        if single_file:
            with open(path_to_json, "r") as infile:
                content = json.load(infile)

            scripts = {
                x[0]: x[1]
                for x in Collection._find_datasets(names=list(content.keys()))
            }

            collection = Collection()
            for dataset_name in content.keys():
                info = ds.load_dataset_builder(
                    str(scripts[dataset_name]), download_mode=download_mode
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

                    dic = pd.DataFrame.from_records(
                        content[dataset_name][split]
                    ).to_dict("series")
                    dic = {k: list(v) for (k, v) in dic.items()}
                    dataset_dict[split_name] = ds.Dataset.from_dict(
                        dic, info.features, info, split
                    )
                collection[dataset_name] = ds.DatasetDict(dataset_dict)
            return collection
        else:
            # TODO add ability to load directory dump (single_file=False)??
            raise NotImplementedError

    # Deprecated
    def save_to_disk(self, path_to_directory="datasets"):
        for name, dataset_dict in self._cache.items():
            dataset_dict.save_to_disk(f"{path_to_directory}/{name}")

    # Deprecated
    def load_from_disk(self, path_to_directory="datasets"):
        for name in next(os.walk(path_to_directory))[1]:
            self._cache[name] = ds.load_from_disk(os.path.join(path_to_directory, name))

    @property
    def all_train(self):
        """
        It takes the training sets all the datasets in the cache and concatenates them into one big dataset
        :return: A concatenated dataset of all the training data.
        """
        return ds.concatenate_datasets(
            [self._cache[name]["train"] for name in self._cache]
        )

    @property
    def all_validation(self):
        """
        It takes the validation sets all the datasets in the cache and concatenates them into one big dataset
        :return: A concatenated dataset of all the validation data.
        """
        return ds.concatenate_datasets(
            [
                self._cache[name]["validation"]
                for name in self._cache
                if "validation" in self._cache[name]
            ]
        )

    @property
    def all_test(self):
        """
        It takes the testing sets all the datasets in the cache and concatenates them into one big dataset
        :return: A concatenated dataset of all the testing data.
        """
        return ds.concatenate_datasets(
            [
                self._cache[name]["test"]
                for name in self._cache
                if "test" in self._cache[name]
            ]
        )
