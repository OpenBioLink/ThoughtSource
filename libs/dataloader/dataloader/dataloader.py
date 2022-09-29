import pathlib
import importlib
import io
import json
import os
from typing import Iterable
import pandas as pd
import datasets as ds
import pkgutil
from pathlib import Path
from collections import defaultdict

class Collection:

    def __init__(self, names=None):
        self._cache = {}
        if names == "all":
            self.load_datasets()
        elif isinstance(names, list):
            self.load_datasets(names)

    def __getitem__(self, key):
        if key not in self._cache:
            self.load_datasets(names=[key])
        return self._cache[key]
  
    def __setitem__(self, key, newvalue):
        self._cache[key] = newvalue

    def __iter__(self):
        yield from self._cache.items()

    def __repr__(self):
        data = [
            (
                name, 
                self._cache[name]['train'].num_rows,
                self._cache[name]['validation'].num_rows if 'validation' in self._cache[name] else '-',
                self._cache[name]['test'].num_rows if 'test' in self._cache[name] else '-'
            )
            for name in self._cache.keys()
        ]
        table = pd.DataFrame.from_records(data, columns=["Name", "Train", "Valid", "Test"])
        table = table.to_markdown(index=False, tablefmt="github")
        not_loaded = [name for name, _ in self._find_datasets() if name not in self._cache]
        return table + "\n\nNot loaded: " + str(not_loaded)

    def _find_datasets(self, names=None):
        path_to_biodatasets = (pathlib.Path(__file__).parent.absolute() / "datasets").resolve()
        if names is None:
            dataloader_scripts = sorted(
                path_to_biodatasets.glob(os.path.join("*", "*.py"))
            )
            dataloader_scripts = [
                (el.name.replace(".py", ""), el) for el in dataloader_scripts if el.name != "__init__.py"
            ]
        else:
            dataloader_scripts = [
                (name, path_to_biodatasets / name / (name + ".py")) for name in names
            ]
        return dataloader_scripts

    def _get_metadata(self):
        for name, script_path in self._find_datasets():
            spec = importlib.util.spec_from_file_location("foo", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            break

    def load_datasets(self, names=None):
        """
        It takes a list of names, finds the corresponding scripts, and loads the datasets
        
        :param names: A list of dataset names to load. If None, all datasets are loaded
        """
        datasets = self._find_datasets(names)
        for name, script in datasets:
            print(f"Loading {name}")
            self._cache[name] = ds.load_dataset(str(script))

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

    """
    def dump(self, path_to_directory = "./dump", single_file=False):
        if single_file:
            d_dict = defaultdict(dict)
            for name, dataset_dict in self._cache.items():
                for split, data in dataset_dict.items():
                    data_stream = io.BytesIO()
                    data.to_json(data_stream)
                    data_stream.seek(0)
                    d_dict[name][split] = [json.loads(x.decode()) for x in data_stream.readlines()]

            with open(f"all.json", "w") as outfile:
                # use json library to prettify output
                json.dump(d_dict, outfile, indent = 4)
        else:
            for name, dataset_dict in self._cache.items():
                for split, data in dataset_dict.items():
                    data.to_json(pathlib.Path(path_to_directory) / name / f"{split}.json")
    """

    def save_to_disk(self, path_to_directory = "datasets"):
        for name, dataset_dict in self._cache.items():
            dataset_dict.save_to_disk(f"{path_to_directory}/{name}")

    def load_from_disk(self, path_to_directory = "datasets"):
        for name in next(os.walk(path_to_directory))[1]:
            self._cache[name] = ds.load_from_disk(os.path.join(path_to_directory, name))





    @property
    def all_train(self):
        return ds.concatenate_datasets([self._cache[name]["train"] for name in self._cache])

    @property
    def all_validation(self):
        return ds.concatenate_datasets([self._cache[name]["validation"] for name in self._cache if "validation" in self._cache[name]])

    @property
    def all_test(self):
        return ds.concatenate_datasets([self._cache[name]["test"] for name in self._cache if "test" in self._cache[name]])

