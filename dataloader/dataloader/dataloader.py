import pathlib
import os
import pandas as pd
from datasets import load_dataset

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

    def load_datasets(self, names=None):
        """
        It takes a list of names, finds the corresponding scripts, and loads the datasets
        
        :param names: A list of dataset names to load. If None, all datasets are loaded
        """
        datasets = self._find_datasets(names)
        for name, script in datasets:
            print(f"Loading {name}")
            self._cache[name] = load_dataset(str(script))

    def dump(self, path_to_directory = "./dump"):
        for name, dataset_dict in self._cache.items():
            for split, data in dataset_dict.items():
                data.to_json(pathlib.Path(path_to_directory) / name / f"{split}.json")

