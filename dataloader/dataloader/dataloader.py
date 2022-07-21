import pathlib
import os
from datasets import load_dataset

class Collection:

    def __init__(self):
        self._cache = {}

    def __getitem__(self, key):
        return self._cache[key]
  
    def __setitem__(self, key, newvalue):
        self._cache[key] = newvalue

    @property
    def statistics(self):
        table = "| Name | Train | Validation | Test |\n"
        table += "|----|----|----|----|\n"

        for name in self._cache.keys():
            table = table + f"| {name} | {self._cache[name]['train'].num_rows} | {self._cache[name]['validation'].num_rows if 'validation' in self._cache[name] else '-'} | {self._cache[name]['test'].num_rows if 'test' in self._cache[name] else '-'} |" + "\n"
        return table




def load_datasets(names=None):
    path_to_here = pathlib.Path(__file__).parent.absolute()
    path_to_biodatasets = (path_to_here / "datasets").resolve()
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
    
    datasets = Collection()
    for name, scripts in dataloader_scripts:
        print(f"Loading {name}")
        datasets[name] = load_dataset(str(scripts))
    return datasets