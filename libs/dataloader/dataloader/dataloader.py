import pathlib
import importlib
import json
import os
from typing import Iterable
import pandas as pd
import datasets as ds
import pkgutil

TEMPLATES = json.loads(pkgutil.get_data(__name__, "templates.json"))
print(TEMPLATES)

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

    def dump(self, path_to_directory = "./dump"):
        for name, dataset_dict in self._cache.items():
            for split, data in dataset_dict.items():
                data.to_json(pathlib.Path(path_to_directory) / name / f"{split}.json")

    @property
    def all_train(self):
        return ds.concatenate_datasets([self._cache[name]["train"] for name in self._cache])

    @property
    def all_validation(self):
        return ds.concatenate_datasets([self._cache[name]["validation"] for name in self._cache if "validation" in self._cache[name]])

    @property
    def all_test(self):
        return ds.concatenate_datasets([self._cache[name]["test"] for name in self._cache if "test" in self._cache[name]])


def apply_templates(item, answer_extraction=False, instruction_keys=None, cot_trigger_keys=None, answer_extraction_keys=None):

    if not instruction_keys:
        instruction_keys = TEMPLATES["instructions"].keys()
    if not cot_trigger_keys:
        cot_trigger_keys = TEMPLATES["cot-triggers"].keys()
    if not answer_extraction_keys:
        answer_extraction_keys = TEMPLATES["answer-extractions"].keys()

    prompts = {}
    if not answer_extraction:
        for instruction_key in instruction_keys:
            for cot_trigger_key in cot_trigger_keys: 
                choices = '\n'.join([f'{chr(65+i)}) {example}' for i, example in enumerate(item['choices'])])
                prompt = TEMPLATES["instructions"][instruction_key] + "\n\n" + item['question'] + "\n" + choices + "\n\n" + TEMPLATES["cot-triggers"][cot_trigger_key]
                prompts[(f"instruction-{instruction_key}", f"cot-triggers-{cot_trigger_key}")] = prompt
    else:
        for answer_extraction_key in answer_extraction_keys:
            choices = '\n'.join([f'{chr(65+i)}) {example}' for i, example in enumerate(item['choices'])])
            cot = "\n".join(item["cot"])
            prompt = item['question'] + "\n" + choices + "\n\n" + cot + "\n" + TEMPLATES["answer-extractions"][answer_extraction_key]
            prompts[f"answer-extraction-{answer_extraction_key}"] = prompt

    return prompts

