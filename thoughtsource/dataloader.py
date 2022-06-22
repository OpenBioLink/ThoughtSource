import pathlib
import os
from datasets import load_dataset

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
    
    datasets = {}
    for name, scripts in dataloader_scripts:
        print(f"Loading {name}")
        datasets[name] = load_dataset(str(scripts))
    return datasets