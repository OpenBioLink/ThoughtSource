import json
from re import X
from datasets import DownloadConfig, DownloadMode, load_dataset
dsd = load_dataset("thoughtsource/datasets/strategy_qa/strategy_qa.py")

print(dsd)
print(dsd["train"][0])

"""
from thoughtsource import load_datasets

a = load_datasets(["qed"])
print(a)
"""