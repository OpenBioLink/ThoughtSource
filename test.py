"""
from datasets import load_dataset
dsd = load_dataset("thoughtsource/datasets/open_book_qa/open_book_qa.py")

print(dsd)
print(dsd["train"][0])
"""

from thoughtsource import load_datasets

a = load_datasets(["gsm8k", "open_book_qa"])
print(a)