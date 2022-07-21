# ThoughtSource
This repository is dedicated to datasets that are converted to our intended COT format.

## Dataloader Usage

1. Clone repository
2. Run `pip install -e ./dataloader`
   
```python
from dataloader import load_datasets
a = load_datasets(["gsm8k", "open_book_qa"])
print(a)
print(a["gsm8k"]["train"][0])
```

## Datasets

The following table represents statistics of datasets that can currently be loaded with the dataloader.

| Name | Train | Validation | Test |
|----|----|----|----|
| aqua | 97467 | 254 | 254 |
| asdiv | 1217 | - | - |
| entailment_bank | 1313 | 187 | 340 |
| gsm8k | 7473 | - | 1319 |
| mawps | 1921 | - | - |
| open_book_qa | 4957 | 500 | 500 |
| qed | 5154 | 1021 | - |
| strategy_qa | 2290 | - | 490 |
| svamp | 1000 | - | - |
| worldtree | 2207 | 496 | 1664 |
