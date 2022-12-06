# ThoughtSource⚡ Dataloader

Library for efficient retrieval and processing of ThoughtSource datasets. We provide code to build dataset objects in the [🤗 Datasets format](https://huggingface.co/docs/datasets/index)

## Installation

1. Clone repository
2. Run `pip install -e ./libs/cot`
   
## Usage examples

```python
from cot import Collection

# load all available datasets
collection = Collection("all")

# create an empty collection
collection = Collection()

# load only selected datasets
collection = Collection(["gsm8k", "open_book_qa"])

print(collection)
```
```batch
| Name         |   Train | Valid   |   Test |
|--------------|---------|---------|--------|
| gsm8k        |    7473 | -       |   1319 |
| open_book_qa |    4957 | 500     |    500 |

Not loaded: ['aqua', 'asdiv', 'commonsense_qa', 'entailment_bank', 'mawps', 'qed', 'strategy_qa', 'svamp', 'worldtree']
```
```python
# datasets not found in the current collection are loaded on the fly
sample = collection["commonsense_qa"]["train"][0]

# or can be loaded explicitly with
collection.load_datasets(["commonsense_qa"])

# Single datasets can be unloaded
collection.unload_datasets(["commonsense_qa"])

# Or the whole collection can be cleared
collection.clear()

# iterate over datasets
for name, data in collection:
  pass

# dump all loaded dataset as json
collection.dump()

# concatenates training sets of loaded datasets
# see also all_test, all_validation
print(collection.all_train)
```
```text
Dataset({
    features: ['id', 'ref_id', 'question', 'type', 'cot_type', 'choices', 'context', 'answer', 'cot', 'feedback', 'cot_after_feedback', 'answer_after_feedback'],
    num_rows: 12430
})
```

## Statistics

The following table represents statistics of datasets that can currently be loaded with the dataloader.

| Name            |   Train | Valid   | Test   |
|-----------------|---------|---------|--------|
| aqua            |   97467 | 254     | 254    |
| asdiv           |    1217 | -       | -      |
| commonsense_qa  |    9741 | 1221    | 1140   |
| entailment_bank |    1313 | 187     | 340    |
| gsm8k           |    7473 | -       | 1319   |
| mawps           |    1921 | -       | -      |
| open_book_qa    |    4957 | 500     | 500    |
| qed             |    5154 | 1021    | -      |
| strategy_qa     |    2290 | -       | 490    |
| svamp           |    1000 | -       | -      |
| worldtree       |    2207 | 496     | 1664   |
