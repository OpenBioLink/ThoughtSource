# ThoughtSource⚡
__A framework for the science of machine thinking__

ThoughtSource⚡ aims to provide a central, open resource for data and tools related to chain-of-thought (CoT) reasoning in large language models ([Wei 2022](https://arxiv.org/abs/2201.11903)). Our goal is to enable trustworthy and robust reasoning in advance AI systems for driving scientific and technological development.

## Dataloader Usage

1. Clone repository
2. Run `pip install -e ./dataloader`
   
```python
from dataloader import Collection

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
# access specific dataset 
sample = collection["gsm8k"]["train"][0]

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
    features: ['id', 'question_id', 'document_id', 'question', 'type', 'cot_type', 'choices', 'context', 'answer', 'cot', 'feedback', 'cot_after_feedback', 'answer_after_feedback'],
    num_rows: 12430
})
```

## Datasets

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


* __[aqua](https://github.com/deepmind/AQuA):__ Math word problems from the AQUA-RAT (Algebra Question Answering with Rationales) dataset ([Ling 2017](https://arxiv.org/pdf/1705.04146.pdf)).
* __asdiv:__ Math word problems from the Academia Sinica Diverse MWP dataset ([Miao 2020](https://aclanthology.org/2020.acl-main.92/)).
* __[entailment_bank](https://allenai.org/data/entailmentbank):__ Science exam questions with expert-authored explanations from the EntailmentBank dataset([Dalvi 2022](https://arxiv.org/pdf/2104.08661.pdf)).
* __gsm8k:__  Math word problems from the GSM8K dataset ([Cobbe 2021](https://arxiv.org/abs/2110.14168)).
* __mawps:__ Math word problems from MAWPS, the Math Word Problem Repository dataset ([Koncel-Kedziorski 2016](https://aclanthology.org/N16-1136.pdf)).
* __open_book_qa:__ Scientific question-answering modeled after open book exams for assessing human understanding from the OpenBookQA dataset ([Mihaylov 2018](https://aclanthology.org/D18-1260.pdf)).
* __[qed](https://github.com/google-research-datasets/QED):__ General-domain question-answering data from the QED dataset ([Lamm 2020](https://arxiv.org/abs/2009.06354)).
* __[strategy_qa](https://allenai.org/data/strategyqa):__ Source: General-domain question-answering data from the StrategyQA dataset ([Geva 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100680/Did-Aristotle-Use-a-Laptop-A-Question-Answering)).
* __svamp:__ Math word problems. Source: SVAMP ([Patel 2021](https://aclanthology.org/2021.naacl-main.168/))
* __worldtree:__ Scientific question-answering data from the WorldTree v2 dataset ([Xie 2020](https://aclanthology.org/2020.lrec-1.671/))

