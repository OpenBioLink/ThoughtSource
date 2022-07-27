# ThoughtSource⚡
__A framework for the science of machine thinking__

We aim to build a central, open resource and community around data and tools related to _chain-of-thought reasoning_ in large language models ([Wei 2022](https://arxiv.org/abs/2201.11903)). Our long-term goal is to enable trustworthy and robust reasoning in advanced AI systems for driving scientific research and development.

## Roadmap

1. Create a central repository of chain-of-thought (CoT) datasets converted to a standardized format. ✅
2. Create a conceptual model of different CoT reasoning styles and phenomena.
3. Create tools for exploring, diagnosing and evaluating CoT reasoning (based on performance, transparency and value-alignment).
4. Unify CoT reasoning with learning from natural language feedback ([Scheurer 2022](https://arxiv.org/abs/2204.14146)) and argumentation/debate.
5. Adapt CoT approaches to high-impact scientific use-cases such as biomedical research.

## Framework

### Applications

* __[dataset-viewer](./apps/dataset-viewer/):__ Streamlit application for browsing ThoughtSource datasets

### Libraries

* __[dataloader](./libs/dataloader/):__ Library for creating and processing of ThoughtSource datasets (based on the Hugging Face 🤗 Datasets library).


## Current datasets
__Datasets can be browsed online through the [ThoughtSource⚡ Dataset Viewer](http://thought.samwald.info/)__. We have converted the following datasets into a common chain-of-thought format:

* __[aqua](https://github.com/deepmind/AQuA):__ Math word problems from the AQUA-RAT (Algebra Question Answering with Rationales) dataset ([Ling 2017](https://arxiv.org/pdf/1705.04146.pdf)).
* __[asdiv](https://github.com/chaochun/nlu-asdiv-dataset):__ Math word problems from the Academia Sinica Diverse MWP dataset ([Miao 2020](https://aclanthology.org/2020.acl-main.92/)).
* __[commonsense_qa](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa):__ Multiple-choice commonsense knowledge question answering dataset ([Talmor 2018](https://arxiv.org/abs/1811.00937)) enriched with explanations [ECQA](https://github.com/dair-iitd/ECQA-Dataset) ([Aggarwal 2021](https://aclanthology.org/2021.acl-long.238/)).
* __[entailment_bank](https://allenai.org/data/entailmentbank):__ Science exam questions with expert-authored explanations from the EntailmentBank dataset ([Dalvi 2022](https://arxiv.org/pdf/2104.08661.pdf)).
* __[gsm8k](https://github.com/openai/grade-school-math):__  Math word problems from the GSM8K dataset ([Cobbe 2021](https://arxiv.org/abs/2110.14168)).
* __[mawps](https://github.com/sroy9/mawps):__ Math word problems from MAWPS, the Math Word Problem Repository dataset ([Koncel-Kedziorski 2016](https://aclanthology.org/N16-1136.pdf)).
* __[open_book_qa](https://allenai.org/data/open-book-qa):__ Scientific question-answering modeled after open book exams for assessing human understanding from the OpenBookQA dataset ([Mihaylov 2018](https://aclanthology.org/D18-1260.pdf)).
* __[qed](https://github.com/google-research-datasets/QED):__ General-domain question-answering data from the QED dataset ([Lamm 2020](https://arxiv.org/abs/2009.06354)).
* __[strategy_qa](https://allenai.org/data/strategyqa):__ General-domain question-answering data from the StrategyQA dataset ([Geva 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100680/Did-Aristotle-Use-a-Laptop-A-Question-Answering)).
* __[svamp](https://github.com/arkilpatel/SVAMP):__ Math word problems. Source: SVAMP ([Patel 2021](https://aclanthology.org/2021.naacl-main.168/))
* __[worldtree](http://cognitiveai.org/explanationbank/):__ Scientific question-answering data from the WorldTree v2 dataset ([Xie 2020](https://aclanthology.org/2020.lrec-1.671/))

We are working on collecting and generating additional datasets, and on further improving the quality of existing datasets (see [dataset issues](https://github.com/OpenBioLink/ThoughtSource/issues?q=is%3Aissue+label%3Adataset)). We welcome suggestions for the inclusion of other datasets!

