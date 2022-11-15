# ThoughtSource⚡
__A framework for the science of machine thinking__

ThoughtSource is a central, open resource and community around data and tools related to _chain-of-thought reasoning_ in large language models ([Wei 2022](https://arxiv.org/abs/2201.11903)). Our long-term goal is to enable trustworthy and robust reasoning in advanced AI systems for driving scientific research and development.

<p align="center">
  <img alt="ThoughtSource overview 3" src="./resources/images/thoughtsource-overview-3.svg">
</p>

## Generate interpretable reasoning chains
<p align="center">
  <img alt="ThoughtSource overview 1" src="./resources/images/thoughtsource-overview-1.svg">
</p>

## Annotate, evaluate and improve
<p align="center">
  <img alt="ThoughtSource overview 2" src="./resources/images/thoughtsource-overview-2.svg">
</p>


## Roadmap

1. Create a __repository of chain-of-thought (CoT) datasets__ converted to a unified format. 
2. Create tools for __generating, diagnosing, annotating and evaluating__ CoT reasoning with a wide variety of large language models. 
3. Create a __conceptual model__ of different CoT reasoning styles and errors.
4. Provide models __fine-tuned on high-quality CoT data__.
5. Apply CoT reasoning to __high-impact use-cases__ such as biomedical research or clinical decision making.

## Current datasets
__Datasets can be [browsed online through the Dataset Viewer 🔎](http://thought.samwald.info/)__. 
 
 We created [dataloaders](./libs/dataloader/) that allow you to access the following datasets in a standardized chain-of-thought format. The dataloaders create objects in the [Hugginface 🤗 Datasets format](https://huggingface.co/docs/datasets/index). We (sometimes extensively) post-processed the source datasets in different ways to create coherent reasoning chains.


### General question answering
* __[commonsense_qa](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa):__ Multiple-choice commonsense knowledge question answering dataset ([Talmor 2018](https://arxiv.org/abs/1811.00937), _License:_ Unknown).  Reasoning chains from three different sources are included:

  * __Human-generated__ reasoning chains derived from the __[ECQA dataset](https://github.com/dair-iitd/ECQA-Dataset)__ ([Aggarwal 2021](https://aclanthology.org/2021.acl-long.238/)). Used as gold standard. _License:_ Community Data License Agreements Sharing license 1.0.
  * __AI-generated (few-shot prompting)__ reasoning chains from __[Wei 2022](https://arxiv.org/abs/2201.11903)__. Only available for __validation split__. _License:_ Unknown
  * __AI-generated (zero-shot prompting)__  generated reasoning chains from __[Kojima 2022](https://arxiv.org/abs/2205.11916)__. Only available for __validation split__. _License:_ Unknown
* __[strategy_qa](https://allenai.org/data/strategyqa):__ General-domain question-answering data from the StrategyQA dataset, reasoning chains are derived from original dataset. ([Geva 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00370/100680/Did-Aristotle-Use-a-Laptop-A-Question-Answering)). _License:_ MIT.
  * __Human-generated__ reasoning chains derived from the original dataset. Used as gold standard. _License:_ MIT.
  * __AI-generated (few-shot)__ reasoning chains from __[Wei 2022](https://arxiv.org/abs/2201.11903)__. Only available for __train split__. _License:_ Unknown
  * __AI-generated (zero-shot)__  generated reasoning chains from __[Kojima 2022](https://arxiv.org/abs/2205.11916)__. Only available for __train split__. _License:_ Unknown
* __[qed](https://github.com/google-research-datasets/QED):__ General-domain question-answering data and justifications from the QED dataset ([Lamm 2020](https://arxiv.org/abs/2009.06354)). _License:_ CC BY-SA 3.0.

### Scientific question answering
* __[worldtree](http://cognitiveai.org/explanationbank/):__ Scientific question-answering data from the WorldTree v2 dataset ([Xie 2020](https://aclanthology.org/2020.lrec-1.671/)). __Human-generated__ reasoning chains derived from the original dataset. _License:_ Unknown.
* __[entailment_bank](https://allenai.org/data/entailmentbank):__ Science exam questions with expert-authored explanations from the EntailmentBank dataset ([Dalvi 2022](https://arxiv.org/pdf/2104.08661.pdf)). __Human-generated__ reasoning chains derived from the original dataset. _License:_ CC BY 4.0. (Note: significant overlap with worldtree v2)
* __[open_book_qa](https://allenai.org/data/open-book-qa):__ Scientific question-answering modeled after open book exams for assessing human understanding from the OpenBookQA dataset ([Mihaylov 2018](https://aclanthology.org/D18-1260.pdf)). __Human-generated__ reasoning chains derived from the original dataset. _License:_ Unknown.
* _Planned_: __Medical question answering__ datasets (USMLE, MedMCQA) from [Liévin 2022](https://arxiv.org/abs/2207.08143).

### Math word problems
* __[aqua](https://github.com/deepmind/AQuA):__ Math word problems from the AQUA-RAT (Algebra Question Answering with Rationales) dataset ([Ling 2017](https://arxiv.org/pdf/1705.04146.pdf)). Reasoning chains derived from the original dataset. _License:_ Apache 2.0.
* __[asdiv](https://github.com/chaochun/nlu-asdiv-dataset):__ Math word problems from the Academia Sinica Diverse MWP dataset ([Miao 2020](https://aclanthology.org/2020.acl-main.92/)). Reasoning chains derived from the original dataset. _License:_ Unknown.
* __[gsm8k](https://github.com/openai/grade-school-math):__  Math word problems from the GSM8K dataset ([Cobbe 2021](https://arxiv.org/abs/2110.14168)). Reasoning chains derived from the original dataset. _License:_ MIT.
* __[mawps](https://github.com/sroy9/mawps):__ Math word problems from MAWPS, the Math Word Problem Repository dataset ([Koncel-Kedziorski 2016](https://aclanthology.org/N16-1136.pdf)). Reasoning chains derived from the original dataset. _License:_ Unknown.
* __[svamp](https://github.com/arkilpatel/SVAMP):__ Math word problems. Source: SVAMP ([Patel 2021](https://aclanthology.org/2021.naacl-main.168/)). Reasoning chains derived from the original dataset. _License:_ MIT.


We are working on collecting and generating additional datasets, and on further improving the quality of existing datasets (see [dataset issues](https://github.com/OpenBioLink/ThoughtSource/issues?q=is%3Aissue+label%3Adataset)). We welcome suggestions for the inclusion of other datasets.

__We welcome dataset contributions! 👉 Have a look at our [contribution guide](CONTRIBUTING.md)!__

## Code
### Libraries

* __[dataloader](./libs/dataloader/):__ Library for creating and processing of ThoughtSource datasets (based on the Hugging Face 🤗 Datasets library).

### Applications

* __[dataset-viewer](./apps/dataset-viewer/):__ Streamlit application for browsing ThoughtSource datasets
* __annotator:__ Web-based tool for annotating chain-of-thought data (soon to be released)

<p align="center">
  <img alt="Demonstration of the annotator tool" src="./resources/images/annotator-demo.webp" width="80%">

  The annotator allows for highlighting similarities between different generated reasoning chains, making it easier to spot strenghts and weaknesses and to select best results.
</p>

