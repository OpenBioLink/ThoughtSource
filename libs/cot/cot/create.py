""" This file contains the functions used to initially create the thoughtsource_100 collection.
If you want to load the already created thoughtsource_100 collection, use the following code:
collection = Collection.load_thoughtsource_100()"""

import random

from cot import Collection

def create_thoughtsource(number_samples, load_pregenerated_cots=False):
    number_samples = number_samples
    # start alphabetically with the commonsense_qa collection
    collection = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    collection = collection.select(split="validation", number_samples=number_samples)
    # add the med_qa collection
    _coll = Collection(["med_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    _coll = _coll.select(split="test", number_samples=number_samples)
    collection["med_qa"] = _coll["med_qa"]

    # add the medmc collection, then add the other datasets
    # it is special because we only select examples from the validation set where lievin has generated a COT
    medmc_qa = create_special_medmc(number_samples, load_pregenerated_cots=load_pregenerated_cots)
    collection["medmc_qa"] = medmc_qa["medmc_qa"]

    # define the other datasets
    dataset_split = [
        ("open_book_qa", "test"),
        ("strategy_qa", "train"),
        ("worldtree", "test"),
    ]

    for dataset, split in dataset_split:
        _coll = Collection([dataset], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
        _coll = _coll.select(split=split, number_samples=number_samples)
        collection[dataset] = _coll[dataset]

    return collection

def create_special_medmc(number_samples, load_pregenerated_cots=False):

    # creating the medmc collection will run two times and be also printed twice, since it is a special case
    collection = Collection(["medmc_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    collection = collection.select(split="validation")

    # create random ids
    random_ids = create_special_medmc_ids(number_samples)

    # filter the collection
    collection["medmc_qa"]["validation"] = collection["medmc_qa"]["validation"].select(random_ids)

    # check if every item has a generated_cot
    if load_pregenerated_cots==True:
        count = 0
        for i,ex in enumerate(collection["medmc_qa"]["validation"]):
            if ex["generated_cot"] != []:
                count += 1
        assert count == number_samples, f"count: {count}, number_samples: {number_samples}"

    # return the filtered collection
    return collection

def create_special_medmc_ids(number_samples):
    collection = Collection(["medmc_qa"], verbose=False, load_pregenerated_cots=True)
    collection = collection.select(split="validation")
    # get indices of examples with generated_cot
    idx = []
    for i,ex in enumerate(collection["medmc_qa"]["validation"]):
        # take only the examples where lievin has generated a COT
        # Important: take also only the examples where human generated COTs are available
        if ex["generated_cot"] != [] and ex["cot"] != []:
            idx.append(i)
    # get random samples from the idx
    random.seed(0)
    random_ids = random.sample(idx, number_samples)
    random_ids = sorted(random_ids)

    return random_ids

def create_thoughtsource_100(load_pregenerated_cots=False):
    number_samples = 100
    # start alphabetically with the commonsense_qa collection
    collection = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    collection = collection.select(split="validation", number_samples=number_samples)
    # add the med_qa collection
    _coll = Collection(["med_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    _coll = _coll.select(split="test", number_samples=number_samples)
    collection["med_qa"] = _coll["med_qa"]

    # add the medmc collection, then add the other datasets
    # it is special because we only select examples from the validation set where lievin has generated a COT
    medmc_qa = create_special_medmc_100(load_pregenerated_cots=load_pregenerated_cots)
    collection["medmc_qa"] = medmc_qa["medmc_qa"]

    # define the other datasets
    dataset_split = [
        ("open_book_qa", "test"),
        ("strategy_qa", "train"),
        ("worldtree", "test"),
    ]

    for dataset, split in dataset_split:
        _coll = Collection([dataset], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
        _coll = _coll.select(split=split, number_samples=number_samples)
        collection[dataset] = _coll[dataset]

    return collection

def create_thoughtsource_1(load_pregenerated_cots=False):
    # this is not an official collection, but it is useful for debugging/testing
    number_samples = 1
    # start alphabetically with the commonsense_qa collection
    collection = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    collection = collection.select(split="validation", number_samples=number_samples)
    # add the med_qa collection
    _coll = Collection(["med_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    _coll = _coll.select(split="test", number_samples=number_samples)
    collection["med_qa"] = _coll["med_qa"]

    # add the medmc collection, then add the other datasets
    # it is special because we only select examples from the validation set where lievin has generated a COT
    medmc_qa = create_special_medmc_100(load_pregenerated_cots=load_pregenerated_cots)
    medmc_qa = medmc_qa.select(split="validation", number_samples=number_samples)
    collection["medmc_qa"] = medmc_qa["medmc_qa"]

    # define the other datasets
    dataset_split = [
        ("open_book_qa", "test"),
        ("strategy_qa", "train"),
        ("worldtree", "test"),
    ]

    for dataset, split in dataset_split:
        _coll = Collection([dataset], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
        _coll = _coll.select(split=split, number_samples=number_samples)
        collection[dataset] = _coll[dataset]

    return collection


def create_special_medmc_ids_100():
    collection = Collection(["medmc_qa"], verbose=False, load_pregenerated_cots=True)
    collection = collection.select(split="validation")
    # get indices of examples with generated_cot
    idx = []
    for i,ex in enumerate(collection["medmc_qa"]["validation"]):
        # take only the examples where lievin has generated a COT
        # Important: take also only the examples where human generated COTs are available
        if ex["generated_cot"] != [] and ex["cot"] != []:
            idx.append(i)
    # get 100 random samples from the idx
    random.seed(0)
    random_ids = random.sample(idx, 100)
    random_ids = sorted(random_ids)

    return random_ids

def create_special_medmc_100(load_pregenerated_cots=False):

    # creating the medmc collection will run two times and be also printed twice, since it is a special case
    collection = Collection(["medmc_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    collection = collection.select(split="validation")

    # create 100 random ids
    random_ids = create_special_medmc_ids_100()

    # filter the collection
    collection["medmc_qa"]["validation"] = collection["medmc_qa"]["validation"].select(random_ids)

    # check if every item has a generated_cot
    if load_pregenerated_cots==True:
        count = 0
        for i,ex in enumerate(collection["medmc_qa"]["validation"]):
            if ex["generated_cot"] != []:
                count += 1
        assert count == 100

    # return the filtered collection
    return collection

def unique_index_correction(self):
    for name in self._cache:
        if name in ["aqua", "entailment_bank", "gsm8k", "mawps", "med_qa", "open_book_qa", "worldtree", "svamp"]: 
            for split in self._cache[name]:
                self[name][split] = self[name][split].map(
                    _unique_index_correction,
                    fn_kwargs={
                        "name": name,
                        "split": split,
                    },
                    features=self[name][split].info.features,
                    load_from_cache_file=False,
                )
    return self

def _unique_index_correction(item, name, split):
    if (name or split) in item["id"]:
        print("The id '" + item["id"] + "' already contains the dataset name or split and will not be changed.")
    else:
        item["id"] = name + "_" + split + "_" + item["id"]
    return item