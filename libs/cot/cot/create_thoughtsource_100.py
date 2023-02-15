from cot import Collection
import random

def create_medmc_ids():
    collection = Collection(["medmc_qa"], verbose=False, load_pregenerated_cots="all")
    collection = collection.select(split="validation")
    # get indices of examples with generated_cot
    idx = []
    for i,ex in enumerate(collection["medmc_qa"]["validation"]):
        if ex["generated_cot"] != []:
            idx.append(i)
    # get 100 random samples from the idx
    random.seed(0)
    random_ids = random.sample(idx, 100)
    random_ids = sorted(random_ids)

    return random_ids

def create_medmc_100(load_pregenerated_cots=None):
    collection = Collection(["medmc_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    collection = collection.select(split="validation")

    # get 100 random ids
    random_ids = create_medmc_ids()
    # filter the collection
    collection["medmc_qa"]["validation"] = collection["medmc_qa"]["validation"].select(random_ids)

    # check if every item has a generated_cot
    if load_pregenerated_cots=="all" or load_pregenerated_cots==["lievin"]:
        count = 0
        for i,ex in enumerate(collection["medmc_qa"]["validation"]):
            if ex["generated_cot"] != []:
                count += 1
        assert count == 100

    # return the filtered collection
    return collection

def create_thoughtsource_100(load_pregenerated_cots=None):
    number_samples = 100
    # start alphabetically with the commonsense_qa collection
    collection = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
    collection = collection.select(split="validation", number_samples=number_samples)
    # add the medmc collection, then add the other datasets
    # it is special because we only select examples from the validation set where lievin has generated a COT
    medmc_qa = create_medmc_100(load_pregenerated_cots=load_pregenerated_cots)
    collection["medmc_qa"] = medmc_qa["medmc_qa"]

    # define the other datasets
    dataset_split = [
        ("med_qa", "test"),
        ("open_book_qa", "test"),
        ("strategy_qa", "train"),
        ("worldtree", "test"),
    ]

    for dataset, split in dataset_split:
        _coll = Collection([dataset], verbose=False, load_pregenerated_cots=load_pregenerated_cots)
        _coll = _coll.select(split=split, number_samples=number_samples)
        collection[dataset] = _coll[dataset]

    return collection