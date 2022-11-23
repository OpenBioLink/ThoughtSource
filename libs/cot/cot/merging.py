

# matches examples based on dataset_name->split->id
# matches generated_cots based on 


# options how = outer
import copy
import pandas as pd
from cot import Collection


def merge(collection_1, collection_2, options):

    datasets_1 = pd.Series(collection_1.loaded)
    datasets_2 = pd.Series(collection_2.loaded)

    datasets = datasets_1.merge(datasets_2, how=options["how"])

    base = None
    if options["how"] == "left":
        base = collection_1
    elif options["how"] == "right":
        base = collection_2
    else:
        base = Collection() # empty collection

    base = copy.deepcopy(collection_1) if options["how"] is not "right" else copy.deepcopy(collection_2)

    for dataset in datasets:
        # one collection contains a dataset the other one does not contain, add dataset and continue, no merge needed
        if (dataset not in datasets_1) or (dataset not in datasets_2):
            # do nothing if how = "left"/"right"
            # already in base collection
            # add dataset and continue only if outer
            if options["how"] == "outer":
                base[dataset] = collection_1[dataset] if dataset in datasets_1 else collection_2[dataset]
            assert (options["how"] != "inner"), "Impossibile"
            continue
        else: # merge splits
            splits_1 = pd.Series(collection_1[dataset])
            splits_2 = pd.Series(collection_2[dataset])
            splits = splits_1.merge(splits_2, how=options["how"])
            for split in splits:
                # one collection contains a split the other one does not contain, add split and continue, no merge needed
                if (split not in splits_1) or (split not in splits_2):
                    # do nothing if how = "left"/"right"
                    # already in base collection
                    # add split and continue only if outer
                    if options["how"] == "outer":
                        # TODO check dataset_dict if the following is even possible (to add splits with [split]=)
                        base[dataset][split] = collection_1[dataset][split] if split in splits_1 else collection_2[dataset][split]
                    assert (options["how"] != "inner"), "Impossibile"
                    continue
                else: # merge examples
                    examples_1 = pd.Series([x["id"] for x in collection_1[dataset][split]])
                    examples_2 = pd.Series([x["id"] for x in collection_2[dataset][split]])
                    examples = examples_1.merge(examples_2, how=options["how"])
                    for example in examples:
                        # one collection contains an example the other one does not contain, add example and continue, no merge needed
                        if (example not in examples_1) or (example not in examples_2):
                            # do nothing if how = "left"/"right"
                            # already in base collection
                            # add split and continue only if outer
                            if options["how"] == "outer":
                                # TODO check if the following is even possible:
                                if example in 

                                base[dataset][split] = collection_1[dataset][split] if split in splits_1 else collection_2[dataset][split]
                            assert (options["how"] != "inner"), "Impossibile"
                            continue
                        else: # merge example








