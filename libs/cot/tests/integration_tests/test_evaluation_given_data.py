from tests.unit_tests.utils import compare_nested_dict_float_values
from cot import Collection

# test for all the datasets which include evaluations
def test_evaluation_included_datasets():
    # commonsense_qa validation set
    collection = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots="all")
    collection = collection.select(split="validation")
    evaluation = collection.evaluate(overwrite=False, warn=False)
    correct = {
        "commonsense_qa": {
            "validation": {
                "accuracy": {
                    "text-davinci-002": {
                        "None_None_None": 0.734426,
                        "None_kojima-01_kojima-A-E": 0.647494,
                    }
                }
            }
        }
    }
    assert evaluation == correct

    # compare with own calculation of the evaluation
    evaluation = collection.evaluate(overwrite=True, warn=False)
    assert compare_nested_dict_float_values(evaluation, correct, 0.021)

    # med_qa test set
    collection = Collection(["med_qa"], verbose=False, load_pregenerated_cots="all")
    collection = collection.select(split="test")
    evaluation = collection.evaluate(overwrite=False, warn=False)
    correct = {
        "med_qa": {
            "test": {
                "accuracy": {
                    "text-davinci-002": {
                        "None_kojima-01_kojima-A-D": 0.471328,
                        "None_lievin-01_kojima-A-D": 0.450903,
                        "None_lievin-02_kojima-A-D": 0.459544,
                        "None_lievin-03_kojima-A-D": 0.456402,
                        "None_lievin-10_kojima-A-D": 0.468185,
                    },
                    "code-davinci-002": {"None_kojima-01_kojima-03": 0.506838},
                }
            }
        }
    }
    assert evaluation == correct

    # compare with own calculation of the evaluation
    evaluation = collection.evaluate(overwrite=True, warn=False)
    assert compare_nested_dict_float_values(evaluation, correct, 1e-6)

    # medmc_qa validation set
    collection = Collection(["medmc_qa"], verbose=False, load_pregenerated_cots="all")
    collection = collection.select(split="validation")
    evaluation = collection.evaluate(overwrite=False, warn=False)
    correct = {
        "medmc_qa": {
            "validation": {
                "accuracy": {
                    "text-davinci-002": {
                        "None_kojima-01_kojima-A-D": 0.408,
                        "None_lievin-01_kojima-A-D": 0.421,
                        "None_lievin-02_kojima-A-D": 0.388,
                        "None_lievin-03_kojima-A-D": 0.371,
                        "None_lievin-10_kojima-A-D": 0.433,
                    },
                    "code-davinci-002": {"None_kojima-01_kojima-03": 0.494724},
                }
            }
        }
    }
    assert evaluation == correct

    # compare with own calculation of the evaluation
    evaluation = collection.evaluate(overwrite=True, warn=False)
    assert compare_nested_dict_float_values(evaluation, correct, 0.0005)

    # pubmed_qa test set
    collection = Collection(["pubmed_qa"], verbose=False, load_pregenerated_cots="all")
    collection = collection.select(split="test")
    evaluation = collection.evaluate(overwrite=False, warn=False)
    correct = {
        "pubmed_qa": {
            "test": {
                "accuracy": {
                    "text-davinci-002": {
                        "None_kojima-01_kojima-A-C": 0.6,
                        "None_lievin-01_kojima-A-C": 0.556,
                        "None_lievin-02_kojima-A-C": 0.662,
                        "None_lievin-03_kojima-A-C": 0.58,
                        "None_lievin-10_kojima-A-C": 0.598,
                    }
                }
            }
        }
    }
    assert evaluation == correct

    # compare with own calculation of the evaluation
    evaluation = collection.evaluate(overwrite=True, warn=False)
    assert compare_nested_dict_float_values(evaluation, correct, 1e-6)

    # strategy_qa train set
    collection = Collection(["strategy_qa"], verbose=False, load_pregenerated_cots="all")
    collection = collection.select(split="train")
    evaluation = collection.evaluate(overwrite=False, warn=False)
    correct = {
        "strategy_qa": {
            "train": {
                "accuracy": {
                    "text-davinci-002": {
                        "None_None_None": 0.625439,
                        "None_kojima-01_kojima-yes-no": 0.549734,
                    }
                }
            }
        }
    }
    assert evaluation == correct

    # compare with own calculation of the evaluation
    evaluation = collection.evaluate(overwrite=True, warn=False)
    assert compare_nested_dict_float_values(evaluation, correct, 1e-6)


# Could be a test for the evaluation function not changing the collection
# for name in ['aqua', 'asdiv', 'commonsense_qa', 'entailment_bank', 'gsm8k', 'mawps',
#             'med_qa', 'medmc_qa', 'open_book_qa', 'pubmed_qa', 'qed', 'strategy_qa', 'svamp', 'worldtree']:
#     print(name)
#     collection_1 = Collection([name], verbose=False)
#     collection_2 = Collection([name], verbose=False)

#     #only evaluate one collection
#     collection_1.evaluate(warn=False)

#     collection_1_json = collection_1.to_json()
#     collection_2_json = collection_2.to_json()

#     assert collection_1_json == collection_2_json
