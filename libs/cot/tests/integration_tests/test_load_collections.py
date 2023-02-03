from cot import Collection

# these tests take a very long time
# they were moved to integration tests


def test_thoughtsource() -> None:
    collection = Collection("all", generate_mode="recache")
    """Test that id is unique within a dataset"""
    for name, dataset in collection:
        for split in dataset:
            pd_ = dataset[split].to_pandas()
            assert (
                len(pd_["id"]) == pd_["id"].nunique()
            ), f"IDs are not unique in {name} {split}"


def test_source() -> None:
    collection = Collection("all", generate_mode="recache", source=True)
    assert collection


def test_keep_generated_cots() -> None:
    # load collection with pregenerated cots of all authors
    commonsense = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots="all")
    commonsense_1 = commonsense.select(split="validation", number_samples=1)
    commonsense_1_all = commonsense_1.to_json()
    # check if both authors are loaded
    assert len(commonsense_1["commonsense_qa"]["validation"][0]["generated_cot"]) == 2

    commonsense = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots=["kojima","wei"])
    commonsense_1 = commonsense.select(split="validation", number_samples=1)
    commonsense_1_both = commonsense_1.to_json()
    # check if both authors are kept
    assert commonsense_1_all == commonsense_1_both

    commonsense = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots=["kojima"])
    commonsense_1_kojima = commonsense.select(split="validation", number_samples=1)
    # check if only one author is loaded
    assert len(commonsense_1_kojima["commonsense_qa"]["validation"][0]["generated_cot"]) == 1
    assert commonsense_1_kojima["commonsense_qa"]["validation"][0]["generated_cot"][0]["author"] == "kojima"

    commonsense = Collection(["commonsense_qa"], verbose=False, load_pregenerated_cots=["wei"])
    commonsense_1_wei = commonsense.select(split="validation", number_samples=1)
    # check if only one author is loaded
    assert len(commonsense_1_wei["commonsense_qa"]["validation"][0]["generated_cot"]) == 1
    assert commonsense_1_wei["commonsense_qa"]["validation"][0]["generated_cot"][0]["author"] == "wei"