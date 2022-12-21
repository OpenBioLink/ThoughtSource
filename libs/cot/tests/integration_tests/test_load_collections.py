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