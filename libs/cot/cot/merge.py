from jsonmerge import Merger


def merge(base_collection, head_collection):
    schema = {
        "patternProperties": {
            ".*": {
                "patternProperties": {
                    ".*": {
                        "mergeStrategy": "arrayMergeById",
                        "mergeOptions": {"idRef": "id"},
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "generated_cot": {
                                    "mergeStrategy": "arrayMergeById",
                                    "mergeOptions": {"idRef": "id"},
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "answers": {
                                                "mergeStrategy": "arrayMergeById",
                                                "mergeOptions": {"idRef": "id"},
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                },
                                            },
                                            "annotations": {
                                                "mergeStrategy": "arrayMergeById",
                                                "mergeOptions": {"idRef": "key"},
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                },
                                            },
                                        },
                                    },
                                }
                            },
                        },
                    }
                }
            }
        }
    }
    merger = Merger(schema)
    merged_json = merger.merge(base_collection.to_json(), head_collection.to_json())
    # cannot call Collection.from_json because of circular dependency otherwise
    return base_collection.__class__.from_json(merged_json)

# merge all files in a directory
def merge_all_files_in_dir(dir):
    from cot import Collection
    filenames = os.listdir(dir)
    filenames = [filename for filename in filenames if filename.endswith(".json")]
    filenames = sorted(filenames)
    collection = Collection.from_json(os.path.join(dir, filenames[0]))
    for filename in filenames[1:]:
        collection_add = Collection.from_json(os.path.join(dir, filename))
        collection = collection.merge(collection_add)
    return collection