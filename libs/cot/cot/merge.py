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
