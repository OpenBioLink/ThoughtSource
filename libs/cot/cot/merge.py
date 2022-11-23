from jsonmerge import Merger

def merge(base, head):
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
                                                    }
                                                },
                                                "annotation": {
                                                    "mergeStrategy": "arrayMergeById",
                                                    "mergeOptions": {"idRef": "id"},
                                                    "type": "array",
                                                    "items": {
                                                        "type": "object",
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }
    merger = Merger(schema)
    return merger.merge(base, head)