"""
Chain-of-Thought Schema
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question_id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "type": datasets.Value("string"),
        "cot_type": datasets.Value("string"),
        "choices": [datasets.Value("string")],
        "context": datasets.Value("string"),
        "cot": [datasets.Value("string")],
        "answer": [datasets.Value("string")],
        "generated_cot": [{
            "instruction": datasets.Value("string"),
            "cot-trigger": datasets.Value("string"),
            "answer": [{
                "answer-extraction": datasets.Value("string"),
                "answer": [datasets.Value("string")],
            }],
            "cot": [datasets.Value("string")],
            "author": datasets.Value("string"),
            "date": datasets.Value("string"),
            "model": datasets.Value("string"),
            "comment": datasets.Value("string"),
            "annotation": [{
                "author": datasets.Value("string"),
                "date": datasets.Value("string"),
                "key": datasets.Value("string"),
                "value": datasets.Value("string"),
                "comment": datasets.Value("string"),
            }],
        }],
        "feedback": [datasets.Value("string")],
    }
)
