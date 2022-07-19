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
        "answer": datasets.Sequence(datasets.Value("string")),
        "cot": [datasets.Value("string")],
        "answer": [datasets.Value("string")],
        "feedback": [datasets.Value("string")],
        "cot_after_feedback": [datasets.Value("string")],
        "answer_after_feedback": [datasets.Value("string")],
    }
)
