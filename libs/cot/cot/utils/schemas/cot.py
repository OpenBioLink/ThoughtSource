"""
Chain-of-Thought Schema
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "ref_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "type": datasets.Value("string"),
        "choices": [datasets.Value("string")],
        "context": datasets.Value("string"),
        "cot": [datasets.Value("string")],
        "answer": [datasets.Value("string")],
        "generated_cot": [
            {
                "id": datasets.Value("string"),
                "fragments_version": datasets.Value("string"),
                "instruction": datasets.Value("string"),
                "multiple_choice_formatting": datasets.Value("string"),
                "cot_trigger": datasets.Value("string"),
                "cot_trigger_template": datasets.Value("string"),
                "prompt_text": datasets.Value("string"),
                "answers": [
                    {
                        "id": datasets.Value("string"),
                        "answer_extraction": datasets.Value("string"),
                        "answer_extraction_template": datasets.Value("string"),
                        "answer_extraction_text": datasets.Value("string"),
                        "answer": datasets.Value("string"),
                        "correct_answer": datasets.Value("bool"),
                    }
                ],
                "cot": datasets.Value("string"),
                "author": datasets.Value("string"),
                "date": datasets.Value("string"),
	            "api_service": datasets.Value("string"),
                "model": datasets.Value("string"),
                "comment": datasets.Value("string"),
                "annotation": [
                    {
                        "author": datasets.Value("string"),
                        "date": datasets.Value("string"),
                        "key": datasets.Value("string"),
                        "value": datasets.Value("string"),
                    }
                ],
            }
        ],
        "feedback": [datasets.Value("string")],
    }
)
