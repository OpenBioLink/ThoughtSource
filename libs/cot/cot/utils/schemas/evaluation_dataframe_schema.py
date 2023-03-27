"""
Evaluation dataframe schema
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "ref_id": datasets.Value("string"),
        "generated_cot": [
            {
                "id": datasets.Value("string"),
                "instruction": datasets.Value("string"),
                "cot_trigger": datasets.Value("string"),
                "cot_trigger_template": datasets.Value("string"),
                "answers": [
                    {
                        "id": datasets.Value("string"),
                        "answer_extraction": datasets.Value("string"),
                        "answer_extraction_template": datasets.Value("string"),
                        "correct_answer": datasets.Value("bool"),
                    }
                ],
                "author": datasets.Value("string"),
                "date": datasets.Value("string"),
                "api_service": datasets.Value("string"),
                "model": datasets.Value("string"),
                "comment": datasets.Value("string")
            }
        ],
        "feedback": [datasets.Value("string")],
    }
)
