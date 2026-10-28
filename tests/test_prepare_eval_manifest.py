import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_eval_manifest.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_eval_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vqav2_record_keeps_dataset_fields_and_groups_by_image():
    module = load_script()
    annotation = {
        "question_id": 42,
        "image_id": 7,
        "question_type": "how many",
        "answer_type": "number",
        "multiple_choice_answer": "2",
        "answers": [
            {"answer": "2", "answer_confidence": "yes", "answer_id": i}
            for i in range(10)
        ],
    }
    record = module.vqav2_record(
        annotation,
        {42: "How many cats?"},
        {7: [42, 43]},
        "images/vqav2/x.jpg",
        "http://x/y.jpg",
    )
    assert record["id"] == "vqav2-val-42" and record["dataset"] == "vqav2"
    assert record["image_group"] == "vqav2:7" and record["task_type"] == "counting"
    assert (
        record["question"] == "How many cats?"
        and record["accepted_answers"] == ["2"] * 10
    )
    assert record["source"]["other_question_ids_on_image"] == [43]
    assert record["scoring"] == "vqav2_vqa_accuracy"


def test_textvqa_record_uses_most_common_answer():
    module = load_script()
    question = {
        "question_id": 9,
        "image_id": "abc",
        "question": "What brand?",
        "set_name": "val",
        "answers": ["sony", "sony", "samsung"],
        "image_classes": ["Television"],
        "image_width": 10,
        "image_height": 5,
        "flickr_original_url": "http://f/x.jpg",
    }
    record = module.textvqa_record(
        question, [10, 11], "images/textvqa/abc.jpg", "http://x"
    )
    assert record["id"] == "textvqa-val-9" and record["task_type"] == "ocr"
    assert (
        record["expected_answer"] == "sony" and record["image_group"] == "textvqa:abc"
    )
    assert record["source"]["other_question_ids_on_image"] == [10, 11]
