"""Check embedded-image integrity, official MCQ scoring, and partial-run reporting."""

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_script(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extraction_preserves_images_groups_repeated_images_and_audits_overlap(
    tmp_path,
):
    module = load_script("scripts/prepare_mmstar.py")
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), "red").save(buffer, format="PNG")
    image = buffer.getvalue()
    tsv = tmp_path / "MMStar.tsv"
    with tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=(
                "index",
                "question",
                "answer",
                "category",
                "l2_category",
                "bench",
                "image",
            ),
        )
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "index": str(index),
                    "question": "Which color?\nA: red, B: blue",
                    "answer": "A",
                    "category": "coarse perception",
                    "l2_category": "recognition",
                    "bench": "example",
                    "image": base64.b64encode(image).decode() if index == 0 else "0",
                }
            )
    training = tmp_path / "train.jsonl"
    training.write_text(
        json.dumps(
            {
                "id": "train-1",
                "source": {
                    "image_sha256": hashlib.sha256(image).hexdigest(),
                    "pixel_sha256": "other",
                },
            }
        )
        + "\n"
    )
    output = tmp_path / "extracted"
    metadata = module.extract(tsv, output, [training], expected_count=2)
    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert (output / rows[0]["image_path"]).read_bytes() == image
    assert rows[0]["image_group"] == rows[1]["image_group"]
    assert metadata["unique_decoded_images"] == 1
    assert metadata["overlap_check"]["overlapping_questions"] == 2
    assert module.extract(tsv, output, [training], expected_count=2) == metadata
    (output / rows[0]["image_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Existing extracted image differs"):
        module.extract(tsv, output, [training], expected_count=2)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("A", "A"),
        (" (B). ", "B"),
        ("Option C", "C"),
        ("The answer is D.", "D"),
        ("", None),
        ("I cannot tell", None),
        ("option ", None),
        ("(", None),
        # This surprising case is deliberately compatible with the official rule.
        ("Answer: B", "A"),
    ],
)
def test_official_prefix_parser(answer, expected):
    module = load_script("encoders/Qwen3.8-vision-compression/scripts/score_mmstar.py")
    assert module.option_letter(answer) == expected


def test_scoring_keeps_missing_questions_visible_and_uses_paired_actual_bits(tmp_path):
    module = load_script("encoders/Qwen3.8-vision-compression/scripts/score_mmstar.py")
    config = {"pipeline": "c", "examples": {"ids": ["q1", "q2", "missing"]}}
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    records = []
    for id_, tokens, teacher, compressed in (("q1", 1, "A", "B"), ("q2", 3, "B", "A")):
        records.append(
            {
                "id": id_,
                "image_group": id_,
                "category": "math",
                "l2_category": "geometry",
                "expected_answer": "A",
                "identity_check": {"passed": True},
                "original": {
                    "answer": teacher,
                    "generation": {"hit_token_limit": False},
                },
                "compressed": {
                    "answer": compressed,
                    "generation": {"hit_token_limit": False},
                },
                "learned": {"roundtrip_exact": True},
                "feature_shape": [tokens, 5120],
                "sizes": {"coded_bits": 100},
                "reconstruction": {"mse": 0.1, "relative_l2": 0.2, "cosine_mean": 0.9},
            }
        )
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    summary = module.score_run(tmp_path)
    assert not summary["complete"] and not summary["full_benchmark"]
    assert summary["missing_ids"] == ["missing"]
    assert summary["original"]["accuracy"] == summary["compressed"]["accuracy"] == 0.5
    assert summary["actual_bits_per_visual_token_pooled"] == 50
    assert summary["paired_comparison"]["difference_pp"] == 0
    path.write_text(path.read_text() + json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate result ids"):
        module.score_run(tmp_path)


def test_teacher_only_scoring_needs_no_student_result(tmp_path):
    module = load_script("encoders/Qwen3.8-vision-compression/scripts/score_mmstar.py")
    (tmp_path / "run_config.json").write_text(
        json.dumps({"pipeline": "teacher", "examples": {"ids": ["q"]}})
    )
    record = {
        "id": "q",
        "image_group": "image",
        "category": "math",
        "l2_category": "geometry",
        "expected_answer": "B",
        "identity_check": {"passed": True},
        "original": {"answer": "Option B", "generation": {"hit_token_limit": False}},
    }
    (tmp_path / "results.jsonl").write_text(json.dumps(record) + "\n")
    summary = module.score_run(tmp_path)
    assert summary["complete"] and not summary["full_benchmark"]
    assert summary["original"]["accuracy"] == 1
    assert "compressed" not in summary


def test_option_b_is_scored_as_a_paired_compressed_run(tmp_path):
    module = load_script("encoders/Qwen3.8-vision-compression/scripts/score_mmstar.py")
    record = {
        "id": "q",
        "image_group": "image",
        "category": "math",
        "l2_category": "geometry",
        "expected_answer": "B",
        "identity_check": {"passed": True},
        "original": {"answer": "B", "generation": {"hit_token_limit": False}},
        "compressed": {"answer": "C", "generation": {"hit_token_limit": False}},
        "learned": {"roundtrip_exact": True},
        "feature_shape": [4, 5120],
        "sizes": {"coded_bits": 100},
        "reconstruction": {"mse": 0.1, "relative_l2": 0.2, "cosine_mean": 0.9},
    }
    (tmp_path / "results.jsonl").write_text(json.dumps(record) + "\n")
    config = tmp_path / "run_config.json"
    config.write_text(json.dumps({"pipeline": "b", "examples": {"ids": ["q"]}}))
    summary = module.score_run(tmp_path)
    assert summary["pipeline"] == "b" and summary["compressed"]["accuracy"] == 0
    assert summary["teacher_correct_b_wrong"] == ["q"]
    assert summary["actual_bits_per_visual_token_pooled"] == 25
    assert summary["roundtrip_checks_passed"] == 1
    # an unrecognised pipeline must not be silently scored as teacher-only
    config.write_text(json.dumps({"pipeline": "a", "examples": {"ids": ["q"]}}))
    with pytest.raises(ValueError, match="Unknown pipeline"):
        module.score_run(tmp_path)


def test_chunked_logit_difference_equals_the_direct_maximum():
    import sys

    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from vit_compression_lab.entropic.qwen.injection import load_inject_eval

    ie = load_inject_eval()
    torch.manual_seed(0)
    a, b = torch.randn(1, 37, 50), torch.randn(1, 37, 50)
    for positions in (1, 8, 37, 256):
        assert ie.max_abs_logit_diff(a, b, positions) == (a - b).abs().max().item()
    assert ie.max_abs_logit_diff(a, a) == 0.0
