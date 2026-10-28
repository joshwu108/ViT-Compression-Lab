import json
import random

import pytest

from vit_compression_lab.eval_sampling import (
    excluded_images_from_manifests,
    select_textvqa,
    select_vqav2,
)


def vqav2_annotations(num_images=50, per_image=3):
    return [
        {"question_id": image * 10 + k, "image_id": image}
        for image in range(num_images)
        for k in range(per_image)
    ]


def textvqa_questions(num_images=50, per_image=2):
    return [
        {"question_id": image * 10 + k, "image_id": f"img{image:04d}"}
        for image in range(num_images)
        for k in range(per_image)
    ]


def test_vqav2_selection_is_reproducible_and_one_per_image():
    first = select_vqav2(vqav2_annotations(), 20, seed=7)
    second = select_vqav2(vqav2_annotations(), 20, seed=7)
    assert [a["question_id"] for a in first] == [a["question_id"] for a in second]
    assert len(first) == 20
    assert len({a["image_id"] for a in first}) == 20


def test_vqav2_selection_ignores_input_order_and_depends_on_seed():
    annotations = vqav2_annotations()
    shuffled = list(annotations)
    random.Random(123).shuffle(shuffled)
    assert select_vqav2(annotations, 10, seed=1) == select_vqav2(shuffled, 10, seed=1)
    assert select_vqav2(annotations, 10, seed=1) != select_vqav2(
        annotations, 10, seed=2
    )


def test_vqav2_selection_excludes_pilot_images_and_fails_when_short():
    excluded = {0, 1, 2}
    selected = select_vqav2(
        vqav2_annotations(num_images=10), 7, seed=0, excluded_image_ids=excluded
    )
    assert not {a["image_id"] for a in selected} & excluded
    with pytest.raises(ValueError, match="Only 7"):
        select_vqav2(
            vqav2_annotations(num_images=10), 8, seed=0, excluded_image_ids=excluded
        )


def test_textvqa_selection_is_reproducible_one_per_image_and_excludes():
    questions = textvqa_questions()
    excluded = {"img0000", "img0001"}
    first = select_textvqa(questions, 15, seed=3, excluded_image_ids=excluded)
    second = select_textvqa(questions, 15, seed=3, excluded_image_ids=excluded)
    assert first == second
    chosen_images = [q["image_id"] for q, _ in first]
    assert len(chosen_images) == 15 and len(set(chosen_images)) == 15
    assert not set(chosen_images) & excluded
    for chosen, others in first:
        assert chosen["question_id"] not in others and len(others) == 1
    with pytest.raises(ValueError, match="Only 48"):
        select_textvqa(questions, 49, seed=3, excluded_image_ids=excluded)


def test_excluded_images_from_manifests(tmp_path):
    manifest = tmp_path / "subset.jsonl"
    rows = [
        {"id": "vqav2-val-1", "source": {"dataset": "VQA v2", "image_id": 5}},
        {"id": "textvqa-val-2", "source": {"dataset": "TextVQA", "image_id": "abc"}},
        {"id": "vqav2-val-3", "source": {"dataset": "VQA v2", "image_id": 9}},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert excluded_images_from_manifests([manifest]) == {
        "VQA v2": {5, 9},
        "TextVQA": {"abc"},
    }
