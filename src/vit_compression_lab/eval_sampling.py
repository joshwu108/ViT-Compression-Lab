"""Seeded question selection for the large VQA v2 / TextVQA evaluation.

Pure Python so it can be tested without datasets. Both selectors take one
question per image, skip excluded images (the pilot set), and are independent
of input order: candidates are sorted before being shuffled with the seed.
"""

import json
import random
from collections import defaultdict
from pathlib import Path


def excluded_images_from_manifests(paths):
    """Image ids per dataset from earlier manifests (``source.dataset`` / ``image_id``)."""
    excluded = defaultdict(set)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                source = json.loads(line)["source"]
                excluded[source["dataset"]].add(source["image_id"])
    return dict(excluded)


def select_vqav2(annotations, count, seed, excluded_image_ids=frozenset()):
    """Pick ``count`` VQA v2 annotations uniformly at random, one per image.

    The question mix (yes/no, counting, other) follows the split's natural
    distribution; stratification is not applied.
    """
    rng = random.Random(seed)
    shuffled = sorted(annotations, key=lambda a: a["question_id"])
    rng.shuffle(shuffled)
    selected, used_images = [], set()
    for annotation in shuffled:
        image_id = annotation["image_id"]
        if image_id in excluded_image_ids or image_id in used_images:
            continue
        selected.append(annotation)
        used_images.add(image_id)
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Only {len(selected)} VQA v2 images available after exclusions; "
        f"{count} requested"
    )


def select_textvqa(questions, count, seed, excluded_image_ids=frozenset()):
    """Pick ``count`` TextVQA images at random, then one question per image.

    Returns (question, other_question_ids_on_image) pairs, as the pilot did.
    """
    by_image = defaultdict(list)
    for question in questions:
        by_image[question["image_id"]].append(question)
    rng = random.Random(seed)
    image_ids = [i for i in sorted(by_image) if i not in excluded_image_ids]
    if len(image_ids) < count:
        raise ValueError(
            f"Only {len(image_ids)} TextVQA images available after exclusions; "
            f"{count} requested"
        )
    rng.shuffle(image_ids)
    selected = []
    for image_id in image_ids[:count]:
        candidates = sorted(by_image[image_id], key=lambda q: q["question_id"])
        chosen = rng.choice(candidates)
        others = [q["question_id"] for q in candidates if q is not chosen]
        selected.append((chosen, others))
    return selected
