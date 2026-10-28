import numpy as np
import pytest

from vit_compression_lab.paired_stats import decide, paired_bootstrap_retention


def test_identical_scores_pass_with_retention_exactly_one():
    scores = [1.0, 0.6, 0.3, 1.0, 0.0] * 20
    result = paired_bootstrap_retention(scores, scores, replicates=2000, seed=1)
    assert result["retention"] == 1.0
    assert result["retention_lower_95_one_sided"] == 1.0
    assert result["retention_upper_95_one_sided"] == 1.0
    assert result["difference_pp"] == 0.0
    assert result["decision"] == "PASS"


def test_clear_degradation_fails():
    rng = np.random.default_rng(0)
    original = rng.choice([0.0, 0.3, 0.6, 1.0], size=2000, p=[0.1, 0.1, 0.2, 0.6])
    compressed = np.where(rng.random(2000) < 0.3, 0.0, original)
    result = paired_bootstrap_retention(original, compressed, replicates=2000, seed=0)
    assert result["retention"] < 0.8
    assert result["retention_upper_95_one_sided"] < 0.99
    assert result["decision"] == "FAIL"


def test_small_sample_is_inconclusive_and_bounds_bracket_estimate():
    original = [1.0, 0.6, 1.0, 0.3, 1.0, 1.0, 0.6, 1.0]
    compressed = [1.0, 0.3, 1.0, 0.3, 1.0, 0.6, 0.6, 1.0]
    result = paired_bootstrap_retention(original, compressed, replicates=4000, seed=5)
    lower, upper = (
        result["retention_lower_95_one_sided"],
        result["retention_upper_95_one_sided"],
    )
    assert lower <= result["retention"] <= upper
    assert lower < 0.99 <= upper
    assert result["decision"] == "INCONCLUSIVE"


def test_same_seed_reproduces_bounds_and_different_seed_differs():
    rng = np.random.default_rng(3)
    original = rng.random(300)
    compressed = original * rng.uniform(0.9, 1.0, size=300)
    a = paired_bootstrap_retention(original, compressed, replicates=3000, seed=42)
    b = paired_bootstrap_retention(original, compressed, replicates=3000, seed=42)
    c = paired_bootstrap_retention(original, compressed, replicates=3000, seed=43)
    assert a["retention_lower_95_one_sided"] == b["retention_lower_95_one_sided"]
    assert a["retention_ci95_two_sided"] == b["retention_ci95_two_sided"]
    assert a["retention_lower_95_one_sided"] != c["retention_lower_95_one_sided"]
    assert a["bootstrap"]["seed"] == 42 and a["bootstrap"]["replicates"] == 3000


def test_zero_baseline_is_undefined_not_pass():
    result = paired_bootstrap_retention([0.0] * 10, [0.0] * 10, replicates=500, seed=0)
    assert result["retention"] is None
    assert result["decision"] == "UNDEFINED"
    assert "zero" in result["note"]


def test_clustering_by_image_widens_the_interval():
    rng = np.random.default_rng(9)
    # 100 images with 4 questions each; whole images degrade together.
    image_effect = rng.random(100) < 0.5
    original = np.ones(400)
    compressed = np.where(np.repeat(image_effect, 4), 0.8, 1.0)
    groups = np.repeat(np.arange(100), 4)
    clustered = paired_bootstrap_retention(
        original, compressed, groups, replicates=3000, seed=1
    )
    unclustered = paired_bootstrap_retention(
        original, compressed, None, replicates=3000, seed=1
    )
    assert clustered["n_clusters"] == 100 and unclustered["n_clusters"] == 400

    def width(r):
        return r["retention_upper_95_one_sided"] - r["retention_lower_95_one_sided"]

    assert width(clustered) > width(unclustered)


def test_decision_thresholds_and_input_validation():
    assert decide(0.99, 1.0, 0.99) == "PASS"
    assert decide(0.98, 0.989, 0.99) == "FAIL"
    assert decide(0.98, 0.995, 0.99) == "INCONCLUSIVE"
    assert decide(None, None, 0.99) == "UNDEFINED"
    with pytest.raises(ValueError):
        paired_bootstrap_retention([1.0], [1.0, 0.5])
    with pytest.raises(ValueError):
        paired_bootstrap_retention([], [])
