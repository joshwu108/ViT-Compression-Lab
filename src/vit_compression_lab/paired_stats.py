"""Paired bootstrap for the retention of a score under compression.

retention = mean(compressed score) / mean(original score)

Original and compressed scores of each example are resampled together, by
cluster (image), so the paired structure is kept. Bounds are percentiles of
the bootstrap distribution. The 5th and 95th percentiles serve as one-sided
95% lower and upper bounds for the decision rule.
"""

import numpy as np

REPLICATE_CHUNK = 500  # bounds memory: chunk x clusters indices at a time


def decide(lower, upper, threshold):
    if lower is None or upper is None:
        return "UNDEFINED"
    if lower >= threshold:
        return "PASS"
    if upper < threshold:
        return "FAIL"
    return "INCONCLUSIVE"


def paired_bootstrap_retention(
    original, compressed, groups=None, replicates=10_000, seed=0, threshold=0.99
):
    """Point estimates, bootstrap bounds, and the decision for one set of examples.

    ``groups`` labels the cluster of each example (for example its image);
    None treats every example as its own cluster. A zero original mean makes
    retention undefined; replicates whose resampled original sum is zero are
    dropped and counted.
    """
    original = np.asarray(original, dtype=float)
    compressed = np.asarray(compressed, dtype=float)
    if original.shape != compressed.shape or original.ndim != 1 or original.size == 0:
        raise ValueError("Need equal-length, nonempty paired score lists")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    n = original.size
    groups = np.arange(n) if groups is None else np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    clusters = int(inverse.max()) + 1
    original_sums = np.bincount(inverse, weights=original, minlength=clusters)
    compressed_sums = np.bincount(inverse, weights=compressed, minlength=clusters)
    sizes = np.bincount(inverse, minlength=clusters).astype(float)

    rng = np.random.default_rng(seed)
    ratios, differences = [], []
    dropped = 0
    for start in range(0, replicates, REPLICATE_CHUNK):
        size = min(REPLICATE_CHUNK, replicates - start)
        index = rng.integers(0, clusters, size=(size, clusters))
        boot_original = original_sums[index].sum(axis=1)
        boot_compressed = compressed_sums[index].sum(axis=1)
        boot_n = sizes[index].sum(axis=1)
        differences.append(100 * (boot_compressed - boot_original) / boot_n)
        valid = boot_original > 0
        dropped += int((~valid).sum())
        ratios.append(boot_compressed[valid] / boot_original[valid])
    ratios = np.concatenate(ratios)
    differences = np.concatenate(differences)

    original_mean = float(original.mean())
    compressed_mean = float(compressed.mean())
    result = {
        "n_examples": int(n),
        "n_clusters": int(clusters),
        "original_mean": original_mean,
        "compressed_mean": compressed_mean,
        "difference_pp": 100 * (compressed_mean - original_mean),
        "difference_pp_ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "retention": None,
        "retention_lower_95_one_sided": None,
        "retention_upper_95_one_sided": None,
        "retention_ci95_two_sided": None,
        "bootstrap": {
            "method": "paired percentile bootstrap, clusters resampled with replacement",
            "replicates": int(replicates),
            "replicates_dropped_zero_original": dropped,
            "seed": int(seed),
            "numpy": np.__version__,
        },
        "threshold": threshold,
    }
    if original_mean == 0:
        result["decision"] = "UNDEFINED"
        result["note"] = "Original mean score is zero, so retention is undefined."
        return result
    result["retention"] = compressed_mean / original_mean
    if ratios.size < 0.9 * replicates:
        result["decision"] = "INCONCLUSIVE"
        result["note"] = (
            "More than 10% of bootstrap replicates had a zero original sum."
        )
        return result
    lower = float(np.quantile(ratios, 0.05))
    upper = float(np.quantile(ratios, 0.95))
    result["retention_lower_95_one_sided"] = lower
    result["retention_upper_95_one_sided"] = upper
    result["retention_ci95_two_sided"] = [
        float(np.quantile(ratios, 0.025)),
        float(np.quantile(ratios, 0.975)),
    ]
    result["decision"] = decide(lower, upper, threshold)
    return result
