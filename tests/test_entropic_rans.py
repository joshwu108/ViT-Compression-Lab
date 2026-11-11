import random

from vit_compression_lab.entropic import rans


def make_tables(num_channels, pmf_len=9, seed=0):
    rng = random.Random(seed)
    cdfs, lengths, offsets = [], [], []
    for _ in range(num_channels):
        pmf = [rng.random() + 0.05 for _ in range(pmf_len)]
        total = sum(pmf)
        cdf = rans.pmf_to_quantized_cdf([p / total for p in pmf], 1e-6)
        cdfs.append(cdf)
        lengths.append(len(cdf))
        offsets.append(-(pmf_len // 2))
    return cdfs, lengths, offsets


def test_quantized_cdf_is_monotone_and_complete():
    cdf = rans.pmf_to_quantized_cdf([0.5, 0.3, 0.2], 1e-9)
    assert cdf[0] == 0 and cdf[-1] == 1 << rans.PRECISION
    assert all(b > a for a, b in zip(cdf, cdf[1:]))
    assert len(cdf) == 3 + 2  # pmf bins + escape bin + leading zero


def test_round_trip_in_range_symbols():
    cdfs, lengths, offsets = make_tables(4)
    rng = random.Random(1)
    indexes = [i % 4 for i in range(400)]
    symbols = [rng.randint(-4, 4) for _ in indexes]
    string = rans.RansEncoder().encode_with_indexes(symbols, indexes, cdfs, lengths, offsets)
    decoded = rans.RansDecoder().decode_with_indexes(string, indexes, cdfs, lengths, offsets)
    assert decoded == symbols


def test_round_trip_with_escape_symbols():
    cdfs, lengths, offsets = make_tables(2)
    indexes = [0, 1, 0, 1, 0, 1]
    symbols = [0, 57, -300, 4, -5, 100000]  # three are far outside the tables
    string = rans.RansEncoder().encode_with_indexes(symbols, indexes, cdfs, lengths, offsets)
    decoded = rans.RansDecoder().decode_with_indexes(string, indexes, cdfs, lengths, offsets)
    assert decoded == symbols


def test_empty_stream_is_only_the_flushed_state():
    cdfs, lengths, offsets = make_tables(1)
    string = rans.RansEncoder().encode_with_indexes([], [], cdfs, lengths, offsets)
    assert len(string) == 4
    assert rans.RansDecoder().decode_with_indexes(string, [], cdfs, lengths, offsets) == []
