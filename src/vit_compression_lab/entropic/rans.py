"""Pure-Python rANS entropy coder with escape symbols and bypass bits.

Interface and bitstream conventions follow CompressAI's ``RansEncoder`` /
``RansDecoder`` (``encode_with_indexes`` / ``decode_with_indexes``) so that a
compiled backend can replace this module without touching callers:

* Each symbol ``s`` is coded with the quantised CDF of its channel
  (``cdfs[index]``, a list of ``cdf_length`` integers from 0 to 2**PRECISION).
* ``value = s - offsets[index]`` indexes the CDF; the last bin
  (``max_value = cdf_length - 2``) is the escape symbol. Out-of-range values
  are zig-zag mapped to a non-negative ``raw`` integer and coded as 4-bit
  bypass chunks: first the chunk count in unary-of-15s, then the chunks.
* The coder is the byte-wise rANS of Giesen (ryg_rans): 32-bit state, lower
  bound 2**23, bytes emitted LIFO and reversed once at the end so the decoder
  reads the stream forwards.

This implementation exists for correctness and exact byte counts. It is slow
(Python loop per symbol); latency measurements should use a C backend.
"""

from bisect import bisect_right

PRECISION = 16
BYPASS_PRECISION = 4
MAX_BYPASS_VAL = (1 << BYPASS_PRECISION) - 1
RANS_BYTE_L = 1 << 23


def pmf_to_quantized_cdf(pmf, tail_mass, precision=PRECISION):
    """Integer CDF with one extra escape bin carrying ``tail_mass``.

    Every bin receives at least one count; the total is exactly 2**precision.
    Returns a list of length len(pmf) + 2.
    """
    probabilities = [max(float(p), 0.0) for p in pmf] + [max(float(tail_mass), 0.0)]
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("pmf has no mass")
    scale = (1 << precision) / total
    freqs = [max(1, int(round(p * scale))) for p in probabilities]
    diff = (1 << precision) - sum(freqs)
    while diff != 0:
        if diff > 0:
            i = max(range(len(freqs)), key=freqs.__getitem__)
            freqs[i] += diff
            diff = 0
        else:
            # Take counts from the largest bins, never below one count.
            i = max(range(len(freqs)), key=freqs.__getitem__)
            take = min(freqs[i] - 1, -diff)
            if take <= 0:
                raise ValueError("Cannot quantise pmf: too many bins for the precision")
            freqs[i] -= take
            diff += take
    cdf = [0]
    for f in freqs:
        cdf.append(cdf[-1] + f)
    return cdf


def _symbol_ops(symbols, indexes, cdfs, cdf_lengths, offsets):
    """Ops in *decoder* order: ('sym', start, freq) or ('raw', value, nbits)."""
    ops = []
    for s, index in zip(symbols, indexes):
        cdf = cdfs[index]
        max_value = cdf_lengths[index] - 2
        value = int(s) - int(offsets[index])
        raw = None
        if value < 0:
            raw = -2 * value - 1
            value = max_value
        elif value >= max_value:
            raw = 2 * (value - max_value)
            value = max_value
        ops.append(("sym", cdf[value], cdf[value + 1] - cdf[value]))
        if raw is not None:
            n_chunks = 0
            while (raw >> (BYPASS_PRECISION * n_chunks)) != 0:
                n_chunks += 1
            count = n_chunks
            while count >= MAX_BYPASS_VAL:
                ops.append(("raw", MAX_BYPASS_VAL, BYPASS_PRECISION))
                count -= MAX_BYPASS_VAL
            ops.append(("raw", count, BYPASS_PRECISION))
            for j in range(n_chunks):
                ops.append(("raw", (raw >> (BYPASS_PRECISION * j)) & MAX_BYPASS_VAL, BYPASS_PRECISION))
    return ops


class RansEncoder:
    def encode_with_indexes(self, symbols, indexes, cdfs, cdf_lengths, offsets):
        ops = _symbol_ops(symbols, indexes, cdfs, cdf_lengths, offsets)
        x = RANS_BYTE_L
        out = bytearray()
        for kind, a, b in reversed(ops):
            if kind == "sym":
                start, freq, scale_bits = a, b, PRECISION
            else:
                start, freq, scale_bits = a, 1, b
            x_max = ((RANS_BYTE_L >> scale_bits) << 8) * freq
            while x >= x_max:
                out.append(x & 0xFF)
                x >>= 8
            x = ((x // freq) << scale_bits) + (x % freq) + start
        for shift in (24, 16, 8, 0):
            out.append((x >> shift) & 0xFF)
        out.reverse()
        return bytes(out)


class RansDecoder:
    def decode_with_indexes(self, string, indexes, cdfs, cdf_lengths, offsets):
        data = memoryview(string)
        if len(data) < 4:
            raise ValueError("rANS stream too short")
        pos = 4
        x = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)

        def advance(start, freq, scale_bits):
            nonlocal x, pos
            mask = (1 << scale_bits) - 1
            x = freq * (x >> scale_bits) + (x & mask) - start
            while x < RANS_BYTE_L:
                if pos >= len(data):
                    raise ValueError("rANS stream ended early")
                x = (x << 8) | data[pos]
                pos += 1

        def get_raw(nbits):
            value = x & ((1 << nbits) - 1)
            advance(value, 1, nbits)
            return value

        out = []
        for index in indexes:
            cdf = cdfs[index]
            max_value = cdf_lengths[index] - 2
            cum = x & ((1 << PRECISION) - 1)
            value = bisect_right(cdf, cum, 0, cdf_lengths[index]) - 1
            advance(cdf[value], cdf[value + 1] - cdf[value], PRECISION)
            if value == max_value:
                n_chunks = 0
                chunk = get_raw(BYPASS_PRECISION)
                while chunk == MAX_BYPASS_VAL:
                    n_chunks += MAX_BYPASS_VAL
                    chunk = get_raw(BYPASS_PRECISION)
                n_chunks += chunk
                raw = 0
                for j in range(n_chunks):
                    raw |= get_raw(BYPASS_PRECISION) << (BYPASS_PRECISION * j)
                value = -(raw + 1) // 2 if raw & 1 else max_value + raw // 2
            out.append(value + int(offsets[index]))
        return out
