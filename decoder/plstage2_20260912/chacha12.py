"""The gamma stream of relay_bp's RelayDecoder, reproduced in numpy: StdRng
(ChaCha12, seeded by seed_from_u64) feeding a Uniform<f64>(low, high). Each
relay set draws one contiguous run of ncols values in variable order; the
pre-leg draws nothing. The decoder uses np.float32 of these f64 draws; that
cast is left to the caller.
"""
import numpy as np

CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)
_U32 = np.uint32
_U64 = np.uint64
_2P52 = 4503599627370496.0  # 2**52


def seed_from_u64(state, nbytes=32):
    """rand_core 0.6.4 SeedableRng::seed_from_u64 -> the raw seed bytes."""
    mul = 6364136223846793005
    inc = 11634580027462260723
    mask = (1 << 64) - 1
    s = state & mask
    out = bytearray()
    while len(out) < nbytes:
        s = (s * mul + inc) & mask
        xorshifted = (((s >> 18) ^ s) >> 27) & 0xFFFFFFFF
        rot = (s >> 59) & 0xFFFFFFFF
        r = rot & 31
        x = ((xorshifted >> r) | (xorshifted << ((32 - r) & 31))) & 0xFFFFFFFF
        out += int(x).to_bytes(4, "little")
    return bytes(out[:nbytes])


def _rotl(v, n):
    return ((v << _U32(n)) | (v >> _U32(32 - n))).astype(_U32)


def chacha_blocks(key_words, counters, drounds=6):
    """ChaCha block function over an array of block counters.

    Returns uint32 array of shape (nblocks, 16) -- the little-endian output
    words of each block, which is exactly what ChaCha12Core::generate leaves in
    the BlockRng buffer.
    """
    ctr = np.ascontiguousarray(counters, dtype=_U64)
    n = ctr.size
    st = np.zeros((16, n), dtype=_U32)
    for i, c in enumerate(CONSTANTS):
        st[i] = _U32(c)
    for i in range(8):
        st[4 + i] = _U32(key_words[i])
    st[12] = (ctr & _U64(0xFFFFFFFF)).astype(_U32)
    st[13] = (ctr >> _U64(32)).astype(_U32)
    # words 14, 15 are the stream identifier, zero for a from_seed ChaCha.
    x = st.copy()

    def qr(a, b, c, d):
        x[a] = x[a] + x[b]
        x[d] = _rotl(x[d] ^ x[a], 16)
        x[c] = x[c] + x[d]
        x[b] = _rotl(x[b] ^ x[c], 12)
        x[a] = x[a] + x[b]
        x[d] = _rotl(x[d] ^ x[a], 8)
        x[c] = x[c] + x[d]
        x[b] = _rotl(x[b] ^ x[c], 7)

    with np.errstate(over="ignore"):
        for _ in range(drounds):
            qr(0, 4, 8, 12)
            qr(1, 5, 9, 13)
            qr(2, 6, 10, 14)
            qr(3, 7, 11, 15)
            qr(0, 5, 10, 15)
            qr(1, 6, 11, 12)
            qr(2, 7, 8, 13)
            qr(3, 4, 9, 14)
        out = (x + st).astype(_U32)
    return np.ascontiguousarray(out.T)


def uniform_scale(low, high):
    """rand 0.8.5 UniformFloat::<f64>::new -> the stored `scale`."""
    max_rand = float.fromhex("0x1.fffffffffffffp+0") - 1.0
    scale = float(high) - float(low)
    if not np.isfinite(scale):
        raise ValueError("Uniform::new: range overflow")
    # decrease_masked = f64::from_bits(bits - 1); for a positive finite scale
    # that is the next representable value toward zero.
    while scale * max_rand + float(low) >= float(high):
        scale = float(np.nextafter(np.float64(scale), np.float64(-np.inf)))
    return scale


class GammaStream:
    """The RelayDecoder's `posterior_update_state`, as a random-access stream.

    Draw j of the decoder's life is self.at(j); nothing is stateful, so a shot
    that starts at stream offset k is served by at(k), at(k+1), ...
    """

    BLOCK_WORDS = 16

    def __init__(self, seed=0, low=-0.24, high=0.6):
        self.key = np.frombuffer(seed_from_u64(seed), dtype="<u4").copy()
        self.low = float(low)
        self.high = float(high)
        self.scale = uniform_scale(low, high)

    def u64(self, offset, count):
        """Raw next_u64() outputs `offset` .. `offset+count` of the stream."""
        if count <= 0:
            return np.zeros(0, dtype=_U64)
        w0 = 2 * int(offset)
        w1 = w0 + 2 * int(count)
        b0 = w0 // self.BLOCK_WORDS
        b1 = (w1 - 1) // self.BLOCK_WORDS
        words = chacha_blocks(self.key,
                              np.arange(b0, b1 + 1, dtype=_U64)).reshape(-1)
        seg = words[w0 - b0 * self.BLOCK_WORDS: w1 - b0 * self.BLOCK_WORDS]
        return (seg[0::2].astype(_U64)
                | (seg[1::2].astype(_U64) << _U64(32)))

    def at(self, offset, count):
        """Uniform f64 samples `offset` .. `offset+count` of the stream."""
        u = self.u64(offset, count)
        # (u >> 12) < 2**52, so the cast and the divide are exact and equal
        # Rust's into_float_with_exponent(0) - 1.0.
        value0_1 = (u >> _U64(12)).astype(np.float64) / _2P52
        return value0_1 * self.scale + self.low

    def gammas(self, offset, num_sets, ncols):
        """Gammas for `num_sets` consecutive relay sets starting at a stream
        offset, shaped (num_sets, ncols) -- row s is the set that consumes
        draws offset + s*ncols .. offset + (s+1)*ncols.
        """
        flat = self.at(offset, num_sets * ncols)
        return flat.reshape(num_sets, ncols)


def chunked_gammas(stream, offset, num_sets, ncols, block_sets=16):
    """gammas() computed a few sets at a time, to bound peak memory."""
    out = np.empty((num_sets, ncols), dtype=np.float64)
    for s0 in range(0, num_sets, block_sets):
        s1 = min(s0 + block_sets, num_sets)
        out[s0:s1] = stream.at(offset + s0 * ncols,
                               (s1 - s0) * ncols).reshape(s1 - s0, ncols)
    return out
