"""Bit-packed GF(2) linear algebra: rref, rank, kernel and CSS logicals.

logicals(H, Hother) returns a [k, n] uint8 basis of ker(H) modulo
rowspace(Hother), with k = n - rank(H) - rank(Hother) for a CSS pair.
"""
import numpy as np


def _pack(M):
    """[m, n] 0/1 uint8 -> [m, W] uint64 words (little-bit-first per word)."""
    M = np.asarray(M, dtype=np.uint8) % 2
    m, n = M.shape
    W = (n + 63) // 64
    P = np.zeros((m, W * 8), dtype=np.uint8)
    P[:, :  (n + 7) // 8] = np.packbits(M, axis=1, bitorder="little")
    return P.view(np.uint64), n


def _unpack(P, n):
    B = np.unpackbits(P.view(np.uint8), axis=1, bitorder="little")
    return B[:, :n].astype(np.uint8)


def _getcol(P, c):
    return (P[:, c >> 6] >> np.uint64(c & 63)) & np.uint64(1)


def rref(M):
    """Packed in-place RREF. Returns (P, n, pivots) with pivots the list of
    pivot column indices in order (row i of the reduced matrix has pivot
    pivots[i])."""
    P, n = _pack(M)
    m = P.shape[0]
    pivots = []
    r = 0
    for c in range(n):
        if r == m:
            break
        col = _getcol(P, c)
        nz = np.flatnonzero(col[r:])
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            P[[r, p]] = P[[p, r]]
        rows = np.flatnonzero(_getcol(P, c))
        rows = rows[rows != r]
        if len(rows):
            P[rows] ^= P[r]
        pivots.append(c)
        r += 1
    return P[:r], n, pivots


def rank(M):
    return rref(M)[0].shape[0]


def kernel_basis(M):
    """[n - rank, n] uint8 basis of the right kernel of M over GF(2)."""
    P, n, pivots = rref(M)
    piv = set(pivots)
    free = [c for c in range(n) if c not in piv]
    R = _unpack(P, n)
    K = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        K[i, f] = 1
        # each pivot row r with a 1 in column f forces pivot bit = 1
        rows = np.flatnonzero(R[:, f])
        for r0 in rows:
            K[i, pivots[r0]] ^= 1
    return K


def logicals(H, Hother):
    """Basis of ker(H) / rowspace(Hother): CSS logical representatives."""
    K = kernel_basis(H)                       # spans ker(H) >= rowspace(Hother)
    S, n = _pack(Hother)
    Sr, _, spiv = rref(Hother)
    srank = Sr.shape[0]
    out = []
    # reduce each kernel vector against the (growing) stabilizer+chosen span;
    # keep it iff it is independent, i.e. reduction is nonzero
    span = Sr.copy()
    span_piv = list(spiv)
    for v in K:
        Pv, _ = _pack(v[None, :])
        w = Pv[0].copy()
        # eliminate against current span (rows in rref-ish order)
        for r0 in range(span.shape[0]):
            c = span_piv[r0]
            if (w[c >> 6] >> np.uint64(c & 63)) & np.uint64(1):
                w ^= span[r0]
        if not w.any():
            continue
        # first set bit becomes this row's pivot
        nzw = np.flatnonzero(w)
        word = int(nzw[0])
        low = int(w[word]) & -int(w[word])          # lowest set bit, python int
        bit = word * 64 + low.bit_length() - 1
        # clear the new pivot bit from the existing span rows, then append
        hit = np.flatnonzero((span[:, bit >> 6] >> np.uint64(bit & 63)) & np.uint64(1))
        if len(hit):
            span[hit] ^= w
        span = np.vstack([span, w[None, :]])
        span_piv.append(bit)
        out.append(_unpack(w[None, :], H.shape[1])[0])
    return np.array(out, dtype=np.uint8)


if __name__ == "__main__":
    # self-test vs the unpacked reference on small codes
    import sys
    sys.path.insert(0, "/u/zdliu/lp/cand")
    rng = np.random.default_rng(5)

    H7 = np.array([[0, 0, 0, 1, 1, 1, 1],
                   [0, 1, 1, 0, 0, 1, 1],
                   [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)
    L = logicals(H7, H7)
    assert L.shape[0] == 1 and not np.any((H7 @ L.T) % 2)
    assert rank(H7) == 3 and kernel_basis(H7).shape == (4, 7)

    for _ in range(20):
        m, n = rng.integers(4, 30), rng.integers(10, 60)
        M = (rng.random((m, n)) < 0.4).astype(np.uint8)
        K = kernel_basis(M)
        assert K.shape[0] == n - rank(M)
        if K.shape[0]:
            assert not np.any((M @ K.T) % 2)

    # Steane code (H7 with itself): k = 7 - 3 - 3 = 1
    assert 7 - 2 * rank(H7) == 1
    try:
        from sqetch_smoke import logicals as slow_logicals
        Ls = np.asarray(slow_logicals(H7, H7), dtype=np.uint8) % 2
        assert Ls.shape[0] == L.shape[0] == 1
        print("cross-checked against sqetch_smoke.logicals")
    except ImportError:
        print("(sqetch_smoke not importable here; membership checks passed)")
    print("fast_gf2 self-test OK")
