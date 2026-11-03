#!/usr/bin/env python3
"""Checks a code of this folder and, optionally, logical operators found for it.

    python verify.py <HX.mtx> <HZ.mtx> [<witness.json>]

Prints n, k (from the GF(2) ranks), the largest check weights and whether H_X H_Z^T = 0. A witness file
(records/exact_bb_20261006_61/c2_W24_operators.json, records/witness_cell0_20261103/*.json) lists supports of
logical operators (0-based qubit indices); every X operator is checked to commute with all Z checks (H_Z x = 0)
and to lie outside the row space of H_X, every Z operator likewise, so that its weight is an upper bound on the
distance. Exit status 1 if any check fails. Requires numpy and scipy."""
import json
import sys

import numpy as np
from scipy.io import mmread


def gf2_rank(m):
    m = (np.asarray(m, dtype=np.uint8) & 1).copy()
    r = 0
    for c in range(m.shape[1]):
        rows = np.nonzero(m[r:, c])[0]
        if rows.size == 0:
            continue
        p = r + rows[0]
        if p != r:
            m[[r, p]] = m[[p, r]]
        others = np.nonzero(m[:, c])[0]
        m[others[others != r]] ^= m[r]
        r += 1
        if r == m.shape[0]:
            break
    return r


def supports(obj):
    """(type, support) of every operator in a witness file: lists 'wx' / 'wz' of supports, or
    'witnesses': {'x': [{'support': [...]}, ...], 'z': [...]}, anywhere in the file."""
    out = []
    if isinstance(obj, dict):
        for t in "xz":
            out += [(t, s) for s in obj.get("w" + t, [])]
            w = obj.get("witnesses")
            if isinstance(w, dict):
                out += [(t, e["support"]) for e in w.get(t, [])]
        for v in obj.values():
            if isinstance(v, (dict, list)):
                out += supports(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)) and not (v and isinstance(v[0] if isinstance(v, list) else None, int)):
                out += supports(v)
    return out


def main(hx_path, hz_path, witness_path=None):
    hx = (np.asarray(mmread(hx_path).todense(), dtype=np.int64) % 2).astype(np.uint8)
    hz = (np.asarray(mmread(hz_path).todense(), dtype=np.int64) % 2).astype(np.uint8)
    n = hx.shape[1]
    ok = hz.shape[1] == n and not ((hx.astype(np.int64) @ hz.T.astype(np.int64)) % 2).any()
    rx, rz = gf2_rank(hx), gf2_rank(hz)
    print("n = %d, k = %d (rank H_X = %d, rank H_Z = %d), check weights w_X = %d, w_Z = %d, H_X H_Z^T = 0: %s"
          % (n, n - rx - rz, rx, rz, hx.sum(1).max(), hz.sum(1).max(), ok))
    if witness_path:
        for t, s in supports(json.load(open(witness_path))):
            v = np.zeros(n, dtype=np.uint8)
            v[list(s)] = 1
            h_other, h_same, r_same = (hz, hx, rx) if t == "x" else (hx, hz, rz)
            commutes = not ((h_other.astype(np.int64) @ v) % 2).any()
            outside = gf2_rank(np.vstack([h_same, v])) == r_same + 1
            good = commutes and outside
            ok &= good
            print("%s operator of weight %d: commutes %s, outside the stabilizer group %s" % (t.upper(), v.sum(), commutes, outside))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
