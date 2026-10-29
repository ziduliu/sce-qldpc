"""Automorphism group and qubit orbits of a CSS code's Tanner graph (vertex colours: qubits,
X checks, Z checks), computed with pynauty.

Every generator is checked on the matrices (a permutation of the qubits that preserves both row
spaces); the orbits written are regenerated from the verified generators, and the output records
the matrix fingerprint so that exact_bb.py can refuse an orbit file that belongs to another code.

  python code_orbits.py --npz code.npz --n 585 --out orbits.json
"""
import argparse
import json
import sys

from exact_bb import InputError, load_code, matrix_fingerprint, archive_name, validate_orbits, atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        from pynauty import Graph, autgrp
    except ImportError:
        print("pynauty is required", file=sys.stderr)
        return 2
    try:
        HX, HZ = load_code(a.npz, a.n)
    except InputError as e:
        print("INPUT ERROR: %s" % e, file=sys.stderr)
        return 2
    n, mx, mz = HX.shape[1], HX.shape[0], HZ.shape[0]
    N = n + mx + mz          # vertices: qubits, X checks, Z checks
    adj = {v: [] for v in range(N)}
    for r in range(mx):
        for c in HX[r].nonzero()[0]:
            adj[int(c)].append(n + r)
    for r in range(mz):
        for c in HZ[r].nonzero()[0]:
            adj[int(c)].append(n + mx + r)
    g = Graph(N, directed=False, adjacency_dict=adj,
              vertex_coloring=[set(range(n)), set(range(n, n + mx)), set(range(n + mx, N))])
    gens, grpsize1, grpsize2, orbits, norb = autgrp(g)
    qgens = []
    for gi, perm in enumerate(gens):
        qp = [int(v) for v in perm[:n]]
        if sorted(qp) != list(range(n)):
            print("generator %d does not map qubits to qubits" % gi, file=sys.stderr)
            return 3
        qgens.append(qp)
    rec = dict(npz=a.npz, n=n, matrix_sha256=matrix_fingerprint(HX, HZ), archive_name=archive_name(HX, HZ),
               aut_order=[grpsize1, grpsize2], n_generators=len(qgens), generators=qgens)
    try:
        orbs = validate_orbits(rec, HX, HZ)     # checks every generator on both row spaces, regenerates orbits
    except InputError as e:
        print("GENERATOR CHECK FAILED: %s" % e, file=sys.stderr)
        return 3
    nauty_orbits = {}
    for i, lab in enumerate(orbits[:n]):
        nauty_orbits.setdefault(lab, []).append(i)
    if sorted(sorted(o) for o in nauty_orbits.values()) != sorted(orbs):
        print("orbits from the generators differ from nauty's qubit orbits", file=sys.stderr)
        return 3
    rec["orbits"] = orbs
    atomic_json(rec, a.out)
    print("n=%d mx=%d mz=%d |Aut|=%.3ge%d generators=%d qubit orbits=%d (sizes %s); written to %s" % (
        n, mx, mz, grpsize1, grpsize2, len(qgens), len(orbs), sorted(set(len(o) for o in orbs)), a.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
