"""Random sampling and mutation of lifted-product constructions at three
levels: group family (L3), protograph shapes (L2) and cell entries (L1).
Also provides the lift and lifted-product builders used by the evaluators
(A lifts by the left regular action, B by the right; n = (na*nb+ma*mb)*|G|).

A spec is {gtype, a0, A_shape, B_shape, A, B};
build(spec, n) -> (H_X, H_Z); valid_size(spec, n); mutate(spec, rng, level).
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- group algebra
# Each group, given its order q, exposes: mul, inv, idx (element -> 0..q-1),
# elements (list of all q canonical elements). Entries are stored as small int
# tuples (a,b) and reduced into the group at lift time.

def group_ops(gtype: str, q: int, a0: int = 3):
    if gtype == "cyclic":
        l = q
        red = lambda g: (int(g[0]) % l, 0)
        mul = lambda g, h: ((int(g[0]) + int(h[0])) % l, 0)
        inv = lambda g: ((-int(g[0])) % l, 0)
        idx = lambda g: int(g[0]) % l
        elements = [(a, 0) for a in range(l)]
    elif gtype == "abelian":
        b0 = q // a0
        red = lambda g: (int(g[0]) % a0, int(g[1]) % b0)
        mul = lambda g, h: ((int(g[0]) + int(h[0])) % a0, (int(g[1]) + int(h[1])) % b0)
        inv = lambda g: ((-int(g[0])) % a0, (-int(g[1])) % b0)
        idx = lambda g: (int(g[0]) % a0) * b0 + (int(g[1]) % b0)
        elements = [(x, y) for x in range(a0) for y in range(b0)]
    elif gtype == "dihedral":
        m = q // 2
        red = lambda g: (int(g[0]) % m, int(g[1]) & 1)
        def mul(g, h):
            a1, b1 = int(g[0]) % m, int(g[1]) & 1
            a2, b2 = int(h[0]) % m, int(h[1]) & 1
            return ((a1 + (a2 if b1 == 0 else -a2)) % m, (b1 + b2) & 1)
        inv = lambda g: (((-int(g[0])) % m, 0) if (int(g[1]) & 1) == 0 else (int(g[0]) % m, 1))
        idx = lambda g: (int(g[1]) & 1) * m + (int(g[0]) % m)
        elements = [(a, b) for b in (0, 1) for a in range(m)]
    elif gtype == "dicyclic":
        m = q // 4
        red = lambda g: (int(g[0]) % (2 * m), int(g[1]) & 1)
        def mul(g, h):
            a1, b1 = int(g[0]) % (2 * m), int(g[1]) & 1
            a2, b2 = int(h[0]) % (2 * m), int(h[1]) & 1
            if b1 == 0:
                return ((a1 + a2) % (2 * m), b2)
            if b2 == 0:
                return ((a1 - a2) % (2 * m), 1)
            return ((a1 - a2 + m) % (2 * m), 0)
        inv = lambda g: (((-int(g[0])) % (2 * m), 0) if (int(g[1]) & 1) == 0
                         else ((int(g[0]) + m) % (2 * m), 1))
        idx = lambda g: (int(g[1]) & 1) * (2 * m) + (int(g[0]) % (2 * m))
        elements = [(a, b) for b in (0, 1) for a in range(2 * m)]
    else:
        raise ValueError(f"unknown group {gtype}")
    return dict(q=q, red=red, mul=mul, inv=inv, idx=idx, elements=elements)


def _action(g, ops, side: str) -> np.ndarray:
    q = ops["q"]; mul = ops["mul"]; idx = ops["idx"]; els = ops["elements"]
    M = np.zeros((q, q), dtype=np.uint8)
    if side == "left":
        for h in els:
            M[idx(mul(g, h)), idx(h)] = 1
    else:
        invg = ops["inv"](g)
        for h in els:
            M[idx(mul(h, invg)), idx(h)] = 1
    return M


def _poly_block(entry, ops, side: str) -> np.ndarray:
    q = ops["q"]; red = ops["red"]
    M = np.zeros((q, q), dtype=np.uint8)
    seen = set()
    for ab in entry:
        g = red(ab)
        if g in seen:
            continue
        seen.add(g)
        M ^= _action(g, ops, side)
    return M


def _lift(proto, ops, side: str) -> np.ndarray:
    q = ops["q"]
    rows = len(proto); cols = len(proto[0])
    M = np.zeros((rows * q, cols * q), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            M[i * q:(i + 1) * q, j * q:(j + 1) * q] ^= _poly_block(proto[i][j], ops, side)
    return M


def _lifted_product(A: np.ndarray, B: np.ndarray, q: int):
    ma, na = A.shape[0] // q, A.shape[1] // q
    mb, nb = B.shape[0] // q, B.shape[1] // q
    ncols = (na * nb + ma * mb) * q
    HX = np.zeros((ma * nb * q, ncols), dtype=np.uint8)
    HZ = np.zeros((na * mb * q, ncols), dtype=np.uint8)
    for i in range(ma):
        for j in range(na):
            Ab = A[i * q:(i + 1) * q, j * q:(j + 1) * q]
            for beta in range(nb):
                HX[(i * nb + beta) * q:(i * nb + beta + 1) * q,
                   (j * nb + beta) * q:(j * nb + beta + 1) * q] ^= Ab
    for i in range(ma):
        for alpha in range(mb):
            for beta in range(nb):
                Bt = B[alpha * q:(alpha + 1) * q, beta * q:(beta + 1) * q].T
                HX[(i * nb + beta) * q:(i * nb + beta + 1) * q,
                   (na * nb + i * mb + alpha) * q:(na * nb + i * mb + alpha + 1) * q] ^= Bt
    for j in range(na):
        for alpha in range(mb):
            for beta in range(nb):
                Bb = B[alpha * q:(alpha + 1) * q, beta * q:(beta + 1) * q]
                HZ[(j * mb + alpha) * q:(j * mb + alpha + 1) * q,
                   (j * nb + beta) * q:(j * nb + beta + 1) * q] ^= Bb
    for i in range(ma):
        for j in range(na):
            At = A[i * q:(i + 1) * q, j * q:(j + 1) * q].T
            for alpha in range(mb):
                HZ[(j * mb + alpha) * q:(j * mb + alpha + 1) * q,
                   (na * nb + i * mb + alpha) * q:(na * nb + i * mb + alpha + 1) * q] ^= At
    return HX, HZ


# ----------------------------------------------------------------- spec & build
GROUPS = ["cyclic", "abelian", "dihedral", "dicyclic"]   # families present in the seed pool
A0_CHOICES = [2, 3, 5]                                    # Z_a0 factor for the abelian family
MINQ = 6                                                  # minimum group order (non-degenerate)


def factor_of(spec) -> int:
    ma, na = spec["A_shape"]; mb, nb = spec["B_shape"]
    return na * nb + ma * mb


def _q_ok(gtype, q, a0):
    if q < MINQ:
        return False
    if gtype == "cyclic":
        return True
    if gtype == "abelian":
        return q % a0 == 0 and q // a0 >= 2
    if gtype == "dihedral":
        return q % 2 == 0 and q // 2 >= 3
    if gtype == "dicyclic":
        return q % 4 == 0 and q // 4 >= 2
    return False


def valid_size(spec, n: int) -> bool:
    n = int(n)
    f = factor_of(spec)
    if n % f != 0:
        return False
    return _q_ok(spec["gtype"], n // f, spec["a0"])


def first_valid_n(spec, lo: int, hi: int):
    return next((n for n in range(lo, hi + 1) if valid_size(spec, n)), None)


def build(spec, n: int):
    q = n // factor_of(spec)
    ops = group_ops(spec["gtype"], q, spec["a0"])
    A = _lift(spec["A"], ops, "left")
    B = _lift(spec["B"], ops, "right")
    HX, HZ = _lifted_product(A, B, q)
    if HX.shape[1] != n or HZ.shape[1] != n:
        raise ValueError("size mismatch")
    if np.any((HX @ HZ.T) & 1):
        raise ValueError("CSS condition violated")
    return HX.astype(np.uint8), HZ.astype(np.uint8)


# --------------------------------------------------------------- random sampling
def _rand_cell(rng, nterms_max=2, amax=12, bmax=2):
    if os.environ.get("MONOMIAL"):          # monomial baseline: exactly ONE group element per cell
        nterms_max = 1
    nt = int(rng.integers(1, nterms_max + 1))
    s = set()
    tries = 0
    while len(s) < nt and tries < 20:
        s.add((int(rng.integers(0, amax)), int(rng.integers(0, bmax))))
        tries += 1
    return [list(t) for t in s]


def random_spec(rng):
    gtype = GROUPS[int(rng.integers(0, len(GROUPS)))]
    a0 = A0_CHOICES[int(rng.integers(0, len(A0_CHOICES)))] if gtype == "abelian" else 3
    bmax = 2 if gtype in ("dihedral", "dicyclic") else (a0 if gtype == "abelian" else 1)
    ma = int(rng.integers(2, 7)); na = int(rng.integers(2, 7))
    mb = int(rng.integers(2, 7)); nb = int(rng.integers(2, 7))
    amax = 12
    A = [[_rand_cell(rng, 2, amax, bmax) for _ in range(na)] for _ in range(ma)]
    B = [[_rand_cell(rng, 2, amax, bmax) for _ in range(nb)] for _ in range(mb)]
    return {"gtype": gtype, "a0": a0, "A_shape": (ma, na), "B_shape": (mb, nb), "A": A, "B": B}


def _bmax_of(spec):
    g = spec["gtype"]
    return 2 if g in ("dihedral", "dicyclic") else (spec["a0"] if g == "abelian" else 1)


def mutate(spec, rng, level: int):
    """level 1: change one entry term; 2: change a shape + refill; 3: change the group."""
    import copy
    s = copy.deepcopy(spec)
    if level == 3:                                   # L3: new group, refill entries for it
        ns = random_spec(rng)
        ns["A_shape"], ns["B_shape"] = s["A_shape"], s["B_shape"]
        ma, na = ns["A_shape"]; mb, nb = ns["B_shape"]
        bmax = _bmax_of(ns)
        ns["A"] = [[_rand_cell(rng, 2, 12, bmax) for _ in range(na)] for _ in range(ma)]
        ns["B"] = [[_rand_cell(rng, 2, 12, bmax) for _ in range(nb)] for _ in range(mb)]
        return ns
    if level == 2:                                   # L2: resample protograph shape(s) + refill
        bmax = _bmax_of(s)
        # default: resample one protograph (A or B at random); L2_BOTH=1 resamples both
        both = bool(os.environ.get("L2_BOTH"))
        do_A = both or (rng.random() < 0.5)
        do_B = both or (not do_A)
        if do_A:
            s["A_shape"] = (int(rng.integers(2, 7)), int(rng.integers(2, 7)))
            ma, na = s["A_shape"]
            s["A"] = [[_rand_cell(rng, 2, 12, bmax) for _ in range(na)] for _ in range(ma)]
        if do_B:
            s["B_shape"] = (int(rng.integers(2, 7)), int(rng.integers(2, 7)))
            mb, nb = s["B_shape"]
            s["B"] = [[_rand_cell(rng, 2, 12, bmax) for _ in range(nb)] for _ in range(mb)]
        return s
    # level 1: change one term in one random cell
    bmax = _bmax_of(s)
    M = s["A"] if rng.random() < 0.5 else s["B"]
    i = int(rng.integers(0, len(M))); j = int(rng.integers(0, len(M[0])))
    cell = M[i][j]
    term = [int(rng.integers(0, 12)), int(rng.integers(0, bmax))]
    # monomial baseline: only CHANGE the single term (never add/remove -> stays 1 term)
    op = 0 if os.environ.get("MONOMIAL") else int(rng.integers(0, 3))
    if op == 0 or len(cell) == 0:
        cell[int(rng.integers(0, len(cell)))] = term
    elif op == 1 and len(cell) < 3:
        cell.append(term)
    elif op == 2 and len(cell) > 1:
        del cell[int(rng.integers(0, len(cell)))]
    else:
        cell[int(rng.integers(0, len(cell)))] = term
    M[i][j] = cell
    return s


# ----------------------------------------------------------------------- driver
def _rank2(Mx):
    Mx = Mx.copy() % 2; rows, cols = Mx.shape; pr = 0
    for c in range(cols):
        piv = next((i for i in range(pr, rows) if Mx[i, c]), None)
        if piv is None:
            continue
        Mx[[pr, piv]] = Mx[[piv, pr]]
        for i in range(rows):
            if i != pr and Mx[i, c]:
                Mx[i] ^= Mx[pr]
        pr += 1
        if pr == rows:
            break
    return pr


def quality(spec, n, p_phys=0.07, trials=500):
    import evaluator_layout_bp_osd as E
    try:
        hx, hz = build(spec, n)
        k = n - _rank2(hx) - _rank2(hz)
        if k <= 0:
            return None
        fr = E.estimate_bp_osd_failure_rate(hx, hz, p_phys=p_phys, trials=trials)
        fr = fr[0] if isinstance(fr, (tuple, list)) else fr
        return (int(k), round(1 - fr, 4))
    except Exception:
        return None


if __name__ == "__main__":
    # smoke test: every group family must yield a VALID CSS code with k>0.
    rng = np.random.default_rng(0)
    for gt in GROUPS:
        for _ in range(200):
            s = random_spec(rng)
            s["gtype"] = gt
            s["a0"] = 3 if gt != "abelian" else A0_CHOICES[int(rng.integers(0, 3))]
            bmax = _bmax_of(s)
            ma, na = s["A_shape"]; mb, nb = s["B_shape"]
            s["A"] = [[_rand_cell(rng, 2, 12, bmax) for _ in range(na)] for _ in range(ma)]
            s["B"] = [[_rand_cell(rng, 2, 12, bmax) for _ in range(nb)] for _ in range(mb)]
            n = first_valid_n(s, 200, 900)
            if not n:
                continue
            try:
                hx, hz = build(s, n)
                css = not np.any((hx @ hz.T) & 1)
                k = n - _rank2(hx) - _rank2(hz)
                print(f"{gt:9s} n={n} shape A{s['A_shape']} B{s['B_shape']} CSS={css} k={k}")
                break
            except Exception as e:
                continue
