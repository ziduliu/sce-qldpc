"""Initial program: lifted-product CSS code, ONE (group, shape) bed per
program, N_SLOTS (env CM_NSLOTS) freely-filled candidates per generation.

SEARCH SPACE (the only constraints; everything else is free):
- The group and the protograph shapes are fixed for the whole program
  (genotype, editable between generations). Each of the 20 slots fills
  the cells however it wants: PER SLOT, every cell holds 0..5 DISTINCT
  group elements (an empty cell is a zero block) - term counts AND
  elements are both the slot's choice. Identical fillings are
  deduplicated downstream and waste the slot.
- Size law: n = (ca*cb + ra*rb) * q for A of shape (ra, ca), B of shape
  (rb, cb), |G| = q. n must land in [CM_NMIN, CM_NMAX) (default
  [100, 600)); the scaffold picks the largest valid q.
- PER-SLOT validity (scaffold-checked; a violating slot is voided, not
  repaired): max row weight <= CM_WMAX (an HX row sums A-row i with
  B-column j; an HZ row sums a B-row with an A-column - check BOTH),
  no all-zero protograph row or column on either side (a zero A-row
  leaves its sector-2 block untouched by every Z check, which is a
  weight-1 logical - d=1 - unless the matching B column is a single
  1-term cell; symmetric for the other three cases), cells distinct,
  sizes 0..5.
- k > 0 required; k is measured on the built matrices per slot.

GROUPS: GTYPE "cyclic" (Z_q), "abelian" (Z_A0 x Z_{q/A0}) or "custom".
A custom group is a code string defining valid_q(q) -> bool and
custom_ops(q) -> dict(elements, ident, mul, inv). Any finite group you
can write down qualifies; the scaffold validates identity, inverses and
associativity on samples before building.
"""
import json
import os
import random

N_MIN = int(os.environ.get("CM_NMIN", "100"))
N_MAX = int(os.environ.get("CM_NMAX", "600"))
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Metacyclic Z8:Z1, s=1, order 8, shapes (3,8)x(3,8): predominantly degree-three component columns and balanced short-kernel screening should improve worse-basis circuit fitness above the rolled parent's 4.0702.")
GROUP = ("Retain GTYPE custom and the same metacyclic construction rule with r=1 and s=1. The new factor 73 makes order 17 inadmissible; order 8 is the largest compatible cyclic lift and gives n=584. This bed is untried. Over the power-of-two cyclic lift, augmentation ranks motivate full-rank A and minimally deficient B, but actual ranks are checked using the verified matrices.")
SHAPES = ("Target cell 0.350 with equal (3,8) components: each has 64 variables and 24 checks. n=73*8=584, kb=25*8=200, guaranteed rate 25/73=0.342466. Require measured k=210, rate 0.359589. Parent failures and weight-five witnesses spread across Acol/Bcol sector-1 blocks with only 1.1-1.2x enrichment, motivating global redundancy rather than a single-block repair.")
FILLING = ("Twenty deliberate profiles combine two or three A double cells in distinct rows/columns with ten normalized B-voltage parity distributions. Other A cells and all B cells are monomials. A row/column totals are <=9/4 and B totals are 8/3, so HX and HZ are <=12. Each slot searches 96 labelings through verified lifts, retains six by short component kernels and cycles, then screens full-code short syndromes and component weight-five/six collisions. Require k=210 and CSS validity. The higher CNOT count is a noise-cost tradeoff, not a predicted fitness gain; the hypothesis targets widespread weak columns and slot 50's 125/1000 Z versus 100/13500 X failed shots.")
FAMILY_NAME = "r035_m8r1s1_38x38_degree3_paritykernel"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    # Same metacyclic rule; order reduction is required by factor 73.
    # Z_8 x| Z_1: y x y^-1 = x^1.
    if q != 8:
        raise ValueError("this shape requires the order-eight lift")
    m, r, s = 8, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 8)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def _columns(rows, n):
    cols = [0] * n
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= tag
            row ^= low
    return cols


def _small_kernel(cols):
    """Singleton/pair syndrome statistics, ordered by short support."""
    singles = {}
    for x in cols:
        singles[x] = singles.get(x, 0) + 1
    n2 = sum(v * (v - 1) // 2 for v in singles.values())
    pairs = {}
    n3 = n4 = 0
    for i, x in enumerate(cols):
        for j in range(i):
            s = x ^ cols[j]
            n3 += singles.get(s, 0)
            old = pairs.get(s, 0)
            n4 += old
            pairs[s] = old + 1
    # Raw collision counts remain useful when shorter words exist.
    return (singles.get(0, 0), n2, n3, n4), pairs


def _long_kernel(cols, pairs):
    """Pair/triple and triple/triple collision multiplicities.

    With no words of weight <=4 these count weight-five/six
    supports up to fixed partition factors. Otherwise they are proxies.
    """
    triples = {}
    five = six = 0
    for i in range(2, len(cols)):
        x = cols[i]
        for j in range(1, i):
            xy = x ^ cols[j]
            for k in range(j):
                s = xy ^ cols[k]
                five += pairs.get(s, 0)
                old = triples.get(s, 0)
                six += old
                triples[s] = old + 1
    return five, six


def _cycles4(rows):
    total = 0
    for i, x in enumerate(rows):
        for j in range(i):
            t = (x & rows[j]).bit_count()
            total += t * (t - 1) // 2
    return total


def _balanced(a, b):
    return tuple(v for x, y in zip(a, b)
                 for v in (max(x, y), x + y))


def _components(G, hx, hz):
    """Extract both components from the scaffold's actual lifts."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            compact = 0
            for j in range(ca):
                compact |= ((source >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
    return ar, br


def _degree3_design(rng, G, doubles, occupancy):
    """Two support types and ten inequivalent parity-count targets."""
    q = G.q
    e = G.e
    A = [[[e] for _ in range(8)]]
    A.extend([[[G.pick(rng)] for _ in range(8)] for _ in range(2)])
    for i in range(doubles):
        # Odd differences have order eight, rather than two or four.
        anchor = A[i][i][0]
        delta = rng.randrange(1, q, 2)
        A[i][i].append(G.mul(anchor, (delta, 0)))

    # Normalize B's first row by column gauges. The other two
    # voltages have a deliberate parity occupancy, not a rolled grid.
    voltages = []
    for count, (pu, pv) in zip(
            occupancy, ((0, 0), (0, 1), (1, 0), (1, 1))):
        options = [(u, v) for u in range(pu, q, 2)
                   for v in range(pv, q, 2)]
        voltages.extend(rng.sample(options, count))
    rng.shuffle(voltages)
    B = [
        [[e] for _ in voltages],
        [[(u, 0)] for u, v in voltages],
        [[(v, 0)] for u, v in voltages],
    ]
    return A, B


def NEW_CTX(rng, G, idx):
    """Optimize labels within one of twenty deliberate designs."""
    profiles = (
        (2, 2, 2, 2),
        (1, 2, 2, 3),
        (1, 1, 3, 3),
        (1, 1, 2, 4),
        (1, 1, 1, 5),
        (0, 2, 3, 3),
        (0, 2, 2, 4),
        (0, 1, 3, 4),
        (0, 1, 2, 5),
        (0, 1, 1, 6),
    )
    design = idx % 20
    doubles = 2 + design // 10
    occupancy = profiles[design % 10]
    pool = []
    for attempt in range(96):
        A, B = _degree3_design(rng, G, doubles, occupancy)
        hx, hz, n = lifted_matrices(G, A, B)
        ar, br = _components(G, hx, hz)
        # Augmentation design predicts these ranks; verify the truth.
        if gf2_rank(ar) != 24 or gf2_rank(br) != 22:
            continue
        ac = _columns(ar, 64)
        bc = _columns(br, 64)
        sa, apairs = _small_kernel(ac)
        sb, bpairs = _small_kernel(bc)
        ca, cb = _cycles4(ar), _cycles4(br)
        quick = (_balanced(sa, sb)
                 + _balanced((ca,), (cb,))
                 + (rng.random(),))
        if len(pool) == 6 and quick >= pool[-1][0]:
            continue
        pool.append((quick, A, B, hx, hz, n, ac, bc,
                     sa, sb, apairs, bpairs, ca, cb))
        pool.sort(key=lambda item: item[0])
        pool = pool[:6]

    best = None
    for item in pool:
        (_, A, B, hx, hz, n, ac, bc,
         sa, sb, apairs, bpairs, ca, cb) = item
        if n - gf2_rank(hx) - gf2_rank(hz) != 210:
            continue
        # Full-code statistics catch mixed-sector short words.
        sx, _ = _small_kernel(_columns(hx, n))
        sz, _ = _small_kernel(_columns(hz, n))
        la = _long_kernel(ac, apairs)
        lb = _long_kernel(bc, bpairs)
        key = (_balanced(sx, sz)
               + _balanced(sa, sb)
               + _balanced(la, lb)
               + _balanced((ca,), (cb,)))
        if best is None or key < best[0]:
            best = key, A, B, hx, hz

    if best is None:
        raise ValueError("no verified k=210 candidate")
    _, A, B, hx, hz = best
    sa = [[len(cell) for cell in row] for row in A]
    sb = [[len(cell) for cell in row] for row in B]
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) <= W_MAX
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert css_ok(hx, hz)
    return {"cells": {"A": A, "B": B}, "design": design}


def CELL(rng, G, side, i, j, ctx):
    """Return the jointly selected full cell without subsequent draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A shared d2s=5 screen ensures balanced circuit protection in the parent bed; slot 50 had 125/1000 failed Z shots versus 100/13500 X shots, with s=3.892 despite X-basis s=5.132.
# EVOLVE-BLOCK-END


class _G(object):
    def __init__(self, ops, q):
        self.q = q
        self.elements = list(ops["elements"])
        self.e = ops["ident"]
        self.mul = ops["mul"]
        self.inv = ops["inv"]
        self._pos = {self._key(g): i for i, g in enumerate(self.elements)}

    @staticmethod
    def _key(g):
        return tuple(g) if isinstance(g, (list, tuple)) else g

    def idx(self, g):
        return self._pos[self._key(g)]

    def pick(self, rng):
        return self.elements[rng.randrange(self.q)]


# ---- verified construction helpers (2026-09-09) --------------------------
# lifted_matrices(G, cells_A, cells_B) builds EXACTLY the HX / HZ the
# evaluator builds (same lift conventions, same column order), as Python-int
# bit rows; gf2_rank(rows) is the rank behind k; lifted_k(G, A, B) is the k the table will report;
# css_ok(hx, hz) checks HX HZ^T = 0. Use these for any in-program screening -
# a hand-written lift that gets the left/right actions wrong screens a
# different (usually non-CSS) matrix pair than the one that is measured.
def lifted_matrices(G, cells_A, cells_B):
    """cells_A[i][j], cells_B[a][b]: lists of group elements (as CELL returns
    them) or element indices. Returns (hx_rows, hz_rows, n): lists of Python
    ints, bit c set = a 1 in column c; n = number of columns (qubits)."""
    q = G.q
    ra, ca = len(cells_A), len(cells_A[0])
    rb, cb = len(cells_B), len(cells_B[0])
    els = G.elements

    def gi(g):
        try:
            return G.idx(g)
        except (KeyError, TypeError):
            if isinstance(g, int) and 0 <= g < q:
                return g
            raise
    mul_idx = [[G.idx(G.mul(els[x], els[y])) for y in range(q)] for x in range(q)]
    inv_idx = [G.idx(G.inv(els[x])) for x in range(q)]
    A = [[sorted(set(gi(g) for g in cell)) for cell in row] for row in cells_A]
    B = [[sorted(set(gi(g) for g in cell)) for cell in row] for row in cells_B]
    n1 = ca * cb * q
    n = n1 + ra * rb * q
    hx = []
    for i in range(ra):
        for beta in range(cb):
            for r in range(q):
                row = 0
                for j in range(ca):                       # sector 1: A_{ij} left-lifted
                    base = (j * cb + beta) * q
                    for g in A[i][j]:
                        row ^= 1 << (base + mul_idx[inv_idx[g]][r])   # h with g*h = r
                for alpha in range(rb):                   # sector 2: B_{alpha beta}^T right-lifted
                    base = n1 + (i * rb + alpha) * q
                    for g in B[alpha][beta]:
                        row ^= 1 << (base + mul_idx[r][inv_idx[g]])   # c = r*g^-1
                hx.append(row)
    hz = []
    for j in range(ca):
        for alpha in range(rb):
            for r in range(q):
                row = 0
                for beta in range(cb):                    # sector 1: B_{alpha beta} right-lifted
                    base = (j * cb + beta) * q
                    for g in B[alpha][beta]:
                        row ^= 1 << (base + mul_idx[r][g])           # h with h*g^-1 = r
                for i in range(ra):                       # sector 2: A_{ij}^T left-lifted
                    base = n1 + (i * rb + alpha) * q
                    for g in A[i][j]:
                        row ^= 1 << (base + mul_idx[g][r])           # c = g*r
                hz.append(row)
    return hx, hz, n


def _popcount(x):
    return bin(x).count("1")



def gf2_rank(rows):
    """Rank over GF(2) of a list of bit rows."""
    pivots = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    return len(pivots)


def lifted_k(G, cells_A, cells_B):
    """k = n - rank(HX) - rank(HZ) of the code the evaluator will build."""
    hx, hz, n = lifted_matrices(G, cells_A, cells_B)
    return n - gf2_rank(hx) - gf2_rank(hz)


def css_ok(hx, hz):
    """True iff every X check commutes with every Z check (HX HZ^T = 0)."""
    return all(_popcount(x & z) % 2 == 0 for x in hx for z in hz)


def _grid_weights(sa, sb):
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    wx = max(sum(sa[i]) + sum(sb[r_][j] for r_ in range(rb))
             for i in range(ra) for j in range(cb))
    wz = max(sum(sb[r_]) + sum(sa[i][c] for i in range(ra))
             for r_ in range(rb) for c in range(ca))
    return max(wx, wz)


def _no_dead_lines(sa, sb):
    return all(
        all(sum(row) > 0 for row in g)
        and all(sum(r_[c] for r_ in g) > 0 for c in range(len(g[0])))
        for g in (sa, sb))


def roll_supports(rng):
    """Random per-cell term counts (1..CELL_MAX), repaired down to the
    weight cap with 1 as the floor - the roller never emits zero blocks
    (a slot's CELL may still return [] deliberately). A scaffold
    utility available to NEW_CTX; always returns legal grids."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    sa = [[rng.randint(1, CELL_MAX) for _ in range(ca)] for _ in range(ra)]
    sb = [[rng.randint(1, CELL_MAX) for _ in range(cb)] for _ in range(rb)]
    for _ in range(4000):
        if _grid_weights(sa, sb) <= W_MAX:
            return sa, sb
        g = sa if rng.random() < 0.5 else sb
        i = rng.randrange(len(g))
        c = rng.randrange(len(g[0]))
        if g[i][c] >= 2:            # 1 is the floor: the roller never
            g[i][c] -= 1            # emits zero blocks (evolution may)
    raise ValueError("roll_supports could not fit the weight cap "
                     "(all-ones row exceeds it: shape too wide)")


def _registry_ops(gtype, q, a0):
    if gtype == "cyclic":
        return dict(elements=list(range(q)), ident=0,
                    mul=lambda g, h: (g + h) % q,
                    inv=lambda g: (-g) % q)
    if gtype == "abelian":
        a = int(a0)
        if a < 1 or q % a:
            raise ValueError("A0 must divide q")
        b = q // a
        return dict(elements=[(x, y) for x in range(a) for y in range(b)],
                    ident=(0, 0),
                    mul=lambda g, h: ((g[0] + h[0]) % a, (g[1] + h[1]) % b),
                    inv=lambda g: ((-g[0]) % a, (-g[1]) % b))
    raise ValueError("GTYPE %r needs CUSTOM_GROUP" % gtype)


def _make_ops(q):
    if GTYPE == "custom":
        ns = {}
        exec(CUSTOM_GROUP, ns)
        if not ns["valid_q"](q):
            raise ValueError("valid_q rejects q=%d" % q)
        return ns["custom_ops"](q)
    return _registry_ops(GTYPE, q, A0)


def _valid_q(q):
    if GTYPE == "custom":
        ns = {}
        exec(CUSTOM_GROUP, ns)
        return bool(ns["valid_q"](q))
    if GTYPE == "abelian":
        return A0 is not None and int(A0) >= 1 and q % int(A0) == 0
    return q >= 2


def _validate_group(G):
    rng = random.Random(7)
    sample = [G.e] + [G.pick(rng) for _ in range(min(12, G.q))]
    for g in sample:
        assert G._key(G.mul(G.e, g)) == G._key(g), "identity fails"
        assert G._key(G.mul(g, G.inv(g))) == G._key(G.e), "inverse fails"
    for g in sample[:6]:
        for h in sample[:6]:
            for k in sample[:6]:
                assert G._key(G.mul(G.mul(g, h), k)) \
                    == G._key(G.mul(g, G.mul(h, k))), "associativity fails"
    assert len(G._pos) == len(G.elements) == G.q, "element list broken"


def _choose_q():
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if min(ra, ca, rb, cb) < 1:
        raise ValueError("protograph sides need >= 1 row and >= 1 column")
    f = ca * cb + ra * rb
    best = None
    for q in range(max(2, N_MIN // f), N_MAX // f + 1):
        if f * q < N_MIN or f * q >= N_MAX:
            continue
        if _valid_q(q):
            best = q
    if best is None:
        raise ValueError("no valid q puts n in [%d, %d)" % (N_MIN, N_MAX))
    return best


def _fill_slot(rng, G, idx):
    ctx = NEW_CTX(rng, G, idx)
    grids = {}
    rgrids = {}
    for side, (rows, cols) in (("A", A_SHAPE), ("B", B_SHAPE)):
        grid = []
        rgrid = []
        for i in range(rows):
            row = []
            rrow = []
            for j in range(cols):
                cell = CELL(rng, G, side, i, j, ctx)
                if not isinstance(cell, (list, tuple)):
                    raise ValueError("%s[%d][%d]: not a list" % (side, i, j))
                if len(cell) > CELL_MAX:
                    raise ValueError("%s[%d][%d]: >%d terms"
                                     % (side, i, j, CELL_MAX))
                keys = {G._key(g) for g in cell}
                if len(keys) != len(cell):
                    raise ValueError("%s[%d][%d]: duplicate element"
                                     % (side, i, j))
                pairs = sorted((G.idx(g), G._key(g)) for g in cell)
                row.append([i for i, _ in pairs])
                rrow.append([list(e) if isinstance(e, tuple) else e
                             for _, e in pairs])
            grid.append(row)
            rgrid.append(rrow)
        grids[side] = grid
        rgrids[side] = rgrid
    sa = [[len(c) for c in row] for row in grids["A"]]
    sb = [[len(c) for c in row] for row in grids["B"]]
    w = _grid_weights(sa, sb)
    if w > W_MAX:
        raise ValueError("row weight %d exceeds cap %d" % (w, W_MAX))
    if not _no_dead_lines(sa, sb):
        raise ValueError("all-zero protograph row or column")
    return grids["A"], grids["B"], rgrids["A"], rgrids["B"]


def get_candidates():
    """-> request dict for the worker: shared group data + 20 slots."""
    q = _choose_q()
    ops = _make_ops(q)
    G = _G(ops, q)
    _validate_group(G)
    mul_table = [[G.idx(G.mul(G.elements[a], G.elements[b]))
                  for b in range(q)] for a in range(q)]
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    slots = []
    for s in range(N_SLOTS):
        rng = random.Random(BASE_SEED * 10007 + s)
        try:
            A, B, Ar, Br = _fill_slot(rng, G, s)
            slots.append({"idx": s, "A": A, "B": B, "Ar": Ar, "Br": Br})
        except Exception as e:
            slots.append({"idx": s, "error": str(e)[:120]})
    return {"family": FAMILY_NAME, "concept": CONCEPT,
            "gtype": GTYPE, "q": q, "mul": mul_table,
            "a_shape": [ra, ca], "b_shape": [rb, cb],
            "n": (ca * cb + ra * rb) * q,
            "slots": slots}


if __name__ == "__main__":
    r = get_candidates()
    ok = sum(1 for s in r["slots"] if "error" not in s)
    print(json.dumps({"family": r["family"], "q": r["q"], "n": r["n"],
                      "slots_ok": ok}))
