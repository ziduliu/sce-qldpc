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
CONCEPT = ("Metacyclic Z29:Z2 with inversion, q=58, shapes (1,3)x(1,3): distinct reflection-gap schedules screened for short component kernels should improve the rate-0.400 frontier's d30s=4 readings.")
GROUP = ("Retain the parent's metacyclic inversion construction; resize m=59 to m=29 for the new size factor. q=58 is the largest even order with 10q<600. The existing q=58 rolled bed scored 6.4, so improvement from this filling remains a testable hypothesis.")
SHAPES = ("Equal (1,3) sides give n=580, bound k=232 and rate 0.400, with equal-length 174-variable/58-check component matrices. Parent minimum logicals were in sector-1 Acol/Bcol blocks; screen both component kernels and the complete verified CSS matrices.")
FILLING = ("All six cells have identity plus two distinct reflections: three terms each, HX/HZ weight 12. Twenty prescribed gap schedules define separate designs. Optimize reflection centres within each schedule using weight-4 collisions, component weight-6 witnesses and overlaps; verify CSS and exact k=232 before returning cells. Component screening is not a quantum-distance certificate.")
FAMILY_NAME = "r040_m29r2s28_13x13_gap_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 58

def custom_ops(q):
    # Same metacyclic inversion construction, resized for n=10q.
    # Z_29 x| Z_2: y x y^-1 = x^28, and 28^2 = 1 mod 29.
    m, r, s = 29, 2, 28
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


# Each slot prescribes a different B gap set, rather than taking an
# automorphism/permutation image of one filling. A fixes the scale.
_GAP_DESIGNS = (
    (1, 3, 5), (1, 3, 7), (1, 3, 9), (1, 3, 11), (1, 3, 13),
    (1, 4, 6), (1, 4, 8), (1, 4, 10), (1, 4, 12), (1, 4, 14),
    (2, 3, 6), (2, 3, 8), (2, 3, 10), (2, 3, 12), (2, 3, 14),
    (2, 5, 7), (2, 5, 9), (2, 5, 11), (2, 5, 13), (3, 6, 10),
)


def _screen_columns(rows, nc):
    """Transpose rows obtained from the verified construction helper."""
    cols = [0] * nc
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            low = row & -row
            cols[low.bit_length() - 1] |= bit
            row ^= low
    return cols


def _screen_four(cols, cap=64):
    """Count capped pair-syndrome collisions; zero excludes kernels <=4.

    All columns screened here have odd weight, excluding odd kernels.
    With distinct nonzero columns, colliding different pairs are disjoint.
    """
    if 0 in cols or len(set(cols)) != len(cols):
        return cap
    seen = set()
    hits = 0
    for a in range(len(cols) - 1):
        x = cols[a]
        for b in range(a + 1, len(cols)):
            syndrome = x ^ cols[b]
            if syndrome in seen:
                hits += 1
                if hits >= cap:
                    return cap
            else:
                seen.add(syndrome)
    return hits


def _screen_component(rows, q):
    cols = _screen_columns(rows, 3 * q)
    four = _screen_four(cols)
    # A representative in each fibre suffices for this overlap objective.
    # These overlaps are a selection heuristic, not a distance bound.
    cycles = 0
    for a in (0, q, 2 * q):
        x = cols[a]
        for b, y in enumerate(cols):
            if a != b:
                overlap = (x & y).bit_count()
                cycles += overlap * (overlap - 1) // 2
    return cols, four, cycles


def _screen_six(cols, q, cap=128):
    """Capped weight-6 component witnesses using anchored triple matching.

    The component lifts have a transitive group action on each fibre.
    Translate any six-support to include one of 0,q,2q, then split it
    into two disjoint triples. A zero result exhausts this test.
    """
    length = len(cols)
    bits = [1 << a for a in range(length)]
    anchored = {}
    for a in (0, q, 2 * q):
        for b in range(length - 1):
            if b == a:
                continue
            ab = cols[a] ^ cols[b]
            mask_ab = bits[a] | bits[b]
            for c in range(b + 1, length):
                if c == a:
                    continue
                key = ab ^ cols[c]
                anchored.setdefault(key, []).append(mask_ab | bits[c])

    hits = 0
    lookup = anchored.get
    for a in range(length - 2):
        for b in range(a + 1, length - 1):
            ab = cols[a] ^ cols[b]
            mask_ab = bits[a] | bits[b]
            for c in range(b + 1, length):
                mates = lookup(ab ^ cols[c])
                if mates:
                    mask = mask_ab | bits[c]
                    for other in mates:
                        if not (mask & other):
                            hits += 1
                            if hits >= cap:
                                return cap
    return hits


def NEW_CTX(rng, G, idx):
    """Optimize centres within a prescribed algebraic gap design."""
    q = G.q
    m = q // 2
    gaps_a = (1, 2, 4)
    gaps_b = _GAP_DESIGNS[idx % len(_GAP_DESIGNS)]
    # Additional harness slots retain the design schedule but optimize
    # fresh centres with a different tie-breaking tradeoff.
    tradeoff = 1 + (idx % 5)
    mask_q = (1 << q) - 1
    mask_component = (1 << (3 * q)) - 1
    pool = []

    for attempt in range(32):
        grids = []
        for gaps in (gaps_a, gaps_b):
            row = []
            for gap in gaps:
                centre = rng.randrange(m)
                row.append([G.e, (centre, 1),
                            ((centre + gap) % m, 1)])
            grids.append([row])
        A, B = grids
        hx, hz, n = lifted_matrices(G, A, B)

        # First HX beta-block exposes A's left-lift component.
        # First HZ j-block exposes B's right-lift component.
        # Extract both from the verified matrices, not a separate lift.
        ar = [
            sum(((row >> (j * 3 * q)) & mask_q) << (j * q)
                for j in range(3))
            for row in hx[:q]
        ]
        br = [row & mask_component for row in hz[:q]]
        ac, af, ao = _screen_component(ar, q)
        bc, bf, bo = _screen_component(br, q)
        cheap = (max(af, bf), af + bf,
                 5 * max(ao, bo) + tradeoff * (ao + bo))
        pool.append((cheap, attempt, A, B, hx, hz, n, ac, bc))
        pool.sort(key=lambda item: (item[0], item[1]))
        if len(pool) > 6:
            pool.pop()

    finalists = []
    for cheap, attempt, A, B, hx, hz, n, ac, bc in pool:
        if cheap[0]:
            # Do not spend the six-support budget on an existing
            # shorter component witness.
            six_a = six_b = 10000 + cheap[0]
        else:
            six_a = _screen_six(ac, q)
            six_b = _screen_six(bc, q)
        score = (max(six_a, six_b), six_a + six_b, cheap, attempt)
        finalists.append((score, A, B, hx, hz, n))
    finalists.sort(key=lambda item: item[0])

    fallback = None
    for score, A, B, hx, hz, n in finalists:
        # The reported k is computed only with the verified ranks.
        if n - gf2_rank(hx) - gf2_rank(hz) != 4 * q:
            continue
        if fallback is None:
            fallback = (A, B, hx, hz)
        # Screen the COMPLETE opposite-check kernels as well.
        # Rejecting even a short stabilizer is conservative here.
        if _screen_four(_screen_columns(hx, n), cap=1):
            continue
        if _screen_four(_screen_columns(hz, n), cap=1):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS check")
        return {"idx": idx, "cells": {"A": A, "B": B}}

    # Preserve a legal exact-rate design if the stronger screen fails;
    # the evaluator, not this heuristic, determines its distance.
    if fallback is None:
        raise ValueError("no exact-rate candidate in prescribed gap design")
    A, B, hx, hz = fallback
    if not css_ok(hx, hz):
        raise ValueError("verified construction failed CSS check")
    return {"idx": idx, "cells": {"A": A, "B": B}}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without further sampling."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Cap-legal rolled fillings do not reliably avoid distance-2 logicals: parent slots 6, 9, 11 and many others returned d2s=2.
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
