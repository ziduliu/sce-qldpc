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
CONCEPT = ("Metacyclic Z12:Z2(s=7), q=24, shapes (1,2)x(3,7): minimum degree three, bounded density and short-dependency expurgation should improve circuit fitness over the parent's 5.312; this is a measured-pL hypothesis, not a distance claim.")
GROUP = ("Keep the exact parent group and order as a controlled comparison. Its q=24 bed achieved 5.312; the new size n=408 is legal without changing the metacyclic presentation.")
SHAPES = ("Target cell 0.250: (1,2)x(3,7) gives n=408 and k>=96, rate 4/17=0.235294. Component lengths/check counts are 48/24 and 168/72. Parent short witnesses lay entirely in sector 2, with diffuse Arow/Brow failure enrichment of only 1.1-1.2x; replace that geometry rather than patch one block.")
FILLING = ("Twenty explicit density profiles keep every qubit's X and Z degree >=3 and both maximum check weights <=12. Prefer B component girth >=6, expurgate component kernel words of weight <=4, minimize four-cycles in verified HX/HZ, then penalize A words of weights five and six. Select only verified k/n in [0.225,0.275). Lower baseline check weights and fewer small traps are intended to improve pL; parent d2s ordering did not predict measured fitness ordering.")
FAMILY_NAME = "r025_m12r2s7_12x37_degree3_expurgated"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # metacyclic Z_12 x| Z_2: y x y^-1 = x^7 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 12, 2, 7
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


# Distinct density designs, not automorphism images of one filling.
# B starts with one term everywhere; listed cells receive a second.
# For A=(3,3): B rows <=9, columns <=5, so HX<=11, HZ<=12.
# For A=(3,4): B rows <=8, columns <=5, so HX<=12, HZ<=12.
# For A=(4,4): B rows <=8, columns <=4, so HX<=12, HZ<=12.
_DENSITY_DESIGNS = (
    ((3, 3), ()),
    ((3, 3), ((0, 0),)),
    ((3, 3), ((0, 0), (1, 1))),
    ((3, 3), ((0, 0), (1, 0))),
    ((3, 3), ((0, 0), (1, 1), (2, 2))),
    ((3, 3), ((0, 0), (1, 0), (2, 1))),
    ((3, 3), ((0, 0), (0, 1), (1, 2), (2, 3))),
    ((3, 3), ((0, 0), (0, 1), (1, 0), (1, 1))),
    ((3, 3), ((0, 0), (0, 1), (1, 2), (1, 3), (2, 4))),
    ((3, 3), ((0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5))),
    ((3, 4), ()),
    ((3, 4), ((0, 0),)),
    ((3, 4), ((0, 0), (1, 1))),
    ((3, 4), ((0, 0), (1, 0))),
    ((3, 4), ((0, 0), (1, 1), (2, 2))),
    ((3, 4), ((0, 0), (1, 0), (2, 1))),
    ((4, 4), ()),
    ((4, 4), ((0, 0),)),
    ((4, 4), ((0, 0), (1, 1))),
    ((4, 4), ((0, 0), (1, 1), (2, 2))),
)


def _b_no_four(B, mt, iv):
    """Right-lift voltage prefilter only; final screening uses HX/HZ."""
    for row in B:
        seen = set()
        for cell in row:
            for g in cell:
                for h in cell:
                    if g != h:
                        t = mt[g][iv[h]]
                        if t in seen:
                            return False
                        seen.add(t)
    for i in range(len(B)):
        for a in range(i):
            seen = set()
            for j in range(len(B[0])):
                for g in B[i][j]:
                    for h in B[a][j]:
                        t = mt[g][iv[h]]
                        if t in seen:
                            return False
                        seen.add(t)
    return True


def _girth_b(rng, q, identity, extras, mt, iv):
    """Greedy monomial core, followed by constrained second terms."""
    B = [[[] for _ in range(7)] for _ in range(3)]
    for j in range(7):
        B[0][j] = [identity]
    for i in (1, 2):
        for j in range(7):
            choices = list(range(q))
            rng.shuffle(choices)
            found = False
            for g in choices:
                B[i][j] = [g]
                if _b_no_four(B, mt, iv):
                    found = True
                    break
            if not found:
                return None
    order = list(extras)
    rng.shuffle(order)
    for i, j in order:
        choices = [g for g in range(q) if g not in B[i][j]]
        rng.shuffle(choices)
        found = False
        for g in choices:
            B[i][j].append(g)
            if _b_no_four(B, mt, iv):
                found = True
                break
            B[i][j].pop()
        if not found:
            return None
    return B


def _screen_columns(rows, width):
    """Transpose a restriction of the verified matrices."""
    cols = [0] * width
    for i, row in enumerate(rows):
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= 1 << i
            row ^= bit
    return cols


def _kernel_le_four(cols):
    """Exact existence test for component kernel words of weight <=4."""
    if any(c == 0 for c in cols) or len(set(cols)) != len(cols):
        return True
    singles = set(cols)
    pairs = set()
    for i, c in enumerate(cols):
        for j in range(i):
            s = c ^ cols[j]
            if s in singles or s in pairs:
                return True
            pairs.add(s)
    return False


def _a_five_six(cols):
    """Short-word multiplicity tie-breaker, after excluding weights <=4."""
    pairs = {
        cols[i] ^ cols[j]
        for i in range(len(cols)) for j in range(i)
    }
    triples = {}
    five = six = 0
    for i in range(len(cols)):
        for j in range(i):
            s = cols[i] ^ cols[j]
            for h in range(j):
                t = s ^ cols[h]
                five += int(t in pairs)
                previous = triples.get(t, 0)
                six += previous
                triples[t] = previous + 1
    return five, six


def _matrix_four_count(rows):
    total = 0
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def NEW_CTX(rng, G, idx):
    """Bounded search within the slot's explicit density design."""
    aw, extras = _DENSITY_DESIGNS[idx % len(_DENSITY_DESIGNS)]
    sa = [list(aw)]
    sb = [[1] * B_SHAPE[1] for _ in range(B_SHAPE[0])]
    for i, j in extras:
        sb[i][j] += 1
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)

    q = G.q
    mt = [[G.idx(G.mul(g, h)) for h in G.elements]
          for g in G.elements]
    iv = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)
    mask = (1 << q) - 1
    bmask = (1 << (B_SHAPE[1] * q)) - 1
    best = None
    best_key = None
    best_matrices = None

    for _ in range(160):
        Bi = _girth_b(rng, q, identity, extras, mt, iv)
        if Bi is None:
            # A legal fallback proposal, still subject to all screening.
            Bi = [[rng.sample(range(q), sb[i][j])
                   for j in range(B_SHAPE[1])]
                  for i in range(B_SHAPE[0])]
        Ai = [[rng.sample(range(q), count) for count in aw]]
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in Ai]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in Bi]

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (0.225 * n <= k < 0.275 * n):
            continue

        # Extract both component matrices from the verified lift:
        # HX's beta=0 block contains A; HZ's j=0 block contains B.
        ar = [(row & mask) |
              (((row >> (B_SHAPE[1] * q)) & mask) << q)
              for row in hx[:q]]
        br = [row & bmask for row in hz[:B_SHAPE[0] * q]]
        ac = _screen_columns(ar, A_SHAPE[1] * q)
        bc = _screen_columns(br, B_SHAPE[1] * q)
        bad_a = _kernel_le_four(ac)
        bad_b = _kernel_le_four(bc)
        fours = _matrix_four_count(hx) + _matrix_four_count(hz)
        head = (int(bad_a) + int(bad_b), fours)
        if best_key is not None and head > best_key[:2]:
            continue
        tail = (10**9, 10**9) if bad_a else _a_five_six(ac)
        key = head + tail
        if best_key is None or key < best_key:
            best_key = key
            best = {"A": A, "B": B}
            best_matrices = (hx, hz)
        if key == (0, 0, 0, 0):
            break

    if best is None:
        raise ValueError("no filling passed the verified target-rate screen")
    hx, hz = best_matrices
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    return {"idx": idx, "cells": best}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete preselected cell, with no mutable shared state."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: distance-screen ordering equals circuit-fitness ordering; parent slot 58 had d2s=9, s=5.872, whereas slot 12 had d2s=8, s=6.018.
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
