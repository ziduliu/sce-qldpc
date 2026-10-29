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
CONCEPT = ("For metacyclic Z17:Z2, s=16, q=34, A=(2,5), B=(3,7), "
           "component-column degree >=3 plus short-syndrome screening should "
           "beat the parent's five distance readings of 4 at rate near 0.300.")
GROUP = ("Keep the metacyclic inversion family. The new size factor 41 "
         "requires a smaller order; q=34 gives n=1394. Prime odd rotation "
         "order 17 makes distinct same-coset binomial labels differ by a "
         "rotation of order 17, rather than an involution. This does not "
         "certify the quantum distance.")
SHAPES = ("A=(2,5), B=(3,7): k>=34*(5-2)*(7-3)=408, "
          "k/n>=12/41=0.292683. Five-variable/two-check and "
          "seven-variable/three-check components receive column degree >=3. "
          "The parent's measured weight-4 logicals occupied sector 1 across "
          "many Acol/Bcol blocks, so protection is distributed.")
FILLING = ("Twenty explicit support designs exchange A-column degree for "
           "B-row extras, with both check-weight maxima <=12. Binomials "
           "stay in one coset; larger cells mix cosets. Each slot selects "
           "among 24 proposals using short syndrome collisions and Tanner "
           "four-cycles of verified lifted matrices, then exact-rank screens "
           "for 0.275<=k/n<0.325. Local collision screening is not exhaustive.")
HOW_AB = FILLING
FAMILY_NAME = "r030_m17r2s16_25x37_degree3_collision_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 34

def custom_ops(q):
    # Same metacyclic inversion rule, with odd prime rotation order.
    # Z_17 x| Z_2: y x y^-1 = x^16, and 16^2 = 1 mod 17.
    m, r, s = 17, 2, 16
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


# Each entry specifies A-column degrees, B-row extra terms, and whether
# degree-four A columns use a balanced 2+2 rather than concentrated 1+3.
# The final two entries differ in cell multiplicities, not just labels.
_SUPPORT_DESIGNS = (
    ((3,3,3,3,3), (0,0,0), 0),
    ((3,3,3,3,3), (0,0,1), 0),
    ((3,3,3,3,3), (0,0,2), 0),
    ((3,3,3,3,3), (0,1,1), 0),
    ((3,3,3,3,3), (0,1,2), 0),
    ((3,3,3,3,3), (0,2,2), 0),
    ((3,3,3,3,3), (1,1,1), 0),
    ((3,3,3,3,3), (1,1,2), 0),
    ((3,3,3,3,3), (1,2,2), 0),
    ((3,3,3,3,3), (2,2,2), 0),
    ((4,3,3,3,3), (0,0,0), 0),
    ((4,3,3,3,3), (0,0,1), 0),
    ((4,3,3,3,3), (0,1,1), 0),
    ((4,3,3,3,3), (1,1,1), 0),
    ((4,4,3,3,3), (0,0,0), 0),
    ((5,3,3,3,3), (0,0,0), 0),
    ((4,4,4,3,3), (0,0,0), 0),
    ((5,4,3,3,3), (0,0,0), 0),
    ((4,4,3,3,3), (0,0,0), 1),
    ((4,3,3,3,3), (1,1,1), 1),
)


def _support_choices(degrees, balanced):
    """Enumerate nonzero cell splits with balanced A-row totals."""
    import itertools
    choices = []
    for d in degrees:
        if d == 3:
            choices.append((1, 2))
        elif d == 4:
            choices.append((2,) if balanced else (1, 3))
        else:
            choices.append((2, 3))
    total = sum(degrees)
    limit = (total + 1) // 2
    return [p for p in itertools.product(*choices)
            if max(sum(p), total - sum(p)) <= limit]


def _draw_cells(rng, G, support, bias):
    """Same-coset binomials; genuinely mixed larger cells."""
    pools = {
        b: [g for g in G.elements if g[1] == b]
        for b in (0, 1)
    }
    result = []
    for row in support:
        out = []
        for count in row:
            if count == 1:
                b = int(rng.random() < bias)
                cell = [pools[b][rng.randrange(len(pools[b]))]]
            elif count == 2:
                b = int(rng.random() < bias)
                cell = rng.sample(pools[b], 2)
            else:
                # Both cosets occur, with the favored coset receiving
                # the larger share when the cell has odd weight.
                favored = int(rng.random() < bias)
                nf = (count + 1) // 2
                cell = (rng.sample(pools[favored], nf)
                        + rng.sample(pools[1 - favored], count - nf))
            out.append(cell)
        result.append(out)
    return result


def _short_collision_score(rows, n):
    """Conservative local null-word surrogate plus Tanner four-cycles.

    Only pairs sharing a check are enumerated. Consequently this is a
    local screen, not an exhaustive weight-four logical search; it also
    conservatively penalizes short null words that might be stabilizers.
    """
    columns = [0] * n
    row_columns = []
    for r, word in enumerate(rows):
        positions = []
        while word:
            bit = word & -word
            c = bit.bit_length() - 1
            positions.append(c)
            columns[c] |= 1 << r
            word ^= bit
        row_columns.append(positions)

    singles = {}
    short = 0
    for c, syndrome in enumerate(columns):
        if syndrome == 0:
            short += 1000
        if syndrome in singles:
            short += 100
        else:
            singles[syndrome] = c

    pair_occurrences = {}
    pair_syndromes = {}
    cycles4 = 0
    for positions in row_columns:
        for t, a in enumerate(positions):
            for b in positions[t + 1:]:
                key = a * n + b
                old = pair_occurrences.get(key, 0)
                pair_occurrences[key] = old + 1
                cycles4 += old
                if old:
                    continue
                syndrome = columns[a] ^ columns[b]
                if syndrome == 0:
                    short += 100
                c = singles.get(syndrome)
                if c is not None and c != a and c != b:
                    short += 10
                if syndrome in pair_syndromes:
                    short += 1
                else:
                    pair_syndromes[syndrome] = key
    return short, cycles4


def NEW_CTX(rng, G, idx):
    """Select a deliberate support design and screen actual LP matrices."""
    degrees, extras, balanced = _SUPPORT_DESIGNS[idx % 20]
    splits = _support_choices(degrees, balanced)
    if not splits:
        raise ValueError("support design has no balanced split")

    # If the harness supplies more than twenty slots, subsequent cohorts
    # have distinct coset-concentration objectives as well as new labels.
    bias = (0.50, 0.25, 0.75)[(idx // 20) % 3]
    proposals = []
    for trial in range(24):
        p = splits[rng.randrange(len(splits))]
        sa = [list(p), [d - x for d, x in zip(degrees, p)]]
        sb = [[1] * 7 for _ in range(3)]

        # At most one extra per B column: B-column degree is 3 or 4.
        order = list(range(7))
        rng.shuffle(order)
        cursor = 0
        for i, extra in enumerate(extras):
            for j in order[cursor:cursor + extra]:
                sb[i][j] += 1
            cursor += extra

        # A degrees 3: A-row <=8, B-column <=4, B-row <=9.
        # A degrees 4: A-row <=9, B-row <=8 (or 7).
        # A degrees 5: B is monomial, so B-row=7.
        # These profiles give BOTH maximum row sums <=12.
        if _grid_weights(sa, sb) > 12:
            raise ValueError("explicit support exceeds weight cap")
        if not _no_dead_lines(sa, sb):
            raise ValueError("explicit support has a dead line")

        A = _draw_cells(rng, G, sa, bias)
        B = _draw_cells(rng, G, sb, 1.0 - bias)
        hx, hz, n = lifted_matrices(G, A, B)
        sx, cx = _short_collision_score(hx, n)
        sz, cz = _short_collision_score(hz, n)
        quality = (sx + sz, max(sx, sz), cx + cz)
        proposals.append((quality, trial, A, B, hx, hz, n))

    # Rank screen uses exactly the matrices used for collision screening.
    for quality, trial, A, B, hx, hz, n in sorted(
            proposals, key=lambda item: (item[0], item[1])):
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k > 0 and 275 * n <= 1000 * k < 325 * n:
            return {"idx": idx, "cells": {"A": A, "B": B},
                    "k": k, "quality": quality}
    raise ValueError("no proposal has exact rate in the target cell")


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without further randomness."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Cap-saturating rolled supports on Z26:Z2, (2,6)x(2,4), did not suppress small logicals: all five measured codes had d=4, with abundant sector-1 witnesses.
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
