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
CONCEPT = ("For SL(2,F5), order 120, shapes (1,2)x(3,4), prescribed constituent column degrees and rejection of constituent kernel words of weight <=4 will improve on the parent's repeated weight-six logical readings.")
GROUP = ("SL(2,F5) is a new nonabelian matrix-group bed with trivial abelianization, unlike the parent's Z2xZ22. Order 120 gives n=1320; noncommutativity is a hypothesis to test, not a distance certificate.")
SHAPES = ("(1,2)x(3,4) guarantees k>=120 and rate>=1/11, targeting the empty 0.100 cell. Constituents have lengths 240/480 and 120/360 checks. The parent's measured lightest logicals were in sector-1 Acol/Bcol blocks, whose degrees these supports strengthen.")
FILLING = ("Twenty deliberate degree cards use A counts (4,4) or (3,5), singleton B cells and at most one doubled cell per B column. Both check-weight sums are <=12. Screen 24 labelings per card using verified lifts, constituent words through weight four, pair expansion, four-cycles and actual rank; none of these certifies CSS distance.")
HOW_AB = FILLING
FAMILY_NAME = "r010_SL2F5_12x34_degreecards_cc4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("SL(2,F5) has order 120")
    els = [(a, b, c, d)
           for a in range(5) for b in range(5)
           for c in range(5) for d in range(5)
           if (a*d - b*c) % 5 == 1]
    pos = {g: i for i, g in enumerate(els)}

    def red(g):
        return tuple(int(x) % 5 for x in g)

    def mul(g, h):
        a, b, c, d = g
        e, f, u, v = h
        return ((a*e + b*u) % 5, (a*f + b*v) % 5,
                (c*e + d*u) % 5, (c*f + d*v) % 5)

    def inv(g):
        a, b, c, d = g
        return (d % 5, (-b) % 5, (-c) % 5, a % 5)

    def idx(g):
        return pos[red(g)]

    return dict(elements=els, ident=(1, 0, 0, 1),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 4)               # (rb, cb)


# Canonical partitions of the number of doubled B columns among its rows.
# These are support designs, not row/column permutation copies.
_DEGREE_PARTITIONS = (
    (0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0),
    (3, 0, 0), (2, 1, 0), (1, 1, 1), (4, 0, 0),
    (3, 1, 0), (2, 2, 0), (2, 1, 1)
)
_ASYMMETRIC_CARDS = (1, 2, 3, 4, 5, 6, 8, 9, 10)


def _degree_card(idx):
    card = idx % 20
    if card < 11:
        sa = [[4, 4]]
        partition = _DEGREE_PARTITIONS[card]
    else:
        sa = [[3, 5]]
        partition = _DEGREE_PARTITIONS[
            _ASYMMETRIC_CARDS[card - 11]]
    sb = [[1] * 4 for _ in range(3)]
    column = 0
    for row, count in enumerate(partition):
        for _ in range(count):
            sb[row][column] = 2
            column += 1
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _element_orders(G):
    orders = []
    for g in G.elements:
        h = G.e
        for order in range(1, G.q + 1):
            h = G.mul(h, g)
            if h == G.e:
                orders.append(order)
                break
        else:
            raise ValueError("element order failed")
    return orders


def _screened_cell(rng, G, count, orders, minimum_order):
    """Distinct terms without small-order internal relative labels."""
    for _ in range(2048):
        cell = rng.sample(G.elements, count)
        legal = True
        for u in range(count):
            inverse = G.inv(cell[u])
            for v in range(u):
                relative = G.mul(inverse, cell[v])
                if orders[G.idx(relative)] < minimum_order:
                    legal = False
                    break
            if not legal:
                break
        if legal:
            return cell
    raise ValueError("could not form a relative-order legal cell")


def _a_generates_group(G, A):
    """Internal A differences must not remain in a proper subgroup."""
    generators = [
        G.mul(G.inv(cell[0]), g)
        for cell in A[0] for g in cell[1:]
    ]
    reached = {G.e}
    pending = [G.e]
    while pending:
        h = pending.pop()
        for g in generators:
            z = G.mul(h, g)
            if z not in reached:
                reached.add(z)
                pending.append(z)
    return len(reached) == G.q


def _constituent_profile(rows, width):
    """Exact kernel-word exclusion through weight four, then soft scores.

    Rows are restrictions of the VERIFIED lifted matrices, not an
    independently implemented left/right lift. Pair expansion and
    four-cycle counts are heuristics, not CSS distance lower bounds.
    """
    columns = [0] * width
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    singles = set(columns)
    if 0 in singles:
        return (1, 0, -10**9)
    if len(singles) != width:
        return (2, 0, -10**9)

    weights = [x.bit_count() for x in columns]
    pairs = set()
    expansion = 10**9
    squares = 0
    for i, a in enumerate(columns):
        for j in range(i):
            syndrome = a ^ columns[j]
            if syndrome in singles:
                return (3, 0, -10**9)
            if syndrome in pairs:
                return (4, 0, -10**9)
            pairs.add(syndrome)
            weight = syndrome.bit_count()
            expansion = min(expansion, weight)
            overlap = (weights[i] + weights[j] - weight) // 2
            squares += overlap * (overlap - 1) // 2
    return (5, expansion, -squares)


def _profiles_from_verified_lift(G, hx, hz):
    q = G.q
    mask = (1 << q) - 1
    # HX at B-column zero, restricted to its two sector-1 A blocks.
    a_rows = [
        (hx[r] & mask) |
        (((hx[r] >> (B_SHAPE[1] * q)) & mask) << q)
        for r in range(q)
    ]
    # HZ at A-column zero, restricted to the four sector-1 B blocks.
    b_mask = (1 << (B_SHAPE[1] * q)) - 1
    b_rows = [row & b_mask for row in hz[:B_SHAPE[0] * q]]
    return (_constituent_profile(a_rows, A_SHAPE[1] * q),
            _constituent_profile(b_rows, B_SHAPE[1] * q))


def NEW_CTX(rng, G, idx):
    """Choose a genuinely new labeling for the slot's deliberate card."""
    sa, sb = _degree_card(idx)
    orders = _element_orders(G)
    best = None
    best_score = None

    for _ in range(24):
        A = [[_screened_cell(rng, G, count, orders, 4)
              for count in sa[0]]]
        B = [[_screened_cell(rng, G, count, orders, 5)
              for count in row] for row in sb]
        if not _a_generates_group(G, A):
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Keep the entire program in the intended open-ended low-rate cell.
        if k <= 0 or 8 * k >= n:
            continue

        pa, pb = _profiles_from_verified_lift(G, hx, hz)
        score = (
            min(pa[0], pb[0]),
            min(pa[1], pb[1]),
            pa[2] + pb[2],
            -k
        )
        if best_score is None or score > best_score:
            best_score = score
            best = (A, B)

    if best is None:
        raise ValueError("no rank-legal labeling for degree card")
    A, B = best
    hx, hz, n = lifted_matrices(G, A, B)
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= 12
    return {"idx": idx, "cells": {"A": A, "B": B},
            "screen": best_score}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete, already-screened cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Weight legality plus uniform abelian labels does not eliminate light sector-1 logicals: all five measured parent codes had distance reading six, with minimum witnesses only in sector 1.
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
