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
CONCEPT = ("On dicyclic Dic4 of order 16 with A=(1,4), B=(2,7), "
           "screened noncommuting A gaps and B-column paths can remove "
           "weight-four sector-1 logicals at rate near 0.500.")
GROUP = ("Keep the dicyclic presentation, with parameter 4 and order "
         "16 so the factor-30 shape has n=480. The parent's order-36 "
         "codes gained k but all five measured distances remained four.")
SHAPES = ("(1,4)x(2,7) guarantees k>=240 and rate>=0.500. Its "
          "four-column one-row A and seven-column two-row B components "
          "have unequal lengths. The parent's lightest X/Z logicals "
          "occupied sector-1 Acol/Bcol blocks across both components.")
FILLING = ("Indexed profiles use A counts (3,2,2,2) with two disjoint "
           "B doubletons per row, or A counts (2,2,2,2) with three "
           "B doubletons per row. Both check weights stay <=12. Fresh "
           "order-eight, order-four and nonrotation gaps are screened "
           "on verified component lifts through weight four; exact k "
           "selects candidates in cell 0.500, testing the parent's "
           "weight-four witnesses and k above the new bound.")
FAMILY_NAME = "r050_dic4_14x27_two_row_component_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Dic4: x^8=1, y^2=x^4, y*x*y^-1=x^-1.
    m, half = 8, 4
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % 2)

    def idx(g):
        a, b = red(g)
        return m * b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-1 if b else 1) * c + half * b * d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + half) % m, 1)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _dic_step(rng, kind):
    """Nonidentity relative elements from distinct Dic4 order classes."""
    if kind == "r8":
        return (rng.choice((1, 3, 5, 7)), 0)
    if kind == "r4":
        return (rng.choice((2, 6)), 0)
    if kind == "r2":
        return (4, 0)
    return (rng.randrange(8), 1)  # Order four, outside <x>.


def _dic_pair(rng, G, kind):
    base = G.pick(rng)
    return [base, G.mul(base, _dic_step(rng, kind))]


def _dic_triple(rng, G, mode):
    """Three distinct elements with indexed rotation-coset occupancy."""
    if mode % 2:
        return rng.sample(G.elements, 3)
    rotations = rng.sample(range(8), 2)
    return [(rotations[0], 0), (rotations[1], 0),
            (rng.randrange(8), 1)]


def _component_relations(G, rows, side):
    """Count short relations in both check rows of the B component."""
    q = G.q
    blocks = A_SHAPE[1] if side == "A" else B_SHAPE[1]
    check_rows = q if side == "A" else B_SHAPE[0] * q
    stride = B_SHAPE[1] * q if side == "A" else q
    signatures = [0] * (blocks * q)
    mask = (1 << q) - 1
    for r, row in enumerate(rows[:check_rows]):
        for j in range(blocks):
            bits = (row >> (j * stride)) & mask
            while bits:
                bit = bits & -bits
                signatures[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit

    singles = set(signatures)
    seen = set()
    bad2 = int(0 in singles) + len(signatures) - len(singles)
    bad3 = bad4 = 0
    for i, first in enumerate(signatures):
        for second in signatures[i + 1:]:
            syndrome = first ^ second
            if syndrome in singles:
                bad3 += 1
            if syndrome in seen:
                bad4 += 1
            else:
                seen.add(syndrome)
    return bad2, bad3, bad4


def _draw_design(rng, G, idx, trial):
    """Independent labels for an indexed A-order and B-path motif."""
    mode = idx % 5
    order_mode = (idx // 5) % 4
    a_orders = (
        ("ord4", "r8", "ord4"),
        ("r8", "ord4", "r8"),
        ("ord4", "r4", "r8"),
        ("r8", "r8", "r4"),
    )[order_mode]
    if mode < 3:
        counts = [3, 2, 2, 2]  # A sum 9; B-column sum at most 3.
        doubles = 2            # B-row sum 9; HZ at most 9+3.
    else:
        counts = [2, 2, 2, 2]  # A sum 8; B-column sum at most 4.
        doubles = 3            # B-row sum 10; HZ at most 10+2.

    rng.shuffle(counts)
    kinds = list(a_orders)
    rng.shuffle(kinds)
    A = [[]]
    for count in counts:
        if count == 3:
            A[0].append(_dic_triple(rng, G, mode + trial))
        else:
            A[0].append(_dic_pair(rng, G, kinds.pop()))

    sites = rng.sample(range(7), 2 * doubles)
    heavy = (set(sites[:doubles]), set(sites[doubles:]))
    B = [[], []]
    b_kinds = (
        ("ord4", "r8", "r4"),
        ("r8", "ord4", "r8"),
        ("r4", "ord4", "r8"),
        ("ord4", "ord4", "r4"),
        ("r8", "r8", "ord4"),
    )[mode]
    for r in range(2):
        for j in range(7):
            if j not in heavy[r]:
                B[r].append([G.pick(rng)])
            else:
                kind = b_kinds[(j + r + trial) % 3]
                B[r].append(_dic_pair(rng, G, kind))
    return A, B


def NEW_CTX(rng, G, idx):
    """Screen full fresh designs, then rank the best verified lifts."""
    shortlist = []
    for trial in range(32):
        A, B = _draw_design(rng, G, idx, trial)
        hx, hz, n = lifted_matrices(G, A, B)
        sa = _component_relations(G, hx, "A")
        sb = _component_relations(G, hz, "B")
        short = tuple(sa[t] + sb[t] for t in range(3))
        shortlist.append((short, trial, A, B, hx, hz, n))
    shortlist.sort(key=lambda item: (item[0], item[1]))

    best = None
    for short, trial, A, B, hx, hz, n in shortlist[:9]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        in_cell = 240 <= k <= 251  # 252/480 reaches cell 0.550.
        quality = (int(short[0] == 0), int(short[1] == 0),
                   int(in_cell), int(short[2] == 0),
                   k if in_cell else -abs(k - 248),
                   -short[2], -trial)
        if best is None or quality > best[0]:
            best = (quality, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Balanced order-34 dihedral (1,3)x(1,5) fillings did not
# reliably remove short logicals: four of five measured slots had d30s=2.
# REFUTED: Order-22 two-term differences alone remove weight-four
# logicals on Dic11 (1,3)x(1,4): all five selected parent slots had d30s=4.
# REFUTED: Screening the parent's profiles keeps its selected codes in
# the 0.450 cell: their mean k/n was 0.486, in the 0.500 cell.
# REFUTED: Increasing k on Dic11 (1,3)x(1,4) removes its weight-four
# sector-1 logicals: all five current measured slots still had d30s=4.
# REFUTED: Higher k alone removes weight-four logicals on Dic9
# (1,3)x(1,5): all five measured slots had k=296 and d30s=4.
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
