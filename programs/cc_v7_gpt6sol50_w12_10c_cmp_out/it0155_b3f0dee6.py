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
CONCEPT = ("On C3 x Q16 of order 48 with A=(1,2), B=(2,5), screened balanced components should yield more than one distance-12 code among five rate-0.250 measurements.")
GROUP = ("C3 x Q16 has centre of order six and is not the parent's Dic6. Order-48 (1,2)x(2,5) beds have reached fitness 37.4; this untried direct-product group tests that lift size with a different multiplication law.")
SHAPES = ("A=(1,2), B=(2,5) give n=576 and guaranteed k=144, rate 0.250. The one-check A and two-check B components have unequal lengths 96 and 240. The parent's lightest measured logicals occupied sector-1 Acol/Bcol blocks, so both components are screened.")
FILLING = ("Four distinct terms per A cell and eight per B row keep HX and HZ weights at most 12; B-column counts are at most four. Twenty indexed relative-order and support motifs receive independent labels. Verified component screens target the parent's distance-8/9 witnesses, while exact rank monitors k above 144.")
FAMILY_NAME = "r025_c3xq16_12x25_order_support_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # C3 x Q16: x^8=1, y^2=x^4, y*x*y^-1=x^-1;
    # the third coordinate is independent and central.
    els = [(a, b, c) for c in range(3)
           for b in range(2) for a in range(8)]

    def red(g):
        return (g[0] % 8, g[1] % 2, g[2] % 3)

    def idx(g):
        a, b, c = red(g)
        return 16 * c + 8 * b + a

    def mul(g, h):
        a, b, c = red(g)
        u, v, t = red(h)
        return ((a + (-u if b else u) + 4 * b * v) % 8,
                (b + v) % 2, (c + t) % 3)

    def inv(g):
        a, b, c = red(g)
        return ((a + 4 if b else -a) % 8, b, -c % 3)

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _component_columns(hx, hz, q):
    """Extract both component codes from the verified LP lifts."""
    mask = (1 << q) - 1
    acol = [0] * (2 * q)
    bcol = [0] * (5 * q)

    for r in range(q):
        row = hx[r]                 # first B column, sole A check row
        for j in range(2):
            bits = (row >> (j * 5 * q)) & mask
            while bits:
                bit = bits & -bits
                acol[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit

    for alpha in range(2):
        for r in range(q):
            bits = hz[alpha * q + r] & ((1 << (5 * q)) - 1)
            while bits:
                bit = bits & -bits
                bcol[bit.bit_length() - 1] |= 1 << (alpha * q + r)
                bits ^= bit
    return acol, bcol


def _short_obstructions(cols, rng):
    """Count relations through four and sample disjoint weight-five words."""
    singles = set(cols)
    bad2 = len(cols) - len(singles) + int(0 in singles)
    bad3 = bad4 = bad5 = 0
    pairs = {}
    for i, x in enumerate(cols):
        for j in range(i):
            syndrome = x ^ cols[j]
            bad3 += syndrome in singles
            if syndrome in pairs:
                bad4 += 1
            else:
                pairs[syndrome] = (i, j)
    for _ in range(1200):
        i, j, k = rng.sample(range(len(cols)), 3)
        match = pairs.get(cols[i] ^ cols[j] ^ cols[k])
        if match is not None and match[0] not in (i, j, k) \
                and match[1] not in (i, j, k):
            bad5 += 1
    return bad2, bad3, bad4, bad5


def _element_orders(G):
    """Buckets for genuine relative orders in C3 x Q16."""
    buckets = {order: [] for order in (4, 6, 8, 12, 24)}
    for g in G.elements:
        power = G.e
        for order in range(1, 25):
            power = G.mul(power, g)
            if power == G.e:
                if order in buckets:
                    buckets[order].append(g)
                break
    return buckets


def _ordered_pair(rng, G, buckets, order):
    first = G.pick(rng)
    second = G.mul(first, rng.choice(buckets[order]))
    return [first, second]


def _a_cell(rng, G, buckets, order, mode):
    cell = _ordered_pair(rng, G, buckets, order)
    first_coset = cell[0][2]
    third = rng.choice([g for g in G.elements
                        if g[2] == (first_coset + 1) % 3
                        and g not in cell])
    cell.append(third)
    fourth_coset = (first_coset + (1 if mode & 1 else 2)) % 3
    cell.append(rng.choice([g for g in G.elements
                            if g[2] == fourth_coset and g not in cell]))
    return cell


def _b_cell(rng, G, buckets, count, order):
    if count == 1:
        return [G.pick(rng)]
    cell = _ordered_pair(rng, G, buckets, order)
    if count == 3:
        coset = (cell[0][2] + 1) % 3
        cell.append(rng.choice([g for g in G.elements
                                if g[2] == coset and g not in cell]))
    return cell


def _b_support(rng, mode):
    """Eight terms per row, with each column containing three or four."""
    if mode & 1:
        rows = [[3, 2, 1, 1, 1], [1, 1, 2, 2, 2]]
    else:
        rows = [[2, 2, 2, 1, 1], [1, 1, 2, 2, 2]]
    permutation = rng.sample(range(5), 5)
    rows = [[row[j] for j in permutation] for row in rows]
    if rng.randrange(2):
        rows.reverse()
    return rows


def NEW_CTX(rng, G, idx):
    """Choose independently labelled candidates within indexed motifs."""
    orders = (24, 12, 8, 6, 4)
    a_order = orders[idx % 5]
    other_a_order = orders[(idx + 2) % 5]
    mode = (idx // 5) % 4
    b_orders = ((24, 8), (12, 24), (8, 12), (6, 24))[mode]
    buckets = _element_orders(G)
    shortlisted = []

    for _ in range(16):
        A = [[_a_cell(rng, G, buckets, a_order, mode),
              _a_cell(rng, G, buckets, other_a_order, mode + 1)]]
        counts = _b_support(rng, mode)
        B = [[_b_cell(rng, G, buckets, count,
                      b_orders[(i + j) % 2])
              for j, count in enumerate(row)]
             for i, row in enumerate(counts)]
        hx, hz, n = lifted_matrices(G, A, B)
        acol, bcol = _component_columns(hx, hz, G.q)
        sa = _short_obstructions(acol, rng)
        sb = _short_obstructions(bcol, rng)
        short_grade = (-(sa[0] + sb[0]), -(sa[1] + sb[1]),
                       -(sa[2] + sb[2]), -(sa[3] + sb[3]),
                       -max(sa[3], sb[3]), rng.random())
        shortlisted.append((short_grade, A, B, hx, hz, n))

    shortlisted.sort(key=lambda item: item[0], reverse=True)
    best = best_grade = None
    for short_grade, A, B, hx, hz, n in shortlisted[:5]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        grade = (144 <= k <= 157, short_grade, -abs(k - 150))
        if best_grade is None or grade > best_grade:
            best_grade = grade
            best = {"A": A, "B": B}
    return best


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# REFUTED: Uniform rolled order-24 labels did not reliably avoid short
# sector-two logicals: the parent's selected readings were 6, 7, 8, 8, 9.
# REFUTED: Extra k alone did not improve that bed: parent slot 25 had
# k-kb=12 but d2s=2, versus slot 58's k-kb=0 and d30s=9.
# REFUTED: The parent's order-48 (1,2)x(2,5) screening did not make
# distance 12 consistent: selected slots 5, 6 and 41 had d30s=10.
# REFUTED: Moving to Dic6 with two-check A and screened supports did not
# improve the 0.250 cell: fitness fell to 17.801, and the five measured
# distances were 8, 8, 9, 8 and 8.
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
