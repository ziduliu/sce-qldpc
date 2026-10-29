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
CONCEPT = ("In dicyclic Dic5 of order 20, A=(2,5) and B=(2,5), screened zero/binomial profiles should produce five distinct cell-0.350 codes with fitness above 5.0.")
GROUP = ("Jump from cyclic Z26 to nonabelian Dic5. Measured order-20 (2,5)x(2,5) beds reach roughly 5.1-5.2, motivating this different order-20 regular action.")
SHAPES = ("Equal (2,5)x(2,5) sides give n=580, equal 100-variable components and guaranteed k>=180. Verified k>=189 is needed for cell 0.350. The parent's failures were dispersed across sector-1 Acol/Bcol data blocks, including Bcol4 and Bcol5, rather than concentrated at one check block.")
FILLING = ("Put four binomials and one zero in every row, with different zero columns on each side. Cross twenty rotation/reflection gap profiles, screen verified component lifts for short relations, and select target-cell k by verified rank. HX and HZ row weights are at most 8+4=12; the test is fewer limiting-basis failed shots, not a higher proxy.")
FAMILY_NAME = "r035_dic5_25x25_zero_binomial_gap_profiles_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Dic5 = <x,y | x^10=1, y^2=x^5, yxy^-1=x^-1>.
    els = [(a, b) for b in range(2) for a in range(10)]

    def red(g):
        return (int(g[0]) % 10, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-1 if b else 1) * c + 5 * b * d) % 10,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 10, 0) if b == 0 else ((a + 5) % 10, 1)

    def idx(g):
        a, b = red(g)
        return 10 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb); n = 29 * 20 = 580


def _component_columns(rows, q, stride, width):
    """Extract component columns from the scaffold's exact LP lift."""
    columns = [0] * (width * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:2 * q]):
        for j in range(width):
            bits = (row >> (j * stride * q)) & mask
            while bits:
                low = bits & -bits
                columns[j * q + low.bit_length() - 1] |= 1 << check
                bits ^= low
    return columns


def _component_cost(columns):
    """Screen short component words and shared-check four-cycles."""
    singles = set(columns)
    pairs = {}
    twos = threes = fours = cycles = 0
    for i, left in enumerate(columns):
        for j in range(i):
            right = columns[j]
            value = left ^ right
            twos += value == 0
            threes += value in singles
            previous = pairs.get(value)
            if previous is None:
                pairs[value] = (i, j)
            elif i not in previous and j not in previous:
                fours += 1
            shared = (left & right).bit_count()
            cycles += shared * (shared - 1) // 2
    return twos, threes, fours, cycles


def _support(rng):
    """One zero in each row, in different columns on each side."""
    missing = rng.sample(range(5), 2)
    return [[0 if j == missing[i] else 2 for j in range(5)]
            for i in range(2)]


def _gap_row(rng, reflection_count, mode, row_number):
    """Four distinct unoriented gaps, mixing orders five, ten and four."""
    even = [(2, 0), (4, 0)]
    odd = [(1, 0), (3, 0)]
    reflection = [(a, 1) for a in range(5)]
    rotations = 4 - reflection_count
    preference = (1, -1, 0, 1 if row_number % 2 else -1)[mode]
    even_count = max(rotations - 2, min(2, rotations // 2 + preference))
    gaps = (rng.sample(even, even_count)
            + rng.sample(odd, rotations - even_count)
            + rng.sample(reflection, reflection_count))
    rng.shuffle(gaps)
    return gaps


def _draw_side(rng, G, counts, reflections, mode, side_number):
    grid = []
    for i, supports in enumerate(counts):
        gaps = iter(_gap_row(
            rng, reflections[i], mode, 2 * side_number + i))
        row = []
        for count in supports:
            if not count:
                row.append([])
                continue
            first = G.pick(rng)
            gap = next(gaps)
            if rng.randrange(2):
                gap = G.inv(gap)
            row.append([first, G.mul(first, gap)])
        grid.append(row)
    return grid


def NEW_CTX(rng, G, idx):
    """Screen fresh labels for one of twenty gap-order profiles."""
    profile, mode = idx % 5, (idx // 5) % 4
    reflection_profiles = (
        ((0, 0), (0, 0)),
        ((1, 0), (0, 1)),
        ((1, 1), (1, 1)),
        ((2, 0), (0, 2)),
        ((2, 1), (1, 2)),
    )
    sa, sb = _support(rng), _support(rng)
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    shortlist = []
    for _ in range(28 + idx % 7):
        A = _draw_side(rng, G, sa, reflection_profiles[profile][0],
                       mode, 0)
        B = _draw_side(rng, G, sb, reflection_profiles[profile][1],
                       mode, 1)
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, G.q, B_SHAPE[1], A_SHAPE[1])
        bc = _component_columns(hz, G.q, 1, B_SHAPE[1])
        da, db = _component_cost(ac), _component_cost(bc)
        key = (da[0] + db[0], da[1] + db[1],
               max(da[2], db[2]), da[2] + db[2],
               max(da[3], db[3]), da[3] + db[3], rng.random())
        shortlist.append((key, A, B, hx, hz, n))

    shortlist.sort(key=lambda entry: entry[0])
    fallback = None
    for key, A, B, hx, hz, n in shortlist[:12]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k > 0 and fallback is None:
            fallback = {"A": A, "B": B}
        if 189 <= k <= 217:
            return {"A": A, "B": B}
    return fallback if fallback is not None else {
        "A": shortlist[0][1], "B": shortlist[0][2]}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Parent slot 12 refutes that w=12 and ASC depth 13 suffice for low pL:
# its X basis had 109 failures in 750 shots and scored s=3.499.
# Screened A trinomials with four B extras do not make failures rare:
# the current parent's five measured slots all reached the failed-shot
# cap in both bases, and its fitness was 3.8805.
# Sidon-screened sparse B columns on Z26 did not fix that failure mode:
# all five measured slots again hit both failed-shot caps; fitness was 3.8867.
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
