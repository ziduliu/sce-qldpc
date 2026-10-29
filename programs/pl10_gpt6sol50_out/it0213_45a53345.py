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
CONCEPT = ("In dicyclic Dic8 of order 32 with A=(1,3) and B=(2,5), screened trinomial A cells and column-budgeted B cells should reduce worst-basis circuit failures in rate cell 0.350.")
GROUP = ("Keep the dicyclic construction rule while increasing its order from 28 to 32, the largest dicyclic lift fitting the new factor 17. The parent's Dic7 bed scored 6.9902 and supplies a measured baseline for this shape test.")
SHAPES = ("A=(1,3), B=(2,5) give n=544, guaranteed k=192 and rate 0.353; their component lengths are 96 and 160 rather than equal. Parent slot 1's 19 X failures enriched Arow1/Brow1 and Acol3/Bcol2, while slot 34's Z failures enriched Acol1/Bcol2. Both block layouts change.")
FILLING = ("Cross five support budgets with four gap spectra. Trinomial A cells and sparse B rows keep both row sums at most 12; exact-lift component screens choose independent labels. Test for fewer worse-basis failures and decoder fall-throughs, not a larger distance proxy or excess k.")
FAMILY_NAME = "r035_dic8_q32_13x25_trinomial_budget_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    # Dic8 = <x,y | x^16=1, y^2=x^8, yxy^-1=x^-1>.
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        return (g[0] % 16, g[1] % 2)

    def mul(g, h):
        return ((g[0] + (-h[0] if g[1] else h[0])
                 + 8 * g[1] * h[1]) % 16, (g[1] + h[1]) % 2)

    def inv(g):
        return ((-g[0] if g[1] == 0 else g[0] + 8) % 16, g[1])

    def idx(g):
        a, b = red(g)
        return 16 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb); n = 17 * 32 = 544


def _component_columns(rows, q, side):
    """Extract the new A or B component from the exact verified lift."""
    nrows, ncols, stride = ((1, 3, 5) if side == "A"
                            else (2, 5, 1))
    cols = [0] * (ncols * q)
    mask = (1 << q) - 1
    for i in range(nrows):
        for r in range(q):
            # Fix B-column zero in HX, or A-column zero in HZ.
            row = rows[(i * stride * q + r) if side == "A"
                       else (i * q + r)]
            for j in range(ncols):
                block = (row >> (j * stride * q)) & mask
                while block:
                    bit = block & -block
                    col = j * q + bit.bit_length() - 1
                    cols[col] |= 1 << (i * q + r)
                    block ^= bit
    return cols


def _component_relations(cols):
    """Count zero-syndrome pairs, triples, and disjoint pairs of pairs."""
    singles = {}
    for value in cols:
        singles[value] = singles.get(value, 0) + 1
    pairs = {}
    twos = triples = fours = 0
    for u, left in enumerate(cols):
        for v in range(u + 1, len(cols)):
            value = left ^ cols[v]
            twos += value == 0
            triples += singles.get(value, 0)
            previous = pairs.get(value)
            if previous is not None:
                fours += sum(u not in pair and v not in pair
                             for pair in previous)
                previous.append((u, v))
            else:
                pairs[value] = [(u, v)]
    return twos, triples // 3, fours


def _component_cycles(cols, q):
    """Rooted component Tanner four-, six-, and eight-cycle counts."""
    var_checks = []
    check_vars = [[] for _ in range(2 * q)]
    for v, bits in enumerate(cols):
        neighbors = []
        while bits:
            bit = bits & -bits
            c = bit.bit_length() - 1
            neighbors.append(c)
            check_vars[c].append(v)
            bits ^= bit
        var_checks.append(neighbors)

    counts = [0, 0, 0]
    for root in range(0, len(check_vars), q):
        spokes = []
        root_edges = {}
        for first in check_vars[root]:
            for middle in var_checks[first]:
                if middle != root:
                    spokes.append((middle, first))
                    root_edges.setdefault(middle, []).append(first)
        counts[0] += sum(len(v) * (len(v) - 1) // 2
                         for v in root_edges.values())

        paths = {}
        for middle, first in spokes:
            for second in check_vars[middle]:
                if second == first:
                    continue
                for end in var_checks[second]:
                    if end == root or end == middle:
                        continue
                    for last in root_edges.get(end, ()):
                        if last != first and last != second:
                            counts[1] += 1
                    previous = paths.setdefault(end, [])
                    for old_middle, old_first, old_second in previous:
                        if old_middle != middle and len({
                                first, second, old_first, old_second}) == 4:
                            counts[2] += 1
                    previous.append((middle, first, second))
    return tuple(counts)


def NEW_CTX(rng, G, idx):
    """Screen an indexed support budget and voltage spectrum."""
    mode, spectrum = divmod(idx % 20, 4)
    order = rng.sample(range(5), 5)
    sa = [[3, 3, 3] if mode in (1, 2, 3) else [3, 3, 2]]
    rng.shuffle(sa[0])

    if mode == 0:
        # B rows each have nine terms and distinct singleton columns.
        sb = [[1 if j == order[0] else 2 for j in range(5)],
              [1 if j == order[1] else 2 for j in range(5)]]
    elif mode == 1:
        # Disjoint sets of two and three doubled columns.
        sb = [[2 if j in order[:2] else 1 for j in range(5)],
              [2 if j in order[2:] else 1 for j in range(5)]]
    elif mode == 2:
        sb = [[2 if j == order[0] else 1 for j in range(5)],
              [2 if j in order[1:4] else 1 for j in range(5)]]
    elif mode == 3:
        sb = [[1] * 5,
              [1 if j == order[0] else 2 for j in range(5)]]
    else:
        sb = [[2 if j in order[:3] else 1 for j in range(5)],
              [1 if j == order[0] else 2 for j in range(5)]]

    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)

    domains = ((1, 3, 5, 7), (9, 11, 13, 15),
               (2, 6, 10, 14), (1, 5, 9, 13))

    def draw(supports):
        grid = []
        for row in supports:
            cells = []
            for terms in row:
                first = G.pick(rng)
                cell = [first]
                if terms >= 2:
                    gap = rng.choice(domains[spectrum])
                    if rng.randrange(2):
                        gap = -gap
                    # The last spectrum also tests quotient-crossing
                    # binomials, rather than merely shifting old labels.
                    crossing = int(spectrum == 3 and rng.randrange(3) == 0)
                    cell.append(G.mul(first, (gap, crossing)))
                if terms == 3:
                    # Opposite quotient parity makes the third element
                    # distinct from either same-parity element.
                    offset = rng.randrange(16)
                    cell.append(G.mul(first, (offset, 1 - crossing)))
                cells.append(cell)
            grid.append(cells)
        return grid

    trials = []
    for _ in range(22 + idx % 6):
        A, B = draw(sa), draw(sb)
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, G.q, "A")
        bc = _component_columns(hz, G.q, "B")
        ar, br = _component_relations(ac), _component_relations(bc)
        ay, by = _component_cycles(ac, G.q), _component_cycles(bc, G.q)
        key = (ar[0] + br[0], ar[1] + br[1],
               max(ar[2], br[2]), ar[2] + br[2],
               max(ay[0], by[0]), ay[0] + by[0],
               max(ay[1], by[1]), ay[1] + by[1],
               max(ay[2], by[2]), ay[2] + by[2])
        trials.append((key, rng.random(), A, B, hx, hz, n))

    trials.sort(key=lambda item: (item[0], item[1]))
    for _, _, A, B, hx, hz, n in trials[:8]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # For n=544, k=192..203 stays below the cell's upper edge.
        if 192 <= k <= 203:
            return {"A": A, "B": B}

    _, _, A, B, _, _, _ = trials[0]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Predominantly single-term rolled D7 cells suffice for low circuit failure:
# selected parent codes had 101-113 worst-basis failures in 250 shots and
# reported weight-four sector-2 logical operators.
# Unrestricted 2/3-term labels selected by repeated-pair screening alone
# improve this bed: measured five-code fitness fell to 3.3584 versus the
# preceding program's 3.4438, with weight-three witnesses in several slots.
# Unit-gap screening of component relations and Tanner four-cycles alone
# guarantees balanced circuit failures: refuted by current slot 8, with
# 100 X failures in 26250 shots versus only 3 Z failures in 200000.
# Complementary odd-gap supports reliably balance every measured slot:
# refuted by this parent's slot 1 with 19 X failures in 200000 shots,
# while slot 54 had one X failure in the same number of shots.
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
