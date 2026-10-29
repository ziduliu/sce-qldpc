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
CONCEPT = ("In dicyclic Dic7 of order 28, balanced (2,4)x(2,4) components with order-14 binomials and screened short relations may reduce worst-basis circuit failures in rate cell 0.200.")
GROUP = ("Jump from the parent's order-44 C11 semidirect C4 to Dic7, whose unique involution and order-four elements give a genuinely different group law. Recorded order-28 (2,4)x(2,4) beds scored 6.3-6.4; the new law and filling require measurement.")
SHAPES = ("Equal (2,4)x(2,4) sides give n=20*28=560, guaranteed k>=112 and rate 0.200. Each component has 112 variables and 56 checks. Parent Z failures enriched Acol1/Bcol2 and Arow/Brow blocks; its measured X failure enriched Arow1/Brow1, motivating balanced screening of both sides.")
FILLING = ("Use seven terms per component row: distinct singleton and order-14 binomial cells, with indexed three-term coset couplers. Screen independent complete fillings for short relations on both sides, then use verified lifts to prefer k=98-125. This tests failed-shot rates rather than treating the distance proxy as fitness.")
FAMILY_NAME = "r020_dic7_24x24_balanced_order14_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Dic7 = <x,y | x^14=1, y^2=x^7, yxy^-1=x^-1>.
    els = [(a, b) for b in range(2) for a in range(14)]

    def red(g):
        return (g[0] % 14, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c) + 7 * b * d) % 14,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a + 7, 1) if b else ((-a) % 14, 0))

    def idx(g):
        a, b = red(g)
        return 14 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _component_columns(rows, q, side):
    """Extract both component check rows from the verified HX or HZ."""
    columns = [0] * (4 * q)
    mask = (1 << q) - 1
    for i in range(2):
        for r in range(q):
            check = i * q + r
            # HX has four B-column check blocks per A row; HZ has
            # four A-column check blocks per B row.
            row = rows[i * 4 * q + r] if side == "A" else rows[check]
            for j in range(4):
                base = j * (4 if side == "A" else 1) * q
                bits = (row >> base) & mask
                while bits:
                    low = bits & -bits
                    columns[j * q + low.bit_length() - 1] |= 1 << check
                    bits -= low
    return columns


def _short_relations(columns):
    """Count component zero/pair, triple and disjoint-pair relations."""
    singles = {}
    zero = 0
    for i, value in enumerate(columns):
        zero += value == 0
        singles.setdefault(value, i)
    pairs = {}
    twos = threes = fours = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            value = left ^ columns[j]
            twos += value == 0
            third = singles.get(value)
            threes += third is not None and third != i and third != j
            previous = pairs.get(value)
            if previous is None:
                pairs[value] = (i, j)
            elif i not in previous and j not in previous:
                fours += 1
    return zero + twos, threes, fours


def _component_four_cycles(columns):
    """Count pairs of component variables sharing two checks."""
    cycles = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            common = (left & right).bit_count()
            cycles += common * (common - 1) // 2
    return cycles


def _counts(rng, mode):
    """One singleton per row; couplers preserve seven terms per row."""
    a = [[2] * 4 for _ in range(2)]
    b = [[2] * 4 for _ in range(2)]

    def initialise(grid):
        first = rng.randrange(4)
        second = (first + rng.randrange(1, 4)) % 4
        grid[0][first] = grid[1][second] = 1

    def add_coupler(grid, row):
        own = grid[row].index(1)
        other = grid[1 - row].index(1)
        available = [j for j in range(4) if j not in (own, other)]
        rng.shuffle(available)
        grid[row][available[0]] = 3
        grid[row][available[1]] = 1

    initialise(a)
    initialise(b)
    if mode in (1, 4):
        add_coupler(a, 0)
    if mode == 3:
        add_coupler(a, 1)
    if mode == 2:
        add_coupler(b, 0)
    if mode in (3, 4):
        add_coupler(b, 1)
    assert _grid_weights(a, b) <= 12
    return a, b


def _draw_cell(rng, count):
    """Use order-14 binomial gaps; a triple also crosses the cosets."""
    coset = rng.randrange(2)
    start = rng.randrange(14)
    if count == 1:
        return [(start, coset)]
    gap = rng.choice((1, 3, 5, 9, 11, 13))
    cell = [(start, coset), ((start + gap) % 14, coset)]
    if count == 3:
        cell.append((rng.randrange(14), 1 - coset))
    rng.shuffle(cell)
    return cell


def _draw_side(rng, counts):
    """Frustrate the two-row coset pattern rather than separating halves."""
    while True:
        grid = [[_draw_cell(rng, count) for count in row]
                for row in counts]
        differences = {
            grid[0][j][0][1] ^ grid[1][j][0][1]
            for j in range(4)
            if len(grid[0][j]) == len(grid[1][j]) == 2
        }
        if len(differences) == 2:
            return grid


def NEW_CTX(rng, G, idx):
    """Screen independent designs, then verify target-cell k on finalists."""
    finalists = []
    for _ in range(18 + idx % 7):
        sa, sb = _counts(rng, idx % 5)
        A = _draw_side(rng, sa)
        B = _draw_side(rng, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, G.q, "A")
        bc = _component_columns(hz, G.q, "B")
        a2, a3, a4 = _short_relations(ac)
        b2, b3, b4 = _short_relations(bc)
        quality = (a2 + b2, a3 + b3,
                   max(a4, b4), a4 + b4,
                   _component_four_cycles(ac) +
                   _component_four_cycles(bc))
        finalists.append((quality, A, B, hx, hz, n))
    finalists.sort(key=lambda item: item[0])

    best = None
    best_key = None
    for quality, A, B, hx, hz, n in finalists[:8]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        key = (not (98 <= k <= 125), quality, abs(k - 120))
        if best_key is None or key < best_key:
            best_key, best = key, {"A": A, "B": B}
    if best is None:
        raise ValueError("no positive-k, weight-legal filling")
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without redrawing labels."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled singleton-heavy supports suffice to suppress short sector-1 logicals at this bed: all five measured parent slots had weight-4 witnesses and worse-basis scores of only 3.17-3.34.
# REFUTED: Coset-spanning triples alone remove short sector-1 witnesses: the current program's five measured slots still had weight-6 or weight-8 witnesses and worse-basis scores of 4.19-4.54.
# REFUTED: Short-relation screening on the order-44 (2,3)x(2,3) bed already beats the 0.100 frontier: its measured fitness was 7.065 versus the frontier's 7.322.
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
