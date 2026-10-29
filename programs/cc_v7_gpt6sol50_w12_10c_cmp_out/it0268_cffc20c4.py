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
CONCEPT = ("On cyclic Z17 of order 17 with A=(2,6), B=(2,5), "
           "two degree-two A columns and a degree-four/degree-two B "
           "profile screened through weight six should give five "
           "k=208 codes with d30s at least eight.")
GROUP = ("Keep the measured cyclic Z17 bed: the current program produced "
         "60 distinct n=578 codes, with k=204 and d30s=8 in all five "
         "measured slots. This isolates a filling-only rank-and-distance test.")
SHAPES = ("Keep unequal A=(2,6), B=(2,5): n=578, guaranteed k=204 "
          "and rate 0.353. Their two-check components have 102 and 85 "
          "variables, unlike equal shapes with equal-length components. "
          "The parent's lightest X and Z witnesses occupied sector-1 "
          "Acol/Bcol blocks across the grid.")
FILLING = ("Use two (1,1) and four (2,1)/(1,2) A columns, and two "
           "(1,1) and three (2,2) B columns. Every component row weighs "
           "eight, so HX and HZ weigh at most twelve. Even B-column "
           "parities are intended to add four qubits above the k=204 "
           "bound; screen both verified components through weight six "
           "to avoid trading the parent's weight-eight witnesses for "
           "weight-six ones.")
FAMILY_NAME = "r035_z17_26x25_even_B_rank_screen_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Cyclic Z17, encoded as one-coordinate tuples.
    els = [(a,) for a in range(17)]

    def red(g):
        return (g[0] % 17,)

    def idx(g):
        return g[0] % 17

    def mul(g, h):
        return ((g[0] + h[0]) % 17,)

    def inv(g):
        return ((-g[0]) % 17,)

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _columns(rows, starts, q):
    """Read component signatures from the scaffold's verified lift."""
    result = [0] * (len(starts) * q)
    mask = (1 << q) - 1
    for r, row in enumerate(rows):
        mark = 1 << r
        for j, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                bit = bits & -bits
                result[j * q + bit.bit_length() - 1] |= mark
                bits ^= bit
    return result


def _component_grade(cols):
    """Count component zero-syndrome relations through weight six."""
    singles = set(cols)
    bad2 = int(0 in singles) + len(cols) - len(singles)
    pairs = {}
    bad3 = bad4 = 0
    for i, ci in enumerate(cols):
        for cj in cols[i + 1:]:
            syndrome = ci ^ cj
            bad3 += syndrome in singles
            bad4 += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    if bad2 or bad3 or bad4:
        return (bad2, bad3, bad4, 0, 0)

    seen3 = set()
    bad5 = bad6 = 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            partial = cols[i] ^ cols[j]
            for t in range(j + 1, len(cols)):
                syndrome = partial ^ cols[t]
                bad5 += syndrome in pairs
                bad6 += syndrome in seen3
                seen3.add(syndrome)
            if bad5 + bad6 > 3000:
                return (0, 0, 0, bad5, bad6)
    return (0, 0, 0, bad5, bad6)


def _full_short(rows, n):
    """Count short kernels in an exact full check matrix."""
    cols = [0] * n
    for ri, row in enumerate(rows):
        mark = 1 << ri
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= mark
            row ^= bit
    singles = set(cols)
    bad2 = int(0 in singles) + n - len(singles)
    if bad2:
        return (bad2, 0, 0)
    pairs = {}
    bad3 = bad4 = 0
    for i, ci in enumerate(cols):
        for cj in cols[i + 1:]:
            syndrome = ci ^ cj
            bad3 += syndrome in singles
            bad4 += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    return (0, bad3, bad4)


def NEW_CTX(rng, G, idx):
    """Independently screen two deliberately changed component profiles."""
    q = G.q

    def pair(gap):
        start = rng.randrange(17)
        return [(start,), ((start + gap) % 17,)]

    def a_side():
        # Two degree-two columns, four degree-three columns. Each A row
        # has two doubles and four singletons, hence eight terms.
        order = rng.sample(range(6), 6)
        light = set(order[:2])
        doubled_in_first = set(order[2:4])
        grid = [[None] * 6 for _ in range(2)]
        low_gaps = rng.sample(range(1, 9), 2)
        for j in range(6):
            if j in light:
                grid[0][j] = [(rng.randrange(17),)]
                grid[1][j] = [(rng.randrange(17),)]
            else:
                row = 0 if j in doubled_in_first else 1
                grid[row][j] = pair(rng.randrange(1, 17))
                grid[1 - row][j] = [(rng.randrange(17),)]
        # The relative shifts of the two low-degree columns are
        # indexed motifs, not a common translation of a stored code.
        for j, gap in zip(sorted(light), low_gaps):
            base = grid[0][j][0][0]
            grid[1][j] = [((base + gap + idx % 8) % 17,)]
        return grid

    def b_side():
        # Two (1,1) and three (2,2) columns give eight terms per row.
        # At x=1 their two parity-check rows coincide; unlike the
        # parent's all-degree-three B, this can add four logical qubits.
        order = rng.sample(range(5), 5)
        light = set(order[:2])
        grid = [[None] * 5 for _ in range(2)]
        for j in range(5):
            if j in light:
                grid[0][j] = [(rng.randrange(17),)]
                grid[1][j] = [(rng.randrange(17),)]
            else:
                for row in range(2):
                    grid[row][j] = pair(rng.randrange(1, 17))
        for j, gap in zip(sorted(light),
                          rng.sample(range(1, 9), 2)):
            base = grid[0][j][0][0]
            grid[1][j] = [((base + gap + 3 * idx % 8) % 17,)]
        return grid

    best_a = best_b = None
    grade_a = grade_b = None
    for trial in range(36):
        A = best_a if grade_a == (0, 0, 0, 0, 0) else a_side()
        B = best_b if grade_b == (0, 0, 0, 0, 0) else b_side()
        hx, hz, _ = lifted_matrices(G, A, B)

        if grade_a != (0, 0, 0, 0, 0):
            a_rows = [hx[i * 5 * q + r] for i in range(2)
                      for r in range(q)]
            candidate = _component_grade(
                _columns(a_rows, [j * 5 * q for j in range(6)], q))
            if grade_a is None or candidate < grade_a:
                grade_a, best_a = candidate, A

        if grade_b != (0, 0, 0, 0, 0):
            candidate = _component_grade(
                _columns(hz[:2 * q], [j * q for j in range(5)], q))
            if grade_b is None or candidate < grade_b:
                grade_b, best_b = candidate, B

        if grade_a == grade_b == (0, 0, 0, 0, 0):
            break

    return {"A": best_a, "B": best_b}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell; each sampled exponent is distinct."""
    return ctx[side][i][j]

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled support variation on the parent's (1,3)x(1,3) bed
# suffices to remove weight-4 logicals: all five measured slots had d30s=4.
# REFUTED: The previous (1,4)x(1,4) penalty screen suffices to remove
# short logicals: its five measured slots all retained d30s=3 witnesses.
# REFUTED: Triangle-free three-binomial gaps eliminate weight-four
# logicals: all five measured parent slots had d30s=4, with 2489-2659
# verified weight-four operators in each reported spectrum.
# REFUTED: Splitting selected triples across cosets improves that
# distance-four ceiling: the current five measured slots all had d30s=4.
# REFUTED: Mixed-order Q16 binomials on (2,5)x(2,6) yield five
# distance-eight codes: all five measured slots had d30s=6.
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
