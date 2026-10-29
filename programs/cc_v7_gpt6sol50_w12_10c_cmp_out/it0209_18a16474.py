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
CONCEPT = ("On dihedral D8 of order 16 with A=(2,5), B=(2,6), "
           "degree-three component columns screened through weight six "
           "will suppress the parent's weight-four sector-1 logicals "
           "while keeping measured rates in cell 0.350.")
GROUP = ("Keep the dihedral construction rule, changing its rotation "
         "order from 13 to 8 solely to fit the new shape: n=34*16=544. "
         "The related cyclic order-17 (2,6)x(2,5) bed reached d30s=8; "
         "the parent's D13 readings were four and five.")
SHAPES = ("A=(2,5), B=(2,6) guarantees k>=12*16=192 and rate "
          "0.353. Its unequal two-check components have 80 and 96 "
          "variables. The parent's lightest Z witnesses used Acol0-1 "
          "across all B columns; light X witnesses also used Bcol0/3.")
FILLING = ("Put one doubleton and one singleton in each component "
           "column. A rows total seven and eight terms and B rows "
           "nine: HX<=11 and HZ<=12. Twenty indexed order-four/eight "
           "gap-count profiles receive fresh labels. Screen both "
           "complete verified components through weight six, then "
           "check full-lift short relations and exact in-cell k.")
FAMILY_NAME = "r035_d8_q16_25x26_dual_component_gap6"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Same dihedral rule: D8 = <x,y | x^8=y^2=1, yxy=x^-1>.
    p = 8
    els = [(a, b) for b in range(2) for a in range(p)]

    def red(g):
        return (g[0] % p, g[1] % 2)

    def idx(g):
        a, b = red(g)
        return p * b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c)) % p, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((a if b else -a) % p, b)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb)


_GAP_PROFILES = None


def _gap_profiles():
    """Twenty pairs of distinct order-four/order-eight gap counts.

    An automorphism of D8 preserves the order of a rotation gap, so
    changing either count is not merely multiplying every label by
    the same rotation-group unit.
    """
    global _GAP_PROFILES
    if _GAP_PROFILES is None:
        _GAP_PROFILES = [
            (a_even, b_even)
            for b_even in range(4)
            for a_even in range(5)
        ]
    return _GAP_PROFILES


def _component_columns(rows, starts, q):
    """Read component syndromes exclusively from verified LP rows."""
    columns = [0] * (len(starts) * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows):
        mark = 1 << check
        for j, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                bit = bits & -bits
                columns[j * q + bit.bit_length() - 1] |= mark
                bits ^= bit
    return columns


def _all_columns(rows, n):
    columns = [0] * n
    for check, row in enumerate(rows):
        mark = 1 << check
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= mark
            row ^= bit
    return columns


def _short_grade(columns, through_five=False):
    """Detect zero-syndrome words of weights at most four or five."""
    singles = set(columns)
    bad2 = int(0 in singles) + len(columns) - len(singles)
    if bad2:
        return (bad2, 0, 0, 0)

    pairs = set()
    bad3 = bad4 = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            bad3 += syndrome in singles
            bad4 += syndrome in pairs
            pairs.add(syndrome)
        if bad3 or bad4 >= 8:
            return (0, bad3, bad4, 0)
    if not through_five:
        return (0, 0, 0, 0)

    fives = 0
    for i in range(len(columns) - 2):
        x = columns[i]
        for j in range(i + 1, len(columns) - 1):
            partial = x ^ columns[j]
            for k in range(j + 1, len(columns)):
                fives += (partial ^ columns[k]) in pairs
                if fives >= 8:
                    return (0, 0, 0, fives)
    return (0, 0, 0, fives)


def _draw_design(rng, G, gap_counts):
    """Two complete degree-three components with fresh dihedral labels."""
    a_even, b_even = gap_counts

    def side(width, first_doubles, even_count):
        heavy = set(rng.sample(range(width), first_doubles))
        gaps = [2] * even_count
        gaps += [rng.choice((1, 3)) for _ in range(width - even_count)]
        rng.shuffle(gaps)
        grid = [[None] * width for _ in range(2)]
        for j, gap in enumerate(gaps):
            doubled = 0 if j in heavy else 1
            base = G.pick(rng)
            grid[doubled][j] = [base, G.mul(base, (gap, 0))]
            grid[1 - doubled][j] = [G.pick(rng)]
        return grid

    # A row totals: 7, 8. B row totals: 9, 9.
    # Every A and B column totals three, hence HX<=11, HZ<=12.
    return side(5, 2, a_even), side(6, 3, b_even)


def _six_grade(columns, ceiling=8):
    """Count exact triple-syndrome collisions after the four-word screen.

    With odd degree-three columns, kernel words have even weight.
    When no weight-four word exists, different triples with equal
    syndrome therefore give genuine weight-six component words.
    """
    seen = set()
    hits = 0
    size = len(columns)
    for i in range(size - 2):
        first = columns[i]
        for j in range(i + 1, size - 1):
            pair = first ^ columns[j]
            for k in range(j + 1, size):
                syndrome = pair ^ columns[k]
                if syndrome in seen:
                    hits += 1
                    if hits >= ceiling:
                        return hits
                else:
                    seen.add(syndrome)
    return hits


def NEW_CTX(rng, G, idx):
    """Screen both complete components and retain an in-cell exact k."""
    gap_counts = _gap_profiles()[idx % 20]
    q = G.q
    shortlist = []
    for trial in range(32):
        A, B = _draw_design(rng, G, gap_counts)
        hx, hz, n = lifted_matrices(G, A, B)

        # HX interleaves A-check rows with the six B-column indices.
        # HZ has its two B-check rows consecutively at A-column zero.
        a_rows = [hx[i * 6 * q + r] for i in range(2)
                  for r in range(q)]
        ac = _component_columns(
            a_rows, [j * 6 * q for j in range(5)], q)
        bc = _component_columns(
            hz[:2 * q], [j * q for j in range(6)], q)
        sa, sb = _short_grade(ac), _short_grade(bc)
        short = (sa[0] + sb[0], sa[1] + sb[1],
                 sa[2] + sb[2])
        shortlist.append((short, trial, A, B, hx, hz, n, ac, bc))

    best = None
    for short, trial, A, B, hx, hz, n, ac, bc in sorted(
            shortlist, key=lambda item: item[:2])[:8]:
        if short == (0, 0, 0):
            six_a = _six_grade(ac)
            six_b = _six_grade(bc)
        else:
            six_a = six_b = 8

        full_x = _short_grade(_all_columns(hx, n))
        full_z = _short_grade(_all_columns(hz, n))
        k = n - gf2_rank(hx) - gf2_rank(hz)
        merit = (not 192 <= k < 204,
                 full_x[0] + full_z[0],
                 full_x[1] + full_z[1],
                 full_x[2] + full_z[2],
                 short, max(six_a, six_b), six_a + six_b,
                 -k, trial)
        if best is None or merit < best[0]:
            best = (merit, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Rolled singleton-heavy supports do not reliably avoid light logicals:
# all five measured parent slots had d30s=5, despite distinct labels.
# REFUTED: Disjoint doubles plus screening through component weight five
# suffice to raise distance above six: all five current measured slots
# retained d30s=6, with weight-six sector-1 X witnesses.
# REFUTED: The previous gap-orbit screen yielded five distance-eight
# readings: its measured distances were 8, 8, 8, 6, 6. Its A extraction
# inspected only the first of two component-check rows.
# REFUTED: Switching that cyclic order-17 bed to cyclic order 26 with
# (1,3)x(2,7) and gap-orbit screening improves its measured distance:
# the current parent's five selected readings were all d30s=6.
# REFUTED: Rotation-gap D13 fillings with sampled component screening
# on (1,3)x(2,7) exceed distance six: the current program's five
# measured distances were 4, 4, 4, 5, and 5.
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
