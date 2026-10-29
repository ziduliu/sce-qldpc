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
           "a parity-rank-one B with degree-three columns can raise "
           "k above 204 while screening out weight-six witnesses.")
GROUP = ("Keep cyclic Z17 for a filling-only comparison. Its n=578 "
         "bed has produced distance-eight readings at k=204, including "
         "two in the current measured slots.")
SHAPES = ("Keep A=(2,6), B=(2,5): n=34*17=578 and guaranteed "
          "k=204, rate 0.353 in cell 0.350. The unequal two-row "
          "components have 102 and 85 variables. The current lightest "
          "sector-one Z witnesses spread across Acol/Bcol blocks, "
          "particularly Bcol1-3.")
FILLING = ("Keep one singleton and one binomial in each A column. "
           "In B, put one row-zero trinomial above a zero cell and "
           "four row-zero singletons above binomials: all columns "
           "have degree three but B has parity rank one. A rows "
           "weigh 9/9 and B rows 7/8, so LP rows weigh at most 12. "
           "Screen both verified components through weight six, "
           "seeking k=208 without the current weight-six or "
           "weight-seven sector-one witnesses.")
FAMILY_NAME = "r035_m17r1s1_26x25_rankone_B_trinomial_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # metacyclic Z_17 x| Z_1: y x y^-1 = x^1 (s^r = 1 mod m);
    # rolled, not designed - see make_rate_seeds.py
    m, r, s = 17, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


_GAP_PROFILES = None


def _gap_profiles():
    """Twenty distinct simultaneous multiplier orbits of gap spectra."""
    global _GAP_PROFILES
    if _GAP_PROFILES is None:
        from itertools import combinations

        def transform(gaps, unit):
            return tuple(sorted(min((unit * d) % 17,
                                    (-unit * d) % 17) for d in gaps))

        orbits = {}
        for a in combinations(range(1, 9), 5):
            for b in combinations(range(1, 9), 6):
                representative = min(
                    (transform(a, unit), transform(b, unit))
                    for unit in range(1, 17))
                orbits.setdefault(representative, (a, b))
        distinct = [orbits[key] for key in sorted(orbits)]
        _GAP_PROFILES = [
            distinct[t * len(distinct) // 20] for t in range(20)]
    return _GAP_PROFILES


def _component_columns(rows, q, blocks, stride, check_blocks):
    """Read both component-check rows from the verified LP matrices."""
    columns = [0] * (blocks * q)
    mask = (1 << q) - 1
    for i in range(2):
        for r in range(q):
            # HX interleaves its A-check blocks with B-column indices;
            # HZ places its B-check blocks consecutively at A-column 0.
            row = rows[(i * check_blocks) * q + r]
            check = 1 << (i * q + r)
            for j in range(blocks):
                bits = (row >> (j * stride * q)) & mask
                while bits:
                    bit = bits & -bits
                    columns[j * q + bit.bit_length() - 1] |= check
                    bits ^= bit
    return columns


def _component_merit(columns):
    """Lexicographically count words of weights 2, 4, 6 and short cycles.

    All constructed columns have odd weight three, so odd-weight
    component words cannot be in the kernel. Pair and triple syndrome
    collisions detect the remaining weights through six.
    """
    if 0 in columns or len(set(columns)) != len(columns):
        return (1, 0, 0, 0)

    pairs = set()
    overlap = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in pairs:
                return (0, 1, 0, 0)
            pairs.add(syndrome)
            shared = (x & y).bit_count()
            overlap += shared * (shared - 1) // 2

    triples = set()
    sixes = 0
    for i in range(len(columns) - 2):
        x = columns[i]
        for j in range(i + 1, len(columns) - 1):
            xy = x ^ columns[j]
            for k in range(j + 1, len(columns)):
                syndrome = xy ^ columns[k]
                if syndrome in triples:
                    sixes += 1
                    if sixes >= 102:
                        return (0, 0, sixes, overlap)
                else:
                    triples.add(syndrome)
    return (0, 0, sixes, overlap)


def _label_side(rng, gaps, columns, first_row_doubles):
    """Use each gap once; random bases are fresh, not orbit relabelings."""
    gaps = list(gaps)
    rng.shuffle(gaps)
    positions = list(range(columns))
    rng.shuffle(positions)
    heavy = set(positions[:first_row_doubles])
    grid = [[None] * columns for _ in range(2)]
    for j, gap in enumerate(gaps):
        doubled = 0 if j in heavy else 1
        start = rng.randrange(17)
        grid[doubled][j] = [(start, 0), ((start + gap) % 17, 0)]
        grid[1 - doubled][j] = [(rng.randrange(17), 0)]
    return grid


def _mixed_grids(rng, short_gaps, long_gaps, profile):
    """Balance A at 8/8 and B at 8/8 or 9/7, including a B four-cell."""
    a_order = list(range(6))
    rng.shuffle(a_order)
    light = set(a_order[:2])
    a_row_zero_doubles = set(a_order[2:4])
    gaps = rng.sample(list(long_gaps), 4)
    A = [[None] * 6 for _ in range(2)]
    for j in range(6):
        if j in light:
            A[0][j] = [(rng.randrange(17), 0)]
            A[1][j] = [(rng.randrange(17), 0)]
        else:
            row = 0 if j in a_row_zero_doubles else 1
            start = rng.randrange(17)
            gap = gaps.pop()
            A[row][j] = [(start, 0), ((start + gap) % 17, 0)]
            A[1 - row][j] = [(rng.randrange(17), 0)]

    heavy = rng.randrange(5)
    others = [j for j in range(5) if j != heavy]
    rng.shuffle(others)
    gaps = rng.sample(list(short_gaps), 4)
    B = [[None] * 5 for _ in range(2)]
    if profile == 0:
        # Four-cell (2,2); two of the other doubles go in each row.
        double_row_zero = set(others[:2])
        for row in range(2):
            start = rng.randrange(17)
            gap = rng.choice(range(1, 9))
            B[row][heavy] = [(start, 0), ((start + gap) % 17, 0)]
    else:
        # Four-cell (3,1). Profile 1 gives rows 8/8; profile 2
        # gives rows 9/7. The heavy row may be either protograph row.
        triple_row = rng.randrange(2)
        double_in_triple_row = 1 if profile == 1 else 2
        selected = set(others[:double_in_triple_row])
        double_row_zero = (selected if triple_row == 0
                           else set(others) - selected)
        B[triple_row][heavy] = [
            (t, 0) for t in rng.sample(range(17), 3)]
        B[1 - triple_row][heavy] = [(rng.randrange(17), 0)]

    for j in others:
        row = 0 if j in double_row_zero else 1
        start = rng.randrange(17)
        gap = gaps.pop()
        B[row][j] = [(start, 0), ((start + gap) % 17, 0)]
        B[1 - row][j] = [(rng.randrange(17), 0)]
    return A, B


def _mixed_component_merit(columns):
    """Grade exact component relations through weight six.

    Mixed column degrees permit odd-weight relations, unlike the
    parent's all-degree-three components.
    """
    singles = set(columns)
    if 0 in singles:
        return (1, 0)
    if len(singles) != len(columns):
        return (2, 0)

    pairs = set()
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in singles:
                return (3, 0)
            if syndrome in pairs:
                return (4, 0)
            pairs.add(syndrome)

    triples = set()
    sixes = 0
    for i in range(len(columns) - 2):
        x = columns[i]
        for j in range(i + 1, len(columns) - 1):
            partial = x ^ columns[j]
            for k in range(j + 1, len(columns)):
                syndrome = partial ^ columns[k]
                if syndrome in pairs:
                    return (5, 0)
                if syndrome in triples:
                    sixes += 1
                else:
                    triples.add(syndrome)
    return (6 if sixes else 7, min(sixes, 102))


def _rank_one_b(rng, gap_pool, idx):
    """A degree-three B whose column parities are all (1,0)."""
    q = 17
    special = rng.randrange(5)
    gaps = rng.sample(list(gap_pool), 4)
    rng.shuffle(gaps)
    B = [[None] * 5 for _ in range(2)]

    # Avoid an arithmetic-progression triple; vary its actual shape,
    # not just a translation or multiplier of a stored cell.
    while True:
        triple = rng.sample(range(q), 3)
        distances = {
            min((triple[i] - triple[j]) % q,
                (triple[j] - triple[i]) % q)
            for i in range(3) for j in range(i)
        }
        if len(distances) == 3:
            break
    B[0][special] = [(x, 0) for x in triple]
    B[1][special] = []

    ordinary = [j for j in range(5) if j != special]
    rng.shuffle(ordinary)
    for j, gap in zip(ordinary, gaps):
        base = rng.randrange(q)
        B[0][j] = [(rng.randrange(q), 0)]
        B[1][j] = [(base, 0), ((base + gap) % q, 0)]
    return B


def NEW_CTX(rng, G, idx):
    """Independently screen two degree-three verified components."""
    short_gaps, long_gaps = _gap_profiles()[idx % 20]
    best_a = best_b = None
    merit_a = merit_b = None

    for trial in range(48):
        A = (best_a if merit_a is not None
             and merit_a[:3] == (0, 0, 0)
             else _label_side(rng, long_gaps, 6, 3))
        B = (best_b if merit_b is not None
             and merit_b[:3] == (0, 0, 0)
             else _rank_one_b(rng, short_gaps, idx))
        hx, hz, _ = lifted_matrices(G, A, B)

        if merit_a is None or merit_a[:3] != (0, 0, 0):
            candidate = _component_merit(
                _component_columns(hx, G.q, 6, 5, 5))
            if merit_a is None or candidate < merit_a:
                merit_a, best_a = candidate, A

        if merit_b is None or merit_b[:3] != (0, 0, 0):
            candidate = _component_merit(
                _component_columns(hz, G.q, 5, 1, 1))
            if merit_b is None or candidate < merit_b:
                merit_b, best_b = candidate, B

        if merit_a[:3] == (0, 0, 0) \
                and merit_b[:3] == (0, 0, 0):
            break

    return {"A": best_a, "B": best_b}


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
# REFUTED: Removing component words through weight six alone raises
# selected distance beyond eight: all five current measured slots
# retained d30s=8, with numerous sector-one weight-eight witnesses.
# REFUTED: Balanced degree-two A and degree-four B columns improve the
# five-code score: the mixed-degree program scored 17.5764; its five
# measured distances were 6, 6, 8, 7, 8, all with k=204.
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
