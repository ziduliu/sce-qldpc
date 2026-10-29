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
CONCEPT = ("On dicyclic C9 semidirect C4 of order 36 with A=(2,3), "
           "B=(2,4), screened support profiles will produce five "
           "rate-0.150 codes with fitness above 39.2.")
GROUP = ("Jump from cyclic Z17 to C9 semidirect C4 with inversion "
         "action and a central involution. Its order-36 (2,3)x(2,4) "
         "bed is untried; the neighbouring cyclic Z37 bed scored 39.2.")
SHAPES = ("A=(2,3), B=(2,4) give n=16*36=576 and guaranteed "
          "k=2*36=72. Exact-rank selection targets k=73..100 in "
          "cell 0.150. The unequal two-row components have 108 and "
          "144 variables. The parent's lightest weight-eight logicals "
          "occupied sector-one Acol/Bcol blocks across both sides.")
FILLING = ("Cross four A and five B support profiles, then draw fresh "
           "noncommuting labels with indexed relative-element orders. "
           "Both sides have seven terms per row and at most five per "
           "column, bounding HX and HZ row weights by twelve. Screen "
           "verified components through weight six and select by exact "
           "rank, seeking fewer light witnesses and useful k above 72.")
FAMILY_NAME = "r015_dic9_23x24_cross_profile_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # <x,y | x^9=1, y^4=1, y*x*y^-1=x^-1>.
    # In particular y^2 is central, but y has order four.
    els = [(a, b) for b in range(4) for a in range(9)]

    def red(g):
        return (g[0] % 9, g[1] % 4)

    def idx(g):
        a, b = red(g)
        return 9 * b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b & 1 else c)) % 9, (b + d) % 4)

    def inv(g):
        a, b = red(g)
        return (((a if b & 1 else -a) % 9), (-b) % 4)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


_A_PROFILES = (
    ((3, 2, 2), (2, 3, 2)),
    ((4, 2, 1), (1, 2, 4)),
    ((3, 3, 1), (2, 1, 4)),
    ((4, 1, 2), (1, 4, 2)),
)
_B_PROFILES = (
    ((2, 2, 2, 1), (2, 2, 2, 1)),
    ((3, 2, 1, 1), (1, 2, 3, 1)),
    ((1, 2, 2, 2), (3, 2, 2, 0)),
    ((3, 3, 0, 1), (1, 1, 2, 3)),
    ((3, 2, 1, 1), (2, 1, 3, 1)),
)


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
    """Screen the actual, possibly mixed-degree, component through six."""
    singles = set(columns)
    if 0 in singles:
        return (1, 0, 0)
    if len(singles) != len(columns):
        return (2, 0, 0)

    pairs = set()
    four = False
    overlap = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            if syndrome in singles:
                return (3, 0, 0)
            if syndrome in pairs:
                four = True
            pairs.add(syndrome)
            shared = (x & y).bit_count()
            overlap += shared * (shared - 1) // 2
    if four:
        return (4, 0, -overlap)

    triples = set()
    sixes = 0
    for i in range(len(columns) - 2):
        x = columns[i]
        for j in range(i + 1, len(columns) - 1):
            partial = x ^ columns[j]
            for k in range(j + 1, len(columns)):
                syndrome = partial ^ columns[k]
                if syndrome in pairs:
                    return (5, 0, -overlap)
                if syndrome in triples:
                    sixes += 1
                else:
                    triples.add(syndrome)
    return (6 if sixes else 7, -sixes, -overlap)


def _relative_buckets(G):
    """Nonidentity relative elements of several attainable orders."""
    buckets = {order: [] for order in (4, 6, 9, 18)}
    for element in G.elements:
        power = G.e
        for order in range(1, 19):
            power = G.mul(power, element)
            if power == G.e:
                if order in buckets:
                    buckets[order].append(element)
                break
    return buckets


def _label_profile(rng, G, profile, orders):
    """Fresh labels on a permuted profile, without duplicate cell terms."""
    permutation = rng.sample(range(len(profile[0])), len(profile[0]))
    counts = [[row[j] for j in permutation] for row in profile]
    if rng.randrange(2):
        counts.reverse()
    grid = []
    for i, row in enumerate(counts):
        cells = []
        for j, count in enumerate(row):
            if not count:
                cells.append([])
                continue
            first = G.pick(rng)
            cell = [first]
            if count >= 2:
                order = orders[(i + 2 * j) % len(orders)]
                cell.append(G.mul(first, rng.choice(orders[order])))
            while len(cell) < count:
                candidate = G.pick(rng)
                if candidate not in cell:
                    cell.append(candidate)
            cells.append(cell)
        grid.append(cells)
    return grid


def NEW_CTX(rng, G, idx):
    """Select fresh labels for one of twenty crossed support motifs."""
    motif = idx % 20
    a_profile = _A_PROFILES[motif // 5]
    b_profile = _B_PROFILES[motif % 5]
    buckets = _relative_buckets(G)
    order_cycle = (18, 9, 4, 6)
    rotation = (idx // 20 + motif) % 4
    available = {order: buckets[order] for order in order_cycle}
    a_orders = {order: available[order] for order in
                order_cycle[rotation:] + order_cycle[:rotation]}
    b_orders = {order: available[order] for order in
                order_cycle[(rotation + 1):] +
                order_cycle[:(rotation + 1)]}
    # The tuple sets the indexed relative-order schedule; the dictionary
    # supplies the corresponding actual group elements.
    a_schedule = order_cycle[rotation:] + order_cycle[:rotation]
    b_rotation = (rotation + 1) % 4
    b_schedule = order_cycle[b_rotation:] + order_cycle[:b_rotation]

    shortlist = []
    for _ in range(9):
        A = _label_profile(rng, G, a_profile,
                           (a_schedule, a_orders))
        B = _label_profile(rng, G, b_profile,
                           (b_schedule, b_orders))
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, G.q, 3, 4, 4)
        bc = _component_columns(hz, G.q, 4, 1, 1)
        sa, sb = _component_merit(ac), _component_merit(bc)
        grade = (min(sa[0], sb[0]), sa[0] + sb[0],
                 sa[1] + sb[1], sa[2] + sb[2], rng.random())
        shortlist.append((grade, A, B, hx, hz, n))

    best = None
    for grade, A, B, hx, hz, n in sorted(
            shortlist, key=lambda entry: entry[0], reverse=True)[:4]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        quality = (73 <= k <= 100, grade, -abs(k - 83))
        if best is None or quality > best[0]:
            best = (quality, {"A": A, "B": B})
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
# REFUTED: Exchanging component lengths and screening all-degree-three
# Z17 components breaks the weight-eight ceiling: all five current
# measured slots still had d30s=8.
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
