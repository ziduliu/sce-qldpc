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
CONCEPT = ("On (C13 semidirect C3) x C3 of order 117 with (1,2)x(1,2), screening both order-39 quotient components through weight seven should yield five distinct codes improving on the parent's measured distances 22-23.")
GROUP = ("Replace C13 semidirect C9 by (C13 semidirect C3) x C3: the latter has no element of order nine and is genuinely different. A related order-117 bed reached fitness 94.8; the parent's bed reached 107.6. Keeping q=117 gives n=585.")
SHAPES = ("Keep equal (1,2)x(1,2) sides: their equal-length two-column components give guaranteed k>=117 and rate 0.200. The parent's lightest X and Z logicals occupied sector-one Acol/Bcol blocks, so screen both components.")
FILLING = ("Four distinct terms per cell give HX and HZ row weight 12. Indexed action-class profiles receive independently sampled coordinates in the new central C3. Screen verified order-39 quotient components through weight seven, then check full-lift rank; this targets the parent's distance-22 outlier without seeking excess k.")
FAMILY_NAME = "r020_c13sdC3_xC3_12x12_quotient_seven"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 117

def custom_ops(q):
    # (C13 semidirect_3 C3) x C3. The final coordinate is central.
    els = [(a, b, c) for c in range(3)
           for b in range(3) for a in range(13)]

    def red(g):
        return (g[0] % 13, g[1] % 3, g[2] % 3)

    def idx(g):
        a, b, c = red(g)
        return a + 13 * b + 39 * c

    def mul(g, h):
        a, b, c = red(g)
        u, v, t = red(h)
        return ((a + pow(3, b, 13) * u) % 13,
                (b + v) % 3, (c + t) % 3)

    def inv(g):
        a, b, c = red(g)
        bi = (-b) % 3
        return ((-pow(3, bi, 13) * a) % 13, bi, (-c) % 3)

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 2)               # (rb, cb)


def _component_columns(G, A, B):
    """Extract both component-check matrices from the verified LP lift."""
    hx, hz, _ = lifted_matrices(G, A, B)
    q = G.q
    mask = (1 << q) - 1
    a_columns = [0] * (2 * q)
    b_columns = [0] * (2 * q)
    for r in range(q):
        # HX at B-column 0 contains A-column blocks at bases 0, 2q.
        a_row = (hx[r] & mask) | (((hx[r] >> (2 * q)) & mask) << q)
        # HZ at A-column 0 contains both adjacent B-column blocks.
        b_row = hz[r] & ((1 << (2 * q)) - 1)
        for columns, row in ((a_columns, a_row), (b_columns, b_row)):
            while row:
                bit = row & -row
                columns[bit.bit_length() - 1] |= 1 << r
                row ^= bit
    return a_columns, b_columns


def _quotient_group():
    """The verified-lift input for the quotient by the central C3."""
    els = [(a, b) for b in range(3) for a in range(13)]

    def mul(g, h):
        return ((g[0] + pow(3, g[1], 13) * h[0]) % 13,
                (g[1] + h[1]) % 3)

    def inv(g):
        bi = (-g[1]) % 3
        return ((-pow(3, bi, 13) * g[0]) % 13, bi)

    return _G(dict(elements=els, ident=(0, 0),
                   mul=mul, inv=inv), 39)


def _short_component(columns, q, test_seven=False):
    """Detect component kernel words through six, optionally through seven."""
    size = len(columns)
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != size:
        return 2

    pairs = set()
    for i in range(size):
        for j in range(i + 1, size):
            syndrome = columns[i] ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                return 4
            pairs.add(syndrome)

    triples = {}
    for i in range(size - 2):
        for j in range(i + 1, size - 1):
            partial = columns[i] ^ columns[j]
            for k in range(j + 1, size):
                syndrome = partial ^ columns[k]
                if syndrome in pairs:
                    return 5
                if syndrome in triples:
                    return 6
                triples[syndrome] = (1 << i) | (1 << j) | (1 << k)

    if not test_seven:
        return 7

    # Regular-lift translation puts some coordinate of any word at
    # position zero in one of the two protograph blocks.
    for pivot in (0, q):
        bit = 1 << pivot
        for syndrome, used in triples.items():
            if used & bit:
                continue
            other = triples.get(syndrome ^ columns[pivot])
            if other is not None and not (other & (used | bit)):
                return 7
    return 8


def _overlap_penalty(columns):
    penalty = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            common = (x & y).bit_count()
            if common >= 2:
                penalty += common * (common - 1) // 2
    return penalty


def _four_term_cell(rng, doubled_class, mode):
    """Distinct a labels; independent central heights and action classes."""
    heights = rng.sample(range(3), 3)
    other = rng.choice([h for h in range(3)
                        if h != heights[doubled_class]])
    actions = [0, 1, 2, doubled_class]
    centres = heights + [other]
    a_values = rng.sample(range(13), 4)
    cell = list(zip(a_values, actions, centres))
    if mode & 1:
        rng.shuffle(cell)
    return cell


def NEW_CTX(rng, G, idx):
    """Screen indexed designs in the quotient, then rank full lifts."""
    quotient = _quotient_group()
    signature = tuple((idx // (3 ** position)) % 3
                      for position in range(4))
    mode = (idx // 9) % 3
    shortlist = []

    for _ in range(24):
        A = [[_four_term_cell(rng, signature[j], mode)
              for j in range(2)]]
        B = [[_four_term_cell(rng, signature[2 + j], mode + 1)
              for j in range(2)]]
        projected_a = [[[(a, b) for a, b, c in cell]
                        for cell in A[0]]]
        projected_b = [[[(a, b) for a, b, c in cell]
                        for cell in B[0]]]
        ac, bc = _component_columns(quotient, projected_a, projected_b)
        da = _short_component(ac, quotient.q)
        db = _short_component(bc, quotient.q)
        shortlist.append(((min(da, db), da + db, rng.random()),
                          A, B, ac, bc, da, db))

    qualified = []
    for preliminary, A, B, ac, bc, da, db in sorted(
            shortlist, key=lambda item: item[0], reverse=True)[:10]:
        qa = (_short_component(ac, quotient.q, True)
              if da == 7 else da)
        qb = (_short_component(bc, quotient.q, True)
              if db == 7 else db)
        overlap = _overlap_penalty(ac) + _overlap_penalty(bc)
        grade = (min(qa, qb), qa + qb, -overlap, preliminary[2])
        qualified.append((grade, A, B))

    best = best_grade = None
    for quotient_grade, A, B in sorted(
            qualified, key=lambda item: item[0], reverse=True)[:5]:
        ac, bc = _component_columns(G, A, B)
        k = lifted_k(G, A, B)
        full_overlap = _overlap_penalty(ac) + _overlap_penalty(bc)
        grade = (117 <= k <= 131, quotient_grade[0],
                 quotient_grade[1], -full_overlap,
                 -abs(k - 121), quotient_grade[2], quotient_grade[3])
        if best_grade is None or grade > best_grade:
            best_grade, best = grade, {"A": A, "B": B}
    return best


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Uniform rolled dihedral-58 (1,3)x(1,3) fillings do not reliably exceed distance 4: all five measured parent slots had d30s=4.
# REFUTED: Maximizing noncommutator weight among eight action-spanning labelings does not reliably suppress short witnesses: the order-117 parent had numerous d2s=9 slots and a measured d30s=18 slot.
# REFUTED: Screening only component relations of weight three or four and check-neighbour overlaps guarantees five distance-23 readings: the current parent's measured slot 6 had d30s=22.
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
