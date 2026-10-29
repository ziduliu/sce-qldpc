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
CONCEPT = ("In dihedral Z17:Z2 of order 34 with A=(2,3) and B=(2,4), screened binomial components should reduce worse-basis circuit failures while keeping row weight at most 12.")
GROUP = ("Keep the dihedral construction rule but use order 34: (2,3)x(2,4) would exceed the size window at order 58, and the measured order-34 bed at these shapes has scored 7.2.")
SHAPES = ("At order 34, (2,3)x(2,4) gives n=544 and guaranteed k>=68, rate>=0.125, in the target 0.150 cell. Its unequal components have 102 and 136 variables. The parent's witnesses and failures spread across sector-1 Acol/Bcol blocks, not a persistent check-block hotspot; extra component rows test protection across those blocks.")
FILLING = ("Use two distinct same-coset elements per cell: HX rows weigh 6+4=10 and HZ rows weigh 8+4=12. Cross twenty indexed coset-pattern and rotation-gap designs, screening verified component relations and check cycles before checking k. Test for fewer of the parent's roughly 100 worse-basis failures within 1750-4000 shots.")
FAMILY_NAME = "r015_d17_23x24_indexed_binomial_components_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 34

def custom_ops(q):
    # Same metacyclic dihedral rule: Z_17 x| Z_2,
    # with y x y^-1 = x^16 and 16^2 = 1 mod 17.
    m, r, s = 17, 2, 16
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 16 * 34 = 544


def _component_columns(rows, q, side):
    """Extract both component check rows from the verified LP lift."""
    if side == "A":
        nchecks, ncols, stride = A_SHAPE[0], A_SHAPE[1], B_SHAPE[1]
    else:
        nchecks, ncols, stride = B_SHAPE[0], B_SHAPE[1], 1
    columns = [0] * (ncols * q)
    mask = (1 << q) - 1
    for i in range(nchecks):
        for r in range(q):
            row = rows[i * stride * q + r]
            check_bit = 1 << (i * q + r)
            for j in range(ncols):
                bits = (row >> (j * stride * q)) & mask
                while bits:
                    low = bits & -bits
                    columns[j * q + low.bit_length() - 1] |= check_bit
                    bits ^= low
    return columns


def _short_signature(columns):
    """Count weight-2 and weight-4 component-kernel witnesses."""
    two = len(columns) - len(set(columns)) + sum(c == 0 for c in columns)
    pairs = {}
    four = 0
    short_cycles = 0
    for j, right in enumerate(columns):
        for i in range(j):
            left = columns[i]
            shared = left & right
            if shared & (shared - 1):
                short_cycles += 1
            syndrome = left ^ right
            first = pairs.get(syndrome)
            if first is None:
                pairs[syndrome] = (i, j)
            elif i not in first and j not in first:
                four += 1
    return two, four, short_cycles


_A_COSETS = ((0, 0, 1), (0, 1, 1), (1, 0, 1),
             (1, 1, 0), (0, 1, 0))
_B_COSETS = ((0, 0, 0, 1), (0, 0, 1, 1),
             (0, 1, 0, 1), (0, 1, 1, 1))
_A_GAPS = ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 7, 8),
           (1, 3, 4, 5, 6, 8), (2, 3, 4, 5, 7, 8),
           (1, 2, 5, 6, 7, 8))


def _binomial(rng, gap, coset):
    """A nonzero rotation gap makes two distinct elements in one coset."""
    start = rng.randrange(17)
    step = gap if rng.randrange(2) else -gap
    cell = [(start, coset), ((start + step) % 17, coset)]
    rng.shuffle(cell)
    return cell


def _root_sixes(columns, q):
    """Count component six-cycle configurations in both check blocks."""
    checks = [0] * (2 * q)
    for variable, column in enumerate(columns):
        while column:
            bit = column & -column
            checks[bit.bit_length() - 1] |= 1 << variable
            column ^= bit

    total = 0
    for root_index in (0, q // 2, q, q + q // 2):
        root = checks[root_index]
        nearby = [(r, root & checks[r]) for r in range(2 * q)
                  if r != root_index and root & checks[r]]
        for t, (u, ru) in enumerate(nearby):
            for v, rv in nearby[:t]:
                outside = checks[u] & checks[v] & ~root
                if outside:
                    total += ((ru & ~rv).bit_count()
                              * (rv & ~ru).bit_count()
                              * outside.bit_count())
    return total


def _root_check_fours(rows, q):
    """Sample both cosets in every verified full-check block."""
    worst = total = 0
    for block in range(len(rows) // q):
        block_cost = 0
        for offset in (0, q // 2):
            root_index = block * q + offset
            root = rows[root_index]
            for j, row in enumerate(rows):
                if j != root_index:
                    shared = (root & row).bit_count()
                    block_cost += shared * (shared - 1) // 2
        worst = max(worst, block_cost)
        total += block_cost
    return worst, total


def NEW_CTX(rng, G, idx):
    """Screen fresh labels of one indexed two-row binomial design."""
    design = idx % 20
    a_pattern = _A_COSETS[design % 5]
    b_pattern = _B_COSETS[design // 5]
    a_gaps = list(_A_GAPS[design % 5])
    b_gaps = list(range(1, 9))
    counts_a = [[2] * 3 for _ in range(2)]
    counts_b = [[2] * 4 for _ in range(2)]
    assert _grid_weights(counts_a, counts_b) == 12
    assert _no_dead_lines(counts_a, counts_b)

    shortlist = []
    for _ in range(24):
        rng.shuffle(a_gaps)
        rng.shuffle(b_gaps)
        A = [[_binomial(rng, a_gaps[3 * i + j],
                        0 if i == 0 else a_pattern[j])
              for j in range(3)] for i in range(2)]
        B = [[_binomial(rng, b_gaps[4 * i + j],
                        0 if i == 0 else b_pattern[j])
              for j in range(4)] for i in range(2)]
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, G.q, "A")
        bc = _component_columns(hz, G.q, "B")
        da, db = _short_signature(ac), _short_signature(bc)
        cheap = (da[0] + db[0], max(da[1], db[1]),
                 da[1] + db[1], max(da[2], db[2]),
                 da[2] + db[2], rng.random())
        shortlist.append((cheap, A, B, hx, hz, n, ac, bc))
    shortlist.sort(key=lambda item: item[0])

    best_key = best = None
    for cheap, A, B, hx, hz, n, ac, bc in shortlist[:6]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        six_a = _root_sixes(ac, G.q)
        six_b = _root_sixes(bc, G.q)
        xf = _root_check_fours(hx, G.q)
        zf = _root_check_fours(hz, G.q)
        key = (not 68 <= k <= 94, cheap[:3],
               max(six_a, six_b), six_a + six_b,
               max(xf[0], zf[0]), xf[1] + zf[1],
               cheap[3:], abs(k - 82))
        if best_key is None or key < best_key:
            best_key, best = key, {"A": A, "B": B}
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell; NEW_CTX owns all slot randomness."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled singleton-heavy supports suffice to suppress short sector-1 logicals at this bed: all five measured parent slots had weight-4 witnesses and worse-basis scores of only 3.17-3.34.
# REFUTED: The parent's sparse 3/2-term profiles screen as well as its full triples: many sparse slots had d2s=2, while all five measured slots used full triples and still reached the worse-basis failure cap within 1500-3250 shots.
# REFUTED: Indexed full triples and verified-check-cycle screening on the
# order-58 (1,3)x(1,3) bed make failures rare: its measured slots still
# reached about 100 worse-basis failures within 1750-4000 shots.
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
