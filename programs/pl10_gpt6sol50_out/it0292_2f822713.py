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
CONCEPT = ("In cyclic Z13 of order 13 with A=(2,5), B=(3,8), balanced zero/binomial B cells can remove monomial-B short words without raising B row weight; test the worse-basis circuit failure rate.")
GROUP = ("Keep cyclic Z13 and order 13 for a filling-only comparison: n remains 598. This bed has reached about 4.8, while the parent's measured fitness is 4.5493.")
SHAPES = ("Keep (2,5)x(3,8): guaranteed k>=195 of n=598, rate>=0.326 in cell 0.350; component lengths 65 and 104 are unequal, unlike equal-shaped sides. Parent failures and weight-six witnesses occupy sector-1 Acol/Bcol blocks, without a persistent Arow/Brow or check-block hotspot.")
FILLING = ("Cross five A-density profiles with four balanced B motifs. Replace selected (1,1,1) B columns by permutations of (2,0,1), keeping each B row at weight eight and both check weights <=12. Screen verified component relations, rank and Tanner cycles to test whether the parent's 100-failure caps after 2500-8250 shots and worst-logical maxima around 40-49 per 100 failures decline; k excess and d2s are not circuit scores.")
FAMILY_NAME = "r035_z13_25x38_balanced_zero_binomial_B_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 13

def custom_ops(q):
    # Same cyclic presentation and element encoding; only its order changes.
    m = 13
    els = [(a, 0) for a in range(m)]

    def red(g):
        return (g[0] % m, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % m, 0)

    def inv(g):
        return ((-g[0]) % m, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=lambda g: g[0] % m, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb); n = 46 * 13 = 598


def _four_cycles(rows):
    """Count Tanner four-cycles in the verified check rows."""
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            shared = (row & other).bit_count()
            total += shared * (shared - 1) // 2
    return total


def _component_columns(hx, hz, q):
    """Extract both sector-1 component checks from the verified lifts."""
    ca, cb, rb = A_SHAPE[1], B_SHAPE[1], B_SHAPE[0]
    a = []
    for j in range(ca):
        for t in range(q):
            mask = 0
            for i in range(A_SHAPE[0]):
                for r in range(q):
                    row = hx[(i * cb) * q + r]  # beta = 0
                    mask |= ((row >> ((j * cb) * q + t)) & 1) << (i * q + r)
            a.append(mask)
    b = []
    for beta in range(cb):
        for t in range(q):
            mask = 0
            for alpha in range(rb):
                for r in range(q):
                    row = hz[alpha * q + r]     # A-column j = 0
                    mask |= ((row >> (beta * q + t)) & 1) << (alpha * q + r)
            b.append(mask)
    return a, b


def _relations(cols):
    """Count disjoint component relations through weight six.

    Keep the first representative of each pair or triple syndrome.
    Shorter relations take priority over the weight-six multiplicity.
    """
    singles = {value: i for i, value in enumerate(cols)}
    pairs = {}
    bad2 = bad3 = bad4 = 0
    for i, left in enumerate(cols):
        for j in range(i + 1, len(cols)):
            value = left ^ cols[j]
            bad2 += value == 0
            other = singles.get(value)
            bad3 += other is not None and other != i and other != j
            old = pairs.get(value)
            if old is None:
                pairs[value] = (i, j)
            elif i not in old and j not in old:
                bad4 += 1

    triples = {}
    bad5 = bad6 = 0
    for i in range(len(cols) - 2):
        for j in range(i + 1, len(cols) - 1):
            partial = cols[i] ^ cols[j]
            for k in range(j + 1, len(cols)):
                value = partial ^ cols[k]
                old_pair = pairs.get(value)
                if old_pair is not None and all(
                        x not in old_pair for x in (i, j, k)):
                    bad5 += 1
                old_triple = triples.get(value)
                if old_triple is None:
                    triples[value] = (i, j, k)
                elif all(x not in old_triple for x in (i, j, k)):
                    bad6 += 1
    return bad2, bad3, bad4, bad5, bad6


def _counts(rng, idx):
    """Twenty A-density/B-incidence designs, repeated with fresh labels."""
    a_mode, b_mode = (idx // 4) % 5, idx % 4
    first_size, second_size = ((4, 4), (4, 3), (3, 3),
                               (3, 2), (4, 2))[a_mode]
    first = set(rng.sample(range(5), first_size))
    remaining = set(range(5)) - first
    second = remaining | set(rng.sample(
        sorted(first), second_size - len(remaining)))
    sa = [[1 + (j in chosen) for j in range(5)]
          for chosen in (first, second)]

    # An arc (double row, zero row) turns a B column's (1,1,1)
    # into (2,0,1). Oppositely directed arcs balance row sums.
    pair = [(0, 1), (1, 0)]
    if b_mode == 0:
        arcs = pair * 2                 # four altered columns
    elif b_mode == 1:
        arcs = pair * 3                 # six altered columns
    elif b_mode == 2:
        arcs = pair * 4                 # eight; binary B rank <= 2
    else:
        triangle = [(0, 1), (1, 2), (2, 0)]
        arcs = triangle * 2 + pair      # eight; three B row types
    permutation = rng.sample(range(3), 3)
    locations = rng.sample(range(8), len(arcs))
    rng.shuffle(arcs)
    sb = [[1] * 8 for _ in range(3)]
    for j, (double, zero) in zip(locations, arcs):
        sb[permutation[double]][j] = 2
        sb[permutation[zero]][j] = 0

    assert all(sum(row) == 8 for row in sb)
    assert all(sum(sb[i][j] for i in range(3)) == 3
               for j in range(8))
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return sa, sb


def _label_A(rng, sa):
    """Independent origins; unequal gaps in the two doubles of a column."""
    A = [[None] * 5 for _ in range(2)]
    for j in range(5):
        used_gaps = set()
        for i in range(2):
            start = rng.randrange(13)
            if sa[i][j] == 1:
                A[i][j] = [(start, 0)]
            else:
                choices = [d for d in range(1, 7)
                           if d not in used_gaps]
                gap = rng.choice(choices)
                used_gaps.add(gap)
                if rng.randrange(2):
                    gap = -gap
                A[i][j] = [(start, 0), ((start + gap) % 13, 0)]
    return A


def _label_B(rng, sb):
    """Label zero, monomial and binomial cells independently."""
    B = []
    for row in sb:
        filled = []
        for count in row:
            if count == 0:
                filled.append([])
            elif count == 1:
                filled.append([(rng.randrange(13), 0)])
            else:
                start = rng.randrange(13)
                gap = rng.randrange(1, 13)
                filled.append([(start, 0), ((start + gap) % 13, 0)])
        B.append(filled)
    return B


def NEW_CTX(rng, G, idx):
    """Screen fresh labels of one balanced incidence design."""
    sa, sb = _counts(rng, idx)
    best = None
    for _ in range(14 + idx % 4):
        A = _label_A(rng, sa)
        B = _label_B(rng, sb)
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _component_columns(hx, hz, G.q)
        ar, br = _relations(ac), _relations(bc)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not 195 <= k <= 224:
            continue
        # Reject shorter component words before comparing weight-six
        # multiplicities. B comes first in a tie because the parent
        # selected only monomial-B codes with X-type weight-six witnesses.
        quality = (max(ar[0], br[0]), ar[0] + br[0],
                   max(ar[1], br[1]), ar[1] + br[1],
                   max(ar[2], br[2]), br[2] + ar[2],
                   max(ar[3], br[3]), br[3] + ar[3],
                   max(ar[4], br[4]), br[4], ar[4],
                   _four_cycles(hx) + _four_cycles(hz),
                   abs(k - 198), rng.random())
        if best is None or quality < best[0]:
            best = (quality, A, B)
    if best is None:
        raise ValueError("no target-cell filling")
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without redrawing its labels."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# A rolled, weight-12 Z17 (2,5)x(2,6) filling suffices for balanced
# memory performance: parent slots 31 and 50 had worst-Z-basis scores
# 3.691 and 3.817 despite better X-basis scores.
# Unit-separated, four-cycle-screened Z35 (1,3)x(2,5) fillings remove
# the weak X-basis mode: refuted by measured X-basis failure caps after
# 1500-3250 shots and the resulting 4.342 fitness.
# Ranking Z13 fillings by their first short component word, then by
# rank excess and four-cycles, suffices for rare circuit failures:
# refuted by the current 4.5405 fitness; measured worse bases reached
# roughly 100 failures within 3250-4250 shots.
# Adding up to three B doubles would make the proxy screen test those
# supports: refuted by the 4.5493 parent's five measured slots, all
# of which had exclusively monomial B cells.
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
