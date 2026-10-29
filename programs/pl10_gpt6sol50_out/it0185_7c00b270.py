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
CONCEPT = ("In cyclic Z17 of order 17, A=(2,5) and B=(2,6), distinct paired gap spectra screened for short component relations should reduce worse-basis circuit failures.")
GROUP = ("Keep the cyclic construction rule. Order 17 gives n=578 for the new shapes; the measured Z17 (2,5)x(2,6) bed has reached about 4.6 versus this parent's 4.5405.")
SHAPES = ("(2,5)x(2,6) guarantees k>=204 of n=578, rate>=0.3529 in the 0.350 cell. Its unequal-length components have 85 and 102 variables. The parent's weight-six witnesses occupied sector-1 Acol/Bcol blocks; both bases reached the failure cap within 3250-4250 shots.")
FILLING = ("Give every component column one binomial and one monomial cell: HX<=11 and HZ<=12. Assign slots distinct paired gap-spectrum orbits, screen independent labels on verified lifts for component weight-four and weight-six relations, then check rank and Tanner four-cycles. Target fewer worse-basis failed shots, not proxy alone.")
FAMILY_NAME = "r035_z17_25x26_paired_gap_orbits_dual_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
    # Same cyclic presentation and element encoding; only its order changes.
    m = 17
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
B_SHAPE = (2, 6)               # (rb, cb); n = 34 * 17 = 578


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


def _gap_orbits():
    """Paired undirected spectra, identified under common Z17 automorphisms."""
    from itertools import combinations
    representatives = set()
    for ag in combinations(range(1, 9), 5):
        for bg in combinations(range(1, 9), 6):
            orbit = []
            for unit in range(1, 17):
                orbit.append((
                    tuple(sorted(min(unit * d % 17, (-unit * d) % 17)
                                 for d in ag)),
                    tuple(sorted(min(unit * d % 17, (-unit * d) % 17)
                                 for d in bg))))
            representatives.add(min(orbit))
    designs = sorted(representatives)
    random.Random(5318008).shuffle(designs)
    return tuple(designs)


_GAP_ORBITS = _gap_orbits()


def _degree_three_side(rng, width, gaps):
    """One distinct-element binomial and one monomial per column."""
    columns = list(range(width))
    rng.shuffle(columns)
    doubled = set(columns[:width // 2])
    assigned = list(gaps)
    rng.shuffle(assigned)
    grid = [[None] * width for _ in range(2))
    for j, gap in enumerate(assigned):
        row = 0 if j in doubled else 1
        start = rng.randrange(17)
        step = gap if rng.randrange(2) else -gap
        grid[row][j] = [(start, 0), ((start + step) % 17, 0)]
        grid[1 - row][j] = [(rng.randrange(17), 0)]
    return grid


def _component_defects(cols):
    """Count weight-two/four relations, then equal-syndrome triples.

    Every column has odd weight three. Once weight-two/four relations
    are absent, a repeated triple syndrome witnesses a weight-six word.
    """
    pair_values = set()
    twos = fours = 0
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            value = left ^ right
            if value == 0:
                twos += 1
            elif value in pair_values:
                fours += 1
            else:
                pair_values.add(value)
    if twos or fours:
        return twos, fours, 1000000

    triples = {}
    sixes = 0
    for i in range(len(cols) - 2):
        for j in range(i + 1, len(cols) - 1):
            partial = cols[i] ^ cols[j]
            for k in range(j + 1, len(cols)):
                value = partial ^ cols[k]
                previous = triples.get(value, 0)
                sixes += previous
                triples[value] = previous + 1
                if sixes > 256:
                    return twos, fours, sixes
    return twos, fours, sixes


def NEW_CTX(rng, G, idx):
    """Screen fresh labels of one indexed, non-automorphic spectrum pair."""
    ag, bg = _GAP_ORBITS[idx % len(_GAP_ORBITS)]
    finalists = []
    for _ in range(12 + 2 * (idx % 5)):
        A = _degree_three_side(rng, 5, ag)
        B = _degree_three_side(rng, 6, bg)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= 12
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _component_columns(hx, hz, G.q)
        da, db = _component_defects(ac), _component_defects(bc)
        short = (da[0] + db[0], da[1] + db[1],
                 max(da[2], db[2]), da[2] + db[2])
        finalists.append((short, rng.random(), A, B, hx, hz, n))

    finalists.sort(key=lambda item: item[:2])
    best = None
    for short, _, A, B, hx, hz, n in finalists[:4]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        quality = (not 204 <= k <= 216, short,
                   _four_cycles(hx) + _four_cycles(hz), k - 204)
        if best is None or quality < best[0]:
            best = (quality, A, B)
    if best is None:
        raise ValueError("no positive-k degree-three filling")
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
# REFUTED: balanced Z13 (2,5)x(3,8) short-word screening solves both bases; fitness was 4.5405 and measured worse bases reached about 100 failures within 3250-3750 shots.
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
