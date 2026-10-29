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
CONCEPT = ("In cyclic Z46 of order 46 with A=(2,3) and B=(2,3), distinct A gaps and indexed single-repeat B spectra should retain k=70 with fewer circuit failures and isomorphic duplicates.")
GROUP = ("Keep cyclic Z46 and order 46: its n=598 bed has produced five measured zero-failure codes. This round tests only the filling.")
SHAPES = ("Keep equal (2,3)x(2,3) sides, each with 138 component variables; n=598 and rank counting guarantees k>=46, rate>=1/13. The parent's 27 Z failures modestly enriched Acol/Bcol1 data blocks; short X-logical witnesses also touched Arow/Brow blocks. k=70 remains in cell 0.100.")
FILLING = ("Keep distinct-element binomials in every cell for row weight 10. Give A six distinct even gaps and B one repeated even gap, with B spectra indexed by automorphism orbit. Screen complete fillings for short component relations before four-cycles and verify k=70, testing whether the parent's Z-failure outlier and 45 isomorphic duplicates decrease.")
FAMILY_NAME = "r010_z46_23x23_distinctA_single_repeatB_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 46

def custom_ops(q):
    # The same cyclic, trivial-action metacyclic rule as Z19.
    m, r, s = 46, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (int(g[0]) % m, 0)

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        return ((-g[0]) % m, 0)

    def idx(g):
        return g[0] % m

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb); n = 13 * 46 = 598


def _hybrid_gap_catalogs():
    """Distinct and single-repeat spectra modulo Z23 automorphisms."""
    from itertools import combinations

    distinct, repeated = {}, {}

    def orbit(gaps):
        return min(
            tuple(sorted(min((u * d) % 23, (-u * d) % 23)
                         for d in gaps))
            for u in range(1, 12)
        )

    for gaps in combinations(range(1, 12), 6):
        distinct.setdefault(orbit(gaps), gaps)
    for five in combinations(range(1, 12), 5):
        for repeat in five:
            if repeat >= 4:
                gaps = tuple(sorted(five + (repeat,)))
                repeated.setdefault(orbit(gaps), gaps)

    a = [distinct[key] for key in sorted(distinct)]
    b = [repeated[key] for key in sorted(repeated)]
    random.Random(460203).shuffle(a)
    random.Random(460204).shuffle(b)
    assert len(a) >= 20 and len(b) >= 60
    return tuple(a), tuple(b)


_HYBRID_GAPS = _hybrid_gap_catalogs()


def _draw_component(rng, spectrum):
    """Place prescribed gaps, separating any repeat by row and column."""
    gaps = list(spectrum)
    while True:
        rng.shuffle(gaps)
        if (len(set(gaps[:3])) == 3
                and len(set(gaps[3:])) == 3
                and all(gaps[j] != gaps[j + 3] for j in range(3))):
            break

    while True:
        starts = [rng.randrange(46) for _ in range(6)]
        # Couple both parity sheets rather than duplicating one component.
        if len({(starts[j] ^ starts[j + 3]) & 1
                for j in range(3)}) == 2:
            break

    component = []
    for i in range(2):
        row = []
        for j in range(3):
            t = 3 * i + j
            gap = 2 * gaps[t]
            if rng.randrange(2):
                gap = -gap
            row.append([(starts[t], 0),
                        ((starts[t] + gap) % 46, 0)])
        component.append(row)
    return component


def _component_defects(G, grid, side):
    """Screen weight-two through weight-four component relations."""
    q = G.q
    signatures = []
    for j in range(3):
        for h in G.elements:
            syndrome = 0
            for i in range(2):
                for g in grid[i][j]:
                    check = (G.mul(g, h) if side == "A"
                             else G.mul(h, G.inv(g)))
                    syndrome ^= 1 << (i * q + G.idx(check))
            signatures.append(syndrome)

    singles = set(signatures)
    pairs = {}
    twos = threes = fours = cycles = 0
    for i, first in enumerate(signatures):
        for second in signatures[i + 1:]:
            difference = first ^ second
            twos += difference == 0
            threes += difference in singles
            fours += pairs.get(difference, 0)
            pairs[difference] = pairs.get(difference, 0) + 1
            shared = (first & second).bit_count()
            cycles += shared * (shared - 1) // 2
    return twos, threes, fours, cycles


def NEW_CTX(rng, G, idx):
    """Screen one indexed pair of distinct-A and single-repeat-B spectra."""
    pool_a, pool_b = _HYBRID_GAPS
    spectrum_a = pool_a[(idx + 7 * (idx // 20)) % len(pool_a)]
    # Different B orbits distinguish the first sixty slot designs;
    # independently drawn origins are not used as their identities.
    spectrum_b = pool_b[idx % len(pool_b)]
    counts = [[2] * 3 for _ in range(2)]
    assert _grid_weights(counts, counts) == 10
    assert _no_dead_lines(counts, counts)

    shortlist = []
    for _ in range(44 + idx % 7):
        A = _draw_component(rng, spectrum_a)
        B = _draw_component(rng, spectrum_b)
        da = _component_defects(G, A, "A")
        db = _component_defects(G, B, "B")
        cost = (da[0] + db[0], da[1] + db[1],
                max(da[2], db[2]), da[2] + db[2],
                max(da[3], db[3]), da[3] + db[3],
                rng.random())
        shortlist.append((cost, A, B))
    shortlist.sort(key=lambda entry: entry[0])

    for _, A, B in shortlist[:12]:
        if lifted_k(G, A, B) == 70:
            return {"A": A, "B": B}
    # An unusual rank does not void an otherwise legal indexed slot.
    _, A, B = shortlist[0]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the independently screened full cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled (2,4)x(3,5) supports protect both bases comparably:
# all five measured parent slots had X as their worse basis.
# REFUTED: The Z23 (2,3)x(2,5) gap-repeat screen balances the bases:
# its five measured slots had 95-100 Z failures, versus 0-17 X failures.
# REFUTED: Column-covering singleton-heavy Z19 (2,5)x(2,5) supports
# sufficiently suppress circuit failures: its five-code fitness was
# 5.3789; measured slot 17 had 100 Z failures in 13500 shots.
# REFUTED: Mixed odd/even Z46 gap profiles alone attain this bed's
# highest observed score ceiling: all five measured slots had zero
# failures, but their k=62-65 limited fitness to 7.3129.
# REFUTED: Repeating gaps on both Z46 components reliably preserves
# zero failures: parent slot 24 had 27 Z failures in 200000 shots,
# and only 15 of its 60 slots were distinct.
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
