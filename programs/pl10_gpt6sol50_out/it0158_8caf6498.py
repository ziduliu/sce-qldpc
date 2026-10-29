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
CONCEPT = ("In cyclic Z19 of order 19, shapes (2,5)x(2,5), complementary sparse binomials should reduce worst-basis circuit failures by removing short sector-1 Z logicals.")
GROUP = ("Jump from Z35 to Z19. The measured Z19 (2,5)x(2,5) bed reached 5.4 in cell 0.300; that is a relevant comparison, not a prediction for these new fillings.")
SHAPES = ("Equal (2,5)x(2,5) sides give n=551, guaranteed k>=171 and rate 0.3103 in cell 0.300; each component has 95 variables. The parent's X failures affected every logical qubit, while weight-five Z witnesses occupied sector-1 Acol/Bcol blocks. A second A row tests that weakness.")
FILLING = ("Complementary double-cell positions give each row seven or eight terms, every component column three or four terms, and both HX and HZ weight <=12. Indexed gap palettes and independent labels are screened on verified lifted incidences, short relations, rank and Tanner cycles. Test for fewer X failed shots without transferring the limitation to Z.")
FAMILY_NAME = "r030_z19_25x25_complementary_gap_relation_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    # Additive cyclic group Z19, encoded as two-coordinate tuples.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (g[0] % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=lambda g: g[0] % q, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb); n = 29 * 19 = 551


def _four_cycles(rows):
    """Count four-cycles in the evaluator's exact lifted check rows."""
    total = 0
    for i, row in enumerate(rows):
        for earlier in rows[:i]:
            shared = (row & earlier).bit_count()
            total += shared * (shared - 1) // 2
    return total


def _support(rng, extras):
    """Complementary doubled positions; column totals are three or four."""
    first_count, second_count = extras
    order = rng.sample(range(5), 5)
    first = set(order[:first_count])
    second = set(order[first_count:])
    if second_count == 3:
        second.add(rng.choice(tuple(first)))
    assert len(second) == second_count
    return [[1 + (j in selected) for j in range(5)]
            for selected in (first, second)]


def _label_side(rng, G, supports, palette):
    """Independent starts and signs for this design's gap spectrum."""
    gaps = list(palette)
    rng.shuffle(gaps)
    grid = []
    for row in supports:
        cells = []
        for count in row:
            start = rng.randrange(G.q)
            if count == 1:
                cells.append([(start, 0)])
            else:
                gap = gaps.pop()
                step = gap if rng.randrange(2) else -gap
                cells.append([(start, 0), ((start + step) % G.q, 0)])
        grid.append(cells)
    assert not gaps
    return grid


def _component_columns(hx, hz, q):
    """Extract both component matrices from the verified LP check rows."""
    a_rows = [hx[i * 5 * q + r] for i in range(2) for r in range(q)]
    b_rows = [hz[alpha * q + r]
              for alpha in range(2) for r in range(q)]
    a_cols = [
        sum(((row >> ((j * 5) * q + t)) & 1) << r
            for r, row in enumerate(a_rows))
        for j in range(5) for t in range(q)
    ]
    b_cols = [
        sum(((row >> (beta * q + t)) & 1) << r
            for r, row in enumerate(b_rows))
        for beta in range(5) for t in range(q)
    ]
    return a_cols, b_cols


def _relation_profile(columns):
    """Count weight-two/three/four component relations and four-cycles."""
    singles = set(columns)
    pair_counts = {}
    first_pairs = {}
    bad2 = bad3 = bad4 = cycles = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            value = left ^ right
            bad2 += value == 0
            bad3 += value in singles
            previous = pair_counts.get(value, 0)
            bad4 += previous
            pair_counts[value] = previous + 1
            if not previous:
                first_pairs[value] = (i, j)
            shared = (left & right).bit_count()
            cycles += shared * (shared - 1) // 2
    return (bad2, bad3, bad4, cycles), first_pairs


def _sample_long_relations(columns, pairs, rng, draws=2200):
    """Sample only disjoint triple/pair relations, on verified columns."""
    found = 0
    for _ in range(draws):
        a, b, c = rng.sample(range(len(columns)), 3)
        previous = pairs.get(columns[a] ^ columns[b] ^ columns[c])
        if previous is not None and all(x not in previous
                                        for x in (a, b, c)):
            found += 1
    return found


def NEW_CTX(rng, G, idx):
    """Screen an indexed, independently labelled two-component design."""
    from itertools import combinations

    profiles = (
        ((3, 3), (3, 3)),
        ((3, 2), (3, 3)),
        ((3, 3), (3, 2)),
        ((3, 2), (3, 2)),
        ((3, 3), (2, 3)),
    )
    a_extra, b_extra = profiles[idx % len(profiles)]
    palettes_a = list(combinations(range(1, 10), sum(a_extra)))
    palettes_b = list(combinations(range(1, 10), sum(b_extra)))
    design = idx % 20
    gaps_a = palettes_a[(11 * design + 7 * (idx // 20))
                         % len(palettes_a)]
    gaps_b = palettes_b[(13 * design + 17 + 5 * (idx // 20))
                         % len(palettes_b)]
    shortlist = []

    for _ in range(16):
        sa, sb = _support(rng, a_extra), _support(rng, b_extra)
        assert _grid_weights(sa, sb) <= 12
        A = _label_side(rng, G, sa, gaps_a)
        B = _label_side(rng, G, sb, gaps_b)
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _component_columns(hx, hz, G.q)
        aq, ap = _relation_profile(ac)
        bq, bp = _relation_profile(bc)
        # Neither component may be sacrificed for a better other side.
        short = (max(aq[0], bq[0]), aq[0] + bq[0],
                 max(aq[2], bq[2]), aq[2] + bq[2],
                 aq[1] + bq[1],
                 max(aq[3], bq[3]), aq[3] + bq[3])
        shortlist.append((short, rng.random(), A, B, hx, hz, n,
                          ac, bc, ap, bp))

    shortlist.sort(key=lambda item: item[:2])
    best = None
    for short, _, A, B, hx, hz, n, ac, bc, ap, bp in shortlist[:5]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        longer = (_sample_long_relations(ac, ap, rng),
                  _sample_long_relations(bc, bp, rng))
        key = (k > 178, short[:5], max(longer), sum(longer),
               short[5:],
               _four_cycles(hx) + _four_cycles(hz),
               max(0, k - 171))
        if best is None or key < best[0]:
            best = (key, {"A": A, "B": B})

    if best is None:
        raise ValueError("no positive-k Z19 filling")
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the screened complete cell with distinct group elements."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Singleton-heavy rolled labels suffice on this bed: refuted by the seed's
# 4.0553 fitness, weight-5 X-type witnesses, and 109/750 Z-basis failures
# in measured slot 50.
# A mixed-coset (1,3)x(2,5) filling by itself removes the short
# sector-1 failure mode: refuted by the measured D17 bed's 3.0007
# fitness and X-basis failures reaching the cap after 250 shots.
# Complementary supports and four-cycle selection alone make X-basis
# failures rare: refuted by current-parent slots 22 and 31, with
# 106/2500 and 108/4250 X-basis failed shots respectively.
# Screening the sparse Z17 profiles through component weight five
# suffices to match the 0.350-cell frontier: refuted by their measured
# 4.3509 fitness; every selected slot's worse Z basis reached 100
# failures within 2000-4000 shots.
# Uniform degree-three Z17 columns balance the two measured bases:
# refuted by the five measured slots here; Z reached 100 failures in
# 2500-7250 shots while X needed 13250-20500.
# A triple-heavy A row plus denser B rows repairs the weak X basis:
# refuted by this parent's 3.8763 fitness; its measured X bases reached
# about 100 failures within 500-1250 shots, despite Z bases reaching
# the 200000-shot cap.
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
