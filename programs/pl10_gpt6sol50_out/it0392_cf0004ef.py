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
CONCEPT = ("In (C7 x V4) semidirect C3 of order 84, A=(1,2) and B=(1,3), an order-14 A-binomial, an A-trinomial and three B-trinomials should reduce worst-basis circuit failures.")
GROUP = ("Keep the centre-trivial order-84 group fixed. The current bed produced 60 distinct codes, permitting a direct filling-only test against its measured failures.")
SHAPES = ("Keep (1,2)x(1,3): n=588, guaranteed k=168 and rate 2/7 in cell 0.300. Its one-row component lengths are unequal, 168 and 252. The parent's limiting Z failures were broadly spread across sector-1 Acol/Bcol data blocks, without a persistent check-block hotspot.")
FILLING = ("Cross five A voltage motifs with four B quotient motifs at counts A=(2,3), B=(3,3,3). Screen distinct labels on verified component relations, check cycles, full-group generation and target-cell rank. Test for fewer than the parent's five Z-basis 100-failure caps in 16000-119000 shots while preserving its rare X failures; d2s is only a selection proxy.")
FAMILY_NAME = "r030_c7v4_semidirect_c3_12x13_order14_A_gap_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 84

def custom_ops(q):
    # (C7 x V4) semidirect C3. Its generator doubles C7 and cycles
    # the three nonzero elements of V4.
    cycle = (0, 2, 3, 1)
    elements = [(a, b, c) for c in range(3)
                for b in range(4) for a in range(7)]

    def red(g):
        return (int(g[0]) % 7, int(g[1]) % 4, int(g[2]) % 3)

    def act(c, b):
        for _ in range(c % 3):
            b = cycle[b]
        return b

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + pow(2, c, 7) * d) % 7,
                b ^ act(c, e), (c + f) % 3)

    def inv(g):
        a, b, c = red(g)
        ci = (-c) % 3
        return ((-pow(2, ci, 7) * a) % 7, act(ci, b), ci)

    def idx(g):
        a, b, c = red(g)
        return 28 * c + 7 * b + a

    return dict(elements=elements, ident=(0, 0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb); n = 7 * 84 = 588


def _component_columns(hx, hz, q):
    """Read both component incidences from the verified LP lifts."""
    mask = (1 << q) - 1
    a = [0] * (2 * q)
    b = [0] * (3 * q)
    for r in range(q):
        for j in range(2):
            bits = (hx[r] >> (j * 3 * q)) & mask
            while bits:
                bit = bits & -bits
                a[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit
        for j in range(3):
            bits = (hz[r] >> (j * q)) & mask
            while bits:
                bit = bits & -bits
                b[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit
    return a, b


def _component_cost(columns):
    """Count short component relations and Tanner four-cycles."""
    singles = set(columns)
    pairs = {}
    two = three = four = cycles = 0
    for i, left in enumerate(columns):
        for right in columns[:i]:
            syndrome = left ^ right
            two += syndrome == 0
            three += syndrome in singles
            four += pairs.get(syndrome, 0)
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
            shared = (left & right).bit_count()
            cycles += shared * (shared - 1) // 2
    return two, three, four, cycles


def _root_fours(rows, q):
    """Count four-cycles through one check in each lifted check block."""
    total = 0
    for root in range(0, len(rows), q):
        first = rows[root]
        for i, row in enumerate(rows):
            if i != root:
                common = (first & row).bit_count()
                total += common * (common - 1) // 2
    return total


def _root_sixes(columns, q):
    """Count simple six-cycles through component check zero."""
    checks = [0] * q
    for variable, column in enumerate(columns):
        while column:
            bit = column & -column
            checks[bit.bit_length() - 1] |= 1 << variable
            column ^= bit
    root = checks[0]
    nearby = [(r, root & checks[r]) for r in range(1, q)
              if root & checks[r]]
    total = 0
    for i, (u, ru) in enumerate(nearby):
        for v, rv in nearby[:i]:
            total += ((ru & ~rv).bit_count()
                      * (rv & ~ru).bit_count()
                      * (checks[u] & checks[v] & ~root).bit_count())
    return total


def _generated_order(G, grid):
    """Order generated by voltage differences within cells."""
    generators = []
    for row in grid:
        for cell in row:
            first = cell[0]
            generators.extend(G.mul(g, G.inv(first)) for g in cell[1:])
    seen = {G.e}
    pending = [G.e]
    while pending:
        h = pending.pop()
        for g in generators:
            product = G.mul(h, g)
            if product not in seen:
                seen.add(product)
                pending.append(product)
    return len(seen)


def _triple(rng, cs, bs):
    """Three distinct voltages with prescribed quotient and V4 motifs."""
    residues = rng.sample(range(7), 3)
    return [(residues[t], bs[t], cs[t]) for t in range(3)]


def _draw_design(rng, idx):
    """Five A motifs crossed with four B quotient motifs."""
    a_mode = idx % 5
    b_mode = (idx // 5) % 4

    # The binomial's within-cell ratio has C3 coordinate zero,
    # nonzero C7 coordinate and nonzero V4 coordinate: order 14.
    c = rng.randrange(3)
    start = rng.randrange(7)
    gap = rng.randrange(1, 7)
    axis = (1, 2, 3, 1, 2)[a_mode]
    binomial = [(start, 0, c), ((start + gap) % 7, axis, c)]

    a_cs = ((0, 1, 2), (0, 1, 2), (0, 0, 1),
            (0, 1, 1), (0, 0, 2))[a_mode]
    a_bs = ((0, 1, 2), (0, 1, 3), (0, 1, 2),
            (0, 2, 3), (1, 2, 3))[a_mode]
    A = [[binomial, _triple(rng, a_cs, a_bs)]]
    if a_mode == 4:
        A[0].reverse()

    b_cs = (
        ((0, 1, 2), (0, 1, 2), (0, 1, 2)),
        ((0, 1, 2), (0, 1, 2), (0, 0, 1)),
        ((0, 1, 2), (0, 1, 1), (0, 1, 2)),
        ((0, 1, 2), (0, 1, 1), (0, 2, 2)),
    )[b_mode]
    b_bs = ((0, 1, 2), (0, 2, 3), (1, 2, 3))
    B = [[_triple(rng, b_cs[j], b_bs[j]) for j in range(3)]]
    return A, B


def NEW_CTX(rng, G, idx):
    """Screen independently labeled instances of one indexed design."""
    sa, sb = [[2, 3]], [[3, 3, 3]]
    assert _grid_weights(sa, sb) == 12
    assert _no_dead_lines(sa, sb)

    shortlist = []
    for _ in range(18):
        A, B = _draw_design(rng, idx)
        hx, hz, n = lifted_matrices(G, A, B)
        ac, bc = _component_columns(hx, hz, G.q)
        da, db = _component_cost(ac), _component_cost(bc)
        generation = (G.q - _generated_order(G, A)
                      + G.q - _generated_order(G, B))
        key = (generation, da[0] + db[0], da[1] + db[1],
               max(da[2], db[2]), da[2] + db[2],
               _root_fours(hx, G.q), _root_fours(hz, G.q),
               da[3] + db[3], rng.random())
        shortlist.append((key, A, B, hx, hz, n, ac, bc))
    shortlist.sort(key=lambda entry: entry[0])

    best = None
    for key, A, B, hx, hz, n, ac, bc in shortlist[:5]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        six_a = _root_sixes(ac, G.q)
        six_b = _root_sixes(bc, G.q)
        quality = (not 168 <= k <= 190, key[:-1],
                   six_a, six_b, abs(k - 168), key[-1])
        if best is None or quality < best[0]:
            best = (quality, A, B)
    if best is None:
        raise ValueError("no valid voltage filling")
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without redrawing elements."""
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
# Rank-excess and Tanner-four-cycle screening alone protect the Z13
# (2,5)x(3,8) bed: refuted by this parent's 4.6001 fitness; measured
# slots 22 and 31 had 106/2500 and 108/4250 X-basis failures.
# All-trinomial voltage motifs on the present order-84 bed do not
# reliably suppress its limiting Z failures: all five measured slots
# reached 100-101 Z failures in 16000-119000 shots, despite only 3-21
# X failures per 200000 shots; fitness was 5.4719.
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
