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
CONCEPT = ("In cyclic Z29 of order 29 with A=(2,4) and B=(2,4), screening distinct singleton-placement designs against weight-five and weight-six component-kernel words may reduce worse-basis circuit failures.")
GROUP = ("Keep cyclic Z29 at order 29: the measured (2,4)x(2,4) bed reaches fitness 7.3, and prime order gives every nonzero voltage gap full order. This round isolates the filling.")
SHAPES = ("Keep equal (2,4)x(2,4) sides: n=580, guaranteed k>=116 and rate>=0.200, with two length-116 components. Parent X failures included Acol0/Bcol1 and Acol3/Bcol2 data and Xchk:Arow1/Bcol1 in slot 50; slot 30 enriched Zchk:Acol1/Brow1.")
FILLING = ("Cross five singleton profiles with four mixed-gap palettes, keeping one or two distinct terms per cell and check weight <=11. On exact lifted incidences screen weight-five component words in every proposal and weight-six words in finalists, emphasizing the B component implicated by the parent's X-basis-limited slots. Test failed shots and decoder fall-throughs, not d2s or excess k.")
FAMILY_NAME = "r020_z29_24x24_rooted_kernel_words_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 29

def custom_ops(q):
    # Additive cyclic group Z29, encoded with two-coordinate elements.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (g[0] % q, 0)

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    def idx(g):
        return g[0] % q

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 20 * 29 = 580


# Mask bits mark singleton cells. Separated singleton columns keep
# each component column at degree three or four; shared positions
# deliberately provide contrasting designs. Row sums are at most
# seven and column sums at most four, bounding either check by 11.
_SUPPORT_MOTIFS = (
    ((1, 2), (1, 2)),       # degree-balanced on both components
    ((1, 1), (1, 2)),       # light A column, degree-balanced B
    ((1, 2), (1, 1)),       # degree-balanced A, light B column
    ((1, 1), (1, 1)),       # shared light columns on both sides
    ((3, 4), (1, 2)),       # lighter first A row, balanced B
)
# Each palette contains both parity classes. Distinct gaps are drawn
# within a side; palettes change the available spectra, not an orbit
# relabeling of a completed code.
_GAP_PALETTES = (
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
    (1, 2, 3, 5, 7, 8, 11, 12, 13, 14),
    (1, 3, 4, 6, 7, 9, 10, 11, 13, 14),
    (2, 3, 4, 5, 6, 8, 9, 10, 12, 14),
)


def _counts(masks):
    return [[2 - ((mask >> j) & 1) for j in range(4)]
            for mask in masks]


def _draw_side(rng, counts, palette):
    """Independent starts and distinct undirected gaps within each side."""
    pairs = sum(value == 2 for row in counts for value in row)
    gaps = rng.sample(palette, pairs)
    cells = []
    for row in counts:
        filled = []
        for count in row:
            start = rng.randrange(29)
            cell = [(start, 0)]
            if count == 2:
                gap = gaps.pop()
                if rng.randrange(2):
                    gap = -gap
                cell.append(((start + gap) % 29, 0))
            filled.append(cell)
        cells.append(filled)
    return cells


def _component_columns(hx, hz, q, side):
    """Extract both component incidences from the verified LP matrices."""
    if side == "A":
        rows = [hx[(i * B_SHAPE[1]) * q + r]
                for i in range(A_SHAPE[0]) for r in range(q)]
        stride = B_SHAPE[1]
    else:
        rows = [hz[i * q + r]
                for i in range(B_SHAPE[0]) for r in range(q)]
        stride = 1
    columns = [0] * (4 * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows):
        for j in range(4):
            block = (row >> (j * stride * q)) & mask
            while block:
                bit = block & -block
                columns[j * q + bit.bit_length() - 1] |= 1 << check
                block ^= bit
    return columns


def _component_defects(columns):
    """Short kernel relations and pairs of variables sharing checks."""
    singles = set(columns)
    pairs = {}
    bad2 = sum(value == 0 for value in columns)
    bad3 = bad4 = cycles = 0
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            value = left ^ right
            bad2 += value == 0
            bad3 += value in singles
            bad4 += pairs.get(value, 0)
            pairs[value] = pairs.get(value, 0) + 1
            common = (left & right).bit_count()
            cycles += common * (common - 1) // 2
    return bad2, bad3, bad4, cycles


def _kernel_fives(columns, q):
    """Count rooted weight-five kernel words, up to a screening cap.

    Translation symmetry lets a word be rooted at lift coordinate zero
    in one of the four component columns. A rooted triple is compared
    with disjoint pairs of equal syndrome.
    """
    pairs = {}
    size = len(columns)
    for u in range(size):
        for v in range(u + 1, size):
            syndrome = columns[u] ^ columns[v]
            old = pairs.get(syndrome)
            if old is None:
                pairs[syndrome] = ((u, v),)
            elif len(old) < 4:
                pairs[syndrome] = old + ((u, v),)

    hits = 0
    for root in range(0, size, q):
        for u in range(size):
            if u == root:
                continue
            partial = columns[root] ^ columns[u]
            for v in range(u + 1, size):
                if v == root:
                    continue
                for a, b in pairs.get(partial ^ columns[v], ()):
                    if a not in (root, u, v) and b not in (root, u, v):
                        hits += 1
                        if hits >= 64:
                            return hits
    return hits


def _kernel_sixes(columns, q):
    """Count rooted weight-six words among shortlisted components.

    Store several triples per syndrome: retaining only the first can
    miss a disjoint partner when it is the rooted triple itself.
    """
    triples = {}
    size = len(columns)
    for a in range(size):
        for b in range(a + 1, size):
            partial = columns[a] ^ columns[b]
            for c in range(b + 1, size):
                syndrome = partial ^ columns[c]
                old = triples.get(syndrome)
                if old is None:
                    triples[syndrome] = ((a, b, c),)
                elif len(old) < 4:
                    triples[syndrome] = old + ((a, b, c),)

    hits = 0
    for root in range(0, size, q):
        for u in range(size):
            if u == root:
                continue
            partial = columns[root] ^ columns[u]
            for v in range(u + 1, size):
                if v == root:
                    continue
                for a, b, c in triples.get(partial ^ columns[v], ()):
                    if (a not in (root, u, v) and
                            b not in (root, u, v) and
                            c not in (root, u, v)):
                        hits += 1
                        if hits >= 128:
                            return hits
    return hits


def _root_sixes(columns, q):
    """Count component six-cycle configurations at both check-row roots."""
    checks = [0] * (2 * q)
    for variable, column in enumerate(columns):
        while column:
            bit = column & -column
            checks[bit.bit_length() - 1] |= 1 << variable
            column ^= bit

    total = 0
    for root_index in (0, q):
        root = checks[root_index]
        nearby = [(i, root & checks[i]) for i in range(2 * q)
                  if i != root_index and root & checks[i]]
        for a, (u, ru) in enumerate(nearby):
            for v, rv in nearby[:a]:
                total += ((ru & ~rv).bit_count()
                          * (rv & ~ru).bit_count()
                          * (checks[u] & checks[v] & ~root).bit_count())
    return total


def NEW_CTX(rng, G, idx):
    """Select a fresh, complete filling for one of twenty designs."""
    motif = idx % 5
    mode = (idx // 5) % 4
    a_masks, b_masks = _SUPPORT_MOTIFS[motif]
    sa, sb = _counts(a_masks), _counts(b_masks)
    assert _grid_weights(sa, sb) <= 11
    assert _no_dead_lines(sa, sb)

    shortlist = []
    for _ in range(25 + idx % 7):
        A = _draw_side(rng, sa, _GAP_PALETTES[mode])
        B = _draw_side(rng, sb, _GAP_PALETTES[(mode + 1) % 4])
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, hz, G.q, "A")
        bc = _component_columns(hx, hz, G.q, "B")
        da, db = _component_defects(ac), _component_defects(bc)
        five_a = _kernel_fives(ac, G.q)
        five_b = _kernel_fives(bc, G.q)
        # The parent was predominantly X-basis-limited; retain
        # balance while breaking ties in favor of its B component.
        key = (da[0] + db[0], da[1] + db[1],
               max(five_a, five_b), five_a + 2 * five_b,
               max(da[2], db[2]), da[2] + db[2],
               max(da[3], db[3]), da[3] + db[3],
               rng.random())
        shortlist.append((key, A, B, hx, hz, n, ac, bc))
    shortlist.sort(key=lambda item: item[0])

    best = None
    for key, A, B, hx, hz, n, ac, bc in shortlist[:6]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        six_a = _kernel_sixes(ac, G.q)
        six_b = _kernel_sixes(bc, G.q)
        # At n=580, k=116..130 remains in the 0.200 rate cell.
        quality = (not 116 <= k <= 130, key[:4],
                   max(six_a, six_b), six_a + 2 * six_b,
                   key[4:8], abs(k - 122), key[8])
        if best is None or quality < best[0]:
            best = (quality, A, B)
    if best is None:
        raise ValueError("no positive-k Z29 filling")
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without redrawing its labels."""
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
# Sparse Z17 component-word screening suffices to match the 0.350
# frontier: refuted by this parent's 4.3509 fitness; all five measured
# slots reached the Z-basis failure cap within 2000-4000 shots.
# Dense noncommuting order-84 action-mix cells balance memory failures:
# refuted by the current program's 5.4331 fitness; measured slot 27
# had 105 Z failures in 4750 shots but only 3 X failures in 200000.
# Distinct-gap palettes plus short-pair and rooted-cycle screening
# suffice to saturate this Z29 bed: refuted by slot 50's 8 X failures
# and slot 7's 3 X failures, each in 200000 scored shots.
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
