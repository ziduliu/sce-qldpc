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
CONCEPT = ("In Q8 x C2 of order 16, A=(2,5) and B=(2,6), full-group-generating binomial voltages screened for short component relations should reduce worst-basis circuit failures.")
GROUP = ("Jump from cyclic Z34 to noncommutative Q8 x C2. Its order gives n=544; its centre has order four, unlike the recorded order-16 nonabelian (2,5)x(2,6) bed with centre two.")
SHAPES = ("Use (2,5)x(2,6): n=34*16=544 and guaranteed k=192, rate 0.3529 in cell 0.350. The two-row components have lengths 80 and 96. Parent failures and weight-four witnesses spread across sector-1 Acol/Bcol blocks, without a persistent check-block hotspot.")
FILLING = ("Cross indexed singleton/binomial support profiles, require each side's noncentral voltage ratios to generate Q8 x C2, and screen exact lifted components for short relations. Both check-row sums stay at most 12. Test for fewer widely distributed logical failures than the parent's 103-196 failed shots per 250-500 shots, while screening k into cell 0.350.")
FAMILY_NAME = "r035_q8xc2_25x26_generating_component_screen_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Q8 x C2: x^4=1, y^2=x^2, y*x*y^-1=x^-1;
    # z is an independent central involution.
    els = [(a, b, c) for c in range(2)
           for b in range(2) for a in range(4)]

    def red(g):
        return (int(g[0]) % 4, int(g[1]) % 2, int(g[2]) % 2)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + (d if b == 0 else -d) + 2*b*e) % 4,
                (b + e) % 2, (c + f) % 2)

    def inv(g):
        a, b, c = red(g)
        return ((-a if b == 0 else a + 2) % 4, b, c)

    def idx(g):
        a, b, c = red(g)
        return a + 4*b + 8*c

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 6)               # (rb, cb); n = 34 * 16 = 544


def _component_columns(rows, q, width, stride, block_step):
    """Extract both check rows of a component from the verified LP lift."""
    columns = [0] * (width * q)
    mask = (1 << q) - 1
    for i in range(2):
        for r in range(q):
            check = i*q + r
            row = rows[i*block_step*q + r]
            for j in range(width):
                bits = (row >> (j*stride*q)) & mask
                while bits:
                    bit = bits & -bits
                    columns[j*q + bit.bit_length() - 1] |= 1 << check
                    bits ^= bit
    return columns


def _component_cost(columns):
    """Short kernel relations and repeated check-pair incidences."""
    singles = set(columns)
    pairs = {}
    two = three = four = cycles = 0
    for i, left in enumerate(columns):
        two += left == 0
        for right in columns[i+1:]:
            value = left ^ right
            two += value == 0
            three += value in singles
            four += pairs.get(value, 0)
            pairs[value] = pairs.get(value, 0) + 1
            overlap = (left & right).bit_count()
            cycles += overlap*(overlap - 1)//2
    return two, three, four, cycles


def _generated_order(G, grid):
    """Subgroup generated by within-cell voltage ratios."""
    generators = [G.mul(G.inv(cell[0]), cell[1])
                  for row in grid for cell in row if len(cell) == 2]
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


def _supports(idx):
    """Twenty indexed support placements across five weight profiles."""
    mode, motif = idx % 5, (idx // 5) % 4
    a_sizes = ((3, 3), (2, 2), (3, 2), (2, 3), (2, 2))[mode]
    b_sizes = ((2, 2), (3, 3), (2, 3), (3, 2), (2, 2))[mode]
    start = motif % 5

    if mode == 0:
        a0 = {(start + t) % 5 for t in (0, 1, 2)}
        a1 = {(start + t) % 5 for t in (0, 2, 3)}
    else:
        a0 = {(start + t) % 5 for t in range(a_sizes[0])}
        available = [j for j in range(5) if j not in a0]
        a1 = set(available[:a_sizes[1]])
    b0 = {(motif + t) % 6 for t in range(b_sizes[0])}
    b1 = {(motif + 2*t + mode + 1) % 6
          for t in range(b_sizes[1])}
    # When three positions are requested, step two visits three
    # distinct columns modulo six.
    sa = [[1 + (j in doubles) for j in range(5)]
          for doubles in (a0, a1)]
    sb = [[1 + (j in doubles) for j in range(6)]
          for doubles in (b0, b1)]
    return sa, sb


def NEW_CTX(rng, G, idx):
    """Screen fresh, full-generation labels for an indexed support design."""
    sa, sb = _supports(idx)
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    noncentral = [g for g in G.elements if g[1] == 1 or g[0] % 2]
    assert len(noncentral) == 12

    def draw(counts):
        grid = []
        for row in counts:
            filled = []
            for count in row:
                first = G.pick(rng)
                cell = [first]
                if count == 2:
                    cell.append(G.mul(first, rng.choice(noncentral)))
                filled.append(cell)
            grid.append(filled)
        return grid

    shortlist = []
    for trial in range(28):
        A, B = draw(sa), draw(sb)
        if _generated_order(G, A) != G.q or _generated_order(G, B) != G.q:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        ac = _component_columns(hx, G.q, 5, 6, 6)
        bc = _component_columns(hz, G.q, 6, 1, 1)
        da, db = _component_cost(ac), _component_cost(bc)
        key = (da[0] + db[0], da[1] + db[1],
               max(da[2], db[2]), da[2] + db[2],
               max(da[3], db[3]), da[3] + db[3], rng.random())
        shortlist.append((key, A, B, hx, hz, n))
    if not shortlist:
        raise ValueError("no full-generation filling")
    shortlist.sort(key=lambda item: item[0])

    best = None
    for key, A, B, hx, hz, n in shortlist[:8]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        # Cell 0.350 occupies k/n from 0.325 to below 0.375.
        quality = (not 177 <= k <= 203, key, abs(k - 192))
        if best is None or quality < best[0]:
            best = (quality, A, B)
    if best is None:
        raise ValueError("no positive-k filling")
    return {"A": best[1], "B": best[2]}


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected cell, without redrawing its terms."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Parent 6x2 rolled bed did not protect its sector-2 blocks: all five
# measured slots had weight-four logicals there and >=100 failures/basis
# within 250-1750 shots.
# REFUTED: Shrinking to (1,3)x(1,5) with screened mixed-count cells
# removes the dominant failures: all five measured slots still had
# sector-1 weight-two witnesses and 201-243 Z-basis failures per
# 250 shots, for five-code fitness 2.4912.
# REFUTED: Short-component-relation screening in the dihedral order-34
# (1,4)x(1,4) bed reliably suppresses sector-1 failures: its five
# measured slots scored 3.10-3.29, and all reached the failure cap
# after only 250 shots in their worse basis.
# REFUTED: The indexed four-binomial Z34 gap rule supplies twenty distinct initial designs: slots 0-19 collapsed to two isomorphism classes; measured mixed-support slots still had widespread weight-four witnesses.
# REFUTED: Three odd binomials plus one parity-mixed trinomial per side
# improve the earlier Z34 filling: fitness fell from 3.4845 to 3.4296;
# measured slots still had sector-1 weight-four witnesses and roughly
# 100-120 worse-basis failures within 250 shots.
# REFUTED: Joint all-odd four-binomial spectra provide twenty distinct,
# lower-failure designs: only five of 60 slots were distinct, fitness
# was 3.3174, and each measured basis reached its failure cap in
# 250 shots.
# REFUTED: Triangle-free mixed-parity Z34 trinomials reliably beat the
# earlier 3.4845 filling: current fitness was 3.4286, with sector-1
# weight-four witnesses and 103-196 failures in measured bases.
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
