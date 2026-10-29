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
CONCEPT = ("In cyclic Z17 of order 17, A=(2,5) and B=(2,6), balanced singleton/binomial components should reduce worst-basis circuit failures relative to one-row Z34 codes.")
GROUP = ("Jump to Z17, genuinely different from Z34. Its measured (2,5)x(2,6) bed reached fitness 4.7; nonidentity gaps have full order, and q=17 gives n=578.")
SHAPES = ("Use two-row (2,5)x(2,6) components of lengths 85 and 102: n=34*17=578 and guaranteed k>=204, rate>=0.353 in cell 0.350. Parent failures and weight-four witnesses spread across sector-1 Acol/Bcol blocks, with check-block enrichment near 1.0.")
FILLING = ("Cross five singleton-placement profiles with four gap palettes for twenty indexed designs. Use only nonzero singleton/binomial cells, verify both row-weight sums <=12, and screen exact lifted component columns for short relations and four-cycles. Test for fewer broadly distributed worse-basis failed shots, not greater proxy or excess k.")
FAMILY_NAME = "r035_z17_25x26_balanced_degree_gap_palettes_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 17

def custom_ops(q):
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
B_SHAPE = (2, 6)               # (rb, cb); n = 34 * 17 = 578


def _verified_component_columns(hx, hz, q):
    """Read both two-row components from the exact LP matrices."""
    mask = (1 << q) - 1
    a, b = [0] * (5 * q), [0] * (6 * q)
    for i in range(2):
        for r in range(q):
            row = hx[(i * 6) * q + r]
            for j in range(5):
                bits = (row >> (j * 6 * q)) & mask
                while bits:
                    bit = bits & -bits
                    a[j * q + bit.bit_length() - 1] |= 1 << (i * q + r)
                    bits ^= bit
    for alpha in range(2):
        for r in range(q):
            row = hz[alpha * q + r]
            for j in range(6):
                bits = (row >> (j * q)) & mask
                while bits:
                    bit = bits & -bits
                    b[j * q + bit.bit_length() - 1] |= 1 << (alpha * q + r)
                    bits ^= bit
    return a, b


def _component_defects(columns):
    """Count short kernel relations and Tanner four-cycles."""
    singles = set(columns)
    pairs = {}
    twos = sum(value == 0 for value in columns)
    triples = fours = cycles = 0
    for i, left in enumerate(columns):
        for j in range(i):
            right = columns[j]
            value = left ^ right
            twos += value == 0
            triples += value in singles
            overlap = (left & right).bit_count()
            cycles += overlap * (overlap - 1) // 2
            entry = pairs.get(value)
            if entry is None:
                pairs[value] = [1, {i: 1, j: 1}]
            else:
                count, incidence = entry
                fours += count - incidence.get(i, 0) - incidence.get(j, 0)
                entry[0] = count + 1
                incidence[i] = incidence.get(i, 0) + 1
                incidence[j] = incidence.get(j, 0) + 1
    return twos, triples // 3, fours // 3, cycles


_GAP_PALETTES = (
    (1, 2, 3, 5, 7, 8),
    (1, 3, 4, 6, 7, 8),
    (2, 3, 4, 5, 6, 8),
    (1, 2, 4, 5, 6, 7),
)


def _support_profile(rng, mode, phase):
    """Two-row counts; every column remains nonzero."""
    a = [[2] * 5 for _ in range(2)]
    b = [[1] * 6 for _ in range(2)]

    if mode == 4:
        # One A column is double in both rows. Four B columns have
        # one double each; this keeps every B row at weight eight.
        both_double = rng.randrange(5)
        other = [j for j in range(5) if j != both_double]
        first = set(rng.sample(other, 2))
        for j in other:
            a[0 if j in first else 1][j] = 1
        first_b = set(rng.sample(range(6), 2))
        remaining = [j for j in range(6) if j not in first_b]
        second_b = set(rng.sample(remaining, 2))
    else:
        first_count = 1 + phase if mode in (0, 2) else 2
        first = set(rng.sample(range(5), first_count))
        for j in range(5):
            a[0 if j in first else 1][j] = 1
        if mode in (1, 3):
            extra_row = phase % 2
            choices = [j for j in range(5) if a[extra_row][j] == 2]
            a[extra_row][rng.choice(choices)] = 1
        first_b = set(rng.sample(range(6), 3))
        second_b = set(range(6)) - first_b
        if mode in (2, 3):
            # Turn one B binomial into a singleton.
            chosen = rng.choice(tuple(sorted(first_b | second_b)))
            if chosen in first_b:
                first_b.remove(chosen)
            else:
                second_b.remove(chosen)

    for j in first_b:
        b[0][j] = 2
    for j in second_b:
        b[1][j] = 2
    assert _grid_weights(a, b) <= 12
    assert _no_dead_lines(a, b)
    return a, b


def _draw_component(rng, counts, palette):
    grid = []
    for row in counts:
        cells = []
        for count in row:
            start = rng.randrange(17)
            cell = [(start, 0)]
            if count == 2:
                gap = rng.choice(palette)
                if rng.randrange(2):
                    gap = -gap
                cell.append(((start + gap) % 17, 0))
            cells.append(cell)
        grid.append(cells)
    return grid


def NEW_CTX(rng, G, idx):
    """Screen labels for an indexed support profile and gap-palette pair."""
    mode, phase = idx % 5, (idx // 5) % 4
    ac, bc = _support_profile(rng, mode, phase)
    ap = _GAP_PALETTES[phase]
    bp = _GAP_PALETTES[(phase + 1 + mode) % 4]

    shortlist = []
    for _ in range(14 + idx % 5):
        A = _draw_component(rng, ac, ap)
        B = _draw_component(rng, bc, bp)
        hx, hz, n = lifted_matrices(G, A, B)
        acol, bcol = _verified_component_columns(hx, hz, G.q)
        da, db = _component_defects(acol), _component_defects(bcol)
        key = (da[0] + db[0], da[1] + db[1],
               max(da[2], db[2]), da[2] + db[2],
               max(da[3], db[3]), da[3] + db[3],
               rng.random())
        shortlist.append((key, A, B, hx, hz, n))
    shortlist.sort(key=lambda item: item[0])

    best = None
    for key, A, B, hx, hz, n in shortlist[:5]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        # At n=578, this range stays inside the 0.350 rate cell.
        quality = (not 188 <= k <= 216, key, abs(k - 204))
        if best is None or quality < best[0]:
            best = (quality, A, B)
    if best is None:
        raise ValueError("no positive-k Z17 filling")
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
# REFUTED: Triangle-free mixed-gap pairs with parity-mixed trinomials
# improve the Z34 bed: fitness was 3.4286 versus its earlier 3.4845;
# measured slots retained widespread sector-1 weight-four witnesses.
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
