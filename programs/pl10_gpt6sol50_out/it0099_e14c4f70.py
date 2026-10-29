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
CONCEPT = ("In metacyclic Z12:Z1 of order 12, A=(2,6) and B=(3,7), suppressing short A-component kernel words while retaining varied B supports should lower worse-basis circuit failures.")
GROUP = ("Keep the assigned order-12 cyclic metacyclic group: these frozen shapes give n=576, and the current bed's 3.7332 fitness permits a filling-only comparison.")
SHAPES = ("Keep (2,6)x(3,7): k>=192 and rate>=1/3 in the 0.350 cell. The component lengths are unequal, 72 and 84. The parent's weight-five Z-type witnesses occupy sector-1 Acol/Bcol blocks; all five measured slots reached the X-basis failure cap within 500-750 shots, without one dominant check-block hotspot.")
FILLING = ("Spread eight terms across each A row, constrain its two monomial columns against short cyclic relations, and screen labels for component words through weight five. Independently vary B-row totals from seven to nine with extras in distinct columns. Select complete codes by verified rank, component defects, Tanner four-cycles and gate count, targeting failed shots rather than d2s alone.")
FAMILY_NAME = "r035_m12r1s1_26x37_component_balance_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    # Same metacyclic rule, now Z_12 x| Z_1 at the order required by the shape.
    m, r, s = 12, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


def _four_cycles(rows, n):
    """Count repeated pairs of data neighbours among checks."""
    seen = {}
    count = 0
    for bits in rows:
        neighbours = []
        while bits:
            low = bits & -bits
            neighbours.append(low.bit_length() - 1)
            bits -= low
        for p, a in enumerate(neighbours):
            for b in neighbours[p + 1:]:
                key = a * n + b
                count += seen.get(key, 0)
                seen[key] = seen.get(key, 0) + 1
    return count


def _component_stats(grid, side):
    """Count actual component kernel words of weights two through five.

    A columns meet check g*h; B columns meet check h*g^-1.
    The returned counts are proportional to word multiplicities, which
    suffices when comparing independent fillings of the same profile.
    """
    q = 12
    columns = []
    for j in range(len(grid[0])):
        for h in range(q):
            bits = 0
            for i, row in enumerate(grid):
                for a, _ in row[j]:
                    check = (h + a if side == "A" else h - a) % q
                    bits ^= 1 << (i * q + check)
            columns.append(bits)

    singles = {}
    for i, bits in enumerate(columns):
        singles.setdefault(bits, []).append(1 << i)

    pairs = {}
    bad2 = bad3 = bad4 = bad5 = cycles = 0
    for i, left in enumerate(columns):
        mi = 1 << i
        for j in range(i + 1, len(columns)):
            mask = mi | (1 << j)
            value = left ^ columns[j]
            bad2 += value == 0
            bad3 += sum(not (mask & other)
                        for other in singles.get(value, ()))
            old = pairs.setdefault(value, [])
            bad4 += sum(not (mask & other) for other in old)
            old.append(mask)
            overlap = (left & columns[j]).bit_count()
            cycles += overlap * (overlap - 1) // 2

    if not (bad2 or bad3):
        for i in range(len(columns)):
            mi = 1 << i
            for j in range(i + 1, len(columns)):
                partial = columns[i] ^ columns[j]
                mij = mi | (1 << j)
                for t in range(j + 1, len(columns)):
                    mask = mij | (1 << t)
                    bad5 += sum(not (mask & other)
                                for other in pairs.get(
                                    partial ^ columns[t], ()))
    return (bad2, bad3, bad4, bad5, cycles)


def _a_support(rng):
    """Two disjoint doubles per row: A rows weigh eight, columns <=3."""
    sa = [[1] * 6 for _ in range(2)]
    order = rng.sample(range(6), 6)
    for j in order[:2]:
        sa[0][j] = 2
    for j in order[2:4]:
        sa[1][j] = 2
    return sa


def _b_support(rng, idx):
    """Indexed density choices, with no B column receiving two extras."""
    profiles = ((8, 8, 8), (9, 8, 8), (9, 9, 8),
                (8, 8, 7), (9, 8, 7), (8, 9, 7))
    totals = list(profiles[idx % len(profiles)])
    rng.shuffle(totals)
    sb = [[1] * 7 for _ in range(3)]
    order = rng.sample(range(7), 7)
    position = 0
    for i, total in enumerate(totals):
        surplus = total - 7
        for j in order[position:position + surplus]:
            sb[i][j] = 2
        position += surplus
    return sb


def _draw_a(rng, G, sa):
    """Avoid small-order double gaps and a short monomial-only cycle."""
    allowed_gaps = (1, 2, 3, 5, 7, 9, 10, 11)
    for _ in range(90):
        A = []
        for row in sa:
            out = []
            for count in row:
                start = rng.randrange(12)
                if count == 1:
                    out.append([(start, 0)])
                else:
                    gap = rng.choice(allowed_gaps)
                    out.append([(start, 0), ((start + gap) % 12, 0)])
            A.append(out)
        mono = [j for j in range(6) if sa[0][j] == sa[1][j] == 1]
        offsets = [((A[0][j][0][0] - A[1][j][0][0]) % 12)
                   for j in mono]
        if (offsets[0] - offsets[1]) % 12 in (1, 5, 7, 11):
            return A
    raise ValueError("could not find unit-separated monomial columns")


def NEW_CTX(rng, G, idx):
    """Screen independent A and B components, then their verified lifts."""
    sa = _a_support(rng)
    sb = _b_support(rng, idx)
    assert _grid_weights(sa, sb) <= 12

    a_trials = []
    for _ in range(16):
        A = _draw_a(rng, G, sa)
        stats = _component_stats(A, "A")
        a_trials.append((stats, rng.random(), A))
    a_trials.sort(key=lambda trial: (trial[0], trial[1]))

    b_trials = []
    for _ in range(10):
        B = [[rng.sample(G.elements, count) for count in row]
             for row in sb]
        stats = _component_stats(B, "B")
        b_trials.append((stats, rng.random(), B))
    b_trials.sort(key=lambda trial: (trial[0], trial[1]))

    best_key = None
    best = None
    for ast, _, A in a_trials[:3]:
        for bst, _, B in b_trials[:3]:
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k <= 0:
                continue
            key = (k > 210, ast[:3], bst[:3], ast[3], bst[3],
                   2 * _four_cycles(hx, n) + _four_cycles(hz, n),
                   sum(row.bit_count() for row in hx + hz),
                   k - 192)
            if best_key is None or key < best_key:
                best_key, best = key, {"A": A, "B": B}
    if best is None:
        raise ValueError("no positive-k candidate")
    return best


def CELL(rng, G, side, i, j, ctx):
    """Return the screened full cell; its elements are already distinct."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# A high distance-screen proxy alone did not ensure good memory performance:
# parent slots 31 and 50 had d2s=5 but Z failures of 146/750 and 109/750.
# Adding a third B row and selecting mainly for fewer Tanner four-cycles
# improves this cell over its order-17 predecessor: refuted by the current
# 3.7332 versus 4.0553 fitness; all five measured slots reached the
# X-basis failure cap within 500-750 shots.
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
