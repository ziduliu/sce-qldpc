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
CONCEPT = ("Dicyclic Dic6, order 24, A=(2,4), B=(2,5): three-term B columns and order-12 doubled-cell differences should reduce worst-basis circuit failures.")
GROUP = ("Dic6 has y^2=x^6 and yxy^-1=x^-1, unlike cyclic Z23 and the earlier order-24 metacyclic bed. Its order gives n=576 on this untried group-and-shape bed.")
SHAPES = ("Keep (2,4)x(2,5): n=576 and guaranteed k/n=144/576=0.250. The unequal four- and five-column components leave sector 1 larger. The parent's worse Z-basis failures were spread across Acol/Bcol data blocks, including Bcol3, without a pronounced check-block hotspot.")
FILLING = ("Cross five A and four B count profiles. Every B column has three terms; HX weighs at most 11 and HZ at most 12. Use relative rotations of order 12 within doubled cells, then screen independent labels with verified lifts for short component relations, rank and four-cycles. Test whether this reduces the parent's Z-basis failed-shot rate without relying on k excess.")
FAMILY_NAME = "r025_dic6_24x25_order12_component_screen"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    # Dic6 = <x,y | x^12=1, y^2=x^6, yxy^-1=x^-1>.
    els = [(a, b) for b in range(2) for a in range(12)]

    def red(g):
        return (g[0] % 12, g[1] % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 6 * b * d) % 12,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 12, 0) if b == 0 else ((a + 6) % 12, 1)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=lambda g: 12 * (g[1] % 2) + g[0] % 12, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _four_cycles(rows, n):
    """Count pairs of checks sharing two qubits, separately per basis."""
    seen = {}
    cycles = 0
    for bits in rows:
        cols = []
        while bits:
            low = bits & -bits
            cols.append(low.bit_length() - 1)
            bits ^= low
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                key = cols[a] * n + cols[b]
                previous = seen.get(key, 0)
                cycles += previous
                seen[key] = previous + 1
    return cycles


def _component_relations(hx, hz, q):
    """Screen component columns extracted from the verified LP matrices."""
    mask = (1 << q) - 1

    def columns(rows, width, stride, row_start):
        values = [0] * (width * q)
        for i in range(2):
            for r in range(q):
                bits = rows[row_start(i) + r]
                check_bit = 1 << (i * q + r)
                for j in range(width):
                    chunk = (bits >> (j * stride * q)) & mask
                    while chunk:
                        low = chunk & -chunk
                        values[j * q + low.bit_length() - 1] |= check_bit
                        chunk ^= low
        return values

    # HX: beta=0, both A rows. HZ: j=0, both B rows.
    components = (
        columns(hx, 4, 5, lambda i: i * 5 * q),
        columns(hz, 5, 1, lambda i: i * q),
    )
    bad2 = bad3 = bad4 = 0
    for values in components:
        singles = set(values)
        pairs = {}
        for a, left in enumerate(values):
            for b in range(a + 1, len(values)):
                syndrome = left ^ values[b]
                bad2 += syndrome == 0
                bad3 += syndrome in singles
                old = pairs.get(syndrome)
                if old is None:
                    pairs[syndrome] = (a, b)
                elif a not in old and b not in old:
                    bad4 += 1
    return bad2, bad3, bad4


def NEW_CTX(rng, G, idx):
    """Select independent labels for one indexed, weight-legal design."""
    a_profiles = (
        ((2, 2, 2, 1), (1, 2, 2, 2)),
        ((2, 2, 1, 2), (2, 1, 2, 2)),
        ((2, 2, 2, 2), (1, 2, 2, 1)),
        ((2, 2, 2, 1), (2, 1, 2, 2)),
        ((2, 1, 2, 2), (1, 2, 2, 2)),
    )
    # The second row complements the first: every B column totals three.
    b_double_positions = (
        (0, 1), (0, 2), (0, 1, 3), (0, 2, 4),
    )
    sa = a_profiles[(idx // 4) % len(a_profiles)]
    doubled = set(b_double_positions[idx % 4])
    sb = (
        tuple(2 if j in doubled else 1 for j in range(5)),
        tuple(1 if j in doubled else 2 for j in range(5)),
    )
    assert _grid_weights(sa, sb) <= W_MAX

    def draw(count):
        first = G.pick(rng)
        if count == 1:
            return [first]
        # A rotation difference coprime to 12 has order 12. In
        # particular, no doubled cell introduces an involution gap.
        difference = (rng.choice((1, 5, 7, 11)), 0)
        return [first, G.mul(first, difference)]

    trials = []
    for _ in range(10):
        A = [[draw(count) for count in row] for row in sa]
        B = [[draw(count) for count in row] for row in sb]
        hx, hz, n = lifted_matrices(G, A, B)
        short = _component_relations(hx, hz, G.q)
        trials.append((short, A, B, hx, hz, n))

    # Rank and Tanner-cycle checks are reserved for promising component
    # fillings. Every final candidate remains independently labelled.
    trials.sort(key=lambda item: item[0])
    bound = G.q * (A_SHAPE[1] - A_SHAPE[0]) * (B_SHAPE[1] - B_SHAPE[0])
    best = None
    for short, A, B, hx, hz, n in trials[:4]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        cost = (abs(k - bound), short,
                _four_cycles(hx, n) + _four_cycles(hz, n))
        if best is None or cost < best[0]:
            best = (cost, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell, without changing its labels."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Parent (6,2)x(6,2) mostly-monomial fillings did not suppress sector-2
# weight-4 logicals: its five measured scores were 3.30--3.63.
# Rank and Tanner-four-cycle screening alone did not prevent weak X-basis
# slots at q=24: slots 37 and 43 failed 100/10750 and 102/25500 shots.
# REFUTED: The Z23 balanced-cycle filling already matches the stronger
# neighbouring rate cells: it scored 5.7578 at rate 0.250, and measured
# slot 3 had 101 Z-basis failures in 35750 shots.
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
