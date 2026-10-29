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
CONCEPT = ("In cyclic Z28 of order 28, A=(2,4) and B=(2,4), balanced singleton/binomial supports with screened order-28 gaps should lower five-code worst-basis circuit failures.")
GROUP = ("Z28 is genuinely different from Dic7 and is untried at these shapes. The related cyclic Z29 bed reached fitness 7.0, versus 6.3262 for the parent's Dic7 program; this is a measured comparison, not a prediction of Z28 fitness.")
SHAPES = ("Retain equal (2,4)x(2,4) shapes: n=560, guaranteed k=112 and rate 0.200, with length-112 two-check components on both sides. Parent failures were broad across sector-1 Acol/Bcol blocks, including Bcol3 in slot 18's Z map and Bcol2 in its X map, without a consistent check-block hotspot.")
FILLING = ("Cross five A and four B support profiles, each with no two-singleton component column, and use three distinct unit-gap spectrum rules. Screen exact-lift component relations and Tanner four-cycles, then require verified k in the 0.200 cell. The target is the parent's slot-8 X rate of 100/26250 rather than an increased distance proxy or excess k.")
FAMILY_NAME = "r020_z28_24x24_balanced_unitgap_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Additive cyclic Z28; the second coordinate is fixed at zero.
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
B_SHAPE = (2, 4)               # (rb, cb); n = 20 * 28 = 560


def _component_relations(rows, q, side):
    """Short component-column relations extracted from verified HX/HZ."""
    cols = [0] * (4 * q)
    mask = (1 << q) - 1
    for i in range(2):
        for r in range(q):
            # HX's A component: X check (i, B-column 0, r).
            # HZ's B component: Z check (A-column 0, i, r).
            row = rows[(i * 4 * q + r) if side == "A"
                       else (i * q + r)]
            for j in range(4):
                block = (row >> (j * (4 if side == "A" else 1) * q)) & mask
                while block:
                    bit = block & -block
                    cols[j * q + bit.bit_length() - 1] |= 1 << (i * q + r)
                    block ^= bit

    pairs = {}
    twos = 0
    for u, left in enumerate(cols):
        for right in cols[u + 1:]:
            syndrome = left ^ right
            twos += syndrome == 0
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    triples = sum(pairs.get(col, 0) for col in cols)
    fours = sum(v * (v - 1) // 2 for v in pairs.values())
    return twos, triples, fours


def _tanner_fours(rows):
    """Count check pairs sharing a pair of data qubits."""
    seen = {}
    cycles = 0
    for row in rows:
        positions = []
        while row:
            bit = row & -row
            positions.append(bit.bit_length() - 1)
            row ^= bit
        for u, left in enumerate(positions):
            for right in positions[u + 1:]:
                key = (left, right)
                cycles += seen.get(key, 0)
                seen[key] = seen.get(key, 0) + 1
    return cycles


def NEW_CTX(rng, G, idx):
    """Screen one of twenty prescribed support-profile pairings."""
    a_profiles = (
        ((2, 2, 1, 1), (1, 1, 2, 2)),  # row sums 6,6
        ((2, 2, 2, 1), (1, 1, 2, 2)),  # 7,6
        ((2, 2, 2, 1), (2, 2, 1, 2)),  # 7,7
        ((2, 2, 2, 2), (2, 2, 1, 1)),  # 8,6
        ((2, 2, 2, 2), (2, 2, 2, 1)),  # 8,7
    )
    b_profiles = (
        ((2, 2, 1, 1), (1, 1, 2, 2)),
        ((2, 2, 2, 1), (1, 1, 2, 2)),
        ((2, 2, 2, 1), (2, 2, 1, 2)),
        ((2, 2, 2, 2), (2, 2, 1, 1)),
    )
    design = idx % 20
    sa = [list(row) for row in a_profiles[design // 4]]
    sb = [list(row) for row in b_profiles[design % 4]]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sa[0][j] + sa[1][j] >= 3 for j in range(4))
    assert all(sb[0][j] + sb[1][j] >= 3 for j in range(4))

    # Representatives of the six inverse pairs of order-28 gaps.
    unit_gaps = (1, 3, 5, 9, 11, 13)
    mode = (idx // 20) % 3

    def draw(supports):
        grid = []
        across_side = list(unit_gaps)
        rng.shuffle(across_side)
        for row in supports:
            cells = []
            within_row = list(unit_gaps)
            rng.shuffle(within_row)
            for count in row:
                start = rng.randrange(28)
                cell = [(start, 0)]
                if count == 2:
                    if mode == 0:       # distinct gaps within this row
                        gap = within_row.pop()
                    elif mode == 1:     # spread gaps over the whole side
                        if not across_side:
                            across_side = list(unit_gaps)
                            rng.shuffle(across_side)
                        gap = across_side.pop()
                    else:              # independent gaps; screen collisions
                        gap = rng.choice(unit_gaps)
                    if rng.randrange(2):
                        gap = -gap
                    cell.append(((start + gap) % 28, 0))
                cells.append(cell)
            grid.append(cells)
        return grid

    trials = []
    for _ in range(20 + idx % 5):
        A, B = draw(sa), draw(sb)
        hx, hz, _ = lifted_matrices(G, A, B)
        at, ath, af = _component_relations(hx, G.q, "A")
        bt, bth, bf = _component_relations(hz, G.q, "B")
        cx, cz = _tanner_fours(hx), _tanner_fours(hz)
        key = (at + bt, ath + bth, max(af, bf), af + bf,
               max(cx, cz), cx + cz, rng.random())
        trials.append((key, A, B))
    trials.sort(key=lambda item: item[0])

    fallback = None
    for key, A, B in trials[:8]:
        k = lifted_k(G, A, B)
        if 112 <= k <= 125:
            return {"A": A, "B": B}
        choice = (max(0, k - 125), key)
        if fallback is None or choice < fallback[0]:
            fallback = (choice, A, B)
    return {"A": fallback[1], "B": fallback[2]}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Predominantly single-term rolled D7 cells suffice for low circuit failure:
# selected parent codes had 101-113 worst-basis failures in 250 shots and
# reported weight-four sector-2 logical operators.
# Unrestricted 2/3-term labels selected by repeated-pair screening alone
# improve this bed: measured five-code fitness fell to 3.3584 versus the
# preceding program's 3.4438, with weight-three witnesses in several slots.
# Dic7 unit-gap screening reliably balances the bases: refuted by slot 8's 100 X failures in 26250 shots versus slot 18's two in 200000, despite both having d2s=11.
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
