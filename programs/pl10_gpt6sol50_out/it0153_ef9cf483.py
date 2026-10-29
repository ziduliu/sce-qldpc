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
CONCEPT = ("In abelian C14 x C2 of order 28, equal (2,4)x(2,4) binomial sides screened on their verified component lifts may reduce worse-basis circuit failures in the 0.250 rate cell.")
GROUP = ("Jump from the parent's dihedral order-28 group to noncyclic abelian C14 x C2, an untried group-and-shapes bed. Related order-28 (2,4)x(2,4) beds reached 6.4, but that does not predict this group's measured pL.")
SHAPES = ("Equal (2,4)x(2,4) sides give n=560 and guaranteed k=112, rate 0.200; select verified k=126..153 for the 0.250 cell. Both component codes have 112 variables. The parent's X-heavy failures and short logicals occupied sector-one Acol/Bcol blocks, without a dominant check block.")
FILLING = ("Use twenty indexed, independently labelled binomial gap designs. Every cell has two distinct elements: HX and HZ rows both weigh 8+4=12. Screen short cycles in the verified component lifts, then verified k; test whether this lowers the parent's 103-135 X-basis failures within 500-1000 shots. Cycles select candidates, not fitness.")
FAMILY_NAME = "r025_c14xc2_24x24_verified_component_cycles_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 28

def custom_ops(q):
    # Direct product C14 x C2; noncyclic and not dihedral.
    els = [(a, b) for b in range(2) for a in range(14)]

    def red(g):
        return (g[0] % 14, g[1] % 2)

    def idx(g):
        a, b = red(g)
        return 14 * b + a

    def mul(g, h):
        return ((g[0] + h[0]) % 14, (g[1] + h[1]) % 2)

    def inv(g):
        return ((-g[0]) % 14, (-g[1]) % 2)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 4)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb); n = 20 * 28 = 560


def _component_rows(hx, hz, q, side):
    """Read a two-row component directly from the verified LP matrices."""
    mask = (1 << q) - 1
    out = []
    for i in range(2):
        for r in range(q):
            source = (hx[(i * 4) * q + r] if side == "A"
                      else hz[i * q + r])
            row = 0
            for j in range(4):
                base = (j * 4 * q if side == "A" else j * q)
                row |= ((source >> base) & mask) << (j * q)
            out.append(row)
    return out


def _component_cycles(rows, q):
    """Count rooted four-, six-, eight-cycles in the actual component."""
    checks = [[] for _ in range(2 * q)]
    variables = [[] for _ in range(4 * q)]
    for c, value in enumerate(rows):
        while value:
            bit = value & -value
            v = bit.bit_length() - 1
            checks[c].append(v)
            variables[v].append(c)
            value ^= bit

    adjacent = [[] for _ in checks]
    for c, vs in enumerate(checks):
        for v in vs:
            adjacent[c].extend((other, v) for other in variables[v]
                               if other != c)

    four = six = eight = 0
    # Regular translation relates lift coordinates within each check row.
    for root in (0, q):
        incident = {}
        for other, first in adjacent[root]:
            previous = incident.setdefault(other, [])
            four += len(previous)
            previous.append(first)

        paths = {}
        for middle, first in adjacent[root]:
            for end, second in adjacent[middle]:
                if end == root or second == first:
                    continue
                for last in incident.get(end, ()):
                    if last != first and last != second:
                        six += 1
                earlier = paths.setdefault(end, [])
                for old_middle, old_first, old_second in earlier:
                    if old_middle != middle and len({
                            first, second, old_first, old_second}) == 4:
                        eight += 1
                earlier.append((middle, first, second))
    return four, six, eight


def _binomial_grid(rng, idx, side):
    """One independent draw from this slot's prescribed voltage profile."""
    mode = idx % 5
    phase = idx // 5
    grid = []
    for i in range(2):
        line = []
        for j in range(4):
            t = 4 * i + j
            start = rng.randrange(14)
            sign = rng.randrange(2)
            if mode == 0:
                gap = 2 * rng.randrange(7) + 1
            elif mode == 1:
                gap = rng.choice((1, 3, 5, 9, 11, 13))
            elif mode == 2:
                gap = rng.choice((1, 3, 5, 7, 9, 11, 13)
                                 if t % 3 else (2, 4, 6, 8, 10, 12))
            elif mode == 3:
                gap = rng.choice((1, 3, 5, 9, 11, 13)
                                 if t % 4 else (2, 6, 10))
            else:
                gap = 2 * rng.randrange(7) + 1
            change_coset = (rng.randrange(2) if mode in (0, 2)
                            else (t + phase + (side == "B")) % 2)
            line.append([(start, sign),
                         ((start + gap) % 14, sign ^ change_coset)])
        grid.append(line)
    return grid


def NEW_CTX(rng, G, idx):
    """Choose a complete, independently labelled design for each slot."""
    finalists = []
    q = G.q
    for _ in range(24 + idx % 7):
        A = _binomial_grid(rng, idx, "A")
        B = _binomial_grid(rng, idx, "B")
        hx, hz, _ = lifted_matrices(G, A, B)
        ac = _component_cycles(_component_rows(hx, hz, q, "A"), q)
        bc = _component_cycles(_component_rows(hx, hz, q, "B"), q)
        key = (max(ac[0], bc[0]), ac[0] + bc[0],
               max(ac[1], bc[1]), ac[1] + bc[1],
               max(ac[2], bc[2]), ac[2] + bc[2], rng.random())
        finalists.append((key, A, B))
    finalists.sort(key=lambda entry: entry[0])

    # n=560: the interior of the 0.250 cell is 126 <= k <= 153.
    for _, A, B in finalists[:10]:
        if 126 <= lifted_k(G, A, B) <= 153:
            return {"A": A, "B": B}
    # Retain a legal design even if this slot's trials miss the rate cell.
    _, A, B = finalists[0]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full, distinct-element binomial cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Refuted by parent slots 5 and 52: legal uniform labels do not reliably
# balance bases; X reached 100 failures after 7500 and 18000 shots while
# the corresponding Z measurements remained below 100 at 200000 shots.
# Refuted by this bed's slot 0: mixed-coset two-term A support was adequate
# against short logicals; it produced weight-two Z logicals and 234 X
# failures in 250 shots. Slot 31's all-three-term A row did better (4.14).
# Refuted by the current parent's measured slots: screening odd-rotation
# pairs and short component words did not make X failures rare; its five
# measured worse-basis scores were only 3.57-3.97.
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
