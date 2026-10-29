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
CONCEPT = ("On dicyclic Dic9 of order 36, shapes (2,3)x(2,4), mixing order-4 cross-coset pairs with order-18 rotation pairs may raise the five-code distance readings in cell 0.150.")
GROUP = ("Jump from cyclic Z37 to the genuinely different Dic9 multiplication law. Order 36 gives n=576 at these shapes; this group-and-shape bed is absent from BEDS, while the cyclic parent reached fitness 38.1752.")
SHAPES = ("Keep A=(2,3), B=(2,4): n=576 and guaranteed k>=36*(3-2)*(4-2)=72, rate>=0.125. The unequal two-row components have three and four columns. The parent's lightest measured X logicals occupied sector-1 Acol/Bcol blocks across all four B columns.")
FILLING = ("Cross five inequivalent legal count profiles with four relative-order rules using order-4 cross-coset, order-18 rotation and order-9 rotation pairs. Both row-weight sums stay at most 12. Screen verified component kernels and exact k to target the parent's distance-16 witnesses while retaining cell 0.150.")
FAMILY_NAME = "r015_dic9_23x24_relative_orders_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    # Dic9: x^18=e, y^2=x^9, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(18)]

    def red(g):
        return (g[0] % 18, g[1] % 2)

    def idx(g):
        a, b = red(g)
        return 18 * b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c) + 9 * b * d) % 18,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 18, 0) if b == 0 else ((a + 9) % 18, 1)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _component_columns(rows, starts, q):
    """Read component columns from rows of the verified LP matrices."""
    columns = [0] * (len(starts) * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows):
        for block, start in enumerate(starts):
            bits = (row >> start) & mask
            while bits:
                bit = bits & -bits
                columns[block * q + bit.bit_length() - 1] |= 1 << check
                bits ^= bit
    return columns


def _kernel_probe(columns, rng):
    """Small verified component-kernel words under independent pivot orders."""
    best = len(columns) + 1
    for _ in range(4):
        order = list(range(len(columns)))
        rng.shuffle(order)
        pivots = {}
        words = []
        for j in order:
            syndrome, word = columns[j], 1 << j
            while syndrome:
                pivot = syndrome.bit_length() - 1
                previous = pivots.get(pivot)
                if previous is None:
                    pivots[pivot] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            else:
                words.append(word)
                best = min(best, word.bit_count())
        words.sort(key=int.bit_count)
        short = words[:20]
        for i, word in enumerate(short):
            for other in short[:i]:
                best = min(best, (word ^ other).bit_count())
        for i, word in enumerate(short[:10]):
            for j, other in enumerate(short[:i]):
                pair = word ^ other
                for third in short[:j]:
                    best = min(best, (pair ^ third).bit_count())
    return best


def _four_cycles(columns):
    """Shared check-neighbour pairs in an exact component lift."""
    total = 0
    for i, left in enumerate(columns):
        for right in columns[:i]:
            shared = (left & right).bit_count()
            total += shared * (shared - 1) // 2
    return total


def _trial_filling(rng, G, profile, motif):
    """Label one of five count profiles with Dic9 relative-order rules."""
    profiles = (
        (((2, 2, 2), (2, 2, 2)),
         ((3, 2, 2, 1), (1, 2, 2, 3))),
        (((3, 2, 2), (1, 2, 2)),
         ((2, 2, 2, 2), (2, 2, 2, 2))),
        (((2, 2, 2), (2, 2, 2)),
         ((2, 2, 2, 2), (2, 2, 2, 2))),
        (((3, 3, 2), (1, 1, 2)),
         ((2, 2, 2, 2), (2, 2, 2, 2))),
        (((2, 2, 2), (2, 2, 2)),
         ((3, 3, 1, 1), (1, 1, 3, 3))),
    )
    ac, bc = profiles[profile]
    assert _grid_weights(ac, bc) <= 12

    def label(counts, side):
        # A common column permutation preserves each count profile's
        # column sums and hence both row-weight budgets.
        permutation = rng.sample(range(len(counts[0])), len(counts[0]))
        grid = []
        for i, row in enumerate(counts):
            labeled = []
            for j in permutation:
                count = row[j]
                coset = rng.randrange(2)
                start = rng.randrange(18)
                if count == 1:
                    cell = [(start, coset)]
                elif count == 2:
                    cross = ((motif == 0 and side == "A") or
                             (motif == 1 and side == "B") or
                             motif == 2 or
                             (motif == 3 and (i + j) % 2 == 0))
                    if cross:
                        cell = [(start, coset),
                                (rng.randrange(18), 1 - coset)]
                    else:
                        gaps = ((2, 4, 8, 10, 14, 16)
                                if motif == 3 and side == "B"
                                else (1, 5, 7, 11, 13, 17))
                        gap = rng.choice(gaps)
                        cell = [(start, coset),
                                ((start + gap) % 18, coset)]
                else:
                    # Every triple spans both cosets; samples within
                    # either coset remain distinct.
                    same = rng.sample(range(18), count - 1)
                    cell = [(a, coset) for a in same]
                    cell.append((rng.randrange(18), 1 - coset))
                labeled.append(cell)
            grid.append(labeled)
        return grid

    A, B = label(ac, "A"), label(bc, "B")
    assert _grid_weights([[len(c) for c in row] for row in A],
                         [[len(c) for c in row] for row in B]) <= 12
    return A, B


def NEW_CTX(rng, G, idx):
    """Screen both complete components, then verify the selected rate."""
    profile, motif = divmod(idx % 20, 4)
    q = G.q
    options = []
    for _ in range(16):
        A, B = _trial_filling(rng, G, profile, motif)
        hx, hz, _ = lifted_matrices(G, A, B)
        a_rows = [hx[i * 4 * q + r] for i in range(2)
                  for r in range(q)]
        ac = _component_columns(
            a_rows, [j * 4 * q for j in range(3)], q)
        bc = _component_columns(
            hz[:2 * q], [j * q for j in range(4)], q)
        da = _kernel_probe(ac, rng)
        db = _kernel_probe(bc, rng)
        cycles = _four_cycles(ac) + _four_cycles(bc)
        grade = (min(da, db), da + db, -cycles, rng.random())
        options.append((grade, A, B, hx, hz))

    fallback = None
    for grade, A, B, hx, hz in sorted(
            options, key=lambda item: item[0], reverse=True):
        k = 576 - gf2_rank(hx) - gf2_rank(hz)
        if fallback is None or (72 <= k <= 100 and
                                not 72 <= fallback[0] <= 100):
            fallback = (k, A, B)
        if 78 <= k <= 96:
            return {"grid": {"A": A, "B": B}}
    return {"grid": {"A": fallback[1], "B": fallback[2]}}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected full cell without redrawing its labels."""
    return list(ctx["grid"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# Mostly-singleton rolled supports suffice for d30s >= 10 in this bed:
# refuted by the parent's five measured slots (four d30s=6, one d30s=8).
# REFUTED: All-two-term A cells and one doubled B cell per column
# reliably attain d30s>=12: four of the current program's five
# measured slots had d30s=11.
# REFUTED: Expanding the verified component-screening profiles alone
# raises this bed beyond distance 12: all five current measured slots
# had d30s=12, with lightest Z witnesses in sector-1 blocks.
# REFUTED: Monomial B cells with denser A rows raise the order-23 bed
# beyond fitness 28.826: its five measured slots had d30s=13 and
# fitness 28.826, with sector-1 Z witnesses still present.
# REFUTED: Uniform all-binomial Z37 supports suffice for five distance-17 readings: the measured all-binomial slot 41 had d30s=16.
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
