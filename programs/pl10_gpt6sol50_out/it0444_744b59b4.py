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
CONCEPT = ("In dihedral D13 of order 26, A=(1,3) and B=(2,7), independently screened reflection patterns and sparse B supports should reduce worst-basis circuit failures.")
GROUP = ("Replace cyclic order-26 Z13 x Z2 by nonabelian D13 at the same lift order. The parent scored 4.1191, but all five measured slots reached the failed-shot cap in both bases; n stays 598 for a direct test.")
SHAPES = ("Keep (1,3)x(2,7): n=598, guaranteed k>=260 and rate>=0.435 in cell 0.450. Its unequal components have 78 and 182 variables. Parent failures were diffuse across Acol/Bcol data blocks, with slot 40's Z failures enriched 1.2x at Bcol4 and little check-block concentration.")
FILLING = ("Cross five genuinely different B support profiles with four reflection protocols. Screen component relations extracted from verified lifts, then retain target-cell k; both check types have weight at most 12. Test for fewer worst-basis failed shots and decoder fallthroughs, not higher proxy alone.")
FAMILY_NAME = "r045_dihedral13_13x27_reflection_support_screen_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 26

def custom_ops(q):
    # D13 = <x,y | x^13=y^2=1, y*x*y=x^-1>.
    els = [(a, b) for b in range(2) for a in range(13)]

    def red(g):
        return (int(g[0]) % 13, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-c if b else c)) % 13, (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return (a if b else (-a) % 13, b)

    def idx(g):
        a, b = red(g)
        return 13 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb); n = 23 * 26 = 598


def _verified_component_columns(rows, q, stride, length, check_rows=1):
    """Read all component check rows from the scaffold's exact lift."""
    columns = [0] * (length * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:check_rows * q]):
        for j in range(length):
            bits = (row >> (j * stride * q)) & mask
            while bits:
                low = bits & -bits
                columns[j * q + low.bit_length() - 1] |= 1 << check
                bits ^= low
    return columns


def _component_defects(columns):
    """Count short relations and pairs of columns sharing two checks."""
    singles = {}
    for col in columns:
        singles[col] = singles.get(col, 0) + 1
    twos = sum(v * (v - 1) // 2 for v in singles.values())

    pairs = {}
    threes = fours = 0
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            syndrome = left ^ right
            if syndrome:
                threes += singles.get(syndrome, 0)
                fours += pairs.get(syndrome, 0)
                pairs[syndrome] = pairs.get(syndrome, 0) + 1

    check_pairs = {}
    cycles = 0
    for col in columns:
        checks = []
        while col:
            low = col & -col
            checks.append(low.bit_length() - 1)
            col ^= low
        for a in range(len(checks)):
            for b in range(a + 1, len(checks)):
                pair = (checks[a], checks[b])
                cycles += check_pairs.get(pair, 0)
                check_pairs[pair] = check_pairs.get(pair, 0) + 1
    return twos, threes // 3, fours, cycles


def _dihedral_cell(rng, G, count, reflections):
    """Choose distinct voltages with controlled rotation/reflection mix."""
    if not count:
        return []
    rotations = [(a, 0) for a in range(13)]
    mirrors = [(a, 1) for a in range(13)]
    for _ in range(40):
        cell = (rng.sample(mirrors, reflections) +
                rng.sample(rotations, count - reflections))
        if count < 3:
            return cell
        # Avoid an immediate involution relation inside a cell.
        differences = [
            G.mul(G.inv(cell[u]), cell[v])
            for u in range(count) for v in range(u)
        ]
        if len(set(differences)) == len(differences):
            return cell
    return cell


def NEW_CTX(rng, G, idx):
    """Screen one of twenty support/reflection designs on exact lifts."""
    design = idx % 20
    profile, mode = design % 5, design // 5
    shortlist = []

    for _ in range(30 + idx % 7):
        # All A cells are trinomials; the four protocols give genuinely
        # different reflection populations, not automorphism copies.
        reflection_counts = (
            (1, 1, 1), (0, 1, 2), (2, 2, 2), (1, 2, 0)
        )[mode]
        A = [[_dihedral_cell(rng, G, 3, reflection_counts[j])
              for j in range(3)]]

        order = rng.sample(range(7), 7)
        B = [[_dihedral_cell(rng, G, 1, (mode + row + j) % 2)
              for j in range(7)] for row in range(2)]

        if profile == 0:
            # Four split binomials; B row sums 9/9.
            isolated, reinforced = [], [
                (0, order[0]), (0, order[1]),
                (1, order[2]), (1, order[3])]
        elif profile == 1:
            # Two isolated trinomials; B row sums 8/8.
            isolated, reinforced = [
                (0, order[0]), (1, order[1])], []
        elif profile == 2:
            # Two isolated trinomials and two split binomials: 9/9.
            isolated, reinforced = [
                (0, order[0]), (1, order[1])], [
                (0, order[2]), (1, order[3])]
        elif profile == 3:
            # Four isolated trinomials, balanced two per row: 9/9.
            isolated, reinforced = [
                (0, order[0]), (1, order[1]),
                (0, order[2]), (1, order[3])], []
        else:
            # Two split binomials; B row sums 8/8.
            isolated, reinforced = [], [
                (0, order[0]), (1, order[1])]

        for row, j in isolated:
            B[row][j] = _dihedral_cell(
                rng, G, 3, 1 + (mode + row) % 2)
            B[1 - row][j] = []
        for row, j in reinforced:
            first = B[row][j][0]
            for _attempt in range(20):
                second = rng.choice(G.elements)
                if second != first and G.mul(G.inv(first), second) != G.e:
                    break
            B[row][j] = [first, second]

        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _grid_weights(sa, sb) <= 12
        assert _no_dead_lines(sa, sb)
        assert all(len(cell) == len(set(cell))
                   for grid in (A, B) for row in grid for cell in row)

        hx, hz, n = lifted_matrices(G, A, B)
        ac = _verified_component_columns(
            hx, G.q, B_SHAPE[1], A_SHAPE[1])
        bc = _verified_component_columns(
            hz, G.q, 1, B_SHAPE[1], B_SHAPE[0])
        a = _component_defects(ac)
        b = _component_defects(bc)
        # Preserve balance: a low count on one side cannot conceal
        # numerous short relations on the other.
        quality = (a[0] + b[0], a[1] + b[1],
                   max(a[2], b[2]), a[2] + b[2],
                   max(a[3], b[3]), a[3] + b[3],
                   rng.random())
        shortlist.append((quality, A, B, hx, hz, n))

    shortlist.sort(key=lambda item: item[0])
    for _, A, B, hx, hz, n in shortlist[:10]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if 260 <= k <= 284:
            return {"A": A, "B": B}
    _, A, B, _, _, _ = shortlist[0]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Predominantly single-term rolled D7 cells suffice for low circuit failure:
# selected parent codes had 101-113 worst-basis failures in 250 shots and
# reported weight-four sector-2 logical operators.
# Pair-signature screening alone did not eliminate the current bed's weight-3/4 sector-1 logicals or its frequent measured failures.
# Penalizing component-kernel weights 2-4 across five profiles did not
# balance the bases: fitness fell to 3.3540 and all five measured slots
# were Z-limited, including slot 48 with 160 failures in 250 shots.
# Further density-and-cycle screening on Z7 x| Z6 did not remove
# frequent failures: the current parent scored 3.5312, with slot 10
# reaching 192 X-basis failures in 500 shots.
# Balanced cyclic order-26 supports screened for component defects did
# not make failures rare: the present parent's five measured slots all
# reached the failed-shot cap in both bases and had weight-six witnesses.
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
