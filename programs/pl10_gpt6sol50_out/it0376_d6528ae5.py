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
CONCEPT = ("In C5 x C5 of order 25, A=(1,3) and B=(2,7), screened distinct sparse-support motifs should reduce worst-basis circuit failures in rate cell 0.450.")
GROUP = ("Jump from the parent's order-42 nonabelian group to C5 x C5. Its (1,3)x(2,7) bed is absent from BEDS; the parent's measured fitness was 3.6049.")
SHAPES = ("A=(1,3), B=(2,7) give n=575, guaranteed k>=250 and rate>=0.435. Their unequal components have 75 and 175 variables. Parent failures were diffuse across sector-1 Acol/Bcol blocks, with modest Bcol1/Bcol3 enrichment and no dominant check block.")
FILLING = ("Cross five support motifs with four voltage protocols: split B binomials, isolated B trinomials, or a shared reinforced column. Screen complete verified lifts for short component relations and check target-cell k. Test for fewer limiting X failures than the parent's 125 in 500 shots in slot 2, not for excess k.")
FAMILY_NAME = "r045_c5xc5_13x27_five_support_four_voltage_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 25

def custom_ops(q):
    # Additive C5 x C5, not cyclic: every nonidentity element has order 5.
    els = [(a, b) for a in range(5) for b in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 5)

    def mul(g, h):
        return ((g[0] + h[0]) % 5, (g[1] + h[1]) % 5)

    def inv(g):
        return ((-g[0]) % 5, (-g[1]) % 5)

    def idx(g):
        a, b = red(g)
        return 5 * a + b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb); n = 23 * 25 = 575


def _component_columns(rows, q, stride, width, check_blocks):
    """Extract both component incidences from the exact scaffold lifts."""
    columns = [0] * (width * q)
    mask = (1 << q) - 1
    for check, row in enumerate(rows[:check_blocks * q]):
        for j in range(width):
            bits = (row >> (j * stride * q)) & mask
            while bits:
                low = bits & -bits
                columns[j * q + low.bit_length() - 1] |= 1 << check
                bits ^= low
    return columns


def _component_defects(columns):
    """Count zero pairs, zero triples, distinct zero fours and 4-cycles."""
    singles = set(columns)
    pairs = {}
    twos = threes = fours = cycles = 0
    for i, left in enumerate(columns):
        for j in range(i):
            right = columns[j]
            syndrome = left ^ right
            twos += syndrome == 0
            threes += syndrome in singles
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = (i, j)
            elif i not in previous and j not in previous:
                fours += 1
            overlap = (left & right).bit_count()
            cycles += overlap * (overlap - 1) // 2
    return twos, threes, fours, cycles


def _direction(g, h):
    """Canonical one-dimensional subgroup of a nonzero difference."""
    x, y = ((h[0] - g[0]) % 5, (h[1] - g[1]) % 5)
    return min(((t * x) % 5, (t * y) % 5) for t in range(1, 5))


def _draw_cell(rng, G, count, spread):
    if not count:
        return []
    for _ in range(40):
        cell = rng.sample(G.elements, count)
        if count != 3 or not spread or \
                _direction(cell[0], cell[1]) != \
                _direction(cell[0], cell[2]):
            return cell
    return rng.sample(G.elements, count)


def _support_motif(rng, motif):
    """Five different component profiles, all with maximum check weight 12."""
    ac = [3, 3, 3] if motif < 3 else [3, 3, 2]
    rng.shuffle(ac)
    sb = [[1] * 7 for _ in range(2)]
    order = rng.sample(range(7), 7)

    if motif == 0:
        # Four split binomials: row sums 9/9, column sums at most 3.
        for row, marked in ((0, order[:2]), (1, order[2:4])):
            for j in marked:
                sb[row][j] = 2
    elif motif == 1:
        # One 3/0 isolated column on each row, then two split doubles.
        for row, j in enumerate(order[:2]):
            sb[row][j], sb[1 - row][j] = 3, 0
        sb[0][order[2]] = 2
        sb[1][order[3]] = 2
    elif motif == 2:
        # Four isolated trinomials, two per row: row sums 9/9.
        for t, j in enumerate(order[:4]):
            row = t % 2
            sb[row][j], sb[1 - row][j] = 3, 0
    elif motif == 3:
        # Both rows reinforce one shared column; A sums only to 8.
        sb[0][order[0]] = sb[1][order[0]] = 2
        sb[0][order[1]] = sb[1][order[2]] = 2
    else:
        # A 3/1 column and two other B-row doubles.
        sb[0][order[0]] = 3
        sb[1][order[1]] = sb[1][order[2]] = 2

    sa = [ac]
    assert _grid_weights(sa, sb) <= 12
    assert _no_dead_lines(sa, sb)
    return sa, sb


def NEW_CTX(rng, G, idx):
    """Search one of twenty indexed support-and-voltage design classes."""
    motif = idx % 5
    protocol = (idx // 5) % 4
    sa, sb = _support_motif(rng, motif)
    shortlist = []

    for _ in range(13 + idx % 5):
        A = [[_draw_cell(rng, G, count, protocol != 0)
              for count in sa[0]]]
        B = [[_draw_cell(rng, G, count, protocol >= 2)
              for count in row] for row in sb]

        if protocol >= 2:
            # Distinct reinforced directions within each B row avoid
            # repeatedly presenting the same short component cycles.
            for row in B:
                doubled = [cell for cell in row if len(cell) == 2]
                directions = [_direction(cell[0], cell[1])
                              for cell in doubled]
                if len(directions) != len(set(directions)):
                    continue

        hx, hz, n = lifted_matrices(G, A, B)
        ad = _component_defects(_component_columns(
            hx, G.q, B_SHAPE[1], A_SHAPE[1], A_SHAPE[0]))
        bd = _component_defects(_component_columns(
            hz, G.q, 1, B_SHAPE[1], B_SHAPE[0]))
        key = (ad[0] + bd[0], ad[1] + bd[1],
               max(ad[2], bd[2]), ad[2] + bd[2],
               max(ad[3], bd[3]), ad[3] + bd[3],
               rng.random())
        shortlist.append((key, A, B, hx, hz, n))

    shortlist.sort(key=lambda item: item[0])
    best = None
    for key, A, B, hx, hz, n in shortlist[:5]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # For n=575, k=250..273 remains in cell 0.450.
        quality = (not 250 <= k <= 273, key, abs(k - 250))
        if k > 0 and (best is None or quality < best[0]):
            best = (quality, A, B)
    if best is None:
        raise ValueError("no positive-k screened filling")
    return {"A": best[1], "B": best[2]}


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
# Parity-constrained support screening on the order-42 (1,3)x(1,4) bed
# eliminates limiting failures: refuted by fitness 3.6049, weight-four
# Z witnesses, and 125 X failures in 500 shots in measured slot 2.
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
