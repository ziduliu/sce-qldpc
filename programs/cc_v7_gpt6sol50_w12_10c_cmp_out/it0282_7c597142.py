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
CONCEPT = ("On Dic11 of order 44 with A=(1,3), B=(1,4), centrally coupled but unequal A pairs can raise k above 284 while preserving distance four.")
GROUP = ("Keep Dic11 at q=44: this n=572 bed repeatedly measured d30s=4, making its remaining rank potential a filling-only question.")
SHAPES = ("Keep (1,3)x(1,4): k>=264, guaranteed rate 6/13, and n=572. Its unequal single-check components have three and four columns. The parent's lightest words occupied sector-1 Acol/Bcol blocks, prominently Acol1 across B columns.")
FILLING = ("Use legal (4,4,2) A counts and four B binomials. Couple each A quadruple's pair bases by Dic11's central involution, but use unequal pair steps. Cross four A-step motifs with five B order spectra; screen exact component and full-lift short relations before rewarding k within the 0.500 cell.")
FAMILY_NAME = "r050_dic11_13x14_central_frustrated_pairs"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 44

def custom_ops(q):
    # Dic11: x^22=1, y^2=x^11, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(22)]

    def red(g):
        return (g[0] % 22, g[1] % 2)

    def idx(g):
        a, b = red(g)
        return 22 * b + a

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (-1 if b else 1) * c + 11 * b * d) % 22,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 22, 0) if b == 0 else ((a + 11) % 22, 1)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _dic_step(rng, kind):
    """Choose a relative element with the specified order motif."""
    if kind == "r22":
        return (rng.choice((1, 3, 5, 7, 9, 13, 15, 17, 19, 21)), 0)
    if kind == "r11":
        return (rng.choice((2, 4, 6, 8, 10, 12, 14, 16, 18, 20)), 0)
    return (rng.randrange(22), 1)  # A non-rotation has order four.


def _dic_cell(rng, G, count, kind, paired):
    """Construct one full cell without duplicate group elements."""
    if count == 2:
        base = G.pick(rng)
        return [base, G.mul(base, _dic_step(rng, kind))]
    if count == 4 and paired:
        for _ in range(40):
            first, second = G.pick(rng), G.pick(rng)
            cell = [first, G.mul(first, _dic_step(rng, kind)),
                    second, G.mul(second, _dic_step(rng, kind))]
            if len(set(cell)) == 4:
                return cell

    if count == 3:
        cosets = (0, 0, 1) if rng.randrange(2) else (0, 1, 1)
    else:
        cosets = (0, 0, 1, 1)
    cell = []
    for b in cosets:
        a = rng.randrange(22)
        while (a, b) in cell:
            a = rng.randrange(22)
        cell.append((a, b))
    return cell


def _component_relations(G, rows, side):
    """Count relations of weights 2, 3 and 4 in an exact lifted component."""
    q = G.q
    cb = B_SHAPE[1]
    blocks = A_SHAPE[1] if side == "A" else cb
    signatures = [0] * (blocks * q)
    mask = (1 << q) - 1
    for r, row in enumerate(rows[:q]):
        for j in range(blocks):
            base = j * cb * q if side == "A" else j * q
            bits = (row >> base) & mask
            while bits:
                bit = bits & -bits
                signatures[j * q + bit.bit_length() - 1] |= 1 << r
                bits ^= bit

    singles = set(signatures)
    seen = set()
    bad2 = bad3 = bad4 = 0
    for i, first in enumerate(signatures):
        for second in signatures[i + 1:]:
            syndrome = first ^ second
            if not syndrome:
                bad2 += 1
            elif syndrome in singles:
                bad3 += 1
            elif syndrome in seen:
                bad4 += 1
            else:
                seen.add(syndrome)
    return bad2, bad3, bad4


def _central_quad(rng, G, kind, offset):
    """Couple pair bases by central x^11, without making equal pairs."""
    for _ in range(80):
        first = G.pick(rng)
        step = _dic_step(rng, kind)
        other_step = G.mul(step, (offset, 0))
        second = G.mul(first, (11, 0))
        cell = [first, G.mul(first, step),
                second, G.mul(second, other_step)]
        if len(set(cell)) == 4:
            return cell
    raise ValueError("could not draw a distinct central quadruple")


def _order_binomials(rng, G, motif):
    """Five different multisets of relative orders for B's four cells."""
    orders = (
        ("ord4", "ord4", "ord4", "r22"),
        ("ord4", "ord4", "r22", "r11"),
        ("ord4", "r22", "r22", "r11"),
        ("ord4", "ord4", "r11", "r11"),
        ("ord4", "r22", "r11", "central"),
    )
    kinds = list(orders[motif])
    rng.shuffle(kinds)
    cells = []
    for kind in kinds:
        base = G.pick(rng)
        step = (11, 0) if kind == "central" else _dic_step(rng, kind)
        cells.append([base, G.mul(base, step)])
    return [cells]


def _full_short_grade(rows, n):
    """Detect zero columns, duplicate columns, and three-column words."""
    columns = [0] * n
    for r, row in enumerate(rows):
        mark = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= mark
            row ^= bit
    singles = set(columns)
    bad2 = int(0 in singles) + n - len(singles)
    if bad2:
        return bad2, 0
    bad3 = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            bad3 += (x ^ y) in singles
    return 0, bad3


def NEW_CTX(rng, G, idx):
    """Screen independent fillings of twenty central-pair/order motifs."""
    design = idx % 20
    a_motif, b_motif = divmod(design, 5)
    a_kind = ("ord4", "ord4", "r22", "r22")[a_motif]
    offset = (2, 6, 4, 8)[a_motif]
    shortlist = []

    for trial in range(48):
        counts = [4, 4, 2]
        rng.shuffle(counts)
        A = [[]]
        for count in counts:
            if count == 4:
                A[0].append(_central_quad(rng, G, a_kind, offset))
            else:
                base = G.pick(rng)
                step = _dic_step(rng, "r22" if a_motif % 2 else "ord4")
                A[0].append([base, G.mul(base, step)])
        B = _order_binomials(rng, G, b_motif)
        hx, hz, n = lifted_matrices(G, A, B)
        sa = _component_relations(G, hx, "A")
        sb = _component_relations(G, hz, "B")
        bad2, bad3 = sa[0] + sb[0], sa[1] + sb[1]
        if bad2 or bad3:
            quality = (-bad2, -bad3, 0, -10**6, -sa[2]-sb[2])
        else:
            k = n - gf2_rank(hx) - gf2_rank(hz)
            in_cell = 272 <= k <= 300
            quality = (0, 0, int(in_cell),
                       k if in_cell else -abs(k - 296),
                       -sa[2]-sb[2])
        shortlist.append((quality, trial, A, B, hx, hz, n))

    shortlist.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    best = None
    for quality, trial, A, B, hx, hz, n in shortlist[:5]:
        x2, x3 = _full_short_grade(hx, n)
        z2, z3 = _full_short_grade(hz, n)
        grade = (-(x2 + z2), -(x3 + z3), quality, -trial)
        if best is None or grade > best[0]:
            best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Balanced order-34 dihedral (1,3)x(1,5) fillings did not
# reliably remove short logicals: four of five measured slots had d30s=2.
# REFUTED: Order-22 two-term differences alone remove weight-four
# logicals on Dic11 (1,3)x(1,4): all five selected parent slots had d30s=4.
# REFUTED: Screening the parent's profiles keeps its selected codes in
# the 0.450 cell: their mean k/n was 0.486, in the 0.500 cell.
# REFUTED: Independently drawn mixed-order pairs lift this bed above
# distance four: all five current measured slots retained d30s=4,
# with weight-four X witnesses across sector-1 Acol/Bcol blocks.
# REFUTED: Shared/fractured A pairs with distinct B gap spectra raise
# this bed above distance four: all five measured slots had d30s=4,
# and their highest reported k was 284.
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
