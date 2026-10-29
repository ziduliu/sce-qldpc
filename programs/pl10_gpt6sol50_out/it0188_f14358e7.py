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
CONCEPT = ("In Z5 semidirect Z4 with action x -> x^2, order 20, A=(2,5) and B=(2,5), equal-length components and screened mixed-order binomial gaps should reduce worst-basis circuit failures.")
GROUP = ("Keep the parent's faithful order-20 semidirect group. Its related (2,5)x(2,5) measured bed has reached about 5.2 fitness, compared with this parent's 4.3344.")
SHAPES = ("Equal (2,5)x(2,5) sides give n=580, guaranteed k>=180 and rate>=0.310 in cell 0.300, with two length-100 component codes and 200 check rows per basis. The parent's limiting Z failures were spread across sector-1 Acol/Bcol blocks, including Acol1/Bcol1 and Acol3/Bcol3 in slot 11, without a persistent check-block hotspot.")
FILLING = ("Give every component column a doubled cell, vary order-four cross-coset and order-five within-coset gaps across indexed support designs, and screen independent labels for short component relations and verified target-cell k. Both check-row sums stay <=12. Test for fewer Z failed shots and decoder fallthroughs than the parent's 100-failure caps, not for proxy or excess k alone.")
FAMILY_NAME = "r030_z5_by_z4_s2_25x25_mixedgap_balanced_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 20

def custom_ops(q):
    # Z_5 semidirect Z_4; multiplication by 2 has order four mod 5.
    m, r, s = 5, 4, 2
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return b * m + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb); n = 29 * 20 = 580


def _component_columns(rows, bases, q):
    """Transpose selected columns of verified component-check rows."""
    cols = [0] * (len(bases) * q)
    block_mask = (1 << q) - 1
    for check, row in enumerate(rows):
        for j, base in enumerate(bases):
            bits = (row >> base) & block_mask
            while bits:
                bit = bits & -bits
                cols[j * q + bit.bit_length() - 1] |= 1 << check
                bits ^= bit
    return cols


def _short_defects(cols):
    """Count zero pairs, triples and repeated pair syndromes."""
    pairs = {}
    twos = cycles = 0
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            syndrome = left ^ right
            twos += syndrome == 0
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
            shared = left & right
            weight = shared.bit_count()
            cycles += weight * (weight - 1) // 2
    triples = sum(pairs.get(col, 0) for col in cols)
    fours = sum(v * (v - 1) // 2 for v in pairs.values())
    return twos, triples, fours, cycles


def _spread_doubles(rng, columns, row_totals):
    """Use different columns in the second row before reusing a column."""
    counts = [[1] * columns for _ in row_totals]
    first = rng.sample(range(columns), row_totals[0] - columns)
    for j in first:
        counts[0][j] = 2
    second_n = row_totals[1] - columns
    available = [j for j in range(columns) if j not in first]
    rng.shuffle(available)
    overlap = list(first)
    rng.shuffle(overlap)
    for j in (available + overlap)[:second_n]:
        counts[1][j] = 2
    return counts


def _gap_cell(rng, G, count, mode, side, bank):
    """Use an order-four or order-five gap; neither is the identity."""
    first = G.pick(rng)
    if count == 1:
        return [first]
    within_coset = ((mode == 1 and side == "B")
                    or (mode == 2 and side == "A")
                    or (mode == 3 and rng.randrange(4) < 1 + bank))
    if within_coset:
        step = (rng.randrange(1, 5), 0)  # order five
    else:
        step = (rng.randrange(5), rng.choice((1, 3)))  # order four
    return [first, G.mul(first, step)]


def _higher_relations(cols, cap=24):
    """Bound witnessed disjoint weight-five and weight-six relations."""
    pairs = {}
    for i, left in enumerate(cols):
        for j in range(i + 1, len(cols)):
            pairs.setdefault(left ^ cols[j], []).append((i, j))

    triples = {}
    fives = sixes = 0
    for i in range(len(cols) - 2):
        for j in range(i + 1, len(cols) - 1):
            partial = cols[i] ^ cols[j]
            for k in range(j + 1, len(cols)):
                value = partial ^ cols[k]
                if any(i not in pair and j not in pair and k not in pair
                       for pair in pairs.get(value, ())):
                    fives += 1
                old = triples.get(value)
                if old is None:
                    triples[value] = (i, j, k)
                elif i not in old and j not in old and k not in old:
                    sixes += 1
                if fives >= cap and sixes >= cap:
                    return cap, cap
    return min(fives, cap), min(sixes, cap)


def NEW_CTX(rng, G, idx):
    """Search an indexed support-and-gap design on verified component lifts."""
    profiles = (
        ((8, 7), (8, 7)),
        ((7, 8), (8, 8)),
        ((8, 8), (7, 8)),
        ((8, 7), (8, 8)),
        ((8, 8), (8, 8)),
    )
    design = idx % 20
    mode = design // 5
    bank = (idx // 20) % 3
    a_totals, b_totals = profiles[design % 5]
    q = G.q
    shortlist = []

    for _ in range(16 + idx % 5):
        sa = _spread_doubles(rng, 5, a_totals)
        sb = _spread_doubles(rng, 5, b_totals)
        assert _grid_weights(sa, sb) <= W_MAX
        assert _no_dead_lines(sa, sb)
        A = [[_gap_cell(rng, G, count, mode, "A", bank)
              for count in row] for row in sa]
        B = [[_gap_cell(rng, G, count, mode, "B", bank)
              for count in row] for row in sb]

        hx, hz, n = lifted_matrices(G, A, B)
        # HX has five B-column blocks per A row; HZ's first two
        # check blocks contain the complete right-lifted B component.
        a_rows = hx[:q] + hx[5 * q:6 * q]
        a_cols = _component_columns(
            a_rows, [j * 5 * q for j in range(5)], q)
        b_cols = _component_columns(
            hz[:2 * q], [j * q for j in range(5)], q)
        a2, a3, a4, ac = _short_defects(a_cols)
        b2, b3, b4, bc = _short_defects(b_cols)
        defects = (max(a2, b2), a2 + b2,
                   max(a3, b3), a3 + b3,
                   max(a4, b4), a4 + b4,
                   max(ac, bc), ac + bc)
        shortlist.append((defects, rng.random(), A, B,
                          a_cols, b_cols, hx, hz, n))
    shortlist.sort(key=lambda item: item[:2])

    best_key = best_grids = None
    for defects, _, A, B, a_cols, b_cols, hx, hz, n in shortlist[:4]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k <= 0:
            continue
        ah = _higher_relations(a_cols)
        bh = _higher_relations(b_cols)
        key = (not 180 <= k <= 188, defects[:4],
               max(ah[0], bh[0]), ah[0] + bh[0],
               max(ah[1], bh[1]), ah[1] + bh[1],
               defects[4:], k - 180)
        if best_key is None or key < best_key:
            best_key, best_grids = key, {"A": A, "B": B}

    if best_grids is None:
        raise ValueError("no positive-k filling")
    return best_grids


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
# Parity-constrained screening eliminates frequent failures in the order-42
# bed: refuted by its 3.6049 fitness; its measured worse bases still had
# 108-161 failures within 500 shots in four of the five measured slots.
# Same-Z5-coset A binomials and mostly monomial B cells remove the Z-basis weakness: refuted by 109-138 Z failures within 750-1000 shots in all five measured order-20 slots.
# REFUTED: Cross-coset gaps and short-relation screening on unequal
# (2,4)x(2,6) components balance the bases: all five measured slots
# were limited by Z, with 100-failure caps reached in 1500-9500 shots.
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
