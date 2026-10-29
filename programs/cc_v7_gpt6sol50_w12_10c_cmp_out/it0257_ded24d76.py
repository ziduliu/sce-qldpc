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
CONCEPT = ("On cyclic C37 of order 37 with A=(2,3), B=(2,4), "
           "varying A-column occupancies and screening verified "
           "component-kernel words may lift the parent's weight-16 "
           "through weight-18 sector-1 logicals.")
GROUP = ("Keep cyclic C37 and order 37 fixed. The measured (2,3)x(2,4) "
         "bed has reached fitness 41.0; a filling-only change tests "
         "whether that benchmark can be improved.")
SHAPES = ("Keep (2,3)x(2,4): n=16*37=592 and guaranteed k=74, "
          "rate 0.125 at the lower edge of cell 0.150. Its unequal "
          "two-row components have 111 and 148 variables. All five "
          "measured lightest parent logicals used sector-1 Acol/Bcol "
          "blocks.")
FILLING = ("Cross five A occupancy profiles with four compatible B "
           "profiles each. Every profile has nonzero rows and columns "
           "and verified HX/HZ row weights at most 12. Independently "
           "label each design; screen exact-lift component minors and "
           "kernel words, then check k against the 74 bound. The test "
           "targets the parent's weight-16 through weight-18 witnesses.")
FAMILY_NAME = "r015_c37_23x24_occupancy_kernel_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Cyclic C37, represented by singleton exponent tuples.
    els = [(a,) for a in range(37)]

    def red(g):
        return (g[0] % 37,)

    def idx(g):
        return g[0] % 37

    def mul(g, h):
        return ((g[0] + h[0]) % 37,)

    def inv(g):
        return ((-g[0]) % 37,)

    return dict(elements=els, ident=(0,), mul=mul,
                inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _draw_design(rng, idx):
    """One fresh labeling of an indexed, weight-legal occupancy design."""
    a_profile, b_profile = divmod(idx % 20, 4)
    a_counts = (
        ((3, 2, 2), (1, 2, 2)),  # row weights 7,5; columns at most 4
        ((3, 1, 2), (1, 3, 2)),  # row weights 6,6
        ((3, 3, 2), (1, 1, 2)),  # row weights 8,4
        ((3, 2, 2), (2, 3, 2)),  # row weights 7,7; columns at most 5
        ((3, 2, 2), (2, 2, 3)),  # different tripleton incidence
    )[a_profile]
    balanced_b = (
        ((2, 2, 2, 2), (2, 2, 2, 2)),
        ((3, 1, 3, 1), (1, 3, 1, 3)),
        ((1, 2, 3, 2), (3, 2, 1, 2)),
        ((3, 2, 2, 1), (2, 2, 2, 2)),
    )
    seven_term_b = (
        ((2, 2, 2, 1), (2, 2, 1, 2)),
        ((3, 2, 1, 1), (1, 2, 3, 1)),
        ((3, 2, 1, 1), (2, 1, 2, 2)),
        ((3, 1, 2, 1), (1, 3, 1, 2)),
    )
    if a_profile >= 3:
        b_counts = seven_term_b[b_profile]
    elif a_profile == 2 and b_profile == 3:
        # A's weight-eight row requires B columns of at most four.
        b_counts = ((4, 2, 1, 1), (0, 2, 3, 3))
    else:
        b_counts = balanced_b[b_profile]

    assert _grid_weights(a_counts, b_counts) <= 12
    assert _no_dead_lines(a_counts, b_counts)

    def label(counts, side):
        grid = []
        for i, counts_row in enumerate(counts):
            row = []
            for j, count in enumerate(counts_row):
                if count == 2:
                    # Use different gap sets across the twenty designs;
                    # starts and gaps remain independently drawn.
                    gaps = ((1, 3, 5, 7, 9, 11, 13, 15, 17)
                            if (idx + i + j + (side == "B")) % 3 == 0
                            else range(1, 19))
                    start = rng.randrange(37)
                    gap = rng.choice(tuple(gaps))
                    row.append([(start,), ((start + gap) % 37,)])
                else:
                    row.append([(a,) for a in rng.sample(range(37), count)])
            grid.append(row)
        return grid

    return label(a_counts, "A"), label(b_counts, "B")


def _minor_grade(hx, hz, q):
    """Find short translated two-row component-kernel minor words.

    Row supports come from the verified lifts, including their distinct
    A-left and B-right conventions. The products below are cyclic
    convolutions; each three-column minor is a kernel word.
    """
    from itertools import combinations

    mask = (1 << q) - 1

    def rotate(x, shift):
        if not shift:
            return x
        return ((x << shift) | (x >> (q - shift))) & mask

    def product(x, y):
        result = 0
        while y:
            bit = y & -y
            result ^= rotate(x, bit.bit_length() - 1)
            y ^= bit
        return result

    def one_side(rows, blocks, stride):
        polynomials = [
            [(rows[r] >> (j * stride * q)) & mask
             for j in range(blocks)]
            for r in range(2)
        ]
        words = []
        for triple in combinations(range(blocks), 3):
            parts = [0] * blocks
            for j in triple:
                u, v = [t for t in triple if t != j]
                parts[j] = (product(polynomials[0][u], polynomials[1][v])
                            ^ product(polynomials[1][u], polynomials[0][v]))
            word = sum(part << (q * j) for j, part in enumerate(parts))
            if word:
                words.append(word)
        if not words:
            return (0, 0, 0)

        minimum = min(w.bit_count() for w in words)
        light = 0
        total = sum(w.bit_count() for w in words)
        for i, left in enumerate(words):
            for j in range(i, len(words)):
                right = words[j]
                for shift in range(q):
                    translated = 0
                    for block in range(blocks):
                        part = (right >> (block * q)) & mask
                        translated |= rotate(part, shift) << (block * q)
                    combined = left ^ translated
                    if combined:
                        weight = combined.bit_count()
                        minimum = min(minimum, weight)
                        light += weight <= 16
        return (minimum, -light, total)

    a = one_side((hx[0], hx[4 * q]), 3, 4)
    b = one_side((hz[0], hz[q]), 4, 1)
    return (min(a[0], b[0]), a[0] + b[0],
            a[1] + b[1], a[2] + b[2])


def _component_kernel_grade(hx, hz, q, rng):
    """Probe kernels of both complete components in the verified lifts."""
    mask = (1 << q) - 1

    def columns(side):
        blocks = 3 if side == "A" else 4
        stride = 4 if side == "A" else 1
        cols = [0] * (blocks * q)
        for check in range(2 * q):
            row = (hx[(check // q) * 4 * q + check % q]
                   if side == "A" else hz[check])
            mark = 1 << check
            for j in range(blocks):
                bits = (row >> (j * stride * q)) & mask
                while bits:
                    bit = bits & -bits
                    cols[j * q + bit.bit_length() - 1] |= mark
                    bits ^= bit
        return cols

    def probe(cols):
        found = set()
        for _ in range(4):
            order = list(range(len(cols)))
            rng.shuffle(order)
            pivots = {}
            for j in order:
                syndrome, word = cols[j], 1 << j
                while syndrome:
                    p = syndrome.bit_length() - 1
                    entry = pivots.get(p)
                    if entry is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= entry[0]
                    word ^= entry[1]
                else:
                    found.add(word)

        short = sorted(found, key=int.bit_count)[:9]
        if not short:
            return (len(cols) + 1, 0)
        minimum = min(word.bit_count() for word in short)
        near = 0
        blocks = len(cols) // q
        for i, left in enumerate(short):
            for right in short[i:]:
                for shift in range(q):
                    translated = 0
                    for block in range(blocks):
                        part = (right >> (block * q)) & mask
                        if shift:
                            part = ((part << shift) |
                                    (part >> (q - shift))) & mask
                        translated |= part << (block * q)
                    combined = left ^ translated
                    if combined:
                        weight = combined.bit_count()
                        minimum = min(minimum, weight)
                        near += weight <= 12
        return minimum, -near

    return probe(columns("A")), probe(columns("B"))


def NEW_CTX(rng, G, idx):
    """Select fresh labels using verified minors, kernels and exact rank."""
    shortlisted = []
    for trial in range(20):
        A, B = _draw_design(rng, idx)
        hx, hz, n = lifted_matrices(G, A, B)
        minor = _minor_grade(hx, hz, G.q)
        ac, bc = _component_kernel_grade(hx, hz, G.q, rng)
        key = (min(ac[0], bc[0]), ac[0] + bc[0],
               minor[0], minor[1], ac[1] + bc[1], minor[2])
        shortlisted.append((key, trial, A, B, hx, hz, n))
    shortlisted.sort(key=lambda item: item[0], reverse=True)

    best = None
    for key, trial, A, B, hx, hz, n in shortlisted[:7]:
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # k=74 gives exactly the cell's lower rate boundary.
        eligible = 74 <= k <= 103
        grade = (eligible, key[0], key[1], key[2], key[3],
                 k * key[0] * key[0], key[4], key[5], -trial)
        if best is None or grade > best[0]:
            best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell without redrawing."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled support variation on the parent's (1,3)x(1,3) bed
# suffices to remove weight-4 logicals: all five measured slots had d30s=4.
# REFUTED: The previous (1,4)x(1,4) penalty screen suffices to remove
# short logicals: its five measured slots all retained d30s=3 witnesses.
# REFUTED: Triangle-free binomial gaps on the parent's order-34
# (1,4)x(1,4) bed eliminate weight-four logicals: all five measured
# slots retained d30s=4, with thousands of verified weight-four words.
# REFUTED: Noncommuting order-six A binomials and screened B columns on
# S3 x C3 (1,4)x(2,7) suffice for d30s>=5: all five measured slots
# retained d30s=4, with k at its rank-count bound.
# REFUTED: Binomial-only A cells with four-term B columns reliably lift
# this C37 bed beyond weight 17: the five measured distances were
# 16, 16, 16, 17 and 18, all with sector-1 lightest witnesses.
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
