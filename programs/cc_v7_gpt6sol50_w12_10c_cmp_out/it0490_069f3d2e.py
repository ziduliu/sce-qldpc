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
CONCEPT = ("On S4 x C2 of order 48 with A=(1,2), B=(2,5), "
           "independently screening components across five B support "
           "profiles and four permutation-label regimes will yield "
           "five cell-0.250 codes scoring above 39.0.")
GROUP = ("Jump from cyclic C37 to nonabelian S4 x C2. The run has "
         "measured order-48 groups but lists no order-48 group with "
         "centre size 2 and 10 conjugacy classes, the invariants of "
         "S4 x C2. Order 48 permits n=576 on these shapes.")
SHAPES = ("A=(1,2), B=(2,5) give n=12*48=576 and guaranteed "
          "k>=144, rate>=0.250. Their unequal component lengths are "
          "96 and 240; equal shapes would give equal lengths. The "
          "parent's lightest X logicals spanned sector-1 Acol/Bcol "
          "blocks across B columns, motivating screens on both sides.")
FILLING = ("Use four distinct terms per A cell and five indexed B "
           "profiles with eight terms per B row and at most four per "
           "B column: HX<=8+4=12 and HZ<=8+4=12. Vary central "
           "balance and S4 permutation types, screen short relations "
           "in verified component columns, and select exact ranks "
           "that keep measured rates in cell 0.250.")
FAMILY_NAME = "r025_s4xc2_12x25_component_profiles_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    # (a,b) represents permutation number a in S4 and b in C2.
    from itertools import permutations
    ps = list(permutations(range(4)))
    number = {p: i for i, p in enumerate(ps)}
    els = [(a, b) for b in range(2) for a in range(24)]

    def red(g):
        return (g[0] % 24, g[1] % 2)

    def idx(g):
        a, b = red(g)
        return a + 24 * b

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        p, t = ps[a], ps[c]
        return (number[tuple(p[t[i]] for i in range(4))], b ^ d)

    def inv(g):
        a, b = red(g)
        p = ps[a]
        return (number[tuple(p.index(i) for i in range(4))], b)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _counts(rng, idx):
    """Five distinct B support motifs, each with row sums eight."""
    profiles = (
        ((2, 2, 2, 1, 1), (2, 1, 1, 2, 2)),
        ((3, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((2, 2, 2, 1, 1), (2, 2, 1, 2, 1)),
        ((2, 2, 2, 1, 1), (2, 2, 2, 1, 1)),
        ((3, 1, 1, 2, 1), (1, 3, 2, 1, 1)),
    )
    order = rng.sample(range(5), 5)
    B = [[row[j] for j in order] for row in profiles[idx % 5]]
    if rng.randrange(2):
        B.reverse()
    A = [[4, 4]]
    assert _grid_weights(A, B) == 12
    assert _no_dead_lines(A, B)
    return A, B


def _labels(rng, G, counts, regime):
    """Sample distinct elements from indexed S4 permutation-type mixes."""
    from itertools import permutations
    ps = list(permutations(range(4)))
    odd = [a for a, p in enumerate(ps)
           if sum(p[i] > p[j] for i in range(4)
                  for j in range(i + 1, 4)) % 2]
    even = [a for a in range(24) if a not in odd]
    types = {2: [], 3: [], 4: []}
    for a, p in enumerate(ps):
        current = (0, 1, 2, 3)
        order = 0
        while True:
            current = tuple(p[current[i]] for i in range(4))
            order += 1
            if current == (0, 1, 2, 3):
                break
        if order in types:
            types[order].append(a)
    transpositions = [a for a in types[2] if a in odd]
    double_transpositions = [a for a in types[2] if a in even]

    result = []
    for row in counts:
        labelled = []
        for count in row:
            if regime == 0:
                values = rng.sample(G.elements, count)
            else:
                split = count // 2
                if regime == 1:
                    first, second = odd, even
                elif regime == 2:
                    first, second = types[4], types[3]
                else:
                    first, second = transpositions, double_transpositions
                aa = rng.sample(first, split)
                aa += rng.sample(second, count - split)
                bs = [0] * count
                ones = min(count, (1, 2, 3, 1)[regime])
                for t in rng.sample(range(count), ones):
                    bs[t] = 1
                values = list(zip(aa, bs))
                rng.shuffle(values)
            assert len(values) == len(set(values)) == count
            labelled.append(values)
        result.append(labelled)
    return result


def _component_columns(rows, q, blocks, stride, check_blocks, nrows):
    """Transpose one or two component rows of the verified LP lift."""
    columns = [0] * (blocks * q)
    mask = (1 << q) - 1
    for i in range(nrows):
        for r in range(q):
            row = rows[i * check_blocks * q + r]
            mark = 1 << (i * q + r)
            for j in range(blocks):
                bits = (row >> (j * stride * q)) & mask
                while bits:
                    bit = bits & -bits
                    columns[j * q + bit.bit_length() - 1] |= mark
                    bits ^= bit
    return columns


def _component_merit(columns):
    """Penalize exact component relations through weight four."""
    singles = set(columns)
    bad = int(0 in singles) + len(columns) - len(singles)
    pairs = set()
    short = overlap = 0
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            short += syndrome in singles
            short += syndrome in pairs
            pairs.add(syndrome)
            shared = (x & y).bit_count()
            overlap += shared * (shared - 1) // 2
    return bad, short, overlap


def _minor_merit(rows, q, blocks, stride, check_blocks):
    """Screen cyclic component relations read from verified LP rows."""
    from itertools import combinations

    mask = (1 << q) - 1
    polynomials = [
        [(rows[i * check_blocks * q] >> (j * stride * q)) & mask
         for j in range(blocks)]
        for i in range(2)
    ]

    def rotate(bits, shift):
        if not shift:
            return bits
        return ((bits << shift) | (bits >> (q - shift))) & mask

    def product(left, right):
        value = 0
        while right:
            bit = right & -right
            value ^= rotate(left, bit.bit_length() - 1)
            right ^= bit
        return value

    words = []
    for triple in combinations(range(blocks), 3):
        parts = [0] * blocks
        for j in triple:
            u, v = [t for t in triple if t != j]
            parts[j] = (
                product(polynomials[0][u], polynomials[1][v])
                ^ product(polynomials[1][u], polynomials[0][v])
            )
        word = sum(part << (j * q) for j, part in enumerate(parts))
        if word:
            words.append(word)
    if not words:
        return (0, 0, 0, 0)

    translated = [
        [sum(rotate((word >> (j * q)) & mask, shift) << (j * q)
             for j in range(blocks))
         for shift in range(q)]
        for word in words
    ]
    minimum = blocks * q + 1
    light17 = light19 = light21 = 0

    def observe(word):
        nonlocal minimum, light17, light19, light21
        if word:
            weight = word.bit_count()
            minimum = min(minimum, weight)
            light17 += weight <= 17
            light19 += weight <= 19
            light21 += weight <= 21

    for word in words:
        observe(word)
    for i, word in enumerate(words):
        for j in range(i, len(words)):
            for other in translated[j]:
                observe(word ^ other)

    if blocks == 3:
        # Fix one translate at zero; cyclic symmetry covers the others.
        for s in range(1, q):
            for t in range(s + 1, q):
                observe(words[0] ^ translated[0][s] ^ translated[0][t])
    else:
        for i, j, k in combinations(range(len(words)), 3):
            for second in translated[j]:
                for third in translated[k]:
                    observe(words[i] ^ second ^ third)

    return minimum, -light17, -light19, -light21


def NEW_CTX(rng, G, idx):
    """Select verified components separately, then rank exact pairings."""
    sa, sb = _counts(rng, idx)
    regime = (idx // 5) % 4
    anchor_a = _labels(rng, G, sa, regime)
    anchor_b = _labels(rng, G, sb, (regime + 1) % 4)
    a_options, b_options = [], []

    for trial in range(18):
        A = _labels(rng, G, sa, regime)
        hx, _, _ = lifted_matrices(G, A, anchor_b)
        ac = _component_columns(hx, G.q, 2, 5, 5, 1)
        a_options.append((_component_merit(ac), trial, A))

        B = _labels(rng, G, sb, (regime + 1) % 4)
        _, hz, _ = lifted_matrices(G, anchor_a, B)
        bc = _component_columns(hz, G.q, 5, 1, 1, 2)
        b_options.append((_component_merit(bc), trial, B))

    a_options.sort(key=lambda entry: entry[:2])
    b_options.sort(key=lambda entry: entry[:2])
    best = None
    for am, at, A in a_options[:3]:
        for bm, bt, B in b_options[:3]:
            k = lifted_k(G, A, B)
            grade = (144 <= k <= 158,
                     -(am[0] + bm[0]),
                     -(am[1] + bm[1]),
                     k if k <= 158 else -k,
                     -max(am[2], bm[2]),
                     -at, -bt)
            if best is None or grade > best[0]:
                best = (grade, {"A": A, "B": B})
    return best[1]


def CELL(rng, G, side, i, j, ctx):
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# Rolled singleton-heavy supports do not reliably avoid light logicals:
# all five measured parent slots had d30s=5, despite distinct labels.
# REFUTED: Disjoint doubles plus screening through component weight five
# suffice to raise distance above six: all five current measured slots
# retained d30s=6, with weight-six sector-1 X witnesses.
# REFUTED: The previous gap-orbit screen yielded five distance-eight
# readings: its measured distances were 8, 8, 8, 6, 6. Its A extraction
# inspected only the first of two component-check rows.
# REFUTED: Screening only component relations through weight four with
# 18 label trials routinely removes the lightest logicals on this bed:
# the parent measured distances 16, 17, 18, 17, 17, with lightest X
# witnesses in sector-1 Acol/Bcol blocks.
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
