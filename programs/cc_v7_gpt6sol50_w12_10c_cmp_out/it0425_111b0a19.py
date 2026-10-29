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
           "screening verified divided component minors and their "
           "translated sums will select more distance-18-or-better "
           "codes in rate cell 0.150.")
GROUP = ("Keep C37 at order 37: this exact bed has produced measured "
         "distance-18 codes. Holding the group fixed tests a deeper "
         "witness screen as a filling-only change.")
SHAPES = ("Keep (2,3)x(2,4): q=37 gives n=592 and guaranteed k>=74, "
          "rate>=0.125. Its unequal two-row components have 111 and "
          "148 variables; equal shapes would give equal lengths. "
          "The parent's lightest X witnesses occupied sector-1 "
          "Acol/Bcol blocks across all four B columns.")
FILLING = ("Use balanced four-term component columns, emphasizing "
           "complementary 1/3-term B cells across distinct motifs. "
           "A rows have at most seven terms and B rows eight, so "
           "both check weights are at most 12. Screen divided minors "
           "and three-relation sums only when they verify in the "
           "lifted component; give B more label trials, then check "
           "exact k. Target weight-16/17 X witnesses while retaining "
           "useful excess over the k=74 bound.")
FAMILY_NAME = "r015_c37_23x24_verified_divided_triples_v3"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 37

def custom_ops(q):
    # Cyclic C37, encoded as two-coordinate tuples.
    els = [(a, 0) for a in range(q)]

    def red(g):
        return (g[0] % q, 0)

    def idx(g):
        return g[0] % q

    def mul(g, h):
        return ((g[0] + h[0]) % q, 0)

    def inv(g):
        return ((-g[0]) % q, 0)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 4)               # (rb, cb)


def _counts(rng, idx):
    """Five row-balanced B motifs, with occasional mixed-A controls."""
    A = [[2] * 3 for _ in range(2)]
    B = [[2] * 4 for _ in range(2)]
    order = rng.sample(range(4), 4)
    motif = idx % 5
    if motif in (0, 1):
        used = 2 if motif == 0 else 4
        for t, j in enumerate(order[:used]):
            B[0][j], B[1][j] = ((1, 3) if t % 2 == 0
                                 else (3, 1))
    elif motif in (2, 3):
        B[0][order[0]], B[1][order[0]] = 0, 4
        B[0][order[1]], B[1][order[1]] = 4, 0
        if motif == 2:
            B[0][order[2]], B[1][order[2]] = 1, 3
            B[0][order[3]], B[1][order[3]] = 3, 1
    # Motif four is the all-binomial B control.
    if idx % 10 == 9:
        j = rng.randrange(3)
        A[0][j], A[1][j] = 3, 1
    assert _grid_weights(A, B) <= 12
    assert _no_dead_lines(A, B)
    return A, B


def _labels(rng, counts, regime):
    """Separate unsigned gaps across a side, without fixing its orbit."""
    pools = (tuple(range(1, 19)),
             tuple(range(1, 18, 2)),
             tuple(range(2, 19, 2)),
             tuple(range(4, 19)),
             tuple(range(1, 16)))
    result, used = [], set()
    for row in counts:
        labelled = []
        for count in row:
            if count == 2:
                choices = [d for d in pools[regime] if d not in used]
                gap = rng.choice(choices or pools[regime])
                used.add(gap)
                start = rng.randrange(37)
                direction = gap if rng.randrange(2) else -gap
                shifts = [start, (start + direction) % 37]
            elif count >= 3:
                shifts, best_penalty = None, 100
                for _ in range(32):
                    candidate = rng.sample(range(37), count)
                    gaps = [
                        min((x - y) % 37, (y - x) % 37)
                        for t, x in enumerate(candidate)
                        for y in candidate[:t]
                    ]
                    penalty = len(gaps) - len(set(gaps))
                    penalty += len(set(gaps) & used)
                    if penalty < best_penalty:
                        shifts, best_penalty = candidate, penalty
                    if penalty == 0:
                        break
                used.update(
                    min((x - y) % 37, (y - x) % 37)
                    for t, x in enumerate(shifts) for y in shifts[:t]
                )
            else:
                shifts = rng.sample(range(37), count)
            labelled.append([(a, 0) for a in shifts])
        result.append(labelled)
    return result


def _component_columns(rows, q, blocks, stride, check_blocks):
    """Read both component-check rows from the verified LP matrices."""
    columns = [0] * (blocks * q)
    mask = (1 << q) - 1
    for i in range(2):
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


def _minor_grade(rows, q, blocks, stride, check_blocks, deep=False):
    """Grade only kernel words verified against the exact lifted columns."""
    from itertools import combinations

    mask = (1 << q) - 1
    polys = [
        [(rows[i * check_blocks * q] >> (j * stride * q)) & mask
         for j in range(blocks)]
        for i in range(2)
    ]
    columns = _component_columns(rows, q, blocks, stride, check_blocks)

    def rotate(x, shift):
        shift %= q
        return ((x << shift) | (x >> (q - shift))) & mask

    def product(x, y):
        output = 0
        while y:
            low = y & -y
            output ^= rotate(x, low.bit_length() - 1)
            y ^= low
        return output

    def verified(word):
        syndrome = 0
        bits = word
        while bits:
            low = bits & -bits
            syndrome ^= columns[low.bit_length() - 1]
            bits ^= low
        return syndrome == 0

    def divide_even(x):
        if x.bit_count() & 1:
            return None
        quotient = previous = 0
        for position in range(1, q):
            previous ^= (x >> position) & 1
            quotient |= previous << position
        return quotient if previous == (x & 1) else None

    seeds = set()
    for triple in combinations(range(blocks), 3):
        parts = [0] * blocks
        for j in triple:
            u, v = (other for other in triple if other != j)
            parts[j] = (product(polys[0][u], polys[1][v]) ^
                        product(polys[1][u], polys[0][v]))
        for _ in range(3):
            word = sum(part << (j * q) for j, part in enumerate(parts))
            if word and verified(word):
                seeds.add(word)
            divided = [divide_even(part) for part in parts]
            if any(part is None for part in divided):
                break
            parts = divided
    if not seeds:
        return (0, -100000, -100000, -100000, -100000)

    # Limit the deeper search to the lightest *different* verified seeds.
    seeds = sorted(seeds, key=lambda word: (word.bit_count(), word))[:8]
    translated = [
        [sum(rotate((word >> (j * q)) & mask, shift) << (j * q)
             for j in range(blocks))
         for shift in range(q)]
        for word in seeds
    ]
    minimum = blocks * q + 1
    light16 = light17 = light18 = light20 = 0

    def observe(word):
        nonlocal minimum, light16, light17, light18, light20
        if word:
            weight = word.bit_count()
            minimum = min(minimum, weight)
            light16 += weight <= 16
            light17 += weight <= 17
            light18 += weight <= 18
            light20 += weight <= 20

    for word in seeds:
        observe(word)
    for i, left in enumerate(seeds):
        for j in range(i, len(seeds)):
            for shift in range(q):
                observe(left ^ translated[j][shift])

    if deep:
        for i, j, k in combinations(range(len(seeds)), 3):
            for middle in translated[j]:
                pair = seeds[i] ^ middle
                for right in translated[k]:
                    observe(pair ^ right)
        for i, base in enumerate(seeds):
            for s in range(1, q):
                pair = base ^ translated[i][s]
                for t in range(s + 1, q):
                    observe(pair ^ translated[i][t])
    return minimum, -light16, -light17, -light18, -light20


def NEW_CTX(rng, G, idx):
    """Search B more deeply, since the parent's lightest X words use B."""
    sa, sb = _counts(rng, idx)
    regime = (idx // 5) % 5
    anchor_a = _labels(rng, sa, regime)
    anchor_b = _labels(rng, sb, (regime + 2) % 5)
    a_options, b_options = [], []

    for trial in range(28):
        A = _labels(rng, sa, regime)
        hx, _, _ = lifted_matrices(G, A, anchor_b)
        merit = _minor_grade(hx, G.q, 3, 4, 4)
        a_options.append((merit, -trial, A))
    for trial in range(84):
        B = _labels(rng, sb, (regime + 2) % 5)
        _, hz, _ = lifted_matrices(G, anchor_a, B)
        merit = _minor_grade(hz, G.q, 4, 1, 1)
        b_options.append((merit, -trial, B))

    a_options.sort(key=lambda item: item[:2], reverse=True)
    b_options.sort(key=lambda item: item[:2], reverse=True)

    def finalists(options, side, count):
        graded = []
        for _, tie, grid in options[:count]:
            A, B = ((grid, anchor_b) if side == "A"
                    else (anchor_a, grid))
            hx, hz, _ = lifted_matrices(G, A, B)
            rows = hx if side == "A" else hz
            merit = (_minor_grade(rows, G.q, 3, 4, 4, True)
                     if side == "A" else
                     _minor_grade(rows, G.q, 4, 1, 1, True))
            graded.append((merit, tie, grid))
        graded.sort(key=lambda item: item[:2], reverse=True)
        return graded[:3]

    best = None
    for am, at, A in finalists(a_options, "A", 6):
        for bm, bt, B in finalists(b_options, "B", 10):
            k = lifted_k(G, A, B)
            predicted = min(am[0], bm[0], 21)
            grade = (74 <= k <= 103, k * predicted * predicted,
                     predicted, bm[1:], am[1:], at, bt)
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
# REFUTED: Balanced B support profiles screened only for component
# relations through weight four suffice to clear distance 17 on C37:
# the current five measured distances were 16, 17, 17, 17 and 18.
# REFUTED: Twenty-eight trials of undivided minor-pair screening with
# varied A-column parity beat fitness 41.0074: fitness fell to 38.6494,
# and four of five measured slots retained weight-16 X witnesses.
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
