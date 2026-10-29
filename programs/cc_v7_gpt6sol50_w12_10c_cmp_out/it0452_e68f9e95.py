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
           "independently screening verified divided component minors "
           "should select five new rate-0.150 codes scoring above 39.1566.")
GROUP = ("Keep C37 at order 37: this n=592 bed has measured distance-18 "
         "codes, allowing a direct filling-only test.")
SHAPES = ("Keep (2,3)x(2,4): n=592, guaranteed k>=74 and rate>=0.125; "
          "measured k=76-88 lies in cell 0.150. Its unequal two-row "
          "components have 111 A and 148 B variables, unlike equal "
          "shapes. The parent's lightest X witnesses used sector-1 "
          "Acol/Bcol blocks across all B columns.")
FILLING = ("Cross five legal support profiles with four cyclic-gap "
           "regimes. Select A and B labels independently by divided "
           "minors verified on their exact lifted component rows, "
           "then screen translated sums and check exact k. Both row "
           "weights stay <=12. Target weight-16 to -18 X witnesses "
           "while retaining k above the bound of 74.")
FAMILY_NAME = "r015_c37_23x24_independent_verified_quotients_v3"

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


def _profiles(rng, idx):
    """Five support designs crossed with four distinct gap regimes."""
    motif = idx % 5
    A = [[2, 2, 2], [2, 2, 2]]
    B = [[2] * 4, [2] * 4]
    a_order = rng.sample(range(3), 3)
    b_order = rng.sample(range(4), 4)

    if motif in (0, 1, 3, 4):
        odd = 2 if motif == 0 else 4
        for t, j in enumerate(b_order[:odd]):
            B[0][j], B[1][j] = ((1, 3) if t < odd // 2
                                 else (3, 1))
    # Motif 2 deliberately keeps both components all-binomial.
    if motif == 3:
        A[0][a_order[0]], A[1][a_order[0]] = 3, 1
        A[0][a_order[1]], A[1][a_order[1]] = 1, 3
    elif motif == 4:
        A[0][a_order[0]], A[1][a_order[0]] = 3, 1

    assert _grid_weights(A, B) <= 12
    assert _no_dead_lines(A, B)
    return A, B


def _label_side(rng, counts, regime):
    """Fresh cells with distinct elements and preferentially fresh gaps."""
    pools = (tuple(range(1, 19)), tuple(range(3, 19)),
             tuple(range(1, 19, 2)), tuple(range(2, 19, 2)))
    used = set()
    grid = []
    for counts_row in counts:
        row = []
        for count in counts_row:
            if count == 1:
                shifts = [rng.randrange(37)]
            elif count == 2:
                choices = [d for d in pools[regime] if d not in used]
                gap = rng.choice(choices or pools[regime])
                used.add(gap)
                first = rng.randrange(37)
                sign = 1 if rng.randrange(2) else -1
                shifts = [first, (first + sign * gap) % 37]
            else:
                shifts, lowest = None, 100
                for _ in range(32):
                    candidate = rng.sample(range(37), count)
                    gaps = [
                        min((x - y) % 37, (y - x) % 37)
                        for p, x in enumerate(candidate)
                        for y in candidate[:p]
                    ]
                    penalty = len(gaps) - len(set(gaps))
                    penalty += len(set(gaps) & used)
                    if penalty < lowest:
                        shifts, lowest = candidate, penalty
                    if lowest == 0:
                        break
                used.update(
                    min((x - y) % 37, (y - x) % 37)
                    for p, x in enumerate(shifts)
                    for y in shifts[:p])
            row.append([(x,) for x in shifts])
        grid.append(row)
    return grid


def _certified_component_grade(rows, q, side, deep=False):
    """Grade only component-kernel words certified by verified LP rows."""
    from itertools import combinations

    blocks, stride = (3, 4) if side == "A" else (4, 1)
    check_stride = 4 if side == "A" else 1
    mask = (1 << q) - 1
    checks = [
        rows[i * check_stride * q:(i * check_stride + 1) * q]
        for i in range(2)
    ]
    polys = [
        [(checks[i][0] >> (j * stride * q)) & mask
         for j in range(blocks)]
        for i in range(2)
    ]

    def rotate(x, shift):
        shift %= q
        return ((x << shift) | (x >> (q - shift))) & mask

    def product(x, y):
        out = 0
        while y:
            bit = y & -y
            out ^= rotate(x, bit.bit_length() - 1)
            y ^= bit
        return out

    def divide_even(x):
        if x.bit_count() & 1:
            return None
        quotient = previous = 0
        for position in range(1, q):
            previous ^= (x >> position) & 1
            quotient |= previous << position
        return quotient if previous == (x & 1) else None

    def verified(word):
        expanded = 0
        for j in range(blocks):
            expanded |= ((word >> (j * q)) & mask) << (j * stride * q)
        return all((row & expanded).bit_count() % 2 == 0
                   for group in checks for row in group)

    seeds = set()
    for triple in combinations(range(blocks), 3):
        parts = [0] * blocks
        for j in triple:
            u, v = [t for t in triple if t != j]
            parts[j] = (product(polys[0][u], polys[1][v])
                        ^ product(polys[1][u], polys[0][v]))
        for _ in range(4):
            word = sum(part << (j * q) for j, part in enumerate(parts))
            if word and verified(word):
                seeds.add(word)
            divided = [divide_even(part) for part in parts]
            if any(part is None for part in divided):
                break
            parts = divided

    if not seeds:
        return (0, -100000, -100000, -100000, -100000)
    seeds = sorted(seeds, key=lambda w: (w.bit_count(), w))[:8]
    translated = [
        [sum(rotate((word >> (j * q)) & mask, shift) << (j * q)
             for j in range(blocks))
         for shift in range(q)]
        for word in seeds
    ]
    minimum = blocks * q + 1
    short16 = short17 = short18 = short20 = 0

    def observe(word):
        nonlocal minimum, short16, short17, short18, short20
        if word:
            weight = word.bit_count()
            minimum = min(minimum, weight)
            short16 += weight <= 16
            short17 += weight <= 17
            short18 += weight <= 18
            short20 += weight <= 20

    for word in seeds:
        observe(word)
    for i, word in enumerate(seeds):
        for j in range(i, len(seeds)):
            for shift in range(q):
                observe(word ^ translated[j][shift])

    if deep:
        for i, word in enumerate(seeds):
            for s in range(1, q):
                pair = word ^ translated[i][s]
                for t in range(s + 1, q):
                    observe(pair ^ translated[i][t])
        for i, j, k in combinations(range(len(seeds)), 3):
            for second in translated[j]:
                pair = seeds[i] ^ second
                for third in translated[k]:
                    observe(pair ^ third)

    return minimum, -short16, -short17, -short18, -short20


def NEW_CTX(rng, G, idx):
    """Select independently certified components, then check exact k."""
    acounts, bcounts = _profiles(rng, idx)
    regime = (idx // 5) % 4
    anchor_a = _label_side(rng, acounts, regime)
    anchor_b = _label_side(rng, bcounts, (regime + 2) % 4)
    a_pool, b_pool = [], []

    for trial in range(28):
        A = _label_side(rng, acounts, regime)
        hx, _, _ = lifted_matrices(G, A, anchor_b)
        merit = _certified_component_grade(hx, G.q, "A")
        a_pool.append((merit, -trial, A))
    for trial in range(84):
        B = _label_side(rng, bcounts, (regime + 2) % 4)
        _, hz, _ = lifted_matrices(G, anchor_a, B)
        merit = _certified_component_grade(hz, G.q, "B")
        b_pool.append((merit, -trial, B))

    def finalists(pool, side, limit, keep):
        pool.sort(key=lambda entry: entry[:2], reverse=True)
        out = []
        for _, tie, grid in pool[:limit]:
            A, B = ((grid, anchor_b) if side == "A"
                    else (anchor_a, grid))
            hx, hz, _ = lifted_matrices(G, A, B)
            rows = hx if side == "A" else hz
            merit = _certified_component_grade(rows, G.q, side, True)
            out.append((merit, tie, grid))
        out.sort(key=lambda entry: entry[:2], reverse=True)
        return out[:keep]

    a_best = finalists(a_pool, "A", 6, 3)
    b_best = finalists(b_pool, "B", 10, 4)
    best = None
    for am, at, A in a_best:
        for bm, bt, B in b_best:
            k = lifted_k(G, A, B)
            predicted = min(am[0], bm[0], 22)
            grade = (74 <= k <= 103, k * predicted * predicted,
                     predicted, bm[1:], am[1:], at, bt)
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
# REFUTED: Keeping every C37 A cell binomial while varying B occupancy
# removes weight-16/17 X logicals: four of five measured slots retained
# such sector-1 witnesses.
# REFUTED: Jointly selecting 32 labels by undivided component minors
# improves fitness 39.1566: fitness fell to 38.7652, with measured
# distances 16, 16, 17, 17, and 18.
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
