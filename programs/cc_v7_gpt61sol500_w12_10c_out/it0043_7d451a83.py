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
CONCEPT = ("Metacyclic Z17:Z2, s=16, order 34, A=(1,3), B=(2,5): mixed trinomials and an even-term B check row should produce five rate-0.350 codes with d30s>=8 and positive k-kb; exact ranks and short-kernel screening test this principle.")
GROUP = ("Keep the parent's group and order under L2. Both old and new shapes have size factor 17, so q=34 still gives n=578. Related dihedral (1,3)x(1,3) fillings measured d30s=8, but transfer to this new bed is experimental.")
SHAPES = ("Target cell 0.350: n=578, kb=34*2*3=204 and guaranteed rate 6/17=0.352941. Component lengths/check counts are 102/34 and 170/68. Parent minima were weight-four sector-1 Acol/Bcol operators. Replace A's binomial columns and give B a second check row; gate every emitted code by verified 0.325<=k/n<0.375.")
FILLING = ("Five A rotation-step triples cross four B support/coset architectures. Every component column has three terms. B has an even-term row with deliberate zeros, testing controlled rank surplus; its row totals are (8,7) or (6,9), giving HX weight 12 and HZ weight at most 12. Optimize independent voltages with stabilizer-tested component witnesses, exact component kernel exclusion through weight seven using odd column parity, and full-matrix commuting-weight-four statistics. Witness readings remain upper bounds, not quantum-distance certificates.")
FAMILY_NAME = "r035_m17r2s16_13x25_evenrow_oddcolumns_screen7"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 34

def custom_ops(q):
    # Same metacyclic construction; order adjusted for the new shapes.
    # Z_17 x| Z_2: y x y^-1 = x^16, and 16^2 = 1 mod 17.
    m, r, s = 17, 2, 16
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)

# Five A step geometries crossed with four B architectures.
# These constrain relative voltages, not a common automorphism orbit.
_A_STEPS = (
    (1, 2, 3), (1, 2, 4), (1, 3, 5),
    (1, 3, 6), (1, 4, 7),
)
# (number of nonzero cells in B's even row,
#  number of those cells in the reflection coset after bottom gauge).
_B_ARCHITECTURES = ((4, 0), (4, 1), (4, 2), (3, 1))


def _draw_design(rng, G, idx):
    m = G.q // 2
    steps = _A_STEPS[(idx % 20) // 4]
    active, reflected = _B_ARCHITECTURES[idx % 4]

    # A's first reflection is a gauge anchor; the remaining relative
    # reflection positions are genuinely optimized, not globally shifted.
    A = [[[(0, 0), (d, 0), (0 if j == 0 else rng.randrange(m), 1)]
          for j, d in enumerate(steps)]]

    # Every B column has exactly three terms. Its even row creates
    # structured dependencies without leaving any dead row or column.
    top, bottom = [], []
    anchor = 1 + ((idx // 20) % (m // 2))
    available = [d for d in range(1, m // 2 + 1) if d != anchor]
    differences = [anchor] + rng.sample(available, active - 1)
    for j in range(active):
        a = rng.randrange(m)
        b = int(j < reflected)
        d = differences[j]
        top.append([(a, b), ((a + d) % m, b)])
        bottom.append([(0, 0)])
    for j in range(active, B_SHAPE[1]):
        top.append([])
        bottom.append([(0, 0), (rng.randrange(1, m), 0),
                       (rng.randrange(m), 1)])
    return A, [top, bottom]


def _syndrome_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= marker
            row ^= bit
    return columns


def _component_columns(hx, hz, q):
    """Extract both actions only from the VERIFIED LP matrices."""
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q) - 1
    ar = []
    for row in hx[:q]:          # A-row zero, B-column zero
        packed = 0
        for j in range(ca):
            packed |= ((row >> (j * cb * q)) & mask) << (j * q)
        ar.append(packed)
    bm = (1 << (cb * q)) - 1
    br = [row & bm for row in hz[:B_SHAPE[0] * q]]
    return (_syndrome_columns(ar, ca * q),
            _syndrome_columns(br, cb * q))


def _span_basis(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = row
                break
            row ^= old
    return basis


def _outside_span(word, basis):
    while word:
        p = word.bit_length() - 1
        old = basis.get(p)
        if old is None:
            return True
        word ^= old
    return False


def _odd_column_kernel_floor(columns, through_six=True):
    """Exact component kernel floor through weight seven.

    All columns have odd weight, so every kernel word has even weight.
    After weights two and four are excluded, distinct equal-syndrome
    triples must be disjoint and therefore witness weight six.
    Return eight if weights <=7 are excluded; this is not an LP bound.
    """
    assert all(c.bit_count() & 1 for c in columns)
    if len(set(columns)) != len(columns):
        return 2
    pairs = set()
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in pairs:
                return 4
            pairs.add(syndrome)
    del pairs
    if not through_six:
        return 6

    triples = set()
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if syndrome in triples:
                    return 6
                triples.add(syndrome)
    return 8


def _logical_probe(rng, columns, stabilizers, side, q,
                   trials, best):
    """Actual sector-1 logical witnesses from randomized information sets."""
    ca, cb = A_SHAPE[1], B_SHAPE[1]
    mask = (1 << q) - 1
    order = list(range(len(columns)))

    def consider(word, bound):
        weight = word.bit_count()
        if not word or weight >= bound:
            return bound
        if side == "A":
            for beta in range(cb):
                full = 0
                for j in range(ca):
                    full |= ((word >> (j * q)) & mask) << (
                        (j * cb + beta) * q)
                if _outside_span(full, stabilizers):
                    return weight
        else:
            for j in range(ca):
                full = word << (j * cb * q)
                if _outside_span(full, stabilizers):
                    return weight
        return bound

    for _ in range(trials):
        rng.shuffle(order)
        pivots, dependencies = {}, []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                dependencies.append(word)
                best = consider(word, best)
        short = sorted(dependencies, key=int.bit_count)[:20]
        for i, word in enumerate(short):
            for j in range(i):
                best = consider(word ^ short[j], best)
        small = short[:8]
        for i, word in enumerate(small):
            for j in range(i):
                pair = word ^ small[j]
                for h in range(j):
                    best = consider(pair ^ small[h], best)
        if best <= 4:
            break
    return best


def _small_kernel_stats(rows, n):
    """Exact commuting-vector screening through weight three.

    With distinct nonzero column syndromes, equal pair syndromes
    involve disjoint pairs. Every weight-four kernel vector has
    exactly three pair partitions, permitting exact counting.
    """
    cols = [0] * n
    for r, row in enumerate(rows):
        bit = 1 << r
        while row:
            lo = row & -row
            cols[lo.bit_length() - 1] |= bit
            row ^= lo

    singles = set(cols)
    if 0 in singles or len(singles) != n:
        return None

    pairs = {}
    for i, a in enumerate(cols):
        for j in range(i + 1, n):
            syndrome = a ^ cols[j]
            if syndrome in singles:
                return None
            packed = i * n + j
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = packed
            elif isinstance(previous, int):
                pairs[syndrome] = [previous, packed]
            else:
                previous.append(packed)

    histogram = {}
    participation = [0] * n
    collisions = 0
    for syndrome, value in pairs.items():
        count = 1 if isinstance(value, int) else len(value)
        key = (count, syndrome.bit_count())
        histogram[key] = histogram.get(key, 0) + 1
        if count > 1:
            collisions += count * (count - 1) // 2
            for packed in value:
                i, j = divmod(packed, n)
                participation[i] += count - 1
                participation[j] += count - 1

    return (collisions // 3, tuple(sorted(histogram.items())),
            tuple(c.bit_count() for c in cols),
            tuple(p // 3 for p in participation))


def _incidence_fingerprint(hx, hz, sx, sz):
    # Invariant under qubit/check permutations and a global X/Z swap.
    joint = tuple(sorted(zip(sx[2], sz[2], sx[3], sz[3])))
    swapped = tuple(sorted(zip(sz[2], sx[2], sz[3], sx[3])))
    row_profiles = tuple(sorted((
        tuple(sorted(r.bit_count() for r in hx)),
        tuple(sorted(r.bit_count() for r in hz)))))
    return (tuple(sorted((sx[1], sz[1]))),
            min(joint, swapped), row_profiles)


def NEW_CTX(rng, G, idx):
    """Optimize the slot's prescribed geometry, not a relabeling orbit."""
    q = G.q
    active, _ = _B_ARCHITECTURES[idx % 4]
    sa = [[3, 3, 3]]
    sb = [[2] * active + [0] * (5 - active),
          [1] * active + [3] * (5 - active)]
    assert _grid_weights(sa, sb) <= W_MAX
    assert _no_dead_lines(sa, sb)
    kb = q * (A_SHAPE[1] - A_SHAPE[0]) * (
        B_SHAPE[1] - B_SHAPE[0])

    def merit(item):
        d = min(item["da"], item["db"])
        return (item["k"] * d * d / item["n"],
                item["da"] + item["db"])

    pool = []
    seen = set()
    admitted = 0
    for attempt in range(96):
        A, B = _draw_design(rng, G, idx)
        key = tuple(tuple(sorted(cell)) for grid in (A, B)
                    for row in grid for cell in row)
        if key in seen:
            continue
        seen.add(key)
        hx, hz, n = lifted_matrices(G, A, B)
        xs, zs = _span_basis(hx), _span_basis(hz)
        # These are exactly the ranks used by the evaluator.
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (325 * n <= 1000 * k < 375 * n):
            continue
        # The even B row is useful only if its dependencies survive
        # in the complete LP matrices.
        if k <= kb:
            continue

        ac, bc = _component_columns(hx, hz, q)
        item = {
            "A": A, "B": B, "hx": hx, "hz": hz,
            "xs": xs, "zs": zs, "ac": ac, "bc": bc,
            "n": n, "k": k,
            "da": _logical_probe(rng, ac, zs, "A", q, 8, n + 1),
            "db": _logical_probe(rng, bc, xs, "B", q, 8, n + 1),
        }
        # Require actual witnesses, rather than rewarding search failure.
        if item["da"] > n or item["db"] > n:
            continue
        pool.append(item)
        pool.sort(key=merit, reverse=True)
        del pool[4:]
        admitted += 1
        if admitted >= 24:
            break

    if not pool:
        raise ValueError("no rank-surplus code passed the target-rate gate")

    winner = None
    for item in pool:
        item["da"] = _logical_probe(
            rng, item["ac"], item["zs"], "A", q, 48, item["da"])
        item["db"] = _logical_probe(
            rng, item["bc"], item["xs"], "B", q, 48, item["db"])

        fa = _odd_column_kernel_floor(item["ac"])
        # If A already has a weight-six word, only B weights <=4
        # can lower the joint floor; avoid an unnecessary triple table.
        fb = _odd_column_kernel_floor(
            item["bc"], through_six=(fa == 8))
        floor = min(fa, fb)

        # Component exclusion is not quantum exclusion: also screen
        # complete matrices for short mixed-sector commuting supports.
        sx = _small_kernel_stats(item["hx"], item["n"])
        if sx is None:
            continue
        sz = _small_kernel_stats(item["hz"], item["n"])
        if sz is None:
            continue
        four = sx[0] + sz[0]
        d = min(item["da"], item["db"])
        objective = (
            int(four == 0), floor,
            item["k"] * d * d / item["n"],
            item["da"] + item["db"], -four,
        )
        if winner is None or objective > winner[0]:
            winner = (objective, item)

    if winner is None:
        raise ValueError("all finalists had commuting supports below four")
    chosen = winner[1]
    if not css_ok(chosen["hx"], chosen["hz"]):
        raise ValueError("verified lift failed CSS check")
    return {"A": chosen["A"], "B": chosen["B"]}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell; all new-shape indices are defined."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free binomial steps plus one trinomial per single-row side remove the weight-four bottleneck: all five measured order-34 parent leaders instead had d30s=4 and k-kb=0.
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
