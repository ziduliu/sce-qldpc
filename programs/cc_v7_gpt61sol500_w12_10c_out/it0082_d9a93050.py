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
CONCEPT = ("Inversion-metacyclic C17:C2, order 34, shapes (1,3)x(1,3): balanced trinomials selected against short component dependencies should raise the parent's five measured distance-four readings; readings above eight would improve the 0.400 incumbent.")
GROUP = ("Keep the exact parent's metacyclic group, m=17,r=2,s=16,q=34. The related q=58 (1,3)x(1,3) bed reached five measured distance-eight readings with mixed trinomials. The smaller shapes allow retaining q=34; transfer of that distance is a hypothesis, not a guarantee.")
SHAPES = ("Target cell 0.400: n=10*34=340 and kb=4*34=136, guaranteed rate 0.400. Equal components have 102 variables and 34 checks. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly; replace both binomial-rich components. Verified lifted ranks enforce 0.375<=k/n<0.425 for every emitted code.")
FILLING = ("All six cells contain three distinct terms, so both check-weight maxima are twelve. Twenty recipes cross five relative-step architectures with four pure/mixed-trinomial placements. Search fresh reflection offsets, exactly screen component dependencies through weight six, deepen sampled kernel descent, and screen complete verified matrices through commuting weight three. Incidence and check-overlap invariants exclude detected orbit copies; component readings remain heuristics, not quantum-distance certificates.")
FAMILY_NAME = "r040_m17r2s16_13x13_balanced_component6"

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
B_SHAPE = (1, 3)               # (rb, cb)

from itertools import combinations, combinations_with_replacement
from functools import lru_cache


def _unsigned_step(x, m):
    x %= m
    return min(x, (-x) % m)


def _normal_side(side, u, m):
    steps, tri = side
    ds = tuple(sorted(_unsigned_step(u * d, m) for d in steps))
    if tri[0] == 0:
        # Mixed trinomial: normalize its reflection by conjugation.
        ts = (0, _unsigned_step(u * tri[1], m))
    else:
        # Rotation trinomial: remove translation and independent inversion.
        support = tri[1:]
        ts = (1,) + min(
            tuple(sorted((sgn * u * (x - base)) % m for x in support))
            for base in support for sgn in (1, -1))
    return ds, ts


def _canonical_plan(a, b, m):
    # Remove common group automorphisms, column permutations, and side swap.
    return min(
        tuple(sorted((_normal_side(a, u, m), _normal_side(b, u, m))))
        for u in range(1, m))


def _plan_cells(side):
    steps, tri = side
    cells = [((0, 0), (d, 0)) for d in steps]
    if tri[0] == 0:
        cells.append(((0, 0), (tri[1], 0), (0, 1)))
    else:
        cells.append(tuple((a, 0) for a in tri[1:]))
    return (tuple(cells),)


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


_STEP_ARCHITECTURES = (
    (1, 2, 3), (1, 2, 4), (1, 3, 5),
    (1, 4, 7), (1, 5, 8),
)


def _balanced_draw(rng, recipe):
    """Five relative-step architectures x four pure/mixed placements."""
    geometry, placement = divmod(recipe, 4)
    grids = []
    for side in range(2):
        steps = _STEP_ARCHITECTURES[(geometry + 2*side) % 5]
        unit = rng.randrange(1, 17)
        pure = bool(placement & (1 << side))
        row = []
        for j, step in enumerate(steps):
            d = unit * step % 17
            if pure and j == 0:
                # A genuine three-point rotation architecture, not a
                # translated mixed cell. At most one pure cell per side.
                ratio = (3, 4, 5, 6, 7)[geometry]
                cell = [(0, 0), (d, 0), (ratio*d % 17, 0)]
            else:
                cell = [(0, 0), (d, 0), (rng.randrange(17), 1)]
            assert len(set(cell)) == 3
            row.append(cell)
        grids.append([row])
    return grids[0], grids[1]


def _balanced_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        marker = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length()-1] |= marker
            row ^= bit
    return columns


def _balanced_components(hx, hz, q):
    """Extract both components from the verified LP actions."""
    mask = (1 << q)-1
    ar = [
        sum(((row >> (j*3*q)) & mask) << (j*q)
            for j in range(3))
        for row in hx[:q]
    ]
    br = [row & ((1 << (3*q))-1) for row in hz[:q]]
    return (_balanced_columns(ar, 3*q),
            _balanced_columns(br, 3*q))


def _balanced_floor(columns, through_six=False):
    """Exact exclusion through four, optionally through six.

    Every column here has odd weight three, so every kernel word has
    even weight. With no weight-two or weight-four dependency, equal
    triple syndromes necessarily give a disjoint weight-six support.
    Seven means weights <=6 were excluded, not that distance is seven.
    """
    if any(c == 0 for c in columns):
        return 1
    if len(set(columns)) != len(columns):
        return 2
    pairs = set()
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in pairs:
                return 4
            pairs.add(syndrome)
    if not through_six:
        return 5
    triples = set()
    for i, x in enumerate(columns):
        for j in range(i):
            pair = x ^ columns[j]
            for h in range(j):
                syndrome = pair ^ columns[h]
                if syndrome in triples:
                    return 6
                triples.add(syndrome)
    return 7


def _balanced_probe(rng, columns, trials):
    """Actual classical kernel witnesses; high readings are not proofs."""
    width = len(columns)
    order = list(range(width))
    best = width + 1
    for _ in range(trials):
        rng.shuffle(order)
        pivots = {}
        words = []
        for c in order:
            syndrome, word = columns[c], 1 << c
            while syndrome:
                p = syndrome.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if syndrome == 0:
                words.append(word)
                best = min(best, word.bit_count())

        # XOR reduction stays in the exact component kernel.
        for sweep in range(2):
            words.sort(key=int.bit_count)
            changed = False
            for i in range(len(words)):
                word = words[i]
                weight = word.bit_count()
                for other in words[:i]:
                    trial = word ^ other
                    tw = trial.bit_count()
                    if trial and tw < weight:
                        word, weight = trial, tw
                        changed = True
                words[i] = word
                best = min(best, weight)
            if not changed:
                break
        short = sorted(set(words), key=int.bit_count)[:24]
        for i, word in enumerate(short):
            for other in short[:i]:
                trial = word ^ other
                if trial:
                    best = min(best, trial.bit_count())
        if best <= 4:
            break
    return best


def _balanced_cycles(columns):
    total = 0
    for i, a in enumerate(columns):
        for b in columns[:i]:
            overlap = (a & b).bit_count()
            total += overlap * (overlap-1) // 2
    return total


def _check_overlap_signature(rows):
    """Permutation-invariant local cycle data on the check-overlap graph."""
    count = len(rows)
    overlaps = [[0] * count for _ in range(count)]
    adjacency = [0] * count
    for i, row in enumerate(rows):
        for j in range(i):
            weight = (row & rows[j]).bit_count()
            overlaps[i][j] = overlaps[j][i] = weight
            if weight:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

    profiles = []
    for i, neighbors in enumerate(adjacency):
        remaining = neighbors
        triangle_twice = 0
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length()-1
            triangle_twice += (neighbors & adjacency[j]).bit_count()
            remaining ^= bit
        common = tuple(sorted(
            (neighbors & adjacency[j]).bit_count()
            for j in range(count) if j != i))
        profiles.append((
            tuple(sorted(overlaps[i])),
            triangle_twice // 2,
            common,
        ))
    return tuple(sorted(profiles))


@lru_cache(maxsize=1)
def _design_bank(G):
    """Independently optimize indexed recipes; cache is transparent."""
    assert G.q == 34
    supports = [[3, 3, 3]]
    assert _grid_weights(supports, supports) <= W_MAX
    assert _no_dead_lines(supports, supports)
    seen = set()
    bank = []

    def merit(item):
        da, db = item["da"], item["db"]
        return (min(da, db), da + db, -item["cycles"],
                -abs(item["k"]-136))

    for slot in range(N_SLOTS):
        rng = random.Random(40034019 + 100003*slot)
        recipe = slot % 20
        winner = None

        for batch in range(2):
            pool = []
            local_seen = set()
            for attempt in range(48):
                A, B = _balanced_draw(rng, recipe)
                key = tuple(tuple(sorted(cell))
                            for grid in (A, B)
                            for row in grid for cell in row)
                if key in local_seen:
                    continue
                local_seen.add(key)

                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (375*n <= 1000*k < 425*n):
                    continue
                ac, bc = _balanced_components(hx, hz, G.q)
                fa = _balanced_floor(ac)
                fb = _balanced_floor(bc)
                da = fa if fa < 5 else _balanced_probe(rng, ac, 6)
                db = fb if fb < 5 else _balanced_probe(rng, bc, 6)
                item = {
                    "A": A, "B": B, "hx": hx, "hz": hz,
                    "n": n, "k": k, "ac": ac, "bc": bc,
                    "da": da, "db": db,
                    "cycles": _balanced_cycles(ac) + _balanced_cycles(bc),
                }
                pool.append(item)
                pool.sort(key=merit, reverse=True)
                del pool[6:]

            for item in pool:
                for side in ("a", "b"):
                    columns = item[side+"c"]
                    floor = _balanced_floor(columns, through_six=True)
                    reading = (floor if floor < 7 else
                               _balanced_probe(rng, columns, 32))
                    item["d"+side] = min(item["d"+side], reading)

            for item in sorted(pool, key=merit, reverse=True):
                hx, hz, n = item["hx"], item["hz"], item["n"]
                sx = _small_kernel_stats(hx, n)
                if sx is None:
                    continue
                sz = _small_kernel_stats(hz, n)
                if sz is None:
                    continue
                fingerprint = (
                    _incidence_fingerprint(hx, hz, sx, sz),
                    tuple(sorted((_check_overlap_signature(hx),
                                  _check_overlap_signature(hz)))),
                )
                if fingerprint in seen:
                    continue

                light = sx[0] + sz[0]
                objective = (
                    light == 0,
                    min(item["da"], item["db"]),
                    -light,
                    item["da"] + item["db"],
                    -item["cycles"],
                    -abs(item["k"]-136),
                )
                if winner is None or objective > winner[0]:
                    winner = (objective, item, fingerprint)
            if winner is not None and winner[0][0] and winner[0][1] >= 8:
                break

        if winner is None:
            # A failed recipe never receives an already emitted orbit.
            bank.append(None)
            continue
        _, item, fingerprint = winner
        assert css_ok(item["hx"], item["hz"])
        assert 375*item["n"] <= 1000*item["k"] < 425*item["n"]
        seen.add(fingerprint)
        bank.append((item["A"], item["B"]))

    if sum(item is not None for item in bank) < min(5, N_SLOTS):
        raise ValueError("fewer than five invariant-distinct legal designs")
    return tuple(bank)


def NEW_CTX(rng, G, idx):
    """Assign a different screened structural design to each slot."""
    bank = _design_bank(G)
    if idx >= len(bank) or bank[idx] is None:
        # Never fill a failed recipe with an orbit copy.
        raise ValueError("recipe found no invariant-distinct screened design")
    A, B = bank[idx]
    return {"A": A, "B": B}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell; all new-shape indices are defined."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal nonzero rolled supports with w<=12 do not ensure d>=3: parent slots 0-5 had d2s=2; its measured leaders remained at d30s=4.
# REFUTED: Sum-free rotation binomials plus one trinomial per D34 (1,4) component, selected by weight-four multiplicity, did not remove the distance-four plateau: all five measured parent readings were four, with sector-1 minima.
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
