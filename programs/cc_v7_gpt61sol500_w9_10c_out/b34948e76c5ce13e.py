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
W_MAX = int(os.environ.get("CM_WMAX", "9"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("Dic6, order 24, A=B=(1,4): inverse-distinct unit/binomial constituents should supply twenty inequivalent connected codes with verified k=216 and exact distance three in cell 0.550.")
GROUP = ("Keep the Dic6 presentation and order 24 unchanged. All five measured parent codes had d30s=4, and all 26 distinct slots had d2s=4. The new shape addresses the requested high-rate cell without assuming that this plateau transfers or that group element orders predict distance.")
SHAPES = ("Equal (1,4) constituents have 96 variables and 24 checks. Factor 17 gives n=408; the window permits q<=35 and the frozen group uses q=24. Rank counting gives kb=216 and rate 9/17=0.529412, above 0.525. Unit pivots enforce the actual ranks. Parent minima occupied sector-1 Acol0..2/Bcol0..6 broadly; the new sector-1 layout has sixteen blocks and only one sector-2 block.")
FILLING = ("Both sides use [1,2,2,2], giving HX and HZ maxima nine. Enumerate inverse-distinct direction triples, interleave closure/order prescriptions, and remove common automorphism and side-swap images. Verified full matrices exclude commuting words of weights one and two and verify a non-stabilizer weight-three witness. Connected full Tanner graphs with distinct rooted corner-hypergraph/intersection invariants are retained conservatively. Five novel codes would score 216*9/408=4.764706; strict improvement over the rounded frontier incumbent and unseen archive novelty remain evaluator-dependent.")
FAMILY_NAME = "r055_dic6_14x14_unitbinomial_exact3_rooted_catalog"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    if q != 24:
        raise ValueError("Dic6 has order 24")
    # x^12=1, y^2=x^6, y*x*y^-1=x^-1; canonical element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(12)]

    def red(g):
        a, b = g
        return ((a + 6 * (b // 2)) % 12, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 6 * b * d) % 12,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 12, 0) if b == 0 else ((a + 6) % 12, 1)

    def idx(g):
        a, b = red(g)
        return 12 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


def _column_syndromes(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _component_collisions(rows, width):
    """None rejects kernels <=3; otherwise count pair collisions at weight 4."""
    columns = _column_syndromes(rows, width)
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return None
    pairs = set()
    collisions = 0
    for i, x in enumerate(columns):
        for j in range(i):
            syndrome = x ^ columns[j]
            if syndrome in singles:
                return None
            if syndrome in pairs:
                collisions += 1
            else:
                pairs.add(syndrome)
    return collisions


def _a_catalogue(G):
    """Spanning quotient triples, modulo inversion and group automorphisms."""
    from itertools import combinations

    # Inverse representatives of odd-a elements. The original triples
    # have a bipartite grading and contain no involution differences.
    options = ((1, 0), (3, 0), (5, 0),
               (1, 1), (3, 1), (5, 1))
    representatives = {}

    def unoriented(g):
        return min(g, G.inv(g))

    for triple in combinations(options, 3):
        reached = {G.e}
        frontier = [G.e]
        generators = triple + tuple(G.inv(g) for g in triple)
        while frontier:
            g = frontier.pop()
            for h in generators:
                product = G.mul(g, h)
                if product not in reached:
                    reached.add(product)
                    frontier.append(product)
        if len(reached) != G.q:
            continue

        # Every Dic6 automorphism sends x to x^u and y to x^v*y.
        key = min(
            tuple(sorted(unoriented(((u * a + v * b) % 12, b))
                         for a, b in triple))
            for u in (1, 5, 7, 11) for v in range(12)
        )
        representatives.setdefault(key, triple)

    catalogue = []
    for key in sorted(representatives):
        triple = representatives[key]
        A = [[[G.e, g] for g in triple]]
        # This is a projection of the VERIFIED lift with a dummy side.
        hx, _, _ = lifted_matrices(G, A, [[[G.e]]])
        mask = (1 << (3 * G.q)) - 1
        quality = _component_collisions([r & mask for r in hx], 3 * G.q)
        if quality is not None:
            catalogue.append((triple, quality))
    if not catalogue:
        raise ValueError("No spanning A quotient family passed screening")
    return catalogue


def _b_profiles():
    """Seven support cards: both rows sum to 7; every column sums to 1..3."""
    columns = (
        ((1, 1),) * 7,
        ((2, 0), (0, 2)) + ((1, 1),) * 5,
        ((2, 0),) * 2 + ((0, 2),) * 2 + ((1, 1),) * 3,
        ((2, 0),) * 3 + ((0, 2),) * 3 + ((1, 1),),
        ((1, 0), (1, 2)) + ((1, 1),) * 5,
        ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
        ((1, 0), (3, 0)) + ((1, 1),) * 3 + ((0, 2),) * 2,
    )
    return tuple(tuple(tuple(c[i] for c in profile) for i in range(2))
                 for profile in columns)


def _draw_b(rng, G, profile, mode):
    def cell(count, order_model):
        if count == 0:
            return []
        first = G.pick(rng)
        if count == 1:
            return [first]
        if count == 2:
            if order_model == 0:
                quotient = (rng.choice((1, 5, 7, 11)), 0)  # order 12
            elif order_model == 1:
                quotient = (rng.choice((3, 9)), 0)          # order 4
            else:
                quotient = (rng.randrange(12), 1)          # order 4
            return [first, G.mul(first, quotient)]
        result = [first]
        while len(result) < count:
            g = G.pick(rng)
            if g not in result:
                result.append(g)
        return result

    # Different internal columns can use different quotient-order models.
    return [[cell(profile[i][j], (mode + i + j) % 3)
             for j in range(7)] for i in range(2)]


def _verified_components(hx, hz, q):
    """Project the actual left/right components out of the supplied LP lift."""
    mask = (1 << q) - 1
    arows = []
    for t in range(q):
        source = hx[t]                    # A-row 0, B-column 0
        row = 0
        for j in range(3):
            row |= ((source >> (j * 7 * q)) & mask) << (j * q)
        arows.append(row)
    bmask = (1 << (7 * q)) - 1
    brows = [row & bmask for row in hz[:2 * q]]
    return arows, brows


def _tanner_signature(hx, hz):
    """An uncoloured Tanner-isomorphism invariant, not an isomorphism test.

    Check intersections define two adjacency relations. Each check's
    profile counts common neighbours in both relations for every other
    check. Different signatures certify non-isomorphic Tanner graphs.
    """
    rows = hx + hz
    m = len(rows)
    one, two = [0] * m, [0] * m
    for i in range(m):
        for j in range(i):
            c = (rows[i] & rows[j]).bit_count()
            if c:
                adj = one if c == 1 else two
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    profiles = [{} for _ in rows]
    for i in range(m):
        for j in range(i):
            kind = ((one[i] >> j) & 1) + 2 * ((two[i] >> j) & 1)
            key = (kind,
                   (one[i] & one[j]).bit_count(),
                   (one[i] & two[j]).bit_count()
                   + (two[i] & one[j]).bit_count(),
                   (two[i] & two[j]).bit_count())
            for p in (profiles[i], profiles[j]):
                p[key] = p.get(key, 0) + 1
    return tuple(sorted(tuple(sorted(p.items())) for p in profiles))


def _no_short_kernel(rows, n):
    """Exactly exclude kernel vectors of weights one, two and three."""
    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    seen = set(columns)
    if 0 in seen or len(seen) != n:
        return False
    for i, c in enumerate(columns):
        for j in range(i):
            if (c ^ columns[j]) in seen:
                return False
    return True


from functools import lru_cache


@lru_cache(maxsize=1)
def _highrate_catalogue(G, count):
    """Pure immutable portfolio; memoization changes only computation cost."""
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    q, e = G.q, G.e
    cap = min(9, W_MAX)
    support = [[1, 2, 2, 2]]
    if q != 24 or _grid_weights(support, support) > cap:
        raise ValueError("Dic6 unit/binomial bed requires row cap nine")
    assert _no_dead_lines(support, support)

    def closure(triple):
        reached, pending = {e}, [e]
        steps = triple + tuple(G.inv(g) for g in triple)
        for g in pending:
            for h in steps:
                product = G.mul(g, h)
                if product not in reached:
                    reached.add(product)
                    pending.append(product)
        return len(reached)

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("Invalid element order")

    # Exclude identity and involutions: an involutory binomial has
    # duplicate lifted columns. Three different inverse classes prevent
    # parallel directions within each constituent.
    directions = sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != e and G.mul(g, g) != e
    })
    triples = list(combinations(directions, 3))
    descriptions = [
        (closure(t), tuple(sorted(order(g) for g in t)),
         sum(g[1] for g in t))
        for t in triples
    ]

    # These are actual Dic6 automorphisms, not independent label draws.
    actions = [(u, v) for u in (1, 5, 7, 11) for v in range(12)]
    images = [
        [
            tuple(sorted(
                min(G.idx(h), G.idx(G.inv(h)))
                for h in (((u*a + v*b) % 12, b) for a, b in triple)
            ))
            for u, v in actions
        ]
        for triple in triples
    ]

    def orbit_key(a, b):
        return min(tuple(sorted((x, y)))
                   for x, y in zip(images[a], images[b]))

    # Interleave genuinely different closure/order prescriptions before
    # searching more labels within any one prescription.
    buckets = {}
    for a in range(len(triples)):
        for b in range(a, len(triples)):
            description = tuple(sorted((descriptions[a], descriptions[b])))
            buckets.setdefault(description, []).append((a, b))
    local = random.Random(BASE_SEED ^ 0xD6141455)
    for bucket in buckets.values():
        local.shuffle(bucket)
    bucket_order = sorted(
        buckets, key=lambda key: (
            -int(any(d[0] == q for d in key)),
            -len({d[0] for d in key}), key))

    def plans():
        for turn in range(max(map(len, buckets.values()))):
            for description in bucket_order:
                bucket = buckets[description]
                if turn < len(bucket):
                    yield description, bucket[turn]

    def build(triple):
        return [[[e]] + [[e, g] for g in triple]]

    def no_two(columns):
        return 0 not in columns and len(set(columns)) == len(columns)

    def outside(word, rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p not in pivots:
                    pivots[p] = row
                    break
                row ^= pivots[p]
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                return True
            word ^= pivots[p]
        return False

    def three_witness(hz, hx, columns):
        """Construct from actual syndromes, without assuming a lift action."""
        units = {columns[j]: j for j in range(q)}
        for j in range(q, 2*q):
            syndrome = columns[j]
            if syndrome.bit_count() != 2:
                continue
            bit = syndrome & -syndrome
            other = syndrome ^ bit
            if bit not in units or other not in units:
                continue
            word = ((1 << j) | (1 << units[bit])
                    | (1 << units[other]))
            if word.bit_count() != 3:
                continue
            if any((row & word).bit_count() % 2 for row in hz):
                raise ValueError("Syndrome witness failed commutation")
            if outside(word, hx):
                return word
        return 0

    def fingerprint(hx, hz, n):
        """Intrinsic rooted corner hypergraph plus full check intersections.

        Degree-14 qubits are identified by degree, not sector labels.
        Checks and low-degree qubit corners retain endpoint multiplicity.
        Different fingerprints certify non-isomorphism; equality is
        inconclusive and is conservatively rejected.
        """
        rows = list(hx) + list(hz)
        columns = _column_syndromes(rows, n)
        high = [j for j, c in enumerate(columns) if c.bit_count() == 14]
        if len(high) != q:
            raise ValueError("Unexpected high-qubit degree class")
        position = {j: v for v, j in enumerate(high)}
        ends, tags = [], []
        check_ends = []
        for row in rows:
            endpoints = []
            bits = row
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                if j in position:
                    endpoints.append(position[j])
                bits ^= bit
            endpoints = tuple(sorted(endpoints))
            check_ends.append(endpoints)
            ends.append(endpoints)
            tags.append((0, row.bit_count()))

        for j, syndrome in enumerate(columns):
            if j in position:
                continue
            endpoints, weights = [], []
            bits = syndrome
            while bits:
                bit = bits & -bits
                r = bit.bit_length() - 1
                endpoints.extend(check_ends[r])
                weights.append(rows[r].bit_count())
                bits ^= bit
            ends.append(tuple(sorted(endpoints)))
            tags.append((1, tuple(sorted(weights))))

        incident = [[] for _ in high]
        for edge, endpoints in enumerate(ends):
            for v in endpoints:
                incident[v].append(edge)

        def compress(features):
            palette = {f: i for i, f in enumerate(sorted(set(features)))}
            return [palette[f] for f in features]

        rooted = []
        for root in range(q):
            colours = [int(v == root) for v in range(q)]
            history = []
            for _ in range(5):
                es = [(tag, tuple(sorted(colours[v] for v in endpoints)))
                      for tag, endpoints in zip(tags, ends)]
                ec = compress(es)
                vs = [(colours[v],
                       tuple(sorted(ec[edge] for edge in incident[v])))
                      for v in range(q)]
                history.append((
                    tuple(sorted(Counter(es).items())),
                    tuple(sorted(Counter(vs).items()))))
                colours = compress(vs)
            rooted.append(sha256(repr(history).encode("ascii")).digest())

        # Check projection has the same connectivity as the full Tanner
        # graph because every qubit has positive degree.
        adjacency = [0] * len(rows)
        for column in columns:
            bits = column
            while bits:
                bit = bits & -bits
                adjacency[bit.bit_length() - 1] |= column ^ bit
                bits ^= bit
        reached, pending = {0}, [0]
        for r in pending:
            bits = adjacency[r]
            while bits:
                bit = bits & -bits
                s = bit.bit_length() - 1
                if s not in reached:
                    reached.add(s)
                    pending.append(s)
                bits ^= bit
        if len(reached) != len(rows):
            return None

        return (
            n, tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in columns).items())),
            tuple(sorted(rooted)), _tanner_signature(hx, hz),
        )

    graph_keys, orbit_keys = set(), set()
    results = []
    for description, (a, b) in plans():
        orbit = orbit_key(a, b)
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)
        A, B = build(triples[a]), build(triples[b])
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 408 or max(r.bit_count() for r in hx + hz) > cap:
            continue
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if rx != 4*q or rz != 4*q or k != 216:
            continue
        if 1000*k <= 525*n:
            continue
        cx = _column_syndromes(hx, n)
        cz = _column_syndromes(hz, n)
        if not (no_two(cx) and no_two(cz)):
            continue
        witness = three_witness(hz, hx, cz)
        if not witness:
            continue
        key = fingerprint(hx, hz, n)
        if key is None or key in graph_keys:
            continue
        if not css_ok(hx, hz):
            raise ValueError("Verified high-rate lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("Verified rank helpers disagree")
        graph_keys.add(key)

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        results.append((freeze(A), freeze(B), k, description, witness))
        if len(results) >= count:
            break
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Publish an invariant-separated, exact-distance-three complete filling."""
    catalogue = _highrate_catalogue(G, N_SLOTS)
    if idx >= len(catalogue):
        raise ValueError("Connected exact-distance-three catalogue exhausted")
    A, B, k, card, witness = catalogue[idx]
    return {"A": A, "B": B, "k": k, "card": card,
            "distance_certificate": 3,
            "verified_X_witness": witness,
            "kernel_weights_one_and_two_excluded": True}


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without changing shared state."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free catalogue supplied five measured inequivalent codes; the current table instead retained only four distinct codes from sixty slots, with d30s=4,4,3,3.
# REFUTED: Component weight-four collision minimization on spanning-A Dic6 (1,3)x(2,7) escapes the distance-four plateau: all 26 distinct parent slots had d2s=4 and all five measured slots had d30s=4.
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
