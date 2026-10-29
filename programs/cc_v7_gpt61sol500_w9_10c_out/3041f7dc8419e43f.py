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
CONCEPT = ("C9 semidirect C6 with action 4, order 54, A=B=(1,3): connected binomial constituents with verified kernel exclusion through four, followed by complete CSS screening and an explicit five-logical certificate, should supply five novel distance-five finalists in cell 0.400.")
GROUP = ("Jump from Dic7 to x^9=y^6=1, y*x*y^-1=x^4. The action has order three modulo nine; available long element orders are 6, 9 and 18. This centre-six order-54 group's (1,3)x(1,3) bed is absent from BEDS. The measured order-55 Frobenius bed attained distance-five readings and fitness 10.2002 on these shapes, motivating a directly certified alternative rather than another distance-three high-rate catalogue.")
SHAPES = ("Equal constituents have 162 variables and 54 checks. n=10*54=540, kb=4*54=216 and guaranteed rate 0.400; the window allows q<=59. Primary connected all-binomial pairs must have verified k=222, rate 0.411111, whose distance-five score is 10.277778. The parent's weight-three witnesses occupied sector-1 Acol/Bcol blocks, especially its unit columns. Remove unit pivots and replace four-column constituents by three-column incidence constituents.")
FILLING = ("Enumerate noncommuting inverse-class triples, quotient known semidirect automorphisms, and use only scaffold-derived constituent and complete matrices. Primary counts (2,2,2) on both sides give BOTH maxima eight. Require complete kernel exclusion through four and verify a non-stabilizer weight-five witness. Intrinsic full uncoloured Tanner signatures separate accepted designs. One-trinomial B alternatives have counts (3,2,2), both maxima nine, and the same complete floor-five screen. Only after these branches are exhausted are complete-floor-four binomial fallbacks admitted; no fallback is advertised as distance five. Archive novelty remains evaluator-tested.")
FAMILY_NAME = "r040_c9by6_s4_13x13_verified5_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 54

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C9:C6 with action 4 requires order 54")
    # x^9=y^6=1, y*x*y^-1=x^4; 4^3=1 modulo nine.
    powers = [pow(4, b, 9) for b in range(6)]
    els = [(a, b) for b in range(6) for a in range(9)]

    def red(g):
        return (g[0] % 9, g[1] % 6)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 9, (b + d) % 6)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 6
        return ((-powers[bi] * a) % 9, bi)

    def idx(g):
        a, b = red(g)
        return 9*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (1, 3)               # (rb, cb)


_DIC_PROFILES = (
    (4, 4, 4), (4, 4, 7), (4, 4, 14), (4, 7, 7),
    (4, 7, 14), (4, 14, 14), (7, 7, 7), (7, 7, 14),
    (7, 14, 14), (14, 14, 14),
)
_DIC_PRIMARY = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
    (0, 4), (0, 9), (1, 5), (1, 8), (2, 3),
    (2, 7), (3, 5), (4, 6), (5, 8), (6, 9),
)
_DIC_CARDS = _DIC_PRIMARY + tuple(
    (a, b) for a in range(10) for b in range(a, 10)
    if (a, b) not in _DIC_PRIMARY
)


def _dic_order(G, g):
    h = G.e
    for order in range(1, G.q + 1):
        h = G.mul(h, g)
        if h == G.e:
            return order
    raise ValueError("invalid finite-group order")


def _dic_component(rng, G, orders, pools):
    """A fixed monomial pivot plus three prescribed-order binomials."""
    row = [[G.e]]
    used = set()
    for order in orders:
        available = [h for h in pools[order]
                     if h not in used and G.inv(h) not in used]
        if not available:
            raise ValueError("relative-order card exhausted")
        h = rng.choice(available)
        used.add(h)
        g = G.pick(rng)
        row.append([g, G.mul(g, h)])
    return [row]


def _dic_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _dic_excludes_two(rows, n):
    """Complete exclusion of commuting supports of weights one and two."""
    columns = _dic_columns(rows, n)
    return 0 not in columns and len(set(columns)) == n


def _dic_three_witness(G, B):
    """X witness in Acol1: one B-binomial bit and two B-pivot bits.

    B[0][0]=identity. For binomial labels u,v, selecting its column
    at identity has check syndromes at u^-1 and v^-1, canceled by
    those two columns of the monomial pivot.
    """
    q = G.q
    u, v = B[0][1]
    base = 4*q                  # Acol1/Bcol0
    return ((1 << (base + q + G.idx(G.e)))
            | (1 << (base + G.idx(G.inv(u))))
            | (1 << (base + G.idx(G.inv(v)))))


def _extract_components(G, hx, hz):
    """Extract components from the verified lift, not a separate lift."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    # j=0 HZ rows already have consecutive B-column blocks.
    br = [row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]]
    return ar, br


def _component_probe(rows, n, rng):
    """Exact search through weight 4, then heuristic kernel probes."""
    from itertools import combinations

    columns = [0] * n
    for r, row in enumerate(rows):
        bits = row
        while bits:
            bit = bits & -bits
            columns[bit.bit_length() - 1] |= 1 << r
            bits ^= bit
    if 0 in columns:
        return 1
    singles = set(columns)
    if len(singles) != n:
        return 2

    pairs = set()
    found4 = False
    for i in range(n):
        for j in range(i):
            syndrome = columns[i] ^ columns[j]
            if syndrome in singles:
                return 3
            if syndrome in pairs:
                found4 = True
            pairs.add(syndrome)
    if found4:
        return 4

    best = n + 1
    for _ in range(4):
        order = list(range(n))
        rng.shuffle(order)
        image = [1 << order[c] for c in range(n)]
        pivots = {}
        for original in rows:
            row = 0
            bits = original
            while bits:
                bit = bits & -bits
                row ^= image[bit.bit_length() - 1]
                bits ^= bit
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break

        ascending = sorted(pivots)
        basis = []
        for f in range(n):
            if f in pivots:
                continue
            word = 1 << f
            for p in ascending:
                if (pivots[p] & word).bit_count() & 1:
                    word ^= 1 << p
            basis.append(word)

        for word in basis:
            best = min(best, word.bit_count())
        for a, b in combinations(basis, 2):
            best = min(best, (a ^ b).bit_count())
        for a, b, c in combinations(basis, 3):
            best = min(best, (a ^ b ^ c).bit_count())
        if best <= 5:
            break
    return best


def _four_cycles(rows):
    total = 0
    for i, row in enumerate(rows):
        for other in rows[:i]:
            overlap = (row & other).bit_count()
            total += overlap * (overlap - 1) // 2
    return total


def _retired_dic_context(rng, G, idx):
    """Inactive historical Dic7 catalogue; never publishes this bed's cells."""
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    def signature(hx, hz, n):
        """Necessary Tanner-isomorphism invariant, deliberately uncoloured.

        Check intersections retain their exact multiplicities. Root
        profiles record common-neighbour counts and distance spheres
        separately for each intersection multiplicity. Different
        signatures prove non-isomorphism; equality proves nothing.
        """
        rows = hx + hz
        m = len(rows)
        relations = [[0] * m for _ in range(10)]
        overlap = [[0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i):
                c = (rows[i] & rows[j]).bit_count()
                overlap[i][j] = overlap[j][i] = c
                if c:
                    relations[c][i] |= 1 << j
                    relations[c][j] |= 1 << i
        active = [r for r in relations[1:] if any(r)]

        profiles = [{} for _ in rows]
        for i in range(m):
            for j in range(i):
                common = []
                for a, relation in enumerate(active):
                    for b in range(a + 1):
                        other = active[b]
                        value = (relation[i] & other[j]).bit_count()
                        if a != b:
                            value += (other[i] & relation[j]).bit_count()
                        common.append(value)
                key = (overlap[i][j], tuple(common))
                for p in (profiles[i], profiles[j]):
                    p[key] = p.get(key, 0) + 1

        roots = []
        for root in range(m):
            spheres = []
            for relation in active:
                visited = frontier = 1 << root
                shell = []
                while frontier:
                    neighbors = 0
                    bits = frontier
                    while bits:
                        bit = bits & -bits
                        neighbors |= relation[bit.bit_length() - 1]
                        bits ^= bit
                    frontier = neighbors & ~visited
                    if frontier:
                        shell.append(frontier.bit_count())
                        visited |= frontier
                spheres.append(tuple(shell))
            roots.append((rows[root].bit_count(),
                          tuple(sorted(profiles[root].items())),
                          tuple(spheres)))
        degrees = [0] * n
        for row in rows:
            while row:
                bit = row & -row
                degrees[bit.bit_length() - 1] += 1
                row ^= bit
        return (tuple(sorted(degrees)), tuple(sorted(roots)))

    def cells(steps):
        return [[[G.e]] + [[G.e, h] for h in steps]]

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    def orbit_key(a, b):
        """Common Dic7 automorphisms, step inversion and side interchange."""
        keys = []
        for u in (1, 3, 5, 9, 11, 13):
            for v in range(14):
                pair = []
                for steps in (a, b):
                    transformed = [
                        ((u * g[0] + v * g[1]) % 14, g[1])
                        for g in steps
                    ]
                    pair.append(tuple(sorted(inverse_class(g)
                                             for g in transformed)))
                keys.append(tuple(sorted(pair)))
        return min(keys)

    # The thirteen distinct parent codes, in the frozen element ordering.
    # Rejecting their signatures also rejects every isomorphic image.
    archived = (
        ((5,25),(1,17),(9,26),(3,22),(7,20),(5,14)),
        ((1,21),(0,14),(21,27),(6,22),(6,21),(14,26)),
        ((1,26),(7,14),(18,19),(10,21),(12,15),(16,19)),
        ((13,25),(15,27),(17,27),(11,23),(19,25),(16,26)),
        ((6,21),(15,21),(8,11),(9,21),(11,13),(2,11)),
        ((24,26),(16,22),(0,4),(15,17),(6,10),(2,10)),
        ((0,8),(0,2),(0,9),(15,25),(3,11),(14,27)),
        ((8,20),(1,17),(13,23),(4,21),(16,20),(16,21)),
        ((13,17),(9,22),(2,18),(26,27),(1,12),(4,9)),
        ((9,18),(5,24),(20,24),(10,19),(0,5),(1,4)),
        ((7,15),(7,19),(13,15),(9,22),(3,22),(15,23)),
        ((12,26),(5,7),(17,25),(2,8),(0,2),(19,23)),
        ((8,18),(0,15),(12,16),(12,23),(11,15),(12,21)),
    )
    blocked = set()
    for record in archived:
        A = [[[G.e]] + [[G.elements[a], G.elements[b]]
                       for a, b in record[:3]]]
        B = [[[G.e]] + [[G.elements[a], G.elements[b]]
                       for a, b in record[3:]]]
        hx, hz, n = lifted_matrices(G, A, B)
        blocked.add(signature(hx, hz, n))

    pools = {
        order: [g for g in G.elements
                if _dic_order(G, g) == order
                and G.idx(g) == inverse_class(g)]
        for order in (4, 7, 14)
    }

    def draw_steps(local, orders):
        steps = []
        for order in orders:
            available = [g for g in pools[order] if g not in steps]
            steps.append(local.choice(available))
        return tuple(sorted(steps, key=G.idx))

    # Every slot rebuilds the same accepted prefix. Thus invariant
    # separation is enforced across slots without mutable shared state.
    local = random.Random(BASE_SEED ^ 0xD1C7C3)
    seen_orbits = set()
    accepted = 0
    for attempt in range(8192):
        pa, pb = _DIC_CARDS[attempt % len(_DIC_CARDS)]
        a = draw_steps(local, _DIC_PROFILES[pa])
        b = draw_steps(local, _DIC_PROFILES[pb])
        orbit = orbit_key(a, b)
        if orbit in seen_orbits:
            continue
        seen_orbits.add(orbit)

        A, B = cells(a), cells(b)
        hx, hz, n = lifted_matrices(G, A, B)
        if not (_dic_excludes_two(hx, n)
                and _dic_excludes_two(hz, n)):
            continue
        graph = signature(hx, hz, n)
        if graph in blocked:
            continue

        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if k != 9 * G.q:
            continue
        witness = _dic_three_witness(G, B)
        if witness.bit_count() != 3:
            continue
        if any((row & witness).bit_count() & 1 for row in hz):
            continue
        if gf2_rank(hx + [witness]) != rx + 1:
            continue

        blocked.add(graph)
        if accepted == idx:
            assert css_ok(hx, hz)
            return {"A": A, "B": B, "k": k,
                    "distance_certificate": 3,
                    "card": (pa, pb), "witness": witness}
        accepted += 1

    raise ValueError("new invariant-separated distance-three catalogue exhausted")


def _mc_clean(rows, width, floor):
    """Exactly exclude every nonzero kernel support below floor (4 or 5)."""
    cols = _dic_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for j in range(i):
            value = first ^ cols[j]
            if value in singles:
                return False
            if floor == 5:
                if value in pairs:
                    return False
                pairs.add(value)
    return True


def _mc_five_word(rows, width):
    """Find an actual weight-five constituent word using pair/triple joins."""
    cols = _dic_columns(rows, width)
    pairs = {}
    for i, first in enumerate(cols):
        for j in range(i):
            pairs[first ^ cols[j]] = (1 << i) | (1 << j)
    for i, first in enumerate(cols):
        for j in range(i):
            value = first ^ cols[j]
            for h in range(j):
                other = pairs.get(value ^ cols[h])
                if other is not None:
                    word = other ^ (1 << i) ^ (1 << j) ^ (1 << h)
                    if word.bit_count() == 5:
                        return word
    return None


def _mc_signature(hx, hz, n):
    """Intrinsic invariant of the complete UNCOLOURED Tanner graph.

    Unequal signatures certify non-isomorphism. Equality is inconclusive
    and is rejected conservatively. No group or protograph labels enter.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _dic_columns(rows, n)
    m = len(rows)
    weights = [row.bit_count() for row in rows]
    one, many = [0] * m, [0] * m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            value = (row & rows[j]).bit_count()
            overlaps.append((i, j, value))
            if value:
                relation = one if value == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i, j, value in overlaps:
        common = (
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        profiles[i][(weights[j], value) + common] += 1
        profiles[j][(weights[i], value) + common] += 1
    roots = tuple(sorted(
        (weights[i], tuple(sorted(profiles[i].items())))
        for i in range(m)))

    # These degree classes identify the high-degree sector intrinsically.
    # The projection handles two- and three-endpoint checks alike.
    high = [j for j, col in enumerate(cols) if col.bit_count() >= 10]
    position = {j: v for v, j in enumerate(high)}
    mask = sum(1 << j for j in high)
    adjacency = [Counter() for _ in high]
    markers = [Counter() for _ in high]
    for row in rows:
        bits = row & mask
        endpoints = []
        while bits:
            bit = bits & -bits
            endpoints.append(position[bit.bit_length() - 1])
            bits ^= bit
        for u in endpoints:
            markers[u][(row.bit_count(), len(endpoints))] += 1
            for v in endpoints:
                if u != v:
                    adjacency[u][v] += 1

    colors = [
        (cols[j].bit_count(), tuple(sorted(markers[v].items())))
        for v, j in enumerate(high)]
    walks = []
    size = len(high)
    for root in range(size):
        vector = [0] * size
        vector[root] = 1
        features = [[] for _ in high]
        for _ in range(8):
            following = [0] * size
            for u, count in enumerate(vector):
                if count:
                    for v, multiplicity in adjacency[u].items():
                        following[v] += count * multiplicity
            vector = following
            for v, count in enumerate(vector):
                features[v].append(count)
        walks.append((
            colors[root], tuple(features[root]),
            tuple(sorted((colors[v], tuple(features[v]))
                         for v in range(size)))))

    intrinsic = (
        tuple(sorted(Counter(weights).items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
        roots, tuple(sorted(walks)),
    )
    return sha256(repr(intrinsic).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=1)
def _mc_portfolio(G):
    """Immutable jointly screened portfolio; memoization changes no design."""
    from itertools import combinations, combinations_with_replacement

    if G.q != 54:
        raise ValueError("the semidirect portfolio requires order 54")
    q, width = G.q, 3 * G.q
    mask = (1 << width) - 1
    local = random.Random(BASE_SEED ^ 0x9546405)

    def order(g):
        value = G.e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return length
        raise ValueError("invalid finite-group element order")

    orders = {g: order(g) for g in G.elements}

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    directions = sorted({
        inverse_class(g) for g in G.elements if orders[g] >= 5
    })
    triples = list(combinations(directions, 3))
    local.shuffle(triples)

    # Known valid automorphisms x -> x^u, y -> x^v*y:
    # u is a unit modulo nine and v is divisible by three.
    sums = [sum(pow(4, t, 9) for t in range(b)) % 9
            for b in range(6)]
    actions = [(u, v) for u in (1, 2, 4, 5, 7, 8)
               for v in (0, 3, 6)]

    def triple_key(triple):
        return min(tuple(sorted(
            inverse_class(((u * G.elements[t][0]
                            + v * sums[G.elements[t][1]]) % 9,
                           G.elements[t][1]))
            for t in triple))
            for u, v in actions)

    def component(triple):
        return [[[G.e, G.elements[t]] for t in triple]]

    def constituent(C, side):
        dummy = [[[G.e]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, C, dummy)
        else:
            _, rows, _ = lifted_matrices(G, dummy, C)
        return [row & mask for row in rows]

    strict, backup, used = [], [], set()
    for triple in triples:
        generators = [G.elements[t] for t in triple]
        if any(G.mul(generators[i], generators[j])
               == G.mul(generators[j], generators[i])
               for i in range(3) for j in range(i)):
            continue
        key = triple_key(triple)
        if key in used:
            continue
        used.add(key)
        C = component(key)
        left = constituent(C, "A")
        if gf2_rank(left) != q - 1 or not _mc_clean(left, width, 4):
            continue
        right = constituent(C, "B")
        if gf2_rank(right) != q - 1 or not _mc_clean(right, width, 4):
            continue

        if (_mc_clean(left, width, 5)
                and _mc_clean(right, width, 5)):
            word = _mc_five_word(right, width)
            if word is not None:
                strict.append((key, C, word))
        elif len(backup) < 24:
            backup.append((key, C, None))
        if len(strict) >= 48:
            break

    entries, graph_keys = [], set()

    def freeze(C):
        return tuple(tuple(tuple(cell) for cell in row) for row in C)

    def consider(A, B, floor, word, prescription):
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb):
            return
        if _grid_weights(sa, sb) > min(9, W_MAX):
            return
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 540 or max(r.bit_count() for r in hx + hz) > min(9, W_MAX):
            return
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (375 * n <= 1000 * k < 425 * n):
            return
        if prescription[0] in ("certified-five", "floor-four"):
            if k != 222:
                return
        if not (_mc_clean(hx, n, floor) and _mc_clean(hz, n, floor)):
            return

        # The B constituent word sits in Acol0, entirely in sector one.
        # Verify both commutation and non-stabilizer status on exact lifts.
        if word is not None:
            if word.bit_count() != 5:
                return
            if any((row & word).bit_count() & 1 for row in hz):
                return
            if gf2_rank(hx + [word]) != rx + 1:
                return

        signature = _mc_signature(hx, hz, n)
        if signature in graph_keys:
            return
        if not css_ok(hx, hz):
            raise ValueError("verified semidirect lift failed CSS")
        graph_keys.add(signature)
        entries.append((freeze(A), freeze(B), k, floor, word, prescription))

    # All primary designs have an exact distance-five certificate,
    # not merely a high reading from a limited-budget search.
    plans = list(combinations_with_replacement(range(len(strict)), 2))
    local.shuffle(plans)
    for a, b in plans:
        consider(strict[a][1], strict[b][1], 5, strict[b][2],
                 ("certified-five", strict[a][0], strict[b][0]))
        if len(entries) >= N_SLOTS:
            return tuple(entries)

    # Structured secondary cards: one trinomial and two binomials.
    # The primary A row totals six; B totals seven. Both maxima are nine.
    pools = {
        b: [g for g in G.elements if g[1] == b and orders[g] >= 5]
        for b in range(6)}
    patterns = [(b, c) for b in range(6) for c in range(b, 6)]
    eligible = [G.elements[t] for t in directions]
    if strict:
        for attempt in range(2048):
            b, c = patterns[attempt % len(patterns)]
            if not pools[b] or not pools[c]:
                continue
            g, h = local.choice(pools[b]), local.choice(pools[c])
            if g == h or G.mul(g, h) == G.mul(h, g):
                continue
            s, t = local.sample(eligible, 2)
            B = [[[G.e, g, h], [G.e, s], [G.e, t]]]
            rows = constituent(B, "B")
            if not _mc_clean(rows, width, 5):
                continue
            a = (attempt // len(patterns) + attempt) % len(strict)
            consider(strict[a][1], B, 5, None,
                     ("trinomial-floor-five", strict[a][0],
                      (b, c), tuple(G.idx(z) for z in (g, h, s, t))))
            if len(entries) >= N_SLOTS:
                return tuple(entries)

    # Explicitly lower-priority fallbacks, still novel and fully screened.
    # They never masquerade as certified distance-five entries.
    fallback = strict + backup
    plans = list(combinations_with_replacement(range(len(fallback)), 2))
    local.shuffle(plans)
    for a, b in plans:
        consider(fallback[a][1], fallback[b][1], 4, None,
                 ("floor-four", fallback[a][0], fallback[b][0]))
        if len(entries) >= N_SLOTS:
            break
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed lookup; no slot-dependent relabeling or fresh cell draws."""
    entries = _mc_portfolio(G)
    if idx >= len(entries):
        raise ValueError("invariant-distinct semidirect portfolio exhausted")
    A, B, k, floor, witness, prescription = entries[idx]
    return {"A": A, "B": B, "k": k,
            "verified_kernel_floor": floor,
            "five_logical": witness,
            "prescription": prescription}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh copy of the complete, immutable screened cell."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Distinct Dic7 relative-order cards ensure distinct Tanner graphs: the parent produced only thirteen distinct codes among sixty slots, with numerous cross-card isomorphic duplicates.
# REFUTED: The archive-aware Dic7 signature catalogue supplies twenty evaluator-distinct designs: its sixty-slot evaluation retained sixteen distinct codes, and all five measured finalists remained at distance three and fitness 4.7650.
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
