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
CONCEPT = ("Dic2=Q8, order 8, A=B=(3,7): balanced triangle-core and degree-three reinforcement catalogues should produce five novel distance-eight measured codes in cell 0.300; full-rank finalists would score 128*64/464=17.655 above its 15.207 incumbent.")
GROUP = ("Retain the dicyclic construction rule, resizing only its parameter from seven to two. Factor 58 excludes order 28; order eight gives legal n=464. The recorded Q8 (3,7) bed reached fitness 9.9, so the proposed gain must come from optimized reinforcement and verified matrix screens, not from an asserted group advantage.")
SHAPES = ("Change both sides to (3,7), giving equal 56-variable, 24-check constituents. n=58*8=464 and kb=16*8=128 give guaranteed rate 16/58=0.275862 in cell 0.300, intentionally leaving the requested high-rate distance-three plateau. The shape permits q<=10; the selected dicyclic order is eight. Parent weight-three minima were entirely sector-1 Acol/Bcol witnesses, especially involving pivot columns. Three-row degree-three reinforcement replaces that pivot/binomial mechanism.")
FILLING = ("Optimize balanced count profiles with each row totaling six and each column at most three: BOTH CSS maxima are nine. Interleave a three-edge triangle core with other intrinsic support profiles, changing actual relative voltages rather than emitting orbit copies. Exact verified constituent screens exclude weights one through five and count weight-six words; six-free constituents and stronger actual kernel witnesses receive priority. Final verified lifts require target-cell ranks, CSS, stabilizer-aware logical exclusion through four and actual nontrivial logical witnesses. Complete uncoloured Tanner rooted-refinement invariants conservatively reject repeated outputs. Distance-eight survival and unseen-archive novelty remain evaluator-tested hypotheses.")
FAMILY_NAME = "r030_dic2_37x37_triangle_hyperedge_kernel6_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic2 requires order eight")
    # Same dicyclic rule: x^(2*t)=1, y^2=x^t, y*x*y^-1=x^-1.
    t = 2
    modulus = 2*t
    els = [(a, b) for b in range(2) for a in range(modulus)]

    def red(g):
        a, b = g
        return ((a + t*(b//2)) % modulus, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + t*b*d) % modulus,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a+t) % modulus, b)

    def idx(g):
        a, b = red(g)
        return modulus*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 7)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


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


from functools import lru_cache


@lru_cache(maxsize=1024)
def _dic_joint_incidence_key(hx, hz, q):
    """Deterministic intrinsic invariant of the UNCOLOURED Tanner graph.

    Memoization changes computation cost only, never catalogue ownership.
    Different keys certify non-isomorphism; equal keys are inconclusive.
    """
    from collections import Counter
    from hashlib import sha256

    rows = hx + hz
    n = 17 * q
    m = len(rows)
    columns = _dic_columns(rows, n)

    # Preserve the parent's complete check-intersection information.
    one, many = [0] * m, [0] * m
    overlaps = []
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            overlaps.append((i, j, overlap))
            if overlap:
                relation = one if overlap == 1 else many
                relation[i] |= 1 << j
                relation[j] |= 1 << i

    profiles = [Counter() for _ in rows]
    for i, j, overlap in overlaps:
        common = (
            (one[i] & one[j]).bit_count(),
            (one[i] & many[j]).bit_count()
            + (many[i] & one[j]).bit_count(),
            (many[i] & many[j]).bit_count(),
        )
        profiles[i][(rows[j].bit_count(), overlap) + common] += 1
        profiles[j][(rows[i].bit_count(), overlap) + common] += 1
    intersection = tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profiles[i].items())))
        for i in range(m)
    ))

    # Degree fourteen intrinsically identifies the pivot-sector qubits.
    # Neither CSS colours nor protograph/group labels identify this class.
    high = [j for j, column in enumerate(columns)
            if column.bit_count() == 14]
    if len(high) != q:
        raise ValueError("certified support lost its degree-fourteen class")
    position = {j: i for i, j in enumerate(high)}
    high_mask = sum(1 << j for j in high)
    edges = [[Counter() for _ in high] for _ in high]
    anchors = [Counter() for _ in high]

    for row in rows:
        bits = row & high_mask
        if bits.bit_count() == 1:
            u = position[bits.bit_length() - 1]
            anchors[u][row.bit_count()] += 1
        elif bits.bit_count() == 2:
            first = bits & -bits
            u = position[first.bit_length() - 1]
            v = position[(bits ^ first).bit_length() - 1]
            edges[u][v][row.bit_count()] += 1
            edges[v][u][row.bit_count()] += 1
        else:
            raise ValueError("unexpected intrinsic projected incidence")

    markers = [tuple(sorted(anchor.items())) for anchor in anchors]
    initial = [
        (int(u == v), markers[u], markers[v],
         tuple(sorted(edges[u][v].items())))
        for u in range(q) for v in range(q)
    ]
    palette = {feature: i for i, feature in
               enumerate(sorted(set(initial)))}
    colours = [palette[feature] for feature in initial]
    pair_history = [tuple(sorted(Counter(initial).items()))]

    # Two-dimensional colour refinement keeps correlations between
    # the A and B actions that independent constituent statistics lose.
    for _ in range(4):
        features = []
        for u in range(q):
            for v in range(q):
                routes = Counter(
                    (colours[u * q + w], colours[w * q + v])
                    for w in range(q)
                )
                features.append((
                    colours[u * q + v], tuple(sorted(routes.items()))
                ))
        pair_history.append(tuple(sorted(Counter(features).items())))
        palette = {feature: i for i, feature in
                   enumerate(sorted(set(features)))}
        following = [palette[feature] for feature in features]
        stable = len(set(following)) == len(set(colours))
        colours = following
        if stable:
            break

    # Refine the COMPLETE bipartite graph, rooted separately at every
    # member of the intrinsic high-degree class. Sorting root signatures
    # removes root ordering. Initial colours use degree and root status.
    incidence = [[] for _ in range(m + n)]
    for j, column in enumerate(columns):
        vertex = m + j
        while column:
            bit = column & -column
            check = bit.bit_length() - 1
            incidence[check].append(vertex)
            incidence[vertex].append(check)
            column ^= bit
    degrees = [len(neighbours) for neighbours in incidence]
    rooted = []

    for j in high:
        root = m + j
        initial = [(degree, int(v == root))
                   for v, degree in enumerate(degrees)]
        palette = {feature: i for i, feature in
                   enumerate(sorted(set(initial)))}
        colours = [palette[feature] for feature in initial]
        history = [tuple(sorted(Counter(initial).items()))]
        for _ in range(5):
            features = [
                (colours[v],
                 tuple(sorted(colours[w] for w in neighbours)))
                for v, neighbours in enumerate(incidence)
            ]
            history.append(tuple(sorted(Counter(features).items())))
            palette = {feature: i for i, feature in
                       enumerate(sorted(set(features)))}
            following = [palette[feature] for feature in features]
            stable = len(set(following)) == len(set(colours))
            colours = following
            if stable:
                break
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    description = (
        tuple(sorted(Counter(degrees).items())),
        intersection, tuple(pair_history), tuple(sorted(rooted)),
    )
    return sha256(repr(description).encode("ascii")).digest()


def _retired_dic7_ctx(rng, G, idx):
    """Inactive historical reference; never publishes the new shapes.

    Only deterministic invariant values are memoized. Selection has no
    mutable cross-slot state and does not depend on invocation order.
    """
    from itertools import combinations, combinations_with_replacement
    from math import gcd

    assert G.q == 28
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    def graph_signature(hx, hz):
        return _dic_joint_incidence_key(tuple(hx), tuple(hz), G.q)

    # All thirteen distinct representatives in the reported parent table.
    # Each packet encodes its six binomial cells by exact element indices:
    # three A pairs followed by three B pairs; both pivots are identity.
    # These are exclusion references only, never candidate outputs.
    archived = (
        (5,25, 1,17, 9,26, 3,22, 7,20, 5,14),
        (1,21, 0,14, 21,27, 6,22, 6,21, 14,26),
        (1,26, 7,14, 18,19, 10,21, 12,15, 16,19),
        (13,25, 15,27, 17,27, 11,23, 19,25, 16,26),
        (6,21, 15,21, 8,11, 9,21, 11,13, 2,11),
        (24,26, 16,22, 0,4, 15,17, 6,10, 2,10),
        (0,8, 0,2, 0,9, 15,25, 3,11, 14,27),
        (8,20, 1,17, 13,23, 4,21, 16,20, 16,21),
        (13,17, 9,22, 2,18, 26,27, 1,12, 4,9),
        (9,18, 5,24, 20,24, 10,19, 0,5, 1,4),
        (7,15, 7,19, 13,15, 9,22, 3,22, 15,23),
        (12,26, 5,7, 17,25, 2,8, 0,2, 19,23),
        (8,18, 0,15, 12,16, 12,23, 11,26, 12,21),
    )
    graph_keys = set()
    for packet in archived:
        sides = []
        for offset in (0, 6):
            row = [[G.e]]
            for p in range(offset, offset + 6, 2):
                row.append([G.elements[packet[p]],
                            G.elements[packet[p + 1]]])
            sides.append([row])
        hx, hz, _ = lifted_matrices(G, sides[0], sides[1])
        graph_keys.add(graph_signature(hx, hz))

    # All fourteen distinct representatives in the CURRENT table, plus
    # normalized references reported in the shown earlier Dic7 catalogues.
    # These are exclusion references only, never candidate seeds.
    normalized_archive = (
        ((1, 17, 20), (1, 19, 20)),
        ((3, 16, 19), (4, 5, 17)),
        ((1, 3, 4), (3, 4, 5)),
        ((1, 18, 20), (6, 18, 20)),
        ((1, 3, 6), (3, 17, 18)),
        ((1, 2, 5), (3, 6, 17)),
        ((4, 17, 20), (15, 18, 19)),
        ((4, 18, 20), (6, 16, 17)),
        ((2, 6, 15), (2, 17, 19)),
        ((3, 4, 19), (3, 6, 20)),
        ((1, 6, 19), (3, 4, 16)),
        ((5, 14, 19), (14, 19, 20)),
        ((1, 3, 15), (14, 15, 17)),
        ((2, 5, 15), (2, 5, 15)),
        ((1, 2, 15), (1, 14, 20)),
        ((2, 16, 19), (6, 16, 19)),
        ((1, 3, 4), (4, 6, 19)),
        ((2, 6, 16), (16, 18, 20)),
        ((3, 18, 19), (4, 5, 15)),
        ((3, 5, 19), (5, 6, 17)),
        ((3, 15, 17), (14, 15, 16)),
        ((1, 16, 19), (4, 15, 17)),
        ((5, 14, 18), (5, 18, 19)),
        ((3, 4, 6), (14, 17, 18)),
        ((5, 14, 19), (15, 17, 20)),
    )
    for a, b in normalized_archive:
        A = [[[G.e]] + [[G.e, G.elements[t]] for t in a]]
        B = [[[G.e]] + [[G.e, G.elements[t]] for t in b]]
        hx, hz, _ = lifted_matrices(G, A, B)
        graph_keys.add(graph_signature(hx, hz))

    # Thirteen inverse-pair directions: six noncentral rotations and
    # seven order-four directions. The involution would create duplicate
    # component columns, and is therefore excluded.
    directions = ([G.elements[a] for a in range(1, 7)]
                  + [G.elements[14 + a] for a in range(7)])
    triples = list(combinations(range(len(directions)), 3))

    def inverse_key(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    # All Dic7 automorphisms x -> x^u, y -> x^v y.
    maps = []
    for u in range(14):
        if gcd(u, 14) != 1:
            continue
        for v in range(14):
            maps.append(tuple(
                inverse_key(((u * a + v * b) % 14, b))
                for a, b in directions
            ))

    def orbit_key(a, b):
        return min(tuple(sorted((
            tuple(sorted(mapping[t] for t in a)),
            tuple(sorted(mapping[t] for t in b))
        ))) for mapping in maps)

    def component(triple):
        return [[[G.e]] +
                [[G.e, directions[t]] for t in triple]]

    def generated_order(triple):
        """Group-theoretic card label, not a Tanner inequivalence claim."""
        reached, frontier = {G.e}, [G.e]
        generators = [directions[t] for t in triple]
        while frontier:
            value = frontier.pop()
            for generator in generators:
                following = G.mul(value, generator)
                if following not in reached:
                    reached.add(following)
                    frontier.append(following)
        return len(reached)

    subgroup_orders = [generated_order(triple) for triple in triples]
    local = random.Random(BASE_SEED ^ 0xD7C1404)
    proposals = list(combinations_with_replacement(range(len(triples)), 2))
    local.shuffle(proposals)
    buckets = {}
    for ia, ib in proposals:
        profile = tuple(sorted((subgroup_orders[ia], subgroup_orders[ib])))
        buckets.setdefault(profile, []).append((ia, ib))

    # Deliberately interleave proper/proper, proper/full and full/full
    # constituent prescriptions before full-graph novelty screening.
    profiles = sorted(buckets, key=lambda p: (
        not (p[0] < G.q == p[1]), p
    ))
    plans = [
        buckets[profile][turn]
        for turn in range(max(map(len, buckets.values())))
        for profile in profiles if turn < len(buckets[profile])
    ]
    orbit_keys = set()
    accepted = 0

    for ia, ib in plans:
        ta, tb = triples[ia], triples[ib]
        orbit = orbit_key(ta, tb)
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)

        A, B = component(ta), component(tb)
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 476 or max(row.bit_count() for row in hx + hz) > min(W_MAX, 9):
            continue
        signature = graph_signature(hx, hz)
        if signature in graph_keys:
            continue
        if not (_dic_excludes_two(hx, n)
                and _dic_excludes_two(hz, n)):
            continue

        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if k != 9 * G.q or 1000 * k <= 525 * n:
            continue

        # Validate the explicit witness against the evaluator's exact
        # matrices; do not infer quantum distance from a component graph.
        witness = _dic_three_witness(G, B)
        if witness.bit_count() != 3:
            continue
        if any((row & witness).bit_count() & 1 for row in hz):
            continue
        if gf2_rank(hx + [witness]) != rx + 1:
            continue

        graph_keys.add(signature)
        if accepted == idx:
            if not css_ok(hx, hz):
                raise ValueError("verified lift failed CSS screening")
            return {"A": A, "B": B, "k": k,
                    "distance_certificate": 3,
                    "directions": (ta, tb), "witness": witness,
                    "subgroup_card": tuple(sorted((
                        subgroup_orders[ia], subgroup_orders[ib]))),
                    "joint_incidence_key": signature.hex()}
        accepted += 1

    raise ValueError("archive-disjoint certified catalogue exhausted")


def _tri_basis(rows):
    basis = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            old = basis.get(p)
            if old is None:
                basis[p] = word
                break
            word ^= old
    return basis


def _tri_reduce(word, basis):
    for p in sorted(basis, reverse=True):
        if (word >> p) & 1:
            word ^= basis[p]
    return word


def _tri_rows(G, grid, side):
    """Extract only from the scaffold's verified construction."""
    dummy = [[[G.e]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, dummy)
    else:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in rows[:len(grid) * G.q]]


def _tri_six_count(rows, width):
    """Exclude kernel weights <=5; count every weight-six support.

    After exclusion through five, colliding triples are disjoint.
    Every six-support has ten unordered triple/triple partitions.
    """
    cols = _dic_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return None
    pairs = set()
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs.add(syndrome)
    triples = {}
    collisions = 0
    for i, first in enumerate(cols):
        for j in range(i):
            pair = first ^ cols[j]
            for h in range(j):
                syndrome = pair ^ cols[h]
                if syndrome in pairs:
                    return None
                previous = triples.get(syndrome, 0)
                collisions += previous
                triples[syndrome] = previous + 1
    if collisions % 10:
        raise ValueError("weight-six partition count is inconsistent")
    return collisions // 10


def _tri_quantum_four(checks, stabilizers, width):
    """Exact logical exclusion through four; short stabilizers are legal."""
    basis = _tri_basis(stabilizers)
    residues = [_tri_reduce(1 << j, basis) for j in range(width)]
    cols = _dic_columns(checks, width)
    seen = {0: 0}
    for j, syndrome in enumerate(cols):
        residue = residues[j]
        if syndrome in seen:
            if seen[syndrome] != residue:
                return False
        else:
            seen[syndrome] = residue
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            residue = residues[i] ^ residues[j]
            if syndrome in seen:
                if seen[syndrome] != residue:
                    return False
            else:
                seen[syndrome] = residue
    return True


def _tri_probe(rows, width, local, passes, seed=0, own=None):
    """Return an actual kernel/logical witness, not an unresolved cutoff."""
    cols = _dic_columns(rows, width)
    span = _tri_basis(own) if own is not None else None
    best = seed
    order = list(range(width))

    def inspect(word):
        nonlocal best
        if not word or (best and word.bit_count() >= best.bit_count()):
            return
        if span is not None and not _tri_reduce(word, span):
            return
        best = word

    if seed:
        if any((seed & row).bit_count() & 1 for row in rows):
            raise ValueError("probe seed does not commute")
        if span is not None and not _tri_reduce(seed, span):
            raise ValueError("probe seed is a stabilizer")

    for _ in range(passes):
        local.shuffle(order)
        pivots, nulls = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= old[0]
                word ^= old[1]
            if not syndrome:
                nulls.append(word)
                inspect(word)

        short = sorted(nulls, key=int.bit_count)[:24]
        for word in nulls:
            for other in short:
                trial = word ^ other
                if trial and trial.bit_count() < word.bit_count():
                    word = trial
            inspect(word)
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                inspect(pair)
                if i < 18:
                    for h in range(j):
                        inspect(pair ^ short[h])
        for _ in range(2):
            start = best
            if not start:
                break
            for i, first in enumerate(short):
                inspect(start ^ first)
                for second in short[:i]:
                    inspect(start ^ first ^ second)
            if best == start:
                break
        if best and best.bit_count() <= (6 if own is None else 5):
            break

    if best:
        assert all((best & row).bit_count() % 2 == 0 for row in rows)
        assert span is None or _tri_reduce(best, span)
    return best


def _tri_graph_key(G, hx, hz, width):
    """Intrinsic full-Tanner invariant, with no X/Z check colouring.

    Equality is inconclusive and discarded conservatively. Central
    translation reduces root work only after verification on actual rows.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _dic_columns(rows, width)
    m = len(rows)
    adjacency = [[] for _ in range(m + width)]
    for j, column in enumerate(cols):
        vertex = m + j
        while column:
            bit = column & -column
            check = bit.bit_length() - 1
            adjacency[check].append(vertex)
            adjacency[vertex].append(check)
            column ^= bit
    tags = [(int(v >= m), len(ns))
            for v, ns in enumerate(adjacency)]
    highest = max(c.bit_count() for c in cols)
    roots = [j for j, c in enumerate(cols) if c.bit_count() == highest]
    multiplicity = 1

    # The Q8 central involution commutes with both evaluator actions.
    z = next(g for g in G.elements
             if g != G.e and G.mul(g, g) == G.e)
    move = [G.idx(G.mul(z, g)) for g in G.elements]

    def translate(row):
        out = 0
        while row:
            bit = row & -row
            block, h = divmod(bit.bit_length() - 1, G.q)
            out |= 1 << (block * G.q + move[h])
            row ^= bit
        return out

    if Counter(map(translate, rows)) == Counter(rows):
        roots = [j for j in roots
                 if j % G.q < move[j % G.q]]
        multiplicity = 2

    def refinement(root):
        initial = [(tag, int(v == root))
                   for v, tag in enumerate(tags)]
        palette = {f: i for i, f in enumerate(sorted(set(initial)))}
        colours = [palette[f] for f in initial]
        history = [tuple(sorted(Counter(initial).items()))]
        for _ in range(5):
            features = [
                (colours[v], tuple(sorted(colours[w] for w in ns)))
                for v, ns in enumerate(adjacency)]
            history.append(tuple(sorted(Counter(features).items())))
            palette = {f: i for i, f in enumerate(sorted(set(features)))}
            following = [palette[f] for f in features]
            stable = len(set(following)) == len(set(colours))
            colours = following
            if stable:
                break
        return sha256(repr(history).encode("ascii")).digest()

    rooted = []
    for j in roots:
        rooted.extend([refinement(m + j)] * multiplicity)
    intersections = Counter(
        (min(a.bit_count(), b.bit_count()),
         max(a.bit_count(), b.bit_count()),
         (a & b).bit_count())
        for i, a in enumerate(rows) for b in rows[:i])
    description = (
        width, tuple(sorted(Counter(tags).items())),
        tuple(sorted(intersections.items())),
        refinement(-1), tuple(sorted(rooted)))
    return sha256(repr(description).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _tri_portfolio(G):
    """Deterministic immutable optimized portfolio, independent of slot order."""
    from itertools import permutations
    from collections import Counter

    if G.q != 8 or A_SHAPE != (3, 7) or B_SHAPE != (3, 7):
        raise ValueError("triangle portfolio requires Dic2 and (3,7) sides")
    q, width, cap = G.q, 7 * G.q, min(W_MAX, 9)
    local = random.Random(BASE_SEED ^ 0xD237C681)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def support_key(profile):
        return min(
            tuple(sorted(tuple(profile[i][j] for i in order)
                         for j in range(7)))
            for order in permutations(range(3)))

    # Three two-term edges, one per row pair, and four triple hyperedges.
    seed_profile = (
        (1, 1, 0, 1, 1, 1, 1),
        (1, 0, 1, 1, 1, 1, 1),
        (0, 1, 1, 1, 1, 1, 1),
    )
    profiles, keys = [seed_profile], {support_key(seed_profile)}
    double_types = sorted(set(permutations((1, 1, 0)))
                          | set(permutations((2, 0, 0))))
    triple_types = sorted(set(permutations((2, 1, 0)))
                          | set(permutations((3, 0, 0)))
                          | {(1, 1, 1)})

    # Enumerate genuinely different support structures before voltage
    # optimization. Every retained row totals six and column totals 2/3.
    for attempt in range(32768):
        columns = [local.choice(double_types) for _ in range(3)]
        columns += [local.choice(triple_types) for _ in range(4)]
        profile = tuple(tuple(c[i] for c in columns) for i in range(3))
        if any(sum(row) != 6 for row in profile):
            continue
        key = support_key(profile)
        if key not in keys:
            keys.add(key)
            profiles.append(profile)
        if len(profiles) == 24:
            break

    def draw(profile, policy):
        grid = [[[] for _ in range(7)] for _ in range(3)]
        for j in range(7):
            occupied = [i for i in range(3) if profile[i][j]]
            anchor_row = occupied[0]
            for i in occupied:
                count = profile[i][j]
                labels = local.sample(range(q), count)
                if i == anchor_row:
                    # Fix one column anchor while retaining relative terms.
                    anchor = G.elements[labels[0]]
                    labels = [G.idx(G.mul(G.inv(anchor), G.elements[t]))
                              for t in labels]
                grid[i][j] = [G.elements[t] for t in sorted(labels)]

        if profile == seed_profile and policy < 3:
            # Prescribed triangle holonomy, not a translated saved code.
            quartics = [g for g in G.elements if _dic_order(G, g) == 4]
            g, h = local.sample(quartics, 2)
            if policy == 0:
                t = local.choice(quartics)
            elif policy == 1:
                t = G.mul(g, h)
            else:
                t = G.inv(g)
            grid[0][0], grid[1][0] = [G.e], [g]
            grid[0][1], grid[2][1] = [G.e], [h]
            grid[1][2], grid[2][2] = [G.e], [t]
        return grid

    # Each support/voltage-policy bucket undergoes the same search budget.
    buckets = [[] for _ in range(len(profiles) * 4)]
    fillings = set()
    for attempt in range(max(8192, 192 * len(buckets))):
        card = attempt % len(buckets)
        pi, policy = divmod(card, 4)
        grid = draw(profiles[pi], policy)
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        counts = [[len(cell) for cell in row] for row in grid]
        if any(sum(row) != 6 for row in counts):
            raise ValueError("balanced row budget changed")
        if max(sum(row[j] for row in counts) for j in range(7)) > 3:
            raise ValueError("reinforcement column budget changed")
        if not _no_dead_lines(counts, counts):
            continue
        if _grid_weights(counts, counts) > cap:
            continue

        left = _tri_rows(G, grid, "A")
        right = _tri_rows(G, grid, "B")
        if gf2_rank(left) != 24 or gf2_rank(right) != 24:
            continue
        six_left = _tri_six_count(left, width)
        if six_left is None:
            continue
        six_right = _tri_six_count(right, width)
        if six_right is None:
            continue

        wl = _tri_probe(left, width, local, 3)
        wr = _tri_probe(right, width, local, 3)
        if not wl or not wr:
            continue
        # A counted six-word is evidence, even when probing misses it.
        dl = min(wl.bit_count(), 6 if six_left else width)
        dr = min(wr.bit_count(), 6 if six_right else width)
        quality = (
            min(dl, dr, 8),
            int(six_left == six_right == 0),
            -max(six_left, six_right),
            -six_left - six_right,
            -_four_cycles(left) - _four_cycles(right))
        entry = (quality, attempt, freeze(grid), wl, wr,
                 six_left, six_right, card)
        buckets[card].append(entry)
        buckets[card].sort(key=lambda e: (e[0], -e[1]), reverse=True)
        del buckets[card][2:]

    raw = sorted((e for bucket in buckets for e in bucket),
                 key=lambda e: (e[0], -e[1]), reverse=True)
    if not raw:
        raise ValueError("no kernel-five-clean Dic2 constituent")

    bank = []
    for quality, attempt, grid, wl, wr, sl, sr, card in raw[:64]:
        wl = _tri_probe(_tri_rows(G, grid, "A"),
                        width, local, 16, wl)
        wr = _tri_probe(_tri_rows(G, grid, "B"),
                        width, local, 16, wr)
        dl = min(wl.bit_count(), 6 if sl else width)
        dr = min(wr.bit_count(), 6 if sr else width)
        bank.append((min(dl, dr, 8), grid, wl, wr, sl, sr, card, attempt))
    bank.sort(key=lambda e: (-e[0], max(e[4], e[5]),
                             e[4] + e[5], e[6], e[7]))

    plans = [(a, b) for a in range(len(bank))
             for b in range(a, len(bank))]
    plans.sort(key=lambda ab: (
        -min(bank[ab[0]][0], bank[ab[1]][0]),
        max(bank[ab[0]][4], bank[ab[1]][5]),
        bank[ab[0]][6] == bank[ab[1]][6], ab))

    def embedded_seed(source, kind, checks, own):
        span = _tri_basis(own)
        seeds = []
        for block in range(7):
            if kind == "X":
                word = source << (block * 7 * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 7 + block) * q + h)
                    bits ^= bit
            if any((word & row).bit_count() & 1 for row in checks):
                raise ValueError("verified constituent embedding failed")
            if _tri_reduce(word, span):
                seeds.append(word)
        return min(seeds, key=int.bit_count) if seeds else 0

    results, graph_keys = [], set()
    tested = 0
    for a, b in plans:
        aa, bb = bank[a], bank[b]
        A, B = aa[1], bb[1]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if _grid_weights(sa, sb) > cap or not _no_dead_lines(sa, sb):
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 464 or not (275 * n <= 1000 * k < 325 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        if not _tri_quantum_four(hz, hx, n):
            continue
        if not _tri_quantum_four(hx, hz, n):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified three-row pair failed CSS")

        sx = embedded_seed(bb[3], "X", hz, hx)
        sz = embedded_seed(aa[2], "Z", hx, hz)
        if not sx or not sz:
            continue
        wx = _tri_probe(hz, n, local, 4, sx, hx)
        wz = _tri_probe(hx, n, local, 4, sz, hz)
        reading = min(wx.bit_count(), wz.bit_count())
        tested += 1

        signature = _tri_graph_key(G, hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        quality = (k * min(reading, 8)**2 / n,
                   min(reading, 8), k, -max(aa[4], bb[5]))
        screen = (
            ("k", k),
            ("logical_witness_upper_bound", reading),
            ("logical_witnesses", (wx, wz)),
            ("component_six_counts", (aa[4], bb[5])),
            ("component_cards", (aa[6], bb[6])),
            ("complete_logical_exclusion_through", 4),
            ("tanner_key", signature.hex()),
        )
        results.append((quality, A, B, screen))
        if len(results) >= N_SLOTS + 12:
            break
        if tested >= max(160, 8 * N_SLOTS) and len(results) >= N_SLOTS:
            break

    results.sort(key=lambda e: e[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in results[:N_SLOTS])


def NEW_CTX(rng, G, idx):
    """Pure indexed selection from the jointly optimized immutable portfolio."""
    entries = _tri_portfolio(G)
    if not 0 <= idx < len(entries):
        raise ValueError("distinct target-cell triangle portfolio exhausted")
    A, B, screen = entries[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell of the new three-row shape."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: The parent's rolled uniform (2,4)x(3,5) policy avoids weight-6 logicals: all five measured slots had d30s=6, with Z witnesses across sector-1 Acol0..3.
# REFUTED: Distinct relative-order cards provide distinct Dic7 codes: the measured sixty-slot parent collapsed to thirteen distinct codes, with multiple primary cards marked iso_dup.
# REFUTED: The local check-intersection direction catalogue supplies twenty evaluator-distinct Dic7 designs: the current sixty-slot evaluation retained fourteen classes and forty-six iso_dup slots, with all five measured distances remaining three.
# REFUTED: Complete-incidence and joint-projection refinement supplies twenty evaluator-distinct Dic7 pivot/binomial designs: the current sixty-slot evaluation retained nine classes; all five measured finalists remained k=252,d30s=3.
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
