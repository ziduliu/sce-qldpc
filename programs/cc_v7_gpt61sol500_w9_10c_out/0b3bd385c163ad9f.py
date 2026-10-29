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
CONCEPT = ("Generalized quaternion Q32, order 32, A=(1,2), B=(2,5): independent optimization of noncommuting cubic A constituents and saturated two-row B constituents should supply five novel measured distances >=10 at exact rate 0.250; this is a falsifiable measurement hypothesis.")
GROUP = ("Keep Q32 and CUSTOM_GROUP byte-for-byte. The new factor twelve gives legal n=384 at the existing order 32, so no order repair is needed. Reported Q32 one-row pivot/binomial finalists all measured distance three; that result validates neither a distance ceiling nor a distance advantage on this new bed.")
SHAPES = ("Replace equal (1,4) sides by unequal (1,2)x(2,5): constituent lengths/check counts are 64/32 and 160/64, n=384 and kb=96. Require both constituents and the complete matrices to have full row rank, retaining k=96 and rate 0.250. This deliberately targets the weaker 0.250 incumbent rather than the assigned high-rate plateau. Parent minima were exclusively sector-1 Acol/Bcol words, especially involving monomial columns; remove its monomial A pivot and introduce a two-row B geometry.")
FILLING = ("A has two noncommuting trinomials, row weight six and column weights three. B explores canonical profiles with column degrees (3,3,2,2,2) and both row totals six. Thus BOTH CSS maxima are nine, also below twelve. Optimize the two constituents independently using verified lifts, exact kernel exclusion through four, randomized information sets and nonincreasing actual kernel witnesses. Pair equally refined finalists, then screen ranks, CSS and complete kernels through four and search actual nontrivial full-code words, including mixed sectors and stabilizer reductions. Distinct intrinsic full-Tanner invariants separate immutable catalogue entries. Distance-ten survival and unseen-archive novelty remain evaluator-tested hypotheses; higher probes are upper bounds, not certificates.")
FAMILY_NAME = "r025_quaternion32_12x25_cubic_saturated_independent_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 32

def custom_ops(q):
    if q != 32:
        raise ValueError("Q32 has order 32")
    # x^16=1, y^2=x^8, y*x*y^-1=x^-1; element x^a y^b.
    els = [(a, b) for b in range(2) for a in range(16)]

    def red(g):
        a, b = g
        return ((a + 8 * (b // 2)) % 16, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 8 * b * d) % 16,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 16, 0) if b == 0 else ((a + 8) % 16, 1)

    def idx(g):
        a, b = red(g)
        return 16 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _generator_cards(G):
    """Enumerate inverse-normalized generating triples, not label draws."""
    from itertools import combinations

    # One representative of each inverse pair, excluding e and x^8.
    representatives = [(a, 0) for a in range(1, 8)]
    representatives += [(a, 1) for a in range(8)]

    def order(g):
        h, count = G.e, 0
        while True:
            h = G.mul(h, g)
            count += 1
            if h == G.e:
                return count

    cards = []
    for triple in combinations(representatives, 3):
        reached, frontier = {G.e}, [G.e]
        while frontier:
            h = frontier.pop()
            for g in triple:
                t = G.mul(h, g)
                if t not in reached:
                    reached.add(t)
                    frontier.append(t)
        if len(reached) == G.q:
            cards.append(triple)
    cards.sort(key=lambda t: (tuple(sorted(order(g) for g in t)),
                              sum(g[1] for g in t), t))
    return cards


def _unit_binomial_cells(G, triple):
    return [[[G.e]] + [[G.e, g] for g in triple]]


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


def _check_columns(rows, n):
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _no_weight_two_kernel(rows, n):
    """Exact exclusion of every nonzero kernel vector of weight <=2."""
    columns = _check_columns(rows, n)
    seen = set(columns)
    return 0 not in seen and len(seen) == n


def _degree_signature(hx, hz, n):
    """Additional intrinsic Tanner invariants; no X/Z colouring assumed.

    Project only degree-three qubits onto checks. Quibit degree is
    invariant under any Tanner relabeling, so different signatures
    certify non-isomorphic full graphs.
    """
    rows = hx + hz
    columns = _check_columns(rows, n)
    degree_counts = {}
    adj = [0] * len(rows)
    for c in columns:
        degree = c.bit_count()
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
        if degree == 3:
            bits = c
            while bits:
                bit = bits & -bits
                adj[bit.bit_length() - 1] |= c ^ bit
                bits ^= bit

    profiles = []
    for i, neighbors in enumerate(adj):
        histogram = {}
        for j, other in enumerate(adj):
            if i == j:
                continue
            key = (((neighbors >> j) & 1),
                   other.bit_count(),
                   (neighbors & other).bit_count(),
                   rows[j].bit_count())
            histogram[key] = histogram.get(key, 0) + 1
        profiles.append((rows[i].bit_count(), neighbors.bit_count(),
                         tuple(sorted(histogram.items()))))
    return (tuple(sorted(degree_counts.items())), tuple(sorted(profiles)))


def _verify_three_witness(G, B, hx, hz):
    """Verify a sector-1 X logical of weight three in the actual lift."""
    q = G.q
    g = B[0][1][1]
    # B=[1, 1+g, ...]. Its right-lifted binomial column at h=e
    # is corrected by the two unit-column qubits at e and g^-1.
    word = ((1 << (q + G.idx(G.e)))
            ^ (1 << G.idx(G.e))
            ^ (1 << G.idx(G.inv(g))))
    if word.bit_count() != 3:
        return False
    if any((word & row).bit_count() % 2 for row in hz):
        return False

    pivots = {}
    for row in hx:
        while row:
            p = row.bit_length() - 1
            if p in pivots:
                row ^= pivots[p]
            else:
                pivots[p] = row
                break
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _q32_retired_mixed_reference(rng, G, idx):
    """Inactive historical reference; never publishes the new catalogue."""
    from itertools import combinations

    assert G.q == 32
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    # Enumerate all inverse-class triples, excluding the identity and
    # unique involution. The first component is proper, the second full:
    # neither the generating/generating nor common-subgroup archive rule.
    directions = (tuple((a, 0) for a in range(1, 8))
                  + tuple((a, 1) for a in range(8)))
    cards = list(combinations(directions, 3))
    generating = set(_generator_cards(G))
    proper_ids = [i for i, triple in enumerate(cards)
                  if triple not in generating]
    full_ids = [i for i, triple in enumerate(cards)
                if triple in generating]
    plans = [(a, b) for a in proper_ids for b in full_ids]
    if not proper_ids or not full_ids:
        raise ValueError("mixed component prescriptions unexpectedly absent")

    # All Q32 automorphisms: x -> x^u, y -> x^v*y, u odd.
    # Inverting a binomial direction and permuting columns are gauges.
    actions = [(u, v) for u in range(1, 16, 2) for v in range(16)]

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    images = []
    for triple in cards:
        images.append([
            tuple(sorted(inverse_class(
                ((u * a + v * b) % 16, b))
                for a, b in triple))
            for u, v in actions
        ])

    def orbit_key(a, b):
        return min(tuple(sorted((x, y)))
                   for x, y in zip(images[a], images[b]))

    def coset_signature(hx, hz, n):
        """Intrinsic invariants of the actual uncoloured Tanner graph.

        Check projection preserves full-graph connectivity because every
        qubit has positive degree. Different signatures certify
        non-isomorphism; equal signatures are discarded conservatively.
        """
        rows = list(hx) + list(hz)
        m = len(rows)
        columns = _check_columns(rows, n)
        relations = {}
        adjacency = [0] * m

        # These projections use intrinsic full-graph qubit degrees,
        # never protograph labels or an assumed X/Z check colouring.
        # A Tanner isomorphism preserves every projection separately.
        degree_relations = {}
        for column in columns:
            degree = column.bit_count()
            if not degree:
                raise ValueError("isolated qubit in certified catalogue")
            relation = degree_relations.setdefault(degree, [0] * m)
            bits = column
            while bits:
                bit = bits & -bits
                relation[bit.bit_length() - 1] |= column ^ bit
                bits ^= bit

        for i, row in enumerate(rows):
            for j in range(i):
                overlap = (row & rows[j]).bit_count()
                if not overlap:
                    continue
                if overlap not in relations:
                    relations[overlap] = [0] * m
                relation = relations[overlap]
                relation[i] |= 1 << j
                relation[j] |= 1 << i
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

        def neighbors(bits, relation):
            out = 0
            while bits:
                bit = bits & -bits
                out |= relation[bit.bit_length() - 1]
                bits ^= bit
            return out

        components = []
        remaining = (1 << m) - 1
        while remaining:
            visited = frontier = remaining & -remaining
            while frontier:
                frontier = neighbors(frontier, adjacency) & ~visited
                visited |= frontier
            components.append(visited)
            remaining &= ~visited

        # Do not filter connectivity here: this same invariant must
        # also describe disconnected archive references faithfully.
        # Candidate connectivity is checked after signature extraction.
        component_profiles = []
        for mask in components:
            degrees = {}
            for column in columns:
                if column & mask:
                    degree = column.bit_count()
                    degrees[degree] = degrees.get(degree, 0) + 1
            component_profiles.append(
                (mask.bit_count(), tuple(sorted(degrees.items()))))

        def spheres(root, relation):
            visited = frontier = 1 << root
            counts = []
            while frontier:
                frontier = neighbors(frontier, relation) & ~visited
                if frontier:
                    counts.append(frontier.bit_count())
                    visited |= frontier
            return tuple(counts)

        kinds = sorted(relations)
        root_profiles = []
        for root in range(m):
            root_profiles.append((
                rows[root].bit_count(),
                spheres(root, adjacency),
                tuple((kind, spheres(root, relations[kind]))
                      for kind in kinds),
                tuple((degree, spheres(root, degree_relations[degree]))
                      for degree in sorted(degree_relations)),
            ))

        return (
            tuple(sorted(component_profiles)),
            tuple(sorted(root_profiles)),
            _tanner_signature(hx, hz),
            _degree_signature(hx, hz, n),
        )

    # Exclusion references only: twelve generating/generating codes
    # from the direct-parent table, then twelve common-subgroup codes
    # from the current table. Each packet contains three A directions
    # followed by three B directions in the frozen element ordering.
    archived = (
        (4,18,19, 16,17,21),
        (20,21,22, 4,5,20),
        (4,5,17, 7,16,18),
        (3,17,21, 1,3,22),
        (7,17,18, 19,20,21),
        (2,7,16, 1,20,23),
        (1,7,21, 1,6,17),
        (1,16,21, 1,16,21),
        (7,20,22, 7,20,22),
        (18,19,23, 1,19,23),
        (4,22,23, 1,3,20),
        (1,17,23, 3,22,23),
        (2,4,6, 3,4,7),
        (3,5,6, 5,6,7),
        (2,6,22, 2,16,20),
        (4,18,20, 6,16,18),
        (1,2,5, 3,4,7),
        (4,20,22, 4,20,22),
        (2,6,16, 2,6,22),
        (2,4,22, 6,16,22),
        (1,2,5, 1,2,5),
        (2,16,20, 2,18,22),
        (2,6,18, 6,16,18),
        (4,18,22, 6,16,20),
    )
    graph_keys = set()
    for packet in archived:
        A = _unit_binomial_cells(
            G, tuple(G.elements[t] for t in packet[:3]))
        B = _unit_binomial_cells(
            G, tuple(G.elements[t] for t in packet[3:]))
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(coset_signature(hx, hz, n))

    # Shuffle an exhaustive list of mixed structural prescriptions,
    # not independently sampled completed codes. Every slot rebuilds
    # the same prefix with the same archive and within-program exclusions.
    local = random.Random(BASE_SEED ^ 0xC032A3)
    plans = sorted(plans)
    local.shuffle(plans)
    orbit_keys = set()
    accepted = 0

    for ia, ib in plans:
        orbit = orbit_key(ia, ib)
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)

        A = _unit_binomial_cells(G, cards[ia])
        B = _unit_binomial_cells(G, cards[ib])
        hx, hz, n = lifted_matrices(G, A, B)
        if max(row.bit_count() for row in hx + hz) > min(W_MAX, 9):
            continue
        if not (_no_weight_two_kernel(hx, n)
                and _no_weight_two_kernel(hz, n)):
            continue

        signature = coset_signature(hx, hz, n)
        if len(signature[0]) != 1 or signature in graph_keys:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 544 or k != 288 or 1000 * k <= 525 * n:
            continue
        if not _verify_three_witness(G, B, hx, hz):
            continue

        graph_keys.add(signature)
        if accepted == idx:
            if not css_ok(hx, hz):
                raise ValueError("verified lift failed CSS screening")
            return {
                "A": A, "B": B, "screen_k": k,
                "distance_certificate": 3,
                "component_profile": signature[0],
                "component_prescription": ("proper-subgroup", "generating"),
                "reported_archive_references": len(archived),
                "directions": (cards[ia], cards[ib]),
            }
        accepted += 1

    raise ValueError("archive-screened mixed distance-three catalogue exhausted")


from functools import lru_cache


@lru_cache(maxsize=1)
def _q32_remaining_portfolio(G):
    """Immutable residual catalogue; selected matrices ignore call order.

    Memoization stores only the deterministic immutable result. Unequal
    intrinsic signatures certify different Tanner graphs; equality is
    discarded conservatively, not treated as an isomorphism certificate.
    """
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations, combinations_with_replacement

    if G.q != 32:
        raise ValueError("residual catalogue requires frozen Q32")
    support = [[1, 2, 2, 2]]
    cap = min(W_MAX, 9)
    if not _no_dead_lines(support, support):
        raise ValueError("saturated profile has a dead line")
    if _grid_weights(support, support) > cap:
        raise ValueError("saturated profile exceeds both CSS budgets")

    directions = tuple(range(1, 8)) + tuple(range(16, 24))
    cards = list(combinations(directions, 3))

    def component(labels):
        return _unit_binomial_cells(
            G, tuple(G.elements[label] for label in labels))

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def component_threes(labels, side):
        """Count every weight-three word in the VERIFIED constituent.

        This is a multiplicity tie-break, not a quantum-distance estimate.
        A unit opposite side exposes the exact evaluator action.
        """
        grid, unit = component(labels), [[[G.e]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, unit)
        else:
            _, rows, _ = lifted_matrices(G, unit, grid)
        width = 4 * G.q
        mask = (1 << width) - 1
        rows = [row & mask for row in rows[:G.q]]
        cols = _check_columns(rows, width)
        if 0 in cols or len(set(cols)) != width:
            return None
        positions = {syndrome: i for i, syndrome in enumerate(cols)}
        count = 0
        for i, first in enumerate(cols):
            for j in range(i):
                k = positions.get(first ^ cols[j], -1)
                # Exactly one partition of each triple has k > i > j.
                if k > i:
                    count += 1
        return count

    def graph_key(hx, hz, n):
        """Intrinsic invariant of the complete UNCOLOURED Tanner graph.

        Degree fourteen identifies the high-degree qubit class without
        sector labels. Checks meeting two such qubits become weighted
        multigraph edges; checks meeting one become intrinsic markers.
        Refinement and walks retain multiplicities and check weights.
        """
        rows = list(hx) + list(hz)
        cols = _check_columns(rows, n)
        high = [j for j, col in enumerate(cols)
                if col.bit_count() == 14]
        if len(high) != G.q:
            raise ValueError("saturated intrinsic degree class changed")

        size = len(high)
        position = {j: i for i, j in enumerate(high)}
        mask = sum(1 << j for j in high)
        edges = [Counter() for _ in high]
        markers = [Counter() for _ in high]

        for row in rows:
            bits = row & mask
            degree = bits.bit_count()
            first = bits & -bits
            weight = row.bit_count()
            if degree == 1:
                markers[position[first.bit_length() - 1]][weight] += 1
            elif degree == 2:
                u = position[first.bit_length() - 1]
                v = position[(bits ^ first).bit_length() - 1]
                edges[u][(v, weight)] += 1
                edges[v][(u, weight)] += 1
            else:
                raise ValueError("unexpected high-degree check projection")

        labels = [
            (tuple(sorted(markers[u].items())),
             tuple(sorted(Counter(
                 (weight, multiplicity)
                 for (_, weight), multiplicity in edges[u].items()
             ).items())))
            for u in range(size)
        ]

        roots = []
        for root in range(size):
            initial = [(labels[u], int(u == root))
                       for u in range(size)]
            palette = {feature: i for i, feature in
                       enumerate(sorted(set(initial)))}
            colors = [palette[feature] for feature in initial]
            histories = [tuple(sorted(Counter(initial).items()))]

            for step in range(10):
                features = [
                    (colors[u], tuple(sorted(
                        (weight, multiplicity, colors[v])
                        for (v, weight), multiplicity
                        in edges[u].items())))
                    for u in range(size)
                ]
                histories.append(tuple(sorted(Counter(features).items())))
                palette = {feature: i for i, feature in
                           enumerate(sorted(set(features)))}
                following = [palette[feature] for feature in features]
                stable = len(set(following)) == len(set(colors))
                colors = following
                if stable and step >= 2:
                    break

            vector = [0] * size
            vector[root] = 1
            walks = [[] for _ in high]
            for length in range(10):
                following = [0] * size
                for u, count in enumerate(vector):
                    if count:
                        for (v, _), multiplicity in edges[u].items():
                            following[v] += count * multiplicity
                vector = following
                for v, count in enumerate(vector):
                    walks[v].append(count)

            roots.append((
                tuple(histories),
                tuple(walks[root]),
                tuple(sorted((colors[v], labels[v], tuple(walks[v]))
                             for v in range(size))),
            ))

        intrinsic = (
            tuple(sorted(Counter(row.bit_count()
                                 for row in rows).items())),
            tuple(sorted(Counter(col.bit_count()
                                 for col in cols).items())),
            tuple(sorted(roots)),
            _tanner_signature(hx, hz),
            _degree_signature(hx, hz, n),
        )
        return sha256(repr(intrinsic).encode("ascii")).digest()

    # Exclusion references ONLY: the twenty-four previously shown Q32
    # classes, followed by all nine distinct classes in the current table.
    # Each packet is three A directions followed by three B directions.
    archived = (
        (4,18,19, 16,17,21),
        (20,21,22, 4,5,20),
        (4,5,17, 7,16,18),
        (3,17,21, 1,3,22),
        (7,17,18, 19,20,21),
        (2,7,16, 1,20,23),
        (1,7,21, 1,6,17),
        (1,16,21, 1,16,21),
        (7,20,22, 7,20,22),
        (18,19,23, 1,19,23),
        (4,22,23, 1,3,20),
        (1,17,23, 3,22,23),
        (2,4,6, 3,4,7),
        (3,5,6, 5,6,7),
        (2,6,22, 2,16,20),
        (4,18,20, 6,16,18),
        (1,2,5, 3,4,7),
        (4,20,22, 4,20,22),
        (2,6,16, 2,6,22),
        (2,4,22, 6,16,22),
        (1,2,5, 1,2,5),
        (2,16,20, 2,18,22),
        (2,6,18, 6,16,18),
        (4,18,22, 6,16,20),
        (4,21,23, 2,5,20),
        (2,5,7, 7,19,23),
        (2,18,22, 1,21,22),
        (4,16,18, 4,20,21),
        (4,6,16, 16,21,23),
        (16,20,22, 2,3,16),
        (4,6,17, 4,16,17),
        (4,5,7, 5,7,22),
        (4,16,20, 18,19,22),
    )
    blocked = set()
    for packet in archived:
        hx, hz, n = lifted_matrices(
            G, component(packet[:3]), component(packet[3:]))
        blocked.add(graph_key(hx, hz, n))

    # All Q32 automorphisms x -> x^u, y -> x^v*y, u odd.
    # Independent direction inversions and column permutations are gauges.
    actions = [(u, v) for u in range(1, 16, 2) for v in range(16)]

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    images = []
    for card in cards:
        transformed = []
        for u, v in actions:
            transformed.append(tuple(sorted(
                inverse_class(((u * a + v * b) % 16, b))
                for a, b in (G.elements[label] for label in card))))
        images.append(tuple(transformed))

    def orbit_key(ia, ib):
        return min(tuple(sorted((a, b)))
                   for a, b in zip(images[ia], images[ib]))

    counts_a = [component_threes(card, "A") for card in cards]
    counts_b = [component_threes(card, "B") for card in cards]
    plans = [
        (max(counts_a[a], counts_b[b]),
         counts_a[a] + counts_b[b], a == b, a, b)
        for a, b in combinations_with_replacement(range(len(cards)), 2)
        if counts_a[a] is not None and counts_b[b] is not None
    ]
    # Deliberate multiplicity-ordered structural prescriptions, not fresh
    # per-slot draws. No subgroup category or connectivity is assumed.
    plans.sort()

    entries, orbits = [], set()
    for maximum, total, _, ia, ib in plans:
        orbit = orbit_key(ia, ib)
        if orbit in orbits:
            continue
        orbits.add(orbit)

        A, B = component(cards[ia]), component(cards[ib])
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 544 or max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = graph_key(hx, hz, n)
        if signature in blocked:
            continue
        if not (_no_weight_two_kernel(hx, n)
                and _no_weight_two_kernel(hz, n)):
            continue

        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 288 or 1000 * k <= 525 * n:
            continue
        if not _verify_three_witness(G, B, hx, hz):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified residual pair failed CSS")

        blocked.add(signature)
        entries.append((
            freeze(A), freeze(B), cards[ia], cards[ib],
            counts_a[ia], counts_b[ib],
        ))
        if len(entries) >= N_SLOTS:
            break

    # Exhaustion is exposed, never filled with an archived or orbit copy.
    return tuple(entries)


@lru_cache(maxsize=1)
def _q32_cubic_saturated_portfolio(G, count):
    """Deterministic optimization; cached published cells are immutable.

    Historical (1,4) catalogues above are inactive. Every screening
    matrix here comes from the scaffold's verified construction.
    """
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations_with_replacement

    if G.q != 32:
        raise ValueError("new asymmetric bed requires Q32")
    cap = min(W_MAX, 9)
    q, e = G.q, G.e
    zero = [[[]]]

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def basis(rows):
        out = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = out.get(p)
                if old is None:
                    out[p] = word
                    break
                word ^= old
        return out

    def outside(word, own):
        while word:
            p = word.bit_length() - 1
            old = own.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def clear_four(rows, width):
        """Exactly exclude every nonzero kernel word of weight <=4."""
        cols = _check_columns(rows, width)
        seen = {0}
        for syndrome in cols:
            if syndrome in seen:
                return None
            seen.add(syndrome)
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                if syndrome in seen:
                    return None
                seen.add(syndrome)
        return cols

    def cycle_cost(rows):
        cost = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                overlap = (row & other).bit_count()
                cost += overlap * (overlap - 1) // 2
        return cost

    def component(grid, side):
        """Zero auxiliary cells expose the exact left/right actions."""
        if side == "A":
            rows, _, _ = lifted_matrices(G, grid, zero)
            width, wanted_rank = 2 * q, q
        else:
            _, rows, _ = lifted_matrices(G, zero, grid)
            width, wanted_rank = 5 * q, 2 * q
        mask = (1 << width) - 1
        rows = [row & mask for row in rows]
        if gf2_rank(rows) != wanted_rank:
            return None
        cols = clear_four(rows, width)
        if cols is None:
            return None
        return rows, cols, cycle_cost(rows)

    def probe(local, cols, passes, seed=0, own=None, stabilizers=()):
        """Search actual kernel words; unresolved readings are upper bounds."""
        width = len(cols)
        best = seed
        weight = seed.bit_count() if seed else width + 1

        def inspect(word):
            nonlocal best, weight
            w = word.bit_count()
            if not word or w >= weight:
                return
            if own is not None and not outside(word, own):
                return
            best, weight = word, w

        for _ in range(passes):
            order = list(range(width))
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

            # Basis exchanges preserve the whole kernel.
            local.shuffle(nulls)
            for i, word in enumerate(nulls):
                w = word.bit_count()
                for j, other in enumerate(nulls):
                    if i == j:
                        continue
                    trial = word ^ other
                    tw = trial.bit_count()
                    if 0 < tw < w:
                        word, w = trial, tw
                nulls[i] = word
                inspect(word)

            short = sorted(set(nulls), key=int.bit_count)[:32]
            if stabilizers:
                reduced = []
                for word in short:
                    w = word.bit_count()
                    for row in stabilizers:
                        trial = word ^ row
                        tw = trial.bit_count()
                        if 0 < tw < w:
                            word, w = trial, tw
                    inspect(word)
                    reduced.append(word)
                short = sorted(set(short + reduced), key=int.bit_count)[:32]

            for i, first in enumerate(short):
                for second in short[:i]:
                    inspect(first ^ second)
            tiny = short[:18]
            for i, first in enumerate(tiny):
                for j in range(i):
                    pair = first ^ tiny[j]
                    for third in tiny[:j]:
                        inspect(pair ^ third)

            if best:
                starting = best
                for first in short:
                    inspect(starting ^ first)
                if stabilizers:
                    for i, first in enumerate(stabilizers):
                        inspect(starting ^ first)
                        for second in stabilizers[:i]:
                            inspect(starting ^ first ^ second)
            if weight <= (7 if own is not None else 6):
                break

        if not best:
            raise ValueError("actual kernel witness absent")
        syndrome, bits = 0, best
        while bits:
            bit = bits & -bits
            syndrome ^= cols[bit.bit_length() - 1]
            bits ^= bit
        if syndrome:
            raise ValueError("kernel probe returned a noncommuting word")
        return weight, best

    # Canonical count designs: permutations of one profile are not
    # separate cards. The first profile is the measured related-bed core.
    profiles = [
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
    ]
    keys = {
        tuple(sorted((profiles[0][0][j], profiles[0][1][j])
                     for j in range(5)))
    }
    triples = ((0, 3), (1, 2), (2, 1), (3, 0))
    doubles = ((0, 2), (1, 1), (2, 0))
    for top in combinations_with_replacement(triples, 2):
        for tail in combinations_with_replacement(doubles, 3):
            columns = top + tail
            if sum(c[0] for c in columns) != 6:
                continue
            key = min(tuple(sorted(columns)),
                      tuple(sorted((b, a) for a, b in columns)))
            if key in keys:
                continue
            keys.add(key)
            profiles.append(tuple(tuple(c[i] for c in columns)
                                  for i in range(2)))

    nonidentity = [g for g in G.elements if g != e]

    def draw_a(local, card):
        row = []
        for j in range(2):
            for _ in range(128):
                g, h = local.sample(nonidentity, 2)
                if G.mul(g, h) == G.mul(h, g):
                    continue
                # Four genuinely different quotient prescriptions;
                # voltage optimization still happens within each card.
                kind = (card + j) % 4
                if kind == 0 and not (g[1] != h[1]):
                    continue
                if kind == 1 and not (g[1] == h[1] == 1):
                    continue
                if kind == 2 and not ((g[0] + h[0]) % 2):
                    continue
                row.append([e, g, h])
                break
            else:
                return None
        return [row]

    def draw_b(local, card):
        profile = profiles[card % len(profiles)]
        grid = [[[] for _ in range(5)] for _ in range(2)]
        for j in range(5):
            for i in range(2):
                size = profile[i][j]
                for _ in range(64):
                    cell = local.sample(G.elements, size)
                    if size == 2:
                        difference = G.mul(G.inv(cell[0]), cell[1])
                        if G.mul(difference, difference) == e:
                            continue
                    grid[i][j] = cell
                    break
                else:
                    return None
            # Fix only one common column anchor, retaining the relative
            # voltage between its two cells.
            anchor = next(cell[0] for cell in
                          (grid[0][j], grid[1][j]) if cell)
            inverse = G.inv(anchor)
            for i in range(2):
                grid[i][j] = [G.mul(inverse, g) for g in grid[i][j]]
        return grid

    def optimized_bank(side):
        bank, fillings = [], set()
        cards = 12 if side == "A" else max(12, len(profiles))
        trials = 112 if side == "A" else 192
        for card in range(cards):
            local = random.Random(
                BASE_SEED ^ (0x325120 + 104729 * card
                             + (1000003 if side == "B" else 0)))
            raw = []
            for trial in range(trials):
                grid = (draw_a(local, card) if side == "A"
                        else draw_b(local, card))
                if grid is None:
                    continue
                filling = freeze(grid)
                if filling in fillings:
                    continue
                fillings.add(filling)
                info = component(grid, side)
                if info is None:
                    continue
                rows, cols, cost = info
                raw.append((cost, trial, grid, rows, cols))
            raw.sort(key=lambda item: item[:2])

            shortlist = []
            for cost, trial, grid, rows, cols in raw[:12]:
                reading, witness = probe(local, cols, 3)
                shortlist.append(dict(
                    grid=grid, rows=rows, cols=cols, cost=cost,
                    card=card, trial=trial,
                    reading=reading, witness=witness))
            shortlist.sort(key=lambda r: (
                -r["reading"], r["cost"], r["trial"]))

            # Every retained constituent gets the same 3+18-pass budget.
            for record in shortlist[:3]:
                reading, witness = probe(
                    local, record["cols"], 18, record["witness"])
                record["reading"], record["witness"] = reading, witness
                bank.append(record)
        bank.sort(key=lambda r: (
            -r["reading"], r["cost"], r["card"], r["trial"]))
        if len(bank) < 3:
            raise ValueError("too few optimized kernel-four-clean constituents")
        return bank[:24]

    aa, bb = optimized_bank("A"), optimized_bank("B")

    def graph_key(hx, hz, n):
        """Intrinsic invariant of the complete uncoloured Tanner graph.

        Unequal keys certify different invariant descriptions. Equal
        keys are discarded conservatively, not declared isomorphic.
        """
        rows = hx + hz
        cols = _check_columns(rows, n)
        m = len(rows)
        adjacency = [[] for _ in range(m + n)]
        for j, col in enumerate(cols):
            bits = col
            while bits:
                bit = bits & -bits
                r = bit.bit_length() - 1
                adjacency[r].append(m + j)
                adjacency[m + j].append(r)
                bits ^= bit
        colors = [len(neighbors) for neighbors in adjacency]
        history = []
        for _ in range(6):
            features = [
                (colors[v], tuple(sorted(colors[t] for t in neighbors)))
                for v, neighbors in enumerate(adjacency)]
            history.append(tuple(sorted(Counter(features).items())))
            palette = {f: c for c, f in enumerate(sorted(set(features)))}
            colors = [palette[f] for f in features]
        intrinsic = (
            n, m,
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(history), _tanner_signature(hx, hz))
        return sha256(repr(intrinsic).encode("ascii")).digest()

    def full_reading(a, b, hx, hz, n, local):
        overall = n + 1
        for kind, checks, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            own = basis(stabilizers)
            best = 0
            weight = n + 1
            for block in range(2 if kind == "X" else 5):
                if kind == "X":
                    word = b["witness"] << (block * 5 * q)
                else:
                    source, word = a["witness"], 0
                    while source:
                        bit = source & -source
                        j, h = divmod(bit.bit_length() - 1, q)
                        word |= 1 << ((j * 5 + block) * q + h)
                        source ^= bit
                if any((row & word).bit_count() & 1 for row in checks):
                    raise ValueError("verified component embedding failed")
                if outside(word, own) and word.bit_count() < weight:
                    best, weight = word, word.bit_count()
            if not best:
                return None
            reading, witness = probe(
                local, _check_columns(checks, n), 2,
                best, own, stabilizers)
            if not outside(witness, own):
                raise ValueError("full-code witness is a stabilizer")
            overall = min(overall, reading)
        return overall

    plans = sorted(
        (-min(a["reading"], b["reading"]),
         -a["reading"] - b["reading"], a["cost"] + b["cost"], ai, bi)
        for ai, a in enumerate(aa) for bi, b in enumerate(bb))
    entries, signatures = [], set()
    for _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 384 or max(row.bit_count() for row in hx + hz) > cap:
            continue
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if rx != 160 or rz != 128 or k != 96:
            continue
        if clear_four(hx, n) is None or clear_four(hz, n) is None:
            continue
        signature = graph_key(hx, hz, n)
        if signature in signatures:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified asymmetric Q32 lift failed CSS")

        local = random.Random(BASE_SEED ^ (0x325D00 + 1009 * ai + bi))
        reading = full_reading(a, b, hx, hz, n, local)
        if reading is None:
            continue
        signatures.add(signature)
        quality = (k * reading * reading / n, reading,
                   -a["cost"] - b["cost"])
        entries.append((
            quality, freeze(A), freeze(B), k, reading,
            (a["card"], b["card"]),
            (a["reading"], b["reading"]), signature.hex()))
        if len(entries) >= max(count, 5) + 12:
            break
    entries.sort(key=lambda entry: entry[0], reverse=True)
    return tuple(entries[:count])


def NEW_CTX(rng, G, idx):
    """Publish a distinct, equally screened immutable optimized design."""
    entries = _q32_cubic_saturated_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("optimized asymmetric Q32 portfolio exhausted")
    quality, A, B, k, reading, cards, components, signature = entries[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "verified_kernel_floor": 5,
        "quantum_witness_upper_bound": reading,
        "component_witness_upper_bounds": components,
        "design_cards": cards, "quality": quality,
        "tanner_signature": signature,
        "catalogue_position": idx,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: The Q32 generating-triple catalogue's intersection and qubit-degree gates eliminate isomorphic repeats: its measured sixty slots collapsed to twelve distinct codes, with forty-eight iso_dup slots.
# REFUTED: The Q32 common-proper-subgroup coset catalogue supplies twenty inequivalent slots: its measured sixty slots retained twelve distinct codes and forty-eight iso_dup slots, with all five measured distances remaining three.
# REFUTED: Mixed proper-subgroup/generating triples plus the parent's intrinsic gates supply twenty inequivalent Q32 slots: the current sixty-slot evaluation retained nine distinct codes and fifty-one iso_dup slots, with all five measured distances remaining three.
# REFUTED: Exhaustive residual Q32 triple pairing with root/walk gates supplies twenty evaluator-distinct designs: sixty evaluated slots retained six distinct codes and fifty-four iso_dup slots, with all five measured distances remaining three.
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
