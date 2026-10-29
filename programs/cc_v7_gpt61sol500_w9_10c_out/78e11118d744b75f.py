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
CONCEPT = ("Generalized quaternion Q32, order 32, shapes (1,4)x(1,4): transverse proper component subgroups H,K with HK=Q32 should supply at least five archive-new inequivalent connected codes, preserving verified distance three and exact rate 9/17 in cell 0.550.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. The common-subgroup parent measured five k=288,d30s=3 codes but retained only twelve classes from sixty slots. Transverse proper/proper pairings differ from both its common-subgroup rule and the shown proper/generating rule; actual full-Tanner invariants, not subgroup names alone, screen archive novelty.")
SHAPES = ("Keep equal 128-variable, 32-check components: n=544, kb=288 and guaranteed rate 9/17=0.529412 in cell 0.550. The shape permits q<=35 while this round fixes q=32. Parent weight-three minima occupied sector-1 Acol/Bcol blocks broadly, especially Bcol0 for X and Acol0 for Z; none were reported solely in sector 2. Unit pivots retain exact k=288. No strict gain over the certified distance-three score 81/17 is asserted.")
FILLING = ("Retain [1,2,2,2] on both sides and BOTH check maxima nine. Enumerate all inverse-distinct proper-subgroup direction triples and pair only those whose subgroup product covers Q32. Remove common automorphism copies and exclude all thirty-six reported Q32 representatives. Intrinsic degree-fourteen joint walks, complete-incidence refinement, check intersections and degree-conditioned profiles conservatively separate final graphs. Verified lifts enforce connectivity, CSS, exact ranks, complete commuting-weight-one/two exclusion and a non-stabilizer weight-three witness. An immutable memoized catalogue changes computation cost only; unseen archive novelty and useful-slot yield remain measurement hypotheses.")
FAMILY_NAME = "r055_quaternion32_14x14_unitbinomial_d3_transverse_rootwalk_v4"

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

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (1, 4)               # (rb, cb)


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


def _retired_common_coset_context(rng, G, idx):
    """Inactive historical catalogue; never publishes the new slots."""
    from itertools import combinations, combinations_with_replacement

    assert G.q == 32
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    # Both are proper index-two normal subgroups of the frozen Q32.
    # Identity, the involution, and inverse-class repeats are excluded.
    domains = (
        tuple((a, 0) for a in range(1, 8)),       # <x> = C16
        tuple((a, 0) for a in (2, 4, 6))
        + tuple((a, 1) for a in (0, 2, 4, 6)),  # <x^2,y> = Q16
    )
    cards, positions, plans = [], {}, set()
    for domain in domains:
        ids = []
        for triple in combinations(domain, 3):
            if triple not in positions:
                positions[triple] = len(cards)
                cards.append(triple)
            ids.append(positions[triple])
        for a, b in combinations_with_replacement(ids, 2):
            plans.add(tuple(sorted((a, b))))

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

        # The reported parent has one component: both of its direction
        # triples generate Q32. Never emit one of those connected codes.
        if len(components) < 2:
            return None

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
            ))

        return (
            tuple(sorted(component_profiles)),
            tuple(sorted(root_profiles)),
            _tanner_signature(hx, hz),
            _degree_signature(hx, hz, n),
        )

    # Shuffle an exhaustive list of structural prescriptions, not
    # independently sampled completed fillings. Every slot rebuilds the
    # same accepted prefix and therefore shares its novelty exclusions.
    local = random.Random(BASE_SEED ^ 0xC032D3)
    plans = sorted(plans)
    local.shuffle(plans)
    orbit_keys, graph_keys = set(), set()
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
        if signature is None or signature in graph_keys:
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
                "directions": (cards[ia], cards[ib]),
            }
        accepted += 1

    raise ValueError("new disconnected distance-three catalogue exhausted")


from functools import lru_cache


@lru_cache(maxsize=1)
def _q32_transverse_portfolio(G, count):
    """Immutable catalogue of jointly screened structural prescriptions.

    Memoization affects build cost, not matrices or slot ownership.
    Unequal intrinsic keys certify Tanner non-isomorphism. Equal keys
    are discarded conservatively; equality is not an isomorphism test.
    """
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations

    if G.q != 32:
        raise ValueError("transverse portfolio requires frozen Q32")
    support = [[1, 2, 2, 2]]
    assert _no_dead_lines(support, support)
    assert _grid_weights(support, support) <= min(W_MAX, 9)

    def closure(triple):
        reached, frontier = {G.e}, [G.e]
        while frontier:
            h = frontier.pop()
            for g in triple:
                following = G.mul(h, g)
                if following not in reached:
                    reached.add(following)
                    frontier.append(following)
        return frozenset(G.idx(g) for g in reached)

    # All nonidentity, noninvolutory inverse classes. No independently
    # resampled completed code, repeated inverse direction or anchor gauge
    # is treated as a new prescription.
    directions = (
        tuple((a, 0) for a in range(1, 8))
        + tuple((a, 1) for a in range(8))
    )
    cards = []
    for triple in combinations(directions, 3):
        subgroup = closure(triple)
        if len(subgroup) < G.q:
            cards.append((triple, subgroup))
    if not cards:
        raise ValueError("proper component cards unexpectedly absent")

    # Transversality is tested algebraically, then connectivity is checked
    # again on the actual intrinsic projection of the verified matrices.
    products = {}
    plans = []
    for ia, (_, ha) in enumerate(cards):
        for ib in range(ia, len(cards)):
            hb = cards[ib][1]
            pair = (ha, hb)
            if pair not in products:
                products[pair] = len({
                    G.idx(G.mul(G.elements[a], G.elements[b]))
                    for a in ha for b in hb
                }) == G.q
            if products[pair]:
                plans.append((ia, ib))
    if not plans:
        raise ValueError("no transverse proper component pairings")

    actions = [(u, v) for u in range(1, 16, 2) for v in range(16)]

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    images = []
    for triple, _ in cards:
        images.append(tuple(
            tuple(sorted(inverse_class(
                ((u * a + v * b) % 16, b))
                for a, b in triple))
            for u, v in actions
        ))

    def orbit_key(ia, ib):
        return min(tuple(sorted((a, b)))
                   for a, b in zip(images[ia], images[ib]))

    def graph_key(hx, hz, n):
        """Intrinsic invariant of the complete uncoloured Tanner graph.

        Qubit degree fourteen identifies the contracted sector without
        CSS colours, group labels or protograph coordinates. Its parallel
        edges retain multiplicity. Complete-incidence refinement and the
        supplied full-check invariants supplement that contraction.
        """
        rows = list(hx) + list(hz)
        columns = _check_columns(rows, n)
        m = len(rows)
        high = [j for j, col in enumerate(columns)
                if col.bit_count() == 14]
        if len(high) != G.q:
            raise ValueError("unit-binomial degree class changed")
        position = {j: i for i, j in enumerate(high)}
        mask = sum(1 << j for j in high)
        adjacency = [[] for _ in high]
        markers = [0] * len(high)

        for row in rows:
            bits = row & mask
            first = bits & -bits
            if bits.bit_count() == 1:
                markers[position[first.bit_length() - 1]] += 1
            elif bits.bit_count() == 2:
                u = position[first.bit_length() - 1]
                v = position[(bits ^ first).bit_length() - 1]
                adjacency[u].append(v)
                adjacency[v].append(u)
            else:
                raise ValueError("unexpected high-degree check contraction")

        # Every check meets this class, and every qubit has positive
        # degree. Thus connectivity of this contraction implies
        # connectivity of the complete Tanner graph.
        remaining = set(range(len(high)))
        component_sizes = []
        while remaining:
            root = min(remaining)
            visited, queue = {root}, [root]
            for u in queue:
                for v in adjacency[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            component_sizes.append(len(visited))
            remaining.difference_update(visited)
        components = tuple(sorted(component_sizes))

        # Joint walk profiles retain correlations between return counts
        # and off-diagonal counts, rather than just a graph spectrum.
        size = len(high)
        power = [[int(u == v) for v in range(size)]
                 for u in range(size)]
        walks = [[[] for _ in range(size)] for _ in range(size)]
        for _ in range(10):
            following = [[0] * size for _ in range(size)]
            for u, neighbors in enumerate(adjacency):
                target = following[u]
                for w in neighbors:
                    source = power[w]
                    for v, value in enumerate(source):
                        target[v] += value
            power = following
            for u in range(size):
                for v in range(size):
                    walks[u][v].append(power[u][v])

        labels = [
            (markers[u], len(adjacency[u]), tuple(walks[u][u]))
            for u in range(size)
        ]
        walk_profiles = tuple(sorted(
            (labels[u], tuple(sorted(
                (labels[v], tuple(walks[u][v]))
                for v in range(size))))
            for u in range(size)
        ))

        # Refine the full bipartite incidence graph using intrinsic
        # degrees only. No assumed X/Z or sector colouring enters.
        incidence = [[] for _ in range(m + n)]
        for j, column in enumerate(columns):
            vertex = m + j
            while column:
                bit = column & -column
                check = bit.bit_length() - 1
                incidence[check].append(vertex)
                incidence[vertex].append(check)
                column ^= bit

        colors = [len(neighbors) for neighbors in incidence]
        history = []
        for _ in range(6):
            features = [
                (colors[v],
                 tuple(sorted(Counter(
                     colors[t] for t in neighbors).items())))
                for v, neighbors in enumerate(incidence)
            ]
            history.append(tuple(sorted(Counter(features).items())))
            palette = {feature: c for c, feature in
                       enumerate(sorted(set(features)))}
            colors = [palette[feature] for feature in features]

        intrinsic = (
            components,
            tuple(sorted(Counter(row.bit_count()
                                 for row in rows).items())),
            tuple(sorted(Counter(col.bit_count()
                                 for col in columns).items())),
            walk_profiles, tuple(history),
            _tanner_signature(hx, hz),
            _degree_signature(hx, hz, n),
        )
        return components, sha256(repr(intrinsic).encode("ascii")).digest()

    # Exclusion references only, never candidate outputs:
    # twelve generating/generating representatives, twelve current
    # common-subgroup representatives, and twelve shown Q32 representatives.
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
        (4,5,7, 5,6,7),
        (4,6,22, 4,20,21),
        (16,18,22, 4,20,23),
        (4,6,16, 16,19,21),
        (4,21,22, 16,20,21),
    )
    blocked = set()
    for packet in archived:
        A = _unit_binomial_cells(
            G, tuple(G.elements[t] for t in packet[:3]))
        B = _unit_binomial_cells(
            G, tuple(G.elements[t] for t in packet[3:]))
        hx, hz, n = lifted_matrices(G, A, B)
        blocked.add(graph_key(hx, hz, n))

    local = random.Random(BASE_SEED ^ 0xC0327A)
    local.shuffle(plans)
    orbits, results = set(), []

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for ia, ib in plans:
        orbit = orbit_key(ia, ib)
        if orbit in orbits:
            continue
        orbits.add(orbit)
        ta, ha = cards[ia]
        tb, hb = cards[ib]
        A = _unit_binomial_cells(G, ta)
        B = _unit_binomial_cells(G, tb)
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 544:
            raise ValueError("frozen transverse bed changed size")
        if max(row.bit_count() for row in hx + hz) > min(W_MAX, 9):
            continue
        if not (_no_weight_two_kernel(hx, n)
                and _no_weight_two_kernel(hz, n)):
            continue

        signature = graph_key(hx, hz, n)
        if signature[0] != (G.q,) or signature in blocked:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 288 or 1000 * k <= 525 * n:
            continue
        if not _verify_three_witness(G, B, hx, hz):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified transverse lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified rank helpers disagree")

        blocked.add(signature)
        prescription = (
            ta, tb, len(ha), len(hb), len(ha & hb))
        results.append((
            freeze(A), freeze(B), k,
            prescription, signature[1].hex()))
        if len(results) >= count:
            break

    # Exhaustion is explicit: never pad the portfolio with orbit copies.
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication from an immutable structural portfolio."""
    portfolio = _q32_transverse_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(portfolio)):
        raise ValueError("archive-screened transverse Q32 portfolio exhausted")
    A, B, k, prescription, signature = portfolio[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "distance_certificate": 3,
        "component_prescription": "transverse proper/proper",
        "directions_and_subgroup_orders": prescription,
        "intrinsic_tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh certified cell without mutating the portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: The Q32 generating-triple catalogue's intersection and qubit-degree gates eliminate isomorphic repeats: its measured sixty slots collapsed to twelve distinct codes, with forty-eight iso_dup slots.
# REFUTED: The Q32 common-proper-subgroup catalogue supplies twenty inequivalent slots: sixty evaluated slots retained twelve classes and forty-eight iso_dup slots; all five measured distances remained three.
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
