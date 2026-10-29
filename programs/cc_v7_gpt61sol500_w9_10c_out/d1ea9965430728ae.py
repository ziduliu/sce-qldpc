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
CONCEPT = ("Generalized quaternion Q32, order 32, shapes (1,4)x(1,4): contrasting exact constituent weight-three multiplicities and weighted two-dimensional Tanner refinement should supply twenty archive-new inequivalent codes with certified distance three and exact rate 9/17 in cell 0.550.")
GROUP = ("Keep the frozen Q32 presentation, encoding and order byte-for-byte. This bed repeatedly measured k=288 and d30s=3. The parent's sixty common-subgroup slots retained only twelve distinct codes, so subgroup confinement alone is not a sufficient diversity prescription.")
SHAPES = ("Keep equal 128-variable, 32-check constituents: n=544, kb=288 and guaranteed rate 9/17=0.529412 in cell 0.550. These shapes permit q<=35, while this round fixes q=32. Parent minima occupied sector-1 Acol/Bcol blocks, especially the monomial columns; none were reported in sector 2. Monomial pivots retain exact k=288. The saturated distance-three profile scores 81/17; no strict distance or score increase is asserted.")
FILLING = ("Retain saturated [1,2,2,2], giving BOTH CSS maxima nine, but enumerate all inverse-distinct triple pairings without a common-subgroup restriction. Verified constituent lifts count exact weight-three words; deliberately contrasting multiplicities and subgroup sizes prioritize new pairings. Exclude all thirty-nine Q32 representatives shown here using intrinsic full-Tanner signatures strengthened by weighted two-dimensional refinement of the intrinsic degree-fourteen qubit projection. Different signatures certify inequivalence; equal signatures are discarded conservatively. Final verified lifts certify CSS, k=288, complete commuting-weight-one/two exclusion and an actual non-stabilizer weight-three logical. Unshown archive novelty and twenty-code yield remain evaluator-tested hypotheses; exhaustion is never padded with copies.")
FAMILY_NAME = "r055_quaternion32_14x14_contrast_pairrefinement_d3_v5"

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


def _retired_coset_ctx(rng, G, idx):
    """Inactive historical reference; never selects the new fillings."""
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
def _q32_contrast_portfolio(G):
    """Immutable deterministic portfolio; memoization changes only cost."""
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations, combinations_with_replacement

    if G.q != 32:
        raise ValueError("contrast portfolio requires frozen Q32")
    support = [[1, 2, 2, 2]]
    cap = min(W_MAX, 9)
    if not _no_dead_lines(support, support):
        raise ValueError("dead saturated support")
    if _grid_weights(support, support) > cap:
        raise ValueError("saturated profile requires cap nine")

    def component(labels):
        return _unit_binomial_cells(
            G, tuple(G.elements[t] for t in labels))

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def subgroup_size(labels):
        generators = [G.elements[t] for t in labels]
        reached, pending = {G.e}, [G.e]
        while pending:
            g = pending.pop()
            for h in generators:
                value = G.mul(g, h)
                if value not in reached:
                    reached.add(value)
                    pending.append(value)
        return len(reached)

    def constituent_threes(labels, side):
        """Exact multiplicity, extracted only from verified constituent lifts."""
        C, unit = component(labels), [[[G.e]]]
        if side == "A":
            rows, _, _ = lifted_matrices(G, C, unit)
        else:
            _, rows, _ = lifted_matrices(G, unit, C)
        width = 4 * G.q
        mask = (1 << width) - 1
        rows = [row & mask for row in rows[:G.q]]
        cols = _check_columns(rows, width)
        if 0 in cols or len(set(cols)) != width:
            return None
        positions = {s: i for i, s in enumerate(cols)}
        count = 0
        for i, first in enumerate(cols):
            for j in range(i):
                third = positions.get(first ^ cols[j], -1)
                if third > i:
                    count += 1
        return count

    def graph_key(hx, hz, n):
        """Intrinsic invariant of the complete uncoloured Tanner graph.

        Degree fourteen identifies a qubit class without sector labels.
        Each check meets one or two such qubits. Project it as a marked
        vertex or a weighted multigraph edge, retaining check weight and
        parallel-edge multiplicity. Two-dimensional refinement operates
        on ordered vertex pairs, not on element labels or check types.
        """
        rows = list(hx) + list(hz)
        cols = _check_columns(rows, n)
        high = [j for j, c in enumerate(cols) if c.bit_count() == 14]
        if len(high) != G.q:
            raise ValueError("intrinsic saturated degree class changed")
        size = len(high)
        position = {j: i for i, j in enumerate(high)}
        mask = sum(1 << j for j in high)
        marks = [Counter() for _ in high]
        edges = [[Counter() for _ in high] for _ in high]

        for row in rows:
            bits = row & mask
            degree = bits.bit_count()
            if degree not in (1, 2):
                raise ValueError("unexpected projected check")
            first = bits & -bits
            u = position[first.bit_length() - 1]
            weight = row.bit_count()
            if degree == 1:
                marks[u][weight] += 1
            else:
                v = position[(bits ^ first).bit_length() - 1]
                edges[u][v][weight] += 1
                edges[v][u][weight] += 1

        vertex = [
            (tuple(sorted(marks[u].items())),
             tuple(sorted(Counter(
                 (weight, multiplicity)
                 for v in range(size)
                 for weight, multiplicity in edges[u][v].items()
             ).items())))
            for u in range(size)
        ]
        initial = [
            (int(u == v), vertex[u], vertex[v],
             tuple(sorted(edges[u][v].items())))
            for u in range(size) for v in range(size)
        ]
        palette = {value: i for i, value in
                   enumerate(sorted(set(initial)))}
        colors = [palette[value] for value in initial]
        history = [tuple(sorted(Counter(initial).items()))]

        for step in range(6):
            features = []
            for u in range(size):
                offset = u * size
                for v in range(size):
                    joins = Counter(
                        (colors[offset + t], colors[t * size + v])
                        for t in range(size)
                    )
                    features.append((
                        colors[offset + v],
                        tuple(sorted(joins.items())),
                    ))
            history.append(tuple(sorted(Counter(features).items())))
            palette = {value: i for i, value in
                       enumerate(sorted(set(features)))}
            following = [palette[value] for value in features]
            stable = len(set(following)) == len(set(colors))
            colors = following
            if stable and step >= 1:
                break

        intrinsic = (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(history),
            _tanner_signature(list(hx), list(hz)),
            _degree_signature(list(hx), list(hz), n),
        )
        return sha256(repr(intrinsic).encode("ascii")).digest()

    # Exclusion references ONLY: generating, common-subgroup, mixed and
    # residual representatives explicitly reported in this prompt.
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
        (1,3,5, 1,3,7),
        (1,3,5, 1,4,7),
        (1,3,5, 2,6,16),
        (1,3,16, 1,3,18),
        (1,3,16, 1,4,16),
        (1,3,16, 1,16,18),
    )
    blocked = set()
    for packet in archived:
        hx, hz, n = lifted_matrices(
            G, component(packet[:3]), component(packet[3:]))
        blocked.add(graph_key(hx, hz, n))

    directions = tuple(range(1, 8)) + tuple(range(16, 24))
    cards = list(combinations(directions, 3))
    counts_a = [constituent_threes(card, "A") for card in cards]
    counts_b = [constituent_threes(card, "B") for card in cards]
    sizes = [subgroup_size(card) for card in cards]

    actions = [(u, v) for u in range(1, 16, 2) for v in range(16)]

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    images = []
    for card in cards:
        images.append(tuple(
            tuple(sorted(inverse_class(
                ((u * a + v * b) % 16, b))
                for a, b in (G.elements[t] for t in card)))
            for u, v in actions
        ))

    def orbit_key(a, b):
        return min(tuple(sorted((x, y)))
                   for x, y in zip(images[a], images[b]))

    plans = [
        (a, b)
        for a, b in combinations_with_replacement(range(len(cards)), 2)
        if counts_a[a] is not None and counts_b[b] is not None
    ]
    local = random.Random(BASE_SEED ^ 0x325C02)
    local.shuffle(plans)
    # Contrast is a deliberate structural ordering, not an optimistic
    # distance estimate. It differs from minimum-multiplicity ordering.
    plans.sort(key=lambda ab: (
        -abs(counts_a[ab[0]] - counts_b[ab[1]]),
        sizes[ab[0]] == sizes[ab[1]],
        -abs(sizes[ab[0]] - sizes[ab[1]]),
        ab[0] == ab[1],
    ))

    entries, orbits = [], set()
    for a, b in plans:
        orbit = orbit_key(a, b)
        if orbit in orbits:
            continue
        orbits.add(orbit)
        A, B = component(cards[a]), component(cards[b])
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 544 or max(r.bit_count() for r in hx + hz) > cap:
            continue
        signature = graph_key(hx, hz, n)
        if signature in blocked:
            continue
        if not (_no_weight_two_kernel(hx, n)
                and _no_weight_two_kernel(hz, n)):
            continue

        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if k != 288 or 1000 * k <= 525 * n:
            continue

        # Construct the witness from actual HZ column syndromes in
        # Acol1, avoiding reliance on an assumed inverse convention.
        syndromes = _check_columns(hz, n)
        base = 4 * G.q
        unit_columns = {
            syndromes[base + t]: base + t for t in range(G.q)
        }
        boundary = syndromes[base + G.q]
        witness = 1 << (base + G.q)
        legal = True
        while boundary:
            bit = boundary & -boundary
            position = unit_columns.get(bit)
            if position is None:
                legal = False
                break
            witness ^= 1 << position
            boundary ^= bit
        if not legal or witness.bit_count() != 3:
            continue
        if any((witness & row).bit_count() & 1 for row in hz):
            continue
        if gf2_rank(list(hx) + [witness]) != rx + 1:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified contrast pair failed CSS")

        blocked.add(signature)
        entries.append((
            freeze(A), freeze(B), witness,
            (cards[a], cards[b]),
            (counts_a[a], counts_b[b]),
            (sizes[a], sizes[b]),
        ))
        if len(entries) >= N_SLOTS:
            break

    # Never recycle an archived class or weaken the certificate to pad.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed selection from the immutable contrast portfolio."""
    entries = _q32_contrast_portfolio(G)
    if not (0 <= idx < len(entries)):
        raise ValueError("archive-screened Q32 contrast portfolio exhausted")
    A, B, witness, directions, counts, sizes = entries[idx]
    return {
        "A": A, "B": B, "screen_k": 288,
        "distance_certificate": 3,
        "verified_three_logical": witness,
        "directions": directions,
        "component_three_counts": counts,
        "component_subgroup_sizes": sizes,
        "reported_archive_references": 39,
        "catalogue_position": idx,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell without resampling or shared mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: The Q32 generating-triple catalogue's intersection and qubit-degree gates eliminate isomorphic repeats: its measured sixty slots collapsed to twelve distinct codes, with forty-eight iso_dup slots.
# REFUTED: Common-proper-subgroup confinement and coset signatures supply twenty inequivalent Q32 codes: the measured sixty-slot parent retained twelve distinct codes and forty-eight iso_dup slots, with all five measured distances remaining three.
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
