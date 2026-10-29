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
CONCEPT = ("Dic6, order 24, A=(1,3), B=(2,7): triangle-free binomial/incidence constituents should supply novel cell-0.450 finalists with distance readings at least four, preferentially at k=258 rather than the parent's certified distance-three plateau.")
GROUP = ("Preserve the dicyclic construction rule x^m=1, y^2=x^(m/2), y*x*y^-1=x^-1. The new factor 23 excludes order 32, so repair to m=12 and order 24, giving n=552. Related measured (1,3)x(2,7) beds reached distance four; this is motivation, not a transfer of their distance or archive novelty.")
SHAPES = ("A=(1,3), B=(2,7) give 72-variable/24-check and 168-variable/48-check constituents. n=552, kb=240 and guaranteed rate 10/23=0.434783 target cell 0.450. Actual ranks must retain [0.425,0.475). The parent's weight-three minima were sector-1 Acol/Bcol operators, particularly involving its monomial pivots; the new A has no monomial pivot.")
FILLING = ("A uses three inverse-distinct binomials. B has seven degree-two columns and seven terms in each row, so BOTH CSS maxima are at most nine: HX eight, HZ nine. Optimize balanced internal-edge/cross-matching cards in Q8, C12 and Dic6, but never publish a rotation-only A/B pair that merely revisits the common-cyclic construction. Prioritize actual k, then constituent weight-four multiplicity. Unequal intrinsic full uncoloured Tanner signatures allocate distinct catalogue positions. Verified matrices enforce CSS, target rate and absence of every commuting kernel word through weight three in both sectors. Novelty against the unseen archive and beating the cell's 7.478 incumbent remain measured hypotheses.")
FAMILY_NAME = "r045_dic6_q24_13x27_trianglefree_incidence_rank_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 24

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic6 has order 24")
    # Same dicyclic rule; factor 23 requires the order repair.
    # x^12=1, y^2=x^6, y*x*y^-1=x^-1; element x^a y^b.
    m, half = 12, 6
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        a, b = g
        return ((a + half * (b // 2)) % m, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + half * b * d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + half) % m, 1)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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


def _retired_q32_coset_reference(rng, G, idx):
    """Inactive historical reference; never publishes a candidate."""
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


def _r45_kernel3(rows, width):
    """Exactly exclude every nonzero kernel support of weights 1..3."""
    columns = _check_columns(rows, width)
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return None
    pairs = {}
    collisions = 0
    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            if syndrome in singles:
                return None
            previous = pairs.get(syndrome, 0)
            collisions += previous
            pairs[syndrome] = previous + 1
    # Clean singles imply colliding pairs are disjoint. Each four-word
    # has three unordered pair partitions.
    assert collisions % 3 == 0
    return columns, collisions // 3


def _r45_component(G, grid, side):
    """Extract only the actual scaffold's left/right constituent action."""
    unit = [[[G.e]]]
    width = len(grid[0]) * G.q
    mask = (1 << width) - 1
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    return [row & mask for row in rows]


def _r45_signature(hx, hz, n):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    Check intersections, degree-conditioned projections and rooted
    distance shells survive arbitrary qubit/check permutations.
    Unequal signatures certify non-isomorphism; equality is discarded
    conservatively. No group labels or protograph positions enter.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    m = len(rows)
    columns = _check_columns(rows, n)
    projections = {}
    for column in columns:
        degree = column.bit_count()
        if not degree:
            raise ValueError("isolated qubit in screened design")
        adjacency = projections.setdefault(degree, [0] * m)
        bits = column
        while bits:
            bit = bits & -bits
            adjacency[bit.bit_length() - 1] |= column ^ bit
            bits ^= bit

    degrees = sorted(projections)
    adjacency = [0] * m
    for relation in projections.values():
        for i, neighbors in enumerate(relation):
            adjacency[i] |= neighbors

    profiles = [Counter() for _ in rows]
    weights = [row.bit_count() for row in rows]
    for i in range(m):
        for j in range(i):
            overlap = (rows[i] & rows[j]).bit_count()
            feature = (
                overlap,
                tuple((((projections[d][i] >> j) & 1),
                       (projections[d][i] & projections[d][j]).bit_count())
                      for d in degrees),
            )
            profiles[i][(weights[j], feature)] += 1
            profiles[j][(weights[i], feature)] += 1

    def spheres(root, relation):
        visited = frontier = 1 << root
        result = []
        while frontier:
            following = 0
            bits = frontier
            while bits:
                bit = bits & -bits
                following |= relation[bit.bit_length() - 1]
                bits ^= bit
            frontier = following & ~visited
            if frontier:
                result.append(frontier.bit_count())
                visited |= frontier
        return tuple(result)

    roots = tuple(sorted(
        (weights[i], tuple(sorted(profiles[i].items())),
         spheres(i, adjacency),
         tuple((d, spheres(i, projections[d])) for d in degrees))
        for i in range(m)))
    intrinsic = (
        tuple(sorted(Counter(weights).items())),
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        roots,
    )
    return sha256(repr(intrinsic).encode("ascii")).digest()


from functools import lru_cache


@lru_cache(maxsize=1)
def _r45_portfolio(G):
    """Immutable deterministic optimized portfolio, independent of calls."""
    from itertools import combinations

    if G.q != 24:
        raise ValueError("new incidence bed requires Dic6 of order 24")
    cap = min(9, W_MAX)
    q = G.q
    local = random.Random(BASE_SEED ^ 0xD6C045)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def order(g):
        value = G.e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return length
        raise ValueError("invalid dicyclic element order")

    def inverse_class(g):
        return min(G.idx(g), G.idx(G.inv(g)))

    representatives = sorted({
        inverse_class(g) for g in G.elements if order(g) > 2
    })

    # A cards are exhaustive structural triples, not per-slot draws.
    # Keep relative orientations: constituent isomorphism alone need
    # not imply isomorphism of a pair of nonabelian lifted products.
    abank = []
    for triple in combinations(representatives, 3):
        A = [[[G.e, G.elements[t]] for t in triple]]
        rows = _r45_component(G, A, "A")
        result = _r45_kernel3(rows, 3 * q)
        if result is None:
            continue
        rank = gf2_rank(rows)
        abank.append((rank, result[1], triple, freeze(A)))
    abank.sort()
    if not abank:
        raise ValueError("no triangle-free Dic6 binomial A")

    # Q8=<x^3,y>, C12=<x>, and the full Dic6 domain. A Q8-based
    # degree-seven B can have three incidence components, permitting
    # a different rank surplus from the rotation-only archive.
    domains = (
        tuple(g for g in G.elements if g[0] % 3 == 0),
        tuple(g for g in G.elements if g[1] == 0),
        tuple(G.elements),
    )
    bbank, fillings, component_keys = [], set(), set()

    for attempt in range(6144):
        domain_id = attempt % 3
        domain = domains[domain_id]
        internal = (attempt // 3) % 4
        cross_count = 7 - 2 * internal
        directions = sorted({
            inverse_class(g) for g in domain if order(g) > 2
        })
        if len(directions) < internal:
            continue

        B = [[], []]
        for row in range(2):
            chosen = local.sample(directions, internal)
            for label in chosen:
                B[row].append([G.e, G.elements[label]])
                B[1 - row].append([])
        for g in local.sample(list(domain), cross_count):
            B[0].append([G.e])
            B[1].append([g])

        counts = [[len(cell) for cell in row] for row in B]
        if not (len(B[0]) == len(B[1]) == 7
                and all(sum(row) == 7 for row in counts)
                and all(counts[0][j] + counts[1][j] == 2
                        for j in range(7))):
            raise ValueError("balanced degree-two B card malformed")

        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in B for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        rows = _r45_component(G, B, "B")
        result = _r45_kernel3(rows, 7 * q)
        if result is None:
            continue

        # This conservative constituent gate saves full-lift work.
        signature = _r45_signature(rows, [], 7 * q)
        if signature in component_keys:
            continue
        component_keys.add(signature)
        rank = gf2_rank(rows)
        rotation_only = all(g[1] == 0
                            for row in B for cell in row for g in cell)
        bbank.append((rank, result[1], filling, freeze(B),
                      rotation_only, domain_id, internal))
        if len(bbank) >= 96:
            break
    bbank.sort(key=lambda entry: entry[:3])
    if not bbank:
        raise ValueError("no triangle-free balanced B cards")

    plans = []
    for ai, a in enumerate(abank):
        a_rotation = all(G.elements[t][1] == 0 for t in a[2])
        for bi, b in enumerate(bbank):
            # Do not recycle the common-C12 construction, even with
            # a fresh labeling or an unshown cyclic matching profile.
            if a_rotation and b[4]:
                continue
            plans.append((a[0] + b[0], max(a[1], b[1]),
                          a[1] + b[1], ai, bi))
    plans.sort()

    shortlist, graph_keys = [], set()
    for _, _, _, ai, bi in plans:
        a, b = abank[ai], bbank[bi]
        A, B = a[3], b[3]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        if n != 552 or max(row.bit_count() for row in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 425 * n <= 1000 * k < 475 * n):
            continue

        signature = _r45_signature(hx, hz, n)
        if signature in graph_keys:
            continue
        if (_r45_kernel3(hx, n) is None
                or _r45_kernel3(hz, n) is None):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified incidence pair failed CSS")

        graph_keys.add(signature)
        shortlist.append((
            k, -max(a[1], b[1]), -(a[1] + b[1]),
            signature, A, B, a[2], b[5], b[6],
        ))
        # Optimize a larger pool than the published portfolio, while
        # keeping the expensive full-matrix screening bounded.
        if len(shortlist) >= max(40, 2 * N_SLOTS):
            break

    shortlist.sort(key=lambda entry: entry[:4], reverse=True)
    return tuple(shortlist[:N_SLOTS])


def NEW_CTX(rng, G, idx):
    """Pure indexed selection; no retries publish another slot's design."""
    entries = _r45_portfolio(G)
    if not (0 <= idx < len(entries)):
        raise ValueError("novel target-rate kernel-three-clean catalogue exhausted")
    k, _, _, signature, A, B, triple, domain, internal = entries[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "verified_kernel_floor": 4,
        "signature": signature,
        "A_direction_card": triple,
        "B_domain_card": domain,
        "B_internal_edges_per_row": internal,
        "catalogue_position": idx,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh complete cell from the screened immutable entry."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: The Q32 generating-triple catalogue's intersection and qubit-degree gates eliminate isomorphic repeats: its measured sixty slots collapsed to twelve distinct codes, with forty-eight iso_dup slots.
# REFUTED: Common-proper-subgroup Q32 triples supply twenty inequivalent codes: the measured sixty-slot catalogue retained twelve distinct codes and forty-eight iso_dup slots; all five measured distances remained three.
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
