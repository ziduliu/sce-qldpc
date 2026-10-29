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
CONCEPT = ("Generalized quaternion Q8, order 8, A=B=(2,7): triangle-free-first incidence constituents and invariant-separated coupled lifts should improve the parent's 4.7650 five-code fitness while verified ranks place the portfolio in cell 0.500.")
GROUP = ("Keep the generalized-quaternion/dicyclic construction rule, reducing x's order from sixteen to four and y's square from x^8 to x^2. New size factor 53 excludes Q32; Q8 is the largest generalized-quaternion order that fits, giving n=424. The parent's Q32 one-row catalogues consistently measured distance three. The new order is a size repair, not evidence of greater distance.")
SHAPES = ("Equal (2,7) constituents have 56 variables and 16 checks. n=424, kb=200 and guaranteed rate 25/53=0.471698 lie just below the requested band; actual verified ranks must satisfy [0.475,0.525). Connected degree-two constituents supply rank deficits absent from the parent's unit pivots. Parent weight-three minima occupied sector-1 Acol/Bcol blocks, often involving Bcol0 for X or Acol0 for Z; remove those singleton pivots rather than merely relabeling them.")
FILLING = ("Every constituent column has two terms and every row seven, giving BOTH LP check maxima nine, also below twelve. Enumerate matching/internal-cycle prescriptions and exact-screen their verified constituent kernels through three. Retain different triangle-free deck partitions before coupling; broaden with connected triangle-bearing prescriptions for distinct-slot yield. Complete verified lifts enforce target rate, CSS and absence of weight-one/two kernels. Triangle-free full lifts receive exact weight-three exclusion and a verified non-stabilizer weight-four witness; alternatives require a verified weight-three witness. Full uncoloured Tanner intersection and intrinsic degree-fourteen weighted-walk invariants conservatively separate entries. Distance-four codes rank first; distance-three alternatives prevent missing-finalist zeros. Catalogue yield and unseen archive novelty remain measured hypotheses.")
FAMILY_NAME = "r050_quaternion8_27x27_incidence_trianglefirst_certified_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q8 has order eight")
    # Same quaternion/dicyclic rule: x^4=1, y^2=x^2,
    # y*x*y^-1=x^-1; element x^a y^b.
    m, square = 4, 2
    els = [(a, b) for b in range(2) for a in range(m)]

    def red(g):
        a, b = g
        return ((a + square * (b // 2)) % m, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + square * b * d) % m,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % m, 0) if b == 0 else ((a + square) % m, 1)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
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


def _retired_q32_mixed_context(rng, G, idx):
    """Uncalled historical Q32 context; never fills the new shapes."""
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


def _q8_clear_three(rows, width):
    """Exact exclusion of every kernel support of weights one to three."""
    cols = _check_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return False
    for i, first in enumerate(cols):
        for second in cols[:i]:
            if first ^ second in singles:
                return False
    return True


def _q8_span(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p in pivots:
                word ^= pivots[p]
            else:
                pivots[p] = word
                break
    return pivots


def _q8_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _q8_witness(checks, stabilizers, width, weight):
    """Find an actual logical of weight three or four in verified rows."""
    cols = _check_columns(checks, width)
    own = _q8_span(stabilizers)
    if weight == 3:
        singles = {syndrome: i for i, syndrome in enumerate(cols)}
        for i, first in enumerate(cols):
            for j in range(i):
                h = singles.get(first ^ cols[j])
                if h is None or h == i or h == j:
                    continue
                word = (1 << i) | (1 << j) | (1 << h)
                if _q8_outside(word, own):
                    return word
        return None

    pairs = {}
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            word = (1 << i) | (1 << j)
            previous = pairs.get(syndrome)
            if previous is None:
                pairs[syndrome] = word
                continue
            difference = word ^ previous
            if (difference.bit_count() == 4
                    and _q8_outside(difference, own)):
                return difference
    return None


def _q8_graph_key(hx, hz, n):
    """Intrinsic full, uncoloured Tanner invariants.

    Unequal keys certify non-isomorphism. Equality is inconclusive and
    causes conservative rejection. No CSS colours or element indices
    enter the key.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _check_columns(rows, n)
    high = [j for j, col in enumerate(cols) if col.bit_count() == 14]
    position = {j: i for i, j in enumerate(high)}
    mask = sum(1 << j for j in high)
    size = len(high)
    adjacency = [Counter() for _ in high]

    for row in rows:
        endpoints = []
        bits = row & mask
        while bits:
            bit = bits & -bits
            endpoints.append(position[bit.bit_length() - 1])
            bits ^= bit
        if len(endpoints) != 2:
            raise ValueError("incidence check lost its two intrinsic endpoints")
        u, v = endpoints
        adjacency[u][v] += 1
        adjacency[v][u] += 1

    power = [[int(u == v) for v in range(size)] for u in range(size)]
    features = [[[] for _ in range(size)] for _ in range(size)]
    for _ in range(6):
        following = [[0] * size for _ in range(size)]
        for u, neighbors in enumerate(adjacency):
            target = following[u]
            for v, multiplicity in neighbors.items():
                for w, value in enumerate(power[v]):
                    target[w] += multiplicity * value
        power = following
        for u in range(size):
            for v in range(size):
                features[u][v].append(power[u][v])

    walks = tuple(sorted(
        (sum(adjacency[u].values()), tuple(features[u][u]),
         tuple(sorted(
             (sum(adjacency[v].values()), tuple(features[v][v]),
              tuple(features[u][v]))
             for v in range(size))))
        for u in range(size)))
    intrinsic = (
        n, len(rows),
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in cols).items())),
        _tanner_signature(hx, hz), walks,
    )
    return sha256(repr(intrinsic).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _q8_portfolio(G, count):
    """Immutable, deterministic, jointly screened structural catalogue."""
    from itertools import combinations

    if G.q != 8:
        raise ValueError("new incidence bed requires Q8")
    q, width = G.q, 7 * G.q
    cap = min(9, W_MAX)
    dummy = [[[G.e]]]
    mask = (1 << width) - 1

    # Q8 has three non-involutory inverse classes. Internal binomial
    # edges and cross-layer matchings always retain column weight two.
    directions = sorted({
        min(G.idx(g), G.idx(G.inv(g)))
        for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
    })

    def build(spec):
        upper, lower, labels = spec
        A = [
            [[G.e, G.elements[t]] for t in upper]
            + [[G.e] for _ in labels] + [[] for _ in lower],
            [[] for _ in upper]
            + [[G.elements[t]] for t in labels]
            + [[G.e, G.elements[t]] for t in lower],
        ]
        assert all(len(row) == 7 and sum(map(len, row)) == 7
                   for row in A)
        assert all(sum(len(A[i][j]) for i in range(2)) == 2
                   for j in range(7))
        return A

    def deck_partition(rows):
        """Keep different triangle-free deck partitions before coupling.

        This label-dependent proposal key is NOT a non-isomorphism
        certificate. Only _q8_graph_key separates published codes.
        """
        cols = _check_columns(rows, width)
        adjacency = [[] for _ in rows]
        for col in cols:
            first = col & -col
            u = first.bit_length() - 1
            v = (col ^ first).bit_length() - 1
            adjacency[u].append(v)
            adjacency[v].append(u)
        colors = [-1] * len(rows)
        colors[0] = 0
        queue = [0]
        for u in queue:
            for v in adjacency[u]:
                if colors[v] < 0:
                    colors[v] = colors[u] ^ 1
                    queue.append(v)
                elif colors[v] == colors[u]:
                    return None
        return tuple(colors[:q])

    specs = []
    for p in range(4):
        steps = list(combinations(directions, p))
        for upper in steps:
            for lower in steps:
                for labels in combinations(range(q), 7 - 2 * p):
                    specs.append((upper, lower, labels))
    local = random.Random(BASE_SEED ^ 0x827704)
    local.shuffle(specs)

    clean, broad = [], []
    partitions, broad_counts, fingerprints = set(), {}, set()
    for spec in specs:
        A = build(spec)
        filling = tuple(tuple(G.idx(g) for g in cell)
                        for row in A for cell in row)
        if filling in fingerprints:
            continue
        fingerprints.add(filling)
        left, _, _ = lifted_matrices(G, A, dummy)
        rows = [row & mask for row in left]
        if gf2_rank(rows) != 2 * q - 1:
            continue
        if not _no_weight_two_kernel(rows, width):
            continue
        clear = _q8_clear_three(rows, width)
        if clear:
            partition = deck_partition(rows)
            if partition is None or partition in partitions:
                continue
            partitions.add(partition)
            clean.append((A, spec, True))
        else:
            # Spread the bank over support and intrinsic component
            # structures instead of retaining only one label family.
            key = (len(spec[0]), _tanner_signature(rows, []))
            used = broad_counts.get(key, 0)
            if used >= 8 or len(broad) >= 64:
                continue
            broad_counts[key] = used + 1
            broad.append((A, spec, False))

    bank = clean + broad
    if not bank:
        raise ValueError("no connected simple Q8 incidence constituents")
    plans = sorted(
        (-int(a[2] and b[2]), -int(a[2]) - int(b[2]), ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))

    signatures, results = set(), []
    for _, _, ai, bi in plans:
        A, B = bank[ai][0], bank[bi][0]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 424 or max(row.bit_count() for row in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (475 * n <= 1000 * k < 525 * n):
            continue
        if not (_no_weight_two_kernel(hx, n)
                and _no_weight_two_kernel(hz, n)):
            continue

        signature = _q8_graph_key(hx, hz, n)
        if signature in signatures:
            continue
        clear = (_q8_clear_three(hx, n)
                 and _q8_clear_three(hz, n))
        distance = 4 if clear else 3
        witness = _q8_witness(hz, hx, n, distance)
        kind = "X"
        if witness is None:
            witness = _q8_witness(hx, hz, n, distance)
            kind = "Z"
        if witness is None:
            continue
        checks = hz if kind == "X" else hx
        if any((row & witness).bit_count() & 1 for row in checks):
            raise ValueError("verified logical witness does not commute")
        if not css_ok(hx, hz):
            raise ValueError("verified Q8 incidence lift failed CSS")

        signatures.add(signature)

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = k * distance * distance / n
        results.append((
            quality, freeze(A), freeze(B), k, distance,
            kind, witness, (bank[ai][1], bank[bi][1])))
        # All triangle-free/triangle-free plans precede alternatives.
        # Keep a wider distinct portfolio before choosing final slots.
        if len(results) >= max(count, 5) + 12:
            break

    results.sort(key=lambda entry: entry[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an immutable screened portfolio."""
    entries = _q8_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("invariant-separated Q8 incidence catalogue exhausted")
    quality, A, B, k, distance, kind, witness, prescription = entries[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "exact_distance": distance, "verified_logical_type": kind,
        "verified_logical": witness, "prescription": prescription,
        "certified_score": quality,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the already screened full cell without changing shared state."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The Z11 triangle-free matching catalogue reliably supplies five distinct measured codes: the current evaluation collapsed 60 slots to four distinct codes, with d30s readings 4,4,3,3.
# REFUTED: The Q32 generating-triple catalogue's intersection and qubit-degree gates eliminate isomorphic repeats: its measured sixty slots collapsed to twelve distinct codes, with forty-eight iso_dup slots.
# REFUTED: The Q32 common-proper-subgroup coset catalogue supplies twenty inequivalent slots: its measured sixty slots retained twelve distinct codes and forty-eight iso_dup slots, with all five measured distances remaining three.
# REFUTED: Mixed proper-subgroup/generating Q32 pairings supply twenty evaluator-distinct codes: the current sixty-slot evaluation retained nine distinct codes, and all five measured finalists still had distance three.
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
