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
CONCEPT = ("Quaternion Q8, order 8, A=B=(3,8): balanced incidence profiles without a forced three-parallel-matching core should supply novel distance-seven finalists in cell 0.350; exactly screened distance-six designs remain fallbacks.")
GROUP = ("Jump from cyclic Z8 to nonabelian Q8. Q8 has one involution and six elements of order four, whereas Z8 has elements of order eight. The nonabelian order-eight (3,8)x(3,8) bed is absent from BEDS. The parent's five finalists measured k=200,d30s=6; a group-law and incidence change tests that ceiling without asserting a transferred distance guarantee.")
SHAPES = ("Keep equal 64-variable, 24-check constituents. Factor 73 permits q<=8, giving n=584, kb=200 and guaranteed rate 25/73=0.342466 in cell 0.350. Parent weight-six minima occupied sector-1 Acol/Bcol blocks broadly. Independently change both constituents and screen complete matrices, including sector two.")
FILLING = ("Enumerate balanced pendant/hyperedge count profiles, prioritizing those without three parallel degree-two columns. Each row has six terms and each column at most three, so BOTH CSS maxima are nine, also below twelve. Verified constituent matrices exclude kernel weights through five and exactly count weight-six words; zero-six designs lead. An immutable catalogue conservatively rejects repeated full uncoloured Tanner invariants and requires verified ranks, CSS and complete kernel exclusion through five. Constituent floor seven does not certify mixed-sector quantum distance seven, and archive novelty remains evaluator-tested.")
FAMILY_NAME = "r035_quaternion_q8_38x38_balanced_kernel6_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q8 requires order eight")
    # x^4=1, y^2=x^2, y*x*y^-1=x^-1.
    # Canonical elements are x^a*y^b, 0<=a<4, 0<=b<2.
    els = [(a, b) for b in range(2) for a in range(4)]

    def red(g):
        a, b = int(g[0]), int(g[1])
        return ((a + 2 * (b // 2)) % 4, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 2*b*d) % 4,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 4, 0) if b == 0 else ((a + 2) % 4, 1)

    def idx(g):
        a, b = red(g)
        return 4*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 8)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


def _component_spec(rng, mode, q):
    """Matching columns, or two cycle columns plus five matchings."""
    if mode == 0:
        return (), tuple(sorted(rng.sample(range(q), 7)))
    if mode == 1:
        d = rng.randrange(1, q)
        phase = rng.randrange(q)
        # Five independent vertices of the eleven-cycle with step d.
        labels = tuple(sorted((phase + d * t) % q
                              for t in (0, 2, 4, 6, 8)))
        return (d, d), labels
    return (rng.randrange(1, q), rng.randrange(1, q)), \
        tuple(sorted(rng.sample(range(q), 5)))


def _component_cells(G, spec):
    ds, labels = spec
    e, els = G.e, G.elements
    if not ds:
        return [[[e] for _ in labels],
                [[els[s]] for s in labels]]
    # Column weights remain two; both row weights remain seven.
    return [[[e, els[ds[0]]]] + [[e] for _ in labels] + [[]],
            [[]] + [[els[s]] for s in labels] + [[e, els[ds[1]]]]]


def _translation_key(labels, q):
    return min(tuple(sorted((s + t) % q for s in labels))
               for t in range(q))


def _scaled_spec_key(spec, u, q):
    ds, labels = spec
    s = tuple((u * x) % q for x in labels)
    forward = _translation_key(s, q)
    backward = _translation_key(tuple((-x) % q for x in s), q)
    if not ds:
        return (0, min(forward, backward))
    d0, d1 = ((u * d) % q for d in ds)
    d0, d1 = min(d0, q - d0), min(d1, q - d1)
    return min((1, d0, d1, forward),
               (1, d1, d0, backward))


def _pair_key(a, b, q):
    """Remove translations, row swaps, common automorphisms and A/B swap."""
    keys = []
    for u in range(1, q):
        ka = _scaled_spec_key(a, u, q)
        kb = _scaled_spec_key(b, u, q)
        keys.append(tuple(sorted((ka, kb))))
    return min(keys)


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


def _unused_order11_reference(rng, G, idx):
    """Uncalled historical catalogue; never publishes a candidate.

    Candidate proposals are finite structural cards. Slot ownership is
    a function of the built graph, not its labels or proposal number.
    No mutable cross-slot state or saved parent filling is published.
    """
    from itertools import combinations
    from collections import Counter
    import hashlib

    q, e = G.q, G.e
    cap = min(9, W_MAX)
    element = {G._key(g): g for g in G.elements}

    def x(t):
        return element[(t % q, 0)]

    def labels_key(values):
        return min(tuple(sorted((v - anchor) % q for v in values))
                   for anchor in values)

    def signed(steps):
        return set(steps) | {(-d) % q for d in steps}

    def image(card, scale):
        kind, upper, lower, values = card
        a = tuple(sorted(min(scale*d % q, -scale*d % q)
                         for d in upper))
        b = tuple(sorted(min(scale*d % q, -scale*d % q)
                         for d in lower))
        v = labels_key(tuple(scale*s % q for s in values))
        return kind, a, b, v

    def canonical(card):
        images = []
        for u in range(1, q):
            kind, a, b, v = image(card, u)
            images.append((kind, a, b, v))
            if kind == 0:
                images.append((kind, b, a,
                               labels_key(tuple(-s % q for s in v))))
        return min(images)

    def regular(upper, lower, values):
        assert len(upper) == len(lower)
        assert len(values) + len(upper) + len(lower) == 7
        return [
            [[e, x(d)] for d in upper]
            + [[e] for _ in values] + [[] for _ in lower],
            [[] for _ in upper]
            + [[x(v)] for v in values]
            + [[e, x(d)] for d in lower],
        ]

    def build(card):
        kind, upper, lower, values = card
        if kind == 0:
            grid = regular(upper, lower, values)
        elif kind == 1:
            # Six cross matchings and one grounding singleton.
            # Unlike the parent's boundary, there is no internal ring.
            assert len(values) == 6
            grid = [
                [[e] for _ in values] + [[e]],
                [[x(v)] for v in values] + [[]],
            ]
        else:
            # Five matchings, opposite-layer ring, grounding singleton.
            assert len(values) == 5 and len(lower) == 1
            grid = [
                [[e] for _ in values] + [[], [e]],
                [[x(v)] for v in values] + [[e, x(lower[0])], []],
            ]
        assert all(len(row) == 7 for row in grid)
        assert all(len({G._key(g) for g in cell}) == len(cell)
                   for row in grid for cell in row)
        return grid

    def columns(rows, width):
        out = [0] * width
        mask = (1 << width) - 1
        for r, row in enumerate(rows):
            row &= mask
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def component_info(grid):
        # A zero one-cell side exposes the EXACT scaffold A action.
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
        width = 7*q
        mask = (1 << width) - 1
        rows = [row & mask for row in rows]
        if not _no_short_kernel(rows, width):
            return None
        cols = columns(rows, width)
        multiplicities = {}
        collisions = 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                old = multiplicities.get(syndrome, 0)
                collisions += old
                multiplicities[syndrome] = old + 1
        # After exclusion through three, every collision has disjoint
        # pair supports. A weight-four word has three such partitions.
        assert collisions % 3 == 0
        return gf2_rank(rows), collisions // 3

    descriptors = set()
    for p in range(3):
        options = []
        for steps in combinations(range(1, (q + 1)//2), p):
            ss = signed(steps)
            if not any((a+b) % q in ss for a in ss for b in ss):
                options.append(steps)
        for ai, upper in enumerate(options):
            for lower in options[ai:]:
                forbidden = signed(upper) | signed(lower)
                for tail in combinations(range(1, q), 6 - 2*p):
                    values = (0,) + tail
                    if any((a-b) % q in forbidden
                           for a, b in combinations(values, 2)):
                        continue
                    descriptors.add(canonical(
                        (0, upper, lower, values)))

    # Deliberately vary the intrinsic six-matching graph, not just its
    # grounding location or a translation of one matching set.
    for tail in combinations(range(1, q), 5):
        descriptors.add(canonical((1, (), (), (0,) + tail)))
    for d in range(1, (q + 1)//2):
        for tail in combinations(range(1, q), 4):
            values = (0,) + tail
            if any((a-b) % q in (d, (-d) % q)
                   for a, b in combinations(values, 2)):
                continue
            descriptors.add(canonical((2, (), (d,), values)))

    regular_bank, boundary_bank = [], []
    for card in sorted(descriptors):
        grid = build(card)
        info = component_info(grid)
        if info is None:
            continue
        record = (card, info, grid)
        (regular_bank if card[0] == 0 else boundary_bank).append(record)
    if not regular_bank:
        raise ValueError("no kernel-three-screened regular constituents")

    def fingerprint(hx, hz, width):
        """Intrinsic invariant of the complete uncoloured Tanner graph.

        High-degree qubits identify sector 2 without protograph labels.
        A check meeting two such qubits becomes a weighted edge; a check
        meeting one becomes a marked grounding loop. Check weights label
        these edges, and qubit degrees label vertices.

        Every feature is invariant under check/qubit relabeling.
        Equal fingerprints are discarded conservatively.
        """
        rows = hx + hz
        cols = columns(rows, width)
        degree_counts = Counter(c.bit_count() for c in cols)
        high = [j for j, c in enumerate(cols) if c.bit_count() >= 10]
        nv = len(high)
        if nv != 4*q:
            raise ValueError("unexpected high-degree contraction")
        position = {j: i for i, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        adjacency = [Counter() for _ in high]
        edge_weights = [Counter() for _ in high]
        for row in rows:
            bits = row & high_mask
            degree = bits.bit_count()
            if degree not in (1, 2):
                raise ValueError("check contraction is not an edge or anchor")
            first = bits & -bits
            u = position[first.bit_length() - 1]
            v = (u if degree == 1 else
                 position[(bits ^ first).bit_length() - 1])
            adjacency[u][v] += 1
            edge_weights[u][row.bit_count()] += 1
            if u != v:
                adjacency[v][u] += 1
                edge_weights[v][row.bit_count()] += 1

        neighbours = [tuple(sorted(a.items())) for a in adjacency]
        # Exploit translation only after checking it on the extracted
        # graph and its intrinsic vertex labels.
        vertex_labels = [
            (cols[j].bit_count(), tuple(sorted(edge_weights[v].items())))
            for v, j in enumerate(high)]
        symmetric = all(
            vertex_labels[i*q+h] == vertex_labels[i*q]
            and all(
                adjacency[i*q+h].get(j*q+t, 0)
                == adjacency[i*q].get(j*q+(t-h) % q, 0)
                for j in range(4) for t in range(q))
            for i in range(4) for h in range(q))
        roots = list(range(0, nv, q)) if symmetric else list(range(nv))
        diagonals, features = [], []
        for root in roots:
            vector = [0] * nv
            vector[root] = 1
            diagonal, pair = [], [[] for _ in high]
            for length in range(1, 11):
                following = [0] * nv
                for v, count in enumerate(vector):
                    if count:
                        for w, multiplicity in neighbours[v]:
                            following[w] += count * multiplicity
                vector = following
                diagonal.append(vector[root])
                if length <= 6:
                    for v, count in enumerate(vector):
                        pair[v].append(count)
            diagonals.append(tuple(diagonal))
            features.append(pair)

        colors = [
            (vertex_labels[v],
             diagonals[v//q if symmetric else v])
            for v in range(nv)]
        profiles = []
        for ri, root in enumerate(roots):
            profile = (
                colors[root],
                tuple(sorted((colors[v], tuple(features[ri][v]))
                             for v in range(nv))))
            profiles.extend([profile] * (q if symmetric else 1))
        return (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(degree_counts.items())),
            tuple(sorted(profiles)))

    def matching(values):
        return regular((), (), values)

    def ring(a, b, values):
        return regular((a,), (b,), values)

    two = regular((1, 3), (1, 3), (0, 2, 4))
    other_two = regular((1, 3), (1, 3), (0, 2, 6))
    ring_one = ring(1, 1, (0, 2, 4, 6, 8))
    grounded = build((2, (), (1,), (0, 2, 4, 6, 8)))
    six_grounded = build((1, (), (), tuple(range(6))))
    shown_archive = [
        (matching((0, 1, 2, 4, 5, 6, 10)),
         matching((0, 2, 3, 5, 6, 7, 9))),
        (ring(10, 10, (0, 2, 5, 7, 9)),
         ring(10, 10, (0, 2, 4, 7, 9))),
        (ring(6, 7, (0, 1, 4, 6, 8)),
         ring(4, 4, (0, 3, 6, 8, 9))),
        (ring(1, 7, (3, 4, 5, 6, 8)),
         ring(4, 6, (0, 1, 3, 6, 10))),
        (matching(tuple(range(7))), two),
        (ring_one, two),
        (two, two),
        (ring_one, other_two),
        (matching(tuple(range(7))),
         regular((1, 4), (1, 4), (0, 2, 5))),
        (two, grounded),
        (matching(tuple(range(7))), grounded),
        (matching(tuple(range(7))), six_grounded),
        (two, regular((2, 5), (2, 5), (0, 3, 7))),
        (two, regular((2, 3), (2, 3), (0, 1, 5))),
    ]
    archive_keys = set()
    for A, B in shown_archive:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(fingerprint(hx, hz, n))
    # These references are never candidate outputs. The complete run's
    # archive remains authoritative for unshown previously built codes.

    plans = []
    for ai in range(len(regular_bank)):
        for bi in range(ai, len(regular_bank)):
            for scale in range(1, q):
                plans.append((0, ai, bi, scale))
        for bi in range(len(boundary_bank)):
            for scale in range(1, q):
                plans.append((1, ai, bi, scale))
    rng.shuffle(plans)

    best, best_quality = None, None
    filling_keys, graph_keys = set(), set()
    owned_regular = 0
    # Different slots have different multiplicity pressures, but their
    # ownership bins are disjoint for the ENTIRE built-graph invariant.
    pressure = 1 + idx % 5
    for boundary, ai, bi, scale in plans[:1536]:
        a = regular_bank[ai]
        b = (boundary_bank if boundary else regular_bank)[bi]
        A = a[2]
        bcard = image(b[0], scale)
        B = build(bcard)
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        if n != 583 or max(r.bit_count() for r in hx + hz) > cap:
            continue
        signature = fingerprint(hx, hz, n)
        if signature in archive_keys or signature in graph_keys:
            continue
        graph_keys.add(signature)
        digest = hashlib.blake2b(
            repr(signature).encode("ascii"), digest_size=16).digest()
        if int.from_bytes(digest, "little") % N_SLOTS != idx:
            continue

        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 475*n <= 1000*k < 525*n):
            continue
        if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified owned-bin filling failed CSS")

        qa, qb = a[1][1], b[1][1]
        # Both floors are four. Prefer rank surplus first, then fewer
        # actual constituent weight-four words, with opposing pressures.
        # Multiplicity is a tie-break, NOT a quantum-distance estimate.
        quality = (k, -max(qa, qb),
                   -(pressure*qa + (6-pressure)*qb),
                   -(qa + qb), digest)
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best = {
                "A": A, "B": B, "k": k,
                "card": (a[0], bcard),
                "kernel_floor": 4,
                "component_four_counts": (qa, qb),
                "owner": idx,
            }
        if not boundary:
            owned_regular += 1
            if owned_regular >= 12:
                break

    if best is None:
        raise ValueError("no novel target-rate filling in intrinsic owner bin")
    return best


def _unused_order8_multisupport_catalogue(rng, G, idx):
    """Historical cyclic catalogue; the quaternion entry point never calls it.

    Its cyclic voltage normalization and reserved-position model are
    not used to publish any quaternion cells.
    """
    from itertools import permutations
    from collections import Counter

    if G.q != 8:
        raise ValueError("multisupport catalogue requires order eight")
    local = random.Random(BASE_SEED ^ 0x386704)
    q, width, cap = G.q, 8 * G.q, min(9, W_MAX)
    mask = (1 << width) - 1
    dummy = [[[G.e]]]

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def pair_table(cols):
        """Exactly exclude kernel weights one through four."""
        singles = set(cols)
        if 0 in singles or len(singles) != len(cols):
            return None
        pairs = {}
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                if syndrome in singles or syndrome in pairs:
                    return None
                pairs[syndrome] = (j, i)
        return pairs

    def six_count(rows):
        """Exclude weights <=5, then count every weight-six word."""
        cols = columns(rows, width)
        pairs = pair_table(cols)
        if pairs is None:
            return None
        triples = {}
        collisions = 0
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                for h in range(j):
                    value = syndrome ^ cols[h]
                    if value in pairs:
                        return None
                    previous = triples.get(value, 0)
                    collisions += previous
                    triples[value] = previous + 1
        # Once weights <=5 are excluded, colliding triples are disjoint.
        # Each six-word has ten unordered partitions into two triples.
        assert collisions % 10 == 0
        return collisions // 10

    def support_key(profile):
        return min(
            tuple(sorted(tuple(profile[i][j] for i in order)
                         for j in range(8)))
            for order in permutations(range(3)))

    seed_profile = (
        (1, 1, 0, 0, 0, 2, 2, 0),
        (0, 1, 1, 1, 1, 0, 0, 2),
        (0, 0, 1, 1, 1, 1, 1, 1),
    )
    profiles = [seed_profile]
    support_keys = {support_key(seed_profile)}
    double_types = sorted(set(permutations((1, 1, 0)))
                          | set(permutations((2, 0, 0))))
    triple_types = sorted(set(permutations((2, 1, 0)))
                          | {(1, 1, 1)}
                          | set(permutations((3, 0, 0))))
    # Count profiles are deliberate designs, not voltage orbit copies.
    for attempt in range(32768):
        cols = [(1, 0, 0)]
        cols += [local.choice(double_types) for _ in range(4)]
        cols += [local.choice(triple_types) for _ in range(3)]
        profile = tuple(tuple(c[i] for c in cols) for i in range(3))
        if any(sum(row) != 6 for row in profile):
            continue
        key = support_key(profile)
        if key in support_keys:
            continue
        support_keys.add(key)
        profiles.append(profile)
        if len(profiles) >= 18:
            break

    def draw(profile, seeded):
        C = [[[] for _ in range(8)] for _ in range(3)]
        if seeded:
            # Preserve the successful matching core, but change genuine
            # relative binomial gaps and endpoint voltages.
            C[0][0] = [G.e]
            C[0][1] = [G.e]
            C[1][1] = [G.e]
            for j, voltage in enumerate((0, 1, 3), 2):
                C[1][j] = [G.e]
                C[2][j] = [G.elements[voltage]]
            for j, row in ((5, 0), (6, 0), (7, 1)):
                C[row][j] = [G.e, G.elements[local.randrange(1, q)]]
                C[2][j] = [G.elements[local.randrange(q)]]
            return C
        for j in range(8):
            first = True
            for i in range(3):
                count = profile[i][j]
                if not count:
                    continue
                labels = local.sample(range(q), count)
                if first:
                    # A column gauge fixes one term, not relative labels.
                    anchor = labels[0]
                    labels = [(v - anchor) % q for v in labels]
                    first = False
                C[i][j] = [G.elements[v] for v in sorted(labels)]
        return C

    bank, fill_keys, component_keys = [], set(), set()
    for attempt in range(12288):
        # Half the proposals exploit a measured support; the remainder
        # explore different intrinsic hyperedge incidence structures.
        pi = 0 if attempt % 2 == 0 else (
            1 + (attempt // 2) % max(1, len(profiles) - 1))
        pi %= len(profiles)
        C = draw(profiles[pi], pi == 0)
        filling = tuple(tuple(G.idx(g) for g in cell)
                        for row in C for cell in row)
        if filling in fill_keys:
            continue
        fill_keys.add(filling)
        counts = [[len(cell) for cell in row] for row in C]
        if any(sum(row) != 6 for row in counts):
            raise ValueError("constituent row budget changed")
        if not _no_dead_lines(counts, counts):
            continue

        left, _, _ = lifted_matrices(G, C, dummy)
        _, right, _ = lifted_matrices(G, dummy, C)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        if gf2_rank(left) != 24 or gf2_rank(right) != 24:
            continue
        nl = six_count(left)
        if nl is None:
            continue
        nr = six_count(right)
        if nr is None:
            continue
        signature = (_tanner_signature(left, []),
                     tuple(sorted((nl, nr))))
        if signature in component_keys:
            continue
        component_keys.add(signature)
        bank.append((max(nl, nr), nl + nr, filling, C, pi))
        if len(bank) >= 32:
            break
    if len(bank) < 6:
        raise ValueError("too few kernel-five-clean multisupport constituents")
    bank.sort(key=lambda entry: entry[:3])

    plans = sorted(
        (int(bank[a][0] != 0 or bank[b][0] != 0),
         max(bank[a][0], bank[b][0]),
         bank[a][1] + bank[b][1], a, b)
        for a in range(len(bank)) for b in range(a, len(bank)))
    wanted = {idx + retry * N_SLOTS for retry in range(4)}
    graph_keys = set()
    accepted = 0
    n1 = 64 * q

    def full_key(hx, hz, n):
        cols = columns(hx + hz, n)
        return (
            tuple(sorted(Counter(r.bit_count()
                                 for r in hx + hz).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            _tanner_signature(hx, hz),
        )

    def mixed_five(cols, pairs):
        # Pure sector-1 words decompose into constituent kernel words,
        # already excluded through five. Every remaining five-word has
        # a triple partition containing a sector-2 qubit.
        for i in range(n1, len(cols)):
            first = cols[i]
            for j in range(i):
                syndrome = first ^ cols[j]
                for h in range(j):
                    if (syndrome ^ cols[h]) in pairs:
                        return True
        return False

    for _, _, _, a, b in plans:
        A, B = bank[a][3], bank[b][3]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if _grid_weights(sa, sb) > cap or not _no_dead_lines(sa, sb):
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 584 or not (325*n <= 1000*k < 375*n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        signature = full_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        position = accepted
        accepted += 1
        if position not in wanted:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified multisupport lift failed CSS")

        legal = True
        for checks in (hx, hz):
            cols = columns(checks, n)
            pairs = pair_table(cols)
            if pairs is None or mixed_five(cols, pairs):
                legal = False
                break
        if legal:
            return {
                "A": A, "B": B, "k": k,
                "catalogue_position": position,
                "component_six_counts": (bank[a][0], bank[b][0]),
                "support_cards": (bank[a][4], bank[b][4]),
                "verified_kernel_floor": 6,
            }
    raise ValueError("reserved multisupport designs failed complete screens")


from functools import lru_cache


def _q8_columns(rows, width):
    """Syndromes extracted exclusively from verified scaffold matrices."""
    result = [0] * width
    mask = (1 << width) - 1
    for r, row in enumerate(rows):
        row &= mask
        while row:
            bit = row & -row
            result[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return result


def _q8_pairs(cols):
    """Exactly reject every nonzero kernel support of weights one to four."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = {}
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles or syndrome in pairs:
                return None
            pairs[syndrome] = (j, i)
    return pairs


def _q8_component_quality(rows, width):
    """Exact exclusion through five and exact weight-six multiplicity."""
    cols = _q8_columns(rows, width)
    pairs = _q8_pairs(cols)
    if pairs is None:
        return None
    triples, collisions = {}, 0
    for i, first in enumerate(cols):
        for j in range(i):
            partial = first ^ cols[j]
            for h in range(j):
                syndrome = partial ^ cols[h]
                if syndrome in pairs:
                    return None
                previous = triples.get(syndrome, 0)
                collisions += previous
                triples[syndrome] = previous + 1
    # Smaller words absent: colliding triples are disjoint.
    # Every six-support has ten unordered triple partitions.
    assert collisions % 10 == 0
    sixes = collisions // 10
    return (7 if sixes == 0 else 6), sixes


def _q8_graph_key(rows, width):
    """Conservative full Tanner invariant, not an isomorphism test.

    It includes check-intersection profiles and incidence refinement.
    Equal keys are discarded; inequality certifies nonisomorphism of
    the bipartite graph with its check/qubit vertex classes.
    """
    from collections import Counter
    from hashlib import sha256

    cols = _q8_columns(rows, width)
    incidence = []
    for col in cols:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length() - 1)
            col ^= bit
        incidence.append(neighbors)

    # These initial colors are intrinsic check degrees.
    colors = [row.bit_count() for row in rows]
    history = [
        tuple(sorted(Counter(colors).items())),
        tuple(sorted(Counter(len(c) for c in incidence).items())),
        _tanner_signature(rows, []),
    ]
    for _ in range(4):
        qfeatures = [tuple(sorted(colors[r] for r in neighbors))
                     for neighbors in incidence]
        qpalette = {f: i for i, f in
                    enumerate(sorted(set(qfeatures)))}
        profiles = [[] for _ in rows]
        for neighbors, feature in zip(incidence, qfeatures):
            for r in neighbors:
                profiles[r].append(qpalette[feature])
        refined = [(colors[r], tuple(sorted(profiles[r])))
                   for r in range(len(rows))]
        history.append((
            tuple(sorted(Counter(qfeatures).items())),
            tuple(sorted(Counter(refined).items()))))
        palette = {f: i for i, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    return sha256(repr((width, len(rows), history)).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _q8_portfolio(G, count):
    """Pure deterministic portfolio; cached publication data are immutable."""
    from itertools import combinations_with_replacement, permutations

    if G.q != 8:
        raise ValueError("quaternion portfolio requires order eight")
    q, e, width = G.q, G.e, 8 * G.q
    cap = min(9, 12, W_MAX)
    mask = (1 << width) - 1
    dummy = [[[]]]
    row_orders = tuple(permutations(range(3)))

    def canonical(columns):
        return min(tuple(sorted(tuple(col[i] for i in order)
                                for col in columns))
                   for order in row_orders)

    def grid_counts(columns):
        return tuple(tuple(col[i] for col in columns) for i in range(3))

    def parity_rank(columns):
        return gf2_rank([
            sum((col[i] & 1) << j for j, col in enumerate(columns))
            for i in range(3)])

    # Four degree-two columns are genuine edges, not order-four
    # binomial loops, which would already introduce weight-four words.
    doubles = ((1, 1, 0), (1, 0, 1), (0, 1, 1))
    triples = tuple(sorted(
        set(permutations((2, 1, 0)))
        | set(permutations((3, 0, 0)))
        | {(1, 1, 1)}))
    profiles = set()
    for ds in combinations_with_replacement(doubles, 4):
        for ts in combinations_with_replacement(triples, 3):
            columns = ((1, 0, 0),) + ds + ts
            if any(sum(c[i] for c in columns) != 6 for i in range(3)):
                continue
            if parity_rank(columns) != 3:
                continue
            profiles.add(canonical(columns))

    def support_priority(columns):
        multiplicities = [
            sum(col == edge for col in columns) for edge in doubles]
        isolated = sum(max(col) == 3 for col in columns)
        # Remove the parent's forced three-edge bundle first.
        return (max(multiplicities), isolated, columns)

    profiles = sorted(profiles, key=support_priority)

    # Retain the measured incidence as a quaternion-only fallback card.
    # No archived cyclic labeling is reproduced.
    seed = (
        (1, 1, 0, 0, 0, 2, 2, 0),
        (0, 1, 1, 1, 1, 0, 0, 2),
        (0, 0, 1, 1, 1, 1, 1, 1),
    )
    seed_columns = canonical(tuple(
        tuple(seed[i][j] for i in range(3)) for j in range(8)))
    chosen = profiles[:19]
    if seed_columns not in chosen:
        chosen.append(seed_columns)
    elif len(profiles) > len(chosen):
        chosen.append(profiles[len(chosen)])
    if not chosen:
        raise ValueError("no full-augmentation-rank balanced supports")

    def component_rows(grid, right=False):
        if right:
            _, rows, _ = lifted_matrices(G, dummy, grid)
        else:
            rows, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in rows]

    def draw(local, columns, policy):
        grid = [[[] for _ in range(8)] for _ in range(3)]
        order_four = [g for g in G.elements
                      if g != e and G.mul(g, g) != e]
        for j, counts in enumerate(columns):
            for i, terms in enumerate(counts):
                if not terms:
                    continue
                if terms == 1:
                    cell = [G.pick(local)]
                elif terms == 2 and policy % 2 == 0:
                    anchor = G.pick(local)
                    relative = local.choice(order_four)
                    cell = [anchor, G.mul(anchor, relative)]
                else:
                    cell = local.sample(G.elements, terms)
                assert len(set(cell)) == terms
                grid[i][j] = cell

            # Right multiplication by one common factor fixes a
            # column gauge while retaining all relative cell voltages.
            anchor = next(grid[i][j][0] for i in range(3)
                          if grid[i][j])
            inverse = G.inv(anchor)
            for i in range(3):
                grid[i][j] = [G.mul(g, inverse) for g in grid[i][j]]
        return grid

    bank, component_keys = [], set()
    for card, columns in enumerate(chosen):
        support = grid_counts(columns)
        assert all(sum(row) == 6 for row in support)
        assert _no_dead_lines(support, support)
        assert _grid_weights(support, support) <= cap
        local = random.Random(BASE_SEED ^ 0x083807 ^ (104729 * card))
        records, filling_keys = [], set()
        for trial in range(384):
            grid = draw(local, columns, trial % 4)
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            left = component_rows(grid)
            if gf2_rank(left) != 3 * q:
                continue
            quality = _q8_component_quality(left, width)
            if quality is None:
                continue
            signature = _q8_graph_key(left, width)
            records.append((quality, trial, grid, signature))
        records.sort(key=lambda r: (-r[0][0], r[0][1], r[1]))

        retained = 0
        for quality, trial, grid, signature in records:
            if signature in component_keys:
                continue
            # Verify the right action directly, rather than assuming
            # an inversion convention or transferring a hand-built lift.
            right = component_rows(grid, True)
            if gf2_rank(right) != 3*q:
                continue
            right_quality = _q8_component_quality(right, width)
            if right_quality is None:
                continue
            component_keys.add(signature)
            floor = min(quality[0], right_quality[0])
            sixes = quality[1] + right_quality[1]
            bank.append((floor, sixes, card, trial, grid))
            retained += 1
            if retained >= 4:
                break
    if len(bank) < 5:
        raise ValueError("too few quaternion kernel-five-clean constituents")
    bank.sort(key=lambda r: (-r[0], r[1], r[2], r[3]))

    plans = sorted(
        (-min(a[0], b[0]), a[1] + b[1],
         int(a[2] == b[2]), ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))
    results, graph_keys = [], set()
    sector1 = A_SHAPE[1] * B_SHAPE[1] * q

    def mixed_five(cols, pairs):
        """Exact remaining weight-five test after constituent screening.

        Pure sector-one words split into verified constituent kernels.
        Every other five-support has a triple partition containing
        a sector-two qubit, and hence appears in this enumeration.
        """
        for i in range(sector1, len(cols)):
            first = cols[i]
            for j in range(i):
                partial = first ^ cols[j]
                for h in range(j):
                    if (partial ^ cols[h]) in pairs:
                        return True
        return False

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for negative_floor, multiplicity, same_card, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a[4], b[4]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        # Check both sums explicitly, independently of helper screening.
        wx = max(sum(row) + sum(sb[r][j] for r in range(3))
                 for row in sa for j in range(8))
        wz = max(sum(row) + sum(sa[r][j] for r in range(3))
                 for row in sb for j in range(8))
        assert max(wx, wz) <= cap
        assert _no_dead_lines(sa, sb)

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 584 or not (k > 0 and 325*n <= 1000*k < 375*n):
            continue
        signature = _q8_graph_key(hx + hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)

        passed = True
        for checks in (hx, hz):
            cols = _q8_columns(checks, n)
            pairs = _q8_pairs(cols)
            if pairs is None or mixed_five(cols, pairs):
                passed = False
                break
            del pairs, cols
        if not passed:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified quaternion lift failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= cap

        # Larger constituent floors lead; lower six-word multiplicity
        # breaks ties. Neither is a complete quantum-distance estimate.
        quality = (-negative_floor, -multiplicity, k, -same_card)
        results.append((
            quality, freeze(A), freeze(B), k,
            (a[2], b[2]), (a[0], b[0]), signature.hex()))
        if len(results) >= count + 8:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one immutable, completely kernel-five-screened design."""
    portfolio = _q8_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct quaternion target-rate portfolio exhausted")
    quality, A, B, k, cards, floors, signature = portfolio[idx]
    return dict(
        A=A, B=B, k=k, quality=quality, support_cards=cards,
        constituent_kernel_floors=floors, complete_kernel_floor=6,
        tanner_signature=signature)


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete quaternion cell without further draws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton plus symmetric-layer catalogue supplies twenty inequivalent designs: sixty evaluated slots retained only five distinct codes, all with measured distance four.
# REFUTED: Independent regular layers and joint-walk fingerprints alone supply five novel measured classes: the shown jointwalk12 program retained only two measured classes across sixty slots.
# REFUTED: Independent grounded constituents and intrinsic ownership bins supply twenty inequivalent designs: the current program retained only five distinct codes across sixty slots, all with measured distance four.
# REFUTED: The multisupport Z8 portfolio yields distance-seven finalists: all five measured finalists instead had k=200,d30s=6, with minimum logical families confined to sector one.
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
