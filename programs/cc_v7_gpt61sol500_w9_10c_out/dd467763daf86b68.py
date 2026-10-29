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
CONCEPT = ("Metacyclic Z8:Z1, order 8, A=B=(8,3): independently optimized transposed incidence structures should exceed fitness 12.3290 through distance-seven readings or distance-six codes with useful rank deficits in cell 0.350.")
GROUP = ("Keep the metacyclic group and order implementation unchanged. The parent measured five [584,200,6] finalists and fitness 12.3290. Retain this useful lift while changing shapes and filling selection; neither transposition nor a large search reading establishes a distance advantage.")
SHAPES = ("Use equal tall (8,3) constituents with 24 variables and 64 checks. Factor 73 still gives n=584, kb=200 and guaranteed rate (-5)*(-5)/73=0.342466. Transposing both sides preserves the available code family, so shape change alone is not the improvement hypothesis. Parent minima were sector-1 Acol/Bcol witnesses; the new complete screens cover both sectors and actual ranks retain [0.325,0.375).")
FILLING = ("Twenty support-optimization cards vary incidence geometry, five/six-term dual-row budgets, support exchanges, rank pressure and exact short-word multiplicity. Independently optimized dual constituents are transposed into the published shapes. Every final pair passes both strict weight-nine budgets, verified ranks, CSS and complete kernel exclusion through five. Different full uncoloured Tanner invariants separate catalogue entries; shown parent pairings are conservatively excluded. These gates certify only distance at least six, not seven or novelty against the unseen archive.")
FAMILY_NAME = "r035_m8r1s1_83x83_transposed_rank_support_portfolio_v5"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    # Same metacyclic construction rule; factor 73 requires q <= 8.
    # Z_8 x| Z_1, with s^r = 1 mod m.
    m, r, s = 8, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (8, 3)               # (ra, ca)
B_SHAPE = (8, 3)               # (rb, cb)


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


def _inactive_multisupport_context(rng, G, idx):
    """Uncalled historical reconstruction of the old-shape catalogue.

    Different intrinsic invariants certify different Tanner graphs.
    Reserved positions prevent retries from recycling another slot.
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


def _tall_columns(rows, width):
    """Syndromes of supplied verified rows."""
    out = [0] * width
    mask = (1 << width) - 1
    for r, row in enumerate(rows):
        row &= mask
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _tall_pairs(cols):
    """Exactly exclude every kernel support of weight one through four."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return None
    pairs = set()
    for i, first in enumerate(cols):
        for second in cols[:i]:
            value = first ^ second
            if value in singles or value in pairs:
                return None
            pairs.add(value)
    return pairs


def _tall_clean5(rows, width):
    cols = _tall_columns(rows, width)
    pairs = _tall_pairs(cols)
    if pairs is None:
        return False
    for i, first in enumerate(cols):
        for j in range(i):
            partial = first ^ cols[j]
            for h in range(j):
                if (partial ^ cols[h]) in pairs:
                    return False
    return True


def _tall_spectrum(rows, width):
    """Dirty optimization penalty; exact five/six counts once <=4 clean."""
    from collections import Counter
    cols = _tall_columns(rows, width)
    singles = Counter(cols)
    bad = 10000 * (
        singles.get(0, 0)
        + sum(v * (v - 1) // 2 for v in singles.values()))
    pairs = {}
    for i, first in enumerate(cols):
        for second in cols[:i]:
            value = first ^ second
            bad += 100 * singles.get(value, 0)
            previous = pairs.get(value, 0)
            bad += previous
            pairs[value] = previous + 1
    if bad:
        return 4, bad, 0

    triples, five, six = {}, 0, 0
    for i, first in enumerate(cols):
        for j in range(i):
            partial = first ^ cols[j]
            for h in range(j):
                value = partial ^ cols[h]
                five += int(value in pairs)
                previous = triples.get(value, 0)
                six += previous
                triples[value] = previous + 1
    assert five % 10 == six % 10 == 0
    five, six = five // 10, six // 10
    return (5 if five else 6 if six else 7), five, six


def _tall_key(rows, width):
    """Conservative full uncoloured Tanner invariant.

    Equality is inconclusive and discarded. Different keys separate
    check/qubit-preserving Tanner isomorphisms without CSS colouring.
    """
    from collections import Counter
    cols = _tall_columns(rows, width)
    m = len(rows)
    adjacency = [{} for _ in rows]
    masks = [0] * m
    for i, row in enumerate(rows):
        for j in range(i):
            overlap = (row & rows[j]).bit_count()
            if overlap:
                adjacency[i][j] = adjacency[j][i] = overlap
                masks[i] |= 1 << j
                masks[j] |= 1 << i

    features = []
    for i, neighbors in enumerate(adjacency):
        profile = Counter(
            (rows[j].bit_count(), overlap,
             (masks[i] & masks[j]).bit_count())
            for j, overlap in neighbors.items())
        features.append((rows[i].bit_count(),
                         tuple(sorted(profile.items()))))
    palette = {f: c for c, f in enumerate(sorted(set(features)))}
    colors = [palette[f] for f in features]
    history = [tuple(sorted(Counter(features).items()))]

    incidence = []
    for col in cols:
        neighbors = []
        while col:
            bit = col & -col
            neighbors.append(bit.bit_length() - 1)
            col ^= bit
        incidence.append(neighbors)
    for _ in range(3):
        qfeatures = [tuple(sorted(colors[r] for r in ns))
                     for ns in incidence]
        qp = {f: c for c, f in enumerate(sorted(set(qfeatures)))}
        profiles = [[] for _ in rows]
        for ns, feature in zip(incidence, qfeatures):
            for r in ns:
                profiles[r].append(qp[feature])
        refined = [
            (colors[i], tuple(sorted(profiles[i])),
             tuple(sorted((overlap, colors[j])
                          for j, overlap in adjacency[i].items())))
            for i in range(m)]
        history.append((tuple(sorted(Counter(qfeatures).items())),
                        tuple(sorted(Counter(refined).items()))))
        palette = {f: c for c, f in enumerate(sorted(set(refined)))}
        colors = [palette[f] for f in refined]
    return (width, m,
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(history))


@lru_cache(maxsize=1)
def _tall_portfolio(G, count):
    """Pure deterministic optimization; cached publication data immutable."""
    if G.q != 8:
        raise ValueError("tall portfolio requires the frozen order eight")
    q, width, cap = G.q, 8 * G.q, min(9, W_MAX)
    dummy = [[[]]]
    mask = (1 << width) - 1

    # Structural starting cards, not archived completed fillings.
    profiles = (
        ((1, 1, 0, 0, 0, 2, 2, 0),
         (0, 1, 1, 1, 1, 0, 0, 2),
         (0, 0, 1, 1, 1, 1, 1, 1)),
        ((1, 1, 0, 0, 0, 1, 1, 2),
         (0, 1, 1, 1, 1, 1, 1, 0),
         (0, 0, 1, 1, 1, 1, 1, 1)),
        ((2, 1, 1, 1, 1, 0, 0, 0),
         (1, 1, 0, 0, 0, 1, 1, 2),
         (0, 0, 1, 1, 1, 1, 1, 1)),
        ((3, 0, 1, 1, 1, 0, 0, 0),
         (0, 0, 1, 1, 1, 1, 1, 1),
         (0, 3, 0, 0, 0, 1, 1, 1)),
    )

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    def transpose(grid):
        return [[list(grid[i][j]) for i in range(3)]
                for j in range(8)]

    def normalize(grid):
        out = [[list(cell) for cell in row] for row in grid]
        for j in range(8):
            anchor = next(out[i][j][0][0]
                          for i in range(3) if out[i][j])
            for i in range(3):
                out[i][j] = sorted(
                    [G.elements[(g[0] - anchor) % q]
                     for g in out[i][j]])
        return out

    def rows_of(grid, right=False):
        if right:
            _, rows, _ = lifted_matrices(G, dummy, grid)
        else:
            rows, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in rows]

    def draw(local, profile, budget):
        grid = [
            [[G.elements[t] for t in local.sample(range(q), terms)]
             for terms in row]
            for row in profile]
        if budget == 5:
            for i in range(3):
                degrees = [sum(len(grid[r][j]) for r in range(3))
                           for j in range(8)]
                eligible = [j for j in range(8)
                            if grid[i][j] and degrees[j] > 1]
                if eligible:
                    j = local.choice(eligible)
                    del grid[i][j][local.randrange(len(grid[i][j]))]
        return normalize(grid)

    def mutate(local, grid, column_cap, exchange):
        out = [[list(cell) for cell in row] for row in grid]
        if local.random() < exchange:
            degrees = [sum(len(out[i][j]) for i in range(3))
                       for j in range(8)]
            moves = [
                (i, a, b) for i in range(3)
                for a in range(8) for b in range(8)
                if a != b and out[i][a] and degrees[a] > 1
                and degrees[b] < column_cap]
            if moves:
                i, a, b = local.choice(moves)
                del out[i][a][local.randrange(len(out[i][a]))]
                choices = [g for g in G.elements if g not in out[i][b]]
                out[i][b].append(local.choice(choices))
                return normalize(out)
        occupied = [(i, j) for i in range(3) for j in range(8)
                    if out[i][j]]
        i, j = local.choice(occupied)
        t = local.randrange(len(out[i][j]))
        choices = [g for g in G.elements if g not in out[i][j]]
        out[i][j][t] = local.choice(choices)
        return normalize(out)

    def objective(stats, rank, card):
        floor, five, six = stats
        if floor <= 5:
            return floor, -five, -six, -rank
        # Alternate genuine rank and multiplicity pressures.
        if card % 2:
            return floor, -rank, -six, -abs(six - q * (card % 5))
        return floor, -six, -rank, -abs(six - q * (card % 5))

    bank, fillings = [], set()
    for card in range(20):
        local = random.Random(BASE_SEED ^ (0x835A05 + 104729 * card))
        budget = 5 if card >= 15 else 6
        column_cap = 4 if budget == 5 else 3
        exchange = (0.08, 0.20, 0.40, 0.65)[card // 5]
        records = []
        for restart in range(8):
            grid = draw(local, profiles[(card + restart) % 4], budget)
            rows = rows_of(grid)
            stats, rank = _tall_spectrum(rows, width), gf2_rank(rows)
            quality = objective(stats, rank, card)
            best = (quality, grid, stats, rank)
            for step in range(112):
                proposed = mutate(local, grid, column_cap, exchange)
                pr = rows_of(proposed)
                ps, pk = _tall_spectrum(pr, width), gf2_rank(pr)
                pq = objective(ps, pk, card)
                if pq > best[0]:
                    best = (pq, proposed, ps, pk)
                if pq >= quality or (
                        step % 41 == 40 and ps[0] >= stats[0] - 1):
                    grid, stats, rank, quality = proposed, ps, pk, pq
            _, grid, stats, rank = best
            if stats[0] < 6:
                continue
            right = rows_of(grid, True)
            rs = _tall_spectrum(right, width)
            if rs[0] < 6:
                continue
            filling = freeze(grid)
            if filling in fillings:
                continue
            fillings.add(filling)
            records.append((
                min(stats[0], rs[0]), max(stats[2], rs[2]),
                stats[2] + rs[2], rank + gf2_rank(right),
                card, restart, filling))
        records.sort(key=lambda r: (-r[0], r[1], r[3], r[5]))
        bank.extend(records[:4])

    if len(bank) < 5:
        raise ValueError("too few independently optimized clean constituents")

    # Reference matrices are never published. Include both orientations
    # so exclusion does not rely on a hand-written transposition theorem.
    reference_cards = (
        (6, 1, 1, 2, 7, 6), (1, 2, 3, 7, 4, 5),
        (7, 3, 5, 5, 7, 7), (4, 3, 1, 6, 7, 6),
        (5, 1, 2, 7, 6, 6), (2, 1, 5, 4, 6, 4),
        (3, 7, 1, 2, 5, 5), (5, 2, 3, 2, 6, 7),
        (7, 3, 5, 5, 7, 2), (5, 6, 5, 2, 4, 7),
    )

    def reference(card):
        grid = [[[] for _ in range(8)] for _ in range(3)]
        grid[0][0] = grid[0][1] = grid[1][1] = [G.e]
        for j, v in enumerate((0, 1, 3), 2):
            grid[1][j], grid[2][j] = [G.e], [G.elements[v]]
        for t, (j, i) in enumerate(((5, 0), (6, 0), (7, 1))):
            grid[i][j] = [G.e, G.elements[card[t]]]
            grid[2][j] = [G.elements[card[t + 3]]]
        return grid

    references = [reference(c) for c in reference_cards]
    excluded = set()
    for ai, a in enumerate(references):
        for b in references[ai:]:
            for A, B in ((a, b), (transpose(a), transpose(b))):
                hx, hz, n = lifted_matrices(G, A, B)
                excluded.add(_tall_key(hx + hz, n))

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            preliminary = (
                min(a[0], b[0]), -max(a[1], b[1]),
                -a[2] - b[2], -a[3] - b[3],
                int(a[4] != b[4]))
            plans.append((preliminary, ai, bi))
    plans.sort(key=lambda p: (p[0], -p[1], -p[2]), reverse=True)

    def complete_clean5(hx, hz, n):
        """Exact complete screen, including both pure sectors.

        Sector-two restrictions are extracted from the ACTUAL tall lift.
        No inferred left/right transfer is used for this final gate.
        """
        n1 = A_SHAPE[1] * B_SHAPE[1] * q
        first_mask = (1 << n1) - 1
        block_mask = (1 << q) - 1

        # Pure sector one is small in the tall construction.
        for checks in (hx, hz):
            if not _tall_clean5([r & first_mask for r in checks], n1):
                return False

        # HX sector two separates by A-row; use A-row zero.
        b_rows = [
            (row >> n1) & mask for row in hx[:B_SHAPE[1] * q]]
        # HZ sector two separates by B-row; compact B-row zero.
        a_rows = []
        for j in range(A_SHAPE[1]):
            for t in range(q):
                source = hz[(j * B_SHAPE[0]) * q + t]
                compact = 0
                for i in range(A_SHAPE[0]):
                    offset = n1 + (i * B_SHAPE[0]) * q
                    compact |= ((source >> offset) & block_mask) << (i*q)
                a_rows.append(compact)
        if not (_tall_clean5(a_rows, width)
                and _tall_clean5(b_rows, width)):
            return False

        # Every remaining five-support is mixed and has a triple
        # partition containing at least one endpoint from each sector.
        for checks in (hx, hz):
            cols = _tall_columns(checks, n)
            pairs = _tall_pairs(cols)
            if pairs is None:
                return False
            for i in range(n1, n):
                first = cols[i]
                for j in range(i):
                    partial = first ^ cols[j]
                    for h in range(min(j, n1)):
                        if (partial ^ cols[h]) in pairs:
                            return False
        return True

    results = []
    for preliminary, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = transpose(a[6]), transpose(b[6])
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 584 or not (325*n <= 1000*k < 375*n):
            continue
        if max(r.bit_count() for r in hx + hz) > cap:
            continue
        signature = _tall_key(hx + hz, n)
        if signature in excluded:
            continue
        excluded.add(signature)
        if not css_ok(hx, hz):
            raise ValueError("verified tall lift failed CSS")
        if not complete_clean5(hx, hz, n):
            continue
        quality = (min(a[0], b[0]), k,
                   -max(a[1], b[1]), -a[2] - b[2])
        results.append((quality, freeze(A), freeze(B), k,
                        (a[4], b[4]), (a[1], b[1])))
        if len(results) >= count + 8:
            break
    results.sort(key=lambda r: r[0], reverse=True)
    return tuple(results[:count])


def NEW_CTX(rng, G, idx):
    """Publish one immutable, full-matrix-screened tall design."""
    portfolio = _tall_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct screened tall portfolio exhausted")
    quality, A, B, k, cards, sixes = portfolio[idx]
    return dict(A=A, B=B, screen_k=k, quality=quality,
                optimization_cards=cards,
                dual_component_six_counts=sixes,
                complete_kernel_floor=6)


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell of the new eight-row shapes."""
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
# REFUTED: Prioritizing zero constituent six-counts in the multisupport catalogue produces measured distance-seven finalists: all five measured finalists remained at k=200,d30s=6, with extensive sector-1 weight-six logical families.
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
