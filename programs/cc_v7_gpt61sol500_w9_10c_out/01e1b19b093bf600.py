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
CONCEPT = ("Cyclic Z11 = metacyclic Z11:Z1, order 11, A=(2,7), B=(2,7): an archive-screened portfolio of distance-four candidates and six-internal-column distance-three fallbacks should prevent the parent's two-code collapse in cell 0.500.")
GROUP = ("Keep the frozen group and order byte-for-byte. Order eleven is the largest lift allowed by size factor 53. This bed previously scored 7.7998 with five measured codes; the current score 3.1508 came from only two measured classes, both still reading four. Improve novel-class yield without claiming a group-distance advantage.")
SHAPES = ("Keep equal 77-variable, 22-check constituents, q=11 and n=583. The guaranteed bound is kb=275, rate 25/53=0.47170. Verified full ranks permit k=280 or 287, excess five or twelve, both inside cell 0.500. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly, especially Acol3/5 for X; vary constituent topology rather than treating one block as the sole bottleneck.")
FILLING = ("Exhaustively enumerate regular cards with zero through three internal binomials per layer and two singly grounded families. Regular rows total seven; grounded rows total six/seven; columns contain one or two terms, so BOTH LP maxima are nine. Prioritize complete-kernel-three-clean pairs, but reserve at least half the portfolio for triangle-rich regular pairs, prioritizing six-internal-column constituents absent from the parent's triangle-free rule. Fallbacks must exclude every commuting support through weight two and exhibit an actual non-stabilizer weight-three word in a verified full lift. Reject shown archive fingerprints and repeated intrinsic full-Tanner fingerprints. Five novel k=287 certified-distance-three fallbacks would score 4.431; novel distance-four candidates improve that baseline. Unseen archive coverage and available class counts remain measurement hypotheses.")
FAMILY_NAME = "r050_m11r1s1_27x27_internal6_dualtier_archive_portfolio_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    # Same metacyclic construction rule; factor 53 requires q <= 11.
    # Z_11 x| Z_1, with s^r = 1 mod m.
    m, r, s = 11, 1, 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 7)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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


def _retired_jointwalk_context(rng, G, idx):
    """Uncalled historical catalogue; never publishes candidate cells.

    Constituent automorphisms are quotiented before pairing, but their
    RELATIVE scaling is retained. Different intrinsic fingerprints
    certify nonisomorphism; equal fingerprints are discarded.
    """
    from itertools import combinations
    from collections import Counter

    q, e = G.q, G.e
    cap = min(9, W_MAX)
    element = {G._key(g): g for g in G.elements}

    def x(t):
        return element[(t % q, 0)]

    def label_key(values):
        # Every minimum translated tuple starts at zero.
        return min(tuple(sorted((v - anchor) % q for v in values))
                   for anchor in values)

    def regular(upper, lower, labels):
        p = len(upper)
        assert len(lower) == p and 2*p + len(labels) == 7
        grid = [
            [[e, x(d)] for d in upper]
            + [[e] for _ in labels] + [[] for _ in lower],
            [[] for _ in upper]
            + [[x(s)] for s in labels]
            + [[e, x(d)] for d in lower],
        ]
        assert all(sum(map(len, row)) == 7 for row in grid)
        assert all(len(grid[0][j]) + len(grid[1][j]) == 2
                   for j in range(7))
        return grid

    def scaled(card, u):
        upper, lower, labels = card
        a = tuple(sorted(min(u*d % q, (-u*d) % q) for d in upper))
        b = tuple(sorted(min(u*d % q, (-u*d) % q) for d in lower))
        return a, b, label_key(tuple(u*s % q for s in labels))

    def canonical(card):
        images = []
        for u in range(1, q):
            a, b, values = scaled(card, u)
            images.append((a, b, values))
            images.append((b, a, label_key(
                tuple((-s) % q for s in values))))
        return min(images)

    def signed_steps(steps):
        return {d % q for d in steps} | {(-d) % q for d in steps}

    # Unlike the parent, upper and lower internal steps are independent.
    # Anchoring one matching at zero removes redundant translations.
    descriptors = set()
    for p in range(3):
        options = []
        for steps in combinations(range(1, (q + 1)//2), p):
            signed = signed_steps(steps)
            if any((a+b) % q in signed for a in signed for b in signed):
                continue
            options.append(steps)
        for ai, upper in enumerate(options):
            for lower in options[ai:]:
                forbidden = signed_steps(upper) | signed_steps(lower)
                for tail in combinations(range(1, q), 6 - 2*p):
                    labels = (0,) + tail
                    if any((a-b) % q in forbidden
                           for a, b in combinations(labels, 2)):
                        continue
                    descriptors.add(canonical((upper, lower, labels)))

    bank = []
    component_mask = (1 << (7*q)) - 1
    for card in sorted(descriptors):
        A = regular(*card)
        # Exact constituent action from the scaffold, with a zero side.
        rows, _, _ = lifted_matrices(G, A, [[[]]])
        rows = [row & component_mask for row in rows]
        if gf2_rank(rows) != 2*q - 1:
            continue
        if not _no_short_kernel(rows, 7*q):
            continue
        bank.append(card)
    bank.sort(key=lambda c: (c[0] == c[1], -len(c[0]), c))
    if not bank:
        raise ValueError("no connected triangle-free regular constituents")

    def columns(rows, width):
        cols = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                cols[bit.bit_length() - 1] |= tag
                row ^= bit
        return cols

    def joint_fingerprint(hx, hz, width):
        """Intrinsic invariant extracted from the complete VERIFIED lift.

        Regular codes have degree-four and degree-fourteen qubits.
        Every check meets exactly two degree-fourteen qubits. Contract
        that check to a weighted edge. Any Tanner isomorphism preserves
        this multigraph, without needing to preserve X/Z check types.

        Rooted walk diagonals through length twelve and joint walk
        profiles through length eight are permutation invariants.
        Translation symmetry only accelerates their exact computation.
        """
        rows = hx + hz
        cols = columns(rows, width)
        degrees = tuple(sorted(Counter(c.bit_count() for c in cols).items()))
        high = [j for j, c in enumerate(cols) if c.bit_count() == 14]
        if len(high) != 4*q:
            raise ValueError("regular lift lost its intrinsic degree class")
        position = {j: i for i, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        nv = len(high)
        adjacency = [Counter() for _ in range(nv)]
        for row in rows:
            bits = row & high_mask
            if bits.bit_count() != 2:
                raise ValueError("check does not contract to a high-degree edge")
            first = bits & -bits
            u = position[first.bit_length() - 1]
            v = position[(bits ^ first).bit_length() - 1]
            adjacency[u][v] += 1
            adjacency[v][u] += 1
        neighbours = [tuple(sorted(a.items())) for a in adjacency]

        # Verify the symmetry on the EXTRACTED graph before exploiting it.
        symmetric = all(
            adjacency[i*q+h].get(j*q+t, 0)
            == adjacency[i*q].get(j*q+(t-h) % q, 0)
            for i in range(4) for j in range(4)
            for h in range(q) for t in range(q))
        roots = list(range(0, nv, q)) if symmetric else list(range(nv))
        diagonals, pair_features = [], []
        for root in roots:
            vector = [0] * nv
            vector[root] = 1
            diagonal = []
            features = [[] for _ in range(nv)]
            for length in range(1, 13):
                following = [0] * nv
                for v, count in enumerate(vector):
                    if count:
                        for w, multiplicity in neighbours[v]:
                            following[w] += count * multiplicity
                vector = following
                diagonal.append(vector[root])
                if length <= 8:
                    for v, count in enumerate(vector):
                        features[v].append(count)
            diagonals.append(tuple(diagonal))
            pair_features.append(features)

        colors = [
            (sum(adjacency[v].values()),
             diagonals[v//q if symmetric else v])
            for v in range(nv)]
        profiles = []
        for ri, root in enumerate(roots):
            profile = (
                colors[root],
                tuple(sorted((colors[v], tuple(pair_features[ri][v]))
                             for v in range(nv))))
            profiles.extend([profile] * (q if symmetric else 1))
        return (
            tuple(sorted(Counter(row.bit_count() for row in rows).items())),
            degrees, tuple(sorted(profiles)))

    def matching(values):
        return regular((), (), values)

    def ring(d0, d1, values):
        return regular((d0,), (d1,), values)

    # Shown archived regular distance-four classes. Singleton codes
    # have a different intrinsic qubit-degree distribution and cannot
    # be isomorphic to any candidate submitted by this catalogue.
    two = regular((1, 3), (1, 3), (0, 2, 4))
    other_two = regular((1, 3), (1, 3), (0, 2, 6))
    archived = [
        (matching((0, 1, 2, 4, 5, 6, 10)),
         matching((0, 2, 3, 5, 6, 7, 9))),
        (ring(10, 10, (0, 2, 5, 7, 9)),
         ring(10, 10, (0, 2, 4, 7, 9))),
        (matching(tuple(range(7))), two),
        (ring(1, 1, (0, 2, 4, 6, 8)), two),
        (two, two),
        (ring(1, 1, (0, 2, 4, 6, 8)), other_two),
        (matching(tuple(range(7))),
         regular((1, 4), (1, 4), (0, 2, 5))),
    ]
    archive_keys = set()
    for A, B in archived:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(joint_fingerprint(hx, hz, n))

    pairs = [(ai, bi) for ai in range(len(bank))
             for bi in range(ai, len(bank))]
    pairs.sort(key=lambda ab: (
        bank[ab[0]][0] == bank[ab[0]][1]
        and bank[ab[1]][0] == bank[ab[1]][1],
        -(len(bank[ab[0]][0]) + len(bank[ab[1]][0])), ab))
    scales = tuple(range(2, q-1)) + (1, q-1)

    graph_keys, orbit_keys = set(), set()
    accepted = 0
    # Interleave intrinsic component pairs and relative orientations.
    # Scaling B alone is not treated as a common group automorphism.
    for turn in range(len(scales)):
        for ai, bi in pairs:
            a = bank[ai]
            u = scales[(turn + 3*ai + bi) % len(scales)]
            b = scaled(bank[bi], u)
            orbit = min(tuple(sorted((scaled(a, v), scaled(b, v))))
                        for v in range(1, q))
            if orbit in orbit_keys:
                continue
            orbit_keys.add(orbit)
            A, B = regular(*a), regular(*b)
            sa = [[len(c) for c in row] for row in A]
            sb = [[len(c) for c in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue

            hx, hz, n = lifted_matrices(G, A, B)
            if n != 583 or max(r.bit_count() for r in hx + hz) > cap:
                continue
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if k != 287 or not (475*n <= 1000*k < 525*n):
                continue
            signature = joint_fingerprint(hx, hz, n)
            if signature in archive_keys or signature in graph_keys:
                continue
            if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
                continue
            graph_keys.add(signature)

            if accepted == idx:
                if not css_ok(hx, hz):
                    raise ValueError("verified joint-voltage lift failed CSS")
                return {"A": A, "B": B, "k": k,
                        "card": (a, b, u), "kernel_floor": 4,
                        "joint_walk_depth": 12}
            accepted += 1

    # Exhaustion voids a slot rather than publishing an archived copy.
    raise ValueError("novel regular joint-voltage catalogue exhausted")


from functools import lru_cache


@lru_cache(maxsize=1)
def _r050_dualtier_portfolio(G, count):
    """Deterministic immutable portfolio; memoization changes only cost.

    Enumeration covers structural cards, not independent label draws.
    Unequal intrinsic signatures separate Tanner isomorphism classes.
    Equal signatures are conservatively discarded.
    """
    from itertools import combinations
    from collections import Counter

    q, e, width = G.q, G.e, 7 * G.q
    cap = min(9, W_MAX)
    if q != 11:
        raise ValueError("dual-tier portfolio requires frozen order eleven")
    elements = {G._key(g): g for g in G.elements}

    def x(t):
        return elements[(t % q, 0)]

    def labels_key(values):
        return min(tuple(sorted((v - anchor) % q for v in values))
                   for anchor in values)

    # card = (kind, upper_steps, lower_steps, matching_voltages).
    # kind 0: regular; 1: six matchings plus one grounding;
    # kind 2: five matchings, opposite-layer ring and one grounding.
    def image(card, scale):
        kind, upper, lower, values = card
        a = tuple(sorted(min(scale*d % q, -scale*d % q)
                         for d in upper))
        b = tuple(sorted(min(scale*d % q, -scale*d % q)
                         for d in lower))
        return kind, a, b, labels_key(tuple(scale*v % q for v in values))

    def gauge(card):
        kind, a, b, values = card
        if kind:
            return card
        reverse = (kind, b, a,
                   labels_key(tuple(-v % q for v in values)))
        return min(card, reverse)

    def canonical(card):
        return min(gauge(image(card, u)) for u in range(1, q))

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
            C = regular(upper, lower, values)
        elif kind == 1:
            C = [
                [[e] for _ in values] + [[e]],
                [[x(v)] for v in values] + [[]],
            ]
        else:
            C = [
                [[e] for _ in values] + [[], [e]],
                [[x(v)] for v in values]
                + [[e, x(lower[0])], []],
            ]
        if len(C) != 2 or any(len(row) != 7 for row in C):
            raise ValueError("constituent card changed the frozen shape")
        if any(len({G._key(g) for g in cell}) != len(cell)
               for row in C for cell in row):
            raise ValueError("constituent card has duplicate terms")
        return C

    def columns(rows, n):
        out = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def low_profile(rows, n):
        """Exact exclusion through two and counts at weights three/four."""
        cols = columns(rows, n)
        singles = set(cols)
        if 0 in singles or len(singles) != n:
            return None
        pair_counts = {}
        triples = collisions = 0
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                triples += int(syndrome in singles)
                old = pair_counts.get(syndrome, 0)
                collisions += old
                pair_counts[syndrome] = old + 1
        # Distinct nonzero columns make all equal-syndrome pair
        # collisions disjoint. Each four-word has three partitions.
        assert triples % 3 == 0 and collisions % 3 == 0
        return (3 if triples else 4, triples // 3, collisions // 3)

    def constituent(grid):
        """Screen BOTH actual scaffold actions, not a separate lift."""
        mask = (1 << width) - 1
        zero = [[[]]]
        left, _, _ = lifted_matrices(G, grid, zero)
        _, right, _ = lifted_matrices(G, zero, grid)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        lp, rp = low_profile(left, width), low_profile(right, width)
        if lp is None or rp is None:
            return None
        return (gf2_rank(left), gf2_rank(right),
                min(lp[0], rp[0]), max(lp[1], rp[1]),
                max(lp[2], rp[2]))

    descriptors = set()
    directions = tuple(range(1, (q + 1) // 2))
    for p in range(4):
        steps = list(combinations(directions, p))
        for ai, upper in enumerate(steps):
            for lower in steps[ai:]:
                for tail in combinations(range(1, q), 6 - 2*p):
                    descriptors.add(canonical(
                        (0, upper, lower, (0,) + tail)))
    for tail in combinations(range(1, q), 5):
        descriptors.add(canonical((1, (), (), (0,) + tail)))
    for d in directions:
        for tail in combinations(range(1, q), 4):
            descriptors.add(canonical((2, (), (d,), (0,) + tail)))

    bank = []
    for card in sorted(descriptors):
        grid = build(card)
        info = constituent(grid)
        wanted_rank = 2*q - int(card[0] == 0)
        if info is None or info[0] != wanted_rank or info[1] != wanted_rank:
            continue
        bank.append((card, info, grid))
    if not bank:
        raise ValueError("no low-weight-screened constituent cards")

    def full_key(hx, hz, n):
        """Intrinsic uncoloured full-Tanner invariant.

        Degree >=10 identifies the high-degree qubits intrinsically.
        Checks contract to edges or marked groundings. Exact rooted
        joint walks, vertex labels and check intersections are invariant
        under arbitrary check/qubit relabelings.
        """
        rows = hx + hz
        cols = columns(rows, n)
        high = [j for j, c in enumerate(cols) if c.bit_count() >= 10]
        nv = len(high)
        if nv != 4*q:
            raise ValueError("unexpected high-degree sector")
        position = {j: i for i, j in enumerate(high)}
        high_mask = sum(1 << j for j in high)
        adjacency = [Counter() for _ in high]
        anchors = [Counter() for _ in high]
        weights = [Counter() for _ in high]
        for row in rows:
            bits = row & high_mask
            degree = bits.bit_count()
            if degree not in (1, 2):
                raise ValueError("check is neither contracted edge nor grounding")
            bit = bits & -bits
            u = position[bit.bit_length() - 1]
            if degree == 1:
                anchors[u][row.bit_count()] += 1
                continue
            v = position[(bits ^ bit).bit_length() - 1]
            adjacency[u][v] += 1
            adjacency[v][u] += 1
            weights[u][row.bit_count()] += 1
            weights[v][row.bit_count()] += 1

        labels = [
            (cols[j].bit_count(), tuple(sorted(anchors[i].items())),
             tuple(sorted(weights[i].items())))
            for i, j in enumerate(high)]
        symmetric = all(
            labels[i*q+h] == labels[i*q]
            and all(adjacency[i*q+h].get(j*q+t, 0)
                    == adjacency[i*q].get(j*q+(t-h) % q, 0)
                    for j in range(4) for t in range(q))
            for i in range(4) for h in range(q))
        roots = list(range(0, nv, q)) if symmetric else list(range(nv))
        neighbors = [tuple(sorted(a.items())) for a in adjacency]
        diagonals, joint = [], []
        for root in roots:
            vector = [0] * nv
            vector[root] = 1
            diagonal, features = [], [[] for _ in high]
            for length in range(1, 13):
                following = [0] * nv
                for u, value in enumerate(vector):
                    if value:
                        for v, multiplicity in neighbors[u]:
                            following[v] += value * multiplicity
                vector = following
                diagonal.append(vector[root])
                if length <= 8:
                    for v, value in enumerate(vector):
                        features[v].append(value)
            diagonals.append(tuple(diagonal))
            joint.append(features)
        colors = [
            (labels[v], diagonals[v//q if symmetric else v])
            for v in range(nv)]
        profiles = []
        for ri, root in enumerate(roots):
            profile = (
                colors[root],
                tuple(sorted((colors[v], tuple(joint[ri][v]))
                             for v in range(nv))))
            profiles.extend([profile] * (q if symmetric else 1))
        return (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(profiles)), _tanner_signature(hx, hz))

    def matching(values):
        return regular((), (), values)

    def ring(a, b, values):
        return regular((a,), (b,), values)

    # References are excluded, never emitted. Include the current two
    # classes and the additional regular/grounded classes shown here.
    two = regular((1, 3), (1, 3), (0, 2, 4))
    other_two = regular((1, 3), (1, 3), (0, 2, 6))
    one = ring(1, 1, (0, 2, 4, 6, 8))
    grounded = build((2, (), (1,), (0, 2, 4, 6, 8)))
    references = [
        (matching((0, 1, 2, 4, 5, 6, 10)),
         matching((0, 2, 3, 5, 6, 7, 9))),
        (ring(10, 10, (0, 2, 5, 7, 9)),
         ring(10, 10, (0, 2, 4, 7, 9))),
        (ring(6, 7, (0, 1, 4, 6, 8)),
         ring(4, 4, (0, 3, 6, 8, 9))),
        (ring(1, 7, (3, 4, 5, 6, 8)),
         ring(4, 6, (0, 1, 3, 6, 10))),
        (matching(tuple(range(7))), two),
        (one, two), (two, two), (one, other_two),
        (matching(tuple(range(7))),
         regular((1, 4), (1, 4), (0, 2, 5))),
        (two, grounded), (matching(tuple(range(7))), grounded),
        (matching(tuple(range(7))),
         build((1, (), (), tuple(range(6))))),
        (two, regular((2, 5), (2, 5), (0, 3, 7))),
        (two, regular((2, 3), (2, 3), (0, 1, 5))),
        (other_two, regular((1, 3), (1, 3), (0, 2, 7))),
        (two, regular((1, 4), (1, 4), (0, 2, 8))),
        (matching((0, 1, 2, 3, 4, 6, 7)),
         matching((0, 1, 2, 4, 5, 6, 8))),
        (one, [
            [[e] for _ in range(7)],
            [[x(v)] for v in (0, 1, 3, 4, 7, 9)] + [[]],
        ]),
        (one, regular((1, 2), (2, 5), (0, 3, 7))),
        (two, regular((1, 3), (1, 4), (0, 2, 4))),
        (two, regular((2, 5), (1, 3), (0, 3, 7))),
    ]
    blocked = set()
    for A, B in references:
        hx, hz, n = lifted_matrices(G, A, B)
        blocked.add(full_key(hx, hz, n))

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            # Two full-rank grounded sides would land below cell 0.500.
            if a[0][0] and b[0][0]:
                continue
            floor = min(a[1][2], b[1][2])
            internal6 = any(c[0] == 0 and len(c[1]) == 3
                            for c in (a[0], b[0]))
            tier = 0 if floor == 4 else (1 if internal6 else 2)
            regular_pair = a[0][0] == b[0][0] == 0
            for scale in range(1, q):
                plans.append((
                    tier, -int(regular_pair),
                    a[1][3] + b[1][3], a[1][4] + b[1][4],
                    ai, bi, scale))
    plans.sort()

    def no_two(rows, n):
        cols = columns(rows, n)
        return 0 not in cols and len(set(cols)) == n

    def logical_three(checks, stabilizers, n):
        """Find and verify an actual non-stabilizer weight-three word."""
        cols = columns(checks, n)
        location = {c: i for i, c in enumerate(cols)}
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        for i, first in enumerate(cols):
            for j in range(i):
                h = location.get(first ^ cols[j])
                if h is None or h in (i, j):
                    continue
                word = (1 << i) | (1 << j) | (1 << h)
                remainder = word
                while remainder:
                    p = remainder.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        assert word.bit_count() == 3
                        assert all((row & word).bit_count() % 2 == 0
                                   for row in checks)
                        return word
                    remainder ^= old
        return 0

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    results, orbits = [], set()
    four_count = 0
    # Do not let an unseen archived distance-four family consume every
    # slot: reserve at least half for structurally broader fallbacks.
    four_limit = min(10, count // 2)
    for tier, _, _, _, ai, bi, scale in plans:
        if tier == 0 and four_count >= four_limit:
            continue
        acard, bcard = bank[ai][0], image(bank[bi][0], scale)
        orbit = min(
            tuple(sorted((gauge(image(acard, u)),
                          gauge(image(bcard, u)))))
            for u in range(1, q))
        if orbit in orbits:
            continue
        orbits.add(orbit)
        A, B = bank[ai][2], build(bcard)
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 583 or max(r.bit_count() for r in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k not in (280, 287) or not (475*n <= 1000*k < 525*n):
            continue
        signature = full_key(hx, hz, n)
        if signature in blocked:
            continue
        if not (no_two(hx, n) and no_two(hz, n)):
            continue

        witness, witness_type = 0, None
        if tier == 0:
            if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
                continue
            floor = 4
        else:
            # Fallbacks earn a genuine distance-three certificate:
            # exclusion through two plus a verified logical of weight three.
            witness = logical_three(hz, hx, n)
            witness_type = "X"
            if not witness:
                witness = logical_three(hx, hz, n)
                witness_type = "Z"
            if not witness:
                continue
            floor = 3
        if not css_ok(hx, hz):
            raise ValueError("verified dual-tier lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified rank helpers disagree")
        blocked.add(signature)
        results.append((
            freeze(A), freeze(B), k, floor,
            (acard, bcard), witness_type, witness))
        four_count += int(floor == 4)
        if len(results) >= count:
            break
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an immutable screened structural card."""
    entries = _r050_dualtier_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("archive-screened dual-tier portfolio exhausted")
    A, B, k, floor, cards, kind, witness = entries[idx]
    return {
        "A": A, "B": B, "k": k,
        "verified_kernel_floor": floor,
        "structural_cards": cards,
        "logical_three_type": kind,
        "logical_three_witness": witness,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without modifying the cached portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The opposite-cycle singleton and symmetric-layer catalogue supplies twenty inequivalent designs: sixty measured-table slots collapsed to five distinct codes, all with d30s=4.
# REFUTED: Independent regular layers and joint-walk fingerprints through twelve supply five novel measured classes: the current sixty-slot evaluation retained only two measured classes, both with d30s=4, and fifty-eight isomorphic duplicates.
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
