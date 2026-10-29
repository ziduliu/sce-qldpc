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
CONCEPT = ("Cyclic Z11 = metacyclic Z11:Z1, order 11, A=B=(2,7): exhaustive independent-layer relative-voltage coverage and complete-Tanner rooted screening should find five unarchived regular distance-four codes, raising fitness toward 287*16/583=7.8765.")
GROUP = ("Keep the frozen group and order byte-for-byte. This bed measured distance-four codes, and q=11 is the largest lift allowed by size factor 53. The parent retained three regular and two grounded classes among sixty slots; broader regular coverage and archive avoidance, not a group-distance advantage, are tested.")
SHAPES = ("Keep equal 77-variable, 22-check constituents, n=583, q_max=11 and kb=275, guaranteed rate 25/53=0.47170. Every published full lift must have verified k=287, excess twelve and rate 0.49228 inside cell 0.500. Parent weight-four minima occupied sector-1 Acol/Bcol blocks broadly, so both constituents vary instead of retaining a fixed grounding.")
FILLING = ("Enumerate independent upper/lower internal directions and translated matching sets; restore relative group automorphisms when pairing canonical constituents. All columns have two terms and rows seven, giving both LP maxima nine. An immutable deterministic portfolio rejects shown archive references and equal complete uncoloured Tanner rooted-refinement invariants. Actual full ranks, CSS, short-kernel exclusion and non-stabilizer witnesses certify distance four or three. Distance-four regular pairs lead; genuinely different six-internal-column distance-three pairs are fallbacks, never relabeling padding. Five novel regular finalists and unseen archive coverage remain falsifiable.")
FAMILY_NAME = "r050_m11r1s1_27x27_regular_rank287_rooted_witness_portfolio_v4"

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


def _retired_singleton_cycle_context(rng, G, idx):
    """Uncalled historical catalogue; never publishes the new filling.

    Every accepted code, not merely its first twenty entries, passes
    the complete verified-matrix kernel screen through weight three.
    Different fingerprints certify non-isomorphism; equal fingerprints
    are conservatively discarded without claiming an isomorphism.
    """
    from itertools import combinations

    q, e, els = G.q, G.e, G.elements
    cap = min(9, W_MAX)

    def label_key(labels):
        return min(tuple(sorted((sign * s + t) % q for s in labels))
                   for sign in (1, -1) for t in range(q))

    def regular(upper, lower, labels):
        """Seven degree-two columns; row totals seven."""
        p = len(upper)
        assert len(lower) == p and len(labels) + 2 * p == 7
        return (
            [[e, els[d]] for d in upper]
            + [[e] for _ in labels] + [[] for _ in lower],
            [[] for _ in upper]
            + [[els[s]] for s in labels]
            + [[e, els[d]] for d in lower],
        )

    def card_cells(card):
        _, steps, labels = card
        return [list(row) for row in regular(steps, steps, labels)]

    # The singleton layer has no internal edges. On the opposite
    # layer, step one is an eleven-cycle. Matching offsets form an
    # independent set in that cycle, excluding component triangles.
    singleton_cycle = [
        [[e] for _ in range(5)] + [[], [e]],
        [[els[s]] for s in (0, 2, 4, 6, 8)]
        + [[e, els[1]], []],
    ]
    assert [sum(map(len, row)) for row in singleton_cycle] == [6, 7]

    # Canonical regular constituent cards. Common voltage scaling is
    # NOT removed here: the singleton-cycle fixes a relative step.
    by_profile = []
    for p in range(3):
        cards = set()
        for steps in combinations(range(1, (q + 1) // 2), p):
            signed = {d % q for d in steps}
            signed.update((-d) % q for d in steps)
            # Constructive triangle exclusion for internal edges.
            if any((a + b) % q in signed
                   for a in signed for b in signed):
                continue
            for labels in combinations(range(q), 7 - 2 * p):
                if any((a - b) % q in signed
                       for a, b in combinations(labels, 2)):
                    continue
                cards.add((p, steps, label_key(labels)))
        by_profile.append(sorted(cards))

    # Interleave support profiles instead of spending an initial
    # prefix entirely on one regular matching model.
    cards = []
    for turn in range(max(map(len, by_profile))):
        for profile in (2, 1, 0):
            if turn < len(by_profile[profile]):
                cards.append(by_profile[profile][turn])

    def columns(rows, n):
        result = [0] * n
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def fingerprint(hx, hz, n):
        """Intrinsic, uncoloured Tanner invariants.

        In addition to full check intersections, project each high
        qubit-degree class onto checks. Degree classes and the resulting
        adjacency/common-neighbour profiles survive arbitrary Tanner
        relabelings, including interchange of the two CSS types.
        """
        rows = hx + hz
        cols = columns(rows, n)
        degree_counts = {}
        for col in cols:
            d = col.bit_count()
            degree_counts[d] = degree_counts.get(d, 0) + 1

        projected = []
        for degree in sorted(d for d in degree_counts if d >= 10):
            adj = [0] * len(rows)
            for col in cols:
                if col.bit_count() != degree:
                    continue
                bits = col
                while bits:
                    bit = bits & -bits
                    adj[bit.bit_length() - 1] |= col ^ bit
                    bits ^= bit
            profiles = []
            for i, neighbors in enumerate(adj):
                histogram = {}
                for j, other in enumerate(adj):
                    if i == j:
                        continue
                    key = (rows[j].bit_count(), other.bit_count(),
                           (neighbors >> j) & 1,
                           (neighbors & other).bit_count())
                    histogram[key] = histogram.get(key, 0) + 1
                profiles.append(
                    (rows[i].bit_count(), neighbors.bit_count(),
                     tuple(sorted(histogram.items()))))
            projected.append((degree, tuple(sorted(profiles))))
        return (_tanner_signature(hx, hz),
                tuple(sorted(degree_counts.items())),
                tuple(projected))

    # Explicitly exclude the shown archived Z11 codes. These matrices
    # are only fingerprint references, never submitted as candidates.
    def old_cycle(d0, d1, labels):
        return [list(row) for row in regular((d0,), (d1,), labels)]

    def old_matching(labels):
        return [list(row) for row in regular((), (), labels)]

    archived = [
        (old_matching((0, 1, 2, 4, 5, 6, 10)),
         old_matching((0, 2, 3, 5, 6, 7, 9))),
        (old_cycle(10, 10, (0, 2, 5, 7, 9)),
         old_cycle(10, 10, (0, 2, 4, 7, 9))),
        (old_cycle(6, 7, (0, 1, 4, 6, 8)),
         old_cycle(4, 4, (0, 3, 6, 8, 9))),
        (old_cycle(1, 7, (3, 4, 5, 6, 8)),
         old_cycle(4, 6, (0, 1, 3, 6, 10))),
        (old_matching(tuple(range(7))),
         [list(row) for row in
          regular((1, 3), (1, 3), (0, 2, 4))]),
        (old_matching(tuple(range(7))),
         [[[e] for _ in range(7)],
          [[els[s]] for s in range(6)] + [[]]]),
    ]
    archive_keys = set()
    for A, B in archived:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(fingerprint(hx, hz, n))

    def scaled_card(card, u):
        p, steps, labels = card
        ds = tuple(sorted(min((u * d) % q, (-u * d) % q)
                          for d in steps))
        return p, ds, label_key(tuple((u * s) % q for s in labels))

    def regular_pair_key(a, b):
        # Common automorphisms, column gauges, row swaps and A/B swap.
        return min(tuple(sorted((scaled_card(a, u),
                                 scaled_card(b, u))))
                   for u in range(1, q))

    # Primary asymmetric cards, then cycle-rich regular pairs, then
    # the remaining canonical pairs if extra slots are requested.
    cycle_cards = by_profile[2] + by_profile[1]
    plans = [(a, None) for a in cards]
    for turn in range(len(cycle_cards)):
        for ai, a in enumerate(cards):
            plans.append((a, cycle_cards[(turn + ai) % len(cycle_cards)]))
    plans.extend((a, b) for ai, a in enumerate(cards)
                 for b in cards[ai:])

    graph_keys, orbit_keys = set(), set()
    accepted = 0
    for a, b in plans:
        orbit = (("opposite-cycle-singleton", a) if b is None
                 else ("regular", regular_pair_key(a, b)))
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)

        A = card_cells(a)
        B = singleton_cycle if b is None else card_cells(b)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        if max(row.bit_count() for row in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k not in (280, 287) or not (475 * n <= 1000 * k < 525 * n):
            continue

        signature = fingerprint(hx, hz, n)
        if signature in archive_keys or signature in graph_keys:
            continue
        # This stronger-than-logical screen excludes all commuting
        # supports of weights one, two and three in both CSS types.
        if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified singleton-cycle lift failed CSS")
        graph_keys.add(signature)

        if accepted == idx:
            return {"A": A, "B": B, "k": k,
                    "card": (a, b), "kernel_floor": 4}
        accepted += 1

    raise ValueError("novel kernel-three-screened catalogue exhausted")


from functools import lru_cache


@lru_cache(maxsize=1)
def _regular_rank287_portfolio(G, count):
    """Immutable exhaustive portfolio; memoization changes only its cost.

    Constituent automorphisms prune proposals, but relative orientations
    are restored explicitly. Publication is separated by an intrinsic
    invariant of the COMPLETE uncoloured Tanner graph.
    """
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    q, width, cap = G.q, 7 * G.q, min(9, W_MAX)
    if q != 11 or cap < 9:
        raise ValueError("regular rank287 portfolio requires q=11 and cap nine")
    elements = {G._key(g): g for g in G.elements}
    mask = (1 << width) - 1

    def x(t):
        return elements[(t % q, 0)]

    def translation(values):
        return min(tuple(sorted((v - a) % q for v in values))
                   for a in values)

    def gauge(card):
        upper, lower, values = card
        forward = (upper, lower, translation(values))
        reverse = (lower, upper,
                   translation(tuple((-v) % q for v in values)))
        return min(forward, reverse)

    def image(card, u):
        upper, lower, values = card
        upper = tuple(sorted(min(u*d % q, -u*d % q) for d in upper))
        lower = tuple(sorted(min(u*d % q, -u*d % q) for d in lower))
        return gauge((upper, lower, tuple(u*v % q for v in values)))

    def canonical(card):
        return min(image(card, u) for u in range(1, q))

    def build(card):
        upper, lower, values = card
        p = len(upper)
        if len(lower) != p or len(values) + 2*p != 7:
            raise ValueError("wrong regular constituent card")
        grid = [
            [[G.e, x(d)] for d in upper]
            + [[G.e] for _ in values] + [[] for _ in lower],
            [[] for _ in upper]
            + [[x(v)] for v in values]
            + [[G.e, x(d)] for d in lower],
        ]
        assert all(len(row) == 7 and sum(map(len, row)) == 7
                   for row in grid)
        assert all(sum(len(grid[i][j]) for i in range(2)) == 2
                   for j in range(7))
        assert all(len({G._key(g) for g in cell}) == len(cell)
                   for row in grid for cell in row)
        return grid

    def columns(rows, n):
        result = [0] * n
        for r, original in enumerate(rows):
            row = original & ((1 << n) - 1)
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def low_counts(rows):
        """Exact constituent three/four-word counts in verified coordinates."""
        cols = columns(rows, width)
        singles = set(cols)
        if 0 in singles or len(singles) != width:
            return None
        pair_counts, threes, collisions = {}, 0, 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                threes += int(syndrome in singles)
                old = pair_counts.get(syndrome, 0)
                collisions += old
                pair_counts[syndrome] = old + 1
        # A three-word has three pair/single partitions; a four-word
        # has three unordered pair/pair partitions.
        assert threes % 3 == collisions % 3 == 0
        return threes // 3, collisions // 3

    def graph_key(hx, hz, n):
        """Intrinsic rooted refinement of the complete Tanner graph.

        No CSS colours or protograph labels enter refinement. In this
        frozen abelian lift, simultaneous group translation is a Tanner
        automorphism. The four high-degree block representatives recover
        the complete multiset of high-degree rooted invariants.
        """
        rows = list(hx) + list(hz)
        adjacency = [[] for _ in range(n + len(rows))]
        for r, original in enumerate(rows):
            row = original
            while row:
                bit = row & -row
                j = bit.bit_length() - 1
                adjacency[j].append(n + r)
                adjacency[n + r].append(j)
                row ^= bit
        degrees = [len(neighbours) for neighbours in adjacency]
        high = [j for j in range(n) if degrees[j] >= 12]
        if high != list(range(49*q, n)):
            raise ValueError("unexpected intrinsic high-degree root class")

        rooted = []
        for block in range(4):
            root = 49*q + block*q + G.idx(G.e)
            initial = [(int(v >= n), degrees[v], int(v == root))
                       for v in range(len(adjacency))]
            unique = sorted(set(initial))
            palette = {key: i for i, key in enumerate(unique)}
            colors = [palette[key] for key in initial]
            digest = sha256()
            digest.update(repr(tuple(sorted(Counter(initial).items())))
                          .encode("ascii"))
            previous = len(unique)
            for _ in range(14):
                keys = [
                    (colors[v],
                     tuple(sorted(colors[w] for w in neighbours)))
                    for v, neighbours in enumerate(adjacency)
                ]
                digest.update(repr(tuple(sorted(Counter(keys).items())))
                              .encode("ascii"))
                unique = sorted(set(keys))
                palette = {key: i for i, key in enumerate(unique)}
                colors = [palette[key] for key in keys]
                if len(unique) == previous:
                    break
                previous = len(unique)
            rooted.extend([digest.digest()] * q)

        intersections = Counter(
            (rows[i] & rows[j]).bit_count()
            for i in range(len(rows)) for j in range(i))
        return (
            tuple(sorted(Counter(degrees[:n]).items())),
            tuple(sorted(Counter(degrees[n:]).items())),
            tuple(sorted(intersections.items())),
            tuple(sorted(rooted)),
        )

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return True
            word ^= old
        return False

    def witness(checks, stabilizers, n, weight):
        """Find and verify a non-stabilizer logical of weight three/four."""
        cols, own = columns(checks, n), span(stabilizers)
        if weight == 3:
            singles = {s: i for i, s in enumerate(cols)}
            for i, first in enumerate(cols):
                for j in range(i):
                    h = singles.get(first ^ cols[j])
                    if h is None or h in (i, j):
                        continue
                    word = (1 << i) | (1 << j) | (1 << h)
                    if outside(word, own):
                        assert all((word & row).bit_count() % 2 == 0
                                   for row in checks)
                        return word
            return None

        pairs = {}
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = (1 << i) | (1 << j)
                old = pairs.get(syndrome)
                if old is None:
                    pairs[syndrome] = word
                    continue
                difference = word ^ old
                if (difference.bit_count() == 4
                        and outside(difference, own)):
                    assert all((difference & row).bit_count() % 2 == 0
                               for row in checks)
                    return difference
        return None

    # Exhaust all internal allocations, not just symmetric layers.
    # Six-internal-column cards are new structural fallbacks.
    descriptors = set()
    for p in range(4):
        directions = list(combinations(range(1, (q + 1)//2), p))
        for ai, upper in enumerate(directions):
            for lower in directions[ai:]:
                for tail in combinations(range(1, q), 6 - 2*p):
                    descriptors.add(canonical(
                        (upper, lower, (0,) + tail)))

    clean, broad = [], []
    broad_profiles = set()
    zero = [[[]]]
    for card in sorted(descriptors):
        grid = build(card)
        left, _, _ = lifted_matrices(G, grid, zero)
        _, right, _ = lifted_matrices(G, zero, grid)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        if gf2_rank(left) != 21 or gf2_rank(right) != 21:
            continue
        a, b = low_counts(left), low_counts(right)
        if a is None or b is None:
            continue
        record = (card, grid, max(a[1], b[1]), a[0] == b[0] == 0)
        if record[3]:
            clean.append(record)
        elif len(card[0]) == 3:
            profile = (_tanner_signature(left, []), a, b)
            if profile not in broad_profiles:
                broad_profiles.add(profile)
                broad.append(record)
    if not clean:
        raise ValueError("no verified regular kernel-three-clean constituents")
    bank = clean + broad

    # Exclusion references only: never published as candidate fillings.
    def matching(values):
        return ((), (), values)

    def ring(a, b, values):
        return ((a,), (b,), values)

    def two(a, b, values):
        return ((a, b), (a, b), values)

    archived = [
        (matching((0,1,2,4,5,6,10)), matching((0,2,3,5,6,7,9))),
        (ring(10,10,(0,2,5,7,9)), ring(10,10,(0,2,4,7,9))),
        (ring(6,7,(0,1,4,6,8)), ring(4,4,(0,3,6,8,9))),
        (ring(1,7,(3,4,5,6,8)), ring(4,6,(0,1,3,6,10))),
        (matching(tuple(range(7))), two(1,3,(0,2,4))),
        (ring(1,1,(0,2,4,6,8)), two(1,3,(0,2,4))),
        (two(1,3,(0,2,4)), two(1,3,(0,2,4))),
        (ring(1,1,(0,2,4,6,8)), two(1,3,(0,2,6))),
        (matching(tuple(range(7))), two(1,4,(0,2,5))),
        (two(1,3,(0,2,6)), two(1,3,(0,2,7))),
        (two(1,3,(0,2,4)), two(1,4,(0,2,8))),
        (matching((0,1,2,3,4,6,7)), matching((0,1,2,4,5,6,8))),
        (ring(1,1,(0,2,4,6,8)), ((1,2),(2,5),(0,3,7))),
        (two(1,3,(0,2,4)), ((1,3),(1,4),(0,2,4))),
        (two(1,3,(0,2,4)), ((1,3),(2,5),(0,3,7))),
        (two(1,3,(0,2,4)), two(2,5,(0,3,7))),
        (two(1,3,(0,2,4)), two(2,3,(0,1,5))),
    ]
    blocked = set()
    for a, b in archived:
        hx, hz, n = lifted_matrices(G, build(a), build(b))
        blocked.add(graph_key(hx, hz, n))

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            floor = 4 if a[3] and b[3] else 3
            if floor == 3 and max(len(a[0][0]), len(b[0][0])) != 3:
                continue
            asymmetric = int(a[0][0] != a[0][1]) + int(b[0][0] != b[0][1])
            for u in range(1, q):
                tie = sha256(repr(
                    ("regular-rank287-v4", a[0], b[0], u)
                ).encode("ascii")).digest()
                plans.append((-floor, -asymmetric,
                              max(a[2], b[2]), a[2] + b[2],
                              tie, ai, bi, u))
    plans.sort()

    entries, orbit_keys, graph_keys = [], set(), set()
    for neg_floor, _, _, _, _, ai, bi, u in plans:
        a, b = bank[ai], bank[bi]
        bcard = image(b[0], u)
        orbit = min(
            tuple(sorted((image(a[0], v), image(bcard, v))))
            for v in range(1, q))
        if orbit in orbit_keys:
            continue
        orbit_keys.add(orbit)
        A, B = a[1], build(bcard)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 583 or k != 287:
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        floor = -neg_floor
        legal = True
        for checks in (hx, hz):
            cols = columns(checks, n)
            if 0 in cols or len(set(cols)) != n:
                legal = False
                break
            if floor == 4 and not _no_short_kernel(checks, n):
                legal = False
                break
        if not legal:
            continue
        signature = graph_key(hx, hz, n)
        if signature in blocked or signature in graph_keys:
            continue
        word = witness(hz, hx, n, floor)
        kind = "X"
        if word is None:
            word = witness(hx, hz, n, floor)
            kind = "Z"
        if word is None:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified regular portfolio failed CSS")
        graph_keys.add(signature)

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        entries.append((
            freeze(A), freeze(B), k, floor, kind, word,
            (a[0], bcard), signature))
        if len(entries) >= count:
            break
    # Exhaustion is explicit; no orbit copies fill missing positions.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication of an immutable verified regular design."""
    entries = _regular_rank287_portfolio(G, N_SLOTS)
    if not (0 <= idx < len(entries)):
        raise ValueError("novel regular rank287 catalogue exhausted")
    A, B, k, distance, kind, word, cards, signature = entries[idx]
    return {
        "A": A, "B": B, "k": k,
        "exact_short_distance": distance,
        "verified_logical_type": kind, "verified_logical": word,
        "cards": cards, "intrinsic_signature": signature,
        "certified_score": k * distance * distance / 583,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh complete cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: Opposite-cycle singleton priority supplies twenty inequivalent designs or escapes distance four: sixty evaluated slots retained only five classes, all measured at distance four, including two k=280 grounded finalists.
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
