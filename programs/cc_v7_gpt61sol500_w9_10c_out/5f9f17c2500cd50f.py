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
CONCEPT = ("Generalized quaternion Q16, order 16, A=B=(2,5): girth-eight matching cores with two cubic reinforcements should supply five novel measured distance-eight candidates in cell 0.300; exact constituent screening through seven does not certify the complete quantum distance.")
GROUP = ("Jump from cyclic Z11 to Q16. Order-eight rotations and order-four quaternion elements permit noncommuting matching cores without involutory reflections. The affine order-20 (2,5) bed measured five distance-seven finalists and fitness 15.207, motivating reinforcement screening rather than a transferred distance guarantee. Q16 fixes q=16 and n=464.")
SHAPES = ("Equal (2,5) constituents have 80 variables and 32 checks. The size factor is 29, q_max=20, n=464, kb=144 and guaranteed rate 9/29=0.310345 in cell 0.300. Actual complete ranks enforce [0.275,0.325). Parent weight-four witnesses occupied sector-1 Acol/Bcol blocks broadly; reinforced degree-three columns replace the degree-two constituent bottleneck.")
FILLING = ("Twenty support/core/order cards optimize isolated trinomials or split-binomial cubic columns around three matching columns. Each constituent row has six terms and each column at most three, giving BOTH CSS maxima nine, also below twelve. Verified matching-core girth and exact graph joins cover constituent supports through seven, with floor-seven fallbacks. Complete verified lifts enforce ranks, CSS and kernel exclusion through four. Intrinsic uncoloured Tanner intersection and high-degree-qubit walk invariants separate the immutable portfolio. Explicit non-stabilizer logical witnesses and additional probes rank pairs; larger quantum distance and unseen archive novelty remain measurement hypotheses.")
FAMILY_NAME = "r030_quaternion16_25x25_exactjoin7_reinforced_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Q16 requires order sixteen")
    # x^8=1, y^2=x^4, y*x*y^-1=x^-1.
    # Canonical element (a,b) represents x^a y^b.
    els = [(a, b) for b in range(2) for a in range(8)]

    def red(g):
        a, b = g
        return ((a + 4*(b // 2)) % 8, b % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 4*b*d) % 8,
                (b + d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a) % 8, 0) if b == 0 else ((a + 4) % 8, 1)

    def idx(g):
        a, b = red(g)
        return 8*b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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


def _parent_singleton_catalogue_reference(rng, G, idx):
    """Uncalled historical catalogue; never publishes a candidate.

    Rebuild a deterministic catalogue; no mutable cross-slot state.

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
def _relative_voltage_portfolio(G):
    """Memoize an immutable, deterministic complete portfolio.

    Memoization changes cost, not results or slot ownership. All matrix
    properties come from the scaffold lift. Equal intrinsic fingerprints
    are rejected conservatively, not declared isomorphic.
    """
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    q, cap = G.q, min(9, W_MAX)
    if q != 11:
        raise ValueError("relative-voltage portfolio requires order eleven")
    elements = {G._key(g): g for g in G.elements}

    def x(t):
        return elements[(t % q, 0)]

    def translations(values):
        return min(tuple(sorted((v-anchor) % q for v in values))
                   for anchor in values)

    def normalize(card):
        grounded, upper, lower, values = card
        direct = (grounded, tuple(sorted(upper)), tuple(sorted(lower)),
                  translations(values))
        if grounded:
            return direct
        reverse = (0, direct[2], direct[1],
                   translations(tuple(-v % q for v in values)))
        return min(direct, reverse)

    def scaled(card, u):
        grounded, upper, lower, values = card
        a = tuple(sorted(min(u*d % q, -u*d % q) for d in upper))
        b = tuple(sorted(min(u*d % q, -u*d % q) for d in lower))
        return normalize((grounded, a, b,
                          tuple(u*v % q for v in values)))

    def pair_key(a, b):
        # Only COMMON automorphisms are removed here. Independently
        # scaling the constituents earlier would discard relative twists.
        return min(tuple(sorted((scaled(a, u), scaled(b, u))))
                   for u in range(1, q))

    def build(card):
        grounded, upper, lower, values = card
        if grounded:
            assert not upper and not lower and len(values) == 6
            return [
                [[G.e] for _ in values] + [[G.e]],
                [[x(v)] for v in values] + [[]],
            ]
        assert len(upper) == len(lower)
        assert len(upper) + len(lower) + len(values) == 7
        return [
            [[G.e, x(d)] for d in upper]
            + [[G.e] for _ in values] + [[] for _ in lower],
            [[] for _ in upper]
            + [[x(v)] for v in values]
            + [[G.e, x(d)] for d in lower],
        ]

    def columns(rows, width):
        result = [0] * width
        mask = (1 << width) - 1
        for r, row in enumerate(rows):
            row &= mask
            tag = 1 << r
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= tag
                row ^= bit
        return result

    def four_kernel_count(cols):
        # Called only after exclusion through weight three.
        counts, collisions = {}, 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                old = counts.get(syndrome, 0)
                collisions += old
                counts[syndrome] = old + 1
        assert collisions % 3 == 0
        return collisions // 3

    def constituent_info(grid):
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
        width = 7*q
        mask = (1 << width) - 1
        rows = [row & mask for row in rows]
        if not _no_short_kernel(rows, width):
            return None
        return gf2_rank(rows), four_kernel_count(columns(rows, width))

    def tanner_key(hx, hz, width):
        """Rooted refinement of the actual uncoloured Tanner graph.

        All high-degree qubits are roots intrinsically. In this cyclic
        lift, simultaneous coordinate translation is an automorphism,
        so the four sector-2 orbit representatives each stand for q roots.
        No X/Z check colour or voltage label enters refinement.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, width)
        high = [j for j, c in enumerate(cols) if c.bit_count() >= 10]
        if len(high) != 4*q:
            raise ValueError("unexpected incidence degree partition")

        adjacency = [[] for _ in range(width + len(rows))]
        for r, row in enumerate(rows):
            check = width + r
            while row:
                bit = row & -row
                j = bit.bit_length() - 1
                adjacency[j].append(check)
                adjacency[check].append(j)
                row ^= bit

        degrees = [len(a) for a in adjacency]
        representatives = high[::q]
        roots = []
        for root in representatives:
            features = [(degrees[v], int(v == root))
                        for v in range(len(adjacency))]
            palette = {f: i for i, f in enumerate(sorted(set(features)))}
            colors = [palette[f] for f in features]
            digest = sha256()
            digest.update(repr(tuple(sorted(Counter(features).items())))
                          .encode("ascii"))
            for _ in range(9):
                features = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(len(adjacency))
                ]
                histogram = tuple(sorted(Counter(features).items()))
                digest.update(repr(histogram).encode("ascii"))
                palette = {f: i for i, f in
                           enumerate(sorted(set(features)))}
                following = [palette[f] for f in features]
                colors = following
            roots.extend([digest.digest()] * q)

        return (
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(roots)),
        )

    def basis(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                if p not in pivots:
                    pivots[p] = row
                    break
                row ^= pivots[p]
        return pivots

    def logical_four_count(checks, stabilizers, width):
        """Exactly enumerate weight-four logical supports.

        Kernel-three exclusion makes distinct equal-syndrome pairs
        disjoint. Every four-support has three pair partitions; the set
        removes that repetition. Stabilizer membership is tested exactly.
        """
        cols = columns(checks, width)
        own = basis(stabilizers)
        seen, logicals = {}, set()

        def inspect(word):
            assert word.bit_count() == 4
            reduced = word
            while reduced:
                p = reduced.bit_length() - 1
                if p not in own:
                    logicals.add(word)
                    return
                reduced ^= own[p]

        for i, first in enumerate(cols):
            bit = 1 << i
            for j in range(i):
                syndrome = first ^ cols[j]
                word = bit | (1 << j)
                old = seen.get(syndrome)
                if old is None:
                    seen[syndrome] = word
                elif isinstance(old, int):
                    inspect(word ^ old)
                    seen[syndrome] = [old, word]
                else:
                    for previous in old:
                        inspect(word ^ previous)
                    old.append(word)
        return len(logicals)

    # Exhaustive support/voltage cards, not independent slot draws.
    # Independent upper/lower sets are proposed; actual triangle and
    # verified-kernel screens decide whether they are admissible.
    regular_cards = set()
    for p in range(4):
        options = []
        for steps in combinations(range(1, (q+1)//2), p):
            signed = set(steps) | {-d % q for d in steps}
            if not any((a+b) % q in signed
                       for a in signed for b in signed):
                options.append(steps)
        count = 7 - 2*p
        if count < 1:
            continue
        for upper in options:
            for lower in options:
                forbidden = set(upper + lower)
                forbidden |= {-d % q for d in forbidden}
                for tail in combinations(range(1, q), count-1):
                    values = (0,) + tail
                    if any((a-b) % q in forbidden
                           for a, b in combinations(values, 2)):
                        continue
                    regular_cards.add(normalize(
                        (0, upper, lower, values)))

    grounded_cards = {
        normalize((1, (), (), (0,) + tail))
        for tail in combinations(range(1, q), 5)
    }
    bank = {}
    for card in sorted(regular_cards | grounded_cards):
        grid = build(card)
        info = constituent_info(grid)
        if info is not None:
            bank[card] = info

    regular_cards = sorted(c for c in bank if c[0] == 0)
    grounded_cards = sorted(c for c in bank if c[0] == 1)
    if not regular_cards:
        raise ValueError("no verified regular incidence cards")

    # References are exclusion data only, never portfolio entries.
    def regular(a, b, values):
        return build((0, tuple(a), tuple(b), tuple(values)))

    def matching(values):
        return regular((), (), values)

    def ring(a, b, values):
        return regular((a,), (b,), values)

    two = regular((1, 3), (1, 3), (0, 2, 4))
    ring_one = ring(1, 1, (0, 2, 4, 6, 8))
    opposite_ground = [
        [[G.e] for _ in range(5)] + [[], [G.e]],
        [[x(v)] for v in (0, 2, 4, 6, 8)]
        + [[G.e, x(1)], []],
    ]
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
        (ring_one, two),
        (two, two),
        (ring_one, regular((1, 3), (1, 3), (0, 2, 6))),
        (matching(tuple(range(7))),
         regular((1, 4), (1, 4), (0, 2, 5))),
        (regular((1, 3), (1, 3), (0, 2, 6)),
         regular((1, 3), (1, 3), (0, 2, 7))),
        (two, regular((1, 4), (1, 4), (0, 2, 8))),
        (matching((0, 1, 2, 3, 4, 6, 7)),
         matching((0, 1, 2, 4, 5, 6, 8))),
        (two, regular((2, 5), (2, 5), (0, 3, 7))),
        (two, regular((2, 3), (2, 3), (0, 1, 5))),
        (two, opposite_ground),
        (matching(tuple(range(7))), opposite_ground),
        (matching(tuple(range(7))),
         build((1, (), (), tuple(range(6))))),
    ]
    archive_keys = set()
    for A, B in references:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(tanner_key(hx, hz, n))

    plans = set()
    for ai, a in enumerate(regular_cards):
        for b in regular_cards[ai:]:
            plans.add(pair_key(a, b))
        for b in grounded_cards:
            plans.add(pair_key(a, b))

    # Common scaling preserves component ranks and kernel multiplicities.
    def plan_quality(pair):
        a, b = pair
        ia, ib = constituent_info(build(a)), constituent_info(build(b))
        return (int(b[0] != 0), max(ia[1], ib[1]),
                ia[1] + ib[1], pair)

    ordered = sorted(plans, key=plan_quality)
    winners = {}

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for a, b in ordered:
        A, B = build(a), build(b)
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 583 or max(row.bit_count() for row in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k not in (280, 287) or not (475*n <= 1000*k < 525*n):
            continue
        if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
            continue

        signature = tanner_key(hx, hz, n)
        if signature in archive_keys or signature in winners:
            continue

        nx = logical_four_count(hz, hx, n)
        nz = logical_four_count(hx, hz, n)
        # Counts are exact selection features, not larger-distance claims.
        quality = (k, -max(nx, nz), -(nx+nz), signature)
        if not css_ok(hx, hz):
            raise ValueError("verified relative-voltage pair failed CSS")
        winners[signature] = (
            quality, freeze(A), freeze(B), k, nx, nz, (a, b))

    selected = sorted(winners.values(), key=lambda r: r[0], reverse=True)
    return tuple((A, B, k, nx, nz, cards)
                 for _, A, B, k, nx, nz, cards in selected[:N_SLOTS])


def _q16_columns(rows, width):
    """Column syndromes of the verified scaffold matrices."""
    out = [0] * width
    mask = (1 << width) - 1
    for r, row in enumerate(rows):
        row &= mask
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _q16_clear_four(cols):
    """Exact exclusion of all nonzero kernel supports of weights 1..4."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    pairs = set()
    for i, first in enumerate(cols):
        for second in cols[:i]:
            syndrome = first ^ second
            if syndrome in singles or syndrome in pairs:
                return False
            pairs.add(syndrome)
    return True


def _q16_basis(rows):
    pivots = {}
    for word in rows:
        while word:
            p = word.bit_length() - 1
            if p not in pivots:
                pivots[p] = word
                break
            word ^= pivots[p]
    return pivots


def _q16_outside(word, pivots):
    while word:
        p = word.bit_length() - 1
        if p not in pivots:
            return True
        word ^= pivots[p]
    return False


def _q16_component_rows(G, grid, right=False):
    """Expose the exact constituent action; the zero dummy is never output."""
    width = len(grid[0]) * G.q
    dummy = [[[]]]
    if right:
        _, rows, _ = lifted_matrices(G, dummy, grid)
    else:
        rows, _, _ = lifted_matrices(G, grid, dummy)
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _q16_probe(checks, stabilizers, width, local, seed, passes):
    """Return an actual logical witness, never an unresolved distance cap."""
    cols = _q16_columns(checks, width)
    own = _q16_basis(stabilizers)
    if not seed or not _q16_outside(seed, own):
        return 0
    best, weight = seed, seed.bit_count()

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if word and w < weight and _q16_outside(word, own):
            best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        local.shuffle(order)
        pivots, kernel = {}, []
        for j in order:
            syndrome, word = cols[j], 1 << j
            while syndrome:
                p = syndrome.bit_length() - 1
                previous = pivots.get(p)
                if previous is None:
                    pivots[p] = (syndrome, word)
                    break
                syndrome ^= previous[0]
                word ^= previous[1]
            if not syndrome:
                kernel.append(word)
                inspect(word)
        local.shuffle(kernel)
        for i, word in enumerate(kernel):
            for j, other in enumerate(kernel):
                if i != j:
                    trial = word ^ other
                    if 0 < trial.bit_count() < word.bit_count():
                        word = trial
            kernel[i] = word
            inspect(word)
        for i, word in enumerate(kernel):
            for other in kernel[:i]:
                inspect(word ^ other)
        short = sorted(kernel, key=int.bit_count)[:20]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    inspect(pair ^ short[h])
        if weight <= 6:
            break
    assert all((row & best).bit_count() % 2 == 0 for row in checks)
    assert _q16_outside(best, own)
    return best


def _q16_graph_key(hx, hz, width):
    """Intrinsic invariant of the complete uncoloured Tanner graph.

    Equal keys are conservatively rejected. No group labels, CSS colours
    or protograph coordinates enter the extracted graph features.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _q16_columns(rows, width)
    high = [j for j, col in enumerate(cols) if col.bit_count() >= 10]
    position = {j: i for i, j in enumerate(high)}
    mask = sum(1 << j for j in high)
    adjacency = [Counter() for _ in high]
    incident = [Counter() for _ in high]
    for row in rows:
        endpoints = []
        bits = row & mask
        while bits:
            bit = bits & -bits
            endpoints.append(position[bit.bit_length() - 1])
            bits ^= bit
        for u in endpoints:
            incident[u][(row.bit_count(), len(endpoints))] += 1
            for v in endpoints:
                if u != v:
                    adjacency[u][v] += 1

    size = len(high)
    colors = [(cols[j].bit_count(), tuple(sorted(incident[i].items())))
              for i, j in enumerate(high)]
    power = [[int(i == j) for j in range(size)] for i in range(size)]
    walks = [[[] for _ in range(size)] for _ in range(size)]
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
                walks[u][v].append(power[u][v])
    profiles = tuple(sorted(
        (colors[u], tuple(walks[u][u]),
         tuple(sorted((colors[v], tuple(walks[v][v]),
                       tuple(walks[u][v])) for v in range(size))))
        for u in range(size)))
    intrinsic = (
        width, len(rows),
        tuple(sorted(Counter(row.bit_count() for row in rows).items())),
        tuple(sorted(Counter(col.bit_count() for col in cols).items())),
        _tanner_signature(hx, hz), profiles)
    return sha256(repr(intrinsic).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _q16_join_portfolio(G):
    """Immutable deterministic portfolio of optimized structural cards."""
    from itertools import combinations

    q, e = G.q, G.e
    if q != 16:
        raise ValueError("reinforced quaternion portfolio requires Q16")
    width, cap = 5*q, min(9, W_MAX, 12)

    def order(g):
        value = e
        for length in range(1, q + 1):
            value = G.mul(value, g)
            if value == e:
                return length
        raise ValueError("invalid element order")

    orders = {g: order(g) for g in G.elements}
    nonzero = [g for g in G.elements if g != e]
    pools = {t: [g for g in nonzero if orders[g] == t]
             for t in (2, 4, 8)}

    def core_data(g, h):
        grid = [[[e], [e], [e]], [[e], [g], [h]]]
        rows, cw = _q16_component_rows(G, grid)
        cols = _q16_columns(rows, cw)
        adjacency = [[] for _ in rows]
        edges = set()
        for j, syndrome in enumerate(cols):
            if syndrome.bit_count() != 2:
                return None
            bit = syndrome & -syndrome
            a = bit.bit_length() - 1
            b = (syndrome ^ bit).bit_length() - 1
            edge = (min(a, b), max(a, b))
            if edge in edges:
                return None
            edges.add(edge)
            adjacency[a].append((b, j))
            adjacency[b].append((a, j))

        distances, paths, cycle = [], [], 0
        for root in range(len(rows)):
            dist, path = [10000] * len(rows), [0] * len(rows)
            dist[root] = 0
            queue = [root]
            for a in queue:
                for b, edge in adjacency[a]:
                    if dist[b] == 10000:
                        dist[b] = dist[a] + 1
                        path[b] = path[a] ^ (1 << edge)
                        queue.append(b)
                    else:
                        word = path[a] ^ path[b] ^ (1 << edge)
                        if word and (not cycle or
                                     word.bit_count() < cycle.bit_count()):
                            cycle = word
            distances.append(tuple(dist))
            paths.append(tuple(path))
        if not cycle or cycle.bit_count() < 8:
            return None
        assert all((row & cycle).bit_count() % 2 == 0 for row in rows)
        return (g, h, tuple(cols), tuple(distances), tuple(paths), cycle)

    cores = [[], []]
    for g, h in combinations(nonzero, 2):
        if G.mul(g, h) == G.mul(h, g):
            continue
        data = core_data(g, h)
        if data is not None:
            cores[int(orders[g] == orders[h])].append(data)
    if not any(cores):
        raise ValueError("no verified Q16 matching core of girth eight")
    for kind in range(2):
        if not cores[kind]:
            cores[kind] = list(cores[1-kind])

    def join_screen(rows, data):
        """Exact constituent exclusion through seven, retaining a witness.

        A kernel uses an even number of cubic columns. Supports through
        seven therefore contain zero, two, four or six cubic variables.
        Anchors use only the constituent's verified regular-action symmetry.
        """
        cols = _q16_columns(rows, width)
        if tuple(cols[:3*q]) != data[2] or not _q16_clear_four(cols):
            return None
        cubic = list(range(3*q, width))
        if any(cols[j].bit_count() != 3 for j in cubic):
            raise ValueError("reinforcement lost its cubic column degree")
        anchors = (3*q + G.idx(e), 4*q + G.idx(e))
        distances, paths = data[3], data[4]
        best, memo = data[5], {}

        def tjoin(boundary, budget):
            if not boundary:
                return (0, 0)
            if budget < 0 or boundary.bit_count() > 2*budget:
                return None
            key = (boundary, budget)
            if key in memo:
                return memo[key]
            bit = boundary & -boundary
            a = bit.bit_length() - 1
            rest, answer = boundary ^ bit, None
            choices = rest
            while choices:
                other = choices & -choices
                b = other.bit_length() - 1
                distance = distances[a][b]
                if distance <= budget:
                    tail = tjoin(rest ^ other, budget-distance)
                    if tail is not None:
                        result = (distance + tail[0], paths[a][b] ^ tail[1])
                        if answer is None or result[0] < answer[0]:
                            answer = result
                choices ^= other
            memo[key] = answer
            return answer

        for anchor in anchors:
            for other in cubic:
                if other == anchor:
                    continue
                result = tjoin(cols[anchor] ^ cols[other],
                               min(5, best.bit_count()-3))
                if result is not None:
                    word = (1 << anchor) ^ (1 << other) ^ result[1]
                    if word.bit_count() < best.bit_count():
                        best = word
                        if best.bit_count() <= 6:
                            return None

        triples = {}
        for a, b, c in combinations(cubic, 3):
            syndrome = cols[a] ^ cols[b] ^ cols[c]
            word = (1 << a) | (1 << b) | (1 << c)
            if syndrome in triples:
                return None
            triples[syndrome] = word

        for anchor in anchors:
            others = [j for j in cubic if j != anchor]
            for a, b, c in combinations(others, 3):
                boundary = cols[anchor] ^ cols[a] ^ cols[b] ^ cols[c]
                budget = min(3, best.bit_count()-5)
                if boundary.bit_count() > 2*budget:
                    continue
                result = tjoin(boundary, budget)
                if result is not None:
                    word = ((1 << anchor) | (1 << a) |
                            (1 << b) | (1 << c)) ^ result[1]
                    if word.bit_count() < best.bit_count():
                        best = word
                        if best.bit_count() <= 6:
                            return None
        if best.bit_count() == 7:
            return (7, best)

        for anchor in anchors:
            others = [j for j in cubic if j != anchor]
            for a, b in combinations(others, 2):
                syndrome = cols[anchor] ^ cols[a] ^ cols[b]
                word = (1 << anchor) | (1 << a) | (1 << b)
                for edge, edge_syndrome in enumerate(cols[:3*q]):
                    previous = triples.get(syndrome ^ edge_syndrome)
                    if previous is not None:
                        witness = word ^ previous ^ (1 << edge)
                        if witness and witness.bit_count() <= 7:
                            if witness.bit_count() <= 6:
                                return None
                            return (7, witness)
        assert all((row & best).bit_count() % 2 == 0 for row in rows)
        return (8, best)

    spectra = ((2, 4, 4), (4, 4, 4), (4, 4, 8),
               (2, 8, 8), None)
    triangles = [[] for _ in spectra]
    for r, s in combinations(nonzero, 2):
        spectrum = tuple(sorted(
            (orders[r], orders[s], orders[G.mul(G.inv(r), s)])))
        for policy, requested in enumerate(spectra):
            if requested is None or spectrum == requested:
                triangles[policy].append((r, s))
    for policy in range(4):
        if not triangles[policy]:
            triangles[policy] = list(triangles[4])

    bank, seen_fillings = [], set()
    for card in range(20):
        profile, remainder = divmod(card, 10)
        core_kind, policy = divmod(remainder, 5)
        local = random.Random(BASE_SEED ^ (0x167B91 + 104729*card))
        records = []
        for trial in range(64):
            data = local.choice(cores[core_kind])
            g, h = data[:2]
            C = [[[e], [e], [e], [], []],
                 [[e], [g], [h], [], []]]
            if profile == 0:
                r, s = local.choice(triangles[policy])
                u, v = local.choice(triangles[(policy+1) % 5])
                C[0][3], C[1][4] = [e, r, s], [e, u, v]
            else:
                ratio_policies = ((2, 2), (4, 4), (8, 8), (4, 8), (0, 0))
                p0, p1 = ratio_policies[policy]
                r = local.choice(pools[p0] if p0 else nonzero)
                s = local.choice(pools[p1] if p1 else nonzero)
                a, b = G.pick(local), G.pick(local)
                C[0][3], C[1][3] = [a, G.mul(a, r)], [e]
                C[0][4], C[1][4] = [e], [b, G.mul(b, s)]
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in C for cell in row)
            if filling in seen_fillings:
                continue
            seen_fillings.add(filling)
            assert all(sum(map(len, row)) == 6 for row in C)
            assert all(len(set(cell)) == len(cell)
                       for row in C for cell in row)
            rows, _ = _q16_component_rows(G, C)
            if gf2_rank(rows) != 2*q:
                continue
            screened = join_screen(rows, data)
            if screened is None:
                continue
            floor, word = screened
            penalty = sum(
                (row & other).bit_count() *
                ((row & other).bit_count()-1) // 2
                for i, row in enumerate(rows) for other in rows[:i])
            records.append((floor, penalty, trial, C, word))
        records.sort(key=lambda r: (-r[0], r[1], r[2]))
        for floor, penalty, trial, C, word in records[:3]:
            right_rows, _ = _q16_component_rows(G, C, True)
            right_word, source = 0, word
            while source:
                bit = source & -source
                j, t = divmod(bit.bit_length()-1, q)
                right_word |= 1 << (j*q + G.idx(G.inv(G.elements[t])))
                source ^= bit
            assert gf2_rank(right_rows) == 2*q
            assert all((row & right_word).bit_count() % 2 == 0
                       for row in right_rows)
            bank.append(dict(grid=C, floor=floor, penalty=penalty,
                             left=word, right=right_word,
                             card=card, trial=trial))
    if len(bank) < 5:
        raise ValueError("too few Q16 join-screened constituents")

    plans = sorted(
        (-min(a["floor"], b["floor"]),
         a["penalty"] + b["penalty"],
         int(a["card"] == b["card"]), ai, bi)
        for ai, a in enumerate(bank)
        for bi, b in enumerate(bank[ai:], ai))

    def embedded(record, kind, checks, stabilizers):
        own = _q16_basis(stabilizers)
        best = 0
        for block in range(5):
            if kind == "X":
                word = record["right"] << (block*width)
            else:
                word, source = 0, record["left"]
                while source:
                    bit = source & -source
                    j, t = divmod(bit.bit_length()-1, q)
                    word |= 1 << ((j*5 + block)*q + t)
                    source ^= bit
            if any((row & word).bit_count() % 2 for row in checks):
                raise ValueError("embedded verified constituent word failed")
            if not _q16_outside(word, own):
                continue
            for _ in range(2):
                starting = word
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < word.bit_count():
                        word = trial
                if word == starting:
                    break
            if not best or word.bit_count() < best.bit_count():
                best = word
        return best

    results, graph_keys, tested = [], set(), 0
    for _, _, _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 464 or not (k > 0 and 275*n <= 1000*k < 325*n):
            continue
        signature = _q16_graph_key(hx, hz, n)
        if signature in graph_keys:
            continue
        graph_keys.add(signature)
        tested += 1
        if not (_q16_clear_four(_q16_columns(hx, n))
                and _q16_clear_four(_q16_columns(hz, n))):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified Q16 reinforcement lift failed CSS")
        assert max(row.bit_count() for row in hx + hz) <= cap
        xseed = embedded(b, "X", hz, hx)
        zseed = embedded(a, "Z", hx, hz)
        if not xseed or not zseed:
            continue
        local = random.Random(BASE_SEED ^ (0x16C300 + 257*ai + bi))
        xword = _q16_probe(hz, hx, n, local, xseed, 3)
        zword = _q16_probe(hx, hz, n, local, zseed, 3)
        reading = min(xword.bit_count(), zword.bit_count())
        if reading < 6:
            continue

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        quality = (k*reading*reading/n,
                   min(a["floor"], b["floor"]),
                   -a["penalty"]-b["penalty"], signature)
        results.append((quality, freeze(A), freeze(B), k, reading,
                        (a["card"], b["card"]),
                        (a["floor"], b["floor"])))
        if len(results) >= N_SLOTS:
            break
        if tested >= max(180, 5*N_SLOTS):
            break
    results.sort(key=lambda entry: entry[0], reverse=True)
    return tuple(results)


def NEW_CTX(rng, G, idx):
    """Pure indexed publication; never pad with orbit copies."""
    portfolio = _q16_join_portfolio(G)
    if not (0 <= idx < len(portfolio)):
        raise ValueError("distinct Q16 reinforced portfolio exhausted")
    quality, A, B, k, reading, cards, floors = portfolio[idx]
    return {"A": A, "B": B, "screen_k": k,
            "logical_witness_upper_bound": reading,
            "constituent_kernel_floors": floors,
            "cards": cards, "selection_quality": quality[:3]}


def CELL(rng, G, side, i, j, ctx):
    """Publish a fresh complete cell from the immutable screened design."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The regular triangle-free catalogue supplies twenty distinct codes: sixty evaluated slots retained only four classes and four measured codes, with readings 4,4,3,3.
# REFUTED: A bare singleton replacing one matching supplies broad diversity: the shown asymmetric catalogue retained only one distinct measured code across sixty slots.
# REFUTED: The singleton-opposite-cycle catalogue supplies twenty inequivalent designs: the current sixty-slot evaluation retained only five distinct codes, all with distance reading four, including two k=280 leaders.
# REFUTED: Exhaustive relative-voltage pairing and logical-four-aware selection supply five novel measured leaders: the measured sixty-slot evaluation retained only three distinct codes, all with k=287 and d30s=4, scoring 4.7262.
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
