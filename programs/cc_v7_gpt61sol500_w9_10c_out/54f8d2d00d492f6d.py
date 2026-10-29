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
CONCEPT = ("Direct product C5 x C2, order 10, A=B=(2,7): exhaustive triangle-free incidence constituents with independently chosen internal directions should provide at least five novel distance-four finalists in cell 0.500.")
GROUP = ("Jump from C11 to the direct-product construction C5 x C2, abstractly C10 and genuinely nonisomorphic to C11. Order ten gives n=530. The reported C10 bed best is 4.7, so this proposal tests exact short-kernel rejection and structural diversification, not an assumed group-distance advantage. Unknown archive coverage remains a risk.")
SHAPES = ("Keep equal (2,7) shapes, now with 70 variables and 20 checks per constituent. The size factor is 53, n=530, kb=250 and guaranteed rate 25/53=0.47170. Verified full ranks require k=262, rate 0.49434. Five distance-four readings would score 262*16/530=7.90943, above the cell's 7.7998 record. Parent minima broadly occupied sector-1 Acol/Bcol blocks.")
FILLING = ("Enumerate seven-matchings and independently stepped one/two internal columns per layer; no grounded fallback is submitted. Every row has seven terms and every column two, giving BOTH CSS row-weight sums nine, below twelve as well. Both verified constituent actions must have rank 19 and exclude kernels through weight three. Exact constituent weight-four counts prioritize pairings. Complete lifts enforce k=262, CSS, both budgets and the same kernel-three exclusion. Distinct complete uncoloured Tanner rooted-refinement invariants separate immutable catalogue entries; equality is discarded conservatively. Catalogue yield, unseen archive novelty and distance-four readings remain falsifiable.")
FAMILY_NAME = "r050_direct_c5xc2_27x27_independent_incidence_rooted10_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 10

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C5 x C2 requires order ten")
    # Direct-product construction, not the parent's metacyclic rule.
    els = [(a, b) for b in range(2) for a in range(5)]

    def red(g):
        return (int(g[0]) % 5, int(g[1]) % 2)

    def mul(g, h):
        return ((int(g[0]) + int(h[0])) % 5,
                (int(g[1]) + int(h[1])) % 2)

    def inv(g):
        return ((-int(g[0])) % 5, (-int(g[1])) % 2)

    def idx(g):
        a, b = red(g)
        return a + 5*b

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
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


def _retired_z11_context(rng, G, idx):
    """Uncalled historical C11 catalogue; never publishes new cells.

    Cards remove explicit constituent gauges. Relative group
    orientations are restored when pairing the two constituents.
    No mutable cross-slot cache or distance-search reading is used.
    """
    from itertools import combinations
    from collections import Counter
    import hashlib

    q, e, cap = G.q, G.e, min(9, W_MAX)
    elements = {G._key(g): g for g in G.elements}

    def x(v):
        return elements[(v % q, 0)]

    def translation(values):
        return min(tuple(sorted((v-anchor) % q for v in values))
                   for anchor in values)

    def image(card, scale):
        kind, upper, lower, values = card
        a = tuple(sorted(min(scale*d % q, -scale*d % q)
                         for d in upper))
        b = tuple(sorted(min(scale*d % q, -scale*d % q)
                         for d in lower))
        v = translation(tuple(scale*s % q for s in values))
        return kind, a, b, v

    def canonical(card):
        images = []
        for scale in range(1, q):
            kind, a, b, v = image(card, scale)
            images.append((kind, a, b, v))
            if kind == 0:
                images.append((kind, b, a,
                               translation(tuple(-s % q for s in v))))
        return min(images)

    def regular(upper, lower, values):
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
            grid = [
                [[e] for _ in values] + [[e]],
                [[x(v)] for v in values] + [[]],
            ]
        else:
            grid = [
                [[e] for _ in values] + [[], [e]],
                [[x(v)] for v in values]
                + [[e, x(lower[0])], []],
            ]
        assert all(len(row) == 7 for row in grid)
        assert all(len({G._key(g) for g in c}) == len(c)
                   for row in grid for c in row)
        return grid

    def columns(rows, width):
        result = [0] * width
        mask = (1 << width) - 1
        for r, row in enumerate(rows):
            row &= mask
            while row:
                bit = row & -row
                result[bit.bit_length()-1] |= 1 << r
                row ^= bit
        return result

    def component_info(grid):
        # Obtain the exact scaffold action using a zero dummy side.
        rows, _, _ = lifted_matrices(G, grid, [[[]]])
        width = 7*q
        mask = (1 << width) - 1
        rows = [row & mask for row in rows]
        if not _no_short_kernel(rows, width):
            return None
        cols = columns(rows, width)
        counts, collisions = {}, 0
        for i, first in enumerate(cols):
            for second in cols[:i]:
                syndrome = first ^ second
                old = counts.get(syndrome, 0)
                collisions += old
                counts[syndrome] = old + 1
        # With weights <=3 excluded, every pair collision has four
        # distinct positions. Each four-word has three partitions.
        assert collisions % 3 == 0
        return gf2_rank(rows), collisions // 3

    def signed(steps):
        return set(steps) | {(-d) % q for d in steps}

    descriptors = set()
    for p in range(3):
        options = []
        for steps in combinations(range(1, (q+1)//2), p):
            ss = signed(steps)
            if not any((a+b) % q in ss for a in ss for b in ss):
                options.append(steps)
        for ai, upper in enumerate(options):
            for lower in options[ai:]:
                forbidden = signed(upper) | signed(lower)
                for tail in combinations(range(1, q), 6-2*p):
                    values = (0,) + tail
                    if any((a-b) % q in forbidden
                           for a, b in combinations(values, 2)):
                        continue
                    descriptors.add(canonical(
                        (0, upper, lower, values)))

    # A singleton anchors every translated vertex of its layer.
    # Internal edges on that same layer would create weight-three
    # words. Thus the two clean grounded prescriptions below have
    # no internal edge on the singleton layer.
    for tail in combinations(range(1, q), 5):
        descriptors.add(canonical((1, (), (), (0,) + tail)))
    for step in range(1, (q+1)//2):
        for tail in combinations(range(1, q), 4):
            values = (0,) + tail
            if any((a-b) % q in (step, (-step) % q)
                   for a, b in combinations(values, 2)):
                continue
            descriptors.add(canonical((2, (), (step,), values)))

    bank = []
    for card in sorted(descriptors):
        grid = build(card)
        info = component_info(grid)
        if info is not None:
            bank.append((card, grid, info))
    if not bank:
        raise ValueError("no exact kernel-three-clean constituents")

    def fingerprint(hx, hz, n):
        """Rooted refinement invariant of the COMPLETE Tanner graph.

        Checks are pooled without CSS colors. Qubits and checks retain
        their intrinsic bipartite roles. Every degree>=12 qubit is a
        possible root. In these cyclic lifts, simultaneous translation
        gives four root orbits of size q; inspect one representative
        per orbit and restore its multiplicity.

        Different fingerprints imply nonisomorphic Tanner graphs.
        Equality is inconclusive and is discarded conservatively.
        """
        rows = hx + hz
        nv = n + len(rows)
        adjacency = [[] for _ in range(nv)]
        for r, row in enumerate(rows):
            check = n+r
            while row:
                bit = row & -row
                j = bit.bit_length()-1
                adjacency[j].append(check)
                adjacency[check].append(j)
                row ^= bit
        degrees = [len(a) for a in adjacency]
        high = [j for j in range(n) if degrees[j] >= 12]
        n1 = 49*q
        if high != list(range(n1, n)):
            raise ValueError("unexpected intrinsic high-degree sector")
        roots = [n1 + orbit*q for orbit in range(4)]
        rooted = []
        for root in roots:
            labels = [
                (int(v >= n), degrees[v], int(v == root))
                for v in range(nv)]
            palette = {key: c for c, key in enumerate(sorted(set(labels)))}
            colors = [palette[key] for key in labels]
            digest = hashlib.blake2b(digest_size=24)
            digest.update(repr(tuple(sorted(Counter(labels).items())))
                          .encode("ascii"))
            previous = len(palette)
            for depth in range(8):
                keys = [
                    (colors[v],
                     tuple(sorted(colors[w] for w in adjacency[v])))
                    for v in range(nv)]
                histogram = tuple(sorted(Counter(keys).items()))
                digest.update(repr(histogram).encode("ascii"))
                unique = sorted(set(keys))
                palette = {key: c for c, key in enumerate(unique)}
                colors = [palette[key] for key in keys]
                if len(unique) == previous:
                    break
                previous = len(unique)
            rooted.extend([digest.digest()] * q)
        return (
            tuple(sorted(Counter(degrees[:n]).items())),
            tuple(sorted(Counter(degrees[n:]).items())),
            tuple(sorted(rooted)),
        )

    def matching(values):
        return regular((), (), values)

    def ring(a, b, values):
        return regular((a,), (b,), values)

    two = regular((1, 3), (1, 3), (0, 2, 4))
    other_two = regular((1, 3), (1, 3), (0, 2, 6))
    ring_one = ring(1, 1, (0, 2, 4, 6, 8))
    ground_ring = build((2, (), (1,), (0, 2, 4, 6, 8)))
    ground_six = build((1, (), (), tuple(range(6))))

    # References are only exclusion inputs, never published candidates.
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
        (two, other_two),
        (two, regular((1, 3), (1, 3), (0, 2, 7))),
        (two, regular((1, 4), (1, 4), (0, 2, 8))),
        (matching((0, 1, 2, 3, 4, 6, 7)),
         matching((0, 1, 2, 4, 5, 6, 8))),
        (matching(tuple(range(7))), ground_ring),
        (two, ground_ring),
        (matching(tuple(range(7))), ground_six),
        (ring_one, ground_six),
        (two, ground_six),
    ]
    archive_keys = set()
    for A, B in shown_archive:
        hx, hz, n = lifted_matrices(G, A, B)
        archive_keys.add(fingerprint(hx, hz, n))

    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            # Two grounded sides have exact k=275 in this family and
            # cannot enter the requested rate cell.
            if a[0][0] and b[0][0]:
                continue
            boundary = int(bool(a[0][0] or b[0][0]))
            qa, qb = a[2][1], b[2][1]
            for scale in range(1, q):
                prescription = (a[0], b[0], scale)
                tie = hashlib.blake2b(
                    repr(("relative-rooted8-v4", prescription))
                    .encode("ascii"), digest_size=8).digest()
                plans.append((
                    boundary, max(qa, qb), qa+qb, tie, ai, bi, scale))
    plans.sort()

    filling_keys, graph_keys = set(), set()
    accepted = 0
    for _, _, _, _, ai, bi, scale in plans:
        a, b = bank[ai], bank[bi]
        A = a[1]
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
        if n != 583 or max(row.bit_count() for row in hx+hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 475*n <= 1000*k < 525*n):
            continue

        signature = fingerprint(hx, hz, n)
        if signature in archive_keys or signature in graph_keys:
            continue
        # This gate applies to EVERY slot, including extra harness slots.
        if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified exhaustive incidence lift failed CSS")
        graph_keys.add(signature)

        if accepted == idx:
            return {
                "A": A, "B": B, "k": k,
                "cards": (a[0], bcard),
                "component_four_counts": (a[2][1], b[2][1]),
                "complete_kernels_through_three_excluded": True,
            }
        accepted += 1
    raise ValueError("novel rooted-invariant target-rate catalogue exhausted")


from functools import lru_cache


def _c10_columns(rows, width):
    """Column syndromes of a matrix supplied by the verified helper."""
    out = [0] * width
    mask = (1 << width) - 1
    for r, original in enumerate(rows):
        row = original & mask
        while row:
            bit = row & -row
            out[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return out


def _c10_four_count(rows, width):
    """Exact count after excluding every kernel word of weights 1..3."""
    cols = _c10_columns(rows, width)
    singles = set(cols)
    if 0 in singles or len(singles) != width:
        return None
    pairs, collisions = {}, 0
    for i, first in enumerate(cols):
        for j in range(i):
            syndrome = first ^ cols[j]
            if syndrome in singles:
                return None
            old = pairs.get(syndrome, 0)
            collisions += old
            pairs[syndrome] = old + 1
    # Distinct nonzero columns make colliding pairs disjoint.
    # Each weight-four support has three unordered pair partitions.
    if collisions % 3:
        raise ValueError("verified four-word count is inconsistent")
    return collisions // 3


def _c10_full_signature(G, hx, hz, n):
    """Intrinsic rooted invariant of the complete uncoloured Tanner graph.

    CSS types are pooled; only the bipartite qubit/check roles remain.
    In this ABELIAN lift, simultaneous translation of every group
    coordinate is a Tanner automorphism. Each sector-two block is
    therefore one orbit of q high-degree roots. Four representatives
    recover the complete multiset of high-degree rooted invariants.

    Unequal signatures certify nonisomorphism. Equal signatures are
    inconclusive and are conservatively discarded.
    """
    from collections import Counter
    import hashlib

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
    degrees = [len(neighbors) for neighbors in adjacency]
    n1 = 49 * G.q
    high = [j for j in range(n) if degrees[j] >= 12]
    if high != list(range(n1, n)):
        raise ValueError("unexpected intrinsic sector-two root class")

    rooted = []
    for block in range(4):
        root = n1 + block * G.q + G.idx(G.e)
        initial = [(int(v >= n), degrees[v], int(v == root))
                   for v in range(len(adjacency))]
        unique = sorted(set(initial))
        palette = {key: i for i, key in enumerate(unique)}
        colors = [palette[key] for key in initial]
        digest = hashlib.sha256()
        digest.update(repr(tuple(sorted(Counter(initial).items())))
                      .encode("ascii"))
        previous = len(unique)
        for depth in range(10):
            keys = [
                (colors[v],
                 tuple(sorted(colors[w] for w in neighbors)))
                for v, neighbors in enumerate(adjacency)
            ]
            digest.update(repr(tuple(sorted(Counter(keys).items())))
                          .encode("ascii"))
            unique = sorted(set(keys))
            palette = {key: i for i, key in enumerate(unique)}
            colors = [palette[key] for key in keys]
            if len(unique) == previous:
                break
            previous = len(unique)
        rooted.extend([digest.digest()] * G.q)

    # A supplementary invariant uses exact check-intersection sizes.
    overlaps = Counter(
        (rows[i] & rows[j]).bit_count()
        for i in range(len(rows)) for j in range(i))
    return (
        tuple(sorted(Counter(degrees[:n]).items())),
        tuple(sorted(Counter(degrees[n:]).items())),
        tuple(sorted(overlaps.items())),
        tuple(sorted(rooted)),
    )


@lru_cache(maxsize=1)
def _c10_portfolio(G):
    """Immutable exhaustive structural portfolio; no slot-order dependence."""
    from itertools import combinations
    import hashlib

    if G.q != 10:
        raise ValueError("direct-product portfolio requires order ten")
    q, width, cap = G.q, 7 * G.q, min(9, W_MAX)
    if cap < 9:
        raise ValueError("regular incidence portfolio needs weight nine")
    elements = {G._key(g): g for g in G.elements}

    def element(t):
        # t -> (t mod 5, t mod 2) is a bijection and homomorphism C10
        # -> C5 x C2. This only specifies labels; all lifts are verified.
        return elements[(t % 5, t % 2)]

    def build(upper, lower, labels):
        p = len(upper)
        if len(lower) != p or len(labels) + 2*p != 7:
            raise ValueError("incidence descriptor has the wrong shape")
        return [
            [[G.e, element(d)] for d in upper]
            + [[G.e] for _ in labels] + [[] for _ in lower],
            [[] for _ in upper]
            + [[element(t)] for t in labels]
            + [[G.e, element(d)] for d in lower],
        ]

    def translation_key(labels):
        return min(tuple(sorted((t - anchor) % q for t in labels))
                   for anchor in labels)

    def signed(steps):
        return set(steps) | {(-d) % q for d in steps}

    def internally_triangle_free(steps):
        directions = signed(steps)
        return not any((a + b) % q in directions
                       for a in directions for b in directions)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    # Exclude the involution t=5: its binomial repeats component edges.
    # Independent upper/lower directions are retained, not required equal.
    descriptors = set()
    for p in range(3):
        options = [
            steps for steps in combinations(range(1, 5), p)
            if internally_triangle_free(steps)
        ]
        for upper in options:
            for lower in options:
                forbidden = signed(upper) | signed(lower)
                for tail in combinations(range(1, q), 6 - 2*p):
                    labels = (0,) + tail
                    if any((a - b) % q in forbidden
                           for a, b in combinations(labels, 2)):
                        continue
                    labels = translation_key(labels)
                    # Row exchange inverts the cross voltages.
                    forward = (p, upper, lower, labels)
                    reverse = (
                        p, lower, upper,
                        translation_key(tuple((-t) % q for t in labels)))
                    descriptors.add(min(forward, reverse))

    bank, filling_keys = [], set()
    unit = [[[G.e]]]
    mask = (1 << width) - 1
    for descriptor in sorted(descriptors):
        p, upper, lower, labels = descriptor
        grid = build(upper, lower, labels)
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for row in grid for cell in row)
        if filling in filling_keys:
            continue
        filling_keys.add(filling)
        counts = [[len(cell) for cell in row] for row in grid]
        if any(sum(row) != 7 for row in counts):
            raise ValueError("constituent row budget changed")
        if any(sum(row[j] for row in counts) != 2 for j in range(7)):
            raise ValueError("constituent column budget changed")
        if any(len(set(cell)) != len(cell)
               for row in grid for cell in row):
            raise ValueError("duplicate direct-product cell element")

        # Obtain both evaluator actions separately, without a private lift.
        left, _, _ = lifted_matrices(G, grid, unit)
        _, right, _ = lifted_matrices(G, unit, grid)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        if gf2_rank(left) != 19 or gf2_rank(right) != 19:
            continue
        nl = _c10_four_count(left, width)
        if nl is None:
            continue
        nr = _c10_four_count(right, width)
        if nr is None:
            continue
        bank.append((max(nl, nr), nl + nr, descriptor, freeze(grid)))

    if len(bank) < 4:
        raise ValueError("too few verified triangle-free constituents")
    bank.sort(key=lambda entry: entry[:3])

    # Relative orientations are present explicitly in the exhaustive
    # bank. Prefer internal-column pairs over the old dense-matching
    # model, then minimize actual constituent four-word multiplicities.
    plans = []
    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            internal = min(a[2][0], b[2][0])
            tie = hashlib.blake2b(
                repr(("direct-c5xc2-v1", a[2], b[2])).encode("ascii"),
                digest_size=8).digest()
            plans.append((-internal, max(a[0], b[0]),
                          a[1] + b[1], tie, ai, bi))
    plans.sort()

    entries, graph_keys = [], set()
    for _, _, _, _, ai, bi in plans:
        A, B = bank[ai][3], bank[bi][3]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        if n != 530 or max(row.bit_count() for row in hx + hz) > cap:
            continue
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != 262 or not (475*n <= 1000*k < 525*n):
            continue

        signature = _c10_full_signature(G, hx, hz, n)
        if signature in graph_keys:
            continue
        # Applies to every published slot, including extra harness slots.
        if not (_no_short_kernel(hx, n) and _no_short_kernel(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified direct-product lift failed CSS")
        graph_keys.add(signature)
        entries.append((
            A, B, k, (bank[ai][2], bank[bi][2]),
            (bank[ai][0], bank[bi][0]),
        ))
        if len(entries) >= N_SLOTS:
            break

    # Never pad an exhausted catalogue with relabelings or orbit copies.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure indexed selection from the immutable verified portfolio."""
    entries = _c10_portfolio(G)
    if not (0 <= idx < len(entries)):
        raise ValueError("distinct direct-product target-rate catalogue exhausted")
    A, B, k, cards, counts = entries[idx]
    return {
        "A": A, "B": B, "k": k, "cards": cards,
        "component_four_counts": counts,
        "complete_kernels_through_three_excluded": True,
    }


def CELL(rng, G, side, i, j, ctx):
    """Publish the complete selected cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The matching/one-cycle catalogue supplies twenty inequivalent clean designs: sixty measured slots collapsed to four classes, with distance readings 4,4,3,3.
# REFUTED: Exhaustive C11 relative-voltage cards with rooted-eight refinement supply five novel measured codes: the current sixty-slot evaluation retained only two measured classes and one archive duplicate, scoring 3.1508.
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
