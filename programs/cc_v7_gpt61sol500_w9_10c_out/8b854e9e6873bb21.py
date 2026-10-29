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
CONCEPT = ("On C6 semidirect_5 C2, order 12, A=(2,6), B=(2,7), widened A constituents and spanning B voltages should supply novel distance-four-or-better codes; verified target-band rank prioritization tests whether five-code fitness can exceed 7.4780.")
GROUP = ("Keep the cyclic-by-cyclic semidirect-product construction rule. Factor 46 excludes the parent's order 24; m=6,r=2,s=5 gives order 12 and satisfies 5^2=1 mod 6. This differs from the shown order-12 C3:C4 bed without changing the family rule. Reflections here are involutions and are excluded from isolated binomial columns. Parent readings k=258,d30s=4 motivate rank screening, not a transferred distance guarantee.")
SHAPES = ("A=(2,6), B=(2,7) give n=552, kb=240 and guaranteed rate 20/46=10/23. The size window permits q<=13; the prescribed lift has q=12. Constituents have 72 and 84 variables, each with 24 checks. Parent minima broadly occupied sector-1 Acol0..2/Bcol0..6, especially Acol1 and Bcol5. Six A columns and four sector-2 block pairs replace that layout. Verified ranks enforce [0.425,0.475) without assuming k=258.")
FILLING = ("Twenty prescriptions combine four B support structures with five A/connectivity policies. A rows total six and columns two; B rows total seven and columns at most three, so HX<=9 and HZ=9, below both nine and twelve. Isolated binomials use noninvolutory rotation gaps; cubic columns may use reflection ratios. Verified constituent and complete kernels exclude weights 1..3. Deterministic batches prioritize actual k, then constituent four-word collisions. Rooted intrinsic Tanner invariants conservatively reject repeats and the five shown parent anchors. Neither twenty surviving classes, rank improvement, nor unseen archive novelty is assumed.")
FAMILY_NAME = "r045_m6r2s5_26x27_spanning_rooted_rank_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    # Same semidirect-product rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 46 requires q<=13; 5^2=1 modulo 6.
    m, r, s = 6, 2, 5
    if not valid_q(q):
        raise ValueError("this semidirect product has order 12")
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        return ((g[0] + powers[g[1] % r] * h[0]) % m,
                (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-powers[bi] * g[0]) % m, bi)

    def idx(g):
        a, b = red(g)
        return m * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 6)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


def _kernel_profile(rows, n):
    """Exact low-weight kernel screen, not a distance-search estimate.

    A reading of five means only that weights one through four were
    excluded. Equal syndromes of words of weight <=2 detect all such
    kernel words. Collision counts are only a secondary selection feature.
    """
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= check
            row ^= low
    seen = {0: 0}
    lightest, collisions = 5, 0

    def inspect(syndrome, word):
        nonlocal lightest, collisions
        if syndrome in seen:
            difference = word ^ seen[syndrome]
            if difference:
                lightest = min(lightest, difference.bit_count())
                collisions += 1
        else:
            seen[syndrome] = word

    for j, syndrome in enumerate(columns):
        inspect(syndrome, 1 << j)
    for j, syndrome in enumerate(columns):
        for i in range(j):
            inspect(syndrome ^ columns[i], (1 << j) | (1 << i))
    return lightest, collisions


def _extract_components(G, hx, hz):
    """Both constituent matrices, extracted from the verified full lift."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            row = hx[(i * cb) * q + r]
            compact = 0
            for j in range(ca):
                compact |= ((row >> (j * cb * q)) & mask) << (j * q)
            ar.append(compact)
    br = [row & ((1 << (cb * q)) - 1)
          for row in hz[:rb * q]]
    return ar, br


def _odd_inverse_classes(G):
    """Six undirected odd-parity generator classes, none involutory."""
    classes = {}
    for g in G.elements:
        if g[0] % 2 == 1:
            key = min(G.idx(g), G.idx(G.inv(g)))
            classes.setdefault(key, g)
    pools = [
        [g for _, g in sorted(classes.items()) if g[1] == b]
        for b in range(2)
    ]
    assert len(pools[0]) == len(pools[1]) == 3
    assert all(G.mul(g, g) != G.e for pool in pools for g in pool)
    return pools


def _draw_incidence(rng, G, attempt, pools):
    """Twenty structural cards, optimized rather than emitted as draws.

    A's graph is bipartite under a mod 2. B varies its internal-edge
    allocation. With internal edges present, cross-matchings have one
    common parity, so the B incidence graph is also bipartite.
    """
    card = attempt % 20
    epoch = attempt // 20
    outside_a = card % 4
    generators = (rng.sample(pools[0], 3 - outside_a)
                  + rng.sample(pools[1], outside_a))
    rng.shuffle(generators)
    A = [[[G.e, g] for g in generators]]

    internal = card // 5
    B = [[], []]
    for row in range(2):
        # Independent inverse classes exclude parallel internal edges.
        chosen = rng.sample(pools[0] + pools[1], internal)
        for g in chosen:
            B[row].append([G.e, g])
            B[1 - row].append([])

    cross_count = 7 - 2 * internal
    if internal:
        parity = epoch % 2
        ratio_pools = [
            [g for g in G.elements
             if g[1] == b and g[0] % 2 == parity]
            for b in range(2)
        ]
        outside_b = (card + epoch) % (cross_count + 1)
    else:
        ratio_pools = [
            [g for g in G.elements if g[1] == b]
            for b in range(2)
        ]
        outside_b = 1 + (card + epoch) % 6

    ratios = (rng.sample(ratio_pools[0], cross_count - outside_b)
              + rng.sample(ratio_pools[1], outside_b))
    rng.shuffle(ratios)
    for g in ratios:
        B[0].append([G.e])
        B[1].append([g])

    assert len(B[0]) == len(B[1]) == 7
    assert all(sum(map(len, row)) == 7 for row in B)
    assert all(len(B[0][j]) + len(B[1][j]) == 2 for j in range(7))
    return A, B


def _no_kernel_through_three(rows, n):
    """Exactly exclude all nonzero kernel words of weights 1, 2 and 3."""
    columns = [0] * n
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= tag
            row ^= bit
    singles = set(columns)
    if 0 in singles or len(singles) != n:
        return False
    for i, first in enumerate(columns):
        for j in range(i):
            if (first ^ columns[j]) in singles:
                return False
    return True


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
    return tuple(sorted(
        (rows[i].bit_count(), tuple(sorted(profiles[i].items())))
        for i in range(m)))


def _unused_common_c12_ctx(rng, G, idx):
    """Uncalled historical order-24 catalogue; never publishes new cells.

    The first gate is the proposed rank gain, not an augmentation
    calculation. All component and quantum screens use verified lifts.
    """
    from itertools import combinations

    local = random.Random(BASE_SEED ^ 0xC1225845)
    rotations = {g[0]: g for g in G.elements if g[1] == 0}

    def x(a):
        return rotations[a % 12]

    # These distinct inverse classes give two K6,6 incidence components.
    # Do not vary them by automorphisms and call those new designs.
    A = [[[G.e, x(d)] for d in (1, 3, 5)]]
    sa = [[2, 2, 2]]

    def canon(top, bottom, ratios):
        """Remove known B gauges and common rotation automorphisms.

        Odd A inverse classes are preserved as a set by every unit
        modulo twelve. Internal edges can be reversed independently.
        B-row exchange negates cross voltages and exchanges edge sets.
        Equal keys are rejected; this is not a complete graph-iso test.
        """
        best = None
        for u in (1, 5, 7, 11):
            ta = tuple(sorted(min((u * d) % 12,
                                  (-u * d) % 12) for d in top))
            tb = tuple(sorted(min((u * d) % 12,
                                  (-u * d) % 12) for d in bottom))
            values = [(u * a) % 12 for a in ratios]
            for shift in range(12):
                forward = tuple(sorted((a + shift) % 12
                                       for a in values))
                reverse = tuple(sorted((-a + shift) % 12
                                       for a in values))
                for key in ((ta, tb, forward), (tb, ta, reverse)):
                    if best is None or key < best:
                        best = key
        return best

    # Deliberate orbit representatives, not seven fresh labels per slot.
    dense = sorted({
        canon((), (), values)
        for values in combinations(range(12), 7)
    })
    local.shuffle(dense)

    def draw_card(attempt):
        internal = attempt % 4
        epoch = attempt // 4
        policy = epoch % 6
        if internal == 0:
            return dense[epoch % len(dense)]

        cross_count = 7 - 2 * internal
        if internal == 3:
            # Six internal columns plus one bridge matching.
            top = bottom = (1, 3, 5)
            ratios = (0,)
        elif policy < 3:
            # A sufficient bipartite construction; all these steps
            # have order four or twelve, but labels remain rotations.
            top = tuple(local.sample((1, 3, 5), internal))
            bottom = tuple(local.sample((1, 3, 5), internal))
            parity = policy % 2
            ratios = tuple(local.sample(
                list(range(parity, 12, 2)), cross_count))
        elif policy == 3 and internal == 1:
            # Order-six cycles: this six-voltage set has no difference
            # +/-2, preventing triangles involving an internal edge.
            top = bottom = (2,)
            ratios = tuple(local.sample((0, 1, 4, 5, 8, 9), 5))
        elif policy == 3:
            # Triangle-free but not forced to have the parent's odd
            # grading. The verified screen decides which bridges work.
            top = bottom = (2, 3)
            ratios = tuple(local.sample(range(12), 3))
        else:
            # Broader order-four/six/twelve edge prescriptions.
            # Reject triangles below rather than trusting this proposal.
            top = tuple(local.sample((1, 2, 3, 5), internal))
            bottom = tuple(local.sample((1, 2, 3, 5), internal))
            ratios = tuple(local.sample(range(12), cross_count))
        return canon(top, bottom, ratios)

    def build_B(card):
        top, bottom, ratios = card
        B = [[], []]
        for d in top:
            B[0].append([G.e, x(d)])
            B[1].append([])
        for d in bottom:
            B[0].append([])
            B[1].append([G.e, x(d)])
        for a in ratios:
            B[0].append([G.e])
            B[1].append([x(a)])
        return B

    graph_keys, card_keys = set(), set()
    accepted = 0
    for attempt in range(8192):
        card = draw_card(attempt)
        if card in card_keys:
            continue
        card_keys.add(card)
        B = build_B(card)
        sb = [[len(cell) for cell in row] for row in B]
        assert len(B[0]) == len(B[1]) == 7
        assert all(sum(row) == 7 for row in sb)
        assert all(sb[0][j] + sb[1][j] == 2 for j in range(7))
        assert _no_dead_lines(sa, sb)
        if _grid_weights(sa, sb) > min(9, W_MAX):
            raise ValueError("common-C12 profiles require row cap nine")

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # A failed rank-gain hypothesis must not silently emit k=254.
        if n != 552 or k != 258:
            continue
        assert 425 * n <= 1000 * k < 475 * n

        ar, br = _extract_components(G, hx, hz)
        qa = _kernel_profile(ar, 3 * G.q)
        qb = _kernel_profile(br, 7 * G.q)
        if min(qa[0], qb[0]) < 4:
            continue

        signature = _tanner_signature(hx, hz)
        if signature in graph_keys:
            continue
        # Check mixed-sector words too, not just constituent triangles.
        if not _no_kernel_through_three(hx, n):
            continue
        if not _no_kernel_through_three(hz, n):
            continue
        graph_keys.add(signature)

        if accepted == idx:
            assert css_ok(hx, hz), "verified lift failed CSS"
            assert lifted_k(G, A, B) == k
            assert max(row.bit_count() for row in hx + hz) == 9
            assert all(g[1] == 0
                       for grid in (A, B) for row in grid
                       for cell in row for g in cell)
            return {
                "A": A, "B": B, "screen_k": k,
                "component_screens": (qa, qb),
                "centralizer_weights_through_three_excluded": True,
                "design_card": card,
                "rank_gain_above_bound": k - 240,
            }
        accepted += 1

    raise ValueError("distinct k258 common-C12 catalogue exhausted")


def _shape_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _shape_component_quality(rows, width):
    """Exclude kernels <=3; count pair collisions as a weight-four burden."""
    columns = _shape_columns(rows, width)
    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return None
    pairs, burden = set(), 0
    for i, first in enumerate(columns):
        for second in columns[:i]:
            syndrome = first ^ second
            if syndrome in singles:
                return None
            if syndrome in pairs:
                burden += 1
            else:
                pairs.add(syndrome)
    return burden


def _shape_draw(local, G, attempt):
    """Twenty constructive cards; proposals are optimized before publication."""
    card, epoch = attempt % 20, attempt // 20
    amode, bmode = card % 5, card // 5
    e = G.e
    rotations = [g for g in G.elements if g[1] == 0]
    reflections = [g for g in G.elements if g[1] == 1]
    odd_rotations = [g for g in rotations if g[0] % 2]

    def matching(values):
        return [[[e] for _ in values], [[g] for g in values]]

    if amode == 0:
        # Six C6 matchings give two K6,6 incidence components.
        A = matching(rotations)
    elif amode in (1, 2, 3):
        outside = (1 + epoch % 3 if amode == 1 else
                   3 + epoch % 3 if amode == 2 else
                   1 + epoch % 5)
        values = (local.sample(rotations, 6 - outside)
                  + local.sample(reflections, outside))
        A = matching(values)
    else:
        # Internal order-six rings; equal cross parity excludes triangles.
        parity = epoch % 2
        values = local.sample(
            [g for g in G.elements if g[0] % 2 == parity], 4)
        gap0, gap1 = local.choice(odd_rotations), local.choice(odd_rotations)
        # The rotation x^3 is involutory and must not be an isolated edge.
        if G.mul(gap0, gap0) == e or G.mul(gap1, gap1) == e:
            return None
        A = [
            [[e, gap0], []] + [[e] for _ in values],
            [[], [e, gap1]] + [[g] for g in values],
        ]

    if bmode == 0:
        outside = 1 + (epoch + amode) % 6
        values = (local.sample(rotations, 7 - outside)
                  + local.sample(reflections, outside))
        B = matching(values)
    elif bmode == 1:
        # One internal ring per row and five distinct cross matchings.
        parity = (epoch + amode) % 2
        values = local.sample(
            [g for g in G.elements if g[0] % 2 == parity], 5)
        gaps = [local.choice([(1, 0), (5, 0)]) for _ in range(2)]
        B = [
            [[e, gaps[0]], []] + [[e] for _ in values],
            [[], [e, gaps[1]]] + [[g] for g in values],
        ]
    else:
        # Ground/cubic redistributions: every row still totals seven.
        profiles = (
            ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
            ((1, 0), (1, 2)) + ((1, 1),) * 5,
        )
        profile = profiles[bmode - 2]
        B = [[[] for _ in range(7)] for _ in range(2)]
        cross = [j for j, counts in enumerate(profile) if counts == (1, 1)]
        values = local.sample(list(G.elements), len(cross))
        for j, g in zip(cross, values):
            B[0][j], B[1][j] = [e], [g]
        for j, counts in enumerate(profile):
            if j in cross:
                continue
            degree = sum(counts)
            for i, count in enumerate(counts):
                if count == 1:
                    B[i][j] = [G.pick(local)]
                elif count == 2:
                    first = G.pick(local)
                    # An involution is permitted only in a cubic column:
                    # the other row's term prevents a parallel edge pair.
                    pool = ([(1, 0), (5, 0)] if degree == 2 else
                            [g for g in G.elements if g != e])
                    gap = local.choice(pool)
                    B[i][j] = [first, G.mul(first, gap)]

    assert all(len(row) == 6 and sum(map(len, row)) == 6 for row in A)
    assert all(len(A[0][j]) + len(A[1][j]) == 2 for j in range(6))
    assert all(len(row) == 7 and sum(map(len, row)) == 7 for row in B)
    assert all(1 <= len(B[0][j]) + len(B[1][j]) <= 3 for j in range(7))
    return A, B, (amode, bmode, epoch % 6)


def _shape_graph_key(hx, hz, n):
    """Intrinsic uncoloured Tanner invariant; equality is inconclusive.

    Root all intrinsically high-degree qubits. Check hyperedges and
    low-qubit corners retain endpoint multiplicities. No group-action
    transitivity or protograph labels are used to shorten the root list.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = _shape_columns(rows, n)
    high = [j for j, col in enumerate(columns) if col.bit_count() >= 10]
    positions = {j: i for i, j in enumerate(high)}
    edges, tags, check_ends = [], [], []

    for row in rows:
        ends = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in positions:
                ends.append(positions[j])
            bits ^= bit
        ends = tuple(sorted(ends))
        check_ends.append(ends)
        edges.append(ends)
        tags.append((0, row.bit_count()))

    for j, col in enumerate(columns):
        if j in positions:
            continue
        ends, weights = [], []
        while col:
            bit = col & -col
            r = bit.bit_length() - 1
            ends.extend(check_ends[r])
            weights.append(rows[r].bit_count())
            col ^= bit
        edges.append(tuple(sorted(ends)))
        tags.append((1, tuple(sorted(weights))))

    incident = [[] for _ in high]
    for edge, ends in enumerate(edges):
        for v in ends:
            incident[v].append(edge)

    def compress(features):
        palette = {f: i for i, f in enumerate(sorted(set(features)))}
        return [palette[f] for f in features]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(4):
            ef = [(tag, tuple(sorted(colours[v] for v in ends)))
                  for tag, ends in zip(tags, edges)]
            ec = compress(ef)
            vf = [(colours[v], tuple(sorted(ec[k] for k in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(ef).items())),
                            tuple(sorted(Counter(vf).items()))))
            colours = compress(vf)
        rooted.append(sha256(repr(history).encode("ascii")).digest())

    return (
        n,
        tuple(sorted(Counter(r.bit_count() for r in rows).items())),
        tuple(sorted(Counter(c.bit_count() for c in columns).items())),
        tuple(sorted(rooted)),
        _tanner_signature(hx, hz),
    )


def _shape_parent_exclusions():
    """The five measured parent fillings are references, never candidates."""
    els = [(a, b) for b in range(2) for a in range(12)]

    def mul(g, h):
        return ((g[0] + pow(5, g[1], 12) * h[0]) % 12,
                (g[1] + h[1]) % 2)

    def inv(g):
        b = (-g[1]) % 2
        return ((-pow(5, b, 12) * g[0]) % 12, b)

    old = _G(dict(elements=els, ident=(0, 0), mul=mul, inv=inv), 24)
    e = old.e
    A = [[[e, (d, 0)] for d in (1, 3, 5)]]
    cards = (
        ((), (), (0, 1, 2, 3, 6, 7, 9)),
        ((1,), (5,), (0, 2, 4, 6, 8)),
        ((1, 3), (3, 5), (0, 2, 6)),
        ((1, 3, 5), (1, 3, 5), (0,)),
        ((1, 3), (2, 3), (0, 4, 8)),
    )
    excluded = set()
    for top, bottom, values in cards:
        B = [[], []]
        for d in top:
            B[0].append([e, (d, 0)])
            B[1].append([])
        for d in bottom:
            B[0].append([])
            B[1].append([e, (d, 0)])
        for d in values:
            B[0].append([e])
            B[1].append([(d, 0)])
        hx, hz, n = lifted_matrices(old, A, B)
        excluded.add(_shape_graph_key(hx, hz, n))
    return excluded


from functools import lru_cache


@lru_cache(maxsize=1)
def _shape_portfolio(G, count):
    """Deterministic immutable portfolio; memoization changes only build cost."""
    local = random.Random(BASE_SEED ^ 0x12627451)
    cap = min(9, W_MAX)
    if G.q != 12 or cap < 9:
        raise ValueError("new-shape portfolio requires q=12 and row cap nine")
    graph_keys = _shape_parent_exclusions()
    filling_keys, entries = set(), []

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    for batch in range(96):
        candidates = []
        # Two proposals for each of the twenty structural cards.
        for trial in range(40):
            attempt = 40 * batch + trial
            proposal = _shape_draw(local, G, attempt)
            if proposal is None:
                continue
            A, B, card = proposal
            filling = tuple(
                tuple(sorted(G.idx(g) for g in cell))
                for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            sa = [[len(cell) for cell in row] for row in A]
            sb = [[len(cell) for cell in row] for row in B]

            wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                     for row in sa for j in range(7))
            wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                     for row in sb for j in range(6))
            if not _no_dead_lines(sa, sb) or max(wx, wz) > cap:
                raise ValueError("new-shape support budget failed")
            if any(len(set(cell)) != len(cell)
                   for grid in (A, B) for row in grid for cell in row):
                raise ValueError("new-shape cell contains duplicate elements")

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue

            ar, br = _extract_components(G, hx, hz)
            qa = _shape_component_quality(ar, 6 * G.q)
            qb = _shape_component_quality(br, 7 * G.q)
            if qa is None or qb is None:
                continue
            priority = (-k, max(qa, qb), qa + qb, attempt)
            candidates.append(
                (priority, A, B, hx, hz, n, k, rx, rz, qa, qb, card))

        candidates.sort(key=lambda record: record[0])
        for _, A, B, hx, hz, n, k, rx, rz, qa, qb, card in candidates:
            # Complete screens include both LP sectors.
            if not (_no_kernel_through_three(hx, n)
                    and _no_kernel_through_three(hz, n)):
                continue
            signature = _shape_graph_key(hx, hz, n)
            if signature in graph_keys:
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified new-shape lift failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")
            graph_keys.add(signature)
            diagnostics = (
                ("k", k), ("card", card),
                ("component_four_collision_burdens", (qa, qb)),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("full_kernel_weights_1_to_3_excluded", True),
            )
            entries.append((freeze(A), freeze(B), diagnostics))
            if len(entries) == count:
                return tuple(entries)

    # Never recycle an earlier code to make an exhausted portfolio look full.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of a complete rank-prioritized, invariant-separated design."""
    portfolio = _shape_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("distinct screened new-shape portfolio exhausted")
    A, B, diagnostics = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(diagnostics)}


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected cell without modifying the immutable portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supplied twenty inequivalent initial designs on C12:C2(s=5); slots 0..19 collapsed to four distinct codes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: The common-C12 catalogue supplies twenty inequivalent k=258 designs: sixty slots retained only five downstream classes, exhausted after slot 51, and all five measured codes still had d30s=4.
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
