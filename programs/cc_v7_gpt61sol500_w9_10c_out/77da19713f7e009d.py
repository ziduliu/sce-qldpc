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
CONCEPT = ("Metacyclic C8:C1, order 8, A=(3,8), B=(3,8): saturated three-row constituents with dependent augmentation rows should yield five novel k>200 codes with distance readings >=6, improving the measured cell-0.350 incumbent; verified ranks and explicit witnesses make this falsifiable.")
GROUP = ("Keep GTYPE, A0 and the semidirect-product construction rule, changing only its parameters to m=8,r=1,s=1. Factor 73 excludes order 24 and permits q<=8. The C8 (3,8)x(3,8) bed measured five [584,200,6] leaders and fitness about 12.3, versus 10.1 on a shown nonabelian order-eight bed. This measured comparison motivates the cyclic specialization, not a distance guarantee.")
SHAPES = ("Change both sides to (3,8): equal components have 64 variables and 24 checks, n=584, kb=200 and guaranteed rate 25/73=0.34247. Order eight is the largest admissible lift. Parent weight-four minima broadly occupied sector-1 Acol0..2/Bcol0..6, with a few X minima also entering Arow0/Brow1. Three rows on each side replace that unequal incidence geometry and create nine sector-2 blocks. Complete verified ranks require k>200 and rate in [0.325,0.375).")
FILLING = ("Optimize twenty prescriptions crossing five saturated supports with four relative-label policies. Every component row totals six and every column totals at most three, so BOTH CSS row maxima are nine. Two supports deliberately equate the first two augmentation rows; others supply independent-rank partners. Independent constituent banks use exact kernel screening through six, prioritizing actual weight-six witnesses or exclusion through six and verified rank deficits. Full verified lifts exclude all kernel words through four; embedded nontrivial X/Z witnesses undergo stabilizer descent and information-set probes before selection. Intrinsic uncoloured Tanner invariants conservatively reject repeats. Five fresh k>200 distance-six readings would beat the incumbent, but longer-distance survival and unseen-archive novelty remain hypotheses.")
FAMILY_NAME = "r035_m8r1s1_38x38_saturated_augdef_kernel6_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 8

def custom_ops(q):
    # Same semidirect-product rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 73 requires q<=8; the trivial complement gives C8.
    m, r, s = 8, 1, 1
    if not valid_q(q):
        raise ValueError("this semidirect product has order eight")
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

A_SHAPE = (3, 8)               # (ra, ca)
B_SHAPE = (3, 8)               # (rb, cb)


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


def _unused_common_C12_reference(rng, G, idx):
    """Uncalled historical subgroup catalogue; never publishes candidates.

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


from functools import lru_cache


@lru_cache(maxsize=1)
def _redistributed_portfolio(G, count):
    """Deterministic immutable portfolio, independent of slot call order.

    Every matrix property is computed from the scaffold's verified lift.
    Equal invariants are rejected conservatively, not declared isomorphic.
    """
    from collections import Counter
    from hashlib import sha256
    from itertools import combinations

    local = random.Random(BASE_SEED ^ 0x24801345)
    cap = min(9, W_MAX)

    def closure(generators):
        reached, pending = {G.e}, [G.e]
        steps = tuple(generators) + tuple(G.inv(g) for g in generators)
        while pending:
            first = pending.pop()
            for second in steps:
                product = G.mul(first, second)
                if product not in reached:
                    reached.add(product)
                    pending.append(product)
        return tuple(sorted(reached))

    rotations = closure(((1, 0),))
    outside12 = closure(((2, 0), (1, 1)))
    order8 = closure(((3, 0), (0, 1)))
    assert len(rotations) == len(outside12) == 12
    assert len(order8) == 8

    roots = (
        ((1, 0), (3, 0), (5, 0)),
        ((1, 1), (3, 1), (5, 1)),
    )

    # Initial twenty prescriptions: two A families times ten B supports.
    # Every profile has seven columns and no dead protograph line.
    profiles = (
        ((1, 1),) * 7,
        ((2, 0), (0, 2)) + ((1, 1),) * 5,
        ((2, 0),) * 2 + ((0, 2),) * 2 + ((1, 1),) * 3,
        ((2, 0),) * 3 + ((0, 2),) * 3 + ((1, 1),),
        ((1, 0), (1, 2)) + ((1, 1),) * 5,
        ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
        ((1, 0), (3, 0)) + ((1, 1),) * 3 + ((0, 2),) * 2,
        ((1, 0), (0, 1), (2, 1), (1, 2)) + ((1, 1),) * 3,
        ((1, 0), (0, 1), (3, 0), (0, 3)) + ((1, 1),) * 3,
        ((1, 1),) * 5 + ((0, 2), (1, 0)),
    )
    assert all(len(p) == 7 for p in profiles)
    assert all(max(sum(c[i] for c in p) for i in range(2)) <= 7
               for p in profiles)

    def columns(rows, width):
        result = [0] * width
        for r, source in enumerate(rows):
            tag = 1 << r
            while source:
                bit = source & -source
                result[bit.bit_length() - 1] |= tag
                source ^= bit
        return result

    def logical_floor(checks, stabilizers, width):
        """Exact existence test through four, modulo the stabilizer span.

        Five means that logical weights 1..4 are excluded. It is not a
        heuristic distance reading. One representative per syndrome is
        sufficient for existence because differences lie in one quotient.
        """
        basis = {}
        for source in stabilizers:
            while source:
                p = source.bit_length() - 1
                previous = basis.get(p)
                if previous is None:
                    basis[p] = source
                    break
                source ^= previous

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                previous = basis.get(p)
                if previous is None:
                    return True
                word ^= previous
            return False

        cols = columns(checks, width)
        singles = {0: 0}
        for i, syndrome in enumerate(cols):
            word = 1 << i
            previous = singles.get(syndrome)
            if previous is None:
                singles[syndrome] = word
            elif nontrivial(previous ^ word):
                return (previous ^ word).bit_count()

        # First finish the complete test through three. Do not return
        # four early while an unvisited weight-three logical might exist.
        for i, first in enumerate(cols):
            for j in range(i):
                previous = singles.get(first ^ cols[j])
                if previous is not None:
                    difference = previous ^ (1 << i) ^ (1 << j)
                    if nontrivial(difference):
                        return difference.bit_count()

        representatives = dict(singles)
        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = (1 << i) | (1 << j)
                previous = representatives.get(syndrome)
                if previous is None:
                    representatives[syndrome] = word
                elif nontrivial(previous ^ word):
                    # The preceding pass already excluded weights <=3.
                    assert (previous ^ word).bit_count() == 4
                    return 4
        return 5

    def graph_key(hx, hz, width):
        """Intrinsic rooted contraction of the full uncoloured Tanner graph.

        High-degree qubits are identified by degree, not LP coordinates.
        Checks and low-qubit corners retain endpoint multiplicities.
        Root all high qubits; no unverified transitivity is assumed.
        """
        rows = list(hx) + list(hz)
        cols = columns(rows, width)
        high = [j for j, col in enumerate(cols) if col.bit_count() >= 10]
        position = {j: i for i, j in enumerate(high)}
        endpoints, tags, check_ends = [], [], []

        for row in rows:
            ends, bits = [], row
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                if j in position:
                    ends.append(position[j])
                bits ^= bit
            ends = tuple(sorted(ends))
            check_ends.append(ends)
            endpoints.append(ends)
            tags.append((0, row.bit_count()))

        for j, syndrome in enumerate(cols):
            if j in position:
                continue
            ends, weights = [], []
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                ends.extend(check_ends[r])
                weights.append(rows[r].bit_count())
                syndrome ^= bit
            endpoints.append(tuple(sorted(ends)))
            tags.append((1, tuple(sorted(weights))))

        incident = [[] for _ in high]
        adjacency = [set() for _ in high]
        for edge, ends in enumerate(endpoints):
            vertices = set(ends)
            for v in ends:
                incident[v].append(edge)
            for v in vertices:
                adjacency[v].update(vertices - {v})

        unseen = set(range(len(high)))
        component_sizes = []
        while unseen:
            start = unseen.pop()
            pending, size = [start], 1
            while pending:
                v = pending.pop()
                fresh = adjacency[v] & unseen
                unseen.difference_update(fresh)
                size += len(fresh)
                pending.extend(fresh)
            component_sizes.append(size)

        def compress(features):
            palette = {f: i for i, f in enumerate(sorted(set(features)))}
            return [palette[f] for f in features]

        rooted = []
        for root in range(len(high)):
            colours = [int(v == root) for v in range(len(high))]
            history = []
            for _ in range(3):
                es = [(tag, tuple(sorted(colours[v] for v in ends)))
                      for tag, ends in zip(tags, endpoints)]
                ec = compress(es)
                vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                      for v in range(len(high))]
                history.append((tuple(sorted(Counter(es).items())),
                                tuple(sorted(Counter(vs).items()))))
                colours = compress(vs)
            rooted.append(sha256(repr(history).encode("ascii")).digest())

        invariant = (
            width, len(rows),
            tuple(sorted(Counter(r.bit_count() for r in rows).items())),
            tuple(sorted(Counter(c.bit_count() for c in cols).items())),
            tuple(sorted(component_sizes)), tuple(sorted(rooted)),
            _tanner_signature(hx, hz),
        )
        return sha256(repr(invariant).encode("ascii")).digest()

    def incidence(top, bottom, voltages):
        B = [[], []]
        for g in top:
            B[0].append([G.e, g])
            B[1].append([])
        for g in bottom:
            B[0].append([])
            B[1].append([G.e, g])
        for g in voltages:
            B[0].append([G.e])
            B[1].append([g])
        return B

    # Exclusion references only. The complete unseen archive remains
    # authoritative; these references are never candidate outputs.
    old_A = [[[G.e, g] for g in roots[0]]]
    archive_cards = (
        ((), (), (0, 1, 2, 3, 6, 7, 9)),
        ((1,), (5,), (0, 2, 4, 6, 8)),
        ((1, 3), (3, 5), (0, 2, 6)),
        ((1, 3, 5), (1, 3, 5), (0,)),
        ((1, 3), (2, 3), (0, 4, 8)),
        ((), (), (6, 1, 3, 10, 5, 2, 0)),
        ((), (), (4, 9, 3, 8, 5, 0, 6)),
        ((1,), (5,), (6, 0, 2, 4, 10)),
        ((5, 3), (5, 3), (2, 0, 4)),
    )
    graph_keys = set()
    for top, bottom, voltages in archive_cards:
        B = incidence(tuple((a, 0) for a in top),
                      tuple((a, 0) for a in bottom),
                      tuple((a, 0) for a in voltages))
        hx, hz, n = lifted_matrices(G, old_A, B)
        graph_keys.add(graph_key(hx, hz, n))

    # Enumerate additional inverse-class prescriptions, rather than
    # treating permutations or inverse choices as deliberate designs.
    directions = tuple(sorted({
        min(g, G.inv(g)) for g in G.elements
        if g != G.e and G.mul(g, g) != G.e
    }))
    extra_A = []
    for triple in combinations(directions, 3):
        A = [[[G.e, g] for g in triple]]
        ar, _, _ = lifted_matrices(G, A, [[[]]])
        if _no_kernel_through_three(ar, 3 * G.q):
            extra_A.append(triple)
    bank = tuple(roots) + tuple(t for t in extra_A if t not in roots)

    def draw(attempt):
        card, epoch = attempt % 20, attempt // 20
        amodel, p = divmod(card, 10)
        triple = roots[amodel]
        if epoch >= 24 and epoch % 4 == 3:
            triple = bank[(epoch + card) % len(bank)]
        A = [[[G.e, g] for g in triple]]
        profile = profiles[p]

        # A new term-count branch spends a different pair of budgets.
        # Short constituent words are NOT automatically rejected: the
        # complete stabilizer quotient decides whether they are logical.
        if epoch >= 32 and epoch % 5 == 4:
            if card % 2:
                A = [[[G.e, triple[0]], [G.e, triple[1]],
                      local.sample(list(G.elements), 3)]]
            else:
                A = [[local.sample(list(G.elements), 3),
                      local.sample(list(G.elements), 3), [G.e]]]
            profile = ((1, 0), (0, 1)) + ((1, 1),) * 5

        subgroup = closure(triple)
        policy = epoch % 5
        pools = (order8, subgroup, rotations, outside12, tuple(G.elements))
        labels = pools[policy]
        gaps = tuple(sorted({
            min(g, G.inv(g)) for g in labels
            if g != G.e and G.mul(g, g) != G.e
        }))
        if not gaps:
            return None

        B = [[[] for _ in range(7)] for _ in range(2)]
        cross = [j for j, c in enumerate(profile) if c == (1, 1)]
        if len(cross) > len(labels):
            return None
        for j, voltage in zip(cross, local.sample(list(labels), len(cross))):
            B[0][j], B[1][j] = [G.e], [voltage]

        used = [set(), set()]
        for j, counts in enumerate(profile):
            if j in cross:
                continue
            for i, terms in enumerate(counts):
                if terms == 0:
                    continue
                if terms == 1:
                    B[i][j] = [local.choice(labels)]
                elif terms == 2:
                    choices = [g for g in gaps if g not in used[i]]
                    gap = local.choice(choices or gaps)
                    used[i].add(gap)
                    first = G.e if epoch % 2 == 0 else local.choice(labels)
                    B[i][j] = [first, G.mul(first, gap)]
                else:
                    if epoch % 3 == 0:
                        gap = local.choice(gaps)
                        B[i][j] = [G.e, gap, G.mul(gap, gap)]
                    else:
                        B[i][j] = local.sample(list(labels), terms)
        return A, B, (amodel, p, policy, epoch)

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    entries, filling_keys = [], set()
    wanted = max(count + 24, 40)
    for batch in range(128):
        proposals = []
        # Two trials per initial structural prescription in each batch.
        for trial in range(40):
            attempt = 40 * batch + trial
            proposal = draw(attempt)
            if proposal is None:
                continue
            A, B, card = proposal
            if any(len(cell) > 5 or len(set(cell)) != len(cell)
                   for grid in (A, B) for row in grid for cell in row):
                continue

            sa = [[len(c) for c in row] for row in A]
            sb = [[len(c) for c in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for grid in (A, B) for row in grid for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k > 0
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue
            proposals.append((-k, attempt, A, B, hx, hz, n, k, rx, rz, card))

        proposals.sort(key=lambda item: item[:2])
        accepted_batch = []
        for _, attempt, A, B, hx, hz, n, k, rx, rz, card in proposals:
            xf = logical_floor(hz, hx, n)
            if xf < 4:
                continue
            zf = logical_floor(hx, hz, n)
            if zf < 4:
                continue
            floor = min(xf, zf)
            key = graph_key(hx, hz, n)
            if key in graph_keys:
                continue
            graph_keys.add(key)

            if not css_ok(hx, hz):
                raise ValueError("verified redistributed lift failed CSS")
            if lifted_k(G, A, B) != k:
                raise ValueError("verified rank helpers disagree")

            quality = (k * floor * floor, floor, k, -attempt)
            screen = (
                ("k", k), ("logical_floors", (xf, zf)),
                ("logical_exclusion_through", floor - 1),
                ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
                ("design_card", card), ("tanner_invariant", key.hex()),
            )
            accepted_batch.append((quality, freeze(A), freeze(B), screen))

        accepted_batch.sort(key=lambda item: item[0], reverse=True)
        entries.extend(accepted_batch)
        if len(entries) >= wanted:
            break

    entries.sort(key=lambda item: item[0], reverse=True)
    # Never recycle entries to hide catalogue exhaustion.
    return tuple(entries[:count])


def _s38_columns(rows, width):
    cols = [0] * width
    for r, source in enumerate(rows):
        while source:
            bit = source & -source
            cols[bit.bit_length() - 1] |= 1 << r
            source ^= bit
    return cols


def _s38_basis(rows):
    basis = {}
    for source in rows:
        while source:
            p = source.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = source
                break
            source ^= previous
    return basis


def _s38_outside(word, basis):
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            return True
        word ^= previous
    return False


def _s38_component_rows(G, grid, side):
    """Use the evaluator lift with a unit opposite side."""
    unit = [[[G.e]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
    mask = (1 << (8 * G.q)) - 1
    return [row & mask for row in rows[:3 * G.q]]


def _s38_kernel6(rows, width):
    """Exact existence through six; called after exclusion through four.

    A result of seven means exclusion through six, not a search reading.
    All supports of weight <=6 split into two supports of size <=3.
    Continue after a six-collision so that no later five-word is missed.
    """
    cols = _s38_columns(rows, width)
    bits = [1 << j for j in range(width)]
    small = {0: 0}
    for i, syndrome in enumerate(cols):
        small[syndrome] = bits[i]
    for i, first in enumerate(cols):
        for j in range(i):
            small[first ^ cols[j]] = bits[i] | bits[j]

    triples = {}
    witness, collisions = 0, 0
    for i, first in enumerate(cols):
        for j in range(i):
            pair_s = first ^ cols[j]
            pair_w = bits[i] | bits[j]
            for h in range(j):
                syndrome = pair_s ^ cols[h]
                word = pair_w | bits[h]
                old = small.get(syndrome)
                if old is not None:
                    difference = word ^ old
                    if difference:
                        return difference.bit_count(), difference, collisions
                old = triples.get(syndrome)
                if old is None:
                    triples[syndrome] = word
                else:
                    difference = word ^ old
                    if difference:
                        assert difference.bit_count() == 6
                        collisions += 1
                        if not witness:
                            witness = difference
    return (6 if witness else 7), witness, collisions


def _s38_probe(rng, rows, width, seed=0, passes=12, own=None):
    """Actual kernel/logical witnesses from bounded information-set search."""
    cols = _s38_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def inspect(word):
        nonlocal best, weight
        w = word.bit_count()
        if not word or w >= weight:
            return
        if own is not None and not _s38_outside(word, own):
            return
        best, weight = word, w

    order = list(range(width))
    for _ in range(passes):
        rng.shuffle(order)
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

        active = sorted(kernel, key=int.bit_count)[:72]
        for i in range(len(active)):
            current = active[i]
            w = current.bit_count()
            for j, other in enumerate(active):
                if i == j:
                    continue
                trial = current ^ other
                tw = trial.bit_count()
                if 0 < tw < w:
                    current, w = trial, tw
            active[i] = current
            inspect(current)

        active.sort(key=int.bit_count)
        for i, first in enumerate(active):
            for j in range(i):
                pair = first ^ active[j]
                inspect(pair)
                if i < 20:
                    for h in range(j):
                        inspect(pair ^ active[h])

        if own is not None and weight < 6:
            break
        if own is None and weight <= 6:
            break

    assert best
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    assert own is None or _s38_outside(best, own)
    return best


def _s38_draw(rng, G, card):
    """Five genuine support motifs, each with four relative-label policies."""
    profiles = (
        # One grounded column, four core columns, three cubic columns.
        ((1,1,0,0,0,2,2,0),
         (0,1,1,1,1,0,0,2),
         (0,0,1,1,1,1,1,1)),
        # A connected matching core with pervasive cubic columns.
        ((0,1,1,1,0,1,1,1),
         (0,1,1,0,1,1,1,1),
         (1,0,0,1,1,1,1,1)),
        # Two grounded layers; the remaining core loop stays ungrounded.
        ((1,0,1,0,1,1,1,1),
         (0,1,1,0,1,1,1,1),
         (0,0,0,2,1,1,1,1)),
        # First two augmentation rows equal; split cubic reinforcements.
        ((0,1,1,1,0,2,0,1),
         (0,1,1,1,0,0,2,1),
         (1,0,0,0,2,1,1,1)),
        # The same rank mechanism with all cubic columns pervasive.
        ((0,1,1,1,0,1,1,1),
         (0,1,1,1,0,1,1,1),
         (1,0,0,0,2,1,1,1)),
    )
    pi, mode = divmod(card, 4)
    profile = profiles[pi]
    assert all(sum(row) == 6 for row in profile)
    assert max(sum(profile[i][j] for i in range(3))
               for j in range(8)) == 3

    els = G.elements
    grid = [[[] for _ in range(8)] for _ in range(3)]
    for j in range(8):
        occupied = [i for i in range(3) if profile[i][j]]
        anchor_row = occupied[0]
        for i in occupied:
            count = profile[i][j]
            anchor = 0 if i == anchor_row else rng.randrange(8)
            if count == 1:
                cell = [els[anchor]]
            else:
                exclusive = len(occupied) == 1
                if mode == 0:
                    gap = 1
                elif mode == 1:
                    gap = 3
                elif mode == 2 or exclusive:
                    gap = rng.choice((1, 3))
                else:
                    # Order-four gaps are admitted only in reinforced
                    # columns, not as isolated short incidence cycles.
                    gap = rng.choice((1, 2, 3))
                cell = [els[anchor], els[(anchor + gap) % 8]]
            assert len(set(cell)) == count
            grid[i][j] = cell

    if pi >= 3:
        parity = [
            sum((count & 1) << j for j, count in enumerate(row))
            for row in profile
        ]
        assert parity[0] == parity[1]
        assert gf2_rank(parity) == 2
    return grid


def _s38_component_bank(G, side):
    """Independent, equally screened constituent finalists."""
    local = random.Random(
        BASE_SEED ^ (0x835061 if side == "A" else 0x835062))
    bank, fillings = [], set()
    width = 8 * G.q

    for card in range(20):
        raw = []
        for trial in range(192):
            grid = _s38_draw(local, G, card)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _s38_component_rows(G, grid, side)
            rank = gf2_rank(rows)
            if not 22 <= rank <= 24:
                continue
            if _kernel_profile(rows, width)[0] != 5:
                continue
            penalty = sum(
                ((rows[i] & rows[j]).bit_count()
                 * ((rows[i] & rows[j]).bit_count() - 1) // 2)
                for i in range(len(rows)) for j in range(i))
            rank_priority = 0 if rank == 23 else 1 if rank == 24 else 2
            raw.append(((rank_priority, penalty, trial),
                        grid, rows, rank, penalty, trial))
        raw.sort(key=lambda item: item[0])

        refined = []
        for _, grid, rows, rank, penalty, trial in raw[:40]:
            floor, seed, burden = _s38_kernel6(rows, width)
            if floor < 6:
                continue
            # A six-word is already an exact constituent minimum.
            word = (seed if seed else
                    _s38_probe(local, rows, width, passes=16))
            reading = word.bit_count()
            assert reading >= floor
            quality = (reading, int(rank == 23), -burden, -penalty, -trial)
            refined.append((quality, grid, rows, rank, word,
                            floor, burden, penalty, trial))

        refined.sort(key=lambda item: item[0], reverse=True)
        for _, grid, rows, rank, word, floor, burden, penalty, trial in refined[:6]:
            bank.append({
                "grid": grid, "rows": rows, "rank": rank,
                "word": word, "reading": word.bit_count(),
                "kernel_floor": floor, "burden": burden,
                "penalty": penalty, "card": card, "trial": trial,
            })
    return bank


def _s38_pair_reading(G, a, b, hx, hz, n, rng):
    """Embed actual constituent witnesses and search their logical classes."""
    q = G.q
    best = n + 1
    for kind, checks, stabilizers in (("X", hz, hx), ("Z", hx, hz)):
        own = _s38_basis(stabilizers)
        source = b["word"] if kind == "X" else a["word"]
        seeds = []
        for block in range(8):
            if kind == "X":
                word = source << (block * 8 * q)
            else:
                word, bits = 0, source
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 8 + block) * q + h)
                    bits ^= bit
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            if not _s38_outside(word, own):
                continue

            for _ in range(2):
                previous = word
                weight = word.bit_count()
                for row in stabilizers:
                    trial = word ^ row
                    if trial.bit_count() < weight:
                        word, weight = trial, trial.bit_count()
                starting = word
                moves = sorted(
                    (row for row in stabilizers
                     if (row & starting).bit_count() >= 2),
                    key=lambda row: -(row & starting).bit_count())[:24]
                for i, first in enumerate(moves):
                    for second in moves[:i]:
                        trial = starting ^ first ^ second
                        if trial.bit_count() < weight:
                            word, weight = trial, trial.bit_count()
                if word == previous:
                    break
            assert word and _s38_outside(word, own)
            seeds.append(word)

        if not seeds:
            return None
        seed = min(seeds, key=int.bit_count)
        word = _s38_probe(rng, checks, n, seed, 3, own)
        best = min(best, word.bit_count())
        if best < 6:
            return best
    return best


@lru_cache(maxsize=1)
def _s38_portfolio(G, count):
    """Immutable deterministic portfolio; no slot-dependent state."""
    from hashlib import sha256

    assert G.q == 8 and A_SHAPE == B_SHAPE == (3, 8)
    aa = _s38_component_bank(G, "A")
    bb = _s38_component_bank(G, "B")
    if not aa or not bb:
        raise ValueError("no six-screened saturated constituent bank")

    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            weak = min(a["reading"], b["reading"])
            da, db = 24 - a["rank"], 24 - b["rank"]
            # Ordering hypothesis only; exact full ranks gate publication.
            predicted = 200 + 5 * (da + db) + 2 * min(da, db)
            priority = (predicted * weak * weak, weak, predicted,
                        -a["burden"] - b["burden"],
                        -a["penalty"] - b["penalty"])
            plans.append((priority, ai, bi))
    plans.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)

    def graph_key(hx, hz, n):
        cols = _s38_columns(list(hx) + list(hz), n)
        invariant = (
            n, tuple(sorted(col.bit_count() for col in cols)),
            _tanner_signature(hx, hz),
        )
        return sha256(repr(invariant).encode("ascii")).digest()

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    keys, entries = set(), []
    cap = min(9, W_MAX)
    for _, ai, bi in plans[:max(1600, 32 * count)]:
        a, b = aa[ai], bb[bi]
        A, B = a["grid"], b["grid"]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 9
        if _grid_weights(sa, sb) > cap:
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 584 and k > 200
                and 325 * n <= 1000 * k < 375 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        # Exact full-matrix tests include both LP sectors.
        if _kernel_profile(hx, n)[0] != 5:
            continue
        if _kernel_profile(hz, n)[0] != 5:
            continue
        key = graph_key(hx, hz, n)
        if key in keys:
            continue

        local = random.Random(
            BASE_SEED ^ (0x8350CC + 65537 * ai + 1009 * bi))
        reading = _s38_pair_reading(G, a, b, hx, hz, n, local)
        if reading is None or reading < 6:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified saturated lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified saturated rank helpers disagree")

        keys.add(key)
        quality = (k * reading * reading / n, reading, k, -ai, -bi)
        screen = (
            ("k", k), ("rank_gain_above_bound", k - 200),
            ("logical_witness_upper_bound", reading),
            ("constituent_readings", (a["reading"], b["reading"])),
            ("constituent_kernel_floors",
             (a["kernel_floor"], b["kernel_floor"])),
            ("constituent_ranks", (a["rank"], b["rank"])),
            ("design_cards", (a["card"], b["card"])),
            ("LP_rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("full_kernel_exclusion_through", 4),
            ("tanner_invariant", key.hex()),
        )
        entries.append((quality, freeze(A), freeze(B), screen))
        if len(entries) >= count + 24:
            break

    entries.sort(key=lambda item: item[0], reverse=True)
    return tuple(entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of a fully defined target-cell design; never recycle."""
    portfolio = _s38_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("novel rank-gain saturated portfolio exhausted")
    quality, A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "quality": quality, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without resampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supplied twenty inequivalent initial designs on C12:C2(s=5); slots 0..19 collapsed to four distinct codes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: The common-C12 catalogue supplies twenty inequivalent k=258 designs: sixty evaluated slots retained only five downstream classes, exhausted after slot 51, and all five measured codes still had d30s=4.
# REFUTED: The order-eight-coupling redistribution model improves the common-C12 distance-four record: the measured child scored 7.4200 versus 7.4780, and all five measured leaders still had d30s=4 with abundant sector-1 witnesses.
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
