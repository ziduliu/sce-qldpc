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
CONCEPT = ("Frozen C3 semidirect C4 with action 2, order 12, A=(2,6), B=(2,7): triangle-free non-bipartite B incidence graphs containing order-six internal edges should supply five fresh k>=258, distance-four finalists and improve the parent's fitness 7.4258.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. This bed measured k=258,d30s=4, but sixty parent slots retained only ten classes. The normal C6 contains the order-six inverse class represented by (1,2), which the parent's order-four-only internal-edge policy excluded. Its utility is tested on verified lifts, not inferred from its order.")
SHAPES = ("Keep A=(2,6), B=(2,7), q=12: n=552, kb=240, guaranteed rate 10/23=0.43478, and q<=13 permitted by the size window. Unequal constituents have 72 variables/24 checks and 84 variables/24 checks. Parent minima were entirely sector 1, prominently X Acol4 and Z Acols3..5/Bcols3..5. Two diagonal A incidence components retain the measured rank-deficit mechanism while B connectivity and odd-cycle structure change. Verified full ranks retain [0.425,0.475).")
FILLING = ("Exhaustively enumerate nineteen unordered internal-gap coupling cards with one or two internal columns per B row and at least one order-six gap. Cross-voltage subsets contain identity to remove a bridge gauge. Verified B matrices must be connected, triangle-free and non-bipartite; no random gauge copies are published. A rows total six, B rows seven, and all columns total two, giving HX=8,HZ=9. A finite optimized portfolio prioritizes actual k>=258, longer odd girth and fewer constituent four-word collisions. Complete kernels exclude weights 1..3. Full rooted uncoloured Tanner refinement supplements the parent's high-degree contraction and rejects repeats and shown archive references. Five fresh k=258,d30s=4 readings would score 7.47826; their yield and unseen-archive novelty remain falsifiable.")
FAMILY_NAME = "r045_m3r4s2_26x27_rooted_incidence_rank_v3_order6_oddcycles"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 12

def custom_ops(q):
    # Same semidirect-product rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 46 requires q<=13; 2^4=1 modulo 3.
    m, r, s = 3, 4, 2
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
    """Three order-four edge classes and the normal order-six subgroup."""
    steps = sorted({min(g, G.inv(g))
                    for g in G.elements if g[1] % 2})
    subgroup = [g for g in G.elements if g[1] % 2 == 0]
    assert len(steps) == 3 and len(subgroup) == 6
    assert all(G.mul(g, g) != G.e for g in steps)
    return steps, subgroup


def _draw_incidence(rng, G, attempt, pools):
    """Four A prescriptions times five B support prescriptions."""
    steps, subgroup = pools
    card, epoch = attempt % 20, attempt // 20
    amode, bmode = card % 4, card // 4

    def regular(width, internal, labels=None):
        C = [[], []]
        for i in range(2):
            for gap in rng.sample(steps, internal):
                C[i].append([G.e, gap])
                C[1 - i].append([])
        cross = width - 2 * internal
        if labels is None:
            # Odd internal gaps reverse b-parity. Cross differences
            # preserve it, giving a sufficient triangle-free grading.
            pool = subgroup if internal else G.elements
            labels = rng.sample(pool, cross)
        assert len(labels) == cross
        for voltage in labels:
            C[0].append([G.e])
            C[1].append([voltage])
        assert all(len(row) == width for row in C)
        assert all(sum(map(len, row)) == width for row in C)
        return C

    if amode == 0:
        # Six distinct subgroup matchings give two K6,6 components.
        A = regular(6, 0, list(subgroup))
    elif amode == 1:
        outside = [g for g in G.elements if g[1] % 2]
        count = 1 + epoch % 5
        labels = (rng.sample(subgroup, 6 - count)
                  + rng.sample(outside, count))
        A = regular(6, 0, labels)
    elif amode == 2:
        A = regular(6, 1)
    else:
        A = regular(6, 2 + epoch % 2)

    if bmode < 4:
        B = regular(7, bmode)
    else:
        # Row sums seven, column degrees 1,3,2,2,2,2,2.
        counts = ((1, 2, 1, 1, 1, 1, 0),
                  (0, 1, 1, 1, 1, 1, 2))
        B = [[[] for _ in range(7)] for _ in range(2)]
        for i in range(2):
            for j, count in enumerate(counts[i]):
                if count == 1:
                    B[i][j] = [G.pick(rng)]
                elif count == 2:
                    first = G.pick(rng)
                    gap = rng.choice(steps)
                    B[i][j] = [first, G.mul(first, gap)]

    assert all(sum(map(len, row)) == 6 for row in A)
    assert all(sum(map(len, row)) == 7 for row in B)
    assert all(len(A[0][j]) + len(A[1][j]) == 2
               for j in range(6))
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


def _catalogue_invariant(hx, hz, n):
    """Intrinsic rooted invariant, including nonregular hyperedges.

    Degree-13 qubits identify sector 2 without using block labels.
    Check weights colour its projected hyperedges. Root every such
    qubit, rather than assuming a transitive full-code group action.
    Equality is inconclusive and is conservatively rejected.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    columns = [0] * n
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    high = [j for j, col in enumerate(columns)
            if col.bit_count() == 13]
    positions = {j: t for t, j in enumerate(high)}
    edges, weights = [], []
    incident = [[] for _ in high]
    for row in rows:
        endpoints = []
        bits = row
        while bits:
            bit = bits & -bits
            j = bit.bit_length() - 1
            if j in positions:
                endpoints.append(positions[j])
            bits ^= bit
        edge = len(edges)
        edges.append(tuple(endpoints))
        weights.append(row.bit_count())
        for v in endpoints:
            incident[v].append(edge)

    def compress(signatures):
        table = {s: i for i, s in enumerate(sorted(set(signatures)))}
        return [table[s] for s in signatures]

    rooted = []
    for root in range(len(high)):
        colours = [int(v == root) for v in range(len(high))]
        history = []
        for _ in range(5):
            es = [(w, tuple(sorted(colours[v] for v in endpoints)))
                  for w, endpoints in zip(weights, edges)]
            ec = compress(es)
            vs = [(colours[v], tuple(sorted(ec[e] for e in incident[v])))
                  for v in range(len(high))]
            history.append((tuple(sorted(Counter(es).items())),
                            tuple(sorted(Counter(vs).items()))))
            colours = compress(vs)
        rooted.append(sha256(repr(tuple(history)).encode()).digest())
    return (n, tuple(sorted(Counter(weights).items())),
            tuple(sorted(Counter(c.bit_count()
                                 for c in columns).items())),
            tuple(sorted(rooted)))


def _shown_archive_invariants():
    """Only exclusion references; never submit these archived fillings."""
    els = [(a, b) for b in range(2) for a in range(12)]

    def mul(g, h):
        return ((g[0] + pow(5, g[1], 12) * h[0]) % 12,
                (g[1] + h[1]) % 2)

    def inv(g):
        b = (-g[1]) % 2
        return ((-pow(5, b, 12) * g[0]) % 12, b)

    old = _G(dict(elements=els, ident=(0, 0), mul=mul, inv=inv), 24)
    e = old.e

    def incidence(top, bottom, ratios):
        B = [[], []]
        for gap in top:
            B[0].append([e, gap])
            B[1].append([])
        for gap in bottom:
            B[0].append([])
            B[1].append([e, gap])
        for voltage in ratios:
            B[0].append([e])
            B[1].append([voltage])
        return B

    references = [
        (((3,1),(5,1),(1,1)), (), (),
         ((7,1),(9,1),(10,1),(5,1),(0,0),(3,1),(5,0))),
        (((3,0),(1,0),(5,0)), ((5,1),), ((1,0),),
         ((0,0),(8,1),(6,0),(10,1),(2,0))),
        (((5,1),(1,1),(3,1)), ((1,1),(1,0)), ((3,0),(5,1)),
         ((6,1),(8,1),(10,1))),
        (((3,0),(5,0),(1,0)), ((1,1),(1,0)), ((3,0),(5,0)),
         ((6,1),(10,1),(4,0))),
        (((5,1),(1,1),(3,1)), ((3,1),(1,1),(3,0)),
         ((1,0),(5,1),(3,1)), ((2,1),)),
    ]
    common = (
        ((), (), (0,1,2,3,6,7,9)),
        ((1,), (5,), (0,2,4,6,8)),
        ((1,3), (3,5), (0,2,6)),
        ((1,3,5), (1,3,5), (0,)),
        ((5,1), (5,3), (0,4,8)),
    )
    for top, bottom, ratios in common:
        references.append(
            (((1,0),(3,0),(5,0)),
             tuple((d,0) for d in top),
             tuple((d,0) for d in bottom),
             tuple((d,0) for d in ratios)))

    result = set()
    for generators, top, bottom, ratios in references:
        A = [[[e, g] for g in generators]]
        B = incidence(top, bottom, ratios)
        hx, hz, n = lifted_matrices(old, A, B)
        result.add(_catalogue_invariant(hx, hz, n))
    return result


from functools import lru_cache


@lru_cache(maxsize=1)
def _order6_portfolio(G, count):
    """Immutable exhaustive element-model portfolio, independent of call order.

    All ranks, constituent screens and graph features derive from the
    scaffold's verified matrices. Memoization changes only build cost.
    """
    from itertools import combinations
    from collections import Counter
    from hashlib import sha256

    cap = min(9, W_MAX)
    outside, _ = _odd_inverse_classes(G)
    sixth = next(g for g in G.elements if g == (1, 2))
    assert G.mul(G.mul(sixth, sixth), sixth) == (0, 2)
    assert G.mul((0, 2), (0, 2)) == G.e

    def incidence(width, top, bottom, voltages):
        grid = [[], []]
        for gap in top:
            grid[0].append([G.e, gap])
            grid[1].append([])
        for gap in bottom:
            grid[0].append([])
            grid[1].append([G.e, gap])
        for voltage in voltages:
            grid[0].append([G.e])
            grid[1].append([voltage])
        assert all(len(row) == width for row in grid)
        assert all(len(set(cell)) == len(cell)
                   for row in grid for cell in row)
        return grid

    A = incidence(6, tuple(outside), tuple(outside), ())
    sa = [[len(cell) for cell in row] for row in A]
    unit = [[[G.e]]]
    arows, _, _ = lifted_matrices(G, A, unit)
    arows = [row & ((1 << (6 * G.q)) - 1) for row in arows]
    qa = _kernel_profile(arows, 6 * G.q)
    if qa[0] < 4 or gf2_rank(arows) != 2 * G.q - 2:
        raise ValueError("diagonal A failed verified incidence screening")

    def columns(rows, width):
        result = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                result[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return result

    def odd_girth(rows, width):
        """Exact shortest odd cycle of the verified degree-two constituent.

        Parity-expanded BFS finds odd closed walks. Taking the minimum
        over every root gives the shortest odd cycle. Zero means bipartite.
        """
        adjacency = [[] for _ in rows]
        for syndrome in columns(rows, width):
            if syndrome.bit_count() != 2:
                raise ValueError("B proposal is not a simple incidence column")
            bit = syndrome & -syndrome
            u = bit.bit_length() - 1
            v = (syndrome ^ bit).bit_length() - 1
            adjacency[u].append(v)
            adjacency[v].append(u)

        best = 2 * len(rows) + 1
        for root in range(len(rows)):
            distance = [[-1, -1] for _ in rows]
            distance[root][0] = 0
            queue = [(root, 0)]
            for u, parity in queue:
                d = distance[u][parity]
                if d + 1 >= best:
                    continue
                for v in adjacency[u]:
                    other = parity ^ 1
                    if distance[v][other] != -1:
                        continue
                    distance[v][other] = d + 1
                    if v == root and other:
                        best = min(best, d + 1)
                    else:
                        queue.append((v, other))
        return 0 if best > 2 * len(rows) else best

    def full_root_key(hx, hz, n):
        """Rooted refinement of the COMPLETE uncoloured Tanner graph.

        Roots are selected intrinsically by degree thirteen. Refinement
        retains all checks and low-degree qubits, not just the parent's
        sector-two contraction. Unequal keys certify inequivalence;
        equality remains inconclusive and is rejected conservatively.
        """
        rows = list(hx) + list(hz)
        neighbours = [[] for _ in range(n + len(rows))]
        for r, row in enumerate(rows):
            check = n + r
            while row:
                bit = row & -row
                j = bit.bit_length() - 1
                neighbours[j].append(check)
                neighbours[check].append(j)
                row ^= bit
        degrees = [len(ns) for ns in neighbours]
        roots = [v for v, degree in enumerate(degrees) if degree == 13]

        def compress(features):
            palette = {f: c for c, f in enumerate(sorted(set(features)))}
            return [palette[f] for f in features]

        digests = []
        for root in roots:
            initial = [(degree, int(v == root))
                       for v, degree in enumerate(degrees)]
            colours = compress(initial)
            history = []
            for _ in range(6):
                features = [
                    (colours[v], tuple(sorted(colours[u] for u in ns)))
                    for v, ns in enumerate(neighbours)
                ]
                history.append(tuple(sorted(Counter(features).items())))
                colours = compress(features)
            digests.append(sha256(repr(tuple(history)).encode()).digest())
        return (tuple(sorted(Counter(degrees).items())),
                tuple(sorted(digests)))

    def signature(hx, hz, n):
        return (_catalogue_invariant(hx, hz, n),
                full_root_key(hx, hz, n))

    def freeze(grid):
        return tuple(tuple(tuple(cell) for cell in row) for row in grid)

    # Exclusions only: none of these archived fillings is a proposal.
    references = [
        (((0,1),(1,1)), ((2,1),(1,1)),
         ((1,2),(2,0),(0,2))),
        ((), (), ((2,3),(0,2),(1,3),(2,2),(1,2),(0,0),(2,0))),
        (((1,1),), ((0,1),),
         ((1,2),(2,2),(2,0),(0,2),(0,0))),
    ]

    # Convert the shown isomorphic Dic3 bed's four k=258 anchors.
    # Dic3 x maps to (1,2), and its y maps to (0,1).
    def convert(g):
        a, b = g
        return (a % 3, (2 * a + b) % 4)

    dic3_references = (
        ((), (), ((0,1),(3,1),(2,0),(1,0),(1,1),(0,0),(4,1))),
        (((1,1),(0,1)), ((2,1),(0,1)), ((3,0),(0,0),(4,0))),
        (((1,1),), ((2,1),),
         ((2,0),(0,0),(3,0),(1,0),(5,0))),
        (((2,1),(0,1),(1,1)), ((0,1),(1,1),(2,1)), ((5,0),)),
    )
    references.extend(
        (tuple(map(convert, top)), tuple(map(convert, bottom)),
         tuple(map(convert, voltages)))
        for top, bottom, voltages in dic3_references
    )

    graph_keys = set()
    for top, bottom, voltages in references:
        B = incidence(7, top, bottom, voltages)
        hx, hz, n = lifted_matrices(G, A, B)
        graph_keys.add(signature(hx, hz, n))
    old_high_keys = _shown_archive_invariants()

    # Nineteen finite gap-coupling prescriptions, not twenty label draws.
    # Row exchange is already quotiented by unordered edge-set pairs.
    directions = tuple(sorted(outside + [sixth], key=G.idx))
    cards = []
    for internal in (1, 2):
        edge_sets = list(combinations(directions, internal))
        for i, top in enumerate(edge_sets):
            for bottom in edge_sets[i:]:
                if sixth in top or sixth in bottom:
                    cards.append((top, bottom))
    assert len(cards) == 19

    pool = []
    others = tuple(g for g in G.elements if g != G.e)
    filling_keys = set()
    for card, (top, bottom) in enumerate(cards):
        cross = 7 - len(top) - len(bottom)
        # A bridge gauge can normalize one voltage to identity; the
        # retained order-four/six edge classes are closed under conjugacy
        # and inversion. Full graph screening still decides equivalence.
        for tail in combinations(others, cross - 1):
            voltages = (G.e,) + tail
            B = incidence(7, top, bottom, voltages)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in B for cell in row)
            if filling in filling_keys:
                continue
            filling_keys.add(filling)
            sb = [[len(cell) for cell in row] for row in B]
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > cap:
                continue
            assert all(sum(row) == 7 for row in sb)
            assert all(sb[0][j] + sb[1][j] == 2 for j in range(7))

            _, brows, _ = lifted_matrices(G, unit, B)
            brows = [row & ((1 << (7 * G.q)) - 1) for row in brows]
            qb = _kernel_profile(brows, 7 * G.q)
            if qb[0] < 4:
                continue
            if gf2_rank(brows) != 2 * G.q - 1:
                continue
            og = odd_girth(brows, 7 * G.q)
            if og < 5:
                continue

            hx, hz, n = lifted_matrices(G, A, B)
            rx, rz = gf2_rank(hx), gf2_rank(hz)
            k = n - rx - rz
            if not (n == 552 and k >= 258
                    and 425 * n <= 1000 * k < 475 * n):
                continue
            if max(row.bit_count() for row in hx + hz) > cap:
                continue
            priority = (-k, -og, qb[1], card, filling)
            pool.append((priority, B, hx, hz, n, k, rx, rz, qb,
                         card, og, top, bottom, voltages))

    pool.sort(key=lambda record: record[0])
    entries = []
    target = max(count + 12, 32)
    for (_, B, hx, hz, n, k, rx, rz, qb,
         card, og, top, bottom, voltages) in pool:
        coarse = _catalogue_invariant(hx, hz, n)
        if coarse in old_high_keys:
            continue
        key = (coarse, full_root_key(hx, hz, n))
        if key in graph_keys:
            continue
        if not (_no_kernel_through_three(hx, n)
                and _no_kernel_through_three(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified order-six lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified rank helpers disagree")
        graph_keys.add(key)
        screen = (
            ("k", k),
            ("component_screens", (qa, qb)),
            ("B_odd_girth", og),
            ("B_connected", True),
            ("B_non_bipartite", True),
            ("design_card", (card, top, bottom, voltages)),
            ("LP_rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("complete_kernel_exclusion_through", 3),
        )
        entries.append((freeze(A), freeze(B), screen))
        if len(entries) >= target:
            break

    # No recycled codes or relaxation to archived order-four policies.
    return tuple(entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup of an optimized order-six/odd-cycle filling."""
    portfolio = _order6_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("fresh order-six odd-cycle portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell from the immutable screened portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supply twenty inequivalent initial designs on C12:C2(s=5): slots 0..19 collapsed to four classes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: High-degree-only rooted refinement prevents evaluator duplicate collapse on C3:C4 (2,6)x(2,7): sixty parent slots retained ten classes, and all five measured leaders still had d30s=4 with exclusively sector-1 minima.
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
