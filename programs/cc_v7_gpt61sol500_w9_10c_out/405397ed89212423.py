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
CONCEPT = ("C11 semidirect C1, order 11, A=B=(2,7): singleton-anchored triangle-free constituents should supply five novel k=275 codes with distance readings >=4, scoring 7.5472 in cell 0.450. Verified ranks and complete short-kernel screens make the construction falsifiable.")
GROUP = ("Preserve the cyclic-by-cyclic semidirect-product construction rule and custom encoding. New factor 53 excludes order 24 and permits q<=11; m=11,r=1,s=1 is the valid trivial-quotient specialization of that same rule. The existing Z11 incidence bed measured distance four but its shown rank-deficient leaders occupied cell 0.500. Those readings motivate a full-rank singleton-anchored alternative, not a distance transfer or an assertion that this bed is untried.")
SHAPES = ("Change to equal A=B=(2,7), with 77 variables and 22 checks per constituent. At q=11, n=583, kb=275 and guaranteed rate 25/53=0.47170 lie inside cell 0.450. Singleton anchors are intended to remove augmentation rank deficits; exact full ranks must give k=275. Parent minimum logicals occupied sector-1 Acol0..2/Bcol0..6 broadly, especially Acol1 and Bcol5. Both new seven-column constituents and four sector-2 block pairs replace that layout.")
FILLING = ("Exhaustively catalogue translation-normalized six-matching sets and five-matching/internal-cycle alternatives screened against triangles. Each constituent has row totals six and seven, with column degrees one or two: BOTH CSS row-weight maxima are nine. Constituent weight-four collision diagnostics order deliberate pair prescriptions. Verified full lifts require k=275, CSS and complete kernel exclusion through weight three. Rooted full uncoloured Tanner refinement supplements check-intersection invariants and conservatively rejects repeats without ownership lotteries. Immutable memoization changes build cost only. Distance above four and novelty against the unavailable archive remain unproved.")
FAMILY_NAME = "r045_m11r1s1_27x27_singleton_fullrank_portfolio_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    # Same semidirect-product rule: x^m=y^r=1, y*x*y^-1=x^s.
    # Factor 53 excludes order 24. A trivial quotient remains this rule.
    m, r, s = 11, 1, 1
    if not valid_q(q):
        raise ValueError("this semidirect product has order 11")
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

A_SHAPE = (2, 7)               # (ra, ca)
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


def _unused_parent_quotient_ctx(rng, G, idx):
    """Uncalled historical order-24 prescription; never publishes cells."""
    import hashlib

    local = random.Random(BASE_SEED ^ 0xC1253045 ^ (104729 * idx))
    q = G.q
    sa = [[2, 2, 2]]
    card = idx % 20
    amodel, bmodel = divmod(card, 5)

    # Each column pair is (top count, bottom count). Every profile has
    # seven columns, both row sums seven, and column sums at most three.
    profiles = (
        ((1, 1),) * 7,
        ((2, 0), (0, 2)) + ((1, 1),) * 5,
        ((2, 0),) * 2 + ((0, 2),) * 2 + ((1, 1),) * 3,
        ((2, 0),) * 3 + ((0, 2),) * 3 + ((1, 1),),
        ((1, 0), (3, 0)) + ((1, 1),) * 3 + ((0, 2),) * 2,
        ((1, 0), (2, 1)) + ((1, 1),) * 4 + ((0, 2),),
        ((1, 0), (1, 2)) + ((1, 1),) * 5,
        ((1, 0), (0, 1), (2, 1), (1, 2)) + ((1, 1),) * 3,
    )
    assert all(len(p) == 7 and
               all(sum(c[i] for c in p) == 7 for i in range(2))
               for p in profiles)
    assert min(9, W_MAX) >= 9

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    def logical_floor(checks, stabilizers, width):
        """Exact existence screen through four, modulo stabilizers.

        Equal syndromes of supports of size <=2 cover every support
        of size <=4. One representative per syndrome suffices for
        existence: differences that reduce to zero are stabilizers.
        Five means exclusion through four, not a distance estimate.
        The collision burden is a diagnostic, not an operator count.
        """
        span = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                if p not in span:
                    span[p] = row
                    break
                row ^= span[p]

        def nontrivial(word):
            while word:
                p = word.bit_length() - 1
                old = span.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        cols = columns(checks, width)
        reps = {0: 0}
        floor, burden = 5, 0
        for i, syndrome in enumerate(cols):
            word = 1 << i
            old = reps.get(syndrome)
            if old is None:
                reps[syndrome] = word
            elif nontrivial(old ^ word):
                return (old ^ word).bit_count(), 0

        for i, first in enumerate(cols):
            for j in range(i):
                syndrome = first ^ cols[j]
                word = (1 << i) | (1 << j)
                old = reps.get(syndrome)
                if old is None:
                    reps[syndrome] = word
                    continue
                difference = old ^ word
                if not difference:
                    continue
                weight = difference.bit_count()
                if weight <= 3:
                    if nontrivial(difference):
                        return weight, burden
                elif nontrivial(difference):
                    floor = 4
                    burden += 1
        return floor, burden

    def fingerprint(hx, hz, width):
        """Intrinsic invariant, pooled without X/Z or block labels."""
        rows = hx + hz
        cols = columns(rows, width)
        degrees = [c.bit_count() for c in cols]
        qubit_profiles = []
        for col in cols:
            incident = []
            while col:
                bit = col & -col
                incident.append(rows[bit.bit_length() - 1].bit_count())
                col ^= bit
            qubit_profiles.append(tuple(sorted(incident)))
        return (tuple(sorted(degrees)),
                tuple(sorted(qubit_profiles)),
                _tanner_signature(hx, hz))

    def closure(generators):
        reached, todo = {G.e}, [G.e]
        steps = tuple(generators) + tuple(G.inv(g) for g in generators)
        for first in todo:
            for second in steps:
                product = G.mul(first, second)
                if product not in reached:
                    reached.add(product)
                    todo.append(product)
        return tuple(sorted(reached))

    inverse_classes = tuple(sorted({
        min(g, G.inv(g)) for g in G.elements if g != G.e
    }))
    odd_classes = tuple(g for g in inverse_classes if g[0] % 2)
    rotations = tuple(g for g in G.elements if g[1] == 0)
    dic3 = tuple(g for g in G.elements if (g[0] + g[1]) % 2 == 0)

    def draw_A(model):
        if model == 0:
            generators = ((1, 0), (3, 0), (5, 0))
        elif model == 1:
            generators = ((1, 1), (3, 1), (5, 1))
        elif model == 2:
            # Mixed odd classes normally generate the entire group.
            while True:
                generators = tuple(local.sample(odd_classes, 3))
                if len(closure(generators)) == q:
                    break
        else:
            # Rank-rich smaller closures are allowed as hypotheses.
            # Their short words must pass the complete logical test.
            generators = tuple(local.sample(inverse_classes, 3))
        return [[[G.e, g] for g in generators]], closure(generators)

    def draw_B(profile, pool, policy):
        B = [[[] for _ in range(7)] for _ in range(2)]
        nonidentity = [g for g in pool if g != G.e]
        gaps = [g for g in nonidentity if G.mul(g, g) != G.e]
        if not gaps:
            gaps = nonidentity

        cross = [j for j, c in enumerate(profile) if c == (1, 1)]
        voltages = local.sample(pool, len(cross))
        for j, voltage in zip(cross, voltages):
            B[0][j], B[1][j] = [G.e], [voltage]

        used = [set(), set()]
        for j, counts in enumerate(profile):
            if j in cross:
                continue
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                if count == 1:
                    B[i][j] = [local.choice(pool)]
                elif count == 2:
                    options = [g for g in gaps
                               if min(g, G.inv(g)) not in used[i]]
                    gap = local.choice(options or gaps)
                    used[i].add(min(gap, G.inv(gap)))
                    first = G.e if policy % 2 == 0 else local.choice(pool)
                    B[i][j] = [first, G.mul(first, gap)]
                else:
                    # Complete trinomial support, not a binomial gauge.
                    B[i][j] = local.sample(pool, count)
        return B

    # Only exclusions: never emit an anchor or its invariant image.
    root_A = [[[G.e, (d, 0)] for d in (1, 3, 5)]]
    references = (
        ((), (), (0, 1, 2, 3, 6, 7, 9)),
        ((1,), (5,), (0, 2, 4, 6, 8)),
        ((1, 3), (3, 5), (0, 2, 6)),
        ((1, 3, 5), (1, 3, 5), (0,)),
        ((1, 3), (2, 3), (0, 4, 8)),
    )
    archive_keys = set()
    for top, bottom, voltages in references:
        B = [[], []]
        for d in top:
            B[0].append([G.e, (d, 0)])
            B[1].append([])
        for d in bottom:
            B[0].append([])
            B[1].append([G.e, (d, 0)])
        for d in voltages:
            B[0].append([G.e])
            B[1].append([(d, 0)])
        hx, hz, n = lifted_matrices(G, root_A, B)
        archive_keys.add(fingerprint(hx, hz, n))

    best, best_quality = None, None
    fillings, graph_keys = set(), set()
    owned_survivors = 0
    for attempt in range(1536):
        epoch = attempt // 32
        # Initial search honors the twenty prescriptions. Later epochs
        # broaden supports and A families without emitting orbit copies.
        model = amodel if attempt < 512 else (amodel + epoch) % 4
        p = bmodel if bmodel < 4 else 4 + epoch % 4
        if attempt >= 768:
            p = (p + epoch) % len(profiles)
        profile = profiles[p]
        A, subgroup = draw_A(model)

        # Compare common-subgroup, differing-subgroup and full-group
        # couplings; relative label structure, not gauges, changes.
        policies = (subgroup, rotations, dic3, tuple(G.elements))
        pool = policies[epoch % 4]
        if len(pool) < 7:
            pool = tuple(G.elements)
        B = draw_B(profile, pool, epoch)
        filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                        for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) <= min(9, W_MAX)
        assert all(len(set(cell)) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 552 and 425 * n <= 1000 * k < 475 * n):
            continue

        key = fingerprint(hx, hz, n)
        if key in archive_keys or key in graph_keys:
            continue
        graph_keys.add(key)
        digest = hashlib.blake2b(
            repr(key).encode("ascii"), digest_size=16).digest()
        # An isomorphism preserves this invariant and therefore its
        # owner. Different slots cannot submit the same Tanner graph.
        if int.from_bytes(digest, "little") % N_SLOTS != idx:
            continue

        xf = logical_floor(hz, hx, n)
        if xf[0] < 4:
            continue
        zf = logical_floor(hx, hz, n)
        if zf[0] < 4:
            continue
        floor = min(xf[0], zf[0])
        quality = (k * floor * floor, floor, k, -(xf[1] + zf[1]))
        owned_survivors += 1
        if best_quality is None or quality > best_quality:
            ar, br = _extract_components(G, hx, hz)
            best_quality = quality
            best = {
                "A": A, "B": B, "screen_k": k,
                "design_card": (card, model, p, epoch % 4),
                "logical_exclusion_through": floor - 1,
                "logical_collision_diagnostics": (xf, zf),
                "component_diagnostics": (
                    _kernel_profile(ar, 3 * q),
                    _kernel_profile(br, 7 * q)),
                "rank_deficits": (len(hx) - rx, len(hz) - rz),
                "invariant_owner": idx,
            }
        if floor == 5 and k >= 258:
            break
        if owned_survivors >= 12 and attempt >= 768:
            break

    if best is None:
        raise ValueError("no novel owned filling passed logical-quotient screens")
    hx, hz, n = lifted_matrices(G, best["A"], best["B"])
    assert css_ok(hx, hz), "verified lift failed CSS"
    assert lifted_k(G, best["A"], best["B"]) == best["screen_k"]
    assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
    return best


def _singleton_columns(rows, width):
    """Column syndromes of a supplied verified matrix."""
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _singleton_component_bank(G):
    """Exhaustive structural catalogue, not independent slot draws.

    Translation normalizes the matching voltages. Multipliers are NOT
    independently quotiented on both constituents: their relative
    scaling can change the complete LP graph.
    """
    from itertools import combinations

    q, e = G.q, G.e
    assert q == 11

    def x(a):
        return G.elements[a % q]

    def translated(values):
        return min(tuple(sorted((v - a) % q for v in values))
                   for a in values)

    def build(kind, gap, values):
        if kind == 0:
            # One singleton plus six cross-row matching columns.
            # Row totals 7,6; all matching voltages are distinct.
            C = [[[e]], [[]]]
        else:
            # One singleton, one bottom internal cycle, five matchings.
            # Row totals 6,7. Independence of values excludes triangles
            # involving the internal cycle.
            C = [[[e], []], [[], [e, x(gap)]]]
        for value in values:
            C[0].append([e])
            C[1].append([x(value)])
        assert all(len(row) == 7 for row in C)
        return C

    cards = set()
    for values in combinations(range(q), 6):
        cards.add((0, 0, translated(values)))

    for gap in range(1, (q + 1) // 2):
        for values in combinations(range(q), 5):
            occupied = set(values)
            if any((v + gap) % q in occupied for v in values):
                continue
            cards.add((1, gap, translated(values)))

    dummy = [[[]]]
    mask = (1 << (7 * q)) - 1
    bank = []
    for card in sorted(cards):
        C = build(*card)
        left, _, _ = lifted_matrices(G, C, dummy)
        _, right, _ = lifted_matrices(G, dummy, C)
        left = [row & mask for row in left]
        right = [row & mask for row in right]
        if gf2_rank(left) != 2 * q or gf2_rank(right) != 2 * q:
            continue
        qa = _kernel_profile(left, 7 * q)
        qb = _kernel_profile(right, 7 * q)
        if min(qa[0], qb[0]) < 4:
            continue
        # These are exact truncated kernel diagnostics, not GPU readings.
        penalty = max(qa[1], qb[1])
        frozen = tuple(tuple(tuple(cell) for cell in row) for row in C)
        bank.append((penalty, card, frozen, (qa, qb)))
    bank.sort(key=lambda record: (record[0], record[1]))
    if len(bank) < 5:
        raise ValueError("too few full-rank singleton constituents")
    return tuple(bank)


def _singleton_graph_key(hx, hz, n):
    """Intrinsic invariant of the FULL uncoloured Tanner graph.

    Root every minimum-degree vertex, not a chosen LP coordinate.
    Degree and rooted colour-refinement histories survive arbitrary
    graph relabeling. Unequal keys certify inequivalence; equal keys
    are conservatively rejected without claiming an isomorphism test.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    m = len(rows)
    adjacency = [[] for _ in range(m + n)]
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            j = bit.bit_length() - 1
            adjacency[r].append(m + j)
            adjacency[m + j].append(r)
            row ^= bit

    degrees = [len(neighbours) for neighbours in adjacency]
    minimum = min(degrees)
    roots = [v for v, degree in enumerate(degrees) if degree == minimum]

    def compress(features):
        palette = {feature: colour
                   for colour, feature in enumerate(sorted(set(features)))}
        return [palette[feature] for feature in features]

    rooted = []
    for root in roots:
        colours = compress([(degree, int(v == root))
                            for v, degree in enumerate(degrees)])
        history = []
        for _ in range(5):
            features = [
                (colours[v],
                 tuple(sorted(colours[u] for u in neighbours)))
                for v, neighbours in enumerate(adjacency)
            ]
            history.append(tuple(sorted(Counter(features).items())))
            colours = compress(features)
        rooted.append(sha256(repr(tuple(history)).encode("ascii")).digest())

    return (
        n, tuple(sorted(Counter(degrees).items())),
        tuple(sorted(rooted)), _tanner_signature(hx, hz),
    )


from functools import lru_cache


@lru_cache(maxsize=1)
def _singleton_portfolio(G):
    """Pure immutable portfolio of optimized, invariant-separated pairs.

    Rebuilding with the same group and seed gives the same output.
    Memoization exposes no mutable cross-slot selection state.
    No archived filling is used as a construction seed.
    """
    q = G.q
    assert q == 11 and A_SHAPE == B_SHAPE == (2, 7)
    cap = min(9, W_MAX)
    bank = _singleton_component_bank(G)
    plans = []

    for ai, a in enumerate(bank):
        for bi in range(ai, len(bank)):
            b = bank[bi]
            priority = (
                max(a[0], b[0]), a[0] + b[0],
                int(a[1][0] == b[1][0]), ai, bi,
            )
            plans.append((priority, ai, bi))
    plans.sort(key=lambda record: record[0])

    entries, graph_keys = [], set()
    for _, ai, bi in plans:
        a, b = bank[ai], bank[bi]
        A, B = a[2], b[2]
        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if not _no_dead_lines(sa, sb):
            raise ValueError("singleton card has a dead protograph line")
        wx = max(sum(row) + sum(sb[i][j] for i in range(2))
                 for row in sa for j in range(7))
        wz = max(sum(row) + sum(sa[i][j] for i in range(2))
                 for row in sb for j in range(7))
        if max(wx, wz) > cap:
            raise ValueError("singleton card exceeds a CSS weight budget")
        if any(len(set(cell)) != len(cell) or len(cell) > CELL_MAX
               for grid in (A, B) for row in grid for cell in row):
            raise ValueError("singleton card has an invalid cell")

        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        # Full ranks, rather than a predicted augmentation gain, are
        # essential: even modest deficits can leave this narrow band.
        if not (n == 583 and k == 275
                and rx == len(hx) and rz == len(hz)
                and 425 * n <= 1000 * k < 475 * n):
            continue
        if max(row.bit_count() for row in hx + hz) > cap:
            continue

        # Exact complete-matrix tests include BOTH LP sectors.
        # Excluding kernel words also excludes logical words; stabilizer
        # screening is unnecessary for this conservative floor-four gate.
        if not (_no_kernel_through_three(hx, n)
                and _no_kernel_through_three(hz, n)):
            continue

        key = _singleton_graph_key(hx, hz, n)
        if key in graph_keys:
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified singleton lift failed CSS")
        if lifted_k(G, A, B) != k:
            raise ValueError("verified singleton rank helpers disagree")
        graph_keys.add(key)

        screen = (
            ("k", k),
            ("rank_gain_above_bound", 0),
            ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
            ("component_cards", (a[1], b[1])),
            ("component_diagnostics", (a[3], b[3])),
            ("full_kernel_weights_1_to_3_excluded", True),
            ("CSS_weight_maxima", (wx, wz)),
        )
        entries.append((A, B, screen))
        if len(entries) == N_SLOTS:
            break

    # Missing positions are never filled by recycling an accepted code.
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of one complete optimized new-shape design."""
    portfolio = _singleton_portfolio(G)
    if idx >= len(portfolio):
        raise ValueError("distinct singleton full-rank portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh complete cell from the immutable selected design."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supplied twenty inequivalent initial designs on C12:C2(s=5); slots 0..19 collapsed to four distinct codes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: The common-C12 catalogue supplies twenty inequivalent k=258 designs: sixty slots retained only five downstream classes, exhausted after slot 51, and all five measured codes still had d30s=4.
# REFUTED: Stabilizer-aware quotient screening and mixed-degree B redistributions escape the order-24 (1,3)x(2,7) distance-four plateau: all five current measured leaders had d30s=4, and sixty slots retained only 21 distinct codes.
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
