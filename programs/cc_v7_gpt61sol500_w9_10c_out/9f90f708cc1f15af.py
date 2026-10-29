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
CONCEPT = ("S3 direct product C8, order 48, A=(1,2), B=(2,5): independently optimized cubic A and reinforced-matching B constituents should supply five novel distance-nine finalists in cell 0.250, improving its measured frontier fitness 20.9252; longer searches can refute this.")
GROUP = ("Jump from the parent's order-24 C12 semidirect C2 to S3 x C8 of order 48. This direct product has order-eight, order-twelve and order-twenty-four relative labels. The related but different C3 semidirect C16 bed measured five distance-nine finalists in cell 0.250. Its readings motivate testing a different extension structure, not transferring distance or archive novelty.")
SHAPES = ("A=(1,2), B=(2,5) give n=576, kb=144 and guaranteed rate 0.250. The size window permits q<=49; this group fixes q=48. Constituents have 96 variables/48 checks and 240 variables/96 checks. Parent minima occupied sector-1 Acol0..2/Bcol0..6 broadly, especially Acol1 and Bcol5. Cubic A columns replace that binomial incidence mechanism. Verified full ranks enforce [0.225,0.275).")
FILLING = ("Twenty triangle/support prescriptions combine ten unordered pairs of four relative-triangle policies with two saturated B supports. A has two trinomials and six terms; each B row has six terms and each column at most three. Thus BOTH CSS sums are nine, below either stated cap. B's three-matching core is selected by exact girth computed from verified lifts. Independent constituent pools receive kernel-basis refinement before pairing. Complete CSS logical exclusion through four, actual ranks and stabilizer descent of explicit embedded witnesses gate publication. Different intrinsic uncoloured Tanner invariants certify slot inequivalence; equality is rejected conservatively. Larger distance and novelty against unseen archive entries remain hypotheses.")
FAMILY_NAME = "r025_s3xc8_q48_12x25_cubic_core_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 48

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("S3 x C8 has order 48")
    # (a,b,c)=x^a y^b z^c:
    # x^3=y^2=z^8=1, y*x*y^-1=x^-1, z central.
    els = [(a, b, c) for c in range(8)
           for b in range(2) for a in range(3)]

    def red(g):
        return (int(g[0]) % 3, int(g[1]) % 2, int(g[2]) % 8)

    def mul(g, h):
        a, b, c = red(g)
        d, e, f = red(h)
        return ((a + (-d if b else d)) % 3,
                (b + e) % 2, (c + f) % 8)

    def inv(g):
        a, b, c = red(g)
        return ((a if b else -a) % 3, b, (-c) % 8)

    def idx(g):
        a, b, c = red(g)
        return 6 * c + 3 * b + a

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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


def _unused_order24_context(rng, G, idx):
    """Historical order-24 policy; never called by the new group model."""
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


def _s3_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= 1 << r
            row ^= bit
    return columns


def _s3_span(rows):
    basis = {}
    for row in rows:
        while row:
            p = row.bit_length() - 1
            previous = basis.get(p)
            if previous is None:
                basis[p] = row
                break
            row ^= previous
    return basis


def _s3_reduce(word, basis):
    """Linear canonical remainder, including nonpivot coordinates."""
    remainder = 0
    while word:
        p = word.bit_length() - 1
        previous = basis.get(p)
        if previous is None:
            bit = 1 << p
            remainder |= bit
            word ^= bit
        else:
            word ^= previous
    return remainder


def _s3_logical4_clear(checks, stabilizers, width):
    """Exact exclusion of nontrivial logicals of weights one through four."""
    basis = _s3_span(stabilizers)
    quotient = [_s3_reduce(1 << i, basis) for i in range(width)]
    columns = _s3_columns(checks, width)
    representatives = {0: 0}

    for i, syndrome in enumerate(columns):
        value = quotient[i]
        previous = representatives.get(syndrome)
        if previous is None:
            representatives[syndrome] = value
        elif previous != value:
            return False

    for i, first in enumerate(columns):
        for j in range(i):
            syndrome = first ^ columns[j]
            value = quotient[i] ^ quotient[j]
            previous = representatives.get(syndrome)
            if previous is None:
                representatives[syndrome] = value
            elif previous != value:
                return False
    return True


def _s3_probe(rows, width, passes, rng, seed=0):
    """Heuristic kernel upper bound with an explicit verified witness."""
    columns = _s3_columns(rows, width)
    best = seed
    weight = seed.bit_count() if seed else width + 1

    def record(word):
        nonlocal best, weight
        candidate = word.bit_count()
        if word and candidate < weight:
            best, weight = word, candidate

    for _ in range(passes):
        order = list(range(width))
        rng.shuffle(order)
        pivots, kernel = {}, []
        for i in order:
            syndrome, word = columns[i], 1 << i
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
                record(word)

        rng.shuffle(kernel)
        for i, word in enumerate(kernel):
            current = word.bit_count()
            for j, other in enumerate(kernel):
                if i == j:
                    continue
                candidate = word ^ other
                new_weight = candidate.bit_count()
                if 0 < new_weight < current:
                    word, current = candidate, new_weight
            kernel[i] = word
            record(word)

        for i, first in enumerate(kernel):
            for second in kernel[:i]:
                record(first ^ second)
        short = sorted(kernel, key=int.bit_count)[:24]
        for i, first in enumerate(short):
            for j in range(i):
                pair = first ^ short[j]
                for h in range(j):
                    record(pair ^ short[h])
        for _ in range(2):
            if not best:
                break
            starting = best
            for i, first in enumerate(short):
                record(starting ^ first)
                for second in short[:i]:
                    record(starting ^ first ^ second)
            if best == starting:
                break
        if weight < 7:
            break

    if not best:
        raise ValueError("constituent kernel unexpectedly empty")
    assert all((row & best).bit_count() % 2 == 0 for row in rows)
    return weight, best


def _s3_component(G, grid, side):
    """Expose only the scaffold's actual left/right constituent action."""
    unit = [[[G.e]]]
    if side == "A":
        rows, _, _ = lifted_matrices(G, grid, unit)
        width, checks = 2 * G.q, G.q
    else:
        _, rows, _ = lifted_matrices(G, unit, grid)
        width, checks = 5 * G.q, 2 * G.q
    mask = (1 << width) - 1
    return [row & mask for row in rows[:checks]]


def _s3_girth(rows, width):
    """Exact girth of a verified degree-two incidence matrix."""
    columns = _s3_columns(rows, width)
    adjacency = [[] for _ in rows]
    for edge, syndrome in enumerate(columns):
        if syndrome.bit_count() != 2:
            raise ValueError("matching core is not degree two")
        bit = syndrome & -syndrome
        u = bit.bit_length() - 1
        v = (syndrome ^ bit).bit_length() - 1
        adjacency[u].append((v, edge))
        adjacency[v].append((u, edge))

    best = len(rows) + 1
    for root in range(len(rows)):
        distance = [-1] * len(rows)
        parent_edge = [-1] * len(rows)
        distance[root] = 0
        queue = [root]
        for u in queue:
            for v, edge in adjacency[u]:
                if distance[v] < 0:
                    distance[v] = distance[u] + 1
                    parent_edge[v] = edge
                    queue.append(v)
                elif edge != parent_edge[u] and edge != parent_edge[v]:
                    best = min(best, distance[u] + distance[v] + 1)
        if best < 8:
            break
    return best


def _s3_full_key(hx, hz, width):
    """Uncoloured Tanner invariant; equality is rejected conservatively."""
    from collections import Counter
    rows = hx + hz
    columns = _s3_columns(rows, width)
    profiles = []
    for syndrome in columns:
        weights = []
        while syndrome:
            bit = syndrome & -syndrome
            weights.append(rows[bit.bit_length() - 1].bit_count())
            syndrome ^= bit
        profiles.append(tuple(sorted(weights)))
    return (tuple(sorted(Counter(profiles).items())),
            _tanner_signature(hx, hz))


def _s3_quantum_witness(hx, hz, n, aword, bword, q):
    """Stabilizer descent of actual nontrivial embedded component witnesses.

    This supplies an upper bound only. Missing classes and local minima
    are not certified by the descent.
    """
    best = n + 1
    found = False
    for kind, checks, stabilizers in (
            ("X", hz, hx), ("Z", hx, hz)):
        basis = _s3_span(stabilizers)
        blocks = range(2) if kind == "X" else range(5)
        for block in blocks:
            if kind == "X":
                word = bword << (block * 5 * q)
            else:
                word, bits = 0, aword
                while bits:
                    bit = bits & -bits
                    j, h = divmod(bit.bit_length() - 1, q)
                    word |= 1 << ((j * 5 + block) * q + h)
                    bits ^= bit
            if any((row & word).bit_count() % 2 for row in checks):
                raise ValueError("verified witness embedding disagrees")
            if not _s3_reduce(word, basis):
                continue
            found = True
            weight = word.bit_count()
            for _ in range(2):
                starting = word
                for i, first in enumerate(stabilizers):
                    trial = starting ^ first
                    candidate = trial.bit_count()
                    if candidate < weight:
                        word, weight = trial, candidate
                    for second in stabilizers[:i]:
                        trial = starting ^ first ^ second
                        candidate = trial.bit_count()
                        if candidate < weight:
                            word, weight = trial, candidate
                if word == starting:
                    break
            assert word and _s3_reduce(word, basis)
            assert all((row & word).bit_count() % 2 == 0 for row in checks)
            best = min(best, weight)
    return best if found else None


from functools import lru_cache


@lru_cache(maxsize=1)
def _s3_portfolio(G, count):
    """Deterministic immutable portfolio, independent of slot call order."""
    from itertools import combinations_with_replacement

    local = random.Random(BASE_SEED ^ 0x53812509)
    if G.q != 48:
        raise ValueError("S3 x C8 portfolio requires order 48")
    cap = min(9, W_MAX)
    if cap < 9:
        raise ValueError("saturated cubic profiles require cap nine")

    def element_order(g):
        value = G.e
        for t in range(1, G.q + 1):
            value = G.mul(value, g)
            if value == G.e:
                return t
        raise ValueError("group element has no finite order")

    orders = {g: element_order(g) for g in G.elements}
    triangles = [[] for _ in range(4)]
    for g in G.elements:
        if g == G.e:
            continue
        for h in G.elements:
            if h == G.e or h == g:
                continue
            spectrum = sorted((orders[g], orders[h],
                               orders[G.mul(G.inv(g), h)]))
            commute = G.mul(g, h) == G.mul(h, g)
            if not commute and spectrum[0] >= 6:
                triangles[0].append((g, h))
            if commute and spectrum[0] >= 8:
                triangles[1].append((g, h))
            if not commute and spectrum[0] >= 4:
                triangles[2].append((g, h))
            if spectrum[0] >= 4 and spectrum[-1] >= 12:
                triangles[3].append((g, h))
    if any(not bank for bank in triangles):
        raise ValueError("one prescribed relative-triangle class is empty")
    long_steps = [g for g in G.elements if orders[g] >= 8]

    # Construct exact-girth cores without a hand-written lift convention.
    steps = [g for g in G.elements if orders[g] >= 6]
    firsts = [
        g for g in steps
        if G.idx(g) == min(
            G.idx(G.mul(G.mul(a, g), G.inv(a)))
            for a in G.elements)
    ]
    cores = []
    unit = [[[G.e]]]
    for g in firsts:
        for h in steps:
            if g == h or orders[G.mul(G.inv(g), h)] < 6:
                continue
            core = [[[G.e], [G.e], [G.e]],
                    [[G.e], [g], [h]]]
            _, rows, _ = lifted_matrices(G, unit, core)
            mask = (1 << (3 * G.q)) - 1
            rows = [row & mask for row in rows[:2 * G.q]]
            girth = _s3_girth(rows, 3 * G.q)
            if girth >= 8:
                cores.append((girth, g, h))
    if not cores:
        raise ValueError("no exact-girth-eight matching core")
    maximum_girth = max(record[0] for record in cores)
    cores = [record for record in cores if record[0] == maximum_girth]

    triangle_pairs = tuple(combinations_with_replacement(range(4), 2))
    profiles = (
        ((2, 1, 1, 1, 1), (1, 2, 1, 1, 1)),
        ((3, 1, 1, 1, 0), (0, 1, 1, 1, 3)),
    )

    def translated_triangle(policy):
        g, h = local.choice(triangles[policy])
        first = G.pick(local)
        cell = [first, G.mul(first, g), G.mul(first, h)]
        assert len(set(cell)) == 3
        return cell

    def draw(side, attempt, profile_index=0):
        pair_index = attempt % len(triangle_pairs)
        policies = triangle_pairs[pair_index]
        if side == "A":
            return [[translated_triangle(p) for p in policies]], pair_index
        profile = profiles[profile_index]
        grid = [[[] for _ in range(5)] for _ in range(2)]
        for i in range(2):
            for j, terms in enumerate(profile[i]):
                if terms == 1:
                    grid[i][j] = [G.pick(local)]
                elif terms == 2:
                    first = G.pick(local)
                    grid[i][j] = [first, G.mul(
                        first, local.choice(long_steps))]
                elif terms == 3:
                    grid[i][j] = translated_triangle(policies[i])
        _, g, h = local.choice(cores)
        matching = [j for j in range(5)
                    if profile[0][j] == profile[1][j] == 1]
        assert len(matching) == 3
        for j, voltage in zip(matching, (G.e, g, h)):
            grid[0][j], grid[1][j] = [G.e], [voltage]
        return grid, (profile_index, pair_index)

    def component_bank(side, profile_index=0):
        width = (2 if side == "A" else 5) * G.q
        pool, fillings = [], set()
        for attempt in range(640):
            grid, card = draw(side, attempt, profile_index)
            filling = tuple(tuple(sorted(G.idx(g) for g in cell))
                            for row in grid for cell in row)
            if filling in fillings:
                continue
            fillings.add(filling)
            rows = _s3_component(G, grid, side)
            floor, _ = _kernel_profile(rows, width)
            if floor < 5:
                continue
            cycles = sum(
                overlap * (overlap - 1) // 2
                for i, row in enumerate(rows)
                for overlap in [(row & other).bit_count()
                                for other in rows[:i]])
            pool.append((cycles, attempt, grid, rows, card))
        pool.sort(key=lambda record: record[:2])
        if not pool:
            raise ValueError("no kernel-four-clean constituent proposals")

        refined = []
        for cycles, attempt, grid, rows, card in pool[:64]:
            distance, witness = _s3_probe(rows, width, 6, local)
            refined.append((distance, -cycles, -attempt,
                            grid, rows, card, witness))
        refined.sort(key=lambda record: record[:3], reverse=True)
        finalists = []
        for distance, nc, na, grid, rows, card, witness in refined[:28]:
            distance, witness = _s3_probe(
                rows, width, 18, local, witness)
            finalists.append((distance, nc, na,
                              grid, card, witness))
        finalists.sort(key=lambda record: record[:3], reverse=True)
        return finalists[:24]

    aa = component_bank("A")
    bb = component_bank("B", 0) + component_bank("B", 1)

    # Twenty initial constructive prescriptions: ten triangle pairs
    # crossed with the two B supports. Final slots receive optimized,
    # intrinsic-invariant-distinct matrices rather than raw draws.
    plans = []
    for ai, a in enumerate(aa):
        for bi, b in enumerate(bb):
            reading = min(a[0], b[0])
            plans.append((-reading, -(a[0] + b[0]),
                          -(a[1] + b[1]), ai, bi))
    plans.sort()

    candidates = []
    for _, _, _, ai, bi in plans:
        a, b = aa[ai], bb[bi]
        A, B = a[3], b[3]
        sa = [[len(c) for c in row] for row in A]
        sb = [[len(c) for c in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert _grid_weights(sa, sb) == 9
        assert all(len(set(c)) == len(c) <= 5
                   for grid in (A, B) for row in grid for c in row)
        hx, hz, n = lifted_matrices(G, A, B)
        rx, rz = gf2_rank(hx), gf2_rank(hz)
        k = n - rx - rz
        if not (n == 576 and k > 0
                and 225 * n <= 1000 * k < 275 * n):
            continue
        reading = min(a[0], b[0])
        priority = (k * reading * reading / float(n), reading,
                    a[0] + b[0], a[1] + b[1])
        candidates.append((priority, ai, bi, hx, hz, k, rx, rz))
    candidates.sort(key=lambda record: record[0], reverse=True)

    entries, graph_keys = [], set()
    wanted = max(count + 16, 2 * count)

    def freeze(grid):
        return tuple(tuple(tuple(c) for c in row) for row in grid)

    for priority, ai, bi, hx, hz, k, rx, rz in candidates:
        key = _s3_full_key(hx, hz, 576)
        if key in graph_keys:
            continue
        if not (_s3_logical4_clear(hz, hx, 576)
                and _s3_logical4_clear(hx, hz, 576)):
            continue
        a, b = aa[ai], bb[bi]
        reading = _s3_quantum_witness(
            hx, hz, 576, a[5], b[5], G.q)
        effective = min(a[0], b[0],
                        reading if reading is not None else 576)
        if effective < 7:
            continue
        A, B = a[3], b[3]
        assert css_ok(hx, hz)
        assert lifted_k(G, A, B) == k
        assert max(row.bit_count() for row in hx + hz) <= cap
        graph_keys.add(key)
        quality = (k * effective * effective / 576.0,
                   effective, k, priority)
        screen = (
            ("k", k),
            ("design_cards", (a[4], b[4])),
            ("exact_matching_core_girth", maximum_girth),
            ("component_witness_readings", (a[0], b[0])),
            ("quantum_witness_reading", reading),
            ("selection_reading", effective),
            ("exact_logical_exclusion_through", 4),
            ("rank_deficits", (len(hx) - rx, len(hz) - rz)),
        )
        entries.append((quality, freeze(A), freeze(B), screen))
        if len(entries) >= wanted:
            break

    entries.sort(key=lambda record: record[0], reverse=True)
    return tuple((A, B, screen)
                 for _, A, B, screen in entries[:count])


def NEW_CTX(rng, G, idx):
    """Pure lookup; memoization changes cost, not the selected matrices."""
    portfolio = _s3_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("invariant-distinct target-rate portfolio exhausted")
    A, B, screen = portfolio[idx]
    return {"A": A, "B": B, "screen": dict(screen)}


def CELL(rng, G, side, i, j, ctx):
    """Return the selected complete cell without sampling or mutation."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: A five-valued 2-second reading always survives measurement on the Frobenius (1,3)x(1,3) catalogue; slot 30 fell from d2s=5 to d30s=4 and exposed mixed-sector X and Z witnesses.
# REFUTED: Rank-first odd-incidence cards supplied twenty inequivalent initial designs on C12:C2(s=5); slots 0..19 collapsed to four distinct codes, and all five measured leaders retained sector-1 weight-four logicals at k=254.
# REFUTED: The common-C12 catalogue supplies twenty inequivalent k=258 designs: sixty slots retained only five downstream classes, exhausted after slot 51, and all five measured codes still had d30s=4.
# REFUTED: Multiple quotient families and mixed-degree B supports escape the order-24 parent's distance-four plateau: all 21 distinct current codes had d2s=4, and all five measured finalists retained d30s=4 with abundant sector-1 witnesses.
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
