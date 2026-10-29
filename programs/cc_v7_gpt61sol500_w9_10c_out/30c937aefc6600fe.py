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
CONCEPT = ("Metacyclic C11 semidirect_1 C1, order 11, A=B=(2,7): exhaustive triangle-free constituent prescriptions, exact complete-matrix distance-four certificates and intrinsic Tanner separation should supply five fresh [583,287,4] codes, scoring 7.877 in cell 0.500.")
GROUP = ("L2 order adjustment within the parent's metacyclic semidirect-product construction rule: factor 53 excludes q=55 and permits q<=11, so use (m,r,s)=(11,1,1). The cyclic order-eleven bed already measured distance-four codes near fitness 7.8 in cell 0.500. That evidence motivates a novelty-focused portfolio, not a claim of distance above four.")
SHAPES = ("Equal (2,7) constituents have 77 variables and 22 checks. With q=11, n=583 and kb=275, the guaranteed rate is 25/53=0.471698. Exact full ranks must give k=287, rate 0.492281, inside [0.475,0.525). Parent weight-five minima broadly occupied sector-1 Acol/Bcol blocks; an earlier weight-four failure also touched Arow0/Brow0. Both complete logical types therefore receive exact screening.")
FILLING = ("Enumerate zero-, two- and four-ring degree-two constituent prescriptions modulo cyclic translations, automorphisms and row swaps. Every component row has seven terms and every column two, so BOTH CSS weight maxima are nine, also below twelve. Verified constituent lifts must be connected and exclude kernel weights 1..3. Complete lifts must exclude all kernel weights 1..3 and contain a verified nontrivial weight-four logical, certifying distance four. Distinct uncoloured Tanner invariants, enriched by high-degree contraction walks and low-qubit neighbourhoods, allocate different slots. Shown cyclic archive classes are conservatively excluded; the evaluator remains authoritative for the unshown archive.")
FAMILY_NAME = "r050_m11r1s1_27x27_exhaustive_certified4_walkportfolio"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 11

def custom_ops(q):
    # Same metacyclic rule: x^m=y^r=1, y*x*y^-1=x^s.
    # The new shape requires q<=11; s^r=1 modulo m.
    m, r, s = 11, 1, 1
    if not valid_q(q) or m * r != q:
        raise ValueError("this metacyclic parameter set has order eleven")
    powers = [pow(s, b, m) for b in range(r)]
    els = [(a, b) for b in range(r) for a in range(m)]

    def red(g):
        return (g[0] % m, g[1] % r)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % m, (b + d) % r)

    def inv(g):
        a, b = red(g)
        bi = (-b) % r
        return ((-powers[bi] * a) % m, bi)

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


def _component_rows(G, side, grid):
    """Extract any rectangular constituent using the verified lift."""
    dummy = [[[G.e]]]
    width = len(grid[0]) * G.q
    checks = len(grid) * G.q
    mask = (1 << width) - 1
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, dummy)
        return [row & mask for row in hx[:checks]]
    _, hz, _ = lifted_matrices(G, dummy, grid)
    return [row & mask for row in hz[:checks]]


def _generator_key(G, generators):
    """Quotient known component orbit copies, not arbitrary graph isos.

    Affine-group automorphisms have x->x^u, y->x^v*y. Binomial
    generators can also be inverted and their columns permuted.
    """
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    keys = []
    for u in range(1, 11):
        for v in range(11):
            images = []
            for a, b in generators:
                g = ((u * a + v * sums[b]) % 11, b)
                images.append(min(G.idx(g), G.idx(G.inv(g))))
            keys.append(tuple(sorted(images)))
    return min(keys)


def _component_bank(G):
    """Deterministically optimize different quotient prescriptions."""
    local = random.Random(BASE_SEED ^ 0x550403)
    # Quotient-coordinate profiles distinguish construction prescriptions.
    # Avoid two rotations, which commute and immediately create a 4-word.
    profiles = ((0, 1, 2), (0, 1, 1), (1, 1, 2),
                (1, 2, 2), (0, 2, 2), (1, 2, 3))
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    entries, used = [], set()
    for attempt in range(256):
        profile = profiles[attempt % len(profiles)]
        generators = [local.choice(pools[b]) for b in profile]
        undirected = [min(G.idx(g), G.idx(G.inv(g)))
                      for g in generators]
        if len(set(undirected)) != 3:
            continue
        key = _generator_key(G, generators)
        if key in used:
            continue
        used.add(key)
        A = [[[G.e, g] for g in generators]]
        reading, collisions = _kernel_profile(
            _component_rows(G, "A", A), 3 * G.q)
        entries.append(((reading, -collisions), key, A))
    if not entries:
        raise ValueError("component bank unexpectedly empty")
    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    # Prefer every available clean component. If the hypothesis fails,
    # retain the best legal components instead of invalidating all slots.
    clean = [item for item in entries if item[0][0] == 5]
    return (clean if clean else entries[:12])[:16]


def _draw_B(rng, G, attempt, bank):
    """Indexed support/order prescriptions, subsequently screened."""
    if attempt % 4 == 0:
        source = bank[(attempt // 4) % len(bank)][2]
        # Independent samples of quotients, not copies of source labels.
        profile = [cell[1][1] for cell in source[0]]
        cells = []
        for b in profile:
            pool = [g for g in G.elements if g != G.e and g[1] == b]
            cells.append([G.e, rng.choice(pool)])
        return [cells]

    # One degree-three column and two degree-two columns: total seven.
    # Vary the trinomial's quotient-coordinate pattern deliberately.
    patterns = ((0, 1), (0, 2), (1, 1), (1, 2),
                (1, 3), (2, 2), (2, 3), (2, 4))
    b0, b1 = patterns[(attempt // 4) % len(patterns)]
    pools = [[g for g in G.elements
              if g != G.e and g[1] == b] for b in range(5)]
    g = rng.choice(pools[b0])
    h = rng.choice(pools[b1])
    while h == g:
        h = rng.choice(pools[b1])
    cells = [[G.e, g, h]]
    for j in range(2):
        b = (attempt // (8 + 3 * j) + j) % 5
        cells.append([G.e, rng.choice(pools[b])])
    return [cells]


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


def _legacy_order55_evenfirst_reference(rng, G, idx):
    """Inactive order-55 reference; never called by slot publication.

    The new NEW_CTX uses the order-eleven rectangular portfolio below.
    Historical one-row helpers are not used to select any new cells.
    """
    from itertools import combinations
    from collections import Counter

    assert G.q == 55
    local = random.Random(BASE_SEED ^ 0x5504E1)
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    representatives = sorted({
        min(j, inverse[j]) for j in range(G.q)
        if G._key(G.elements[j]) != G._key(G.e)
    })

    # Precompute the action on undirected generators. Enumerating all
    # 27-choose-3 triples avoids the parent's small sampled bank.
    sums = [sum(pow(3, t, 11) for t in range(b)) % 11
            for b in range(5)]
    actions = []
    for u in range(1, 11):
        for v in range(11):
            image = []
            for a, b in G.elements:
                j = G.idx(((u * a + v * sums[b]) % 11, b))
                image.append(min(j, inverse[j]))
            actions.append(image)

    orbit_keys = set()
    for triple in combinations(representatives, 3):
        key = min(tuple(sorted(action[j] for j in triple))
                  for action in actions)
        orbit_keys.add(key)

    bank = []
    for key in sorted(orbit_keys):
        grid = [[[G.e, G.elements[j]] for j in key]]
        rows = _component_rows(G, "A", grid)
        if gf2_rank(rows) != G.q - 1:
            continue
        if _kernel_profile(rows, 3 * G.q)[0] != 5:
            continue
        bank.append(grid)
    if not bank:
        raise ValueError("no connected binomial component passed kernel4")

    def walk_signature(rows):
        """An additional uncoloured Tanner invariant.

        The relation joins checks sharing exactly one qubit. Histograms
        of joint two- and three-walk counts retain more information than
        the parent's independent check-intersection profiles.
        """
        m = len(rows)
        adj = [0] * m
        for i, row in enumerate(rows):
            for j in range(i):
                if (row & rows[j]).bit_count() == 1:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i

        neighbours = []
        for bits in adj:
            vertices = []
            while bits:
                bit = bits & -bits
                vertices.append(bit.bit_length() - 1)
                bits ^= bit
            neighbours.append(vertices)

        square = [[(adj[i] & adj[j]).bit_count()
                   for j in range(m)] for i in range(m)]
        profiles = []
        for i in range(m):
            third = [0] * m
            for neighbour in neighbours[i]:
                for j, value in enumerate(square[neighbour]):
                    third[j] += value
            histogram = Counter(
                (((adj[i] >> j) & 1), square[i][j], third[j])
                for j in range(m))
            profiles.append((rows[i].bit_count(),
                             len(neighbours[i]),
                             tuple(sorted(histogram.items()))))
        return tuple(sorted(profiles))

    graph_buckets = {}
    filling_keys = set()
    accepted = 0

    def novel(hx, hz):
        """Conservatively merge equal invariants, not label fingerprints."""
        basic = _tanner_signature(hx, hz)
        rows = hx + hz
        bucket = graph_buckets.get(basic)
        if bucket is None:
            graph_buckets[basic] = [(rows, None)]
            return True

        extended = walk_signature(rows)
        for position, (old_rows, old_extended) in enumerate(bucket):
            if old_extended is None:
                old_extended = walk_signature(old_rows)
                bucket[position] = (old_rows, old_extended)
            if extended == old_extended:
                return False
        bucket.append((rows, extended))
        return True

    def consider(A, B, phase, required_k=None):
        nonlocal accepted
        filling = tuple(
            tuple(tuple(sorted(G.idx(g) for g in cell)) for cell in row)
            for grid in (A, B) for row in grid)
        if filling in filling_keys:
            return None
        filling_keys.add(filling)

        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        if (_grid_weights(sa, sb) > min(9, W_MAX)
                or not _no_dead_lines(sa, sb)):
            return None

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 550 or not (375 * n <= 1000 * k < 425 * n):
            return None
        if required_k is not None and k != required_k:
            return None

        # These are hard acceptance conditions, not diagnostics.
        # They include every mixed-sector support of weight <=4.
        if _kernel_profile(hx, n)[0] != 5:
            return None
        if _kernel_profile(hz, n)[0] != 5:
            return None
        if not novel(hx, hz):
            return None

        if accepted == idx:
            assert css_ok(hx, hz), "verified lift failed CSS"
            assert lifted_k(G, A, B) == k
            assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
            return {"A": A, "B": B, "screen_k": k,
                    "phase": phase, "centralizer_cutoff": 5,
                    "bank_size": len(bank)}
        accepted += 1
        return None

    # Inner transformations of A and B independently are gauge copies.
    # Only two outer classes remain: scaling x by 1 or by nonsquare 2.
    # Same-orbit untwisted pairs are omitted: they supply the obvious
    # three-sector-1/one-sector-2 weight-four matching mechanism.
    plans = [(a, b, twist)
             for a in range(len(bank))
             for b in range(a, len(bank))
             for twist in (0, 1)
             if a != b or twist == 1]
    plans.sort(key=lambda plan: (plan[0] == plan[1],
                                plan[2], plan[0], plan[1]))

    for a, b, twist in plans:
        A = bank[a]
        scale = 2 if twist else 1
        B = [[[(scale * g[0] % 11, g[1]) for g in cell]
              for cell in bank[b][0]]]
        answer = consider(A, B, "even-even", required_k=226)
        if answer is not None:
            return answer

    # A rotation-only trinomial has three order-eleven relative
    # quotients. The parent's trinomial patterns always included
    # an order-five relative quotient.
    pure_count = accepted
    patterns = ((1, 1), (1, 2), (2, 2), (1, 3), (2, 4))
    pools = {
        b: [g for g in G.elements if g != G.e and g[1] == b]
        for b in range(5)
    }
    tested_components = {}

    for attempt in range(max(4096, 128 * (idx + 1))):
        card = (accepted - pure_count) % 20
        b0, b1 = patterns[(card + attempt // 32) % len(patterns)]
        A = bank[(card // 5 + attempt // 7) % len(bank)]

        # Cards prescribe a relative rotation ratio; the selected
        # scale and the two order-five generators are jointly optimized.
        ratio = 2 + card % 9
        delta = local.randrange(1, 11)
        triple = [G.e, (delta, 0), (ratio * delta % 11, 0)]
        B = [[triple,
              [G.e, local.choice(pools[b0])],
              [G.e, local.choice(pools[b1])]]]

        component_key = tuple(
            tuple(sorted(G.idx(g) for g in cell)) for cell in B[0])
        clean = tested_components.get(component_key)
        if clean is None:
            rows = _component_rows(G, "B", B)
            clean = (_kernel_profile(rows, 3 * G.q)[0] == 5
                     and gf2_rank(rows) == G.q)
            tested_components[component_key] = clean
        if not clean:
            continue

        answer = consider(A, B, "rotation-trinomial", required_k=222)
        if answer is not None:
            return answer

    raise ValueError("fresh full-kernel4 catalogue exhausted")


from functools import lru_cache


def _q11_columns(rows, width):
    """Syndromes from scaffold-built matrices, never a separate lift."""
    cols = [0] * width
    for r, row in enumerate(rows):
        tag = 1 << r
        while row:
            bit = row & -row
            cols[bit.bit_length() - 1] |= tag
            row ^= bit
    return cols


def _q11_grid(G, spec):
    """Seven degree-two columns; seven terms in each constituent row."""
    upper, lower, labels = spec
    element = {(a, b): g for g in G.elements for a, b in [g]}
    x = lambda t: element[(t % G.q, 0)]
    return [
        [[G.e, x(d)] for d in upper]
        + [[G.e] for _ in labels] + [[] for _ in lower],
        [[] for _ in upper]
        + [[x(t)] for t in labels]
        + [[G.e, x(d)] for d in lower],
    ]


def _q11_spec_key(spec, q):
    """Constituent proposal pruning, not a full-code isomorphism test."""
    upper, lower, labels = spec

    def translate(values):
        return min(tuple(sorted((v - anchor) % q for v in values))
                   for anchor in values)

    keys = []
    for u in range(1, q):
        a = tuple(sorted(min(u*d % q, -u*d % q) for d in upper))
        b = tuple(sorted(min(u*d % q, -u*d % q) for d in lower))
        values = tuple(u*t % q for t in labels)
        keys.append((a, b, translate(values)))
        keys.append((b, a, translate(tuple(-t % q for t in values))))
    return min(keys)


def _q11_four_certificate(hx, hz, n):
    """Exclude ALL kernels <=3 and verify a nontrivial four-logical.

    Pair/single syndrome collisions cover every support of size <=3.
    Pair/pair collisions give four-supports once columns are distinct.
    Nontriviality is checked against the actual stabilizer row space.
    """
    witness = None
    for kind, checks, own in (("X", hz, hx), ("Z", hx, hz)):
        cols = _q11_columns(checks, n)
        singles = set(cols)
        if 0 in singles or len(singles) != n:
            return None

        basis = {}
        for word in own:
            while word:
                p = word.bit_length() - 1
                if p not in basis:
                    basis[p] = word
                    break
                word ^= basis[p]

        def outside(word):
            while word:
                p = word.bit_length() - 1
                old = basis.get(p)
                if old is None:
                    return True
                word ^= old
            return False

        pairs = {}
        for j, first in enumerate(cols):
            for i in range(j):
                syndrome = first ^ cols[i]
                if syndrome in singles:
                    return None
                word = (1 << i) | (1 << j)
                previous = pairs.get(syndrome)
                if previous is None:
                    pairs[syndrome] = word
                elif witness is None:
                    logical = word ^ previous
                    if logical.bit_count() == 4 and outside(logical):
                        assert all((row & logical).bit_count() % 2 == 0
                                   for row in checks)
                        witness = (kind, logical)
    return witness


def _q11_tanner_key(hx, hz, n):
    """Intrinsic uncoloured full-Tanner invariant.

    Published checks have degree 9; qubits have degree 4 or 14.
    Hence all three vertex classes are intrinsically distinguishable.
    Contracting checks onto their two degree-14 neighbours produces a
    weighted graph. Its walk profiles and each degree-4 qubit's complete
    contracted neighbourhood are invariant under arbitrary relabeling.
    Equality is conservatively merged; inequality certifies separation.
    """
    from collections import Counter
    from hashlib import sha256

    rows = list(hx) + list(hz)
    cols = _q11_columns(rows, n)
    high = [j for j, col in enumerate(cols) if col.bit_count() == 14]
    if len(high) != 44:
        raise ValueError("unexpected high-degree sector in contraction")
    position = {j: i for i, j in enumerate(high)}
    high_mask = sum(1 << j for j in high)
    size = len(high)
    adjacency = [[0] * size for _ in high]
    endpoints = []

    for row in rows:
        bits = row & high_mask
        if row.bit_count() != 9 or bits.bit_count() != 2:
            raise ValueError("degree-two contraction assumption failed")
        first = bits & -bits
        u = position[first.bit_length() - 1]
        v = position[(bits ^ first).bit_length() - 1]
        endpoints.append((u, v))
        adjacency[u][v] += 1
        adjacency[v][u] += 1

    neighbours = [
        [(v, value) for v, value in enumerate(row) if value]
        for row in adjacency
    ]
    walks = [[[] for _ in high] for _ in high]
    for root in range(size):
        vector = [0] * size
        vector[root] = 1
        for length in range(1, 11):
            following = [0] * size
            for u, value in enumerate(vector):
                if value:
                    for v, multiplicity in neighbours[u]:
                        following[v] += value * multiplicity
            vector = following
            for v, value in enumerate(vector):
                walks[root][v].append(value)

    colours = [tuple(walks[u][u]) for u in range(size)]
    rooted = tuple(sorted(
        (colours[u], tuple(sorted(
            (colours[v], tuple(walks[u][v])) for v in range(size))))
        for u in range(size)))

    low_profiles = Counter()
    for col in cols:
        if col.bit_count() != 4:
            continue
        incidence = Counter()
        remaining = col
        while remaining:
            bit = remaining & -remaining
            u, v = endpoints[bit.bit_length() - 1]
            incidence[u] += 1
            incidence[v] += 1
            remaining ^= bit
        profile = tuple(sorted(
            (incidence[u], colours[u], tuple(sorted(
                (incidence[v], colours[v], tuple(walks[u][v]))
                for v in incidence)))
            for u in incidence))
        low_profiles[profile] += 1

    signature = (
        _tanner_signature(hx, hz), rooted,
        tuple(sorted(low_profiles.items())),
        tuple(sorted(Counter(c.bit_count() for c in cols).items())),
    )
    return sha256(repr(signature).encode("ascii")).digest()


@lru_cache(maxsize=1)
def _q11_portfolio(G, count):
    """Deterministic exhaustive portfolio with immutable published cells.

    Memoization only avoids rebuilding the same pure result for each
    slot. Different slots receive different intrinsic Tanner invariants.
    """
    from itertools import combinations

    q = G.q
    if q != 11:
        raise ValueError("rectangular portfolio requires order eleven")

    def signed(steps):
        return set(steps) | {(-d) % q for d in steps}

    def triangle_free_steps(steps):
        values = signed(steps)
        return not any((a + b) % q in values
                       for a in values for b in values)

    specs = set()
    for p in range(3):
        options = [
            steps for steps in combinations(range(1, 6), p)
            if triangle_free_steps(steps)
        ]
        matching_count = 7 - 2*p
        for upper in options:
            for lower in options:
                forbidden = signed(upper) | signed(lower)
                for tail in combinations(range(1, q), matching_count - 1):
                    labels = (0,) + tail
                    if any((a - b) % q in forbidden
                           for a, b in combinations(labels, 2)):
                        continue
                    specs.add(_q11_spec_key((upper, lower, labels), q))

    bank = []
    for spec in sorted(specs):
        grid = _q11_grid(G, spec)
        assert len(grid) == 2 and all(len(row) == 7 for row in grid)
        left = _component_rows(G, "A", grid)
        right = _component_rows(G, "B", grid)
        if gf2_rank(left) != 21 or gf2_rank(right) != 21:
            continue
        lf, lc = _kernel_profile(left, 77)
        rf, rc = _kernel_profile(right, 77)
        if min(lf, rf) < 4:
            continue
        bank.append((spec, grid, max(lc, rc), lc + rc))

    if len(bank) < 3:
        raise ValueError("too few connected triangle-free prescriptions")

    # These are exclusion references, never emitted candidate fillings.
    # They cover the shown cyclic ancestor and frontier examples, not
    # the unavailable complete evaluator archive.
    def matching(values):
        return _q11_grid(G, ((), (), tuple(values)))

    def ring(a, b, values):
        return _q11_grid(G, ((a,), (b,), tuple(values)))

    two = _q11_grid(G, ((1, 3), (1, 3), (0, 2, 4)))
    other_two = _q11_grid(G, ((1, 3), (1, 3), (0, 2, 6)))
    ring_one = ring(1, 1, (0, 2, 4, 6, 8))
    known_pairs = (
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
         _q11_grid(G, ((1, 4), (1, 4), (0, 2, 5)))),
        (two, _q11_grid(G, ((2, 5), (2, 5), (0, 3, 7)))),
        (two, _q11_grid(G, ((2, 3), (2, 3), (0, 1, 5)))),
    )
    excluded = set()
    for A, B in known_pairs:
        hx, hz, n = lifted_matrices(G, A, B)
        excluded.add(_q11_tanner_key(hx, hz, n))

    # Intrinsic support prescriptions precede relative label orientation.
    # Each orientation is only a proposal: full Tanner invariants decide
    # whether it is a genuinely different design.
    plans = []
    for a, aa in enumerate(bank):
        for b in range(a, len(bank)):
            bb = bank[b]
            pa, pb = len(aa[0][0]), len(bb[0][0])
            for scale in range(1, q):
                plans.append((
                    int(pa == pb), -(pa + pb),
                    max(aa[2], bb[2]), aa[3] + bb[3],
                    a, b, scale))
    plans.sort()

    entries, fillings = [], set()
    for _, _, _, _, a, b, scale in plans:
        A = bank[a][1]
        B = [[[
            ((scale * g[0]) % q, g[1]) for g in cell
        ] for cell in row] for row in bank[b][1]]
        filling = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for grid in (A, B) for row in grid for cell in row)
        if filling in fillings:
            continue
        fillings.add(filling)

        sa = [[len(cell) for cell in row] for row in A]
        sb = [[len(cell) for cell in row] for row in B]
        assert _no_dead_lines(sa, sb)
        assert all(sum(row) == 7 for grid in (sa, sb) for row in grid)
        assert all(sum(grid[i][j] for i in range(2)) == 2
                   for grid in (sa, sb) for j in range(7))
        if _grid_weights(sa, sb) > min(9, W_MAX):
            continue

        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 583 or k != 287:
            continue
        if not (475*n <= 1000*k < 525*n):
            continue
        signature = _q11_tanner_key(hx, hz, n)
        if signature in excluded:
            continue
        excluded.add(signature)

        certificate = _q11_four_certificate(hx, hz, n)
        if certificate is None:
            continue
        assert css_ok(hx, hz)
        assert max(row.bit_count() for row in hx + hz) <= min(9, W_MAX)
        assert lifted_k(G, A, B) == k

        def freeze(grid):
            return tuple(tuple(tuple(cell) for cell in row) for row in grid)

        entries.append((
            freeze(A), freeze(B), k, certificate,
            (bank[a][0], bank[b][0], scale), signature.hex()))
        if len(entries) >= count:
            break
    return tuple(entries)


def NEW_CTX(rng, G, idx):
    """Pure lookup of an invariant-separated, certified-distance-four code."""
    portfolio = _q11_portfolio(G, N_SLOTS)
    if idx >= len(portfolio):
        raise ValueError("novel certified-four rectangular portfolio exhausted")
    A, B, k, certificate, prescription, signature = portfolio[idx]
    return {
        "A": A, "B": B, "screen_k": k,
        "distance_certificate": 4,
        "logical_witness": certificate,
        "prescription": prescription,
        "tanner_signature": signature,
    }


def CELL(rng, G, side, i, j, ctx):
    """Return a fresh full cell without altering the immutable portfolio."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Rolled legal supports with uniform labels alone do not reliably suppress weight-two logicals: parent slot 5 measured d30s=2, with 252 reported weight-two operators.
# REFUTED: The cyclic order-11 triangle-free catalogue supplies twenty inequivalent distance-four designs; the measured program produced only four distinct codes across sixty slots, two with d30s=3.
# REFUTED: Component-first selection with complete short-word screens used only as diagnostics prevents weight-four failures: current slot 30 had d2s=5 but d30s=4, with minimum witnesses touching sector 2 and three sector-1 blocks.
# REFUTED: The even-first rotation-trinomial catalogue supplies five novel k=226 finalists: its current table measured five k=222 codes, while all k=226 entries were archived or isomorphic repeats.
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
