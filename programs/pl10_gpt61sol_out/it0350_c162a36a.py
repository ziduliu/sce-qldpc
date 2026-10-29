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
W_MAX = int(os.environ.get("CM_WMAX", "12"))
CELL_MAX = 5
N_SLOTS = int(os.environ.get("CM_NSLOTS", "20"))
BASE_SEED = 20260903


# EVOLVE-BLOCK-START
CONCEPT = ("On metacyclic Z3:Z2(s=2), order 6, shapes (2,7)x(3,11), regular-nine components with mixed degree-two/three columns and reduced stabilizer-inequivalent short-fault ambiguity should improve worse-basis circuit fitness above the parent's 3.2001 despite greater extraction work.")
GROUP = ("Keep the same metacyclic construction rule, reducing parameters from (7,2,6) to (3,2,2). The new size factor 83 forces q<=7, and order 6 gives n=498. This group/shape bed is absent from the supplied BEDS; neither the parent's 3.2001 nor other shapes' scores establish an advantage for this group.")
SHAPES = ("Target cell 0.500 with (2,7)x(3,11): n=498, kb=240, guaranteed rate 40/83=0.481928. Components have 42 variables/12 checks and 66 variables/18 checks, with similar classical rates. Require complete verified 240<=k<=261. Parent weight-four witnesses occupied widespread Acol/Bcol blocks, failures affected all 213 logicals, and roughly 1.0-1.2x enrichment did not establish a persistent Acol or Brow repair site.")
FILLING = ("Twenty initial contracts cross four A architectures with five balanced B architectures; additional slots consume further canonical B support contracts. Every component row has nine terms. A has three degree-two and four degree-three columns; B has six degree-two and five degree-three columns. Both CSS weight sums are <=12 and extraction uses 17820 six-round CNOTs versus the parent's measured 13440. Helper-derived component actions guide label annealing, full component ranks and short-word suppression; rooted component invariants conservatively collapse equivalent proposals. Full verified CSS selection minimizes worse-basis stabilizer-quotiented singleton/pair fault-mass ambiguity and its concentration. These are uncalibrated circuit-risk surrogates, not ASC simulations. Test whether fewer short ambiguities compensate for increased gate exposure and reduce the parent's 107-133 failed X shots and 121-133 failed Z shots per 250.")
FAMILY_NAME = "r050_m3r2s2_27x311_regular9_faultmass_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 6

def custom_ops(q):
    # Same metacyclic rule; factor 83 requires q<=7.
    # Z_3 x| Z_2: y x y^-1 = x^2, with s^r=1 mod m.
    if not valid_q(q):
        raise ValueError("this metacyclic parameter set requires q=6")
    m, r, s = 3, 2, 2
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
B_SHAPE = (3, 11)              # (rb, cb)


def _distinct_terms(rng, G, count):
    terms = []
    seen = set()
    while len(terms) < count:
        g = G.pick(rng)
        key = G._key(g)
        if key not in seen:
            seen.add(key)
            terms.append(g)
    return terms


def _b_profile(rng, extra, zeros):
    """Seven live columns; fixed total terms and fixed zero-cell count."""
    for _ in range(2000):
        triple = set(rng.sample(range(7), extra))
        sparse = set(rng.sample(range(7), zeros))
        sb = [[0] * 7 for _ in range(2)]
        for j in range(7):
            degree = 3 if j in triple else 2
            if j in sparse:
                a, b = 0, degree
            else:
                a, b = 1, degree - 1
            if rng.randrange(2):
                a, b = b, a
            sb[0][j], sb[1][j] = a, b
        if 0 < sum(sb[0]) <= 9 and 0 < sum(sb[1]) <= 9:
            return sb
    raise ValueError("could not realize deliberate B profile")


def _syndrome_risk(rows, n):
    """Ambiguity counts, not a distance certificate or circuit score."""
    columns = [0] * n
    for r, row in enumerate(rows):
        check = 1 << r
        while row:
            bit = row & -row
            columns[bit.bit_length() - 1] |= check
            row ^= bit
    if 0 in columns or len(set(columns)) != n:
        return None
    singles = set(columns)
    pairs = {}
    triples = 0
    for a in range(n):
        x = columns[a]
        for b in range(a):
            syndrome = x ^ columns[b]
            triples += syndrome in singles
            pairs[syndrome] = pairs.get(syndrome, 0) + 1
    collisions = sum(c * (c - 1) // 2 for c in pairs.values())
    return triples // 3, collisions


def _retired_odd_systematic_ctx(rng, G, idx):
    """Inactive previous-bed construction retained for provenance."""
    from itertools import combinations

    q = G.q
    els = G.elements
    e = G.e
    mask = (1 << q) - 1
    mul = [[G.idx(G.mul(g, h)) for h in els] for g in els]
    autos = [
        [G.idx(((u * g[0] + v * g[1]) % 7, g[1]))
         for g in els]
        for u in range(1, 7) for v in range(7)
    ]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def four_cycles(cols):
        overlaps = {}
        for syndrome in cols:
            checks = []
            while syndrome:
                bit = syndrome & -syndrome
                checks.append(bit.bit_length() - 1)
                syndrome ^= bit
            for pair in combinations(checks, 2):
                overlaps[pair] = overlaps.get(pair, 0) + 1
        return sum(t * (t - 1) // 2 for t in overlaps.values())

    def component_fours(cols, weights):
        """Exact weight-four words; ordered pairs count each support once."""
        pairs = {}
        count = exposure = 0
        participation = [0] * len(cols)
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                syndrome = cols[a] ^ cols[b]
                bucket = pairs.setdefault(syndrome, [])
                for c, d in bucket:
                    if d < a:
                        count += 1
                        ids = (c, d, a, b)
                        exposure += sum(weights[x] * weights[y]
                                        for x, y in combinations(ids, 2))
                        for x in ids:
                            participation[x] += 1
                bucket.append((a, b))
        return exposure, count, max(participation)

    # Right translations are column gauges. Enumerate their trinomial
    # orbits rather than repeatedly drawing relabeled copies.
    tri_type = {}
    representatives = []
    for triple in combinations(range(q), 3):
        if triple in tri_type:
            continue
        orbit = {
            tuple(sorted(mul[g][h] for g in triple))
            for h in range(q)
        }
        t = len(representatives)
        representatives.append(min(orbit))
        for image in orbit:
            tri_type[image] = t

    type_actions = [
        [tri_type[tuple(sorted(aut[g] for g in triple))]
         for triple in representatives]
        for aut in autos
    ]

    # Obtain every component column from the verified full construction.
    # Only extraction/concatenation is performed here, not a separate lift.
    tri_columns = []
    B_probe = [[[e] for _ in range(7)] for _ in range(2)]
    unit_columns = None
    for triple in representatives:
        A_probe = [[[e], [els[g] for g in triple], [e], [e]]]
        hx, hz, n = lifted_matrices(G, A_probe, B_probe)
        ar = [((hx[t] >> (7 * q)) & mask) for t in range(q)]
        tri_columns.append(columns(ar, q))
        if unit_columns is None:
            unit_columns = columns([hx[t] & mask for t in range(q)], q)

    A_book = []
    seen_A = set()
    for triple_types in combinations(range(len(representatives)), 3):
        canonical = min(
            tuple(sorted(action[t] for t in triple_types))
            for action in type_actions
        )
        if canonical in seen_A:
            continue
        seen_A.add(canonical)
        ac = list(unit_columns)
        for t in canonical:
            ac.extend(tri_columns[t])
        assert len(set(ac)) == 4 * q and all(s.bit_count() % 2 for s in ac)
        # Sector-1 gate degrees are 3 for the unit block and 5 otherwise.
        risk = component_fours(ac, [3] * q + [5] * (3 * q))
        A_book.append((risk + (four_cycles(ac),), canonical))
    A_book.sort()

    # Classify B subsets under left translation and group automorphisms.
    # These generators also cover right translation through conjugation.
    generators = [
        mul[G.idx((1, 0))],
        mul[G.idx((0, 1))],
        autos[14],                 # u=3, v=0
        autos[1],                  # u=1, v=1
    ]

    def image_mask(bits, permutation):
        image = 0
        while bits:
            bit = bits & -bits
            image |= 1 << permutation[bit.bit_length() - 1]
            bits ^= bit
        return image

    unseen = {sum(1 << g for g in subset)
              for subset in combinations(range(q), 7)}
    B_book = []
    A_probe = [[[e] for _ in range(4)]]
    bmask = (1 << (7 * q)) - 1
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        orbit = [seed]
        todo = [seed]
        while todo:
            current = todo.pop()
            for permutation in generators:
                image = image_mask(current, permutation)
                if image in unseen:
                    unseen.remove(image)
                    orbit.append(image)
                    todo.append(image)
        canonical = min(orbit)
        labels = tuple(g for g in range(q) if canonical & (1 << g))
        B_probe = [
            [[e] for _ in range(7)],
            [[els[g]] for g in labels],
        ]
        hx, hz, n = lifted_matrices(G, A_probe, B_probe)
        bc = columns([row & bmask for row in hz[:2 * q]], 7 * q)
        assert len(set(bc)) == 7 * q and all(s.bit_count() == 2 for s in bc)
        risk = component_fours(bc, [1] * (7 * q))
        B_book.append(((risk[1], risk[2]), labels))
    B_book.sort()

    # Twenty slots use four strong A classes crossed with five B classes.
    # Larger harnesses continue to additional A classes, not orbit copies.
    nb = min(5, len(B_book))
    if not nb or idx >= len(A_book) * nb:
        raise ValueError("distinct component design book exhausted")
    a_types = A_book[idx // nb][1]
    b_labels = B_book[idx % nb][1]
    A = [[[e]] + [[els[g] for g in representatives[t]] for t in a_types]]
    assert _grid_weights([[1, 3, 3, 3]], [[1] * 7, [1] * 7]) <= W_MAX

    def span(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        return pivots

    def logical_risk(check_cols, stab_cols, stab_rows, degrees):
        """Exact nontrivial logicals through weight four, plus hook proxy."""
        pivots = span(stab_rows)

        def nontrivial(v):
            while v:
                p = v.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    return True
                v ^= old
            return False

        width = len(check_cols)
        bits = [1 << a for a in range(width)]
        singles = {}
        counts = [0, 0, 0, 0]
        hooks = exposure = 0
        for a, syndrome in enumerate(check_cols):
            bucket = singles.setdefault(syndrome, [])
            if syndrome == 0 and nontrivial(bits[a]):
                counts[0] += 1
            for b in bucket:
                if nontrivial(bits[a] | bits[b]):
                    counts[1] += 1
            bucket.append(a)

        pairs = {}
        for a in range(width):
            for b in range(a + 1, width):
                syndrome = check_cols[a] ^ check_cols[b]
                support = bits[a] | bits[b]
                for c in singles.get(syndrome, ()):
                    if c > b and nontrivial(support | bits[c]):
                        counts[2] += 1
                bucket = pairs.setdefault(syndrome, [])
                for c, d, previous in bucket:
                    if d >= a or not nontrivial(previous | support):
                        continue
                    counts[3] += 1
                    ids = (c, d, a, b)
                    # Common same-error-basis checks can propagate pairs.
                    # This is pessimistic: the actual ASC ordering is not
                    # available here, so it is not a circuit certificate.
                    for x, y in combinations(ids, 2):
                        hooks += (stab_cols[x] & stab_cols[y]).bit_count()
                        exposure += degrees[x] * degrees[y]
                bucket.append((a, b, support))
        return tuple(counts), hooks, exposure

    order = list(range(len(autos)))
    rng.shuffle(order)
    order = [0] + [t for t in order if t != 0][:11]
    best = None
    best_key = None
    for t in order:
        aut = autos[t]
        B = [
            [[e] for _ in range(7)],
            [[els[aut[g]]] for g in b_labels],
        ]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k > 0 and 0.475 <= k / n < 0.525):
            continue
        xc, zc = columns(hx, n), columns(hz, n)
        degrees = [x.bit_count() + z.bit_count() for x, z in zip(xc, zc)]
        zrisk = logical_risk(xc, zc, hz, degrees)
        xrisk = logical_risk(zc, xc, hx, degrees)
        if any(zrisk[0][:3]) or any(xrisk[0][:3]):
            continue
        cx, cz = four_cycles(xc), four_cycles(zc)
        key = (
            max(xrisk[1], zrisk[1]), xrisk[1] + zrisk[1],
            max(xrisk[2], zrisk[2]), xrisk[2] + zrisk[2],
            max(xrisk[0][3], zrisk[0][3]),
            xrisk[0][3] + zrisk[0][3],
            max(cx / len(hx), cz / len(hz)), cx + cz,
            k - 210,
        )
        if best_key is None or key < best_key:
            best_key = key
            best = {"A": A, "B": B}
    if best is None:
        raise ValueError("component brief failed verified short-logical screen")
    hx, hz, n = lifted_matrices(G, best["A"], best["B"])
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    return best


def _regular9_supports(idx):
    """Four A architectures crossed with canonical balanced B contracts."""
    from itertools import permutations

    a_special = (
        ((1, 2), (1, 2), (2, 1), (2, 1)),
        ((0, 3), (1, 2), (2, 1), (3, 0)),
        ((0, 3), (2, 1), (2, 1), (2, 1)),
        ((0, 3), (0, 3), (3, 0), (3, 0)),
    )
    acard = idx % 4
    bcard = idx // 4
    acolumns = list(a_special[acard]) + [(1, 1)] * 3
    sa = [[column[i] for column in acolumns] for i in range(2)]

    pair_p = ((2, 1, 0), (0, 1, 2))
    pair_q = ((2, 0, 1), (0, 2, 1))
    triangle = ((2, 1, 0), (0, 2, 1), (1, 0, 2))
    full = (1, 1, 1)
    edges = [(1, 1, 0), (1, 0, 1), (0, 1, 1)]
    edge_base = [edge for edge in edges for _ in range(2)]
    initial = (
        [full] * 5,
        list(pair_p) + [full] * 3,
        list(triangle) + [full] * 2,
        list(pair_p) + list(pair_q) + [full],
        list(triangle) + list(pair_p),
    )
    row_permutations = tuple(permutations(range(3)))

    def canonical(columns):
        return min(tuple(sorted(tuple(column[i] for i in p)
                                for column in columns))
                   for p in row_permutations)

    book, seen = [], set()

    def offer(columns):
        if len(columns) != 11:
            return
        if tuple(sum(column[i] for column in columns)
                 for i in range(3)) != (9, 9, 9):
            return
        signature = canonical(columns)
        if signature in seen:
            return
        grid = [[column[i] for column in signature] for i in range(3)]
        augmentation = [
            sum((count & 1) << j for j, count in enumerate(row))
            for row in grid
        ]
        if gf2_rank(augmentation) != 3:
            return
        seen.add(signature)
        book.append(grid)

    for triples in initial:
        offer(edge_base + triples)
    assert len(book) == 5

    # Further harness slots change weighted support geometry, not merely
    # origins, priorities or automorphism images of an earlier filling.
    catalogue_rng = random.Random(6032711)
    triple_palette = sorted(
        {full}
        | set(permutations((2, 1, 0)))
        | set(permutations((3, 0, 0))))
    for attempt in range(30000):
        if len(book) > bcard:
            break
        columns = [catalogue_rng.choice(edges) for _ in range(6)]
        columns += [catalogue_rng.choice(triple_palette) for _ in range(5)]
        offer(columns)
    if bcard >= len(book):
        raise ValueError("unused balanced regular-nine supports exhausted")

    sb = book[bcard]
    assert tuple(map(sum, sa)) == (9, 9)
    assert tuple(map(sum, sb)) == (9, 9, 9)
    assert sorted(map(sum, zip(*sa))) == [2] * 3 + [3] * 4
    assert sorted(map(sum, zip(*sb))) == [2] * 6 + [3] * 5
    assert _no_dead_lines(sa, sb)
    assert _grid_weights(sa, sb) == 12
    return sa, sb, (acard, bcard)


def NEW_CTX(rng, G, idx):
    """Regular-nine contracts, verified component search and fault quotient."""
    import hashlib
    import math
    from collections import Counter

    q = G.q
    assert q == 6
    sa, sb, contract = _regular9_supports(idx)
    block_mask = (1 << q) - 1

    # Cache only blocks returned by the verified construction. The final
    # components are independently projected from the complete CSS lift.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & block_mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & block_mask for row in hz1)

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def component(grid, side):
        rows = [0] * (len(grid) * q)
        width = len(grid[0]) * q
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    block = actions[side][G.idx(g)]
                    for t, value in enumerate(block):
                        rows[i * q + t] ^= value << (j * q)
        return rows, width

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            tag = 1 << r
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= tag
                row ^= bit
        return out

    def short_statistics(rows, width):
        """Exact short-kernel counts when columns are distinct/nonzero."""
        cs = columns(rows, width)
        singles = Counter(cs)
        duplicates = sum(v * (v - 1) // 2 for v in singles.values())
        duplicates += singles.get(0, 0)
        pairs = {}
        three = partitions = boundary = 0
        for i, value in enumerate(cs):
            for other in cs[:i]:
                syndrome = value ^ other
                three += singles.get(syndrome, 0)
                partitions += pairs.get(syndrome, 0)
                pairs[syndrome] = pairs.get(syndrome, 0) + 1
                weight = syndrome.bit_count()
                if weight < 3:
                    boundary += (3 - weight) ** 2
        four_cycles = 0
        for i, row in enumerate(rows):
            for other in rows[:i]:
                overlap = (row & other).bit_count()
                four_cycles += overlap * (overlap - 1) // 2
        return (duplicates, three // 3, partitions // 3,
                boundary, four_cycles)

    def rooted_identity(rows, width):
        """Conservative component invariant, not an isomorphism certificate.

        Group translations are transitive within each check-row orbit.
        One root per orbit therefore represents every possible check root.
        Equal invariants are discarded conservatively.
        """
        nr = len(rows)
        adjacency = [[] for _ in range(nr + width)]
        for c, syndrome in enumerate(columns(rows, width)):
            while syndrome:
                bit = syndrome & -syndrome
                r = bit.bit_length() - 1
                adjacency[r].append(nr + c)
                adjacency[nr + c].append(r)
                syndrome ^= bit
        identities = []
        for root in range(0, nr, q):
            colors = [0] * nr + [1] * width
            colors[root] = 2
            digest = hashlib.sha256()
            for iteration in range(9):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(nr + width)
                ]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            identities.append(digest.hexdigest())
        return min(identities)

    def optimize(support, side):
        nr, nc = len(support), len(support[0])
        anchors = [
            next(i for i in range(nr) if support[i][j])
            for j in range(nc)
        ]
        sites = [(i, j) for i in range(nr) for j in range(nc)
                 if support[i][j]
                 and not (i == anchors[j] and support[i][j] == 1)]

        def new_cell(i, j):
            count = support[i][j]
            if i == anchors[j]:
                return [G.e] + rng.sample(
                    [g for g in G.elements if g != G.e], count - 1)
            return rng.sample(G.elements, count)

        def draw():
            grid = [[[] for _ in range(nc)] for _ in range(nr)]
            occupied = {}
            for j in range(nc):
                live = tuple(i for i in range(nr) if support[i][j])
                for i in live:
                    grid[i][j] = new_cell(i, j)
                if len(live) == 2 and all(support[i][j] == 1 for i in live):
                    used = occupied.setdefault(live, set())
                    choices = [g for g in G.elements if G.idx(g) not in used]
                    if choices:
                        g = rng.choice(choices)
                        grid[live[0]][j] = [G.e]
                        grid[live[1]][j] = [g]
                        used.add(G.idx(g))
            return grid

        def assess(grid):
            rows, width = component(grid, side)
            stats = short_statistics(rows, width)
            deficit = nr * q - gf2_rank(rows)
            energy = (
                1000000 * stats[0] + 10000 * stats[1]
                + 100000 * deficit + stats[2]
                + 0.015 * stats[3] + 0.005 * stats[4]) / q
            return energy, stats, deficit, rows, width

        pool, seen_labels, seen_classes = [], set(), set()
        serial = 0

        def remember(grid, stats, deficit, rows, width):
            nonlocal serial
            if stats[0] or deficit:
                return
            key = (stats[1], stats[2], stats[3], stats[4])
            if len(pool) == 6 and key > pool[-1][0]:
                return
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen_labels:
                return
            seen_labels.add(signature)
            identity = rooted_identity(rows, width)
            if identity in seen_classes:
                return
            seen_classes.add(identity)
            serial += 1
            pool.append((key, serial, clone(grid), identity))
            pool.sort(key=lambda item: item[:2])
            del pool[6:]

        for restart in range(5):
            grid = draw()
            energy, stats, deficit, rows, width = assess(grid)
            remember(grid, stats, deficit, rows, width)
            for step in range(420):
                proposal = clone(grid)
                i, j = rng.choice(sites)
                proposal[i][j] = new_cell(i, j)
                proposed, ps, pd, pr, pw = assess(proposal)
                delta = proposed - energy
                temperature = 18.0 * (0.986 ** step) + 0.08
                if delta <= 0 or rng.random() < math.exp(
                        -min(700.0, delta / temperature)):
                    grid, energy = proposal, proposed
                    stats, deficit, rows, width = ps, pd, pr, pw
                    remember(grid, stats, deficit, rows, width)
        if not pool:
            raise ValueError("no full-rank distinct-column component in contract")
        return pool

    def quotient_columns(stabilizers, width):
        """Linear canonical remainders modulo actual stabilizer rows."""
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        out = [0] * width
        for c in range(width):
            row = pivots.get(c)
            if row is None:
                out[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= out[bit.bit_length() - 1]
                tail ^= bit
            out[c] = value
        return out

    def fault_mass(checks, stabilizers, width):
        """Exact quotient classification with an uncalibrated fault model.

        Shared opposite checks supply a diluted possible-spread mass.
        Actual ASC ordering, measurement histories and relay/LSD are absent.
        """
        syndrome = columns(checks, width)
        spread = columns(stabilizers, width)
        if 0 in syndrome or len(set(syndrome)) != width:
            return None
        residue = quotient_columns(stabilizers, width)
        exposure = [1 + a.bit_count() + b.bit_count()
                    for a, b in zip(syndrome, spread)]
        singles = dict(zip(syndrome, residue))
        buckets = {0: {0: 1000000000}}
        triples = 0

        def add(s, label, mass):
            classes = buckets.setdefault(s, {})
            classes[label] = classes.get(label, 0) + mass

        for i, s in enumerate(syndrome):
            add(s, residue[i], 4096 * exposure[i])
        for i, si in enumerate(syndrome):
            for j in range(i):
                s = si ^ syndrome[j]
                label = residue[i] ^ residue[j]
                if s == 0 and label:
                    return None
                third = singles.get(s)
                if third is not None and third != label:
                    triples += 1
                mass = 64 * exposure[i] * exposure[j]
                mass += 256 * (spread[i] & spread[j]).bit_count()
                add(s, label, mass)

        loss = peak = ambiguous = 0
        spectrum = Counter()
        for classes in buckets.values():
            if len(classes) > 1:
                error = sum(classes.values()) - max(classes.values())
                loss += error
                peak = max(peak, error)
                ambiguous += 1
                spectrum[error] += 1
        return (triples // 3, loss, peak, ambiguous,
                tuple(sorted(spectrum.items())))

    apool = optimize(sa, "A")
    bpool = optimize(sb, "B")
    pairs = []
    for ak, at, A, aid in apool:
        for bk, bt, B, bid in bpool:
            coarse = tuple(value for t in range(4)
                           for value in (max(ak[t], bk[t]), ak[t] + bk[t]))
            pairs.append((coarse, at, bt, A, B, aid, bid))
    pairs.sort(key=lambda item: item[:3])

    winner = None
    seen_fault_profiles = set()
    for coarse, at, bt, A, B, aid, bid in pairs[:8]:
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if n != 498 or not (240 <= k <= 261):
            continue
        px = fault_mass(hz, hx, n)
        pz = fault_mass(hx, hz, n)
        if px is None or pz is None:
            continue
        invariant = tuple(sorted((px, pz)))
        if invariant in seen_fault_profiles:
            continue
        seen_fault_profiles.add(invariant)
        # Protect the weaker basis. Normalize ambiguity by the actual k,
        # matching the objective's per-logical-qubit normalization.
        score = (
            max(px[0], pz[0]), px[0] + pz[0],
            max(px[1], pz[1]) / k, (px[1] + pz[1]) / k,
            max(px[2], pz[2]), px[2] + pz[2],
            px[3] + pz[3], coarse, at, bt)
        if winner is None or score < winner[0]:
            winner = (score, A, B, hx, hz, k, px, pz, aid, bid)

    if winner is None:
        raise ValueError("no verified target-cell regular-nine CSS finalist")
    score, A, B, hx, hz, k, px, pz, aid, bid = winner

    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    actual_a = [
        sum(((hx[(i * cb) * q + t] >> (j * cb * q)) & block_mask)
            << (j * q) for j in range(ca))
        for i in range(ra) for t in range(q)
    ]
    actual_b = [
        row & ((1 << (cb * q)) - 1) for row in hz[:rb * q]
    ]
    assert actual_a == component(A, "A")[0]
    assert actual_b == component(B, "B")[0]
    assert gf2_rank(actual_a) == 12
    assert gf2_rank(actual_b) == 18
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == k
    assert 0.475 <= k / 498 < 0.525
    assert max(row.bit_count() for row in hx + hz) <= W_MAX
    assert sum(row.bit_count() for row in hx + hz) * 6 == 17820
    assert all(len({G._key(g) for g in cell}) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "k": k, "contract": contract,
            "component_identities": (aid, bid),
            "fault_profiles": (px, pz), "selection_score": score}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete verified cell of the new regular-nine shapes."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: First-pass syndrome settlement certifies reliable memory on the immediate parent: slot 0 X settled 246/4/0 yet failed 120/250 shots, affecting all 213 logical qubits.
# REFUTED: Odd systematic A and monomial bipartite B plus short-hook balancing suffice for reliable D14 memory: fitness was 3.2001, measured bases failed 107-133/250 shots, all 213 logicals were affected, and sixty slots collapsed to six distinct codes with abundant weight-four witnesses.
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
