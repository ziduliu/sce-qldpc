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
CONCEPT = ("For cyclic C19, order 19, shapes (1,4)x(2,7), sum-free A binomial directions and strengthened B columns selected by independent noisy-syndrome holdouts should improve worse-basis ASC fitness above 3.2914 in cell 0.500.")
GROUP = ("Jump from nonabelian S3 of order six to cyclic C19 of order nineteen, a genuinely different group. The C19 (1,4)x(2,7) bed reached 3.3485. This motivates fresh degree and geometry contracts, not a claim that cyclicity alone improves reliability. Order nineteen is the largest legal lift for size factor thirty.")
SHAPES = ("Target cell 0.500 with (1,4)x(2,7): n=570, kb=285 and guaranteed rate 0.500. Components have 76 variables/19 checks and 133 variables/38 checks. Require complete helper-built k=285, retaining zero k-kb. Parent failures affected all 240 logicals; mostly 1.0-1.3x Acol/Bcol and Acol/Brow enrichment establishes no persistent isolated repair block.")
FILLING = ("Twenty contracts cross five invariant-distinct A geometries and canonical B support architectures with four strengthening levels; further batches consume further A classes. A uses three binomials and one Sidon trinomial, totaling nine terms. Signed three-binomial closures are excluded. B has one to four degree-three columns and otherwise degree-two columns, balanced row totals and full augmentation rank. Helper-derived component actions guide short-ambiguity annealing. Complete verified CSS finalists are screened modulo actual stabilizers and compared by common-stream two-interval noisy-syndrome BP training and independent holdouts. These probes omit actual ASC ordering, six-round histories and relay/LSD and are not calibrated pL estimates. HX weighs twelve; HZ weighs at most twelve. Six-round CNOT budgets are 17784,18354,18924 or19494 versus the parent's 17820. Test whether stronger B protection compensates for the extra A binomial and extraction exposure, reducing the parent's 119-146 X failures/250 in four measured slots and collective flips across all logicals.")
FAMILY_NAME = "r050_C19_14x27_sumfreeA_strengthenedB_holdout_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 19

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C19 requires order nineteen")
    els = [(a,) for a in range(q)]

    def red(g):
        return (int(g[0]) % q,)

    def mul(g, h):
        return ((red(g)[0] + red(h)[0]) % q,)

    def inv(g):
        return ((-red(g)[0]) % q,)

    def idx(g):
        return red(g)[0]

    return dict(elements=els, ident=(0,), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 4)               # (ra, ca)
B_SHAPE = (2, 7)               # (rb, cb)


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


def _retired_regular9_ctx(rng, G, idx):
    """Inactive order-six regular-nine search, retained for provenance."""
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


def NEW_CTX(rng, G, idx):
    """Sum-free A geometries, strengthened B and independent BP holdouts."""
    import hashlib
    import math
    from collections import Counter
    from itertools import combinations, combinations_with_replacement

    q = G.q
    assert q == 19 and A_SHAPE == (1, 4) and B_SHAPE == (2, 7)
    level, marker = divmod(idx % 20, 5)
    extra = (4, 3, 2, 1)[level]
    wanted = 5 * (idx // 20) + marker
    mask = (1 << q) - 1

    def clone(grid):
        return [[list(cell) for cell in row] for row in grid]

    def columns(rows, width):
        out = [0] * width
        for r, row in enumerate(rows):
            while row:
                bit = row & -row
                out[bit.bit_length() - 1] |= 1 << r
                row ^= bit
        return out

    # Only verified singleton actions are cached. Their linear
    # composition is checked against the complete selected lift.
    actions = {"A": {}, "B": {}}
    for g in G.elements:
        hx1, _, _ = lifted_matrices(G, [[[g]]], [[[]]])
        _, hz1, _ = lifted_matrices(G, [[[]]], [[[g]]])
        actions["A"][G.idx(g)] = tuple(row & mask for row in hx1)
        actions["B"][G.idx(g)] = tuple(row & mask for row in hz1)

    def component(grid, side):
        rows = [0] * (len(grid) * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, bits in enumerate(actions[side][G.idx(g)]):
                        rows[i*q+t] ^= bits << (j*q)
        return rows, len(grid[0]) * q

    def short_statistics(rows, width):
        cs = columns(rows, width)
        singles = Counter(cs)
        duplicate = singles.get(0, 0)
        duplicate += sum(v*(v-1)//2 for v in singles.values())
        pairs = {}
        triples = quartets = boundary = 0
        for i, x in enumerate(cs):
            for y in cs[:i]:
                value = x ^ y
                triples += singles.get(value, 0)
                quartets += pairs.get(value, 0)
                pairs[value] = pairs.get(value, 0) + 1
                weight = value.bit_count()
                if weight < 3:
                    boundary += (3-weight)**2
        four = sum(
            v*(v-1)//2
            for i, row in enumerate(rows)
            for other in rows[:i]
            for v in [(row & other).bit_count()])
        return duplicate, triples//3, quartets//3, boundary, four

    def fingerprint(rows, width):
        """Rooted component invariant, not complete Tanner canonization."""
        nr = len(rows)
        adjacency = [[] for _ in range(nr+width)]
        for c, value in enumerate(columns(rows, width)):
            while value:
                bit = value & -value
                r = bit.bit_length()-1
                adjacency[r].append(nr+c)
                adjacency[nr+c].append(r)
                value ^= bit
        images = []
        # Regular translations are transitive in each check-row orbit.
        for root in range(0, nr, q):
            colors = [0]*nr + [1]*width
            colors[root] = 2
            digest = hashlib.sha256()
            for _ in range(10):
                signatures = [
                    (colors[v], tuple(sorted(colors[w]
                                             for w in adjacency[v])))
                    for v in range(nr+width)]
                digest.update(repr(sorted(
                    Counter(signatures).items())).encode("ascii"))
                palette = {s: t for t, s in enumerate(
                    sorted(set(signatures)))}
                colors = [palette[s] for s in signatures]
            images.append(digest.hexdigest())
        return min(images)

    def direction(value):
        value %= q
        return min(value, q-value)

    def normal(cell, multiplier=1):
        values = [(multiplier*a) % q for a in cell]
        return min(
            tuple(sorted((a-origin) % q for a in values))
            for origin in values)

    def geometry_key(cells):
        # Common cyclic automorphisms, independent column origins and
        # column permutations are gauges, not additional submitted codes.
        return min(
            tuple(sorted(normal(cell, u) for cell in cells))
            for u in range(1, q))

    def sum_free(gaps):
        signed = tuple(v for gap in gaps for v in (gap, -gap))
        return all((a+b+c) % q
                   for a in signed for b in signed for c in signed)

    bin_book = [
        gaps for gaps in combinations(range(1, 10), 3)
        if sum_free(gaps)]
    tri_book = sorted({
        normal((0, a, b))
        for a, b in combinations(range(1, q), 2)
        if len({direction(a), direction(b), direction(a-b)}) == 3})
    assert bin_book and tri_book

    raw_book, raw_seen = [], set()
    for gaps in bin_book:
        for triangle in tri_book:
            signature = geometry_key(
                tuple((0, gap) for gap in gaps) + (triangle,))
            if signature in raw_seen:
                continue
            raw_seen.add(signature)
            A = [[[(a,) for a in cell] for cell in signature]]
            ar, width = component(A, "A")
            stats = short_statistics(ar, width)
            if stats[0] or stats[1] or gf2_rank(ar) != q:
                continue
            # Actual short words precede local cycle penalties.
            key = (stats[2], stats[4], stats[3], signature)
            raw_book.append((key, A, ar))

    bank, aclasses = [], set()
    for key, A, ar in sorted(raw_book, key=lambda item: item[0]):
        aid = fingerprint(ar, 4*q)
        if aid in aclasses:
            continue
        aclasses.add(aid)
        bank.append((key, aid, A, ar))
        if len(bank) >= max(24, wanted+4):
            break
    if wanted >= len(bank):
        raise ValueError("distinct sum-free C19 A geometry book exhausted")
    akey, aid, A, ar = bank[wanted]
    sa = [[len(cell) for cell in A[0]]]
    assert sorted(sa[0]) == [2, 2, 2, 3] and sum(sa[0]) == 9

    # Balanced degree contracts. Select one canonical support per
    # zero/concentration category, not row/column permutation copies.
    weak_types = ((1,1), (2,0), (0,2))
    strong_types = ((2,1), (1,2), (3,0), (0,3))
    support_classes = {}
    for weak in combinations_with_replacement(weak_types, 7-extra):
        for strong in combinations_with_replacement(strong_types, extra):
            profile = weak + strong
            totals = tuple(sum(p[i] for p in profile) for i in range(2))
            if abs(totals[0]-totals[1]) > 1 or max(totals) > 9:
                continue
            zeros = sum(0 in p for p in profile)
            concentrated = sum(3 in p for p in profile)
            if zeros > 4 or concentrated > 1:
                continue
            signature = min(
                tuple(sorted(profile)),
                tuple(sorted((b, a) for a, b in profile)))
            sb = [[p[i] for p in signature] for i in range(2)]
            augmentation = [
                sum((w & 1) << j for j, w in enumerate(row))
                for row in sb]
            if gf2_rank(augmentation) != 2:
                continue
            if not _no_dead_lines(sa, sb) or _grid_weights(sa, sb) > W_MAX:
                continue
            category = (zeros, concentrated)
            old = support_classes.get(category)
            if old is None or signature < old:
                support_classes[category] = signature
    support_book = sorted(support_classes.items())
    if len(support_book) < 5:
        raise ValueError("balanced strengthened-B support book exhausted")
    category, profile = support_book[marker]
    sb = [[p[i] for p in profile] for i in range(2)]
    assert sum(map(sum, sb)) == 14+extra
    assert sorted(map(sum, profile)) == [2]*(7-extra) + [3]*extra
    assert _no_dead_lines(sa, sb)
    assert 9 + max(map(sum, profile)) == 12
    assert max(map(sum, sb)) + 3 <= 12
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    expected_cnots = 6*q*(9*9 + 5*(14+extra))

    anchors = [
        next(i for i in range(2) if sb[i][j]) for j in range(7)]
    mutable = [
        (i, j) for i in range(2) for j in range(7)
        if sb[i][j] and not (i == anchors[j] and sb[i][j] == 1)]
    nonidentity = [g for g in G.elements if g != G.e]

    def new_cell(i, j):
        count = sb[i][j]
        if i == anchors[j]:
            if count == 2:
                return [G.e, (rng.randrange(1, 10),)]
            return [G.e] + rng.sample(nonidentity, count-1)
        return rng.sample(G.elements, count)

    def draw_B():
        return [[new_cell(i, j) if sb[i][j] else []
                 for j in range(7)] for i in range(2)]

    def assess(B):
        br, width = component(B, "B")
        stats = short_statistics(br, width)
        deficit = 2*q-gf2_rank(br)
        energy = (
            1000*stats[0] + 100*stats[1] + stats[2]
            + 500*deficit + 0.015*stats[3] + 0.01*stats[4])/q
        return energy, stats, deficit, br

    pool, seen_labels = [], set()
    serial = 0

    def remember(B, stats, deficit, br):
        nonlocal serial
        if stats[0] or deficit:
            return
        key = (stats[1], stats[2], stats[3], stats[4])
        if len(pool) == 8 and key > pool[-1][0]:
            return
        signature = tuple(
            tuple(sorted(G.idx(g) for g in cell))
            for row in B for cell in row)
        if signature in seen_labels:
            return
        seen_labels.add(signature)
        serial += 1
        pool.append((key, serial, clone(B), list(br)))
        pool.sort(key=lambda item: item[:2])
        del pool[8:]

    for restart in range(4):
        B = draw_B()
        energy, stats, deficit, br = assess(B)
        remember(B, stats, deficit, br)
        for step in range(360):
            proposal = clone(B)
            i, j = rng.choice(mutable)
            proposal[i][j] = new_cell(i, j)
            proposed, ps, pd, pr = assess(proposal)
            delta = proposed-energy
            temperature = 9.0*(0.987**step) + 0.04
            if delta <= 0 or rng.random() < math.exp(
                    -min(700.0, delta/temperature)):
                B, energy, stats, deficit, br = proposal, proposed, ps, pd, pr
                remember(B, stats, deficit, br)
    if not pool:
        for _ in range(512):
            B = draw_B()
            _, stats, deficit, br = assess(B)
            remember(B, stats, deficit, br)
            if len(pool) >= 4:
                break
    if not pool:
        raise ValueError("no rank-safe strengthened C19 B component")

    def quotient(stabilizers, width):
        pivots = {}
        for row in stabilizers:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        residue = [0]*width
        for c in range(width):
            row = pivots.get(c)
            if row is None:
                residue[c] = 1 << c
                continue
            tail, value = row ^ (1 << c), 0
            while tail:
                bit = tail & -tail
                value ^= residue[bit.bit_length()-1]
                tail ^= bit
            residue[c] = value
        return residue

    def fault_mass(checks, stabilizers, width):
        """Exact stabilizer classes with uncalibrated exposure weights."""
        sensed = columns(checks, width)
        spread = columns(stabilizers, width)
        if 0 in sensed or len(set(sensed)) != width:
            return None
        residue = quotient(stabilizers, width)
        singles = dict(zip(sensed, residue))
        exposure = [1+a.bit_count()+b.bit_count()
                    for a, b in zip(sensed, spread)]
        buckets = {}
        triples = 0

        def add(s, label, mass):
            classes = buckets.setdefault(s, {})
            classes[label] = classes.get(label, 0)+mass

        for i, s in enumerate(sensed):
            add(s, residue[i], 4096*exposure[i])
        for i, si in enumerate(sensed):
            for j in range(i):
                s = si ^ sensed[j]
                label = residue[i] ^ residue[j]
                other = singles.get(s)
                triples += int(other is not None and other != label)
                mass = 64*exposure[i]*exposure[j]
                # Shared opposite checks indicate possible spread,
                # not hooks certified for the evaluator's ASC order.
                mass += 96*(spread[i] & spread[j]).bit_count()
                add(s, label, mass)
        loss = peak = ambiguous = 0
        for classes in buckets.values():
            if len(classes) > 1:
                error = sum(classes.values())-max(classes.values())
                loss += error
                peak = max(peak, error)
                ambiguous += 1
        assert triples % 3 == 0
        return (triples//3, loss, peak, ambiguous), residue

    def noisy_probe(checks, stabilizers, residue, width, seed, shots):
        """Two-interval noisy-syndrome BP, NOT ASC circuit measurement.

        Decode H*e1+m=s1 and H*(e1+e2)=s2. The closing syndrome
        is perfect. Unknown opposite-check suffix orders approximate
        spread opportunities. Actual ASC ordering and relay/LSD are absent.
        """
        import numpy as np

        def sites(row):
            out = []
            while row:
                bit = row & -row
                out.append(bit.bit_length()-1)
                row ^= bit
            return out

        m = len(checks)
        variables = 2*width+m
        augmented = [
            row | (1 << (2*width+r))
            for r, row in enumerate(checks)
        ] + [row | (row << width) for row in checks]
        neighborhoods = [sites(row) for row in augmented]
        row_width = max(map(len, neighborhoods))
        incidence = np.full(
            (2*m, row_width), variables, dtype=np.int32)
        for r, ns in enumerate(neighborhoods):
            incidence[r, :len(ns)] = ns
        flat = incidence.ravel()
        valid = flat != variables
        real = valid.reshape(2*m, row_width)
        edges = len(flat)
        degree = np.bincount(flat[valid], minlength=variables)
        assert np.all(degree > 0)
        ve = np.full(
            (variables+1, int(degree.max())), edges, dtype=np.int32)
        used = np.zeros(variables, dtype=np.int32)
        for edge in np.flatnonzero(valid):
            c = int(flat[edge])
            ve[c, used[c]] = edge
            used[c] += 1

        sensed = np.asarray([
            c.bit_count() for c in columns(checks, width)], dtype=float)
        opposite = np.asarray([
            c.bit_count() for c in columns(stabilizers, width)], dtype=float)
        levels = np.asarray((0.25, 0.50, 0.75))[np.arange(shots) % 3]
        data_mu = 0.004*(2+sensed+opposite)
        measure_mu = np.asarray([
            0.004*(row.bit_count()+2) for row in checks])
        data_p = 0.5*(1-np.exp(-2*levels[:, None]*data_mu[None, :]))
        measure_p = 0.5*(1-np.exp(
            -2*levels[:, None]*measure_mu[None, :]))
        channel = np.random.default_rng(seed)
        error = np.zeros((shots, variables+1), dtype=bool)
        error[:, :width] = channel.random((shots, width)) < data_p
        error[:, width:2*width] = (
            channel.random((shots, width)) < data_p)
        error[:, 2*width:variables] = (
            channel.random((shots, m)) < measure_p)

        for row in stabilizers:
            ns = sites(row)
            if len(ns) < 2:
                continue
            order = channel.permutation(ns)
            event_p = levels*0.002*(len(ns)+1)
            for interval in range(2):
                active = channel.random(shots) < event_p
                for shot in np.flatnonzero(active):
                    cut = int(channel.integers(1, len(order)))
                    error[shot, interval*width+order[cut:]] ^= True

        marginal_mu = data_mu + 0.0015*opposite
        marginal_p = np.clip(0.5*(1-np.exp(
            -2*levels[:, None]*marginal_mu[None, :])), 0.002, 0.25)
        probabilities = np.concatenate((
            marginal_p, marginal_p, np.clip(measure_p, 0.002, 0.25)),
            axis=1)
        prior = np.full((shots, variables+1), 30.0)
        prior[:, :variables] = np.log((1-probabilities)/probabilities)
        target = (error[:, flat].reshape(
            shots, 2*m, row_width).sum(axis=2) & 1).astype(bool)
        signs_target = np.where(target, -1.0, 1.0)
        messages = prior[:, flat].copy()
        messages[:, ~valid] = 30.0
        true_final = error[:, :width] ^ error[:, width:2*width]
        wrong = work = 0

        for iteration in range(1, 33):
            count = len(prior)
            shaped = messages.reshape(count, 2*m, row_width)
            signs = np.where(shaped < 0, -1.0, 1.0)
            logs = np.log(np.clip(
                np.tanh(np.abs(shaped)/2), 1e-12, 1-1e-8))
            logs[:, ~real] = 0.0
            product = np.minimum(
                np.exp(logs.sum(axis=2, keepdims=True)-logs), 1-1e-8)
            outgoing = (
                (np.log1p(product)-np.log1p(-product))
                * signs * signs.prod(axis=2, keepdims=True)
                * signs_target[:, :, None]).reshape(count, edges)
            outgoing[:, ~valid] = 0.0
            padded = np.zeros((count, edges+1))
            padded[:, :edges] = outgoing
            posterior = prior + padded[:, ve].sum(axis=2)
            posterior[:, variables] = 30.0
            hard = posterior < 0
            matched = (hard[:, flat].reshape(
                count, 2*m, row_width).sum(axis=2) & 1).astype(bool)
            settled = np.all(matched == target, axis=1)
            final = hard[:, :width] ^ hard[:, width:2*width]
            for shot in np.flatnonzero(settled):
                label = 0
                for c in np.flatnonzero(final[shot] ^ true_final[shot]):
                    label ^= residue[int(c)]
                wrong += int(bool(label))
                work += iteration
            keep = ~settled
            if not np.any(keep):
                return wrong, 0, work
            damping = 0.25 if iteration <= 12 else 0.50
            messages = np.clip(
                damping*messages
                + (1-damping)*(posterior[:, flat]-outgoing),
                -30, 30)[keep]
            messages[:, ~valid] = 30.0
            prior = prior[keep]
            target = target[keep]
            signs_target = signs_target[keep]
            true_final = true_final[keep]
        unresolved = len(prior)
        return wrong+unresolved, unresolved, work+33*unresolved

    options, bclasses = [], set()
    for coarse, serial, B, br in pool:
        bid = fingerprint(br, 7*q)
        if bid in bclasses:
            continue
        bclasses.add(bid)
        hx, hz, n = lifted_matrices(G, A, B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if n != 570 or k != 285:
            continue
        xresult = fault_mass(hz, hx, n)
        zresult = fault_mass(hx, hz, n)
        if xresult is None or zresult is None:
            continue
        xc, xr = xresult
        zc, zr = zresult
        structural = (
            max(xc[1], zc[1])/k, (xc[1]+zc[1])/k,
            max(xc[2], zc[2]), xc[2]+zc[2],
            xc[3]+zc[3], coarse, bid)
        guard = (max(xc[0], zc[0]), xc[0]+zc[0])
        options.append((
            guard, structural, serial, B, br, bid, hx, hz,
            xc, zc, xr, zr))
        if len(options) >= 4:
            break
    if not options:
        raise ValueError("no verified k=285 strengthened C19 pairing")
    options.sort(key=lambda item: item[:3])

    def quality(x, z):
        return (max(x[0], z[0]), x[0]+z[0],
                max(x[1], z[1]), x[1]+z[1], x[2]+z[2])

    seed = 190927031 + 1009*idx
    training = []
    for number, option in enumerate(options):
        guard, structural, serial, B, br, bid, hx, hz, xc, zc, xr, zr = option
        x = noisy_probe(hz, hx, xr, 570, seed, 24)
        z = noisy_probe(hx, hz, zr, 570, seed+1, 24)
        key = guard + quality(x, z) + structural + (number,)
        training.append((key, number, option, x, z))
    training.sort(key=lambda item: item[:2])

    finalists = []
    for _, number, option, tx, tz in training[:2]:
        guard, structural, serial, B, br, bid, hx, hz, xc, zc, xr, zr = option
        x = noisy_probe(hz, hx, xr, 570, seed+1000003, 64)
        z = noisy_probe(hx, hz, zr, 570, seed+1000004, 64)
        key = (guard + quality(x, z) + quality(tx, tz)
               + structural + (number,))
        finalists.append((key, option, x, z, tx, tz))
    finalists.sort(key=lambda item: item[0])
    key, option, x, z, tx, tz = finalists[0]
    guard, structural, serial, B, br, bid, hx, hz, xc, zc, xr, zr = option

    # Independently check cached component placement against the
    # complete evaluator-convention CSS matrices.
    actual_a = [
        sum(((row >> (j*7*q)) & mask) << (j*q)
            for j in range(4))
        for row in hx[:q]]
    actual_b = [row & ((1 << (7*q))-1) for row in hz[:2*q]]
    assert actual_a == ar and actual_b == br
    assert gf2_rank(actual_a) == q
    assert gf2_rank(actual_b) == 2*q
    assert css_ok(hx, hz)
    assert lifted_k(G, A, B) == 285
    assert 0.475 <= 285/570 < 0.525
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) <= 12
    assert max(row.bit_count() for row in hx+hz) <= W_MAX
    assert 6*sum(row.bit_count() for row in hx+hz) == expected_cnots
    assert expected_cnots in (17784, 18354, 18924, 19494)
    assert all(len(cell) <= CELL_MAX and len(set(cell)) == len(cell)
               for grid in (A, B) for row in grid for cell in row)
    return {"A": A, "B": B, "k": 285,
            "contract": (extra, marker, wanted, category),
            "component_identities": (aid, bid),
            "A_geometry": akey, "fault_profiles": (xc, zc),
            "training_surrogate": (tx, tz),
            "holdout_surrogate": (x, z),
            "selection_score": key, "cnots": expected_cnots}


def CELL(rng, G, side, i, j, ctx):
    """Emit the complete verified strengthened-C19 cell without redraws."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Legal rolled supports and uniform labels alone did not give reliable memory on the parent D14 (6,2)x(6,2) bed: fitness 3.4511, ubiquitous weight-four sector-2 witnesses, and frequent failures in both bases.
# REFUTED: First-pass syndrome settlement certifies reliable memory on the immediate parent: slot 0 X settled 246/4/0 yet failed 120/250 shots, affecting all 213 logical qubits.
# REFUTED: Odd systematic A and monomial bipartite B plus short-hook balancing suffice for reliable D14 memory: fitness was 3.2001, measured bases failed 107-133/250 shots, all 213 logicals were affected, and sixty slots collapsed to six distinct codes with abundant weight-four witnesses.
# REFUTED: Full-rank regular-nine components and stabilizer-classified short-fault selection suffice to suppress collective order-six memory failures: fitness was 3.2914, all measured codes retained weight-four witnesses, Z failed 163-195/500 shots, and failures affected all 240 logicals.
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
