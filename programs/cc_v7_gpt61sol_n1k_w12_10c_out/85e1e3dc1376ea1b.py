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
CONCEPT = ("AGL(1,8) x C2, order 112, shapes (2,3)x(2,3): nonzero canonical supports, full-generating larger cells, non-progression quotient geometry and independently challenged component kernels should improve five-code fitness above 325.2462 in cell 0.100. Transposition alone offers no distance advantage; readings remain provisional upper bounds.")
GROUP = ("Keep the validated affine-field group and order 112. Its wide and tall beds scored 310.4 and 325.2, respectively; the current measured tall readings were 62..67. Translation and central coordinates provide restricted searches extracted from verified matrices, not a distance-transfer theorem.")
SHAPES = ("Equal wide (2,3) components have 336 variables and 224 checks. q=112 gives n=1456, k>=112 and guaranteed rate 1/13; exact ranks retain k/n<0.125. Parent measured minima occupied sector-2 Arow/Brow families, notably every Arow at Brow1 and every Brow at Arow2. The wide orientation returns corresponding component families to sector 1 and screens their kernels directly.")
FILLING = ("Twenty canonical support pairs have row totals six/seven and column totals four/five, with no zero cells and both CSS maxima <=12. Binomials have relative order fourteen. Larger cells have noncommuting relative generators and mixed central coordinates; trinomial quotient supports and quartic complements exclude arithmetic progressions. Verified localized, full-component, central-fixed and translation-plane kernel probes challenge quotient/cycle selection, and three finalists receive independent longer searches. Original-span verification gates normal-fixed logical witnesses. No heuristic certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r010_affineF8xC2_23x23_nonprogression_multispace_retest_v4"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 112

def custom_ops(q):
    # (u,v,z): affine map t -> alpha^v*t + u over F8,
    # times central C2. F8 uses alpha^3 + alpha + 1 = 0.
    if not valid_q(q):
        raise ValueError("this affine-field bed requires order 112")
    powers = (1, 2, 4, 3, 6, 7, 5)
    els = [(u, v, z) for z in range(2)
           for v in range(7) for u in range(8)]

    def field_mul(a, b):
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            a <<= 1
            if a & 8:
                a ^= 11
        return out

    def red(g):
        return (g[0] & 7, g[1] % 7, g[2] & 1)

    def mul(g, h):
        return (g[0] ^ field_mul(powers[g[1]], h[0]),
                (g[1] + h[1]) % 7, g[2] ^ h[2])

    def inv(g):
        vi = (-g[1]) % 7
        return (field_mul(powers[vi], g[0]), vi, g[2])

    def idx(g):
        u, v, z = red(g)
        return (7 * z + v) * 8 + u

    return dict(elements=els, ident=(0, 0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 3)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


def _wide_support_design(idx):
    """Twenty deliberate weighted protographs, canonical under permutations.

    Both rows have six/seven terms, every column has four/five,
    and every cell is occupied. Binary augmentation rank is two.
    Dense and mixed-density pairs are distinct support designs.
    """
    from itertools import product, permutations

    options = [row for row in product(range(1, 5), repeat=3)
               if sum(row) in (6, 7)]
    permutations_3 = list(permutations(range(3)))
    profiles = set()
    for upper in options:
        for lower in options:
            rows = (upper, lower)
            if not all(4 <= upper[j] + lower[j] <= 5
                       for j in range(3)):
                continue
            augmentation = [
                sum((value & 1) << j for j, value in enumerate(row))
                for row in rows
            ]
            if gf2_rank(augmentation) != 2:
                continue
            key = min(
                tuple(rows[r][c] for r in rr for c in cc)
                for rr in ((0, 1), (1, 0))
                for cc in permutations_3
            )
            profiles.add(key)

    profiles = sorted(profiles)
    plans = [(a, b) for i, a in enumerate(profiles)
             for b in profiles[i:]]
    plans.sort(key=lambda pair: (
        -sum(sum(grid) for grid in pair),
        sum(value == 1 for grid in pair for value in grid),
        sum((value - 2)**2 for grid in pair for value in grid),
        pair,
    ))
    if len(plans) < 20:
        raise ValueError("insufficient canonical nonzero support pairs")
    a, b = plans[idx % 20]
    return [list(a[:3]), list(a[3:])], [list(b[:3]), list(b[3:])]


def _set_bits(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _full_relative_cell(G, cell):
    """For >=3 terms, verify noncommuting relative affine generators.

    Distinct multiplicative coordinates generate C7; noncommutation
    supplies a nonzero translation, whose C7 orbit spans F8. Mixed
    relative central coordinates then give the full order-112 product.
    """
    if len(cell) < 3:
        return True
    root_inverse = G.inv(cell[0])
    ratios = [G.mul(root_inverse, g) for g in cell[1:]]
    return (any(g[2] for g in ratios)
            and any(G.mul(g, h) != G.mul(h, g)
                    for i, g in enumerate(ratios) for h in ratios[:i]))


def _affine_grid(rng, G, supports, phase):
    """Fresh full-generating cells with non-progression quotient geometry."""
    grid = []
    for counts in supports:
        row = []
        for count in counts:
            for attempt in range(256):
                vs = rng.sample(range(7), count)
                if count in (3, 4):
                    triangle = (vs if count == 3 else
                                [v for v in range(7) if v not in vs])
                    # An unordered three-set is a progression exactly
                    # when one member is the midpoint of the other two.
                    if any((triangle[a] + triangle[b]) % 7 ==
                           (2 * triangle[c]) % 7
                           for a, b, c in ((0, 1, 2), (0, 2, 1),
                                           (1, 2, 0))):
                        continue
                flip = rng.randrange(2)
                if count == 0:
                    zs = []
                elif count == 1:
                    zs = [flip]
                elif count == 2:
                    zs = [flip, flip ^ 1]
                else:
                    ones = (rng.randrange(1, count) if phase == 0
                            else count // 2 if phase == 1 else 1)
                    zs = [1] * ones + [0] * (count - ones)
                    rng.shuffle(zs)
                    zs = [z ^ flip for z in zs]
                cell = [(rng.randrange(8), v, z)
                        for v, z in zip(vs, zs)]
                if _full_relative_cell(G, cell):
                    row.append(cell)
                    break
            else:
                raise ValueError("full-generating quotient cell failed")
        grid.append(row)
    return grid


def _component_stats(rows, ncols):
    """Four cycles and check-projection triangle excess, not distance."""
    columns = [0] * ncols
    pairs = {}
    four = 0
    for r, row in enumerate(rows):
        ns = list(_set_bits(row))
        for u in ns:
            columns[u] |= 1 << r
        for a, u in enumerate(ns):
            for v in ns[a + 1:]:
                key = u * ncols + v
                previous = pairs.get(key, 0)
                four += previous
                pairs[key] = previous + 1

    # Reject isolated variables and identical component columns:
    # the latter supply classical weight-two kernel vectors.
    if not all(columns) or len(set(columns)) != ncols:
        return None

    adjacency = [0] * len(rows)
    forced_triangles = 0
    for signature in columns:
        degree = signature.bit_count()
        forced_triangles += degree * (degree - 1) * (degree - 2) // 6
        for r in _set_bits(signature):
            adjacency[r] |= signature ^ (1 << r)

    # Require a connected component Tanner graph.
    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _set_bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _set_bits(neighbors >> (u + 1)):
            v += u + 1
            triangles += (neighbors & adjacency[v] &
                          (-1 << (v + 1))).bit_count()

    # With no four cycles, this is the number of genuine six cycles.
    # Otherwise it remains only a secondary projection statistic.
    return four, max(0, triangles - forced_triangles)


def _verified_components(G, hx, hz):
    """Extract both classical lifts from the scaffold's verified matrices."""
    q = G.q
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for r in range(q):
            lifted_row = hx[i * cb * q + r]  # beta = 0
            ar.append(sum(((lifted_row >> (j * cb * q)) & mask)
                          << (j * q) for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + r] & bmask             # A-column j = 0
          for a in range(rb) for r in range(q)]

    # Wide components carry the relevant families in their kernels.
    # Each authoritative component has 224 rows and 336 columns.
    return ar, br


def NEW_CTX(rng, G, idx):
    """Nonzero wide supports and verified multispace kernel screening."""
    sa, sb = _wide_support_design(idx)
    phase = (idx // 20 + idx // 5) % 3
    phase_b = (phase + idx % 3) % 3
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(sum(row) in (6, 7) for grid in (sa, sb) for row in grid)
    assert all(sum(row[j] for row in grid) in (4, 5)
               for grid in (sa, sb) for j in range(3))
    assert all(1 <= count <= 4
               for grid in (sa, sb) for row in grid for count in row)
    # This encoding orders the eight translations contiguously.
    assert all(G.idx((u, v, z)) == (7*z + v)*8 + u
               for z in range(2) for v in range(7) for u in range(8))
    n = 13 * G.q
    diagnostic_seed = rng.randrange(1 << 62)

    # Cache native operations only. Every screening matrix still comes
    # from lifted_matrices, with unchanged native element indexing.
    q = G.q
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

    class IndexedGroup:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = G.idx(G.e)

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    indexed_group = IndexedGroup()

    def build(A, B):
        aa = [[[G.idx(g) for g in cell] for cell in row] for row in A]
        bb = [[[G.idx(g) for g in cell] for cell in row] for row in B]
        return lifted_matrices(indexed_group, aa, bb)

    def pivots_of(rows):
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

    def remainder(word, pivots):
        """Full linear normal form modulo the ORIGINAL stabilizer span."""
        result = 0
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                bit = 1 << p
                result ^= bit
                word ^= bit
            else:
                word ^= old
        return result

    def project(rows, width):
        """Restrict VERIFIED check rows to translation-constant vectors.
        Keep all check rows: no independently implemented group lift."""
        fw = width // 8
        return list(dict.fromkeys(
            sum((((row >> (8*j)) & 255).bit_count() & 1) << j
                for j in range(fw))
            for row in rows))

    def expand(word):
        """Embed a projected vector in the actual qubit coordinates."""
        out = 0
        for j in _set_bits(word):
            out |= 255 << (8*j)
        return out

    def fixed_minimum(rows, width):
        """EXACT minimum in the normal-fixed classical subspace only."""
        projected = project(rows, width)
        fw = width // 8
        pivots = pivots_of(projected)
        ordered = sorted(pivots.items())
        basis = []
        for free in range(fw):
            if free in pivots:
                continue
            word = 1 << free
            for p, row in ordered:
                if (word & row).bit_count() & 1:
                    word |= 1 << p
            basis.append(word)
        # Avoid spending exponential work on highly deficient components.
        if not basis or len(basis) > 18:
            return 0
        word = 0
        reading = fw + 1
        for t in range(1, 1 << len(basis)):
            word ^= basis[(t & -t).bit_length() - 1]
            reading = min(reading, word.bit_count())
        return 8 * reading

    def columns_of(rows, width):
        columns = [0] * width
        for r, row in enumerate(rows):
            for c in _set_bits(row):
                columns[c] |= 1 << r
        return columns

    def component_reading(rows, width, diagnostic, trials=24):
        """Genuine kernel witnesses from authoritative component columns.

        Localized enumeration is exact only in the inspected subspace
        when its dimension is <=16. Every other reading is sampled.
        Coordinate restrictions need not be invariant group subspaces:
        XORing the actual columns still produces valid expanded words.
        """
        columns = columns_of(rows, width)

        def basis_of(cols, order):
            pivots, basis = {}, []
            for c in order:
                syndrome, word = cols[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
            return basis

        def probe(cols, repetitions):
            reading = len(cols) + 1
            order = list(range(len(cols)))
            for _ in range(repetitions):
                diagnostic.shuffle(order)
                basis = basis_of(cols, order)
                if not basis:
                    continue
                short = sorted(basis, key=int.bit_count)[:16]
                reading = min(reading, short[0].bit_count())
                # Short-vs-all pairs retain cancellations missed by
                # random pairs or a short-vs-short-only search.
                for word in short:
                    for other in basis:
                        candidate = word ^ other
                        if candidate:
                            reading = min(reading, candidate.bit_count())
                descent = list(basis)
                diagnostic.shuffle(descent)
                for initial in short[:6]:
                    word = initial
                    for sweep in range(3):
                        changed = False
                        for other in descent:
                            candidate = word ^ other
                            if candidate and (
                                    candidate.bit_count() < word.bit_count()):
                                word, changed = candidate, True
                        reading = min(reading, word.bit_count())
                        if not changed:
                            break
                if len(basis) >= 3:
                    for _ in range(48):
                        a, b, c = diagnostic.sample(basis, 3)
                        candidate = a ^ b ^ c
                        if candidate:
                            reading = min(reading, candidate.bit_count())
            return reading

        reading = probe(columns, trials)
        blocks = [list(range(j*q, (j+1)*q))
                  for j in range(width // q)]
        restrictions = blocks + [
            blocks[a] + blocks[b]
            for a in range(len(blocks))
            for b in range(a+1, len(blocks))
        ]
        for positions in restrictions:
            basis = basis_of(columns, positions)
            if not basis:
                continue
            if len(basis) <= 16:
                word = 0
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length() - 1]
                    reading = min(reading, word.bit_count())
            else:
                short = sorted(basis, key=int.bit_count)[:16]
                for word in short:
                    reading = min(reading, word.bit_count())
                    for other in basis:
                        candidate = word ^ other
                        if candidate:
                            reading = min(reading, candidate.bit_count())

        if trials >= 32:
            # Central C2 pairs: identical u,v, opposite z.
            central = [
                columns[block*q + t] ^ columns[block*q + 56 + t]
                for block in range(width // q) for t in range(56)
            ]
            d = probe(central, 4)
            if d <= len(central):
                reading = min(reading, 2*d)

            # Seven additive planes of F8. These partitions are explicit
            # coordinate restrictions, never separately implemented lifts.
            planes = sorted({
                tuple(sorted((0, a, b, a ^ b)))
                for a in range(1, 8) for b in range(a+1, 8)
            })
            for plane in planes:
                remaining = set(range(8))
                orbits = []
                while remaining:
                    root = min(remaining)
                    orbit = [root ^ u for u in plane]
                    remaining.difference_update(orbit)
                    orbits.append(orbit)
                restricted = []
                for block in range(width // q):
                    for z in range(2):
                        for v in range(7):
                            for orbit in orbits:
                                syndrome = 0
                                for u in orbit:
                                    syndrome ^= columns[
                                        block*q + (7*z + v)*8 + u]
                                restricted.append(syndrome)
                d = probe(restricted, 2)
                if d <= len(restricted):
                    reading = min(reading, 4*d)
        return reading

    def normal_logical_reading(hx, hz, diagnostic, trials=48):
        """Search fixed-subspace logicals, verifying actual nontriviality.
        Quotient homology alone is insufficient in characteristic two:
        every retained witness is tested modulo the ORIGINAL span."""
        reading = n + 1
        fw = n // 8
        for opposite, stabilizers in ((hz, hx), (hx, hz)):
            qchecks = project(opposite, n)
            columns = columns_of(qchecks, fw)
            stabilizer_pivots = pivots_of(stabilizers)
            signatures = [
                remainder(255 << (8*c), stabilizer_pivots)
                for c in range(fw)
            ]
            # Only actual stabilizers are allowed in the descent.
            qstabs = [
                row for row in project(stabilizers, n)
                if row and not remainder(expand(row), stabilizer_pivots)
            ]
            for _ in range(trials):
                order = list(range(fw))
                diagnostic.shuffle(order)
                pivots = {}
                witnesses = []
                for c in order:
                    syndrome = columns[c]
                    word = 1 << c
                    signature = signatures[c]
                    while syndrome:
                        p = syndrome.bit_length() - 1
                        old = pivots.get(p)
                        if old is None:
                            pivots[p] = (syndrome, word, signature)
                            break
                        syndrome ^= old[0]
                        word ^= old[1]
                        signature ^= old[2]
                    if not syndrome and signature:
                        witnesses.append(word)
                witnesses.sort(key=int.bit_count)
                descent = list(qstabs)
                diagnostic.shuffle(descent)
                for word in witnesses[:3]:
                    for sweep in range(2):
                        changed = False
                        for row in descent:
                            candidate = word ^ row
                            if candidate.bit_count() < word.bit_count():
                                word = candidate
                                changed = True
                        if not changed:
                            break
                    weight = 8 * word.bit_count()
                    if weight >= reading:
                        continue
                    actual = expand(word)
                    if any((actual & row).bit_count() & 1
                           for row in opposite):
                        continue
                    if not remainder(actual, stabilizer_pivots):
                        continue
                    reading = weight
        return reading

    def cycle_key(ast, bst):
        return (max(ast[0], bst[0]), ast[0] + bst[0],
                max(ast[1], bst[1]), ast[1] + bst[1])

    # Select quotient geometry before refining its translation coordinates.
    # Different projected minima are genuine invariants, not orbit copies.
    pool = []
    for trial in range(128):
        A = _affine_grid(rng, G, sa, phase)
        B = _affine_grid(rng, G, sb, phase_b)
        hx, hz, built_n = build(A, B)
        assert built_n == n
        ar, br = _verified_components(G, hx, hz)
        fa = fixed_minimum(ar, 3 * G.q)
        fb = fixed_minimum(br, 3 * G.q)
        if not fa or not fb:
            continue
        quotient_key = (min(fa, fb), fa + fb)
        if len(pool) == 8 and quotient_key < pool[-1][0]:
            continue
        ast = _component_stats(ar, 3 * G.q)
        bst = _component_stats(br, 3 * G.q)
        if ast is None or bst is None:
            continue
        ck = cycle_key(ast, bst)
        pool.append((quotient_key, ck, trial, A, B, hx, hz, ar, br))
        pool.sort(key=lambda item: (-item[0][0], -item[0][1],
                                    item[1], item[2]))
        pool = pool[:8]

    finalists = []
    for number, entry in enumerate(pool):
        quotient_key, ck, _, A, B, hx, hz, ar, br = entry
        selected = (ck, A, B, hx, hz, ar, br)
        # Preserve quotient labels and exact restricted kernel minima.
        # Reject refinements that lose the full-relative-generation rule;
        # independent translations are not a common group gauge.
        for _ in range(4):
            AA = [[[(rng.randrange(8), v, z) for u, v, z in cell]
                   for cell in row] for row in A]
            BB = [[[(rng.randrange(8), v, z) for u, v, z in cell]
                   for cell in row] for row in B]
            if not all(_full_relative_cell(G, cell)
                       for grid in (AA, BB) for row in grid for cell in row):
                continue
            xx, zz, built_n = build(AA, BB)
            assert built_n == n
            aa, bb = _verified_components(G, xx, zz)
            ast = _component_stats(aa, 3 * G.q)
            bst = _component_stats(bb, 3 * G.q)
            if ast is None or bst is None:
                continue
            candidate_key = cycle_key(ast, bst)
            if candidate_key < selected[0]:
                selected = (candidate_key, AA, BB, xx, zz, aa, bb)

        ck, A, B, hx, hz, ar, br = selected
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (0 < k and 8*k < n):
            continue
        diagnostic = random.Random(diagnostic_seed + 100003*number)
        da = min(quotient_key[0], component_reading(
            ar, 3 * G.q, diagnostic, trials=12))
        db = min(quotient_key[0], component_reading(
            br, 3 * G.q, diagnostic, trials=12))
        dn = normal_logical_reading(hx, hz, diagnostic)
        # Failure to find a normal-fixed logical is not evidence of
        # enormous distance: cap its secondary tie-break contribution.
        merit = (min(da, db, dn), quotient_key,
                 min(da, db) + min(dn, n // 2),
                 tuple(-x for x in ck), k)
        finalists.append({
            "merit": merit, "A": A, "B": B, "k": k,
            "readings": (da, db, dn), "ar": ar, "br": br,
            "hx": hx, "hz": hz, "quotient": quotient_key, "cycles": ck,
        })
        finalists.sort(key=lambda item: item["merit"], reverse=True)
        del finalists[3:]

    if not finalists:
        raise ValueError("no verified wide-kernel design passed the rate gate")

    # Independently deepen three finalists. Include central and plane
    # restrictions before choosing; new searches only lower readings.
    best = None
    for number, item in enumerate(finalists):
        diagnostic = random.Random(diagnostic_seed + 900001 + 100003*number)
        da, db, dn = item["readings"]
        da = min(da, component_reading(
            item["ar"], 3 * G.q, diagnostic, trials=48))
        db = min(db, component_reading(
            item["br"], 3 * G.q, diagnostic, trials=48))
        dn = min(dn, normal_logical_reading(
            item["hx"], item["hz"], diagnostic, trials=64))
        merit = (min(da, db, dn), item["quotient"],
                 min(da, db) + min(dn, n // 2),
                 tuple(-x for x in item["cycles"]), item["k"])
        if best is None or merit > best[0]:
            best = (merit, item, (da, db, dn))

    merit, item, readings = best
    # One native construction confirms that operation caching introduced
    # no convention/indexing change in any screening matrix.
    hx, hz, built_n = lifted_matrices(G, item["A"], item["B"])
    assert built_n == n == 1456
    assert hx == item["hx"] and hz == item["hz"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
    assert n - gf2_rank(hx) - gf2_rank(hz) == item["k"]
    return {"cells": {"A": item["A"], "B": item["B"]},
            "screen": merit, "k": item["k"],
            "kernel_readings": readings, "design": idx,
            "supports": (sa, sb), "orientation": "wide-multispace-kernel"}


def CELL(rng, G, side, i, j, ctx):
    """Return the retested nonzero wide-design cell without further draws."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A row-weight cap of 12 alone excludes abundant weight-6 logicals: the rolled C37 (2,6)x(2,6) parent had d=6 in all five measured slots, entirely in sector 1, despite k-kb ranging from 0 to 10.
# REFUTED: Cycle-only selection plus a high short-budget reading reliably excludes sub-60 logicals on this affine-field bed: parent slot 40 fell from d2s=70 to d30s=40, with sector-1 X witnesses spanning multiple Acol/Bcol blocks.
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
