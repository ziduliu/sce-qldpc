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
CONCEPT = ("C37 semidirect_10 C3, order 111, A=B=(3,2): excluding detected normal-fixed cokernel words below 74 and selecting independently challenged, stabilizer-reduced logical witnesses should raise cell-0.100 fitness above 346.9964. All sampled readings remain upper readings, not distance certificates.")
GROUP = ("Jump from AGL(1,8)xC2 of order 112 to the genuinely different order-111 faithful semidirect product. Its tall bed reached 345.5 and its wide bed 347.0. Since 10^3=1 mod 37, the group law is associative. Normal-C37 and complement-C3 restrictions expose additional genuine kernel families using authoritative columns.")
SHAPES = ("Keep equal tall (3,2) sides: q=111 gives n=1443, k>=111 and guaranteed rate 1/13. Raw components have 222 variables and 333 checks; their relevant transpose codes have 333 coordinates and 222 checks. Parent minima occupied sector-2 strips, especially every Arow at Brow1 and every Brow at Arow2. Challenge both transpose kernels and enforce exact k/n<0.125.")
FILLING = ("Twenty canonical support pairs retain column totals seven and row totals four/five, hence both CSS maxima twelve. Compare native bidirectional Sidon cells with pure/mixed binomial and larger-cell quotient occupancies. Enumerate normal-fixed kernels, reject detected minima below 74, challenge complement-fixed and unrestricted kernels, retain witnesses across independent retests, and repair implicated rows. Complete pairings use exact k and original-span-verified sector-2 logicals reduced by stabilizers. Neither cycles nor missed witnesses certify distance.")
HOW_AB = FILLING
FAMILY_NAME = "r010_C37_C3_action10_32x32_native_subgroup_logical_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("C37 semidirect_10 C3 requires order 111")
    # (a,b)=x^a y^b; y*x*y^-1=x^10.
    # 10^2=26 and 10^3=1 modulo 37.
    powers = (1, 10, 26)
    els = [(a,b) for b in range(3) for a in range(37)]

    def red(g):
        return (int(g[0]) % 37, int(g[1]) % 3)

    def mul(g, h):
        a,b = red(g)
        c,d = red(h)
        return ((a+powers[b]*c) % 37, (b+d) % 3)

    def inv(g):
        a,b = red(g)
        bi = (-b) % 3
        return ((-powers[bi]*a) % 37, bi)

    def idx(g):
        a,b = red(g)
        return 37*b+a

    return dict(elements=els, ident=(0,0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 2)               # (ra, ca)
B_SHAPE = (3, 2)               # (rb, cb)


def _dual_support_design(idx):
    """Twenty canonical count pairs, not row/column permutation copies.

    Enumerate the dual 2x3 count grids, then transpose them into the
    submitted 3x2 shapes. Both tall columns have seven terms and
    every tall row has four or five. Counts include legal zeros/fives.
    """
    from itertools import product, permutations

    options = [row for row in product(range(6), repeat=3)
               if sum(row) == 7]
    permutations_3 = list(permutations(range(3)))
    profiles = set()
    for upper in options:
        for lower in options:
            rows = (upper, lower)
            if sorted(upper[j] + lower[j] for j in range(3)) != [4, 5, 5]:
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
        sum(value == 0 for grid in pair for value in grid),
        sum((value - 2)**2 for grid in pair for value in grid),
        pair,
    ))
    if len(plans) < 20:
        raise ValueError("insufficient canonical tall support pairs")
    a, b = plans[idx % 20]

    def tall(flat):
        return [[flat[j], flat[3 + j]] for j in range(3)]

    return tall(a), tall(b)


def _set_bits(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _affine_grid(rng, supports, phase):
    """Fresh labels; every binomial has relative order exactly fourteen."""
    grid = []
    for i, counts in enumerate(supports):
        row = []
        for j, count in enumerate(counts):
            vs = rng.sample(range(7), count)
            flip = rng.randrange(2)
            if count == 0:
                zs = []
            elif count == 1:
                zs = [flip]
            elif count == 2:
                # Nonzero multiplicative difference gives order seven;
                # opposite central coordinates double it to fourteen.
                zs = [flip, flip ^ 1]
            else:
                if phase == 0:
                    ones = rng.randrange(1, count)
                elif phase == 1:
                    ones = count // 2
                else:
                    ones = 1
                zs = [1] * ones + [0] * (count - ones)
                rng.shuffle(zs)
                zs = [z ^ flip for z in zs]
            row.append([(rng.randrange(8), v, z)
                        for v, z in zip(vs, zs)])
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

    def transpose_verified(rows, width):
        # Ordinary binary transpose of an extracted authoritative matrix;
        # no new left/right group-action convention is implemented.
        dual = [0] * width
        for check, row in enumerate(rows):
            for variable in _set_bits(row):
                dual[variable] |= 1 << check
        return dual

    # Tall components carry their useful logical families in the cokernel.
    # Each returned dual has 224 rows and 336 columns on this bed.
    return (transpose_verified(ar, ca * q),
            transpose_verified(br, cb * q))


def _legacy_affine_ctx(rng, G, idx):
    """Unused historical affine screen; NEW_CTX below replaces it."""
    sa, sb = _dual_support_design(idx)
    phase = (idx // 20 + idx // 5) % 3
    phase_b = (phase + idx % 3) % 3
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sum(row) in (4, 5) for grid in (sa, sb) for row in grid)
    assert all(sum(row[j] for row in grid) == 7
               for grid in (sa, sb) for j in range(2))
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
        """Actual classical kernel words; sampled upper bounds only."""
        columns = columns_of(rows, width)
        reading = width + 1
        for _ in range(trials):
            order = list(range(width))
            diagnostic.shuffle(order)
            pivots = {}
            words = []
            for c in order:
                syndrome, word = columns[c], 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    words.append(word)
                    reading = min(reading, word.bit_count())
            if len(words) >= 2:
                for _ in range(160):
                    a, b = diagnostic.sample(words, 2)
                    reading = min(reading, (a ^ b).bit_count())
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
    for trial in range(96):
        A = _affine_grid(rng, sa, phase)
        B = _affine_grid(rng, sb, phase_b)
        hx, hz, built_n = build(A, B)
        assert built_n == n
        ar, br = _verified_components(G, hx, hz)
        fa = fixed_minimum(ar, 3 * G.q)
        fb = fixed_minimum(br, 3 * G.q)
        if not fa or not fb:
            continue
        quotient_key = (min(fa, fb), fa + fb)
        if len(pool) == 6 and quotient_key < pool[-1][0]:
            continue
        ast = _component_stats(ar, 3 * G.q)
        bst = _component_stats(br, 3 * G.q)
        if ast is None or bst is None:
            continue
        ck = cycle_key(ast, bst)
        pool.append((quotient_key, ck, trial, A, B, hx, hz, ar, br))
        pool.sort(key=lambda item: (-item[0][0], -item[0][1],
                                    item[1], item[2]))
        pool = pool[:6]

    finalists = []
    for number, entry in enumerate(pool):
        quotient_key, ck, _, A, B, hx, hz, ar, br = entry
        selected = (ck, A, B, hx, hz, ar, br)
        # Preserve quotient labels and exact fixed cokernel minima while
        # changing actual translation geometry, not applying a common gauge.
        for _ in range(8):
            AA = [[[(rng.randrange(8), v, z) for u, v, z in cell]
                   for cell in row] for row in A]
            BB = [[[(rng.randrange(8), v, z) for u, v, z in cell]
                   for cell in row] for row in B]
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
        da = component_reading(ar, 3 * G.q, diagnostic)
        db = component_reading(br, 3 * G.q, diagnostic)
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
        del finalists[2:]

    if not finalists:
        raise ValueError("no verified cokernel design passed the rate gate")

    # Independently deepen both finalists instead of submitting the winner
    # of one optimistic diagnostic. New searches can only lower readings.
    best = None
    for number, item in enumerate(finalists):
        diagnostic = random.Random(diagnostic_seed + 900001 + 100003*number)
        da, db, dn = item["readings"]
        da = min(da, component_reading(
            item["ar"], 3 * G.q, diagnostic, trials=32))
        db = min(db, component_reading(
            item["br"], 3 * G.q, diagnostic, trials=32))
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
            "supports": (sa, sb), "orientation": "tall-cokernel"}


def NEW_CTX(rng, G, idx):
    """Native metacyclic cokernels with binding witness/logical challenges."""
    import math

    q = G.q
    assert q == 111 and A_SHAPE == B_SHAPE == (3, 2)
    width, mask = 3*q, (1 << q)-1
    sa, sb = _dual_support_design(idx)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    seed = rng.randrange(1 << 62)
    card, cohort = idx % 20, idx // 20
    policy_a = (card + cohort) % 4
    policy_b = (card//4 + card + 1 + cohort) % 4

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = ident

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()
    # Authoritative transposed monomial blocks; dummy sides are probes only.
    monomials, zero = [[[g]] for g in range(q)], [[[]]]
    _, az, _ = lifted_matrices(probe, monomials, zero)
    bx, _, _ = lifted_matrices(probe, zero, monomials)
    assert len(az) == len(bx) == q
    blocks = {
        side: [[(row >> ((1+g)*q)) & mask for row in source]
               for g in range(q)]
        for side, source in (("A", az), ("B", bx))
    }

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for c in _set_bits(word):
            value ^= columns[c]
        return value

    def basis_of(columns, order):
        pivots, basis = {}, []
        for c in order:
            value, word = columns[c], 1 << c
            while value:
                lead = value.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    normal_masks = [
        ((1 << 37)-1) << (j*q+37*b)
        for j in range(3) for b in range(3)
    ]
    complement_masks = {}
    subgroup = [G.idx((0,b)) for b in range(3)]
    for side in ("A", "B"):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {
                table[root][h] if side == "A" else table[h][root]
                for h in subgroup
            }
            assert len(orbit) == 3 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(sorted(orbit))
        complement_masks[side] = [
            sum(1 << (j*q+c) for c in orbit)
            for j in range(3) for orbit in orbits
        ]

    def expand(word, masks):
        actual = 0
        for c in _set_bits(word):
            actual ^= masks[c]
        return actual

    occupancies = (
        {3: (1,1,1), 4: (2,1,1), 5: (3,1,1)},
        {3: (2,1,0), 4: (3,1,0), 5: (3,2,0)},
        {3: (1,1,1), 4: (3,1,0), 5: (2,2,1)},
        {3: (2,1,0), 4: (2,1,1), 5: (3,1,1)},
    )

    def sidon(cell):
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        wanted = len(cell)*(len(cell)-1)
        return (len(set(cell)) == len(cell)
                and len(left) == len(right) == wanted)

    def fresh_cell(count, policy, force_mixed=False):
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        mixed = force_mixed or policy in (1,3)
        occupancy = (((1,1,0) if mixed else (2,0,0))
                     if count == 2 else occupancies[policy][count])
        for _ in range(2048):
            fibers = rng.sample(range(3), 3)
            cell = []
            for b, size in zip(fibers, occupancy):
                cell.extend(G.idx((a,b))
                            for a in rng.sample(range(37), size))
            if sidon(cell):
                return cell
        raise ValueError("native Sidon cell recipe exhausted")

    def fresh(support, policy):
        return [[fresh_cell(count, policy, counts == [2,2])
                 for count in counts] for counts in support]

    def inspect(grid, side):
        mono = blocks[side]
        rows = []
        for j in range(2):
            for r in range(q):
                row = 0
                for i in range(3):
                    value = 0
                    for g in grid[i][j]:
                        value ^= mono[g][r]
                    row |= value << (i*q)
                rows.append(row)
        columns = [0]*width
        for r, row in enumerate(rows):
            for c in _set_bits(row):
                columns[c] |= 1 << r
        if not all(columns) or len(set(columns)) != width:
            return None

        restricted = [syndrome(columns, word) for word in normal_masks]
        basis = basis_of(restricted, range(9))
        if not basis:
            return None
        fixed_words, word = [], 0
        for t in range(1, 1 << len(basis)):
            word ^= basis[(t & -t).bit_length()-1]
            fixed_words.append(expand(word, normal_masks))
        fixed_words.sort(key=lambda w: (w.bit_count(), w))
        if fixed_words[0].bit_count() < 74:
            return None
        stats = _component_stats(rows, width)
        if stats is None:
            return None
        return {
            "grid": copy_grid(grid), "rows": rows, "columns": columns,
            "side": side, "cycles": stats,
            "cost": 700*stats[0] + (1+card % 3)*stats[1],
            "words": set(fixed_words[:24]),
            "d": fixed_words[0].bit_count(),
            "signature": tuple(tuple(sorted(cell))
                               for row in grid for cell in row),
        }

    def challenge(record, trials, turn):
        """Actual kernel words; independent searches only lower readings."""
        diagnostic = random.Random(
            seed + 100003*int(record["side"] == "B") + 700001*turn)
        columns = record["columns"]
        retained = sorted(record["words"], key=int.bit_count)[:28]

        def observe(word):
            if word:
                record["d"] = min(record["d"], word.bit_count())

        def search(cols, repetitions, masks=None):
            order = list(range(len(cols)))
            short_bank = []
            for trial in range(repetitions):
                diagnostic.shuffle(order)
                if masks is None and retained and trial % 3 == 1:
                    active = list(_set_bits(retained[0]))
                    front = diagnostic.sample(active, max(1, len(active)//2))
                    chosen = set(front)
                    order = front + [c for c in order if c not in chosen]
                basis = basis_of(cols, order)
                if masks is not None:
                    basis = [expand(word, masks) for word in basis]
                light = sorted(set(basis+short_bank+retained),
                               key=lambda w: (w.bit_count(), w))[:16]
                additions = []
                for word in light:
                    observe(word)
                    for other in basis:
                        candidate = word ^ other
                        observe(candidate)
                        if candidate and candidate.bit_count() <= record["d"]+4:
                            additions.append(candidate)
                moves = basis+short_bank
                diagnostic.shuffle(moves)
                for initial in light[:4]:
                    word = initial
                    for sweep in range(3):
                        changed = False
                        for other in moves:
                            candidate = word ^ other
                            if candidate and candidate.bit_count() < word.bit_count():
                                word, changed = candidate, True
                        observe(word)
                        if not changed:
                            break
                    additions.append(word)
                short_bank = sorted(set(short_bank+light+additions),
                                    key=lambda w: (w.bit_count(), w))[:24]
            return short_bank

        additions = search(columns, trials)
        parts = complement_masks[record["side"]]
        compressed = [syndrome(columns, word) for word in parts]
        additions += search(compressed, max(3, trials//4), parts)
        # Include manageable one/two-block restrictions.
        if turn >= 100:
            for omitted in range(3):
                positions = [c for c in range(width) if c//q != omitted]
                basis = basis_of(columns, positions)
                if len(basis) <= 14:
                    word = 0
                    for t in range(1, 1 << len(basis)):
                        word ^= basis[(t & -t).bit_length()-1]
                        observe(word)
                        if word.bit_count() <= record["d"]+4:
                            additions.append(word)
                else:
                    additions += sorted(basis, key=int.bit_count)[:8]
        record["words"].update(additions)
        record["words"] = set(sorted(
            record["words"], key=lambda w: (w.bit_count(), w))[:40])
        for word in record["words"]:
            assert syndrome(columns, word) == 0
            observe(word)

    def component_pool(support, side, policy):
        bank = {}
        known = set()

        def remember(record):
            if record is None:
                return
            bank[record["signature"]] = record
            if len(bank) > 12:
                worst = max(bank, key=lambda key: bank[key]["cost"])
                del bank[worst]

        for _ in range(40):
            remember(inspect(fresh(support, policy), side))
        if not bank:
            # A failed preferred occupancy is not allowed to void the card.
            for _ in range(80):
                remember(inspect(fresh(support, 3), side))
        if not bank:
            raise ValueError("normal-fixed gate exhausted native starts")
        current = min(bank.values(), key=lambda r: r["cost"])
        positions = [(i,j) for i in range(3) for j in range(2)
                     if support[i][j]]
        for step in range(64):
            grid = copy_grid(current["grid"])
            i,j = rng.choice(positions)
            grid[i][j] = fresh_cell(
                support[i][j], policy, support[i] == [2,2])
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            delta = proposed["cost"]-current["cost"]
            temperature = q*(10*(1-step/64.0)**2+0.2)
            if delta <= 0 or rng.random() < math.exp(
                    min(0.0, -delta/temperature)):
                current = proposed
                remember(current)

        selected = sorted(bank.values(), key=lambda r: r["cost"])[:6]
        for number, record in enumerate(selected):
            challenge(record, 8, number+1)
            known.update(record["words"])

        def merit(record):
            for word in known:
                if (word.bit_count() < record["d"]
                        and syndrome(record["columns"], word) == 0):
                    record["words"].add(word)
                    record["d"] = word.bit_count()
            return (record["d"], -record["cost"],
                    gf2_rank(record["rows"]))

        current = max(selected, key=merit)
        for step in range(12):
            word = min(current["words"], key=int.bit_count)
            implicated = sorted({c//q for c in _set_bits(word)})
            i = rng.choice(implicated)
            grid = copy_grid(current["grid"])
            for j in range(2):
                grid[i][j] = fresh_cell(
                    support[i][j], (policy+step//4) % 4,
                    support[i] == [2,2])
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            challenge(proposed, 8, 30+step)
            known.update(proposed["words"])
            selected.append(proposed)
            if merit(proposed) > merit(current):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists = selected[:3]
        for number, record in enumerate(finalists):
            challenge(record, 32, 100+number)
            challenge(record, 24, 200+number)
            known.update(record["words"])
        finalists.sort(key=merit, reverse=True)
        return finalists

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_readings(a, b, hx, hz, n):
        """Verified sector-2 embeddings, reduced in original logical classes."""
        readings = []
        for kind, record, opposite, stabilizers in (
                ("X", a, hz, hx), ("Z", b, hx, hz)):
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(record["words"], key=int.bit_count)[:5]:
                for fixed in range(3):
                    if kind == "X":
                        actual = sum(
                            ((word >> (i*q)) & mask)
                            << ((4+3*i+fixed)*q) for i in range(3))
                    else:
                        actual = word << ((4+3*fixed)*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n+1
            diagnostic = random.Random(seed+3100009+int(kind == "Z"))
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual, changed = candidate, True
                    nearby = sorted(
                        stabilizers, key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            candidate = actual ^ first ^ second
                            if candidate.bit_count() < actual.bit_count():
                                actual, changed = candidate, True
                    if not changed:
                        break
                assert outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa = component_pool(sa, "A", policy_a)
    bb = component_pool(sb, "B", policy_b)
    best = None
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(probe, a["grid"], b["grid"])
            k = n-gf2_rank(hx)-gf2_rank(hz)
            if not (n == 1443 and k >= q and 8*k < n):
                continue
            ar, br = _verified_components(probe, hx, hz)
            assert ar == a["rows"] and br == b["rows"]
            logicals = logical_readings(a, b, hx, hz, n)
            detected = min(a["d"], b["d"], *logicals)
            key = (k*detected*detected, detected,
                   a["d"]+b["d"], -a["cost"]-b["cost"])
            if best is None or key > best[0]:
                best = (key, A, B, hx, hz, k, a, b, logicals)
    if best is None:
        raise ValueError("native subgroup/logical pairings missed rate gate")
    key, A, B, hx, hz, k, a, b, logicals = best
    native_x, native_z, n = lifted_matrices(G, A, B)
    assert native_x == hx and native_z == hz and n == 1443
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
    assert all(len(cell) == len(set(cell))
               for grid in (A,B) for row in grid for cell in row)
    return {
        "cells": {"A": A, "B": B}, "k": k, "design": idx,
        "supports": (sa,sb), "policies": (policy_a,policy_b),
        "component_upper_readings": (a["d"],b["d"]),
        "verified_logical_upper_readings": logicals,
        "component_cycles": (a["cycles"],b["cycles"]),
        "selection_quality": key,
        "orientation": "native-tall-subgroup-logical-repair",
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete independently challenged metacyclic design cell."""
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
