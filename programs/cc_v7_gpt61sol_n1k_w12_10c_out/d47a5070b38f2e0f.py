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
CONCEPT = ("C13 semidirect C8 with action s=5, order 104, A=(1,2), B=(3,5): quotient-first selection followed by translation and verified-kernel screening should yield five measured readings >=26 and improve cell 0.150 beyond its recorded fitness 79.5; no distance certificate is claimed.")
GROUP = ("Jump from order-112 AGL(1,8)xC2 to C13 semidirect C8. The action 5 modulo 13 has order four and defines a valid C8 action, with a central involution. Same-fiber ratios have order 13 and odd quotient differences have order eight. The strong affine parent motivates a noncommutative experiment, not an assumed transfer; this order-104 bed is absent from BEDS.")
SHAPES = ("A=(1,2), B=(3,5), q=104 give n=1352, guaranteed k>=208 and rate 2/13=0.153846. Components have 208/520 variables and 104/312 checks. Parent minima occupied sector-1 Acol/Bcol blocks, often one A column across B columns; balance both new A columns and give every B column degree three or four. Exact full ranks enforce [0.125,0.175).")
FILLING = ("Twenty permutation-canonical B profiles have column totals (3,4,4,4,4), row totals (6,6,7), and augmentation rank three. A uses (4,4) or (5,3), giving HX<=12 and HZ<=12. Fresh mixed-fiber labels compete first on verified translation-constant kernels, then cycle geometry and repeated actual component-kernel witnesses. Final translation-constant quantum witnesses are checked modulo the original stabilizer spans. Restricted floors and sampled upper readings are kept distinct.")
HOW_AB = FILLING
FAMILY_NAME = "r015_c13rtimesc8_s5_12x35_orbit_kernel_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 104

def custom_ops(q):
    # x^13=y^8=1, y*x*y^-1=x^5.
    # 5^4=1 modulo 13, hence also 5^8=1.
    if not valid_q(q):
        raise ValueError("C13 semidirect C8 requires order 104")
    powers = (1, 5, 12, 8, 1, 5, 12, 8)
    els = [(a, b) for b in range(8) for a in range(13)]

    def red(g):
        return (int(g[0]) % 13, int(g[1]) % 8)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + powers[b] * c) % 13, (b + d) % 8)

    def inv(g):
        a, b = red(g)
        bi = (-b) % 8
        return ((-powers[bi] * a) % 13, bi)

    def idx(g):
        a, b = red(g)
        return 13 * b + a

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv,
                idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (3, 5)               # (rb, cb)


# Different density allocations, not permutations of one labeling.
# Every column has >=4 terms. Templates 3..8 are paired only with
# template 0: its row/column totals of 6/4 accommodate their density.
_SUPPORT_TEMPLATES = (
    ((2, 2, 2), (2, 2, 2)),  # 0: uniform baseline
    ((3, 2, 2), (2, 2, 2)),  # 1: one reinforced cell
    ((3, 2, 2), (2, 3, 2)),  # 2: balanced reinforcement
    ((3, 2, 2), (3, 2, 2)),  # 3: one degree-six column
    ((3, 3, 2), (2, 2, 3)),  # 4: asymmetric row density
    ((3, 3, 2), (3, 2, 3)),  # 5: distributed dense side
    ((3, 3, 2), (3, 3, 2)),  # 6: two degree-six columns
    ((4, 2, 2), (2, 4, 2)),  # 7: concentrated dense side
    ((4, 2, 2), (2, 3, 3)),  # 8: mixed concentration
    ((3, 3, 1), (2, 2, 3)),  # 9: uneven cells, balanced degrees
    ((4, 1, 2), (1, 4, 2)),  # 10: crossed concentrated cells
    ((3, 3, 1), (2, 1, 3)),  # 11: asymmetric balanced degrees
)
_SUPPORT_PAIRS = (
    (2, 2), (1, 2), (0, 2), (1, 1), (0, 1),
    (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (2, 9), (1, 9), (0, 9), (9, 9),
    (2, 10), (1, 10), (9, 10), (10, 10), (2, 11),
)


def _set_bits(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _affine_grid(rng, supports, phase):
    """Distinct v coordinates make every within-cell ratio non-involutory."""
    grid = []
    for i, counts in enumerate(supports):
        row = []
        for j, count in enumerate(counts):
            vs = list(range(7))
            rng.shuffle(vs)
            flip = rng.randrange(2)
            cell = []
            for t in range(count):
                if phase == 0:
                    z = flip ^ (t & 1)
                elif phase == 1:
                    z = flip ^ ((t & 1) if (i + j) & 1 else 0)
                else:
                    z = rng.randrange(2)
                cell.append((rng.randrange(8), vs[t], z))
            row.append(cell)
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
    return ar, br


def _retired_affine_ctx(rng, G, idx):
    """Unused parent implementation; the C13 constructor below replaces it."""
    # All profiles have columns 4..5 and rows <=7. Profiles 1 and 3
    # add concentration patterns absent from the parent's templates.
    profiles = (
        ((3, 3, 1), (2, 2, 3)),
        ((4, 2, 1), (1, 3, 3)),
        ((3, 3, 1), (2, 1, 3)),
        ((3, 3, 1), (1, 1, 4)),
        ((3, 2, 2), (2, 3, 2)),
    )
    plans = (
        (0,0), (0,1), (1,1), (0,2), (1,2),
        (2,2), (0,3), (1,3), (2,3), (3,3),
        (0,4), (1,4), (2,4), (3,4), (4,4),
        (0,0), (0,1), (1,1), (0,2), (1,2),
    )
    ia, ib = plans[idx % len(plans)]
    sa, sb = profiles[ia], profiles[ib]
    phase = (idx // 5 + idx // 15 + idx // 20) % 3
    phase_b = (phase + (idx // 2) % 2) % 3
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    # This encoding orders the eight translations contiguously.
    assert all(G.idx((u, v, z)) == (7*z + v)*8 + u
               for z in range(2) for v in range(7) for u in range(8))
    n = 13 * G.q
    diagnostic_seed = rng.randrange(1 << 62)

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
        while word:
            p = word.bit_length() - 1
            old = pivots.get(p)
            if old is None:
                return word
            word ^= old
        return 0

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
    for trial in range(64):
        A = _affine_grid(rng, sa, phase)
        B = _affine_grid(rng, sb, phase_b)
        hx, hz, built_n = lifted_matrices(G, A, B)
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

    best = None
    for number, entry in enumerate(pool):
        quotient_key, ck, _, A, B, hx, hz, ar, br = entry
        selected = (ck, A, B, hx, hz, ar, br)
        # These changes preserve v,z and therefore the exact fixed minima;
        # independent u choices alter the actual component Tanner graphs.
        for _ in range(6):
            AA = [[[(rng.randrange(8), v, z) for u, v, z in cell]
                   for cell in row] for row in A]
            BB = [[[(rng.randrange(8), v, z) for u, v, z in cell]
                   for cell in row] for row in B]
            xx, zz, built_n = lifted_matrices(G, AA, BB)
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
        # Protect the weakest detected family first. Exact fixed minima
        # break ties before graph statistics; none is a distance certificate.
        merit = (min(da, db, dn), quotient_key,
                 min(da, db) + dn, tuple(-x for x in ck), k)
        if best is None or merit > best[0]:
            best = (merit, A, B, k, (da, db, dn))

    if best is None:
        raise ValueError("no fixed-kernel screened filling passed rate gate")
    return {"cells": {"A": best[1], "B": best[2]},
            "screen": best[0], "k": best[3],
            "kernel_readings": best[4], "design": idx}


def _c13_profiles():
    """Canonical weighted B graphs, not row/column permutation copies."""
    from itertools import product, permutations, combinations_with_replacement
    options = {
        total: [v for v in product(range(4), repeat=3)
                if sum(v) == total and sum(x > 0 for x in v) >= 2]
        for total in (3, 4)
    }
    representatives = {}
    row_orders = list(permutations(range(3)))
    for four in combinations_with_replacement(options[4], 4):
        for three in options[3]:
            columns = (three,) + four
            totals = tuple(sum(c[i] for c in columns) for i in range(3))
            if sorted(totals) != [6, 6, 7]:
                continue
            augmentation = [
                sum((columns[j][i] & 1) << j for j in range(5))
                for i in range(3)
            ]
            if gf2_rank(augmentation) != 3:
                continue
            key = min(
                tuple(sorted(tuple(c[i] for i in order) for c in columns))
                for order in row_orders
            )
            representatives[key] = key
    profiles = sorted(
        representatives,
        key=lambda p: (sum(x == 0 for c in p for x in c),
                       sum(x == 3 for c in p for x in c), p)
    )
    if len(profiles) < 20:
        raise ValueError("fewer than twenty canonical B designs")
    return [
        [[columns[j][i] for j in range(5)] for i in range(3)]
        for columns in profiles
    ]


def _c13_columns(rows, width):
    columns = [0] * width
    for r, row in enumerate(rows):
        for c in _set_bits(row):
            columns[c] |= 1 << r
    return columns


def _c13_pivots(rows):
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


def _c13_remainder(word, pivots):
    """Canonical linear remainder, including nonpivot coordinates."""
    for p in sorted(pivots, reverse=True):
        if (word >> p) & 1:
            word ^= pivots[p]
    return word


def _c13_project(rows, width):
    """Restrict ALL verified rows to constant-on-13-coset vectors."""
    return list(dict.fromkeys(
        sum((((row >> (13*j)) & 8191).bit_count() & 1) << j
            for j in range(width // 13))
        for row in rows
    ))


def _c13_short_floor(columns, cap=4):
    """Exact capped classical kernel floor, not an upper reading.

    cap=4 returns five if weights 1..4 are excluded.
    cap=6 returns seven if weights 1..6 are excluded.
    """
    singles = set(columns)
    if 0 in singles:
        return 1
    if len(singles) != len(columns):
        return 2
    pairs = set()
    three = four = False
    for i, x in enumerate(columns):
        for y in columns[:i]:
            syndrome = x ^ y
            three |= syndrome in singles
            four |= syndrome in pairs
            pairs.add(syndrome)
    if three:
        return 3
    if four:
        return 4
    if cap == 4:
        return 5
    from itertools import combinations
    triples = set()
    five = six = False
    for a, b, c in combinations(columns, 3):
        syndrome = a ^ b ^ c
        five |= syndrome in pairs
        six |= syndrome in triples
        triples.add(syndrome)
    # Once weights <=4 are absent, these collisions have disjoint
    # endpoints and really give weight-five/six words.
    return 5 if five else (6 if six else 7)


def _c13_fixed_stat(rows, width, side):
    projected = _c13_project(rows, width)
    fw = width // 13
    if side == "B":
        # A capped LOWER floor within this subspace, not full distance.
        return 13 * _c13_short_floor(
            _c13_columns(projected, fw), cap=6)
    pivots = _c13_pivots(projected)
    basis = []
    for free in range(fw):
        if free in pivots:
            continue
        word = 1 << free
        for p, row in sorted(pivots.items()):
            if (word & row).bit_count() & 1:
                word |= 1 << p
        basis.append(word)
    if not basis or len(basis) > 11:
        return 0
    minimum, word = fw + 1, 0
    for t in range(1, 1 << len(basis)):
        word ^= basis[(t & -t).bit_length() - 1]
        minimum = min(minimum, word.bit_count())
    # Exact minimum in the A translation-constant subspace only.
    return 13 * minimum


def _c13_label(rng, support, side, card, cohort):
    grid = []
    mode = (card + cohort) % 4
    rotation_fraction = (card // 4) / 4.0
    for counts in support:
        row = []
        for count in counts:
            b = rng.randrange(8)
            if count == 0:
                bs = []
            elif count == 1:
                bs = [b]
            elif side == "A":
                if mode == 0:
                    bs = rng.sample(range(8), count)
                elif mode == 1:
                    others = rng.sample([v for v in range(8) if v != b], 2)
                    bs = [b] * (count - 2) + others
                elif mode == 2:
                    c = (b + rng.choice((1, 3, 5, 7))) % 8
                    bs = [b] * (count - 1) + [c]
                else:
                    bs = [b, (b + 1) % 8]
                    bs += [rng.randrange(8) for _ in range(count - 2)]
            elif count == 2:
                c = (b if rng.random() < rotation_fraction
                     else (b + rng.choice((1, 3, 5, 7))) % 8)
                bs = [b, c]
            else:
                # A nonzero translation and an odd quotient difference
                # generate the whole group, not merely a cyclic fiber.
                bs = [b, b, (b + rng.choice((1, 3, 5, 7))) % 8]
            cell = []
            for v in sorted(set(bs)):
                cell.extend((a, v) for a in
                            rng.sample(range(13), bs.count(v)))
            row.append(cell)
        grid.append(row)
    return grid


def _c13_retranslate(rng, grid):
    result = []
    for row in grid:
        out = []
        for cell in row:
            bs = [b for a, b in cell]
            out.append([
                (a, b)
                for b in sorted(set(bs))
                for a in rng.sample(range(13), bs.count(b))
            ])
        result.append(out)
    return result


def _c13_component(G, grid, side):
    """Identity probes use the scaffold's exact left/right conventions."""
    unit = [[[G.e]]]
    width = len(grid[0]) * G.q
    if side == "A":
        hx, _, _ = lifted_matrices(G, grid, unit)
        rows = hx
    else:
        _, hz, _ = lifted_matrices(G, unit, grid)
        rows = hz
    mask = (1 << width) - 1
    return [row & mask for row in rows], width


def _c13_query(columns, seed, trials):
    """Repeated genuine null words; every reading is an upper bound."""
    diagnostic = random.Random(seed)
    width = len(columns)
    reading = width + 1
    retained = []
    for _ in range(trials):
        order = list(range(width))
        diagnostic.shuffle(order)
        pivots, words = {}, []
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
            if syndrome:
                continue
            for old in retained:
                candidate = word ^ old
                if candidate and candidate.bit_count() < word.bit_count():
                    word = candidate
            words.append(word)
            reading = min(reading, word.bit_count())
        best_pair = None
        for i, word in enumerate(words):
            for other in words[:i]:
                candidate = word ^ other
                if candidate and candidate.bit_count() < reading:
                    reading = candidate.bit_count()
                    best_pair = candidate
        if best_pair is not None:
            words.append(best_pair)
        retained = sorted(set(retained + words),
                          key=int.bit_count)[:24]
        if reading <= 4:
            break
    return reading


def _c13_pool(rng, G, support, side, card, cohort, seed):
    """Select quotient geometry, then independently refine translations."""
    pool = []
    for trial in range(40):
        grid = _c13_label(rng, support, side, card, cohort)
        rows, width = _c13_component(G, grid, side)
        fixed = _c13_fixed_stat(rows, width, side)
        if not fixed:
            continue
        if len(pool) == 4 and fixed < pool[-1]["fixed"]:
            continue
        cycles = _component_stats(rows, width)
        if cycles is None:
            continue
        item = dict(grid=grid, rows=rows, width=width,
                    fixed=fixed, cycles=cycles)
        pool.append(item)
        pool.sort(key=lambda r: (-r["fixed"], r["cycles"]))
        del pool[4:]
    if not pool:
        raise ValueError("no connected quotient-screened component")

    for item in pool:
        original = item["grid"]
        for _ in range(6):
            grid = _c13_retranslate(rng, original)
            rows, width = _c13_component(G, grid, side)
            cycles = _component_stats(rows, width)
            if cycles is not None and cycles < item["cycles"]:
                item.update(grid=grid, rows=rows, cycles=cycles)
        columns = _c13_columns(item["rows"], item["width"])
        item["columns"] = columns
        item["floor"] = _c13_short_floor(columns)
        item["d"] = _c13_query(columns, seed, 4)
        if item["floor"] <= 4:
            item["d"] = min(item["d"], item["floor"])

    def quality(item):
        return (item["floor"], item["d"], item["fixed"],
                tuple(-c for c in item["cycles"]))

    pool.sort(key=quality, reverse=True)
    finalists = pool[:2]
    for item in finalists:
        # Two further independent probes can only lower the reading.
        item["d"] = min(
            item["d"],
            _c13_query(item["columns"], seed + 101, 12),
            _c13_query(item["columns"], seed + 211, 24)
        )
    finalists.sort(key=quality, reverse=True)
    return finalists


def _c13_normal_logical(hx, hz, n, seed):
    """Actual quantum witnesses in the constant-on-coset subspace.

    Canonical original-span remainders are linear. A nonzero signature
    therefore certifies nontriviality modulo the ORIGINAL stabilizers.
    """
    diagnostic = random.Random(seed)
    fw = n // 13
    reading = n + 1
    for opposite, stabilizers in ((hz, hx), (hx, hz)):
        projected = _c13_project(opposite, n)
        columns = _c13_columns(projected, fw)
        span = _c13_pivots(stabilizers)
        signatures = [
            _c13_remainder(8191 << (13*c), span) for c in range(fw)
        ]
        for _ in range(16):
            order = list(range(fw))
            diagnostic.shuffle(order)
            pivots, words = {}, []
            for c in order:
                syndrome, word, signature = (
                    columns[c], 1 << c, signatures[c])
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word, signature)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                    signature ^= old[2]
                if not syndrome:
                    words.append((word, signature))
                    if signature:
                        reading = min(reading, 13 * word.bit_count())
            for i, (word, signature) in enumerate(words):
                for other, other_signature in words[:i]:
                    if signature ^ other_signature:
                        reading = min(
                            reading, 13 * (word ^ other).bit_count())
    return reading


def NEW_CTX(rng, G, idx):
    """Twenty canonical support cards with fresh screened group labels."""
    card, cohort = idx % 20, idx // 20
    catalog = _c13_profiles()
    sb = catalog[card]
    sa = [[4, 4]] if card < 12 else [[5, 3]]
    assert G.q == 104
    assert _grid_weights(sa, sb) <= min(12, W_MAX)
    assert _no_dead_lines(sa, sb)
    assert all(0 <= count <= 5 for grid in (sa, sb)
               for row in grid for count in row)
    assert all(G.idx((a, b)) == 13*b+a
               for b in range(8) for a in range(13))
    seed = rng.randrange(1 << 62)
    aa = _c13_pool(rng, G, sa, "A", card, cohort, seed + 17)
    bb = _c13_pool(rng, G, sb, "B", card, cohort, seed + 31)
    best = None
    for number, (a, b) in enumerate(
            (a, b) for a in aa for b in bb):
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (n == 1352 and k > 0
                and 125*n <= 1000*k < 175*n):
            continue
        dn = _c13_normal_logical(hx, hz, n, seed + 1009*number)
        detected = min(a["d"], b["d"], dn)
        merit = (
            min(a["floor"], b["floor"]),
            k * detected * detected,
            detected, a["d"] + b["d"],
            tuple(-a["cycles"][t]-b["cycles"][t] for t in range(2))
        )
        if best is None or merit > best[0]:
            best = (merit, A, B, k, hx, hz,
                    (a["d"], b["d"], dn),
                    (a["fixed"], b["fixed"]))
    if best is None:
        raise ValueError("no screened pair passed the 0.150 exact-rate gate")
    assert css_ok(best[4], best[5])
    assert max(row.bit_count() for row in best[4] + best[5]) <= min(12, W_MAX)
    return {
        "cells": {"A": best[1], "B": best[2]},
        "k": best[3], "screen": best[0], "design": idx,
        "kernel_readings": best[6],
        # A statistic is exact in its restricted subspace; B is capped.
        "fixed_subspace_statistics": best[7]
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the full selected C13-semidir-C8 cell without randomization."""
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
