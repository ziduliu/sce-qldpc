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
CONCEPT = ("AGL(1,8) x C2, order 112, A=B=(3,2): dense support cards with binding anchored min-sum/OSD and localized kernel challenges should raise five-code fitness above 351.5690 in cell 0.100. All restricted minima and sampled distance readings remain provisional upper bounds, not quantum-distance certificates.")
GROUP = ("Keep the exact validated affine-field group and order 112 byte-for-byte unchanged. This bed leads cell 0.100 at fitness 351.5690, with measured readings 68,68,68,68,66 and k=112 throughout. Native regular component actions permit verified translation of search witnesses to forced anchors. The complement-C7 and central-C2 restrictions remain useful, but their favorable readings do not exclude lighter unrestricted strip words.")
SHAPES = ("Keep both frozen (3,2) shapes: q=112 gives n=1456, k>=112 and guaranteed rate 1/13 in cell 0.100; q_max=115. Equal raw components have 224 variables and 336 checks, while their transpose codes have 336 variables and 224 checks. Parent measured minima were entirely sector-two Arow/Brow strips, including every Arow at Brow1/2 and every Brow at Arow1/2. Challenge both transpose kernels and original-span strip representatives; exact full ranks retain k/n<0.125.")
FILLING = ("Twenty dense cards use positive row degrees (4,5,5), column totals (7,7), and both CSS maxima twelve. Three support pairs cross four symmetric progression/order policies; eight additional cards use two dense pairs with asymmetric A/B policies instead of thirteen-term supports. Fresh native-Sidon labels are screened by exact translation/C7 restrictions, larger independent pools and witness-directed repair. Finalist searches accumulate manageable one/two-block minima, forced-anchor damped min-sum with reliability-ordered syndrome completion, circuit-pair descent, central-fixed words and regular-action cancellations. Every found word is verified on authoritative columns and remains binding. Complete pairings use actual k and original-span logical tests; missed witnesses certify no distance.")
HOW_AB = FILLING
FAMILY_NAME = "r010_affineF8xC2_32x32_dense_anchored_OSD_strip_v8"

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


def NEW_CTX(rng, G, idx):
    """Dense cards with binding anchored and localized kernel challenges."""
    card, cohort = idx % 20, idx // 20
    model = card % 4
    profiles = (
        ((1, 3), (3, 2), (3, 2)),
        ((1, 3), (2, 3), (4, 1)),
    )
    # The measured winners used the dense profiles. Last-eight cards
    # now change A/B relative geometry instead of losing one term.
    plans = ((0, 0), (0, 1), (1, 1), (0, 0), (0, 1))
    pa, pb = plans[card // 4]
    asymmetric = ((0, 1), (0, 2), (1, 3), (2, 3))
    if card < 12:
        policy_a = policy_b = model
    elif card < 16:
        policy_a, policy_b = asymmetric[model]
    else:
        policy_a, policy_b = model, model ^ 1
    sa = [list(row) for row in profiles[pa]]
    sb = [list(row) for row in profiles[pb]]
    assert A_SHAPE == B_SHAPE == (3, 2)
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sorted(map(sum, grid)) == [4, 5, 5]
               for grid in (sa, sb))
    assert all(sum(row[j] for row in grid) == 7
               for grid in (sa, sb) for j in range(2))
    assert all(1 <= count <= 4
               for grid in (sa, sb) for row in grid for count in row)
    assert all(G.idx((u, v, z)) == (7*z + v)*8 + u
               for z in range(2) for v in range(7) for u in range(8))
    n = 13 * G.q
    diagnostic_seed = rng.randrange(1 << 62)
    # Independent actual labels, never a common gauge or archived image.
    rng = random.Random(
        diagnostic_seed ^ (0xA05D32C711 + 1000003*idx))

    def legal_cell(cell):
        """Native difference tests, not a complete Tanner-girth claim."""
        count = len(cell)
        if len(set(cell)) != count:
            return False
        left = {G.mul(G.inv(g), h)
                for g in cell for h in cell if g != h}
        right = {G.mul(g, G.inv(h))
                 for g in cell for h in cell if g != h}
        if len(left) != count*(count-1) or len(right) != count*(count-1):
            return False
        if count < 3:
            return True
        ratios = [G.mul(G.inv(cell[0]), g) for g in cell[1:]]
        return (any(g[2] for g in ratios) and
                any(G.mul(g, h) != G.mul(h, g)
                    for t, g in enumerate(ratios) for h in ratios[:t]))

    def fresh_cell(count, policy):
        """The four cards change element geometry, not a common gauge."""
        for _ in range(1024):
            vs = rng.sample(range(7), count)
            if count in (3, 4):
                tri = (vs if count == 3 else
                       [v for v in range(7) if v not in vs])
                progression = any(
                    (tri[a]+tri[b]-2*tri[c]) % 7 == 0
                    for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 2, 0)))
                if progression != bool(policy & 1):
                    continue
            flip = rng.randrange(2)
            if count == 1:
                zs = [flip]
            elif count == 2:
                # Distinct v gives affine relative order seven.
                # Equal z retains seven; opposite z gives fourteen.
                zs = [flip, flip if policy >= 2 else flip ^ 1]
            else:
                ones = (count // 2 if (policy+cohort) % 2 == 0 else 1)
                zs = [1]*ones + [0]*(count-ones)
                rng.shuffle(zs)
                zs = [z ^ flip for z in zs]
            cell = [(rng.randrange(8), v, z) for v, z in zip(vs, zs)]
            if legal_cell(cell):
                return cell
        raise ValueError("native progression/relative-order cell exhausted")

    def fresh(support, policy):
        return [[fresh_cell(count, policy) for count in row]
                for row in support]

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

    width = 3*q
    zero = [[[], []] for _ in range(3)]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    # Extract transposed monomial blocks from authoritative probes.
    # GF2-linear assembly avoids rebuilding full dummy LP matrices for
    # every proposed component. Final native extraction checks agreement.
    block_mask = (1 << q)-1
    transpose_monomials = [[], []]
    for start in range(0, q, 16):
        ids = list(range(start, min(start+16, q)))
        gallery = [[[g]] for g in ids]
        _, left, _ = lifted_matrices(indexed_group, gallery, [[[]]])
        right, _, _ = lifted_matrices(indexed_group, [[[]]], gallery)
        assert len(left) == len(right) == q
        for position in range(len(ids)):
            for side, source in ((0, left), (1, right)):
                transpose_monomials[side].append([
                    (row >> ((1+position)*q)) & block_mask
                    for row in source])

    def component_rows(grid, side):
        """Only GF2 sums of scaffold-supplied transpose monomial blocks."""
        mono = transpose_monomials[side]
        rows = []
        for j in range(2):
            for r in range(q):
                row = 0
                for i in range(3):
                    block = 0
                    for g in grid[i][j]:
                        block ^= mono[G.idx(g)][r]
                    row |= block << (i*q)
                rows.append(row)
        assert len(rows) == 2*q
        assert all(row >> width == 0 for row in rows)
        return rows

    def syndrome(columns, word):
        value = 0
        for c in _set_bits(word):
            value ^= columns[c]
        return value

    def dependency_basis(columns, order):
        pivots, basis = {}, []
        for c in order:
            value, word = columns[c], 1 << c
            while value:
                p = value.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    # Coordinate partitions of native regular actions only. Every
    # restricted syndrome below is a sum of authoritative columns.
    complement = [G.idx((0, v, 0)) for v in range(7)]
    complement_masks = {}
    for side in (0, 1):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {
                table[root][h] if side == 0 else table[h][root]
                for h in complement
            }
            assert len(orbit) == 7 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(tuple(sorted(orbit)))
        complement_masks[side] = [
            sum(1 << (block*q+c) for c in orbit)
            for block in range(3) for orbit in orbits
        ]
        assert len(complement_masks[side]) == 48

    central_masks = [
        (1 << (block*q+t)) | (1 << (block*q+56+t))
        for block in range(3) for t in range(56)
    ]

    def expand_masks(pattern, masks):
        word = 0
        for c in _set_bits(pattern):
            word ^= masks[c]
        return word

    def complement_word(columns, side):
        """Exact C7-fixed classical minimum, not a quantum certificate."""
        masks = complement_masks[side]
        restricted = [syndrome(columns, word) for word in masks]
        basis = dependency_basis(restricted, range(len(restricted)))
        # Omit exponentially expensive proposals rather than presenting
        # a sampled minimum as an exact restricted result.
        if not basis or len(basis) > 20:
            return None
        pattern, shortest = 0, None
        for t in range(1, 1 << len(basis)):
            pattern ^= basis[(t & -t).bit_length()-1]
            if shortest is None or pattern.bit_count() < shortest.bit_count():
                shortest = pattern
        word = expand_masks(shortest, masks)
        assert word.bit_count() == 7*shortest.bit_count()
        assert syndrome(columns, word) == 0
        return word

    def fixed_word(rows):
        """Exact minimum in the translation-constant restriction only."""
        projected = project(rows, width)
        columns = columns_of(projected, width//8)
        basis = dependency_basis(columns, range(len(columns)))
        if not basis or len(basis) > 18:
            return None
        pattern, shortest = 0, None
        for t in range(1, 1 << len(basis)):
            pattern ^= basis[(t & -t).bit_length()-1]
            if shortest is None or pattern.bit_count() < shortest.bit_count():
                shortest = pattern
        word = expand(shortest)
        assert all((word & row).bit_count() % 2 == 0 for row in rows)
        return word

    def small_penalty(columns):
        """Exact detection through weight four on this classical component."""
        singles, pairs = set(columns), set()
        if 0 in singles or len(singles) != len(columns):
            return 1000000
        hits = 0
        for i, value in enumerate(columns):
            for other in columns[:i]:
                combined = value ^ other
                hits += int(combined in singles or combined in pairs)
                pairs.add(combined)
        return hits

    def move_word(word, side, multiplier):
        """Regular-action images of words, never images submitted as codes."""
        answer = 0
        for c in _set_bits(word):
            block, h = divmod(c, q)
            image = (table[h][multiplier] if side == 0
                     else table[multiplier][h])
            answer |= 1 << (block*q+image)
        return answer

    def anchored_challenge(record, diagnostic, attempts, seeds):
        """Forced-anchor min-sum/OSD witnesses, never an exclusion test.

        Solve H_rest*y=H_anchor. Restoring the excluded anchor produces
        an actual nonzero kernel word. Reliability ordering and circuit
        descent are finite searches, not optimal decoding.
        """
        columns = record["columns"]
        sources = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:16]
        found = []
        ident = G.idx(G.e)
        for attempt in range(attempts):
            block = attempt % 3
            root = block*q+ident
            root_bit = 1 << root
            target = columns[root]
            edge_variables = []
            check_edges = [[] for _ in record["rows"]]
            for c, value in enumerate(columns):
                if c == root:
                    continue
                for r in _set_bits(value):
                    edge = len(edge_variables)
                    edge_variables.append(c)
                    check_edges[r].append(edge)

            channel = [2.0+diagnostic.uniform(-0.8, 0.8)
                       for _ in columns]
            nearby = sorted({
                edge_variables[e]
                for r in _set_bits(target) for e in check_edges[r]
            })
            warm = [word for word in sources
                    if (word >> (block*q)) & ((1 << q)-1)]
            if attempt % 3 == 1 and warm:
                source = diagnostic.choice(warm)
                anchors = list(_set_bits(
                    (source >> (block*q)) & ((1 << q)-1)))
                multiplier = inverses[diagnostic.choice(anchors)]
                image = move_word(source, record["side"], multiplier)
                assert image & root_bit and syndrome(columns, image) == 0
                active = list(_set_bits(image ^ root_bit))
                # Perturb the warm support, rather than just decoding
                # the already known word under a favorable channel.
                for c in active:
                    channel[c] = diagnostic.uniform(0.05, 0.85)
                for c in diagnostic.sample(
                        active, min(len(active), max(1, len(active)//5))):
                    channel[c] = diagnostic.uniform(1.2, 2.5)
            elif attempt % 3 == 2 and nearby:
                for c in diagnostic.sample(nearby, min(3, len(nearby))):
                    channel[c] = -diagnostic.uniform(0.1, 1.0)

            v_messages = [channel[c] for c in edge_variables]
            c_messages = [0.0]*len(edge_variables)
            posterior = list(channel)
            hard = 0
            for iteration in range(18):
                for r, incident in enumerate(check_edges):
                    parity = -1 if (target >> r) & 1 else 1
                    first = second = 24.0
                    smallest = -1
                    for e in incident:
                        value = v_messages[e]
                        if value < 0:
                            parity = -parity
                        magnitude = abs(value)
                        if magnitude < first:
                            second, first = first, magnitude
                            smallest = e
                        elif magnitude < second:
                            second = magnitude
                    for e in incident:
                        sign = -1 if v_messages[e] < 0 else 1
                        magnitude = second if e == smallest else first
                        proposed = 0.8*parity*sign*magnitude
                        c_messages[e] = (
                            0.55*c_messages[e]+0.45*proposed)
                posterior = list(channel)
                for e, c in enumerate(edge_variables):
                    posterior[c] += c_messages[e]
                hard = sum(1 << c for c in range(width)
                           if c != root and posterior[c] < 0)
                if target ^ syndrome(columns, hard) == 0:
                    found.append(hard ^ root_bit)
                    break
                for e, c in enumerate(edge_variables):
                    proposed = posterior[c]-c_messages[e]
                    v_messages[e] = max(-24.0, min(
                        24.0, 0.4*v_messages[e]+0.6*proposed))

            order = [c for c in range(width) if c != root]
            reliability = [
                abs(value)+diagnostic.uniform(0.0, 0.3)
                for value in posterior]
            order.sort(key=lambda c: reliability[c])
            pivots, circuits = {}, []
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
                    circuits.append(word)

            residual = target ^ syndrome(columns, hard)
            repair = 0
            while residual:
                old = pivots.get(residual.bit_length()-1)
                if old is None:
                    break
                residual ^= old[0]
                repair ^= old[1]
            if residual:
                continue
            candidate = hard ^ repair ^ root_bit
            moves = circuits + [w for w in sources if not w & root_bit]
            for sweep in range(3):
                before = candidate.bit_count()
                diagnostic.shuffle(moves)
                for move in moves:
                    proposed = candidate ^ move
                    if proposed.bit_count() < candidate.bit_count():
                        candidate = proposed
                light = sorted(
                    moves, key=lambda w: (candidate ^ w).bit_count())[:24]
                winner = candidate
                for i, first in enumerate(light):
                    partial = candidate ^ first
                    for second in light[:i]:
                        proposed = partial ^ second
                        if proposed.bit_count() < winner.bit_count():
                            winner = proposed
                candidate = winner
                if candidate.bit_count() == before:
                    break
            assert candidate & root_bit
            assert syndrome(columns, candidate) == 0
            found.append(candidate)
        assert all(word and syndrome(columns, word) == 0 for word in found)
        return sorted(set(found), key=lambda w: (w.bit_count(), w))[:12]

    def challenge(record, seed, trials, transferred=()):
        """Binding localized, anchored and orbit-cancellation witnesses."""
        diagnostic = random.Random(seed)
        columns = record["columns"]
        if trials >= 24 and not record.get("localized_checked", False):
            # Enumerate the inspected restriction exactly only when its
            # nullity is manageable. Larger restrictions contribute
            # genuine basis words, never an asserted exact minimum.
            blocks = [list(range(i*q, (i+1)*q)) for i in range(3)]
            restrictions = blocks + [
                blocks[i]+blocks[j]
                for i in range(3) for j in range(i+1, 3)]
            for positions in restrictions:
                basis = dependency_basis(columns, positions)
                if not basis:
                    continue
                shortest = min(basis, key=lambda w: (w.bit_count(), w))
                if len(basis) <= 16:
                    word = 0
                    for t in range(1, 1 << len(basis)):
                        word ^= basis[(t & -t).bit_length()-1]
                        if word.bit_count() < shortest.bit_count():
                            shortest = word
                assert shortest and syndrome(columns, shortest) == 0
                record["words"].add(shortest)
            record["localized_checked"] = True
        retained = set(record["words"])
        retained.update(
            word for word in transferred
            if word and syndrome(columns, word) == 0)
        retained = sorted(retained, key=lambda w: (w.bit_count(), w))[:24]
        reading = min(map(int.bit_count, retained))
        witness = min(retained, key=lambda w: (w.bit_count(), w))

        def observe(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        order = list(range(width))
        for turn in range(trials):
            diagnostic.shuffle(order)
            if turn % 4 == 3 and retained:
                source = diagnostic.choice(retained[:6])
                active = list(_set_bits(source))
                front = diagnostic.sample(active, max(1, len(active)//2))
                chosen = set(front)
                order = front + [c for c in order if c not in chosen]
            basis = dependency_basis(columns, order)
            for word in basis:
                observe(word)
            light = sorted(set(retained + basis),
                           key=lambda w: (w.bit_count(), w))[:12]
            additions = list(light)
            moves = retained + basis
            for word in light:
                for other in moves:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= reading+4:
                        additions.append(candidate)
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
            additions.append(witness)
            retained = sorted(set(retained + additions),
                              key=lambda w: (w.bit_count(), w))[:24]

        if trials >= 24:
            # Independently challenge the central-fixed subspace.
            # All expanded words are genuine kernels of THIS component.
            compressed = [
                syndrome(columns, mask_) for mask_ in central_masks
            ]
            central_order = list(range(len(compressed)))
            central_retained = []
            additions = []
            for _ in range(4):
                diagnostic.shuffle(central_order)
                basis = dependency_basis(compressed, central_order)
                light = sorted(set(central_retained+basis),
                               key=lambda w: (w.bit_count(), w))[:12]
                candidates = list(light)
                for word in light[:6]:
                    for other in basis:
                        candidate = word ^ other
                        if candidate:
                            candidates.append(candidate)
                moves = central_retained+basis
                diagnostic.shuffle(moves)
                for initial in light[:4]:
                    word = initial
                    for sweep in range(3):
                        changed = False
                        for other in moves:
                            candidate = word ^ other
                            if candidate and candidate.bit_count() < word.bit_count():
                                word, changed = candidate, True
                        if not changed:
                            break
                    candidates.append(word)
                central_retained = sorted(
                    set(central_retained+candidates),
                    key=lambda w: (w.bit_count(), w))[:16]
                for pattern in central_retained[:4]:
                    actual = expand_masks(pattern, central_masks)
                    assert syndrome(columns, actual) == 0
                    observe(actual)
                    additions.append(actual)
            additions.append(witness)
            retained = sorted(set(retained+additions),
                              key=lambda w: (w.bit_count(), w))[:24]

        # A regular component symmetry maps genuine kernels to kernels.
        # Verify every image's actual syndrome before using it. Searching
        # their cancellations is distinct from submitting orbit copies.
        sources = retained[:3]
        additions = []
        for source in sources:
            for multiplier in range(q):
                image = move_word(source, record["side"], multiplier)
                assert syndrome(columns, image) == 0
                for other in retained[:12]:
                    candidate = image ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= reading+4:
                        additions.append(candidate)
        if trials >= 24:
            # Independently shaped decoding landscapes challenge the
            # CURRENT matrix; earlier witnesses cannot be overwritten.
            attempts = 6 if trials >= 48 else 4
            for word in anchored_challenge(
                    record, diagnostic, attempts, retained):
                observe(word)
                additions.append(word)
        additions.append(witness)
        record["words"] = set(sorted(
            set(retained + additions),
            key=lambda w: (w.bit_count(), w))[:32])
        record["words"].add(witness)
        record["d"] = min(record["d"], reading)
        record["word"] = min(record["words"],
                             key=lambda w: (w.bit_count(), w))
        assert syndrome(columns, record["word"]) == 0
        return record["word"]

    def select_component(side, support, policy):
        seen, bank = set(), set()
        side_seed = diagnostic_seed+100003*side

        def inspect(grid):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for row in grid for cell in row)
            if signature in seen:
                return None
            seen.add(signature)
            rows = component_rows(grid, side)
            rank = gf2_rank(rows)
            if rank < 2*q-8:
                return None
            word = fixed_word(rows)
            if word is None:
                return None
            columns = columns_of(rows, width)
            cword = complement_word(columns, side)
            if cword is None:
                return None
            stats = _component_stats(rows, width)
            if stats is None:
                return None
            words = {word, cword}
            shortest = min(words, key=lambda w: (w.bit_count(), w))
            return {
                "grid": copy_grid(grid), "rows": rows,
                "columns": columns, "rank": rank,
                "fixed": word.bit_count(),
                "complement": cword.bit_count(), "cycles": stats,
                "d": shortest.bit_count(), "words": words,
                "word": shortest, "signature": signature, "side": side,
            }

        def refresh(record):
            for word in bank:
                if (word.bit_count() < record["d"] and
                        syndrome(record["columns"], word) == 0):
                    record["d"] = word.bit_count()
                    record["word"] = word
                    record["words"].add(word)

        def merit(record):
            refresh(record)
            four, excess = record["cycles"]
            geometry = ((-four, -excess) if model % 2 == 0
                        else (-(700*four+excess), -four))
            return (-record["small"], record["d"],
                    min(record["fixed"], record["complement"]),
                    record["fixed"]+record["complement"],
                    geometry, record["rank"])

        def diagnose(record, trials, turn):
            word = challenge(record, side_seed+700001*turn, trials, bank)
            bank.add(word)
            bank.update(sorted(record["words"],
                               key=lambda w: (w.bit_count(), w))[:4])
            if len(bank) > 64:
                keep = sorted(bank, key=lambda w: (w.bit_count(), w))[:48]
                bank.clear()
                bank.update(keep)
            refresh(record)

        # Independent pools avoid discarding a strong component because
        # its randomly paired partner is weak.
        pool = []
        for trial in range(112):
            record = inspect(fresh(support, policy))
            if record is not None:
                pool.append(record)
                pool.sort(key=lambda r: (
                    -min(r["fixed"], r["complement"]),
                    -r["fixed"]-r["complement"],
                    r["cycles"], -r["rank"], r["signature"]))
                del pool[8:]
            # Wider independent pools offset the stronger finalist
            # challenges. Favor six protected proposals without turning
            # a provisional reading into an infeasible hard distance gate.
            protected = sum(
                min(r["fixed"], r["complement"]) >= 70 for r in pool)
            if trial >= 79 and protected >= 6:
                break
        if not pool:
            raise ValueError("empty positive independent component pool")

        selected = []
        for number, record in enumerate(pool):
            record["small"] = small_penalty(record["columns"])
            diagnose(record, 10, 1+number)
            selected.append(record)

        current = max(selected, key=merit)
        for step in range(18):
            refresh(current)
            active = [word for word in bank
                      if syndrome(current["columns"], word) == 0]
            word = (min(active, key=lambda w: (w.bit_count(), w))
                    if active else current["word"])
            occupied = sorted({c//q for c in _set_bits(word)})
            row = rng.choice(occupied)
            grid = copy_grid(current["grid"])
            # Replace actual quotient and translation labels independently
            # in the raw row used by a retained transpose-kernel word.
            grid[row] = [fresh_cell(count, policy) for count in support[row]]
            proposed = inspect(grid)
            if proposed is None:
                continue
            if proposed["cycles"][0] > current["cycles"][0]+2*q:
                continue
            proposed["small"] = small_penalty(proposed["columns"])
            diagnose(proposed, 6, 20+step)
            selected.append(proposed)
            pk, ck = merit(proposed), merit(current)
            if pk >= ck or (pk[:2] == ck[:2] and rng.random() < 0.10):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists = selected[:3]
        for number, record in enumerate(finalists):
            diagnose(record, 32, 101+number)
        finalists.sort(key=merit, reverse=True)
        for number, record in enumerate(finalists[:2]):
            diagnose(record, 48, 211+number)
        finalists.sort(key=merit, reverse=True)
        for record in finalists:
            refresh(record)
            assert syndrome(record["columns"], record["word"]) == 0
        return finalists

    def strip_logical_reading(a, b, hx, hz, diagnostic):
        """Actual sector-two witnesses modulo ORIGINAL stabilizer spans.

        A transpose-kernel word supplies an X strip at one Brow;
        B supplies a Z strip at one Arow. A failed embedding is not
        evidence of large distance. Stabilizer descent remains budgeted.
        """
        mask = (1 << q)-1
        reading = n+1
        for kind, record, opposite, stabilizers in (
                ("X", a, hz, hx), ("Z", b, hx, hz)):
            pivots = pivots_of(stabilizers)
            candidates = set()
            sources = sorted(
                record["words"] | {record["word"]},
                key=lambda w: (w.bit_count(), w))[:6]
            for word in sources:
                assert syndrome(record["columns"], word) == 0
                for location in range(3):
                    if kind == "X":
                        actual = sum(
                            ((word >> (i*q)) & mask)
                            << ((4+3*i+location)*q) for i in range(3))
                    else:
                        actual = word << ((4+3*location)*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if remainder(actual, pivots):
                        candidates.add(actual)

            reducers = list(stabilizers)
            diagnostic.shuffle(reducers)
            for actual in sorted(
                    candidates, key=lambda w: (w.bit_count(), w))[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:32]
                    best_word, best_weight = actual, actual.bit_count()
                    for i, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:i]:
                            proposed = partial ^ second
                            weight = proposed.bit_count()
                            if weight < best_weight:
                                best_word, best_weight = proposed, weight
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and remainder(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
        return reading

    aa = select_component(0, sa, policy_a)
    bb = select_component(1, sb, policy_b)
    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        -ab[0]["small"]-ab[1]["small"],
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cycles"][0]-ab[1]["cycles"][0]), reverse=True)

    finalists = []
    for number, (a, b) in enumerate(proposals):
        A, B = a["grid"], b["grid"]
        hx, hz, built_n = build(A, B)
        assert built_n == n
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (k >= q and 8*k < n):
            continue
        actual_a, actual_b = _verified_components(G, hx, hz)
        assert actual_a == a["rows"] and actual_b == b["rows"]
        diagnostic = random.Random(diagnostic_seed+3900001+100003*number)
        dn = normal_logical_reading(hx, hz, diagnostic, trials=32)
        dn = min(dn, strip_logical_reading(a, b, hx, hz, diagnostic))
        da, db = a["d"], b["d"]
        detected = min(da, db, dn)
        cycles = (a["cycles"][0]+b["cycles"][0],
                  a["cycles"][1]+b["cycles"][1])
        key = (-a["small"]-b["small"],
               k*detected*detected, detected, da+db,
               -cycles[0], -cycles[1])
        finalists.append({
            "key": key, "A": A, "B": B, "hx": hx, "hz": hz, "k": k,
            "readings": (da, db, dn), "cycles": cycles,
            "small": a["small"]+b["small"],
            "fixed": (a["fixed"], b["fixed"]),
            "complement": (a["complement"], b["complement"]),
            "component_ranks": (a["rank"], b["rank"]),
            "records": (a, b),
        })
        if len(finalists) == 6:
            break
    if not finalists:
        raise ValueError("dense anchored-challenged pairs missed rate gate")

    finalists.sort(key=lambda r: r["key"], reverse=True)
    best = None
    for number, item in enumerate(finalists[:3]):
        diagnostic = random.Random(diagnostic_seed+4900001+100003*number)
        da, db, dn = item["readings"]
        dn = min(dn, normal_logical_reading(
            item["hx"], item["hz"], diagnostic, trials=64))
        a, b = item["records"]
        dn = min(dn, strip_logical_reading(
            a, b, item["hx"], item["hz"], diagnostic))
        detected = min(da, db, dn)
        key = (-item["small"], item["k"]*detected*detected,
               detected, da+db,
               -item["cycles"][0], -item["cycles"][1])
        if best is None or key > best[0]:
            best = (key, item, (da, db, dn))

    key, item, readings = best
    hx, hz, built_n = lifted_matrices(G, item["A"], item["B"])
    assert built_n == n == 1456
    assert hx == item["hx"] and hz == item["hz"]
    assert css_ok(hx, hz)
    assert max(row.bit_count() for row in hx+hz) <= min(12, W_MAX)
    assert n-gf2_rank(hx)-gf2_rank(hz) == item["k"]
    assert all(legal_cell(cell)
               for grid in (item["A"], item["B"])
               for row in grid for cell in row)
    return {
        "cells": {"A": item["A"], "B": item["B"]},
        "screen": key, "k": item["k"], "kernel_readings": readings,
        "design": idx, "card": (pa, pb, policy_a, policy_b, cohort),
        "supports": (sa, sb), "fixed_minima": item["fixed"],
        "complement_C7_exact_minima": item["complement"],
        "component_ranks": item["component_ranks"],
        "repair_model": "binding localized and anchored min-sum/OSD witnesses",
        "orientation": "tall-dense-anchored-OSD-central-strip-repair",
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the complete dense, anchored-challenged cell without draws."""
    cell = list(ctx["cells"][side][i][j])
    assert len(cell) == ctx["supports"][int(side == "B")][i][j]
    assert len({G.idx(g) for g in cell}) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: A row-weight cap of 12 alone excludes abundant weight-6 logicals: the rolled C37 (2,6)x(2,6) parent had d=6 in all five measured slots, entirely in sector 1, despite k-kb ranging from 0 to 10.
# REFUTED: Cycle-only selection plus a high short-budget reading reliably excludes sub-60 logicals on this affine-field bed: parent slot 40 fell from d2s=70 to d30s=40, with sector-1 X witnesses spanning multiple Acol/Bcol blocks.
# REFUTED: Canonical zero-containing support cards are uniformly competitive with occupied cards on this tall bed: current zero-containing slots 11,12,17,31,32 screened at 42 and slot 37 at 28, while occupied slots 0..4 screened at 65..71; this does not establish that every zero-containing code is weak.
# REFUTED: Positive support cards with accumulated unrestricted/orbit component challenges and normal-fixed logical retests make a screening reading >=72 exclude logicals below64: current slot 20 fell from 72 to60 and slot 21 from74 to56, both with sector-two strip witnesses at k=112 and w=12.
# REFUTED: Exact complement-C7 selection, central-fixed challenges and strip retests make d2s>=72 exclude unrestricted logicals below68: current slot 23 fell from72 to66, with sector-two X strips spanning all Arows at Brow2.
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
