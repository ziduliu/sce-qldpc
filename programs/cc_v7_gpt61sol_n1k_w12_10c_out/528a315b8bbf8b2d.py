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
CONCEPT = ("F9-additive semidirect C8, order 72, A=(1,3), B=(2,5): full-generating cubic cells, translation-line challenges and independent witness retests should produce five new readings >=12 and beat fitness 35.7 in cell 0.350. This is a falsifiable measurement hypothesis, not a distance certificate.")
GROUP = ("Jump to the centre-free affine group F9-additive semidirect C8. Over F3, x^2=-1 and alpha=1+x has exact order eight; its scalar action is irreducible. The measured affine-field beds scored 343.4 at low rate and 73.1 near rate 0.300, motivating another affine action without establishing distance transfer. This order-72 shape bed is absent from BEDS.")
SHAPES = ("Change to A=(1,3), B=(2,5): n=17*72=1224 and k>=72*(3-1)*(5-2)=432 guarantee rate 6/17=0.352941 in cell 0.350. The shape permits q<=88. Components have 216 variables/72 checks and 360 variables/144 checks. Parent minima occupied sector-two Arow/Brow families; the new wide components carry corresponding sector-one Acol/Bcol families. Exact complete ranks retain 0.325<=k/n<0.375.")
FILLING = ("A has three cubic cells; B has five degree-three columns and row totals seven/eight. Both weight sums are checked: HX maximum 9+3=12 and HZ maximum 8+3=11. Twenty cards cross five inequivalent B multiplicity profiles with four A relative-order compositions. Cubic relative generators span the whole group and have spectra (3,8,8) or (4,8,8); binomials avoid involutions. Verified monomial probes supply every component matrix. Small-kernel penalties, exact manageable localized restrictions, translation-line challenges, witness-directed redraws and independent retests select fillings. Complete logical witnesses are checked against original stabilizer spans. No sampled reading certifies quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r035_F9add_C8_13x25_generating_cubic_line_witness_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 72

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("F9-additive semidirect C8 requires order 72")
    # F9 = F3[x]/(x^2+1); alpha=1+x has order eight.
    # (a,b,t) represents z -> alpha^t*z + a+b*x.
    def fm(u, v):
        a,b = u
        c,d = v
        return ((a*c-b*d) % 3, (a*d+b*c) % 3)

    powers = [(1,0)]
    for _ in range(7):
        powers.append(fm(powers[-1], (1,1)))
    assert fm(powers[-1], (1,1)) == (1,0)
    assert len(set(powers)) == 8
    els = [(a,b,t) for t in range(8)
           for b in range(3) for a in range(3)]

    def red(g):
        return (g[0] % 3, g[1] % 3, g[2] % 8)

    def mul(g, h):
        a,b,t = red(g)
        c,d,u = red(h)
        v,w = fm(powers[t], (c,d))
        return ((a+v) % 3, (b+w) % 3, (t+u) % 8)

    def inv(g):
        a,b,t = red(g)
        ti = (-t) % 8
        v,w = fm(powers[ti], ((-a) % 3, (-b) % 3))
        return (v,w,ti)

    def idx(g):
        a,b,t = red(g)
        return 9*t+3*b+a

    return dict(elements=els, ident=(0,0,0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (1, 3)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


def _copy_cells(grid):
    return [[list(cell) for cell in row] for row in grid]


# Each profile has eight columns, column degree three, and row sums eight.
# Profiles 2 and 3 have rank-three binary term-count augmentation.
_THREE_ROW_PROFILES = (
    ((1,1,1),) * 8,
    ((2,1,0), (0,1,2)) + ((1,1,1),) * 6,
    ((2,1,0), (0,2,1), (1,0,2)) + ((1,1,1),) * 5,
    ((2,1,0), (1,2,0), (0,1,2), (1,0,2))
        + ((1,1,1),) * 4,
)
_PROFILE_PAIRS = ((0,2), (1,2), (2,2), (2,3), (3,3))


def _three_row_support(rng, profile):
    """Randomize presentation, not the deliberate multiplicity profile."""
    columns = list(_THREE_ROW_PROFILES[profile])
    rng.shuffle(columns)
    rows = list(range(3))
    rng.shuffle(rows)
    grid = [[column[i] for column in columns] for i in rows]
    assert [sum(row) for row in grid] == [8, 8, 8]
    assert all(sum(row[j] for row in grid) == 3 for j in range(8))
    return grid


def _coset_pattern(rng, supports):
    """Start with a frustrated two-column square on the three-row bed."""
    nr, nc = len(supports), len(supports[0])
    bits = [[rng.randrange(2) for _ in range(nc)] for _ in range(nr)]
    full = [j for j in range(nc)
            if all(supports[i][j] for i in range(nr))]
    j, k = full[:2]
    bits[0][j] = bits[1][j] = 0
    bits[0][k], bits[1][k] = 0, 1
    return bits


def _legal_exponents(exponents, m):
    import math
    if len(set(exponents)) != len(exponents):
        return False
    if len(exponents) < 2:
        return True
    gaps = []
    divisor = m
    for a in range(len(exponents)):
        for b in range(a + 1, len(exponents)):
            d = (exponents[a] - exponents[b]) % m
            if 2 * d == m:
                return False
            gaps.append(min(d, m - d))
            divisor = math.gcd(divisor, d)
    # Distinct signed difference classes prevent internal four-cycles.
    return divisor == 1 and len(set(gaps)) == len(gaps)


def _new_coset_cell(rng, m, count, bit):
    if count == 0:
        return []
    while True:
        exponents = rng.sample(range(m), count)
        if _legal_exponents(exponents, m):
            return [(a, bit) for a in exponents]


def _initial_cubic_cells(rng, G, supports, bits):
    m = G.q // 2
    return [[_new_coset_cell(rng, m, count, bits[i][j])
             for j, count in enumerate(row)]
            for i, row in enumerate(supports)]


def _component_rows(hx, hz, q, side):
    """Extract the actual left/right components from verified matrices."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    if side == "B":
        mask = (1 << (cb * q)) - 1
        return [hz[a * q + r] & mask
                for a in range(rb) for r in range(q)]
    mask = (1 << q) - 1
    rows = []
    for i in range(ra):
        for r in range(q):
            source = hx[(i * cb) * q + r]
            row = 0
            for j in range(ca):
                row |= ((source >> (j * cb * q)) & mask) << (j * q)
            rows.append(row)
    return rows


def _cycle_cost(rows, penalty):
    """Count component four/six cycles, excluding reused variable nodes."""
    h = len(rows)
    common = [[0] * h for _ in range(h)]
    upper = [[] for _ in range(h)]
    c4 = 0
    for a in range(h):
        for b in range(a + 1, h):
            overlap = (rows[a] & rows[b]).bit_count()
            if overlap:
                common[a][b] = overlap
                upper[a].append(b)
                c4 += overlap * (overlap - 1) // 2
    c6 = 0
    local6 = [0] * h
    projection = [0] * h
    for a in range(h):
        for b in upper[a]:
            projection[a] |= 1 << b
            projection[b] |= 1 << a
        ns = upper[a]
        for t, b in enumerate(ns):
            ab = common[a][b]
            for c in ns[t + 1:]:
                bc = common[b][c]
                if bc:
                    ac = common[a][c]
                    triple = (rows[a] & rows[b] & rows[c]).bit_count()
                    count = ab * ac * bc - triple * (ab + ac + bc - 2)
                    c6 += count
                    local6[a] += count
                    local6[b] += count
                    local6[c] += count

    # The projection statistic is NOT an exact Tanner eight-cycle count.
    mode = penalty % 4
    projected4 = 0
    if mode in (1, 3):
        for a in range(h):
            for b in range(a):
                count = (projection[a] & projection[b]).bit_count()
                projected4 += count * (count - 1) // 2
        projected4 //= 2
    extra = (0.0, projected4 / 32.0,
             2.0 * max(local6, default=0), projected4 / 8.0)[mode]
    return penalty * c4 + c6 + extra


def _verified_costs(G, A, B, penalty, side=None):
    hx, hz, _ = lifted_matrices(G, A, B)
    sides = (side,) if side is not None else ("A", "B")
    costs = [_cycle_cost(_component_rows(hx, hz, G.q, s), penalty)
             for s in sides]
    return costs[0] if side is not None else costs


def _small_kernel_penalty(rows, width):
    """Exact <=4 veto plus a connected-triple weight-six witness screen."""
    import itertools
    columns = [0] * width
    adjacency = [set() for _ in range(width)]
    for r, row in enumerate(rows):
        remaining = row
        incident = []
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length() - 1
            columns[j] |= 1 << r
            incident.append(j)
            remaining ^= bit
        for j in incident:
            adjacency[j].update(k for k in incident if k != j)

    singles = set(columns)
    if 0 in singles or len(singles) != width:
        return 1000000

    pairs = set()
    short = 0
    for a in range(width):
        for b in range(a + 1, width):
            syndrome = columns[a] ^ columns[b]
            if syndrome in singles or syndrome in pairs:
                short += 1
            pairs.add(syndrome)
    if short:
        return 100000 + short

    # Enumerate triples connected through shared checks. A collision
    # verifies a six-column kernel word; this is not an exhaustive
    # general weight-six search.
    triples = set()
    for b in range(width):
        for a, c in itertools.combinations(sorted(adjacency[b]), 2):
            triples.add(tuple(sorted((a, b, c))))
    seen = {}
    six = 0
    for a, b, c in sorted(triples):
        syndrome = columns[a] ^ columns[b] ^ columns[c]
        mask = (1 << a) | (1 << b) | (1 << c)
        old = seen.get(syndrome)
        if old is not None:
            if (old ^ mask).bit_count() == 6:
                six += 1
        else:
            seen[syndrome] = mask
    return six


def _optimize_labels(rng, G, A, B, penalty):
    import math
    costs = _verified_costs(G, A, B, penalty)
    current = sum(costs)
    best = current
    best_A, best_B = _copy_cells(A), _copy_cells(B)
    positions = [[(i, j) for i, row in enumerate(grid)
                  for j, cell in enumerate(row) if cell]
                 for grid in (A, B)]
    m = G.q // 2
    steps = 180
    for step in range(steps):
        which = rng.randrange(2)
        grid = A if which == 0 else B
        i, j = rng.choice(positions[which])
        old = list(grid[i][j])
        bit = old[0][1]
        if step % 7 == 0:
            # Change holonomy as well as rotation labels; multi-term
            # cells remain in one coset, with legal rotation gaps.
            bit = rng.randrange(2)
            new = _new_coset_cell(rng, m, len(old), bit)
        else:
            new = None
            for _ in range(40):
                exponents = [g[0] for g in old]
                exponents[rng.randrange(len(old))] = rng.randrange(m)
                if _legal_exponents(exponents, m):
                    new = [(a, bit) for a in exponents]
                    break
            if new is None:
                continue
        grid[i][j] = new
        side = ("A", "B")[which]
        proposed_cost = _verified_costs(G, A, B, penalty, side)
        proposed = current - costs[which] + proposed_cost
        temperature = G.q * (18.0 * (1.0 - step / steps) ** 2 + 0.25)
        accept = proposed <= current
        if not accept:
            accept = rng.random() < math.exp((current - proposed) /
                                             temperature)
        if accept:
            current = proposed
            costs[which] = proposed_cost
            if current < best:
                best = current
                best_A, best_B = _copy_cells(A), _copy_cells(B)
        else:
            grid[i][j] = old
    return best_A, best_B, best


def _component_kernel_reading(rng, rows, width):
    """Provisional upper reading from verified component null words.

    Random elimination and word reduction can miss lighter words.
    This is neither a component-distance lower bound nor a quantum
    distance certificate.
    """
    columns = [0] * width
    for r, row in enumerate(rows):
        while row:
            low = row & -row
            columns[low.bit_length() - 1] |= 1 << r
            row ^= low

    best = width + 1
    short_words = []
    order = list(range(width))
    for trial in range(40):
        rng.shuffle(order)
        pivots = {}
        for v in order:
            syndrome, word = columns[v], 1 << v
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
            # All stored words are actual null words. XOR reductions
            # preserve that property, including when trials differ.
            for old in short_words:
                reduced = word ^ old
                if reduced and reduced.bit_count() < word.bit_count():
                    word = reduced
            weight = word.bit_count()
            best = min(best, weight)
            if word and word not in short_words:
                short_words.append(word)
                short_words.sort(key=lambda w: w.bit_count())
                del short_words[32:]
        if best <= 6:
            break
    return best


def _legacy_m37_ctx(rng, G, idx):
    """Unused historical order-111 experiment; NEW_CTX below replaces it."""
    import math

    q = G.q
    assert q == 111 and A_SHAPE == B_SHAPE == (3, 2)
    width = 3 * q
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    model, trade = card % 4, (card % 4 + cohort) % 4
    seed = rng.randrange(1 << 62)

    # Concentrate slots around the parent's successful profiles rather
    # than allocating eight slots per cohort to a doubly-even raw row.
    # Every profile still has row totals 4,5,5 and columns 7,7.
    profiles = (
        ((1, 3), (3, 2), (3, 2)),
        ((1, 3), (2, 3), (4, 1)),
        ((0, 4), (3, 2), (4, 1)),
        ((0, 4), (2, 3), (5, 0)),
        ((1, 3), (1, 4), (5, 0)),
    )
    plans = ((0, 0), (0, 1), (1, 1), (0, 2), (1, 4))
    pa, pb = plans[card // 4]
    sa = [list(row) for row in profiles[pa]]
    sb = [list(row) for row in profiles[pb]]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    for support in (sa, sb):
        augmentation = [
            sum((count & 1) << j for j, count in enumerate(row))
            for row in support
        ]
        assert gf2_rank(augmentation) == 2

    # Cache native operations, not a separately implemented lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        def __init__(self):
            self.q = q
            self.elements = list(range(q))
            self.e = identity

        def idx(self, g):
            return g

        def mul(self, g, h):
            return table[g][h]

        def inv(self, g):
            return inverses[g]

    probe = Probe()

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length() - 1
            word ^= bit

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    # One verified bulk probe extracts every transposed monomial block.
    # Dummy zero sides are probes only, never submitted candidates.
    all_cells = [[[g]] for g in range(q)]
    zero = [[[]]]
    blocks = {}
    for side in ("A", "B"):
        if side == "A":
            _, source, _ = lifted_matrices(probe, all_cells, zero)
        else:
            source, _, _ = lifted_matrices(probe, zero, all_cells)
        assert len(source) == q
        blocks[side] = [
            [(row >> ((1 + g) * q)) & mask for row in source]
            for g in range(q)
        ]

    occupancies = (
        {3: (1, 1, 1), 4: (2, 1, 1), 5: (2, 2, 1)},
        {3: (2, 1, 0), 4: (3, 1, 0), 5: (3, 2, 0)},
        {3: (2, 1, 0), 4: (2, 2, 0), 5: (3, 1, 1)},
        {3: (3, 0, 0), 4: (4, 0, 0), 5: (5, 0, 0)},
    )

    def sidon(cell):
        """Distinct directed differences in BOTH native orientations."""
        count = len(cell)
        if len(set(cell)) != count:
            return False
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        return len(left) == len(right) == count * (count - 1)

    def fresh_cell(count, policy, i=0, j=0):
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        # Four substantive recipes: pure bins, mixed bins, alternating
        # bins, and mixed bins with pure higher-weight cells.
        # A mixed bin has relative order three; a pure bin has order 37.
        # Neither order is assumed to imply a quantum-distance advantage.
        mixed = (policy in (1, 3) or
                 (policy == 2 and (i + j) % 2 == 0))
        occupancy = (((1, 1, 0) if mixed else (2, 0, 0))
                     if count == 2 else occupancies[policy][count])
        for _ in range(2048):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, size in zip(fibers, occupancy):
                cell.extend(G.idx((a, b))
                            for a in rng.sample(range(37), size))
            if sidon(cell):
                return cell
        raise ValueError("bidirectional Sidon cell construction failed")

    def fresh(support, policy):
        return [[fresh_cell(count, policy, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    def signature(grid):
        return tuple(tuple(sorted(cell))
                     for row in grid for cell in row)

    def dependency_basis(columns, positions):
        """Kernel words in the supplied, original column coordinates."""
        pivots, basis = {}, []
        for c in positions:
            value, word = columns[c], 1 << c
            while value:
                p = value.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = (value, word)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    def expand_partition(word, masks):
        expanded = 0
        for c in bits(word):
            expanded ^= masks[c]
        return expanded

    def partition_columns(columns, masks):
        return [
            _xor_columns(columns, positions)
            for positions in (list(bits(mask_)) for mask_ in masks)
        ]

    def _xor_columns(columns, positions):
        value = 0
        for c in positions:
            value ^= columns[c]
        return value

    # Coordinate partitions only: all restricted syndromes are sums of
    # authoritative component columns. No group-action lift is rewritten.
    normal_masks = [
        sum(1 << (block*q + G.idx((a, b))) for a in range(37))
        for block in range(3) for b in range(3)
    ]
    complement_masks = {}
    subgroup = [G.idx((0, b)) for b in range(3)]
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
            sum(1 << (block*q + c) for c in orbit)
            for block in range(3) for orbit in orbits
        ]

    def normal_fixed_words(columns):
        """Exact minimum of the nine-coordinate normal-fixed restriction."""
        compressed = partition_columns(columns, normal_masks)
        basis = dependency_basis(compressed, range(len(compressed)))
        words, word = [], 0
        for t in range(1, 1 << len(basis)):
            word ^= basis[(t & -t).bit_length() - 1]
            words.append(expand_partition(word, normal_masks))
        words.sort(key=lambda w: (w.bit_count(), w))
        return words[:32]

    def localized_words(columns):
        """Exact restricted enumeration when nullity <=14; words otherwise."""
        blocks_ = [list(range(i*q, (i+1)*q)) for i in range(3)]
        restrictions = blocks_ + [
            blocks_[i] + blocks_[j]
            for i in range(3) for j in range(i+1, 3)
        ]
        found = []
        for positions in restrictions:
            basis = dependency_basis(columns, positions)
            if len(basis) <= 14:
                word = 0
                shortest = None
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length() - 1]
                    if shortest is None or word.bit_count() < shortest.bit_count():
                        shortest = word
                if shortest is not None:
                    found.append(shortest)
            else:
                found.extend(sorted(basis, key=int.bit_count)[:8])
        return sorted(set(found), key=int.bit_count)[:16]

    def orbit_challenge(record, side, diagnostic_seed, trials):
        """Find actual complement-fixed component words; no lower bound."""
        masks_ = complement_masks[side]
        columns = partition_columns(record["columns"], masks_)
        order = list(range(len(columns)))
        diagnostic = random.Random(diagnostic_seed)
        found = []
        for _ in range(trials):
            diagnostic.shuffle(order)
            basis = dependency_basis(columns, order)
            if not basis:
                continue
            basis.sort(key=int.bit_count)
            short = list(basis[:8])
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    candidate = word ^ other
                    if candidate and candidate.bit_count() <= 28:
                        short.append(candidate)
            short = sorted(set(short), key=int.bit_count)[:8]
            moves = list(basis)
            diagnostic.shuffle(moves)
            for word in short:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    if not changed:
                        break
                actual = expand_partition(word, masks_)
                assert _xor_columns(record["columns"], bits(actual)) == 0
                found.append(actual)
        if found:
            record["words"].update(
                sorted(set(found), key=int.bit_count)[:16])
            record["d"] = min(record["d"],
                              min(map(int.bit_count, found)))

    def inspect(grid, side):
        """GF2-linear assembly of blocks supplied by verified probes."""
        rows = []
        mono = blocks[side]
        for j in range(2):
            for r in range(q):
                row = 0
                for i in range(3):
                    block = 0
                    for g in grid[i][j]:
                        block ^= mono[g][r]
                    row |= block << (i * q)
                rows.append(row)

        columns = [0] * width
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        if 0 in columns or len(set(columns)) != width:
            return None
        assert all(columns[i*q+t].bit_count() == sum(
            len(cell) for cell in grid[i])
            for i in range(3) for t in range(q))

        fixed_words = normal_fixed_words(columns)
        if not fixed_words:
            return None
        fixed_reading = fixed_words[0].bit_count()
        # This rejects a DETECTED restricted classical word, not an
        # inferred quantum-distance lower bound.
        if fixed_reading < 74:
            return None

        uses, four = {}, 0
        adjacency = [0] * len(rows)
        forced = 0
        for column in columns:
            neighbors = list(bits(column))
            degree = len(neighbors)
            forced += degree * (degree - 1) * (degree - 2) // 6
            for a, u in enumerate(neighbors):
                adjacency[u] |= column ^ (1 << u)
                for v in neighbors[:a]:
                    key = (v, u)
                    previous = uses.get(key, 0)
                    four += previous
                    uses[key] = previous + 1

        visited = frontier = 1
        while frontier:
            reached = 0
            for r in bits(frontier):
                reached |= adjacency[r]
            frontier = reached & ~visited
            visited |= frontier
        if visited != (1 << len(rows)) - 1:
            return None

        triangles = 0
        for u, neighbors in enumerate(adjacency):
            for v in bits(neighbors & (-1 << (u + 1))):
                triangles += (neighbors & adjacency[v]
                              & (-1 << (v + 1))).bit_count()
        excess = max(0, triangles - forced)
        # Excess is a projection statistic if four-cycles are present.
        peak = max(a.bit_count() for a in adjacency)
        cost = (700 * four + excess +
                (0, 4 * peak, excess, 12 * peak)[trade])
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    cycles=(four, excess, peak), cost=cost,
                    signature=signature(grid), d=fixed_reading,
                    fixed_reading=fixed_reading, words=set(fixed_words))

    def small_penalty(columns):
        """Exact detection of classical kernel words through weight four."""
        singles = set(columns)
        if 0 in singles or len(singles) != len(columns):
            return 1000000
        pairs = set()
        hits = 0
        for i, a in enumerate(columns):
            for b in columns[:i]:
                syndrome = a ^ b
                hits += int(syndrome in singles or syndrome in pairs)
                pairs.add(syndrome)
        return hits

    def syndrome(columns, word):
        value = 0
        for c in bits(word):
            value ^= columns[c]
        return value

    def kernel_probe(record, diagnostic_seed, trials):
        """Genuine null-word upper readings; never exclusion certificates."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        order = list(range(width))
        retained = sorted(record.get("words", ()),
                          key=int.bit_count)[:32]
        reading, witness = width + 1, None

        def observe(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        # Previous independent searches and exact restrictions remain
        # binding for this matrix; a later lucky search cannot erase them.
        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word)

        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)

            light = sorted(set(retained + basis),
                           key=int.bit_count)[:18]
            additions = []
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    observe(candidate)
                    if candidate and candidate.bit_count() <= reading + 4:
                        additions.append(candidate)

            moves = retained + basis
            diagnostic.shuffle(moves)
            for word in light[:6]:
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                            changed = True
                    observe(word)
                    if not changed:
                        break
                additions.append(word)

            if len(basis) >= 3:
                for _ in range(96):
                    a, b, c = diagnostic.sample(basis, 3)
                    observe(a ^ b ^ c)
            retained = sorted(set(retained + light + additions),
                              key=int.bit_count)[:24]
            if reading <= 8:
                break

        if witness is None:
            raise ValueError("transpose component has no kernel witness")
        assert syndrome(columns, witness) == 0
        record["words"].update(retained)
        record["words"].add(witness)
        record["words"] = set(sorted(
            record["words"], key=lambda w: (w.bit_count(), w))[:48])
        return reading, witness

    def pool(support, side, policy, restart):
        offset = 100003 * int(side == "B") + 300007 * restart
        bank = {}
        witnessed = set()

        def remember(record):
            if record is None:
                return
            bank[record["signature"]] = record
            if len(bank) > 8:
                worst = max(bank, key=lambda key: bank[key]["cost"])
                del bank[worst]

        starts = []
        for _ in range(32):
            record = inspect(fresh(support, policy), side)
            if record is not None:
                starts.append(record)
                remember(record)
        starts.sort(key=lambda record: record["cost"])
        positions = [(i, j) for i in range(3) for j in range(2)
                     if support[i][j]]

        for initial in starts[:2]:
            current = initial
            for step in range(96):
                grid = copy_grid(current["grid"])
                i, j = rng.choice(positions)
                grid[i][j] = fresh_cell(support[i][j], policy, i, j)
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                delta = proposed["cost"] - current["cost"]
                temperature = q * (8 * (1-step/96.0)**2 + 0.15)
                if delta <= 0 or rng.random() < math.exp(
                        min(0.0, -delta/temperature)):
                    current = proposed
                    remember(current)

        selected = []
        for record in sorted(bank.values(), key=lambda r: r["cost"]):
            record["rank"] = gf2_rank(record["rows"])
            if record["rank"] < 2*q - 12:
                continue
            record["small"] = small_penalty(record["columns"])
            record["words"].update(localized_words(record["columns"]))
            orbit_challenge(record, side, seed + offset + 17, 4)
            d, word = kernel_probe(record, seed + offset, 8)
            record["d"] = min(record["d"], d)
            witnessed.add(word)
            selected.append(record)
            if len(selected) == 6:
                break
        if not selected:
            return []

        def merit(record):
            # A word from another labeling matters only if its syndrome
            # vanishes on this particular verified component.
            for word in witnessed:
                if (word.bit_count() < record["d"] and
                        syndrome(record["columns"], word) == 0):
                    record["d"] = word.bit_count()
            return (-record["small"], record["d"],
                    -record["cost"], record["rank"])

        current = max(selected, key=merit)
        for step in range(16):
            active = [word for word in witnessed
                      if syndrome(current["columns"], word) == 0]
            if active:
                word = min(active, key=int.bit_count)
                occupied = sorted({c // q for c in bits(word)})
                i = rng.choice(occupied)
            else:
                i = rng.randrange(3)
            grid = copy_grid(current["grid"])
            # Replace the implicated raw row's two cells independently;
            # do not submit regular-action or gauge images.
            for j in range(2):
                grid[i][j] = fresh_cell(support[i][j], policy, i, j)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            proposed["rank"] = gf2_rank(proposed["rows"])
            if proposed["rank"] < 2*q - 12:
                continue
            proposed["small"] = small_penalty(proposed["columns"])
            proposed["words"].update(localized_words(proposed["columns"]))
            orbit_challenge(
                proposed, side, seed + offset + 500009*(step+1), 3)
            d, word = kernel_probe(
                proposed, seed + offset + 700001*(step+1), 6)
            proposed["d"] = min(proposed["d"], d)
            witnessed.add(word)
            selected.append(proposed)
            if merit(proposed) > merit(current):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists, seen = [], set()
        for record in selected:
            if record["signature"] in seen:
                continue
            seen.add(record["signature"])
            orbit_challenge(record, side, seed + offset + 800011, 8)
            d, word = kernel_probe(
                record, seed + offset + 900001, 32)
            record["d"] = min(record["d"], d)
            witnessed.add(word)
            finalists.append(record)
            if len(finalists) == 3:
                break
        finalists.sort(key=merit, reverse=True)
        for record in finalists[:2]:
            orbit_challenge(record, side, seed + offset + 1700021, 16)
            d, word = kernel_probe(
                record, seed + offset + 1900003, 64)
            record["d"] = min(record["d"], d)
            witnessed.add(word)
        finalists.sort(key=merit, reverse=True)
        return finalists

    best_answer, best_key = None, None
    for restart in range(2):
        aa = pool(sa, "A", (model + cohort) % 4, restart)
        bb = pool(sb, "B", (model + card//4 + cohort) % 4, restart)
        proposals = [(a, b) for a in aa for b in bb]
        proposals.sort(key=lambda ab: (
            -ab[0]["small"]-ab[1]["small"],
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"]+ab[1]["d"],
            -ab[0]["cost"]-ab[1]["cost"]), reverse=True)

        for a, b in proposals:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(G, A, B)
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (n == 1443 and k >= q and 8*k < n):
                continue

            # Confirm cached transpose components against the FINAL lift.
            # Sector two is ordered (Arow, Brow), Brow varying first.
            actual_a = []
            for j in range(2):
                for r in range(q):
                    source = hz[(j*3)*q+r]
                    actual_a.append(sum(
                        ((source >> ((4+3*i)*q)) & mask) << (i*q)
                        for i in range(3)))
            actual_b = [
                (hx[beta*q+r] >> (4*q)) & ((1 << width)-1)
                for beta in range(2) for r in range(q)
            ]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx+hz) == 12
            assert _grid_weights(sa, sb) == 12
            detected = min(a["d"], b["d"])
            # Compare complete pairings using measured ranks. Component
            # readings remain heuristic and are not quantum certificates.
            key = (-a["small"]-b["small"],
                   k*detected*detected, detected,
                   a["d"]+b["d"], -a["cost"]-b["cost"])
            if best_key is None or key > best_key:
                best_key = key
                best_answer = {
                    "idx": idx, "cells": {"A": A, "B": B}, "screen_k": k,
                    "card": (pa, pb, model, trade), "cohort": cohort,
                    "component_upper_readings": (a["d"], b["d"]),
                    "component_normal_fixed_minima": (
                        a["fixed_reading"], b["fixed_reading"]),
                    "component_small_penalties": (a["small"], b["small"]),
                    "component_ranks": (a["rank"], b["rank"]),
                    "component_cycles": (a["cycles"], b["cycles"]),
                    "pairing_merit": key,
                }
        if best_answer is not None:
            return best_answer
    raise ValueError("subgroup-challenged pairings missed the low-rate gate")


def NEW_CTX(rng, G, idx):
    """Twenty affine cubic/support cards with authoritative witness screens."""
    from itertools import combinations

    q = G.q
    assert q == 72 and A_SHAPE == (1, 3) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    profile, model = card // 4, card % 4
    seed = rng.randrange(1 << 62)
    ident = G.idx(G.e)
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]

    # Column permutation and row exchange cannot identify these five
    # multiplicity profiles. Every column has three terms.
    upper_profiles = (
        (1,1,1,2,2),
        (0,1,2,2,2),
        (0,1,1,2,3),
        (0,0,2,2,3),
        (0,0,1,3,3),
    )
    upper = upper_profiles[profile]
    sa = [[3,3,3]]
    sb = [list(upper), [3-a for a in upper]]
    assert [sum(row) for row in sb] == [7,8]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

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
    mask = (1 << q)-1
    monomials = [[[g] for g in range(q)]]
    xx, _, _ = lifted_matrices(probe, monomials, [[[]]])
    _, zz, _ = lifted_matrices(probe, [[[]]], monomials)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in xx] for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in zz] for g in range(q)],
    }

    def bits(word):
        while word:
            bit = word & -word
            yield bit.bit_length()-1
            word ^= bit

    def syndrome(columns, word):
        result = 0
        for c in bits(word):
            result ^= columns[c]
        return result

    def dependency_basis(columns, positions, masks_=None):
        pivots, basis = {}, []
        for c in positions:
            value = columns[c]
            word = (1 << c) if masks_ is None else masks_[c]
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

    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid group")

    def generates(a, b):
        reached, pending = {ident}, [ident]
        for x in pending:
            for y in (a,b):
                z = table[x][y]
                if z not in reached:
                    reached.add(z)
                    pending.append(z)
        return len(reached) == q

    def sidon(cell):
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        wanted = len(cell)*(len(cell)-1)
        return len(left) == len(right) == wanted

    spectra = ((3,8,8), (4,8,8))
    libraries = [[], []]
    for a in range(q):
        if a == ident:
            continue
        for b in range(a):
            if b == ident:
                continue
            spectrum = tuple(sorted((
                orders[a], orders[b], orders[table[inverses[a]][b]])))
            if spectrum not in spectra:
                continue
            cell = (ident,a,b)
            if sidon(cell) and generates(a,b):
                libraries[spectra.index(spectrum)].append(cell)
    assert all(libraries)
    # Different spectrum multiplicities, not common automorphism images.
    a_plans = ((0,0,0), (1,1,1), (0,1,1), (0,0,1))
    aplan = a_plans[model]
    ratio_pools = {
        order: [g for g in range(q) if orders[g] == order]
        for order in (4,8)
    }

    def fresh_cell(count, side, i, j):
        if not count:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            preferred = 8 if (model+j+i+cohort) % 2 else 4
            root = rng.randrange(q)
            return [root, table[root][rng.choice(ratio_pools[preferred])]]
        kind = aplan[j] if side == "A" else (
            i+j+model+cohort) % 2
        triple = rng.choice(libraries[kind])
        root = rng.randrange(q)
        return [table[root][g] for g in triple]

    supports = {"A": sa, "B": sb}

    def fresh(side):
        return [[fresh_cell(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(supports[side])]

    def inspect(grid, side):
        nr, nc = len(grid), len(grid[0])
        rows = [0]*(nr*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        columns = [0]*(nc*q)
        for r, row in enumerate(rows):
            for c in bits(row):
                columns[c] |= 1 << r
        if 0 in columns or len(set(columns)) != len(columns):
            return None
        assert all(c.bit_count() == 3 for c in columns)

        singles = set(columns)
        pairs, small = set(), 0
        for a, column in enumerate(columns):
            for other in columns[:a]:
                value = column ^ other
                small += int(value in singles or value in pairs)
                pairs.add(value)

        uses, four = {}, 0
        adjacency = [0]*len(rows)
        forced = 0
        for column in columns:
            checks = list(bits(column))
            forced += 1
            for r in checks:
                adjacency[r] |= column ^ (1 << r)
            for a,b in combinations(checks, 2):
                key = (a,b)
                previous = uses.get(key, 0)
                four += previous
                uses[key] = previous+1
        triangles = 0
        for a, neighbors in enumerate(adjacency):
            for b in bits(neighbors & (-1 << (a+1))):
                triangles += (neighbors & adjacency[b]
                              & (-1 << (b+1))).bit_count()
        excess = max(0, triangles-forced)
        return {
            "grid": _copy_cells(grid), "rows": rows, "columns": columns,
            "small": small, "cycles": (four,excess),
            "cost": 500*four+(1+model % 2)*excess,
            "words": set(), "d": len(columns)+1,
            "signature": tuple(tuple(sorted(cell))
                               for row in grid for cell in row),
        }

    def localized_words(record):
        columns = record["columns"]
        nc = len(columns)//q
        positions = [list(range(j*q,(j+1)*q)) for j in range(nc)]
        restrictions = positions + [
            positions[a]+positions[b]
            for a in range(nc) for b in range(a+1,nc)
        ]
        found = []
        for restriction in restrictions:
            basis = dependency_basis(columns, restriction)
            if not basis:
                continue
            shortest = min(basis, key=int.bit_count)
            if len(basis) <= 12:
                word = 0
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length()-1]
                    if word.bit_count() < shortest.bit_count():
                        shortest = word
            else:
                light = sorted(basis, key=int.bit_count)[:8]
                for word in light:
                    for other in basis:
                        candidate = word ^ other
                        if candidate and (
                                candidate.bit_count() < shortest.bit_count()):
                            shortest = candidate
            assert syndrome(columns, shortest) == 0
            found.append(shortest)
        record["words"].update(found)

    # Four translation lines in F3^2. Restrict actual columns using the
    # commuting action appropriate to each authoritative component.
    line_masks = {}
    for side in ("A","B"):
        nc = 3 if side == "A" else 5
        partitions = []
        for a,b in ((1,0),(0,1),(1,1),(1,2)):
            subgroup = [G.idx(((t*a) % 3,(t*b) % 3,0))
                        for t in range(3)]
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
            partitions.append([
                sum(1 << (j*q+c) for c in orbit)
                for j in range(nc) for orbit in orbits
            ])
        line_masks[side] = partitions

    def kernel_probe(record, side, trials, diagnostic_seed, restricted=False):
        """Accumulate genuine words; a missed word never becomes a bound."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        retained = sorted(record["words"], key=int.bit_count)[:24]
        found = list(retained)

        def search(cols, masks_, repetitions):
            order = list(range(len(cols)))
            for _ in range(repetitions):
                diagnostic.shuffle(order)
                basis = dependency_basis(cols, order, masks_)
                if not basis:
                    continue
                light = sorted(set(retained+basis), key=int.bit_count)[:12]
                additions = []
                for word in light:
                    for other in basis:
                        candidate = word ^ other
                        if candidate:
                            additions.append(candidate)
                moves = list(basis)+retained
                diagnostic.shuffle(moves)
                for initial in light[:5]:
                    word = initial
                    for sweep in range(3):
                        changed = False
                        for other in moves:
                            candidate = word ^ other
                            if candidate and (
                                    candidate.bit_count() < word.bit_count()):
                                word, changed = candidate, True
                        if not changed:
                            break
                    additions.append(word)
                found.extend(light)
                found.extend(sorted(set(additions),
                                    key=int.bit_count)[:16])

        search(columns, None, trials)
        if restricted:
            for masks_ in line_masks[side]:
                compressed = [syndrome(columns, word) for word in masks_]
                search(compressed, masks_, max(3,trials//8))
        record["words"] = set(sorted(
            set(found), key=lambda w: (w.bit_count(),w))[:32])
        assert record["words"]
        assert all(syndrome(columns, word) == 0
                   for word in record["words"])
        record["d"] = min(record["d"],
                          min(map(int.bit_count,record["words"])))

    def component_pool(side):
        offset = 100003*int(side == "B")
        bank, known = {}, set()
        nr, nc = (1,3) if side == "A" else (2,5)
        mutable = [(i,j) for i in range(nr) for j in range(nc)
                   if supports[side][i][j]]

        def remember(record):
            if record is None:
                return
            bank[record["signature"]] = record
            if len(bank) > 8:
                worst = max(bank, key=lambda key: (
                    bank[key]["small"],bank[key]["cost"]))
                del bank[worst]

        starts = []
        for _ in range(40):
            record = inspect(fresh(side), side)
            if record is not None:
                starts.append(record)
                remember(record)
        starts.sort(key=lambda r: (r["small"],r["cost"]))
        for initial in starts[:2]:
            current = initial
            for step in range(48):
                grid = _copy_cells(current["grid"])
                i,j = rng.choice(mutable)
                grid[i][j] = fresh_cell(
                    supports[side][i][j],side,i,j)
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                if (proposed["small"],proposed["cost"]) <= (
                        current["small"],current["cost"]):
                    current = proposed
                    remember(current)
                elif step % 11 == 0 and (
                        proposed["small"] == current["small"]):
                    current = proposed

        selected = []
        for record in sorted(bank.values(),
                             key=lambda r: (r["small"],r["cost"]))[:6]:
            record["rank"] = gf2_rank(record["rows"])
            localized_words(record)
            kernel_probe(record,side,8,seed+offset)
            known.update(record["words"])
            selected.append(record)
        if not selected:
            raise ValueError("empty affine component portfolio")

        def merit(record):
            surviving = [word for word in known
                         if syndrome(record["columns"],word) == 0]
            record["words"].update(surviving)
            if surviving:
                record["d"] = min(record["d"],
                                  min(map(int.bit_count,surviving)))
            return (-record["small"],record["d"],
                    -record["cost"],record["rank"])

        current = max(selected,key=merit)
        for step in range(14):
            active = sorted(current["words"],key=int.bit_count)
            occupied = ({c//q for c in bits(active[0])}
                        if active else set())
            targeted = [p for p in mutable if p[1] in occupied]
            i,j = rng.choice(targeted or mutable)
            grid = _copy_cells(current["grid"])
            grid[i][j] = fresh_cell(supports[side][i][j],side,i,j)
            proposed = inspect(grid,side)
            if proposed is None:
                continue
            proposed["rank"] = gf2_rank(proposed["rows"])
            localized_words(proposed)
            kernel_probe(proposed,side,6,
                         seed+offset+300007*(step+1))
            known.update(proposed["words"])
            selected.append(proposed)
            if merit(proposed) > merit(current):
                current = proposed

        selected.sort(key=merit,reverse=True)
        finalists, seen = [], set()
        for record in selected:
            if record["signature"] in seen:
                continue
            seen.add(record["signature"])
            kernel_probe(record,side,32,seed+offset+700001,True)
            kernel_probe(record,side,16,seed+offset+900001,True)
            known.update(record["words"])
            finalists.append(record)
            if len(finalists) == 3:
                break
        finalists.sort(key=merit,reverse=True)
        return finalists

    def span_pivots(rows):
        pivots = {}
        for row in rows:
            while row:
                p = row.bit_length()-1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = row
                    break
                row ^= old
        return pivots

    def in_span(word,pivots):
        while word:
            p = word.bit_length()-1
            old = pivots.get(p)
            if old is None:
                return False
            word ^= old
        return True

    def logical_reading(hx,hz,a,b,n):
        """Verify embedded component words modulo original stabilizer spans."""
        reading = n+1
        for record,opposite,stabilizers,is_b in (
                (b,hz,hx,True),(a,hx,hz,False)):
            pivots = span_pivots(stabilizers)
            words = sorted(record["words"],key=int.bit_count)[:8]
            for word in words:
                for block in range(3 if is_b else 5):
                    if is_b:
                        actual = word << (block*5*q)
                    else:
                        actual = sum(1 << ((c//q*5+block)*q+c % q)
                                     for c in bits(word))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if in_span(actual,pivots):
                        continue
                    for sweep in range(3):
                        changed = False
                        for row in stabilizers:
                            candidate = actual ^ row
                            if candidate.bit_count() < actual.bit_count():
                                actual,changed = candidate,True
                        if not changed:
                            break
                    assert actual and not in_span(actual,pivots)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    reading = min(reading,actual.bit_count())
        return reading

    aa,bb = component_pool("A"),component_pool("B")
    pairs = [(a,b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        -ab[0]["small"]-ab[1]["small"],
        min(ab[0]["d"],ab[1]["d"]),
        ab[0]["d"]+ab[1]["d"],
        -ab[0]["cost"]-ab[1]["cost"]),reverse=True)
    best = None
    for a,b in pairs:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx,hz,n = lifted_matrices(G,A,B)
        k = n-gf2_rank(hx)-gf2_rank(hz)
        if not (n == 1224 and k >= 432 and
                13*n <= 40*k < 15*n):
            continue
        actual_a = [
            sum(((row >> (j*5*q)) & mask) << (j*q)
                for j in range(3)) for row in hx[:q]
        ]
        actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
        assert actual_a == a["rows"] and actual_b == b["rows"]
        dn = logical_reading(hx,hz,a,b,n)
        detected = min(a["d"],b["d"],dn)
        key = (-a["small"]-b["small"],k*detected*detected,
               detected,-a["cost"]-b["cost"])
        if best is None or key > best[0]:
            best = (key,A,B,k,(a["d"],b["d"],dn),hx,hz)
    if best is None:
        raise ValueError("affine pairings missed the exact 0.350 rate gate")
    key,A,B,k,readings,hx,hz = best
    assert css_ok(hx,hz)
    assert max(row.bit_count() for row in hx) == 12
    assert max(row.bit_count() for row in hz) == 11
    return {
        "idx": idx,"cells": {"A": A,"B": B},"k": k,
        "card": (profile,model,cohort),"screen": key,
        "component_and_logical_upper_readings": readings,
        "supports": (sa,sb),
    }


def CELL(rng, G, side, i, j, ctx):
    """Return the independently screened complete affine-design cell."""
    return list(ctx["cells"][side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Weight-legal rolling and uniform labels alone suppress very short logicals in the order-52 (2,6)x(2,4) bed: all five measured parent codes had distance 4, with many sector-1 witnesses.
# REFUTED: Varying the optimized two-row cubic coset profiles breaks the distance-ten plateau on the order-44 bed: all five current measured codes instead returned ten, with minimum logicals confined to sector 1.
# REFUTED: Adding a third component check row with balanced same-coset cubic columns raises the order-20 (3,8)x(3,8) readings above ten; all five measured selections instead returned eight, entirely in sector 1, and only 12 of 60 submitted slots were distinct.
# REFUTED: Prime-order binomial ratios and bidirectional Sidon cells alone exclude order-37 logical families on this bed; parent slots 0..7 all returned d2s=37 despite w=12 and independent component screening.
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
