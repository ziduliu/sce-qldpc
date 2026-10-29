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
CONCEPT = ("Hypothesis: split-metacyclic C8:C2(s=3), order 16, A=B=(3,9), with verified zero-four-cycle degree-three components and independently retained short-kernel challenges produces five new measured readings >=14 and improves cell 0.400 beyond fitness 59.451. Failure refutes this measurement target; finite readings remain provisional upper bounds.")
GROUP = ("Keep the split-metacyclic construction rule, resizing its parameters to m=8,r=2,s=3. The new factor ninety requires 12<=q<=16, so the parent's order 129 is inadmissible. Since 3^2=1 mod 8 and 3!=1, these parameters give a nonabelian group of order sixteen and n=1440. This bed is absent from BEDS. Its usefulness is an experiment, not distance transfer from the parent's measured order-129 plateau.")
SHAPES = ("Change BOTH sides to (3,9). Equal components have 144 variables and 48 checks; q_max=16, n=1440 and k>=16*(9-3)^2=576 guarantee rate 0.400. Three check-block rows replace the parent's single-row components. Parent minima were entirely sector one, particularly Acol1/2 across B columns and Bcol1/2 across A columns. Every new column has degree three and every row has degree nine. Actual full ranks, not augmentation-defect predictions, require 598<=k and 0.375<=k/n<0.425.")
FILLING = ("Twenty cards cross four balanced B supports with five order-eight A-holonomy and B-binomial-ratio prescriptions. A is monomial; B substitutes 2,4,6 or 8 columns by paired (2,1,0)/(0,1,2) columns. Both components retain row totals nine and column totals three, hence BOTH quantum maxima twelve. Gauge anchors remove proposal freedom but never manufacture submitted orbit copies. Native differences propose zero-four-cycle components; authoritative blocks verify cycles, ranks and kernel-four exclusion. Preserve rank diversity through retained kernel cancellation, bounded anchored parity completion and two independent deeper finalist challenges. Full original-coordinate logical commutation, nontriviality and stabilizer reductions challenge pairings. Selection readings are capped at fourteen; finite searches and voltage prescriptions certify neither distance nor complete Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r040_m8r2s3_39x39_zero4_degree3_rankdiverse_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 16

def custom_ops(q):
    # Same split-metacyclic construction rule; factor ninety needs q<=16.
    # Multiplication is x^a y^b with y*x*y^-1=x^3.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 16")
    m, r, s = 8, 2, 3
    assert pow(s, r, m) == 1 and s != 1
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 9)               # (ra, ca)
B_SHAPE = (3, 9)               # (rb, cb)


def _syndrome_columns(rows, n):
    """Columns of a verified bit-row matrix, as syndrome integers."""
    cols = [0] * n
    mask = (1 << n) - 1
    for r, row in enumerate(rows):
        x = row & mask
        while x:
            bit = x & -x
            cols[bit.bit_length() - 1] |= 1 << r
            x ^= bit
    return cols


def _no_kernel4(cols):
    """Exact exclusion of every nonzero kernel vector of weight <=4."""
    singles = set(cols)
    if 0 in singles or len(singles) != len(cols):
        return False
    # A pair matching a singleton gives weight three; matching an
    # earlier pair gives weight four. Shared indices would imply
    # duplicate columns, already excluded above.
    seen = singles.copy()
    for i, a in enumerate(cols):
        for j in range(i):
            v = a ^ cols[j]
            if v in seen:
                return False
            seen.add(v)
    return True


# Four inequivalent density allocations crossed with five occupancy models.
# Entries are the two cell counts on A and B, respectively.
_COUNT_PAIRS = (
    ((4, 4), (4, 4)),
    ((3, 4), (4, 4)),
    ((3, 5), (3, 4)),
    ((3, 4), (3, 4)),
)
_COSET_COUNTS = (
    {3: (1, 1, 1), 4: (2, 1, 1), 5: (2, 2, 1)},
    {3: (2, 1, 0), 4: (2, 2, 0), 5: (3, 2, 0)},
    {3: (2, 1, 0), 4: (3, 1, 0), 5: (3, 1, 1)},
)


def _bits(x):
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _mixed_cells(rng, counts, side, model, separation):
    """Every cell mixes cosets; no single-coset polynomial pair recipe."""
    cells = []
    for j, count in enumerate(counts):
        if model < 3:
            kind = model
        elif model == 3:
            kind = 0 if side == "A" else 1
        else:
            kind = (0, 1)[j] if side == "A" else (2, 0)[j]
        fibers = list(range(3))
        rng.shuffle(fibers)
        cell = []
        for b, size in zip(fibers, _COSET_COUNTS[kind][count]):
            if size == 0:
                continue
            for _ in range(2000):
                rotations = rng.sample(range(97), size)
                ds = [(a - c) % 97 for a in rotations
                      for c in rotations if a != c]
                if (len(set(ds)) == len(ds)
                        and all(min(d, 97 - d) >= separation for d in ds)):
                    break
            else:
                raise ValueError("rotation support construction failed")
            cell.extend((a, b) for a in rotations)
        assert len(cell) == count
        assert len(set(cell)) == count
        assert len({b for a, b in cell}) >= 2
        cells.append(cell)
    return [cells]


def _verified_components(hx, hz, q):
    """Extract the two components from the scaffold's actual full lift."""
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    ar = []
    for i in range(ra):
        for t in range(q):
            source = hx[(i * cb) * q + t]
            ar.append(sum(((source >> (j * cb * q)) & mask)
                          << (j * q) for j in range(ca)))
    bmask = (1 << (cb * q)) - 1
    br = [hz[a * q + t] & bmask
          for a in range(rb) for t in range(q)]
    return ar, br


def _component_quality(rows, width):
    """Exact four cycles; projection-triangle excess is only a heuristic."""
    columns = _syndrome_columns(rows, width)
    if 0 in columns or len(set(columns)) != width:
        return None
    pair_uses = {}
    four = 0
    for row in rows:
        ns = list(_bits(row))
        for t, u in enumerate(ns):
            for v in ns[t + 1:]:
                key = u * width + v
                previous = pair_uses.get(key, 0)
                four += previous
                pair_uses[key] = previous + 1

    adjacency = [0] * len(rows)
    forced = 0
    for signature in columns:
        degree = signature.bit_count()
        forced += degree * (degree - 1) * (degree - 2) // 6
        for r in _bits(signature):
            adjacency[r] |= signature ^ (1 << r)

    visited = frontier = 1
    while frontier:
        reached = 0
        for r in _bits(frontier):
            reached |= adjacency[r]
        frontier = reached & ~visited
        visited |= frontier
    if visited != (1 << len(rows)) - 1:
        return None

    triangles = 0
    for u, neighbors in enumerate(adjacency):
        for v in _bits(neighbors & (-1 << (u + 1))):
            triangles += (neighbors & adjacency[v]
                          & (-1 << (v + 1))).bit_count()
    return (four, max(0, triangles - forced)), columns


def _kernel_reading(rng, rows, width):
    """Sample actual component kernel words; an upper reading, not a bound below."""
    reading = width + 1
    for _ in range(6):
        permutation = list(range(width))
        rng.shuffle(permutation)
        mapped = [1 << p for p in permutation]
        pivots = {}
        for original in rows:
            row = 0
            for c in _bits(original):
                row |= mapped[c]
            while row:
                p = row.bit_length() - 1
                if p in pivots:
                    row ^= pivots[p]
                else:
                    pivots[p] = row
                    break
        ordered = sorted(pivots.items())
        basis = []
        for free in range(width):
            if free in pivots:
                continue
            word = 1 << free
            for p, row in ordered:
                if (word & row).bit_count() & 1:
                    word |= 1 << p
            basis.append(word)
            reading = min(reading, word.bit_count())
        for i, word in enumerate(basis):
            for other in basis[:i]:
                reading = min(reading, (word ^ other).bit_count())
        if len(basis) >= 3:
            for _ in range(160):
                a, b, c = rng.sample(basis, 3)
                reading = min(reading, (a ^ b ^ c).bit_count())
    return reading


# Removed the unused order-291 optimizer to reserve program-size budget
# for the new three-row component search. The refuted ledger is unchanged.


def _retired_affinetriangle_ctx(rng, G, idx):
    """Inactive order-129 predecessor; never called on the new shape."""
    q = G.q
    width = 3 * q
    mask = (1 << q) - 1
    assert q == 129 and A_SHAPE == B_SHAPE == (1, 3)
    support = [[3, 3, 3]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    card, cohort = idx % 20, idx // 20
    diagnostic_seed = rng.randrange(1 << 62)
    rng = random.Random(diagnostic_seed ^ (0x43AFF173 + 1000003 * idx))

    # Native commutators classify proposals; every matrix screen below
    # still uses authoritative monomial blocks.
    signed_action = {
        (sign * pow(6, b, 43)) % 43
        for sign in (1, -1) for b in range(3)
    }
    commutator_classes = tuple(
        tuple(sorted({pow(3, t, 43) * z % 43 for z in signed_action}))
        for t in range(7)
    )
    assert len(signed_action) == 6
    assert len(set().union(*(set(c) for c in commutator_classes))) == 42
    profiles = (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 2),
        (0, 1, 3),
        (0, 1, 4),
    )
    model, composition = card // 5, card % 5
    pa = composition
    pb = (2 * composition + 1 + cohort) % 5
    triangle = (3, 4, 5, 6)[model]
    trade = (model + cohort) % 3

    # Cache native group operations, not a separately implemented lift.
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
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
            return inverse[g]

    P = Probe()

    # Batches keep unused dummy matrices small. Every permutation block
    # comes from the scaffold; subsequent assembly is GF2 linearity only.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(start + 16, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, [[[]]])
        _, zz, _ = lifted_matrices(P, [[[]]], gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([
                (row >> (j * q)) & mask for row in xx])
            blocks["B"].append([
                (row >> (j * q)) & mask for row in zz])

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell)) for cell in grid[0])

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def expand(word, groups):
        actual = 0
        for c in _bits(word):
            actual ^= groups[c]
        return actual

    # Only coordinate restrictions are formed here. Their syndromes are
    # XORs of authoritative columns, and expanded words are verified.
    h = G.idx((0, 1))
    assert table[table[h][h]][h] == identity
    partitions = {}
    for side in ("A", "B"):
        remaining = set(range(q))
        orbits = []
        while remaining:
            root = min(remaining)
            orbit = [root]
            for _ in range(2):
                previous = orbit[-1]
                orbit.append(table[previous][h] if side == "A"
                             else table[h][previous])
            assert len(set(orbit)) == 3 and set(orbit) <= remaining
            remaining.difference_update(orbit)
            orbits.append(orbit)
        assert len(orbits) == 43
        partitions[side] = [
            sum(1 << (j * q + c) for c in orbit)
            for j in range(3) for orbit in orbits
        ]

    def sidon(cell):
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverse[a]][b])
                    right.add(table[a][inverse[b]])
        return (len(cell) == len(set(cell)) == 3
                and len(left) == len(right) == 6)

    def commutator(cell):
        # Native multiplication supplies this invariant, not a new lift.
        root = inverse[cell[0]]
        u = table[root][cell[1]]
        v = table[root][cell[2]]
        return table[table[u][v]][inverse[table[v][u]]]

    def fresh_cell(kind, a=None):
        for _ in range(128):
            coordinate = rng.randrange(43) if a is None else a
            value = rng.choice(commutator_classes[kind])
            delta = (26 * value) % 43
            cell = [
                identity,
                G.idx((coordinate, 1)),
                G.idx(((7 * coordinate + delta) % 43, 2)),
            ]
            assert commutator(cell) == G.idx((value, 0))
            if sidon(cell):
                return cell
        raise ValueError("balanced commutator class failed Sidon gate")

    def fresh(profile):
        for _ in range(128):
            origin, scale = rng.randrange(43), rng.randrange(1, 43)
            coordinates = [
                (origin + scale * t) % 43 for t in (0, 1, triangle)
            ]
            row = [
                fresh_cell(kind, a)
                for kind, a in zip(profiles[profile], coordinates)
            ]
            if len({commutator(cell) for cell in row}) == 3:
                return [row]
        raise ValueError("affine-coupled balanced geometries exhausted")

    def small_word(columns):
        """Exact detection of kernel words through weight four.

        A returned word is genuine; None excludes all weights 1..4.
        The first returned word need not be the minimum at weights 3..4.
        """
        singles = {}
        for c, value in enumerate(columns):
            bit = 1 << c
            if not value:
                return bit
            if value in singles:
                return bit ^ singles[value]
            singles[value] = bit
        pairs = {}
        for c, value in enumerate(columns):
            for d in range(c):
                v = value ^ columns[d]
                word = (1 << c) | (1 << d)
                old = singles.get(v)
                if old is not None:
                    return word ^ old
                old = pairs.get(v)
                if old is not None:
                    return word ^ old
                pairs[v] = word
        return None

    def cycle_key(cycles):
        four, excess = cycles
        if trade == 0:
            return four, excess
        if trade == 1:
            return 8 * four + excess, four
        return 24 * four + excess, excess

    def inspect(grid, side):
        expected = profiles[pa if side == "A" else pb]
        commutators = [commutator(cell) for cell in grid[0]]
        if (len({cell[1] for cell in grid[0]}) != 3
                or len(set(commutators)) != 3):
            return None
        coordinates = [G.elements[cell[1]][0] for cell in grid[0]]
        gap = (coordinates[1] - coordinates[0]) % 43
        if not gap:
            return None
        if ((coordinates[2] - coordinates[0]) * pow(gap, -1, 43)
                % 43 != triangle):
            return None
        for cell, value, kind in zip(grid[0], commutators, expected):
            assert cell[0] == identity
            assert tuple(G.elements[g][1] for g in cell) == (0, 1, 2)
            assert G.elements[value][0] in commutator_classes[kind]
            assert G.elements[value][1] == 0
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, value in enumerate(blocks[side][g]):
                    rows[r] ^= value << (j * q)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        assert all(column.bit_count() == 3 for column in columns)
        rank = gf2_rank(rows)
        if rank < q - 2:
            return None
        if small_word(columns) is not None:
            return None

        groups = partitions[side]
        fixed_columns = [syndrome(columns, group) for group in groups]
        # Each orbit sums three odd-degree columns, so its parity is odd.
        assert all(value.bit_count() & 1 for value in fixed_columns)
        fw = small_word(fixed_columns)
        words = []
        fixed_beam = []
        fixed_d = width + 1
        if fw is not None:
            actual = expand(fw, groups)
            assert actual.bit_count() == 3 * fw.bit_count()
            assert syndrome(columns, actual) == 0
            words.append(actual)
            fixed_beam.append(fw)
            fixed_d = actual.bit_count()

        return {
            "grid": copy_grid(grid), "signature": signature(grid),
            "side": side, "affine_coordinates": tuple(coordinates),
            "rows": rows, "columns": columns, "rank": rank,
            "cycles": cycles, "cycle_key": cycle_key(cycles),
            "groups": groups, "fixed_columns": fixed_columns,
            "fixed_beam": fixed_beam, "beam": words,
            "fixed_small_free": fw is None,
            "fixed_d": fixed_d,
            "d": fixed_d,
            "word": words[0] if words else None,
        }

    def kernel_search(columns, seed, trials, seeds=(), stop=8):
        """Accumulated genuine witnesses from independently changed bases."""
        local = random.Random(seed)
        count = len(columns)
        beam = sorted({word for word in seeds if word},
                      key=int.bit_count)[:20]
        best_word = None

        def observe(word):
            nonlocal best_word
            if word and (best_word is None or
                         word.bit_count() < best_word.bit_count()):
                best_word = word

        for word in beam:
            assert syndrome(columns, word) == 0
            observe(word)

        order = list(range(count))
        for turn in range(trials):
            local.shuffle(order)
            if turn % 5 == 4 and beam:
                anchor = local.choice(beam[:6])
                order.sort(key=lambda c: not ((anchor >> c) & 1))
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    observe(word)
            if not basis:
                continue

            light = sorted(set(basis + beam), key=int.bit_count)[:12]
            grown = list(light) + beam
            for a in light:
                for b in basis:
                    candidate = a ^ b
                    observe(candidate)
                    if (candidate and candidate.bit_count()
                            <= best_word.bit_count() + 6):
                        grown.append(candidate)
                if len(grown) > 160:
                    grown = sorted(set(grown), key=int.bit_count)[:32]

            moves = basis + beam
            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if (candidate and candidate.bit_count()
                                < word.bit_count()):
                            word, changed = candidate, True
                    observe(word)
                    if not changed:
                        break
                grown.append(word)

            shortened = sorted(set(grown), key=int.bit_count)[:20]
            for i, a in enumerate(shortened):
                for b in shortened[:i]:
                    observe(a ^ b)
            grown.append(best_word)
            beam = sorted({word for word in grown if word},
                          key=int.bit_count)[:20]
            if best_word.bit_count() <= stop:
                break

        if best_word is None:
            raise ValueError("no component kernel witness found")
        assert syndrome(columns, best_word) == 0
        return best_word, beam

    def anchored_sparse_challenge(record, seed, attempts):
        """Find genuine kernel words by forced-anchor min-sum/OSD.

        All syndromes use authoritative columns. A failed decoding or
        finite-budget completion is not an exclusion or distance bound.
        """
        local = random.Random(seed)
        columns = record["columns"]
        best = record["word"]
        found = set(record["beam"] + [best])
        assert best and syndrome(columns, best) == 0

        def note(word):
            nonlocal best
            if not word:
                return
            assert syndrome(columns, word) == 0
            if word.bit_count() < best.bit_count():
                best = word
            if word.bit_count() <= min(18, best.bit_count() + 4):
                found.add(word)

        for attempt in range(attempts):
            block = (attempt + model + composition) % 3
            root = block * q + identity
            root_bit = 1 << root
            target = columns[root]
            edge_variables = []
            check_edges = [[] for _ in range(q)]
            for c, value in enumerate(columns):
                if c == root:
                    continue
                for r in _bits(value):
                    edge = len(edge_variables)
                    edge_variables.append(c)
                    check_edges[r].append(edge)

            channel = [2.0 + local.uniform(-0.65, 0.65)
                       for _ in columns]
            nearby = sorted({
                edge_variables[e]
                for r in _bits(target) for e in check_edges[r]
            })
            warm = [
                word for word in found
                if (word >> (block * q)) & mask
            ]
            if attempt % 3 == 1 and warm:
                source = local.choice(sorted(
                    warm, key=lambda word: (word.bit_count(), word))[:8])
                anchors = list(_bits((source >> (block * q)) & mask))
                multiplier = inverse[local.choice(anchors)]
                moved = 0
                for c in _bits(source):
                    j, g = divmod(c, q)
                    image = (table[g][multiplier]
                             if record["side"] == "A"
                             else table[multiplier][g])
                    moved |= 1 << (j * q + image)
                assert moved & root_bit
                assert syndrome(columns, moved) == 0
                for c in _bits(moved ^ root_bit):
                    channel[c] = local.uniform(0.05, 0.75)
            elif attempt % 3 == 2 and nearby:
                for c in local.sample(nearby, min(3, len(nearby))):
                    channel[c] = -local.uniform(0.1, 1.0)

            vm = [channel[c] for c in edge_variables]
            cm = [0.0] * len(edge_variables)
            posterior = list(channel)
            hard = 0
            for _ in range(20):
                for r, incident in enumerate(check_edges):
                    sign = -1 if (target >> r) & 1 else 1
                    first = second = 24.0
                    smallest = -1
                    for e in incident:
                        value = vm[e]
                        if value < 0:
                            sign = -sign
                        magnitude = abs(value)
                        if magnitude < first:
                            second, first = first, magnitude
                            smallest = e
                        elif magnitude < second:
                            second = magnitude
                    for e in incident:
                        own = -1 if vm[e] < 0 else 1
                        magnitude = second if e == smallest else first
                        proposed = 0.8 * sign * own * magnitude
                        cm[e] = 0.55 * cm[e] + 0.45 * proposed
                posterior = list(channel)
                for e, c in enumerate(edge_variables):
                    posterior[c] += cm[e]
                hard = sum(
                    1 << c for c in range(width)
                    if c != root and posterior[c] < 0)
                if syndrome(columns, hard) == target:
                    note(hard ^ root_bit)
                    break
                for e, c in enumerate(edge_variables):
                    proposed = posterior[c] - cm[e]
                    vm[e] = max(-24.0, min(
                        24.0, 0.4 * vm[e] + 0.6 * proposed))

            # Exact syndrome completion in a reliability-ordered basis.
            # Circuit moves exclude the anchor, preserving nonzero words.
            order = [c for c in range(width) if c != root]
            reliability = [
                abs(value) + local.uniform(0.0, 0.2)
                for value in posterior
            ]
            order.sort(key=lambda c: reliability[c])
            pivots, circuits = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
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
                old = pivots.get(residual.bit_length() - 1)
                if old is None:
                    break
                residual ^= old[0]
                repair ^= old[1]
            if residual:
                continue
            candidate = hard ^ repair ^ root_bit
            assert candidate & root_bit
            for _ in range(2):
                before = candidate.bit_count()
                local.shuffle(circuits)
                for move in circuits:
                    proposed = candidate ^ move
                    if proposed.bit_count() < candidate.bit_count():
                        candidate = proposed
                light = sorted(
                    circuits,
                    key=lambda word: (candidate ^ word).bit_count())[:20]
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
            note(candidate)

        record["beam"] = sorted(
            found | {best}, key=lambda word: (word.bit_count(), word))[:20]
        return best

    def parity_completion(record, seed, budget=20000):
        """Budgeted search through ten; exhaustion is not exclusion."""
        local = random.Random(seed)
        columns = record["columns"]
        best = record["word"]
        cap = min(10, best.bit_count() - 1)
        if cap < 5:
            return best
        singles, pairs = {}, {}
        incidence = [0] * q
        for c, value in enumerate(columns):
            bit = 1 << c
            singles.setdefault(value, bit)
            for r in _bits(value):
                incidence[r] |= bit
            for d in range(c):
                pairs.setdefault(value ^ columns[d],
                                 bit | (1 << d))

        priorities = [local.random() for _ in columns]
        roots = [j * q for j in range(3)]
        local.shuffle(roots)
        all_variables = (1 << width) - 1

        def observe(word):
            nonlocal best
            if word and word.bit_count() < best.bit_count():
                assert syndrome(columns, word) == 0
                best = word

        for root in roots:
            remaining = [max(1, budget // len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    observe(word)
                    return
                left = min(cap, best.bit_count() - 1) - word.bit_count()
                if left <= 0 or value.bit_count() > 3 * left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    observe(word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    observe(word | pair)
                left = min(cap, best.bit_count() - 1) - word.bit_count()
                if left <= 2:
                    return

                chosen, smallest = 0, width + 1
                for r in _bits(value):
                    candidates = incidence[r] & allowed
                    size = candidates.bit_count()
                    if not size:
                        return
                    if size < smallest:
                        chosen, smallest = candidates, size
                candidates = list(_bits(chosen))
                candidates.sort(key=lambda c: (
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in candidates:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            visit(columns[root], 1 << root,
                  all_variables ^ (1 << root))
            if best.bit_count() <= 6:
                break
        return best

    def component_pool(profile, side):
        offset = 100003 * int(side == "B")
        bank, seen = set(), set()

        def probe(record, turn, full_trials, fixed_trials):
            transferred = [
                word for word in bank
                if syndrome(record["columns"], word) == 0
            ]
            word, beam = kernel_search(
                record["columns"],
                diagnostic_seed + offset + 700001 * turn,
                full_trials, record["beam"] + transferred)
            record["beam"] = beam
            fw, fixed_beam = kernel_search(
                record["fixed_columns"],
                diagnostic_seed + offset + 1700003 + 900001 * turn,
                fixed_trials, record["fixed_beam"], stop=4)
            record["fixed_beam"] = fixed_beam
            actual = expand(fw, record["groups"])
            assert actual.bit_count() == 3 * fw.bit_count()
            assert syndrome(record["columns"], actual) == 0
            assert all((row & actual).bit_count() % 2 == 0
                       for row in record["rows"])
            record["fixed_d"] = min(
                record["fixed_d"], actual.bit_count())
            for witness in (word, actual):
                bank.add(witness)
                if witness.bit_count() < record["d"]:
                    record["d"] = witness.bit_count()
                    record["word"] = witness
            record["beam"] = sorted(
                set(record["beam"] + [actual, record["word"]]),
                key=int.bit_count)[:20]
            if full_trials >= 24:
                sparse_word = anchored_sparse_challenge(
                    record,
                    diagnostic_seed + offset + 2300009 + 1000033 * turn,
                    3 if full_trials < 48 else 6)
                bank.add(sparse_word)
                if sparse_word.bit_count() < record["d"]:
                    record["d"] = sparse_word.bit_count()
                    record["word"] = sparse_word
                record["beam"] = sorted(
                    set(record["beam"] + [record["word"]]),
                    key=lambda word: (word.bit_count(), word))[:20]
            if len(bank) > 80:
                keep = sorted(bank, key=int.bit_count)[:56]
                bank.clear()
                bank.update(keep)

        def merit(record):
            for word in bank:
                if (word.bit_count() < record["d"]
                        and syndrome(record["columns"], word) == 0):
                    record["d"], record["word"] = word.bit_count(), word
            found_minima = {
                word for word in record["beam"]
                if word.bit_count() == record["d"]
            }
            return (min(14, record["d"]),
                    record["fixed_small_free"],
                    -len(found_minima),
                    min(18, record["fixed_d"]),
                    tuple(-x for x in record["cycle_key"]),
                    record["rank"])

        for restart in range(3):
            records = []
            for _ in range(128 + 16 * min(cohort, 2)):
                grid = fresh(profile)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                record = inspect(grid, side)
                if record is not None:
                    records.append(record)
            records.sort(key=lambda record: (
                not record["fixed_small_free"], record["cycle_key"]))

            selected = records[:10]
            if not selected:
                continue
            for record in selected:
                probe(record, 1 + 300 * restart, 8, 16)
            current = max(selected, key=merit)
            floor4 = min(record["cycles"][0] for record in selected)

            # Change actual terms, never apply a common gauge or automorphism.
            # The card's three-cell relative-order composition is preserved.
            for step in range(18):
                used = list(_bits(current["word"]))
                j = (rng.choice(used) // q
                     if used and rng.random() < 0.85 else rng.randrange(3))
                grid = copy_grid(current["grid"])
                if step % 6 == 5:
                    grid = fresh(profile)
                else:
                    # Preserve the coupled generator triangle while
                    # changing an actual commutator, not a common gauge.
                    coordinate = G.elements[grid[0][j][1]][0]
                    grid[0][j] = fresh_cell(
                        profiles[profile][j], coordinate)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                if proposed["cycles"][0] > floor4 + 2 * q:
                    continue
                probe(proposed, 11 + step + 300 * restart, 4, 8)
                selected.append(proposed)
                pk, ck = merit(proposed), merit(current)
                if pk > ck or (pk[0] == ck[0] and rng.random() < 0.10):
                    current = proposed

            finalists = sorted(selected, key=merit, reverse=True)[:4]
            for record in finalists:
                probe(record, 101 + 300 * restart, 24, 48)
            finalists.sort(key=merit, reverse=True)
            finalists = finalists[:3]
            for record in finalists:
                word = parity_completion(
                    record, diagnostic_seed + offset + 2700011,
                    20000 + 4000 * min(cohort, 2))
                bank.add(word)
                if word.bit_count() < record["d"]:
                    record["d"], record["word"] = word.bit_count(), word
                record["beam"].append(word)
                # Independent retesting retains every earlier witness.
                probe(record, 211 + 300 * restart,
                      64 + 8 * min(cohort, 2), 96)
            finalists.sort(key=merit, reverse=True)
            return finalists
        raise ValueError("no trinomial component passed authoritative gates")

    def localized_logical_challenge(a, b, hx, hz, n):
        """Found nontrivial logicals give upper readings, never exclusions.

        Component words are placed in the ORIGINAL qubit coordinates.
        Exact commutation and original stabilizer-span membership decide
        whether each placement is a logical before any weight reduction.
        """
        def span(rows):
            pivots = {}
            for word in rows:
                while word:
                    lead = word.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = word
                        break
                    word ^= old
            return pivots

        def outside(word, pivots):
            while word:
                old = pivots.get(word.bit_length() - 1)
                if old is None:
                    return True
                word ^= old
            return False

        readings = []
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            pivots = span(stabilizers)
            words = sorted(
                set(record["beam"] + [record["word"]]),
                key=lambda word: (word.bit_count(), word))[:6]
            candidates = set()
            for word in words:
                assert syndrome(record["columns"], word) == 0
                for location in range(3):
                    if side == "A":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((3*j + location)*q)
                            for j in range(3))
                    else:
                        actual = word << (location * width)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)

            reading = n + 1
            local = random.Random(
                diagnostic_seed + (4100011 if side == "A" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(candidates, key=int.bit_count)[:8]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed

                    # Exhaust all pairs in an overlap-selected neighborhood.
                    # This is only a bounded reduction challenge, not an
                    # exhaustive search of the stabilizer equivalence class.
                    neighborhood = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:32]
                    best_word = actual
                    best_weight = actual.bit_count()
                    for i, first in enumerate(neighborhood):
                        partial = actual ^ first
                        for second in neighborhood[:i]:
                            proposed = partial ^ second
                            weight = proposed.bit_count()
                            if weight < best_weight:
                                best_word, best_weight = proposed, weight
                    actual = best_word
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    aa = component_pool(pa, "A")
    bb = component_pool(pb, "B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        min(14, ab[0]["d"], ab[1]["d"]),
        ab[0]["fixed_small_free"] and ab[1]["fixed_small_free"],
        min(14, ab[0]["d"]) + min(14, ab[1]["d"]),
        min(18, ab[0]["fixed_d"], ab[1]["fixed_d"]),
        tuple(-(x + y) for x, y in
              zip(ab[0]["cycle_key"], ab[1]["cycle_key"]))),
        reverse=True)

    survivors = []
    for a, b in pairs[:6]:
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 10 * q == 1290
        ar, br = _verified_components(hx, hz, q)
        assert ar == a["rows"] and br == b["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        # Quotient balance is a proposal mechanism; measured full ranks,
        # not an assumed defect formula, enforce the rank-gain target.
        if not (k >= 528 and 15 * n <= 40 * k < 17 * n):
            continue
        logicals = localized_logical_challenge(a, b, hx, hz, n)
        detected = min(14, a["d"], b["d"], *logicals)
        quality = (
            k * detected * detected,
            detected,
            a["fixed_small_free"] and b["fixed_small_free"],
            min(14, a["d"]) + min(14, b["d"]),
            tuple(-(x + y) for x, y in
                  zip(a["cycle_key"], b["cycle_key"])),
        )
        survivors.append(
            (quality, A, B, a, b, hx, hz, n, k, logicals))

    survivors.sort(key=lambda entry: entry[0], reverse=True)
    for quality, A, B, a, b, hx, hz, n, k, logicals in survivors:
        if not _no_kernel4(_syndrome_columns(hx, n)):
            continue
        if not _no_kernel4(_syndrome_columns(hz, n)):
            continue
        assert max(row.bit_count() for row in hx + hz) == 12
        assert _grid_weights(support, support) == 12
        assert css_ok(hx, hz)
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (model, composition, pa, pb, trade),
            "affine_triangle_parameter": triangle,
            "affine_generator_coordinates": (
                a["affine_coordinates"], b["affine_coordinates"]),
            "complement_fixed_kernel4_free": (
                a["fixed_small_free"], b["fixed_small_free"]),
            "commutator_class_profiles": (profiles[pa], profiles[pb]),
            "component_readings": (a["d"], b["d"]),
            "C3_fixed_readings": (a["fixed_d"], b["fixed_d"]),
            "verified_localized_logical_readings": logicals,
            "component_cycles": (a["cycles"], b["cycles"]),
            "component_ranks": (a["rank"], b["rank"]),
            "conservative_selection": quality,
        }
    raise ValueError("balanced commutator pairs missed full rank/rate gates")


def NEW_CTX(rng, G, idx):
    """Twenty three-row support/holonomy cards, using authoritative blocks."""
    q, width = G.q, 9 * G.q
    assert q == 16 and A_SHAPE == B_SHAPE == (3, 9)
    topology, model = divmod(idx % 20, 5)
    cohort = idx // 20
    seed = rng.randrange(1 << 62) ^ (0x163939 + 15485863 * (idx + 1))
    rng = random.Random(seed)
    mask = (1 << q) - 1
    ident = G.idx(G.e)
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q + 1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid metacyclic operation")
    pools = {o: [g for g in range(q) if orders[g] == o]
             for o in (4, 8)}
    assert len(pools[8]) == 4 and pools[4]
    non8 = [g for g in range(q) if g != ident and orders[g] != 8]

    # Cache ONLY actions returned by the verified construction.
    gallery = [[[g] for g in G.elements]]
    xx, _, _ = lifted_matrices(G, gallery, [[[]]])
    _, zz, _ = lifted_matrices(G, [[[]]], gallery)
    blocks = {
        side: [[(row >> (j * q)) & mask for row in rows]
               for j in range(q)]
        for side, rows in (("A", xx), ("B", zz))
    }
    del xx, zz

    special = (2, 4, 6, 8)[topology]
    sa = [[1] * 9 for _ in range(3)]
    sb = [[1] * 9 for _ in range(3)]
    for j in range(special):
        sb[0][j], sb[2][j] = ((2, 0) if j % 2 == 0 else (0, 2))
    assert all(sum(row) == 9 for row in sa + sb)
    assert all(sum(grid[i][j] for i in range(3)) == 3
               for grid in (sa, sb) for j in range(9))
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    eight_bins = (special * model + 2) // 4

    def fresh_a():
        # Fix a spanning-tree gauge internally. Different cards constrain
        # actual loop holonomies, not common group automorphism images.
        for _ in range(256):
            gs = rng.sample(pools[8], model) + rng.sample(non8, 8 - model)
            rng.shuffle(gs)
            gs = [ident] + gs
            hs, used, differences = [ident], {ident}, {ident}
            for g in gs[1:]:
                eligible = [h for h in range(q) if h not in used
                            and table[h][inverse[g]] not in differences]
                if not eligible:
                    break
                h = rng.choice(eligible)
                hs.append(h)
                used.add(h)
                differences.add(table[h][inverse[g]])
            if len(hs) == 9:
                return [[[ident] for _ in range(9)],
                        [[g] for g in gs], [[h] for h in hs]]
        raise ValueError("partial holonomy transversal exhausted")

    def fresh_b():
        for _ in range(512):
            grid = [[[] for _ in range(9)] for _ in range(3)]
            grid[1] = [[ident] for _ in range(9)]
            used = [set(), set()]
            differences = set()
            good = True
            dense = list(range(special, 9))
            rng.shuffle(dense)
            for j in dense:
                options = [(a, b) for a in range(q) if a not in used[0]
                           for b in range(q) if b not in used[1]
                           and table[a][inverse[b]] not in differences]
                if not options:
                    good = False
                    break
                a, b = rng.choice(options)
                grid[0][j], grid[2][j] = [a], [b]
                used[0].add(a)
                used[1].add(b)
                differences.add(table[a][inverse[b]])
            if not good:
                continue
            prescribed = [8] * eight_bins + [4] * (special - eight_bins)
            rng.shuffle(prescribed)
            for j, wanted in enumerate(prescribed):
                side = j % 2
                i = 0 if side == 0 else 2
                options = []
                for a in range(q):
                    if a in used[side]:
                        continue
                    for ratio in pools[wanted]:
                        b = table[a][ratio]
                        if b != a and b not in used[side]:
                            options.append((a, b))
                if not options:
                    good = False
                    break
                a, b = rng.choice(options)
                grid[i][j] = [a, b]
                used[side].update((a, b))
            if good:
                assert [[len(c) for c in row] for row in grid] == sb
                return grid
        raise ValueError("balanced binomial holonomy card exhausted")

    def component(grid, side):
        rows = [0] * (3 * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i * q + r] ^= value << (j * q)
        return rows

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def inspect(grid, side):
        rows = component(grid, side)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        # Native differences only propose this condition; this is the
        # authoritative check, with the evaluator's left/right actions.
        if cycles[0] or not _no_kernel4(columns):
            return None
        assert all(c.bit_count() == 3 for c in columns)
        rank = gf2_rank(rows)
        return dict(grid=grid, rows=rows, columns=columns, rank=rank,
                    cycles=cycles, d=width + 1, words=[])

    def retain(rec, word):
        if not word:
            return
        assert syndrome(rec["columns"], word) == 0
        rec["d"] = min(rec["d"], word.bit_count())
        rec["words"].append(word)
        if len(rec["words"]) > 96:
            rec["words"] = sorted(set(rec["words"]),
                                   key=lambda w: (w.bit_count(), w))[:24]

    def probe(rec, trials, diagnostic_seed):
        """Retained genuine kernel circuits; no absence certificate."""
        local = random.Random(diagnostic_seed)
        columns = rec["columns"]
        order = list(range(width))
        retained = sorted(set(rec["words"]), key=int.bit_count)[:24]
        for turn in range(trials):
            local.shuffle(order)
            if retained and turn % 4 == 3:
                active = list(_bits(local.choice(retained[:6])))
                front = set(local.sample(active, max(1, len(active) // 2)))
                order.sort(key=lambda c: c not in front)
            pivots, basis = {}, []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivots.get(lead)
                    if old is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    retain(rec, word)
            light = sorted(set(basis + retained), key=int.bit_count)[:16]
            moves = basis + retained
            for first in light:
                for second in moves:
                    candidate = first ^ second
                    if candidate and candidate.bit_count() <= rec["d"] + 4:
                        retain(rec, candidate)
            local.shuffle(moves)
            for word in light[:6]:
                for _ in range(3):
                    before = word.bit_count()
                    for move in moves:
                        candidate = word ^ move
                        if candidate and candidate.bit_count() < word.bit_count():
                            word = candidate
                    retain(rec, word)
                    if word.bit_count() == before:
                        break
            rec["words"] = sorted(set(rec["words"]),
                                  key=lambda w: (w.bit_count(), w))[:24]
            retained = rec["words"]
            if rec["d"] <= 8:
                break
        assert rec["words"]

    def parity_challenge(rec, budget, diagnostic_seed):
        """Bounded anchored completion through twelve; found words only."""
        local = random.Random(diagnostic_seed)
        columns = rec["columns"]
        singles, pairs = {}, {}
        incidence = [0] * (3 * q)
        for c, value in enumerate(columns):
            singles.setdefault(value, 1 << c)
            for r in _bits(value):
                incidence[r] |= 1 << c
            for b in range(c):
                pairs.setdefault(value ^ columns[b], (1 << c) | (1 << b))
        priorities = [local.random() for _ in columns]
        roots = list(range(0, width, q))
        local.shuffle(roots)
        for root in roots:
            remaining = [max(1, budget // len(roots))]

            def visit(value, word, allowed):
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
                if not value:
                    retain(rec, word)
                    return
                left = min(12, rec["d"] - 1) - word.bit_count()
                if left <= 0 or value.bit_count() > 3 * left:
                    return
                one = singles.get(value, 0)
                if one and one & allowed == one:
                    retain(rec, word | one)
                pair = pairs.get(value, 0)
                if left >= 2 and pair and pair & allowed == pair:
                    retain(rec, word | pair)
                left = min(12, rec["d"] - 1) - word.bit_count()
                if left <= 2:
                    return
                chosen, smallest = 0, width + 1
                for r in _bits(value):
                    available = incidence[r] & allowed
                    count = available.bit_count()
                    if not count:
                        return
                    if count < smallest:
                        chosen, smallest = available, count
                candidates = sorted(_bits(chosen), key=lambda c: (
                    -(columns[c] & value).bit_count(), priorities[c]))
                live = allowed
                for c in candidates:
                    bit = 1 << c
                    live &= ~bit
                    visit(value ^ columns[c], word | bit, live)
                    if remaining[0] <= 0:
                        break

            # Regular COMPONENT translations permit a zero-coordinate
            # anchor in the first occupied block. No full-code symmetry
            # is assumed. Representative pair tables can omit completions.
            allowed = ((1 << width) - 1) & ~((1 << (root + 1)) - 1)
            visit(columns[root], 1 << root, allowed)
            if rec["d"] <= 6:
                break
        rec["words"] = sorted(set(rec["words"]),
                              key=lambda w: (w.bit_count(), w))[:24]

    def merit(rec):
        return (min(14, rec["d"]), -rec["cycles"][1], rec["d"])

    def diverse(records, limit):
        ordered = sorted(records, key=merit, reverse=True)
        answer, ranks, taken = [], set(), set()
        for rec in ordered:
            if rec["rank"] not in ranks:
                answer.append(rec)
                ranks.add(rec["rank"])
                taken.add(id(rec))
                if len(answer) == limit:
                    return answer
        for rec in ordered:
            if id(rec) not in taken:
                answer.append(rec)
                taken.add(id(rec))
                if len(answer) == limit:
                    break
        return answer

    def component_pool(side, restart):
        records, seen = [], set()
        offset = 100003 * int(side == "B") + 900001 * restart
        fresh = fresh_a if side == "A" else fresh_b
        for attempt in range(160 + 16 * min(cohort, 2)):
            grid = fresh()
            signature = tuple(tuple(sorted(c)) for row in grid for c in row)
            if signature in seen:
                continue
            seen.add(signature)
            rec = inspect(grid, side)
            if rec is not None:
                records.append(rec)
        if not records:
            return []
        # Interleave ranks before witness screening. A cheap geometry
        # optimum must not eliminate every feasible full-rate pairing.
        by_rank = {}
        for rec in sorted(records, key=lambda r: r["cycles"]):
            by_rank.setdefault(rec["rank"], []).append(rec)
        target = 45 if side == "A" else 47
        ranks = sorted(by_rank, key=lambda r: (abs(r - target), -r))
        selected = [
            by_rank[r][turn]
            for turn in range(max(map(len, by_rank.values())))
            for r in ranks if turn < len(by_rank[r])
        ][:12]
        for at, rec in enumerate(selected):
            probe(rec, 12, seed + offset + 1009 * at)
            parity_challenge(rec, 6000, seed + offset + 3100009)
        finalists = diverse(selected, 6)
        for at, rec in enumerate(finalists):
            # Every finalist retains both independent challenges.
            probe(rec, 32, seed + offset + 700001 + 1013 * at)
            probe(rec, 64, seed + offset + 1700003 + 1019 * at)
            parity_challenge(rec, 60000, seed + offset + 2700011 + at)
        return diverse(finalists, 6)

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length() - 1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_readings(a, b, hx, hz, n):
        """Actual strip logicals modulo the ORIGINAL stabilizer spans."""
        readings = []
        for side, rec, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in rec["words"][:4]:
                for fixed in range(9):
                    actual = (sum(
                        ((word >> (j * q)) & mask) << ((9 * j + fixed) * q)
                        for j in range(9)) if side == "A"
                        else word << (fixed * width))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n + 1
            for actual in sorted(candidates, key=int.bit_count)[:8]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in stabilizers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    nearby = sorted(
                        stabilizers, key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    winner = actual
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < winner.bit_count():
                                winner = proposed
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            readings.append(reading)
        return tuple(readings)

    for restart in range(3):
        aa, bb = component_pool("A", restart), component_pool("B", restart)
        proposals = []
        for a in aa:
            for b in bb:
                A = [[[G.elements[g] for g in c] for c in row]
                     for row in a["grid"]]
                B = [[[G.elements[g] for g in c] for c in row]
                     for row in b["grid"]]
                hx, hz, n = lifted_matrices(G, A, B)
                assert n == 90 * q == 1440
                k = n - gf2_rank(hx) - gf2_rank(hz)
                assert k >= 576
                if not (k >= 598 and 375 * n <= 1000 * k < 425 * n):
                    continue
                ar, br = _verified_components(hx, hz, q)
                assert ar == a["rows"] and br == b["rows"]
                detected = min(14, a["d"], b["d"])
                key = (k * detected * detected, detected,
                       -a["cycles"][1] - b["cycles"][1])
                proposals.append((key, A, B, hx, hz, k, a, b))
        proposals.sort(key=lambda p: p[0], reverse=True)
        challenged = []
        for _, A, B, hx, hz, k, a, b in proposals[:8]:
            logicals = logical_readings(a, b, hx, hz, 1440)
            witnessed = min(logicals)
            detected = min(14, a["d"], b["d"],
                           witnessed if witnessed <= 1440 else 12)
            key = (k * detected * detected, detected,
                   -a["cycles"][1] - b["cycles"][1])
            challenged.append((key, A, B, hx, hz, k, a, b, logicals))
        challenged.sort(key=lambda p: p[0], reverse=True)
        for key, A, B, hx, hz, k, a, b, logicals in challenged:
            if not _no_kernel4(_syndrome_columns(hx, 1440)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, 1440)):
                continue
            assert [[len(c) for c in row] for row in A] == sa
            assert [[len(c) for c in row] for row in B] == sb
            assert max(row.bit_count() for row in hx + hz) == 12
            assert css_ok(hx, hz)
            return dict(
                A=A, B=B, idx=idx, k=k, target_cell=0.400,
                card=(topology, model, cohort),
                A_order8_anchor_holonomies=model,
                B_order8_binomial_ratios=eight_bins,
                support_counts=(sa, sb),
                component_ranks=(a["rank"], b["rank"]),
                component_cycles=(a["cycles"], b["cycles"]),
                component_readings=(a["d"], b["d"]),
                verified_strip_logical_readings=logicals,
                selection_reading_cap=14, pairing_quality=key)
    raise ValueError("three-row pairings missed authoritative rank/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return every complete new-shape cell without additional draws."""
    cell = list(ctx[side][i][j])
    assert 0 <= len(cell) <= CELL_MAX
    assert len(cell) == len({G.idx(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term within-coset Sidon supports and independent component descent uniformly suppress sub-30 logical readings: the measured parent still screened slots 18, 26 and 48 at 27, 24 and 21.
# REFUTED: The Z43:Z3 occupancy portfolio with complement-fixed/unrestricted component repair beats fitness 58.9 on (1,3)x(1,3); it measured 58.2256, with four selected k=520 codes and one k=528 code, all reading twelve and retaining abundant sector-1 minimum witnesses.
# REFUTED: Balanced commutator-profile component and localized-logical challenges universally remove weight-ten logicals; 43 of the current sixty slots screened at ten, although the five measured winners retained reading twelve.
# REFUTED: Affine-triangle coupling and forced-anchor min-sum/OSD produce five measured readings >=14 on Z43:Z3 (1,3)x(1,3); the current program instead measured five reading-twelve codes and retained fitness 58.9400.
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
