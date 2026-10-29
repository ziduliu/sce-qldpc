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
CONCEPT = ("Hypothesis: on frozen split-metacyclic Z37:Z3(s=10), order 111, A=B=(3,2), dense defect-eight relative-gap fillings selected against physical-weight C3-even kernel witnesses raise five-code fitness above 358.4778 in cell 0.100. Restricted minima and finite unrestricted searches never certify quantum distance.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged at q=111. This exact tall bed leads cell 0.100 at 358.4778; its wide counterpart reached 351.9696. The action satisfies 10^2=26 and 10^3=1 modulo 37. Since characteristic two does not divide three, complement-fixed coordinates and even-parity coordinates on each complement orbit form distinct invariant restrictions. Their syndromes are extracted from authoritative columns, not from a separately implemented lift.")
SHAPES = ("Keep A_SHAPE=B_SHAPE=(3,2) byte-for-byte unchanged: n=1443, k>=111 and guaranteed rate 1/13 target the open-ended 0.100 cell; q_max=115. Equal raw components have 222 variables/333 checks and equal transpose codes have 333 variables/222 checks. Parent measured minima were sector-two strips: weight64/66 Z witnesses used Arow1 across all Brows, while weight66/70 X witnesses crossed all Arows at Brow1/2. Preserve these component lengths and verify actual k=119, rate 0.08247.")
FILLING = ("Both supports are [[1,3],[3,2],[3,2]], giving row totals 4,5,5, column totals 7,7 and CSS maximum twelve. Twenty cards cross ten unordered pairs of four binomial-gap classes modulo signed action and inversion with two commutator-to-binomial alignments; later cohorts change the alignment. Balanced noncommuting trinomials and pure order-37 binomials preserve the quotient-defect proposal, but authoritative full ranks alone establish k=119. Whole-cell repairs preserve the card. Add overlapping two-coordinate masks on every C3 orbit and search their genuine kernel words by EXPANDED physical weight, retaining all earlier witnesses. Keep normal-fixed, C3-fixed, unrestricted and original-span strip challenges. Widen the pool, cap optimistic component selection at seventy, and give every finalist the same independent deepest retest. The parent's even minimum weights motivate this restriction but do not prove their membership in it. Rank, component agreement, kernel-four, weights and CSS remain binding; no sampled reading establishes distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r010_m37r3s10_32x32_gapcomm_C3even_physical_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    # Same split-metacyclic construction rule, resized for factor 13.
    # 10^2=26 and 10^3=1 mod 37; the action has order three.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 111")
    m, r, s = 37, 3, 10
    els = [(a, b) for b in range(r) for a in range(m)]

    def mul(g, h):
        return ((g[0] + pow(s, g[1], m) * h[0]) % m, (g[1] + h[1]) % r)

    def inv(g):
        bi = (-g[1]) % r
        return ((-pow(s, bi, m) * g[0]) % m, bi)

    return dict(elements=els, ident=(0, 0), mul=mul, inv=inv)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (3, 2)               # (ra, ca)
B_SHAPE = (3, 2)               # (rb, cb)


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


def NEW_CTX(rng, G, idx):
    """Dense gap/commutator cards with physical-weight C3-even challenges."""
    q, width, height = G.q, 3 * G.q, 2 * G.q
    assert q == 111 and A_SHAPE == B_SHAPE == (3, 2)
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    model = card % 2
    # Swapping the two equal-degree binomial rows inverts their ratio.
    # Classes 0,1,2,3 represent the six classes modulo that inversion.
    plans = tuple((a, b) for a in range(4) for b in range(a, 4))
    assert len(plans) == 10
    pa, pb = plans[card // 2]
    sa = [[1, 3], [3, 2], [3, 2]]
    sb = [list(row) for row in sa]
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)
    assert all(sorted(map(sum, support)) == [4, 5, 5]
               for support in (sa, sb))
    assert all(sum(row[j] for row in support) == 7
               for support in (sa, sb) for j in range(2))
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0xC3E37D821 + 1000003 * idx))

    signed_action = {
        sign * pow(10, b, 37) % 37
        for sign in (1, -1) for b in range(3)
    }
    classes = tuple(
        tuple(sorted({pow(2, t, 37) * z % 37 for z in signed_action}))
        for t in range(6)
    )
    assert len(signed_action) == 6
    assert len(set().union(*(set(c) for c in classes))) == 36
    gap_roots = {"A": rng.randrange(1, 37), "B": rng.randrange(1, 37)}
    gap_orbits = {
        side: tuple(
            tuple(sorted({gap_roots[side] * z % 37 for z in orbit}))
            for orbit in classes)
        for side in ("A", "B")
    }
    gap_cards = {"A": pa, "B": pb}
    shift = cohort % 6
    commutator_templates = (
        (shift, shift, shift),
        (shift, (shift + 1) % 6, (shift + 3) % 6),
    )
    triple_positions = ((0, 1), (1, 0), (2, 0))

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
    blocks = {"A": [], "B": []}
    # These are exact transpose monomials extracted from scaffold lifts.
    # Subsequent assembly uses only their GF2 linearity.
    for start in range(0, q, 16):
        ids = list(range(start, min(q, start + 16)))
        gallery = [[[g]] for g in ids]
        _, aa, _ = lifted_matrices(P, gallery, [[[]]])
        bb, _, _ = lifted_matrices(P, [[[]]], gallery)
        assert len(aa) == len(bb) == q
        for j in range(len(ids)):
            for side, source in (("A", aa), ("B", bb)):
                block = [(row >> ((j + 1) * q)) & mask
                         for row in source]
                assert all(value.bit_count() == 1 for value in block)
                blocks[side].append(block)

    def syndrome(columns, word):
        answer = 0
        for c in _bits(word):
            answer ^= columns[c]
        return answer

    def component(grid, side):
        rows = [0] * height
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for r, value in enumerate(blocks[side][G.idx(g)]):
                        rows[j * q + r] ^= value << (i * q)
        return rows

    def legal_cell(cell):
        count = len(cell)
        ids = [G.idx(g) for g in cell]
        if len(set(ids)) != count:
            return False
        left, right = set(), set()
        for a in ids:
            for b in ids:
                if a != b:
                    left.add(table[inverse[a]][b])
                    right.add(table[a][inverse[b]])
        wanted = count * (count - 1)
        if len(left) != wanted or len(right) != wanted:
            return False
        if count == 2:
            # A same-coset difference is a nonzero normal element,
            # hence has order 37 rather than order three.
            return cell[0][1] == cell[1][1]
        if count < 3:
            return True
        ratios = [table[inverse[ids[0]]][g] for g in ids[1:]]
        return any(table[a][b] != table[b][a]
                   for t, a in enumerate(ratios) for b in ratios[:t])

    def normal_class(value, side):
        if value % 37 == 0:
            return None
        return next(t for t, orbit in enumerate(gap_orbits[side])
                    if value % 37 in orbit)

    def commutator_delta(cell):
        ordered = sorted(cell, key=lambda g: g[1])
        ids = [G.idx(g) for g in ordered]
        u = table[inverse[ids[0]]][ids[1]]
        v = table[inverse[ids[0]]][ids[2]]
        comm = table[table[u][v]][inverse[table[v][u]]]
        a, b = G.elements[comm]
        assert b == 0
        return a

    def geometry_ok(cell, i, j, side):
        if len(cell) != sa[i][j] or not legal_cell(cell):
            return False
        if len(cell) == 1:
            return True
        if len(cell) == 2:
            target = 0 if i == 1 else gap_cards[side]
            return normal_class(cell[1][0] - cell[0][0], side) == target
        if sorted(g[1] for g in cell) != [0, 1, 2]:
            return False
        target = commutator_templates[model][
            triple_positions.index((i, j))]
        return normal_class(commutator_delta(cell), side) == target

    def fresh_cell(count, i, j, side):
        if count == 1:
            return [G.pick(rng)]
        if count == 2:
            target = 0 if i == 1 else gap_cards[side]
            origin, b = rng.randrange(37), rng.randrange(3)
            delta = rng.choice(gap_orbits[side][target])
            cell = [(origin, b), ((origin + delta) % 37, b)]
            assert geometry_ok(cell, i, j, side)
            return cell
        assert count == 3
        for _ in range(1024):
            cell = [(rng.randrange(37), b) for b in range(3)]
            if geometry_ok(cell, i, j, side):
                return cell
        raise ValueError("commutator-to-binomial alignment exhausted")

    def fresh(support, side):
        return [[fresh_cell(count, i, j, side)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    partitions = {"A": {}, "B": {}}
    subgroups = {
        37: [G.idx((a, 0)) for a in range(37)],
        3: [G.idx((0, b)) for b in range(3)],
    }
    for side in ("A", "B"):
        for order, subgroup in subgroups.items():
            remaining, orbits = set(range(q)), []
            while remaining:
                root = min(remaining)
                orbit = {
                    table[root][h] if side == "A" else table[h][root]
                    for h in subgroup
                }
                assert len(orbit) == order and orbit <= remaining
                remaining.difference_update(orbit)
                orbits.append(tuple(sorted(orbit)))
            partitions[side][order] = [
                sum(1 << (i * q + c) for c in orbit)
                for i in range(3) for orbit in orbits
            ]

    # Two overlapping pair masks span the even-parity subspace of each
    # complement orbit. Pattern popcount is NOT physical word weight.
    anti_partitions = {}
    for side in ("A", "B"):
        masks_ = []
        for orbit in partitions[side][3]:
            positions = list(_bits(orbit))
            assert len(positions) == 3
            root = 1 << positions[0]
            masks_.extend((root ^ (1 << positions[1]),
                           root ^ (1 << positions[2])))
        anti_partitions[side] = masks_
        assert len(masks_) == 2 * width // 3

    def expand(pattern, groups):
        word = 0
        for c in _bits(pattern):
            word ^= groups[c]
        return word

    def dependencies(columns, order):
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
        return basis

    def normal_word(columns, side):
        """Exact minimum in nine normal-fixed coordinates ONLY."""
        groups = partitions[side][37]
        compressed = [syndrome(columns, group) for group in groups]
        assert len(compressed) == 9
        basis = dependencies(compressed, range(9))
        if not basis:
            return None
        pattern, shortest = 0, None
        for t in range(1, 1 << len(basis)):
            pattern ^= basis[(t & -t).bit_length() - 1]
            if shortest is None or pattern.bit_count() < shortest.bit_count():
                shortest = pattern
        word = expand(shortest, groups)
        assert word.bit_count() == 37 * shortest.bit_count()
        assert syndrome(columns, word) == 0
        return word

    def inspect(grid, side):
        rows = component(grid, side)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        rank = gf2_rank(rows)
        if rank < height - 4:
            return None
        nw = normal_word(columns, side)
        if nw is None or nw.bit_count() < 74:
            return None
        groups = partitions[side][3]
        fixed = [syndrome(columns, group) for group in groups]
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cycles": cycles, "eight": None,
            "normal": nw.bit_count(), "normal_word": nw,
            "fixed": fixed, "groups": groups, "fixed_beam": [],
            "fixed_d": width + 1, "beam": [nw],
            "anti_groups": anti_partitions[side],
            "anti_columns": [
                syndrome(columns, group)
                for group in anti_partitions[side]],
            "anti_beam": [], "anti_d": width + 1,
            "d": nw.bit_count(), "word": nw,
        }

    def eight_cycles(record):
        """Exact simple Tanner C8 count; diagnostic geometry, not distance."""
        if record["eight"] is not None:
            return record["eight"]
        checks = [list(_bits(row)) for row in record["rows"]]
        neighbors = [list(_bits(col)) for col in record["columns"]]
        rooted = 0
        for root in range(0, width, q):
            paths = {}
            for c1 in neighbors[root]:
                for middle in checks[c1]:
                    if middle == root:
                        continue
                    for c2 in neighbors[middle]:
                        if c2 == c1:
                            continue
                        for end in checks[c2]:
                            if end == root or end == middle:
                                continue
                            previous = paths.setdefault(end, [])
                            for d1, old_middle, d2 in previous:
                                if (old_middle != middle and
                                        c1 != d1 and c1 != d2 and
                                        c2 != d1 and c2 != d2):
                                    rooted += 1
                            previous.append((c1, middle, c2))
        assert q * rooted % 4 == 0
        record["eight"] = q * rooted // 4
        return record["eight"]

    def kernel_search(columns, local, trials, initial=(), stop=24):
        """Accumulated actual kernel witnesses, never an exclusion claim."""
        m = len(columns)
        beam = sorted({w for w in initial if w},
                      key=lambda w: (w.bit_count(), w))[:24]
        reading, witness = m + 1, None

        def note(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        for word in beam:
            assert syndrome(columns, word) == 0
            note(word)
        order = list(range(m))
        for turn in range(trials):
            local.shuffle(order)
            if turn % 4 == 3 and beam:
                active = list(_bits(local.choice(beam[:6])))
                local.shuffle(active)
                preferred = set(active[:max(1, len(active) // 2)])
                order.sort(key=lambda c: c not in preferred)
            basis = dependencies(columns, order)
            for word in basis:
                note(word)
            light = sorted(set(basis + beam),
                           key=lambda w: (w.bit_count(), w))[:16]
            moves, grown = basis + beam, list(light) + beam
            for first in light:
                for second in moves:
                    candidate = first ^ second
                    note(candidate)
                    if candidate and candidate.bit_count() <= reading + 6:
                        grown.append(candidate)
                if len(grown) > 256:
                    grown = sorted(set(grown),
                                   key=lambda w: (w.bit_count(), w))[:48]
            short = sorted(set(grown),
                           key=lambda w: (w.bit_count(), w))[:24]
            for i, first in enumerate(short):
                for second in short[:i]:
                    candidate = first ^ second
                    note(candidate)
                    if candidate and candidate.bit_count() <= reading + 4:
                        grown.append(candidate)
            local.shuffle(moves)
            for initial_word in short[:6]:
                word = initial_word
                for sweep in range(3):
                    changed = False
                    for move in moves:
                        candidate = word ^ move
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    note(word)
                    if not changed:
                        break
                grown.append(word)
            if witness is not None:
                grown.append(witness)
            beam = sorted({w for w in grown if w},
                          key=lambda w: (w.bit_count(), w))[:24]
            if reading <= stop:
                break
        if witness is None:
            raise ValueError("no verified tall-component kernel witness")
        assert syndrome(columns, witness) == 0
        return reading, witness, beam

    def anti_kernel_search(record, local, trials):
        """Genuine C3-even words, ranked in ORIGINAL physical coordinates.

        Overlapping masks make compressed Hamming weight inappropriate.
        Elimination tracks compressed dependencies, then every comparison
        and cancellation operates on their expanded physical words.
        """
        columns = record["anti_columns"]
        masks_ = record["anti_groups"]
        beam = sorted(set(record["anti_beam"]),
                      key=lambda w: (w.bit_count(), w))[:24]
        reading, witness = width + 1, None

        def note(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        for word in beam:
            assert syndrome(record["columns"], word) == 0
            note(word)
        order = list(range(len(columns)))
        for turn in range(trials):
            local.shuffle(order)
            patterns = dependencies(columns, order)
            basis = [expand(pattern, masks_) for pattern in patterns]
            if not basis:
                raise ValueError("C3-even restriction unexpectedly has no kernel")
            for word in basis:
                note(word)
            grown = list(beam) + sorted(
                set(basis), key=lambda w: (w.bit_count(), w))[:16]

            # Complete pairs of ONE sampled restricted basis, not a
            # complete search of this restricted kernel.
            for i, first in enumerate(basis):
                for second in basis[:i]:
                    candidate = first ^ second
                    note(candidate)
                    if candidate and candidate.bit_count() <= reading + 6:
                        grown.append(candidate)
                if len(grown) > 256:
                    grown = sorted(
                        set(grown), key=lambda w: (w.bit_count(), w))[:64]

            light = sorted(set(grown + basis),
                           key=lambda w: (w.bit_count(), w))[:16]
            moves = basis + beam
            for first in light:
                for second in moves:
                    candidate = first ^ second
                    note(candidate)
                    if candidate and candidate.bit_count() <= reading + 6:
                        grown.append(candidate)
                if len(grown) > 256:
                    grown = sorted(
                        set(grown), key=lambda w: (w.bit_count(), w))[:64]
            shortened = sorted(
                set(grown), key=lambda w: (w.bit_count(), w))[:24]
            for i, first in enumerate(shortened):
                for second in shortened[:i]:
                    candidate = first ^ second
                    note(candidate)
                    if candidate and candidate.bit_count() <= reading + 4:
                        grown.append(candidate)

            local.shuffle(moves)
            for initial in shortened[:4]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for move in moves:
                        candidate = word ^ move
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    note(word)
                    if not changed:
                        break
                grown.append(word)
            grown.append(witness)
            beam = sorted(
                {w for w in grown if w},
                key=lambda w: (w.bit_count(), w))[:24]
            if reading <= 60:
                break

        assert witness is not None
        assert syndrome(record["columns"], witness) == 0
        assert all((witness & group).bit_count() % 2 == 0
                   for group in record["groups"])
        record["anti_beam"] = beam
        record["anti_d"] = min(record["anti_d"], reading)
        return witness, beam

    def optimize(side, support):
        bank, seen = set(), set()
        offset = 0 if side == "A" else 100003

        def signature(grid):
            return tuple(tuple(sorted(G.idx(g) for g in cell))
                         for row in grid for cell in row)

        def probe(record, turn, full_trials, fixed_trials):
            fd, fw, fbeam = kernel_search(
                record["fixed"],
                random.Random(seed + offset + 1700003 + 900001 * turn),
                fixed_trials, record["fixed_beam"], stop=8)
            record["fixed_beam"] = fbeam
            actuals = [expand(w, record["groups"]) for w in fbeam[:6]]
            actual = expand(fw, record["groups"])
            actuals.append(actual)
            assert actual.bit_count() == 3 * fd
            assert all(syndrome(record["columns"], w) == 0
                       for w in actuals)
            record["fixed_d"] = min(record["fixed_d"], 3 * fd)
            anti_word, anti_beam = anti_kernel_search(
                record,
                random.Random(seed + offset + 3700033 + 1100009 * turn),
                max(8, min(64, fixed_trials // 2)))
            actuals.extend(anti_beam[:6])
            actuals.append(anti_word)
            assert all(syndrome(record["columns"], w) == 0
                       for w in actuals)
            transferred = [
                w for w in bank
                if syndrome(record["columns"], w) == 0]
            d, word, beam = kernel_search(
                record["columns"],
                random.Random(seed + offset + 700001 * turn),
                full_trials, record["beam"] + actuals + transferred)
            record["beam"] = beam
            if d < record["d"]:
                record["d"], record["word"] = d, word
            assert all((row & record["word"]).bit_count() % 2 == 0
                       for row in record["rows"])
            bank.update(beam[:6])
            bank.add(record["word"])
            if len(bank) > 96:
                kept = sorted(bank,
                              key=lambda w: (w.bit_count(), w))[:64]
                bank.clear()
                bank.update(kept)

        def merit(record):
            for word in bank:
                if (word.bit_count() < record["d"] and
                        syndrome(record["columns"], word) == 0):
                    record["d"], record["word"] = word.bit_count(), word
                    record["beam"].append(word)
            four, six = record["cycles"]
            geometry = (8 + 4 * model) * six + eight_cycles(record)
            # The cap reduces selection pressure on optimistic readings.
            # It is not a certified lower bound.
            return (min(70, record["d"]),
                    min(72, record["anti_d"]), -four, -geometry,
                    record["d"], min(111, record["normal"]),
                    record["rank"])

        for restart in range(3):
            pool = []
            for trial in range(128):
                grid = fresh(support, side)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                record = inspect(grid, side)
                if record is not None:
                    pool.append(record)
            pool.sort(key=lambda r: (
                r["cycles"][0], -min(111, r["normal"]),
                r["cycles"][1], -r["rank"]))
            selected = []
            for record in pool:
                if not _no_kernel4(record["columns"]):
                    continue
                probe(record, 1 + restart, 8, 16)
                selected.append(record)
                if len(selected) == 10:
                    break
            if not selected:
                continue
            current = max(selected, key=merit)

            for step in range(24):
                grid = [[list(cell) for cell in row]
                        for row in current["grid"]]
                if step % 7 == 0:
                    grid = fresh(support, side)
                else:
                    occupied = list(_bits(current["word"]))
                    i = (rng.choice(occupied) // q if occupied
                         else rng.randrange(3))
                    j = rng.randrange(2)
                    # Binomial replacement moves its origin while retaining
                    # the prescribed gap class; single-term edits would not.
                    if step % 3 == 0 or len(grid[i][j]) <= 2:
                        grid[i][j] = fresh_cell(support[i][j], i, j, side)
                    else:
                        t = rng.randrange(len(grid[i][j]))
                        old = grid[i][j][t]
                        available = []
                        for a in range(37):
                            candidate = (a, old[1])
                            if candidate in grid[i][j]:
                                continue
                            changed = list(grid[i][j])
                            changed[t] = candidate
                            if geometry_ok(changed, i, j, side):
                                available.append(candidate)
                        if not available:
                            continue
                        grid[i][j][t] = rng.choice(available)
                        assert geometry_ok(grid[i][j], i, j, side)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                if proposed["cycles"][0] > current["cycles"][0] + q:
                    continue
                if not _no_kernel4(proposed["columns"]):
                    continue
                probe(proposed, 20 + step, 6, 12)
                selected.append(proposed)
                pk, ck = merit(proposed), merit(current)
                if pk >= ck or (pk[0] == ck[0] and rng.random() < 0.10):
                    current = proposed

            selected.sort(key=merit, reverse=True)
            finalists = selected[:3]
            for number, record in enumerate(finalists):
                probe(record, 101 + number, 40, 64)
            finalists.sort(key=merit, reverse=True)
            # Retest ALL contenders before the final sort: a shallower
            # third record must not replace a newly weakened deep-tested one.
            for number, record in enumerate(finalists):
                probe(record, 211 + number,
                      128 + 8 * (cohort % 3), 128)
            finalists.sort(key=merit, reverse=True)
            return finalists[:2]
        raise ValueError("no positive tall component survived screening")

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

    def strip_reading(a, b, hx, hz, n):
        """Actual nontrivial logicals modulo ORIGINAL stabilizer spans."""
        reading, weights = n + 1, []
        for kind, record, opposite, stabilizers in (
                ("X", a, hz, hx), ("Z", b, hx, hz)):
            pivots = span(stabilizers)
            sources = sorted(
                set(record["beam"] + [record["word"],
                                      record["normal_word"]]),
                key=lambda w: (w.bit_count(), w))[:6]
            candidates = set()
            for word in sources:
                assert syndrome(record["columns"], word) == 0
                for location in range(3):
                    if kind == "X":
                        actual = sum(
                            ((word >> (i * q)) & mask)
                            << ((4 + 3 * i + location) * q)
                            for i in range(3))
                    else:
                        actual = word << ((4 + 3 * location) * q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            local = random.Random(
                seed + (4100009 if kind == "X" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
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
                        reverse=True)[:24]
                    best = actual
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < best.bit_count():
                                best = proposed
                    actual = best
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
                weights.append((kind, actual.bit_count()))
        # n+1 means no nontrivial strip was found, never an exclusion.
        return reading, weights

    aa, bb = optimize("A", sa), optimize("B", sb)
    survivors = []
    for a in aa:
        for b in bb:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 13 * q == 1443
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k == q + 8 and 8 * k < n):
                continue
            assert max(row.bit_count() for row in hx + hz) == 12
            # Verify both cached components against this FINAL pairing.
            actual_a = [
                sum(((hz[(3 * j) * q + t]
                      >> ((4 + 3 * i) * q)) & mask) << (i * q)
                    for i in range(3))
                for j in range(2) for t in range(q)
            ]
            actual_b = [
                (hx[beta * q + t] >> (4 * q)) & ((1 << width) - 1)
                for beta in range(2) for t in range(q)
            ]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            logical_d, weights = strip_reading(a, b, hx, hz, n)
            detected = min(a["d"], b["d"], logical_d)
            capped = min(70, detected)
            geometry = sum(
                (8 + 4 * model) * r["cycles"][1] + eight_cycles(r)
                for r in (a, b))
            key = (
                k * capped * capped, detected,
                -a["cycles"][0] - b["cycles"][0],
                -geometry, a["d"] + b["d"],
            )
            survivors.append(
                (key, a, b, hx, hz, k, logical_d, weights))
    survivors.sort(key=lambda item: item[0], reverse=True)
    for key, a, b, hx, hz, k, logical_d, weights in survivors:
        if not _no_kernel4(_syndrome_columns(hx, 1443)):
            continue
        if not _no_kernel4(_syndrome_columns(hz, 1443)):
            continue
        assert css_ok(hx, hz)
        assert all(
            geometry_ok(cell, i, j, side)
            for side, grid in (("A", a["grid"]), ("B", b["grid"]))
            for i, row in enumerate(grid)
            for j, cell in enumerate(row))
        return {
            "A": a["grid"], "B": b["grid"], "idx": idx, "k": k,
            "card": (pa, pb, model, cohort),
            "supports": (sa, sb),
            "relative_binomial_gap_classes": (pa, pb),
            "commutator_to_gap_template": commutator_templates[model],
            "component_ranks": (a["rank"], b["rank"]),
            "component_readings": (a["d"], b["d"]),
            "normal_fixed_exact_minima": (a["normal"], b["normal"]),
            "C3_fixed_readings": (a["fixed_d"], b["fixed_d"]),
            "C3_even_physical_readings": (a["anti_d"], b["anti_d"]),
            "component_cycles": (
                a["cycles"] + (a["eight"],),
                b["cycles"] + (b["eight"],)),
            "verified_strip_logical_reading": logical_d,
            "verified_strip_weights": weights,
            "pairing_merit": key,
        }
    raise ValueError("tall pairings missed low-rate/full-validity gates")


def _wide_reference_ctx(rng, G, idx):
    """Inactive previous wide generator; NEW_CTX above defines every slot."""
    q, width = G.q, 2 * G.q
    mask = (1 << q) - 1
    assert q == 291 and A_SHAPE == B_SHAPE == (1, 2)
    support = [[4, 4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    card, cohort = idx % 20, idx // 20
    profiles = (
        ((2, 1, 1), (2, 1, 1)),
        ((2, 1, 1), (3, 1, 0)),
        ((2, 1, 1), (2, 2, 0)),
        ((3, 1, 0), (3, 1, 0)),
        ((3, 1, 0), (2, 2, 0)),
    )
    profile_a = card // 4
    profile_b = (0, 1, 2, 3, 4)[profile_a]
    ratio_class = 1 + card % 4 + 4 * (cohort % 3)
    primitive = next(
        a for a in range(2, 97)
        if pow(a, 48, 97) != 1 and pow(a, 32, 97) != 1)
    ratio = pow(primitive, ratio_class, 97)
    diagnostic_seed = rng.randrange(1 << 62)

    # Cache native operations only. Monomial matrices below are obtained
    # from the authoritative lift, not from a separately written action.
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

    # Tall galleries keep the unused matrices narrow. Their sector-two
    # blocks are verified transpose monomials; transpose those exact
    # blocks to obtain the original wide-component monomials.
    gallery = [[[g]] for g in range(q)]
    zero = [[[]]]
    _, a_transpose, _ = lifted_matrices(P, gallery, zero)
    b_transpose, _, _ = lifted_matrices(P, zero, gallery)
    blocks = {}
    for side, raw in (("A", a_transpose), ("B", b_transpose)):
        monomials = []
        for g in range(q):
            rows = [0] * q
            shift = (g + 1) * q
            for r, source in enumerate(raw):
                value = (source >> shift) & mask
                assert value.bit_count() == 1
                rows[value.bit_length() - 1] |= 1 << r
            monomials.append(rows)
        blocks[side] = monomials
    del gallery, a_transpose, b_transpose

    def difference_free(grid):
        """Disjoint directed differences across both cells, both orientations."""
        left, right = set(), set()
        for cell in grid[0]:
            ids = [G.idx(g) for g in cell]
            if len(ids) != 4 or len(set(ids)) != 4:
                return False
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    l = table[inverse[a]][b]
                    r = table[a][inverse[b]]
                    if l in left or r in right:
                        return False
                    left.add(l)
                    right.add(r)
        return len(left) == len(right) == 24

    def fresh(profile, delta=None):
        """Coupled gap classes are preserved, not common voltage gauges."""
        for _ in range(1024):
            base = rng.randrange(1, 97) if delta is None else delta
            cells = []
            for j, occupancy in enumerate(profiles[profile]):
                fibers = list(range(3))
                rng.shuffle(fibers)
                cell = []
                core = True
                for b, count in zip(fibers, occupancy):
                    if not count:
                        continue
                    origin = rng.randrange(97)
                    if count == 1:
                        aa = [origin]
                    else:
                        if core:
                            gap = base * (1 if j == 0 else ratio) % 97
                            gap = gap * rng.choice((1, 35, 61)) % 97
                            if rng.randrange(2):
                                gap = (-gap) % 97
                        else:
                            gap = rng.randrange(1, 97)
                        aa = [origin, (origin + gap) % 97]
                        if count == 3:
                            # Exclude all three-point arithmetic progressions.
                            t = rng.choice([v for v in range(3, 96)
                                            if (2*v - 1) % 97])
                            aa.append((origin + t*gap) % 97)
                        core = False
                    cell.extend((a, b) for a in aa)
                assert len(cell) == len(set(cell)) == 4
                cells.append(cell)
            grid = [cells]
            if difference_free(grid):
                return grid, base
        raise ValueError("bidirectional gap card could not be filled")

    # H=<y> has order three. A-left components commute with right
    # translations, and B-right components with left translations.
    # Even without using that symmetry theorem, XORing authoritative
    # columns and expanding a returned word verifies every restriction.
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
            assert len(set(orbit)) == 3
            assert set(orbit) <= remaining
            remaining.difference_update(orbit)
            orbits.append(orbit)
        assert len(orbits) == 97
        partitions[side] = [
            sum(1 << (j*q + c) for c in orbit)
            for j in range(2) for orbit in orbits
        ]

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

    def inspect(grid, side, delta):
        ids = [[G.idx(g) for g in cell] for cell in grid[0]]
        rows = []
        for r in range(q):
            row = 0
            for j, cell in enumerate(ids):
                block = 0
                for g in cell:
                    block ^= blocks[side][g][r]
                row |= block << (j*q)
            rows.append(row)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        if cycles[0]:
            return None
        groups = partitions[side]
        fixed = [syndrome(columns, group) for group in groups]
        return {
            "grid": grid, "delta": delta, "rows": rows,
            "columns": columns, "cycles": cycles,
            "fixed": fixed, "groups": groups,
            "d": width + 1, "fixed_d": width + 1,
            "beam": [], "fixed_beam": [], "word": None,
        }

    def rank_gate(record):
        if "rank" in record:
            return True
        rows = record["rows"]
        rank = gf2_rank(rows)
        cell_ranks = (
            gf2_rank([row & mask for row in rows]),
            gf2_rank([row >> q for row in rows]),
        )
        if rank < q - 3 or min(cell_ranks) < q - 3:
            return False
        if not _no_kernel4(record["columns"]):
            return False
        record["rank"], record["cell_ranks"] = rank, cell_ranks
        record["fixed_rank"] = gf2_rank(record["fixed"])
        return True

    def kernel_search(columns, local, trials, seeds=(), stop=24):
        """Cross-basis cancellation of genuine, nonzero kernel words.

        The retained beam persists between randomized elimination bases.
        Restricted-space results are still upper readings, not exclusions.
        """
        m = len(columns)
        beam = sorted(set(seeds), key=int.bit_count)[:32]
        reading, witness = m + 1, None

        def observe(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        for word in beam:
            assert syndrome(columns, word) == 0
            observe(word)

        order = list(range(m))
        for trial in range(trials):
            local.shuffle(order)
            if trial % 5 == 4 and beam:
                # Challenge a retained support by admitting it early,
                # with randomized outside coordinates still available.
                anchor = local.choice(beam[:8])
                order.sort(key=lambda c: not ((anchor >> c) & 1))
            pivot_values = [0] * q
            pivot_words = [0] * q
            basis = []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivot_values[lead]
                    if not old:
                        pivot_values[lead] = value
                        pivot_words[lead] = word
                        break
                    value ^= old
                    word ^= pivot_words[lead]
                if not value:
                    basis.append(word)
                    observe(word)
            if not basis:
                continue

            light = sorted(set(basis + beam), key=int.bit_count)[:16]
            moves = basis + beam
            grown = list(light) + beam
            for a in light:
                for b in moves:
                    word = a ^ b
                    observe(word)
                    if word and word.bit_count() <= reading + 8:
                        grown.append(word)
                if len(grown) > 192:
                    grown = sorted(set(grown), key=int.bit_count)[:48]

            # Combining newly shortened words reaches combinations of
            # several fundamental words, including from different bases.
            shortened = sorted(set(grown), key=int.bit_count)[:32]
            for i, a in enumerate(shortened):
                for b in shortened[:i]:
                    word = a ^ b
                    observe(word)
                    if word and word.bit_count() <= reading + 6:
                        grown.append(word)

            local.shuffle(moves)
            for start in shortened[:6]:
                word, weight = start, start.bit_count()
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    observe(word)
                    if not changed:
                        break
                grown.append(word)
            if witness is not None:
                grown.append(witness)
            beam = sorted(set(grown), key=int.bit_count)[:32]
            if reading <= stop:
                break

        if witness is None:
            raise ValueError("no actual component kernel witness")
        assert syndrome(columns, witness) == 0
        return reading, witness, beam

    def optimize(profile, side):
        bank = set()
        seen = set()
        side_offset = 0 if side == "A" else 100003

        def signature(grid):
            return tuple(tuple(sorted(G.idx(g) for g in cell))
                         for cell in grid[0])

        def probe(record, turn, full_trials, fixed_trials):
            transferred = [
                word for word in bank
                if syndrome(record["columns"], word) == 0
            ]
            local = random.Random(
                diagnostic_seed + side_offset + 700001*turn)
            d, word, beam = kernel_search(
                record["columns"], local, full_trials,
                record["beam"] + transferred)
            record["beam"] = beam
            local = random.Random(
                diagnostic_seed + side_offset + 1700003 + 900001*turn)
            fd, fw, fixed_beam = kernel_search(
                record["fixed"], local, fixed_trials,
                record["fixed_beam"], stop=8)
            record["fixed_beam"] = fixed_beam
            actual = expand(fw, record["groups"])
            assert actual.bit_count() == 3 * fw.bit_count()
            assert syndrome(record["columns"], actual) == 0
            assert all((row & actual).bit_count() % 2 == 0
                       for row in record["rows"])
            record["fixed_d"] = min(record["fixed_d"], 3*fd)
            if 3*fd < d:
                d, word = 3*fd, actual
            if d < record["d"]:
                record["d"], record["word"] = d, word
            bank.update((word, actual))
            record["beam"] = sorted(
                set(record["beam"] + [actual, record["word"]]),
                key=int.bit_count)[:32]
            if len(bank) > 64:
                keep = sorted(bank, key=int.bit_count)[:48]
                bank.clear()
                bank.update(keep)

        def merit(record):
            for word in bank:
                if (word.bit_count() < record["d"]
                        and syndrome(record["columns"], word) == 0):
                    record["d"], record["word"] = word.bit_count(), word
            # Both distance terms are witnessed upper readings.
            return (record["d"], record["fixed_d"],
                    -record["cycles"][1], record["rank"])

        for restart in range(3):
            records = []
            for _ in range(48):
                grid, delta = fresh(profile)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                record = inspect(grid, side, delta)
                if record is not None:
                    records.append(record)
            records.sort(key=lambda record: record["cycles"])

            selected = []
            for record in records:
                if not rank_gate(record):
                    continue
                probe(record, 1 + restart, 8, 24)
                selected.append(record)
                if len(selected) >= 6:
                    break
            if not selected:
                continue
            current = max(selected, key=merit)

            # Repair cells implicated by actual surviving component words.
            # Replacements retain the normal-gap card and all four terms.
            for step in range(20):
                grid, delta = fresh(profile, current["delta"])
                if step % 5:
                    used = list(_bits(current["word"]))
                    j = rng.choice(used) // q if used else rng.randrange(2)
                    combined = [[list(cell) for cell in current["grid"][0]]]
                    combined[0][j] = list(grid[0][j])
                    grid = combined
                    if not difference_free(grid):
                        continue
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                proposed = inspect(grid, side, delta)
                if proposed is None:
                    continue
                if proposed["cycles"][1] > current["cycles"][1] + 6*q:
                    continue
                if not rank_gate(proposed):
                    continue
                probe(proposed, 11 + step, 4, 12)
                pk, ck = merit(proposed), merit(current)
                if pk > ck or (pk[0] == ck[0] and rng.random() < 0.10):
                    current = proposed
                selected.append(proposed)

            selected.sort(key=merit, reverse=True)
            finalists = selected[:4]
            for record in finalists:
                probe(record, 101, 24, 96)
            finalists.sort(key=merit, reverse=True)
            finalists = finalists[:2]
            for record in finalists:
                # Independent deeper challenges retain every earlier
                # witness; a restricted search never certifies absence.
                probe(record, 211, 96, 192)
            finalists.sort(key=merit, reverse=True)
            return finalists
        raise ValueError("no bidirectional C3-challenged component survived")

    aa = optimize(profile_a, "A")
    bb = optimize(profile_b, "B")
    pairs = [(a, b) for a in aa for b in bb]
    pairs.sort(key=lambda ab: (
        min(ab[0]["d"], ab[1]["d"]),
        ab[0]["d"] + ab[1]["d"],
        min(ab[0]["fixed_d"], ab[1]["fixed_d"]),
        -ab[0]["cycles"][1] - ab[1]["cycles"][1]),
        reverse=True)

    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 5*q == 1455
        ar, br = _verified_components(hx, hz, q)
        assert ar == a["rows"] and br == b["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= q and 7*n <= 40*k < 9*n):
            continue
        if not _no_kernel4(_syndrome_columns(hx, n)):
            continue
        if not _no_kernel4(_syndrome_columns(hz, n)):
            continue
        assert max(row.bit_count() for row in hx + hz) == 12
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS")
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (profile_a, profile_b, ratio_class),
            "component_readings": (a["d"], b["d"]),
            "C3_fixed_readings": (a["fixed_d"], b["fixed_d"]),
            "C3_fixed_ranks": (a["fixed_rank"], b["fixed_rank"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
        }
    raise ValueError("C3-challenged pairings missed full validity gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete gap/commutator and C3-even-challenged cell."""
    cell = list(ctx[side][i][j])
    assert len(cell) == ctx["supports"][int(side == "B")][i][j]
    assert len(cell) == len({G.idx(g) for g in cell})
    if len(cell) == 2:
        assert cell[0][1] == cell[1][1]
    if len(cell) == 3:
        assert sorted(g[1] for g in cell) == [0, 1, 2]
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The current parent's degree-three allocations match its four-term allocations on this bed; every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The current parent's six-trial component selection prevents sub-50 logical readings; slot 48 fell from d2s=67 to d30s=42, with a sector-1 Z witness using both Acols at Bcol0.
# REFUTED: Bidirectionally cycle-free four-term wide components with C3-fixed cross-basis gap-card challenges raise five-code fitness above 965.8564; the measured current program instead scored 873.5244 with selected readings 67,68,64,67,62.
# REFUTED: Normal-fixed/C3-fixed component challenges and strip reduction make screening readings >=70 exclude logicals below68 on this tall order-111 bed; current slot20 fell from70 to64, and slots0,22,40 returned66, with all reported minima in sector two.
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
