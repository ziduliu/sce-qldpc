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
CONCEPT = ("Hypothesis: split-metacyclic Z37:Z3(s=10), order 111, A=(2,1), B=(5,2), with balanced B-column budgets and independently challenged transpose components produces five NEW measured readings >=24 in cell 0.250. At guaranteed k/n=0.250 this gives fitness >=144, above the cell's 124.3 incumbent. All finite readings remain provisional upper bounds.")
GROUP = ("Keep the split-metacyclic construction RULE, resizing only m,r,s to 37,3,10 and q to 111 for factor-twelve shapes. The action is valid because 10^2=26 and 10^3=1 modulo 37. This parameter group reached measured fitness 358.5 on (3,2)x(3,2) and 352.0 on (2,3)x(2,3); those results motivate this untried bed, not certified distance transfer.")
SHAPES = ("Change A to (2,1) and B to (5,2). n=12*111=1332 and k>=3*111=333 guarantee rate 0.250; q_max=124. Distance-bearing transpose components have 222 variables/111 checks and 555 variables/222 checks, with nominal rates one-half and three-fifths. Parent minima were sector-two strips across both Arows at one Brow or both Brows at one Arow, with selected readings 61..69. Challenge analogous longer B strips and sector-one raw-kernel embeddings. Exact full ranks enforce 0.225<=k/n<0.275.")
FILLING = ("Twenty cards cross five support allocations with four binding coset policies, not twenty relabelings. Cubic A permits B-column totals nine; mixed and quartic A use totals eight. B-row totals are three or four, so both CSS maxima remain <=12. Native bidirectional Sidon cells distinguish normal and complement binomials and full versus doubled cubic cosets. Scaffold-derived transpose blocks supply cycles, localized kernels and accumulated unrestricted witnesses. Repair occupied cells without erasing the card and independently retest finalists. Pairings compete using exactly commuting non-stabilizer logicals from BOTH sectors, reduced by ORIGINAL stabilizers. Final native assembly, ranks, kernel-four exclusion, distinctness, weights and CSS are gates; no sampled reading certifies distance.")
HOW_AB = FILLING
FAMILY_NAME = "r025_split_m37r3s10_21x52_balanced_dualsector_repair_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    # SAME split-metacyclic construction rule, resized for factor twelve.
    # 10^2=26 and 10^3=1 mod 37; the action has exact order three.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 111")
    m, r, s = 37, 3, 10
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

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (5, 2)               # (rb, cb)


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


def _legacy_isotypic_ctx(rng, G, idx):
    """Inactive order-291 reference; the new factor-twelve NEW_CTX follows."""
    q, width = G.q, 2 * G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (2, 1)
    support = [[4], [4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    mask = (1 << q) - 1

    profiles = (
        ((2, 1, 1), (2, 1, 1)),
        ((2, 1, 1), (3, 1, 0)),
        ((3, 1, 0), (3, 1, 0)),
        ((2, 1, 1), (2, 2, 0)),
        ((3, 1, 0), (2, 2, 0)),
    )
    plans = ((0, 0), (0, 1), (1, 2), (0, 3), (1, 4))
    card, cohort = idx % 20, idx // 20
    pa, pb = plans[card // 4]
    trade = card % 4
    seed = rng.randrange(1 << 62)
    # New proposal streams, not archived labelings followed by relabeling.
    rng = random.Random(seed + 104729 * (idx + 1) + 3100009)
    triangle = (19, 23, 29, 31, 37, 41, 43, 47)[
        (trade + 4 * cohort) % 8]

    # Cache native group operations only. Every matrix below is built
    # by lifted_matrices, never by a separately implemented action.
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

    probe_group = Probe()

    # Restrictions of the AUTHORITATIVE transpose-component coordinates.
    # Each C3 orbit splits over F2 into its fixed line and even plane.
    # The two even basis masks overlap; their physical weight must NOT
    # be replaced by twice the binary coordinate weight.
    module_masks = {}
    orbit_count = width // 3
    orbit_low_mask = (1 << orbit_count) - 1
    h = G.idx((0, 1))
    assert table[table[h][h]][h] == identity
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
            closing = (table[orbit[-1]][h] if side == "A"
                       else table[h][orbit[-1]])
            assert closing == root and len(set(orbit)) == 3
            assert set(orbit) <= remaining
            remaining.difference_update(orbit)
            orbits.append(orbit)
        assert len(orbits) == q // 3
        fixed, first, second = [], [], []
        for j in range(2):
            for orbit in orbits:
                a, b, c = [1 << (j * q + t) for t in orbit]
                fixed.append(a | b | c)
                first.append(a | b)
                second.append(b | c)
        assert len(fixed) == orbit_count
        module_masks[side] = {
            "fixed": fixed,
            "even": first + second,
        }

    def whole_sidon(cell):
        """Exclude repeated directed differences in both orientations."""
        if len(cell) != 4 or len(set(cell)) != 4:
            return False
        ids = [G.idx(g) for g in cell]
        left, right = set(), set()
        for a in ids:
            for b in ids:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        return len(left) == len(right) == 12

    def fresh_cell(occupancy):
        for _ in range(512):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, count in zip(fibers, occupancy):
                if count == 3:
                    origin, scale = rng.randrange(97), rng.randrange(1, 97)
                    rotations = [origin, (origin + scale) % 97,
                                 (origin + triangle * scale) % 97]
                else:
                    rotations = rng.sample(range(97), count)
                cell.extend((a, b) for a in rotations)
            if whole_sidon(cell):
                return cell
        raise ValueError("whole-cell Sidon profile could not be filled")

    def difference_free(grid):
        """Disjoint whole-cell differences across BOTH component blocks."""
        left, right = set(), set()
        for row in grid:
            ids = [G.idx(g) for g in row[0]]
            if len(ids) != 4 or len(set(ids)) != 4:
                return False
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    l = table[inverses[a]][b]
                    r = table[a][inverses[b]]
                    if l in left or r in right:
                        return False
                    left.add(l)
                    right.add(r)
        return len(left) == len(right) == 24

    def fresh(profile):
        for _ in range(1024):
            grid = [[fresh_cell(occupancy)]
                    for occupancy in profiles[profile]]
            if difference_free(grid):
                return grid
        raise ValueError("cross-cell difference-disjoint card could not be filled")

    def component(grid, side):
        indexed = [[[G.idx(g) for g in cell] for cell in row]
                   for row in grid]
        unit = [[[identity]]]
        if side == "A":
            _, rows, _ = lifted_matrices(probe_group, indexed, unit)
        else:
            rows, _, _ = lifted_matrices(probe_group, unit, indexed)
        # Sector one of these unit probes occupies exactly q columns.
        # Sector two contains the distance-bearing transpose component.
        rows = [row >> q for row in rows]
        assert len(rows) == q
        assert all(row >> width == 0 for row in rows)
        return rows

    def eight_cycles(rows, columns):
        """Exact simple Tanner eight-cycles, including cycles with chords."""
        checks = [list(_bits(row)) for row in rows]
        neighbors = [list(_bits(column)) for column in columns]
        rooted = 0
        # The regular component action is transitive within each block.
        for root in (0, q):
            paths = {}
            for c1 in neighbors[root]:
                for v in checks[c1]:
                    if v == root:
                        continue
                    for c2 in neighbors[v]:
                        if c2 == c1:
                            continue
                        for end in checks[c2]:
                            if end == root or end == v:
                                continue
                            previous = paths.setdefault(end, [])
                            for d1, u, d2 in previous:
                                if (u != v and c1 != d1 and c1 != d2
                                        and c2 != d1 and c2 != d2):
                                    rooted += 1
                            previous.append((c1, v, c2))
        assert q * rooted % 4 == 0
        return q * rooted // 4

    def cycle_key(stats):
        four, six, eight = stats
        if trade == 0:
            return (four, six, eight)
        if trade == 1:
            return (four, eight, six)
        if trade == 2:
            return (four, 10 * six + eight, six)
        return (four, 30 * six + eight, eight)

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in row[0]))
                     for row in grid)

    def inspect(grid, side):
        if not difference_free(grid):
            return None
        rows = component(grid, side)
        quality = _component_quality(rows, width)
        if quality is None or quality[0][0] != 0:
            return None
        rank = gf2_rank(rows)
        ranks = (
            gf2_rank([row & mask for row in rows]),
            gf2_rank([row >> q for row in rows]),
        )
        if min((rank,) + ranks) < q - 3:
            return None
        stats, columns = quality
        stats = stats + (eight_cycles(rows, columns),)
        modules = {
            name: {
                "columns": [syndrome(columns, word) for word in masks],
                "masks": masks, "beam": [],
            }
            for name, masks in module_masks[side].items()
        }
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cell_ranks": ranks, "cycles": stats,
            "cycle_key": cycle_key(stats), "signature": signature(grid),
            "d": width + 1, "kernel_words": [],
            "modules": modules, "module_readings": {},
        }

    def syndrome(columns, word):
        result = 0
        for c in _bits(word):
            result ^= columns[c]
        return result

    def module_probe(record, name, diagnostic_seed, trials):
        """Weighted search in a restricted kernel; verify expanded words.

        Fixed coordinates cost three. Even coordinates come in two
        halves; (1,0), (0,1), and (1,1) on one orbit ALL cost two.
        Every returned reading is witnessed, never an exclusion claim.
        """
        module = record["modules"][name]
        columns = module["columns"]
        diagnostic = random.Random(diagnostic_seed)

        def cost(word):
            if name == "fixed":
                return 3 * word.bit_count()
            active = ((word & orbit_low_mask)
                      | (word >> orbit_count))
            return 2 * active.bit_count()

        beam = sorted(set(module["beam"]), key=cost)[:24]
        reading, witness = width + 1, None

        def note(word):
            nonlocal reading, witness
            if word:
                weight = cost(word)
                if weight < reading:
                    reading, witness = weight, word

        for word in beam:
            note(word)
        order = list(range(len(columns)))
        for turn in range(trials):
            diagnostic.shuffle(order)
            if turn % 4 == 3 and beam:
                anchor = diagnostic.choice(beam[:6])
                preferred = list(_bits(anchor))
                diagnostic.shuffle(preferred)
                preferred = set(preferred[:max(1, len(preferred) // 2)])
                order.sort(key=lambda c: c not in preferred)

            pivots = [None] * q
            basis = []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    lead = value.bit_length() - 1
                    old = pivots[lead]
                    if old is None:
                        pivots[lead] = (value, word)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(word)

            light = sorted(set(basis + beam), key=cost)[:14]
            grown = list(light) + beam
            moves = basis + beam
            for a in light:
                for b in moves:
                    word = a ^ b
                    note(word)
                    if word and cost(word) <= reading + 10:
                        grown.append(word)
                if len(grown) > 192:
                    grown = sorted(set(grown), key=cost)[:48]

            shortened = sorted(set(grown), key=cost)[:24]
            for i, a in enumerate(shortened):
                for b in shortened[:i]:
                    word = a ^ b
                    note(word)
                    if word and cost(word) <= reading + 6:
                        grown.append(word)

            diagnostic.shuffle(moves)
            for initial in shortened[:4]:
                word, weight = initial, cost(initial)
                for _ in range(3):
                    changed = False
                    for other in moves:
                        proposed = word ^ other
                        if proposed and cost(proposed) < weight:
                            word, weight = proposed, cost(proposed)
                            changed = True
                    note(word)
                    if not changed:
                        break
                grown.append(word)
            if witness is not None:
                grown.append(witness)
            beam = sorted(set(grown), key=cost)[:24]
            if reading <= 36:
                break

        if witness is None:
            raise ValueError("restricted component yielded no kernel witness")
        assert syndrome(columns, witness) == 0
        beam = sorted(set(beam + [witness]), key=cost)[:24]
        module["beam"] = beam
        actuals = []
        for word in beam[:4]:
            actual = 0
            for c in _bits(word):
                actual ^= module["masks"][c]
            assert actual and actual.bit_count() == cost(word)
            assert syndrome(record["columns"], actual) == 0
            actuals.append(actual)
        best = min(actuals, key=int.bit_count)
        assert all((row & best).bit_count() % 2 == 0
                   for row in record["rows"])
        reading = best.bit_count()
        record["module_readings"][name] = min(
            record["module_readings"].get(name, width + 1), reading)
        record["kernel_words"] = sorted(
            set(record["kernel_words"] + actuals),
            key=int.bit_count)[:24]
        return reading

    def kernel_probe(record, diagnostic_seed, trials):
        """Return an actual component kernel witness, not a certificate."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        reading, witness = width + 1, None
        order = list(range(width))

        def note(word):
            nonlocal reading, witness
            if word:
                weight = word.bit_count()
                if weight < reading:
                    reading, witness = weight, word

        # Module words are genuine unrestricted words too. Retain them
        # between independent calls and cancel against fresh bases.
        for word in record["kernel_words"]:
            note(word)

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
                    note(word)

            light = sorted(
                set(basis + record["kernel_words"]),
                key=int.bit_count)[:20]
            for word in light:
                for other in basis + record["kernel_words"]:
                    note(word ^ other)

            diagnostic.shuffle(basis)
            for word in light[:8]:
                weight = word.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word = candidate
                            weight = word.bit_count()
                            changed = True
                    note(word)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(192):
                    a, b, c = diagnostic.sample(basis, 3)
                    note(a ^ b ^ c)
            if reading <= 24:
                break

        if witness is None:
            raise ValueError("transpose component has no kernel witness")
        assert syndrome(columns, witness) == 0
        assert all((row & witness).bit_count() % 2 == 0
                   for row in record["rows"])
        record["kernel_words"] = sorted(
            set(record["kernel_words"] + [witness]),
            key=int.bit_count)[:24]
        return reading, witness

    def optimize(profile, side):
        bank = set()
        seen = set()
        records = []
        offset = 0 if side == "A" else 100003

        def probe(record, trials, turn):
            diagnostic_seed = seed + offset + 700001 * turn
            fd = module_probe(
                record, "fixed", diagnostic_seed + 1700003,
                max(2, trials // 2))
            ed = module_probe(
                record, "even", diagnostic_seed + 3100009,
                max(2, trials // 4))
            d, word = kernel_probe(record, diagnostic_seed, trials)
            record["d"] = min(record["d"], fd, ed, d)
            bank.update(record["kernel_words"])
            bank.add(word)
            if len(bank) > 64:
                keep = sorted(bank, key=int.bit_count)[:48]
                bank.clear()
                bank.update(keep)

        def merit(record):
            # A witness from another labeling is relevant only after its
            # syndrome is verified on this particular component.
            for word in bank:
                if (word.bit_count() < record["d"]
                        and syndrome(record["columns"], word) == 0):
                    record["d"] = word.bit_count()
            return (record["d"],
                    tuple(-x for x in record["cycle_key"]),
                    record["rank"])

        for trial in range(48 + 4 * min(cohort, 2)):
            record = inspect(fresh(profile), side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            # Cheap challenges precede geometry shortlisting. Cap this
            # ordering statistic so unusually high short readings do not
            # displace good geometry merely through search luck.
            fd = module_probe(record, "fixed", seed + offset + 400009, 2)
            ed = module_probe(record, "even", seed + offset + 900001, 1)
            record["d"] = min(fd, ed)
            records.append(record)
        records.sort(key=lambda record: (
            -min(72, record["d"]), record["cycle_key"]))

        selected = []
        for record in records:
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 12, 0)
            selected.append(record)
            if len(selected) >= 8:
                break
        if not selected:
            return []

        floor4 = min(record["cycles"][0] for record in selected)
        allowance = 0 if trade == 0 else q
        current = max(selected, key=merit)

        # Repair actual labels at the protograph rows supporting the
        # current light witnesses. No group automorphism creates proposals.
        for step in range(24):
            active = [word for word in bank
                      if syndrome(current["columns"], word) == 0]
            blocks = []
            if active:
                word = min(active, key=int.bit_count)
                blocks = [c // q for c in _bits(word)]
            j = (rng.choice(blocks) if blocks and rng.random() < 0.85
                 else rng.randrange(2))
            grid = [[list(row[0])] for row in current["grid"]]
            if step % 6 == 0:
                grid[j][0] = fresh_cell(profiles[profile][j])
            else:
                t = rng.randrange(4)
                old = grid[j][0][t]
                choices = [(a, old[1]) for a in range(97)
                           if (a, old[1]) not in grid[j][0]]
                grid[j][0][t] = rng.choice(choices)
                if not whole_sidon(grid[j][0]):
                    continue
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            if proposed["cycles"][0] > floor4 + allowance:
                continue
            probe(proposed, 8, 1 + step)
            pk, ck = merit(proposed), merit(current)
            if pk > ck or (pk[0] == ck[0] and rng.random() < 0.15):
                current = proposed
                selected.append(proposed)

        selected.sort(key=merit, reverse=True)
        finalists = []
        for record in selected:
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 48, 101)
            finalists.append(record)
            if len(finalists) >= 3:
                break
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            # Independent confirmation only lowers a retained reading.
            probe(record, 96 + 16 * min(cohort, 2), 211)
        finalists.sort(key=merit, reverse=True)
        return finalists

    for restart in range(2):
        aa = optimize(pa, "A")
        bb = optimize(pb, "B")
        pairs = [(a, b) for a in aa for b in bb]
        pairs.sort(key=lambda ab: (
            min(ab[0]["d"], ab[1]["d"]),
            ab[0]["d"] + ab[1]["d"],
            tuple(-(x + y) for x, y in
                  zip(ab[0]["cycle_key"], ab[1]["cycle_key"]))),
            reverse=True)

        for a, b in pairs:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 5 * q == 1455
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k >= q and 7 * n <= 40 * k < 9 * n):
                continue

            # Verify unit-probe extraction against the FINAL matrices.
            # Sector two is ordered (Arow, Brow), with Brow varying first.
            actual_a = [
                sum(((hz[t] >> ((1 + 2 * i) * q)) & mask)
                    << (i * q) for i in range(2))
                for t in range(q)
            ]
            actual_b = [
                (hx[t] >> q) & ((1 << width) - 1)
                for t in range(q)
            ]
            assert actual_a == a["rows"] and actual_b == b["rows"]
            if not _no_kernel4(_syndrome_columns(hx, n)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, n)):
                continue
            assert max(row.bit_count() for row in hx + hz) == 12
            if not css_ok(hx, hz):
                raise ValueError("verified construction failed CSS")
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "card": (pa, pb, trade, triangle),
                "component_readings": (a["d"], b["d"]),
                "C3_module_readings": (
                    dict(a["module_readings"]), dict(b["module_readings"])),
                "component_cycles": (a["cycles"], b["cycles"]),
                "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
                "cross_cell_difference_free": True,
            }
    raise ValueError("no isotypically challenged tall pair passed full gates")


def NEW_CTX(rng, G, idx):
    """Twenty unequal-shape cards, screened using authoritative matrices."""
    q = G.q
    assert q == 111 and A_SHAPE == (2, 1) and B_SHAPE == (5, 2)
    ra, ca = A_SHAPE
    rb, cb = B_SHAPE
    mask = (1 << q) - 1
    n1 = ca * cb * q
    card, cohort = idx % 20, idx // 20
    allocation, model = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0x372152111 + 15485863 * (idx + 1)))

    # The first three B supports have column totals (9,9), and
    # row totals (3,3,4,4,4). Their term distributions are different.
    b_supports = (
        ((1,2), (2,1), (2,2), (2,2), (2,2)),
        ((1,2), (2,1), (1,3), (3,1), (2,2)),
        ((1,2), (1,2), (3,1), (2,2), (2,2)),
        ((1,2), (1,2), (2,1), (2,1), (2,2)),
    )
    plans = (
        ((3,3), 0), ((3,3), 1), ((3,3), 2),
        ((3,4), 3), ((4,4), 3),
    )
    a_counts, b_profile = plans[allocation]
    supports = {
        "A": [[a_counts[0]], [a_counts[1]]],
        "B": [list(row) for row in b_supports[b_profile]],
    }
    assert _grid_weights(supports["A"], supports["B"]) == 12 <= W_MAX
    assert _no_dead_lines(supports["A"], supports["B"])
    assert min(map(sum, supports["B"])) >= 3
    assert max(map(sum, supports["B"])) <= 4

    # Cache native group operations, NEVER a separately implemented lift.
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

    def indexed(grid):
        return [[[G.idx(g) for g in cell] for cell in row]
                for row in grid]

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for row in grid for cell in row)

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    # Exact transpose monomials from narrow scaffold galleries.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(start + 16, q)))
        gallery = [[[g]] for g in ids]
        _, az, _ = lifted_matrices(P, gallery, [[[]]])
        bx, _, _ = lifted_matrices(P, [[[]]], gallery)
        for j in range(len(ids)):
            for side, source in (("A", az), ("B", bx)):
                block = [(row >> ((j + 1) * q)) & mask
                         for row in source]
                assert len(block) == q
                assert all(word.bit_count() == 1 for word in block)
                blocks[side].append(block)

    def component(grid, side):
        nr, nc = len(grid), len(grid[0])
        rows = [0] * (nc * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, word in enumerate(blocks[side][G.idx(g)]):
                        rows[j * q + t] ^= word << (i * q)
        assert all(row >> (nr * q) == 0 for row in rows)
        return rows

    def cell_legal(cell):
        ids = [G.idx(g) for g in cell]
        count = len(ids)
        if len(set(ids)) != count:
            return False
        left = {table[inverse[a]][b]
                for a in ids for b in ids if a != b}
        right = {table[a][inverse[b]]
                 for a in ids for b in ids if a != b}
        if len(left) != count * (count - 1) or len(right) != len(left):
            return False
        if count < 3:
            return True
        relatives = [table[inverse[ids[0]]][g] for g in ids[1:]]
        return any(table[a][b] != table[b][a]
                   for t, a in enumerate(relatives)
                   for b in relatives[:t])

    def fresh_cell(side, i, j):
        """Binding count/coset prescriptions survive every repair."""
        count = supports[side][i][j]
        if count == 1:
            return [G.pick(rng)]
        for _ in range(1024):
            phase = rng.randrange(3)
            if count == 2:
                cross = (side == "B" and
                         (model == 3 or
                          (model == 2 and (i + j + cohort) % 2)))
                if cross:
                    cell = [(rng.randrange(37), phase),
                            (rng.randrange(37), (phase + 1) % 3)]
                else:
                    cell = [(a, phase) for a in rng.sample(range(37), 2)]
            else:
                if count == 3:
                    full = (model == 0 or
                            (model == 2 and
                             (side == "A" or (i + j) % 2 == 0)))
                    occupancy = (1,1,1) if full else (2,1,0)
                else:
                    occupancy = (3,1,0) if model == 3 else (2,1,1)
                fibers = list(range(3))
                rng.shuffle(fibers)
                cell = []
                for b, number in zip(fibers, occupancy):
                    if number == 3:
                        origin, delta = rng.randrange(37), rng.randrange(1,37)
                        t = (3,5,7,11)[(allocation + cohort) % 4]
                        aa = (origin, (origin + delta) % 37,
                              (origin + t * delta) % 37)
                    else:
                        aa = rng.sample(range(37), number)
                    cell.extend((a, b) for a in aa)
            if cell_legal(cell):
                return cell
        raise ValueError("factor-twelve native-Sidon recipe exhausted")

    def fresh(side):
        return [[fresh_cell(side, i, j)
                 for j in range(len(row))]
                for i, row in enumerate(supports[side])]

    def dependencies(columns, order):
        """Exact null basis in the supplied coordinate system."""
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

    def small_null_catalog(columns):
        """Exact enumeration at nullity <=10; genuine words otherwise."""
        basis = dependencies(columns, range(len(columns)))
        if len(basis) <= 10:
            word, words = 0, []
            for t in range(1, 1 << len(basis)):
                word ^= basis[(t & -t).bit_length() - 1]
                words.append(word)
        else:
            short = sorted(basis, key=int.bit_count)[:16]
            words = list(short)
            for i, first in enumerate(short):
                words.extend(first ^ second for second in short[:i])
        words = sorted({w for w in words if w},
                       key=lambda w: (w.bit_count(), w))[:24]
        assert all(syndrome(columns, w) == 0 for w in words)
        return words

    def inspect(grid, side):
        rows = component(grid, side)
        width = len(grid) * q
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        rank = gf2_rank(rows)
        if rank < len(rows) - 8:
            return None
        local_words = []
        for i in range(len(grid)):
            for word in small_null_catalog(columns[i*q:(i+1)*q]):
                local_words.append(word << (i*q))

        # Transpose the AUTHORITATIVE transpose component. This raw
        # kernel catalog diagnoses the other LP sector as well.
        raw_columns = _syndrome_columns(columns, len(rows))
        raw_words = small_null_catalog(raw_columns)
        words = sorted(set(local_words),
                       key=lambda w: (w.bit_count(), w))[:24]
        return dict(
            side=side, grid=copy_grid(grid), rows=rows, columns=columns,
            rank=rank, cycles=cycles, signature=signature(grid),
            words=words, raw_words=raw_words,
            d=min(map(int.bit_count, words), default=width+1))

    def query(rec, trials, offset):
        """Accumulated genuine kernel words; no distance certificate."""
        local = random.Random(seed + offset)
        columns = rec["columns"]
        order = list(range(len(columns)))
        retained = list(rec["words"])
        best = min(retained, key=int.bit_count) if retained else None

        def note(word, grown):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            if best is None or weight < best.bit_count():
                best = word
            if weight <= best.bit_count() + 6:
                grown.append(word)

        for turn in range(trials):
            local.shuffle(order)
            if retained and turn % 4 == 3:
                anchor = local.choice(retained[:6])
                active = list(_bits(anchor))
                local.shuffle(active)
                preferred = set(active[:max(1, len(active)//2)])
                order.sort(key=lambda c: c not in preferred)
            basis = dependencies(columns, order)
            grown = list(retained)
            for word in basis:
                note(word, grown)
            light = sorted(set(basis + retained),
                           key=lambda w: (w.bit_count(), w))[:16]
            moves = basis + retained
            for first in light:
                for second in moves:
                    note(first ^ second, grown)
                if len(grown) > 256:
                    grown = sorted(set(grown), key=int.bit_count)[:32]

            # Bounded four-basis-word cancellation, not exhaustive search.
            if trials >= 32 and turn % 8 == 0:
                chosen = sorted(basis, key=int.bit_count)[:56]
                coordinates = local.sample(
                    range(len(columns)), min(9, len(columns)))
                hash_mask = sum(1 << c for c in coordinates)
                buckets = {}
                for i, first in enumerate(chosen):
                    for second in chosen[:i]:
                        pair = first ^ second
                        bucket = buckets.setdefault(pair & hash_mask, [])
                        for other in bucket:
                            note(pair ^ other, grown)
                        if len(bucket) < 8:
                            bucket.append(pair)

            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for move in moves:
                        candidate = word ^ move
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    if not changed:
                        break
                note(word, grown)
            if best is not None:
                grown.append(best)
            retained = sorted({w for w in grown if w},
                              key=lambda w: (w.bit_count(), w))[:24]
            if best is not None and best.bit_count() <= 12:
                break
        if best is None:
            raise ValueError("new transpose component yielded no kernel word")
        assert syndrome(columns, best) == 0
        assert all((row & best).bit_count() % 2 == 0 for row in rec["rows"])
        rec["words"] = sorted(set(retained + [best]),
                              key=lambda w: (w.bit_count(), w))[:24]
        rec["d"] = min(rec["d"], best.bit_count())

    def optimize(side, restart):
        width = (ra if side == "A" else rb) * q
        offset = 100003 * int(side == "B") + 300007 * restart
        seen, records, bank = set(), [], set()

        def refresh(rec):
            for word in bank:
                if (word.bit_count() < rec["d"] and
                        syndrome(rec["columns"], word) == 0):
                    rec["d"] = word.bit_count()
                    rec["words"].append(word)

        def merit(rec):
            refresh(rec)
            four, six = rec["cycles"]
            # Caps limit selection on optimistic finite readings.
            return (min(32, rec["d"]), -four, -six,
                    min(48, rec["d"]), rec["rank"])

        def diagnose(rec, trials, turn):
            query(rec, trials, offset + 700001 * turn)
            bank.update(rec["words"][:6])
            if len(bank) > 96:
                keep = sorted(bank, key=int.bit_count)[:64]
                bank.clear()
                bank.update(keep)
            refresh(rec)

        for _ in range(48 if side == "A" else 64):
            rec = inspect(fresh(side), side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            records.append(rec)
        records.sort(key=lambda rec: (
            rec["cycles"], -min(48, rec["d"]), -rec["rank"]))
        selected = []
        for rec in records:
            if not _no_kernel4(rec["columns"]):
                continue
            diagnose(rec, 8, 1)
            selected.append(rec)
            if len(selected) == 8:
                break
        if not selected:
            return []
        current = max(selected, key=merit)

        for step in range(16):
            active = list(_bits(current["words"][0]))
            i = rng.choice(active) // q if active else rng.randrange(len(supports[side]))
            j = rng.randrange(len(supports[side][0]))
            grid = copy_grid(current["grid"])
            grid[i][j] = fresh_cell(side, i, j)
            rec = inspect(grid, side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            if rec["cycles"][0] > current["cycles"][0] + 2*q:
                continue
            if not _no_kernel4(rec["columns"]):
                continue
            rec["words"] += [w for w in current["words"]
                             if syndrome(rec["columns"], w) == 0]
            diagnose(rec, 6, 10 + step)
            selected.append(rec)
            if merit(rec) >= merit(current):
                current = rec

        finalists = sorted(selected, key=merit, reverse=True)[:3]
        for rec in finalists:
            diagnose(rec, 32, 101)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for rec in finalists:
            diagnose(rec, 64 + 8*(cohort % 3), 211)
            assert all(syndrome(rec["columns"], w) == 0
                       for w in rec["words"])
        finalists.sort(key=merit, reverse=True)
        return finalists

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                p = word.bit_length() - 1
                old = pivots.get(p)
                if old is None:
                    pivots[p] = word
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

    def logical_reading(a, b, hx, hz, n):
        """Exactly verified logicals in BOTH original LP sectors.

        A miss returns n+1, not an exclusion. Original stabilizer
        reduction preserves commutation and nontriviality.
        """
        candidates = {"X": set(), "Z": set()}
        for word in a["words"][:6]:
            for location in range(rb):
                actual = sum(
                    ((word >> (i*q)) & mask)
                    << (n1 + (i*rb + location)*q)
                    for i in range(ra))
                candidates["X"].add(actual)
        for word in b["words"][:6]:
            for location in range(ra):
                candidates["Z"].add(word << (n1 + location*rb*q))
        for word in a["raw_words"][:6]:
            for location in range(cb):
                actual = sum(
                    ((word >> (j*q)) & mask)
                    << ((j*cb + location)*q) for j in range(ca))
                candidates["Z"].add(actual)
        for word in b["raw_words"][:6]:
            for location in range(ca):
                candidates["X"].add(word << (location*cb*q))

        reading, witnesses = n+1, []
        for kind, opposite, stabilizers in (
                ("X", hz, hx), ("Z", hx, hz)):
            pivots = span(stabilizers)
            genuine = []
            for word in candidates[kind]:
                assert all((word & row).bit_count() % 2 == 0
                           for row in opposite)
                if word and outside(word, pivots):
                    genuine.append(word)
            local = random.Random(seed + (4100011 if kind == "X" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(genuine, key=int.bit_count)[:8]:
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposal = actual ^ row
                        if proposal.bit_count() < actual.bit_count():
                            actual = proposal
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    best = actual
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposal = actual ^ first ^ second
                            if proposal.bit_count() < best.bit_count():
                                best = proposal
                    actual = best
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
                sector = (1 if actual >> n1 == 0 else
                          2 if actual & ((1 << n1)-1) == 0 else 3)
                witnesses.append((kind, actual.bit_count(), sector))
        return reading, witnesses

    for restart in range(3):
        aa, bb = optimize("A", restart), optimize("B", restart)
        survivors = []
        for a in aa:
            for b in bb:
                hx, hz, n = lifted_matrices(
                    P, indexed(a["grid"]), indexed(b["grid"]))
                assert n == 12*q == 1332
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k >= 333 and 9*n <= 40*k < 11*n):
                    continue

                # Independent agreement with the COMPLETE construction.
                actual_a = [
                    sum(((hz[(j*rb)*q+t]
                          >> (n1+i*rb*q)) & mask) << (i*q)
                        for i in range(ra))
                    for j in range(ca) for t in range(q)]
                actual_b = [
                    (hx[beta*q+t] >> n1) & ((1 << (rb*q))-1)
                    for beta in range(cb) for t in range(q)]
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logical_d, witnesses = logical_reading(a, b, hx, hz, n)
                detected = min(32, a["d"], b["d"], logical_d)
                key = (k*detected*detected, detected,
                       -a["cycles"][0]-b["cycles"][0],
                       -a["cycles"][1]-b["cycles"][1],
                       a["d"]+b["d"])
                survivors.append((key, a, b, hx, hz, k,
                                  logical_d, witnesses))
        survivors.sort(key=lambda item: item[0], reverse=True)

        for key, a, b, cached_hx, cached_hz, k, logical_d, witnesses in survivors:
            if not _no_kernel4(_syndrome_columns(cached_hx, 1332)):
                continue
            if not _no_kernel4(_syndrome_columns(cached_hz, 1332)):
                continue
            A, B = a["grid"], b["grid"]
            assert [[len(c) for c in row] for row in A] == supports["A"]
            assert [[len(c) for c in row] for row in B] == supports["B"]
            assert all(cell_legal(cell) for grid in (A, B)
                       for row in grid for cell in row)
            hx, hz, n = lifted_matrices(G, A, B)
            assert hx == cached_hx and hz == cached_hz and n == 1332
            assert k == n-gf2_rank(hx)-gf2_rank(hz)
            assert max(row.bit_count() for row in hx+hz) <= 12
            assert css_ok(hx, hz)
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "supports": supports,
                "card": (allocation, model, cohort),
                "component_ranks": (a["rank"], b["rank"]),
                "component_readings": (a["d"], b["d"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "raw_kernel_catalog_sizes": (
                    len(a["raw_words"]), len(b["raw_words"])),
                "verified_logical_upper_reading": logical_d,
                "verified_logical_witness_weights": witnesses,
                "pairing_merit": key,
            }
    raise ValueError("balanced factor-twelve cards missed full/rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return every complete new-shape cell without post-screen draws."""
    cell = list(ctx[side][i][j])
    assert len(cell) == ctx["supports"][side][i][j]
    assert len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Whole-cell Sidon supports plus transpose-kernel repair uniformly exclude sub-50 logicals: current slots 21,43,54,58 instead screened at 30,39,45,36 despite four terms per cell and w=12.
# REFUTED: Verified C3-fixed screening before unrestricted repair improves the 965.8564 tall-bed record: shown Program 1 scored 932.7676, with selected readings 65..71.
# REFUTED: Adding bidirectional difference-disjoint supports and both weighted C3 modules improves the 965.8564 tall-bed record: this current program instead scored 869.0234 with five selected readings 61,63,69,66,68.
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
