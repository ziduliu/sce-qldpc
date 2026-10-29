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
CONCEPT = ("Hypothesis: on frozen Z61:Z3(s=13), order 183, A=(1,2), B=(2,3), selecting zero-four-cycle A components by both fixed and trace-zero C3 kernel challenges raises five-code fitness above 270.7344 in cell 0.150. Measurement can refute this hypothesis; all finite distance readings remain provisional upper bounds.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged. This exact order-183 bed leads cell 0.150 at fitness 270.7344 with measured readings 44,47,47,47,47. The faithful action satisfies 13^3=1 modulo 61. In characteristic two its order-three permutation action splits into fixed and trace-zero spaces; restrictions below use only XORs of authoritative columns and physically verified expansions, not another lift.")
SHAPES = ("Keep A=(1,2) and B=(2,3) byte-for-byte unchanged. q=183 gives n=1464, k>=183 and guaranteed rate 1/8; the size-law maximum is q=187. Unequal components have 366 variables/183 checks and 549 variables/366 checks, nominal rates 1/2 and 1/3. Current measured minimum Z witnesses cross both Acols within individual Bcol1/2 strips; accompanying X readings are 48..50, all sector one. Preserve these lengths and target A rather than additional B density. Authoritative ranks retain 0.125<=k/n<0.175, including exact-bound codes.")
FILLING = ("Twenty cards cross five mixed A occupancy pairs with four binding cubic-B relative-order prescriptions. All A counts are (4,5); positive B columns total three and row totals are at most six, giving HX=12 and HZ<=11. Dense degree-four B cards are removed: none advanced in the current measurement. Require bidirectional Sidon and disjoint A right differences, then independently verify zero A four-cycles. Challenge every retained A proposal before geometric truncation using fixed and trace-zero C3 spaces. Trace-zero coordinates are overlapping two-point orbit masks, so circuit selection uses actual expanded weight, not coefficient weight. Add bounded hashed four-circuit cancellation, preserve all discovered witnesses, repair actual cells and independently rechallenge finalists. Original stabilizer spans, native-component agreement, full ranks, kernel-four, weights and CSS remain gates; no finite challenge certifies quantum distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r015_m61r3s13_12x23_zero4_C3_fixed_tracezero_physical_v2"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 183

def custom_ops(q):
    # Same split-metacyclic construction rule, resized for factor eight.
    # 13^2=47 and 13^3=1 mod 61; the action has order three.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 183")
    m, r, s = 61, 3, 13
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

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 3)               # (rb, cb)


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


def _legacy_tall_C3_ctx(rng, G, idx):
    """Historical order-291 tall experiment; never called by NEW_CTX."""
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

    def affine_class(t):
        """Canonical three-point class under affine maps and point swaps."""
        it = pow(t, 95, 97)
        im = pow((t - 1) % 97, 95, 97)
        return tuple(sorted({
            t, (1 - t) % 97, it, (-im) % 97,
            (t * im) % 97, ((t - 1) * it) % 97,
        }))

    old_classes = {
        affine_class(t) for t in (3, 5, 7, 9, 11, 13, 15, 17)
    }
    new_classes = sorted({
        affine_class(t) for t in range(2, 97)
        if len(affine_class(t)) == 6
        and affine_class(t) not in old_classes
    })
    assert new_classes
    triangle = new_classes[
        (trade + 4 * cohort) % len(new_classes)][0]

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

    # Variable symmetries of the verified transpose components:
    # A commutes with right translation; B with left translation.
    # These permutations define a restricted word space, not a new lift.
    subgroup_generator = G.idx((0, 1))
    orbit_masks = {}
    for side in ("A", "B"):
        unused = set(range(q))
        base_orbits = []
        while unused:
            root = min(unused)
            orbit = []
            value = root
            for _ in range(3):
                orbit.append(value)
                value = (table[value][subgroup_generator]
                         if side == "A"
                         else table[subgroup_generator][value])
            assert value == root and len(set(orbit)) == 3
            assert set(orbit) <= unused
            unused.difference_update(orbit)
            base_orbits.append(sum(1 << c for c in orbit))
        assert len(base_orbits) == q // 3
        orbit_masks[side] = [
            word << (j * q)
            for j in range(2) for word in base_orbits
        ]

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

    def fresh(profile):
        return [[fresh_cell(occupancy)]
                for occupancy in profiles[profile]]

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
        rows = component(grid, side)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        rank = gf2_rank(rows)
        ranks = (
            gf2_rank([row & mask for row in rows]),
            gf2_rank([row >> q for row in rows]),
        )
        if min((rank,) + ranks) < q - 3:
            return None
        stats, columns = quality
        # Restrict the AUTHORITATIVE columns to orbit-constant words.
        # Keep all original syndrome coordinates; no quotient lift or
        # assumed check compression is used in this screening matrix.
        fixed_columns = []
        for orbit in orbit_masks[side]:
            value = 0
            for c in _bits(orbit):
                value ^= columns[c]
            fixed_columns.append(value)
        if (0 in fixed_columns
                or len(set(fixed_columns)) != len(fixed_columns)):
            return None
        stats = stats + (eight_cycles(rows, columns),)
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cell_ranks": ranks, "cycles": stats,
            "cycle_key": cycle_key(stats), "signature": signature(grid),
            "fixed_columns": fixed_columns,
            "fixed_orbits": orbit_masks[side],
            "fixed_words": [], "kernel_words": [],
            "fixed_d": width + 1, "d": width + 1,
        }

    def syndrome(columns, word):
        result = 0
        for c in _bits(word):
            result ^= columns[c]
        return result

    def fixed_probe(record, diagnostic_seed, trials):
        """Search the restricted kernel and verify every expanded witness.

        The smaller information sets directly target weight-3-multiple
        families that unrestricted searches can miss. This is still a
        sampled upper reading, not an exhaustive minimum computation.
        """
        diagnostic = random.Random(diagnostic_seed)
        columns = record["fixed_columns"]
        fixed_width = len(columns)
        order = list(range(fixed_width))
        retained = list(record["fixed_words"])
        reading = (min(word.bit_count() for word in retained)
                   if retained else fixed_width + 1)
        witness = (min(retained, key=int.bit_count) if retained else None)
        candidates = []

        def note(word):
            nonlocal reading, witness
            if not word:
                return
            weight = word.bit_count()
            if weight < reading:
                reading, witness = weight, word
            if weight <= reading + 4:
                candidates.append(word)

        for turn in range(trials):
            diagnostic.shuffle(order)
            # Some orders use half of a surviving support as an
            # information-set hint; other orders remain independent.
            if turn % 3 == 1 and retained:
                hint = list(_bits(retained[0]))
                diagnostic.shuffle(hint)
                preferred = set(hint[:max(1, len(hint) // 2)])
                order = ([c for c in order if c in preferred]
                         + [c for c in order if c not in preferred])

            pivots, basis = {}, []
            candidates = list(retained)
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
            if not basis:
                raise ValueError("C3-fixed restriction unexpectedly injective")

            # Exhaustive pairs of the approximately 98 fundamental
            # words are inexpensive at this restricted length.
            for i, word in enumerate(basis):
                for other in basis[:i]:
                    note(word ^ other)

            light = sorted(set(retained + candidates + basis),
                           key=int.bit_count)[:24]
            for i, word in enumerate(light):
                for other in light[:i]:
                    note(word ^ other)

            diagnostic.shuffle(basis)
            for initial in light[:6]:
                word, weight = initial, initial.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in basis:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < weight:
                            word, weight = proposed, proposed.bit_count()
                            changed = True
                    note(word)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    note(a ^ b ^ c)

            retained = sorted(set(retained + candidates),
                              key=int.bit_count)[:24]
            if reading <= 8:
                break

        if witness is None:
            raise ValueError("no C3-fixed kernel witness")
        assert syndrome(columns, witness) == 0
        retained = sorted(set(retained + [witness]),
                          key=int.bit_count)[:24]
        record["fixed_words"] = retained

        actuals = []
        for word in retained[:8]:
            actual = 0
            for c in _bits(word):
                actual |= record["fixed_orbits"][c]
            assert actual.bit_count() == 3 * word.bit_count()
            assert syndrome(record["columns"], actual) == 0
            assert all((row & actual).bit_count() % 2 == 0
                       for row in record["rows"])
            actuals.append(actual)

        best_actual = min(actuals, key=int.bit_count)
        reading = best_actual.bit_count()
        record["fixed_d"] = min(record["fixed_d"], reading)
        record["kernel_words"] = sorted(
            set(record["kernel_words"] + actuals),
            key=int.bit_count)[:24]
        return reading, best_actual, actuals

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

        # Expanded fixed-space words are actual unrestricted kernel
        # words too. Carry them between independent searches and allow
        # cancellation against fresh unrestricted fundamental words.
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
                for other in basis:
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
        return reading, witness

    def optimize(profile, side):
        bank = set()
        seen = set()
        records = []
        offset = 0 if side == "A" else 100003

        def probe(record, trials, turn):
            diagnostic_seed = seed + offset + 700001 * turn
            fd, fixed_word, fixed_words = fixed_probe(
                record, diagnostic_seed + 3000017, max(8, trials))
            d, word = kernel_probe(
                record, diagnostic_seed, max(4, trials // 3))
            record["d"] = min(record["d"], fd, d)
            record["kernel_words"] = sorted(
                set(record["kernel_words"] + [word, fixed_word]),
                key=int.bit_count)[:24]
            bank.update(fixed_words)
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
            return (record["d"], record["fixed_d"],
                    tuple(-x for x in record["cycle_key"]),
                    record["rank"])

        for trial in range(64 + 4 * min(cohort, 2)):
            record = inspect(fresh(profile), side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            # Diagnose every proposal before geometry discards it.
            # Common seeds reduce candidate-specific search luck.
            fixed_probe(record, seed + offset + 400009, 4)
            records.append(record)
        records.sort(key=lambda record: (
            -record["fixed_d"], record["cycle_key"]))

        selected = []
        for record in records:
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 12, 0)
            selected.append(record)
            if len(selected) >= 10:
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
                fiber = [u for u, g in enumerate(grid[j][0])
                         if g[1] == old[1]]
                if len(fiber) == 3:
                    # Preserve this card's NEW affine class rather than
                    # letting a one-exponent mutation erase its design.
                    origin = rng.randrange(97)
                    scale = rng.randrange(1, 97)
                    rotations = [
                        origin, (origin + scale) % 97,
                        (origin + triangle * scale) % 97,
                    ]
                    for u, a in zip(fiber, rotations):
                        grid[j][0][u] = (a, old[1])
                else:
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
                "component_cycles": (a["cycles"], b["cycles"]),
                "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
            }
    raise ValueError("no repaired tall-protograph pair passed full gates")


def NEW_CTX(rng, G, idx):
    """Twenty cubic cards with fixed/trace-zero physical-kernel selection."""
    from math import exp

    q, m = G.q, G.q // 3
    assert q == 183 and m == 61
    assert A_SHAPE == (1, 2) and B_SHAPE == (2, 3)
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    topology, policy = card // 4, card % 4
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0x61C31223 + 1000003 * (idx + 1)))

    # Five A occupancy pairs crossed with four actual B order policies.
    # All cards retain the allocation used by every current measured winner.
    sa = [[4, 5]]
    sb = ([[1, 1, 1], [2, 2, 2]] if policy < 2 else
          [[1, 1, 2], [2, 2, 1]])
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    bins = [(i, j) for i in range(2) for j in range(3)
            if sb[i][j] == 2]
    # Policies prescribe zero, one, two or three order-61 ratios.
    # In the balanced two-normal card the normal bins occupy both rows.
    # Changing these ratios changes relations, not quotient names.
    normal_bins = set(bins[:policy])
    mixed_bins = set(bins) - normal_bins
    assert len(normal_bins) == policy and len(bins) == 3
    a_occupancies = (
        {4: (2, 1, 1), 5: (2, 2, 1)},
        {4: (2, 2, 0), 5: (2, 2, 1)},
        {4: (2, 1, 1), 5: (3, 2, 0)},
        {4: (3, 1, 0), 5: (3, 1, 1)},
        {4: (3, 1, 0), 5: (3, 2, 0)},
    )

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

    class Probe:
        # Cache native operations only; never implement a new lift.
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

    probe_group = Probe()
    mono = {"A": [], "B": []}
    for start in range(0, q, 16):
        ids = list(range(start, min(start + 16, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(probe_group, gallery, [[[]]])
        _, zz, _ = lifted_matrices(probe_group, [[[]]], gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            mono["A"].append([(row >> (j*q)) & mask for row in xx])
            mono["B"].append([(row >> (j*q)) & mask for row in zz])

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def signature(grid):
        return tuple(tuple(sorted(cell)) for row in grid for cell in row)

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def sidon(cell):
        if len(set(cell)) != len(cell):
            return False
        left, right = set(), set()
        for a in cell:
            for b in cell:
                if a != b:
                    left.add(table[inverses[a]][b])
                    right.add(table[a][inverses[b]])
        wanted = len(cell) * (len(cell) - 1)
        return len(left) == len(right) == wanted

    def binding(cell, side, i, j):
        if not sidon(cell):
            return False
        if side == "B" and len(cell) == 2:
            mixed = G.elements[cell[0]][1] != G.elements[cell[1]][1]
            return mixed == ((i, j) in mixed_bins)
        if side == "A":
            sizes = sorted(sum(G.elements[g][1] == b for g in cell)
                           for b in range(3))
            return sizes == sorted(a_occupancies[topology][len(cell)])
        return True

    def fresh_cell(count, side, i, j):
        if count == 1:
            return [rng.randrange(q)]
        occupancy = (a_occupancies[topology][count] if side == "A"
                     else ((1, 1, 0) if (i, j) in mixed_bins
                           else (2, 0, 0)))
        for _ in range(2048):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, size in zip(fibers, occupancy):
                cell.extend(G.idx((a, b))
                            for a in rng.sample(range(m), size))
            if binding(cell, side, i, j):
                return cell
        raise ValueError("binding Sidon cell construction exhausted")

    def right_differences(cell):
        return {table[a][inverses[b]]
                for a in cell for b in cell if a != b}

    def fresh(support, side):
        if side == "A":
            for attempt in range(2048):
                if attempt % 24 == 0:
                    first = fresh_cell(4, "A", 0, 0)
                    differences = right_differences(first)
                second = fresh_cell(5, "A", 0, 1)
                if differences.isdisjoint(right_differences(second)):
                    return [[first, second]]
            raise ValueError("zero-four-cycle mixed A construction exhausted")
        return [[fresh_cell(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(support)]

    # Coordinate restrictions only. Restricted columns are always XORs
    # of authoritative columns, and expanded words are syndrome-verified.
    fixed_masks, normal_masks, even_masks, even_half = {}, {}, {}, {}
    h = G.idx((0, 1))
    for side, nc in (("A", 2), ("B", 3)):
        unused, orbits = set(range(q)), []
        while unused:
            root = min(unused)
            orbit, value = [], root
            for _ in range(3):
                orbit.append(value)
                value = table[value][h] if side == "A" else table[h][value]
            assert value == root and len(set(orbit)) == 3
            assert set(orbit) <= unused
            unused.difference_update(orbit)
            orbits.append(sum(1 << c for c in orbit))
        fixed_masks[side] = [
            word << (j*q) for j in range(nc) for word in orbits]
        normal_masks[side] = [
            sum(1 << (j*q + G.idx((a, b))) for a in range(m))
            for j in range(nc) for b in range(3)]

        # For each three-point orbit, e01 and e02 span its trace-zero
        # word space. These masks overlap: coefficient Hamming weight
        # is NOT physical weight. Store all e01 coordinates before e02.
        pairs01, pairs02 = [], []
        for j in range(nc):
            for orbit in orbits:
                a, b, c = list(_bits(orbit))
                pairs01.append(((1 << a) | (1 << b)) << (j*q))
                pairs02.append(((1 << a) | (1 << c)) << (j*q))
        even_half[side] = len(pairs01)
        even_masks[side] = pairs01 + pairs02
        assert even_half[side] == nc*m

    def dependency_basis(columns, order, with_pivots=False):
        """Genuine kernel words in the supplied column coordinates."""
        pivots, basis, independent = {}, [], []
        for c in order:
            value, word = columns[c], 1 << c
            while value:
                lead = value.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = (value, word)
                    independent.append(c)
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return (basis, independent) if with_pivots else basis

    def expand(word, groups):
        actual = 0
        for c in _bits(word):
            actual ^= groups[c]
        return actual

    def exact_restricted_words(columns, groups):
        compressed = [syndrome(columns, group) for group in groups]
        basis = dependency_basis(compressed, range(len(compressed)))
        found, word = [], 0
        # At most nine normal-orbit coordinates, so enumeration is exact.
        assert len(basis) <= 9
        for t in range(1, 1 << len(basis)):
            word ^= basis[(t & -t).bit_length() - 1]
            actual = expand(word, groups)
            assert actual and syndrome(columns, actual) == 0
            found.append(actual)
        return sorted(set(found), key=lambda w: (w.bit_count(), w))[:12]

    def localized_words(columns, nc):
        found = []
        for j in range(nc):
            basis = dependency_basis(columns, range(j*q, (j+1)*q))
            if len(basis) <= 12:
                word, shortest = 0, None
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length() - 1]
                    if shortest is None or word.bit_count() < shortest.bit_count():
                        shortest = word
                if shortest is not None:
                    found.append(shortest)
            else:
                # Larger restrictions supply witnesses, not exact minima.
                found.extend(sorted(basis, key=int.bit_count)[:4])
        assert all(syndrome(columns, word) == 0 for word in found)
        return found

    def search(columns, prior, diagnostic_seed, trials, trace_half=None):
        """Verified kernel witnesses, ranked at actual expanded weight.

        For trace-zero coordinates (u,v), the physical orbit bits are
        (u XOR v,u,v). The formula below scores their exact weight
        without repeatedly expanding long masks during cancellation.
        """
        local = random.Random(diagnostic_seed)
        width = len(columns)
        order = list(range(width))
        coefficient_mask = ((1 << trace_half)-1
                            if trace_half is not None else None)

        def weight(word):
            if trace_half is None:
                return word.bit_count()
            u = word & coefficient_mask
            v = word >> trace_half
            return u.bit_count() + v.bit_count() + (u ^ v).bit_count()

        def key(word):
            return weight(word), word

        retained = sorted({w for w in prior if w}, key=key)[:20]
        best = None
        grown = []

        def observe(word):
            nonlocal best
            if not word:
                return
            w = weight(word)
            if best is None or w < weight(best):
                best = word
            if w <= weight(best) + 4:
                grown.append(word)

        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word)
        for turn in range(trials):
            local.shuffle(order)
            if turn % 4 == 3 and retained:
                hint = list(_bits(local.choice(retained[:6])))
                local.shuffle(hint)
                preferred = set(hint[:max(1, len(hint)//2)])
                order.sort(key=lambda c: c not in preferred)
            basis, independent = dependency_basis(
                columns, order, with_pivots=True)
            if not basis:
                continue
            grown = list(retained)
            for word in basis:
                observe(word)
            light = sorted(set(basis + retained), key=key)[:16]
            moves = basis + retained
            for word in light:
                for other in moves:
                    observe(word ^ other)

            # Bounded four-circuit cancellation in systematic coordinates.
            # Hash/list omissions are search misses, not exclusions.
            if independent and (turn % 4 == 0 or turn == trials-1):
                sources = sorted(set(basis + retained[:8]), key=key)[:80]
                sampled = local.sample(independent, min(8, len(independent)))
                hash_mask = sum(1 << c for c in sampled)
                buckets = {}
                for i, first in enumerate(sources):
                    for second in sources[:i]:
                        pair = first ^ second
                        if not pair:
                            continue
                        bucket = buckets.setdefault(pair & hash_mask, [])
                        for other in bucket:
                            observe(pair ^ other)
                        bucket.append(pair)
                        if len(bucket) > 32:
                            bucket[:] = sorted(set(bucket), key=key)[:24]
                    if len(grown) > 256:
                        grown = sorted(set(grown), key=key)[:32]

            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        proposed = word ^ other
                        if proposed and weight(proposed) < weight(word):
                            word, changed = proposed, True
                    observe(word)
                    if not changed:
                        break
                grown.append(word)
            if len(basis) >= 3:
                for _ in range(96):
                    a, b, c = local.sample(basis, 3)
                    observe(a ^ b ^ c)
            grown.append(best)
            retained = sorted({w for w in grown if w}, key=key)[:20]
            if weight(best) <= 8:
                break
        if best is not None:
            assert syndrome(columns, best) == 0
            retained = sorted(set(retained + [best]), key=key)[:20]
        return best, retained

    def inspect(grid, side):
        nr, nc = len(grid), len(grid[0])
        if not all(binding(cell, side, i, j)
                   for i, row in enumerate(grid)
                   for j, cell in enumerate(row)):
            return None
        rows = [0] * (nr*q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    for r, value in enumerate(mono[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        quality = _component_quality(rows, nc*q)
        if quality is None:
            return None
        cycles, columns = quality
        four, excess = cycles
        if side == "A" and four:
            return None
        trade = (policy + topology + cohort) % 4
        cost = (500 + 100*trade)*four + (1 + trade)*excess
        return dict(grid=copy_grid(grid), rows=rows, columns=columns,
                    side=side, nc=nc, cycles=cycles, cost=cost,
                    signature=signature(grid), words=[], d=nc*q+1)

    def prepare(record):
        if "rank" in record:
            return True
        rank = gf2_rank(record["rows"])
        if rank < len(record["rows"]) - 12:
            return False
        if not _no_kernel4(record["columns"]):
            return False
        record["rank"] = rank
        words = exact_restricted_words(
            record["columns"], normal_masks[record["side"]])
        words += localized_words(record["columns"], record["nc"])
        record["words"] = sorted(set(words),
                                 key=lambda w: (w.bit_count(), w))[:20]
        if words:
            record["d"] = min(map(int.bit_count, words))
        groups = fixed_masks[record["side"]]
        record["fixed_columns"] = [
            syndrome(record["columns"], group) for group in groups]
        record["fixed_words"] = []
        if record["side"] == "A":
            record["even_columns"] = [
                syndrome(record["columns"], group)
                for group in even_masks["A"]]
            record["even_words"] = []
            record["even_d"] = record["nc"]*q+1
        return True

    def component_pool(support, side):
        offset = 100003 * int(side == "B")
        geometry, seen, witness_bank = {}, set(), set()

        def probe(record, turn, full_trials, fixed_trials, even_trials=0):
            prior = record["words"] + [
                word for word in witness_bank
                if syndrome(record["columns"], word) == 0]
            fw, fixed_beam = search(
                record["fixed_columns"], record["fixed_words"],
                seed + offset + 1700003 + 700001*turn, fixed_trials)
            record["fixed_words"] = fixed_beam
            if fw is not None:
                actuals = [expand(word, fixed_masks[side])
                           for word in fixed_beam[:6]]
                assert all(syndrome(record["columns"], word) == 0
                           for word in actuals)
                prior += actuals

            if side == "A" and even_trials:
                ew, even_beam = search(
                    record["even_columns"], record["even_words"],
                    seed + 3700039 + 900001*turn, even_trials,
                    trace_half=even_half["A"])
                record["even_words"] = even_beam
                if ew is not None:
                    half = even_half["A"]
                    cmask = (1 << half)-1
                    actuals = []
                    for word in even_beam[:8]:
                        actual = expand(word, even_masks["A"])
                        u, v = word & cmask, word >> half
                        physical = (u.bit_count() + v.bit_count()
                                    + (u ^ v).bit_count())
                        assert actual.bit_count() == physical
                        assert syndrome(record["columns"], actual) == 0
                        assert all((actual & row).bit_count() % 2 == 0
                                   for row in record["rows"])
                        actuals.append(actual)
                    record["even_d"] = min(
                        record["even_d"],
                        min(word.bit_count() for word in actuals))
                    prior += actuals

            word, beam = search(
                record["columns"], prior,
                seed + offset + 900001 + 700001*turn, full_trials)
            if word is None:
                raise ValueError("positive-rate component has no kernel witness")
            record["d"] = min(record["d"], word.bit_count())
            record["words"] = beam
            witness_bank.update(beam[:4])
            if len(witness_bank) > 96:
                keep = sorted(witness_bank,
                              key=lambda w: (w.bit_count(), w))[:64]
                witness_bank.clear()
                witness_bank.update(keep)

        def merit(record):
            transferred = []
            for word in witness_bank:
                if (word.bit_count() <= record["d"]
                        and syndrome(record["columns"], word) == 0):
                    record["d"] = min(record["d"], word.bit_count())
                    transferred.append(word)
            if transferred:
                record["words"] = sorted(
                    set(record["words"] + transferred),
                    key=lambda w: (w.bit_count(), w))[:20]
            return (min(54, record["d"]), record["d"],
                    -record["cost"], record["rank"])

        def remember(record):
            if record is None or record["signature"] in seen:
                return
            seen.add(record["signature"])
            if side == "A":
                if not prepare(record):
                    return
                # Both representation sectors are challenged BEFORE
                # geometry can discard this actual A filling.
                probe(record, 301, 0, 4, 2)
            geometry[record["signature"]] = record
            limit = 24 if side == "A" else 14
            if len(geometry) > limit:
                if side == "A":
                    worst = min(geometry, key=lambda k: merit(geometry[k]))
                else:
                    worst = max(geometry, key=lambda k: geometry[k]["cost"])
                del geometry[worst]

        for _ in range((112 if side == "A" else 56)
                       + 8*min(cohort, 2)):
            remember(inspect(fresh(support, side), side))
        if not geometry:
            return []
        starts = (sorted(geometry.values(), key=merit, reverse=True)[:2]
                  if side == "A" else
                  sorted(geometry.values(), key=lambda r: r["cost"])[:2])
        positions = [(i, j) for i, row in enumerate(support)
                     for j, count in enumerate(row) if count]

        # Actual relative-label proposals, never submitted symmetry images.
        steps = 32 if side == "A" else 64
        for initial in starts:
            current = initial
            for step in range(steps):
                grid = copy_grid(current["grid"])
                i, j = rng.choice(positions)
                if side == "A" and step % 11 == 10:
                    grid = fresh(support, side)
                elif step % 4 == 0 or len(grid[i][j]) == 1:
                    grid[i][j] = fresh_cell(support[i][j], side, i, j)
                else:
                    cell = list(grid[i][j])
                    t = rng.randrange(len(cell))
                    a, b = G.elements[cell[t]]
                    cell[t] = G.idx(((a + rng.randrange(1, m)) % m, b))
                    if not binding(cell, side, i, j):
                        continue
                    grid[i][j] = cell
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                if side == "A":
                    remember(proposed)
                    if "rank" not in proposed:
                        continue
                    pk, ck = merit(proposed), merit(current)
                    if pk > ck or (pk[0] == ck[0] and rng.random() < 0.12):
                        current = proposed
                else:
                    delta = proposed["cost"] - current["cost"]
                    temperature = q*(10*(1-step/float(steps))**2 + 0.2)
                    if delta <= 0 or rng.random() < exp(
                            min(0.0, -delta/temperature)):
                        current = proposed
                        remember(current)

        records = list(geometry.values())
        records.sort(key=(lambda r: merit(r)), reverse=True) if side == "A" \
            else records.sort(key=lambda r: r["cost"])
        selected = []
        for record in records:
            if not prepare(record):
                continue
            probe(record, 0, 12 if side == "A" else 8,
                  12 if side == "A" else 8, 8 if side == "A" else 0)
            selected.append(record)
            if len(selected) == (10 if side == "A" else 6):
                break
        if not selected:
            return []
        current = max(selected, key=merit)

        # Keep occupancy and B ratio prescriptions during witness repair.
        for step in range(20 if side == "A" else 12):
            word = rng.choice(current["words"][:6])
            occupied = list(_bits(word))
            j = (rng.choice(occupied)//q if occupied else
                 rng.randrange(len(support[0])))
            grid = copy_grid(current["grid"])
            if side == "A" and step % 7 == 6:
                grid = fresh(support, side)
            else:
                targeted = [(i, j) for i in range(len(support))]
                changed = targeted if step % 4 == 3 else [rng.choice(targeted)]
                for i, j in changed:
                    grid[i][j] = fresh_cell(support[i][j], side, i, j)
            proposed = inspect(grid, side)
            if proposed is None or proposed["signature"] in seen:
                continue
            seen.add(proposed["signature"])
            if proposed["cycles"][0] > current["cycles"][0] + q:
                continue
            if not prepare(proposed):
                continue
            for known in current["words"]:
                if syndrome(proposed["columns"], known) == 0:
                    proposed["words"].append(known)
                    proposed["d"] = min(proposed["d"], known.bit_count())
            probe(proposed, step + 1, 6, 8, 6 if side == "A" else 0)
            selected.append(proposed)
            if merit(proposed) > merit(current):
                current = proposed

        unique = {record["signature"]: record for record in selected}
        finalists = sorted(unique.values(), key=merit, reverse=True)[:4]
        for record in finalists:
            probe(record, 101, 32, 40, 24 if side == "A" else 0)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            # Independent searches can only lower this matrix's reading.
            probe(record, 211, 64 + 8*min(cohort, 2),
                  64, 40 if side == "A" else 0)
            if side == "A":
                probe(record, 419, 24, 24, 20)
        finalists.sort(key=merit, reverse=True)
        return finalists

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

    def localized_logicals(a, b, hx, hz, n):
        """Exact logical validity; bounded stabilizer reduction upper readings."""
        readings = []
        for side, record, opposite, stabilizers in (
                ("A", a, hx, hz), ("B", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(set(record["words"]), key=int.bit_count)[:4]:
                assert syndrome(record["columns"], word) == 0
                for location in range(3 if side == "A" else 2):
                    if side == "A":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((3*j + location)*q) for j in range(2))
                    else:
                        actual = word << (3*location*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n + 1
            local = random.Random(seed + (4100009 if side == "A" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual = proposed
                    near = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    best = actual
                    for i, first in enumerate(near):
                        for second in near[:i]:
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
            # n+1 means no nontrivial embedding was found, not exclusion.
            readings.append(reading)
        return tuple(readings)

    def native(grid):
        return [[[G.elements[g] for g in cell] for cell in row]
                for row in grid]

    for restart in range(2):
        aa = component_pool(sa, "A")
        bb = component_pool(sb, "B")
        survivors = []
        for a in aa:
            for b in bb:
                A, B = native(a["grid"]), native(b["grid"])
                hx, hz, n = lifted_matrices(G, A, B)
                assert n == 8*q == 1464
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k >= q and 5*n <= 40*k < 7*n):
                    continue
                ar, br = _verified_components(hx, hz, q)
                assert ar == a["rows"] and br == b["rows"]
                assert max(row.bit_count() for row in hx + hz) <= min(12, W_MAX)
                logicals = localized_logicals(a, b, hx, hz, n)
                detected = min(52, a["d"], b["d"], *logicals)
                key = (k*detected*detected, detected,
                       -a["cost"]-b["cost"], a["d"]+b["d"], -(k-q))
                survivors.append((key, A, B, a, b, hx, hz, n, k, logicals))
        survivors.sort(key=lambda item: item[0], reverse=True)
        for key, A, B, a, b, hx, hz, n, k, logicals in survivors:
            if not _no_kernel4(_syndrome_columns(hx, n)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, n)):
                continue
            assert css_ok(hx, hz)
            assert _grid_weights(sa, sb) <= min(12, W_MAX)
            assert _no_dead_lines(sa, sb)
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "card": (topology, policy, cohort),
                "support_pair": (sa, sb),
                "mixed_B_binomials": tuple(sorted(mixed_bins)),
                "component_readings": (a["d"], b["d"]),
                "component_ranks": (a["rank"], b["rank"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "A_tracezero_physical_upper_reading": a["even_d"],
                "A_tracezero_coordinate_count": len(a["even_columns"]),
                "verified_localized_logical_readings": logicals,
                "pairing_merit": key,
            }
    raise ValueError("fixed/trace-zero cubic pairings missed cell 0.150 gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete physically challenged cell without extra draws."""
    cell = list(ctx[side][i][j])
    assert len(cell) == ctx["support_pair"][int(side == "B")][i][j]
    assert 1 <= len(cell) <= 5
    assert len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Whole-cell Sidon supports plus transpose-kernel repair uniformly exclude sub-50 logicals: current accepted slots 21,43,54,58 instead screened at 30,39,45,36, despite four terms per cell and w=12.
# REFUTED: Screening C3-fixed component kernels before unrestricted repair improves the 965.8564 tall-bed record: the current program measured 932.7676, with selected readings 65..71 and k-kb=4.
# REFUTED: Raising B column degrees to four improves the 270.7344 order-183 incumbent: the dense-B portfolio measured 241.3364, and all five measured winners retained A=(4,5) and cubic B columns while degree-four alternatives never advanced.
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
