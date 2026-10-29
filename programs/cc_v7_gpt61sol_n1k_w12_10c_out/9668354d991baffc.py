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
CONCEPT = ("Hypothesis: on split-metacyclic Z97:Z3(s=35), order 291, A=B=(1,2), new binding quartic coupling/gap cards challenged in BOTH C3 modules raise five-code fitness above 982.5226 in cell 0.200. Five readings >=70 at k=295 would score >=993.4708. Measurement can refute this target; finite searches never certify distance.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged at Z97:Z3(s=35), order 291. This horizontal bed leads cell 0.200 at 982.5226, versus 965.8564 for the tall bed. Its faithful order-three action permits fixed and even restrictions of the actual regular components. The even restriction has an F4 scalar action, verified on scaffold-derived binary columns rather than supplied by a separate lift.")
SHAPES = ("Keep A_SHAPE=B_SHAPE=(1,2) byte-for-byte unchanged. Factor five gives n=1455, k>=291 and guaranteed rate 0.200; q_max=299 and the group fixes q=291. Equal horizontal components have 582 variables and 291 checks. Parent minima occupied sector-one strips: X commonly crossed both Bcols at Acol1, Z commonly crossed both Acols at Bcol1, and slot 54 exposed a four-block Z family. Exact full ranks retain the measured k=295 baseline, k-kb=4, inside cell 0.200.")
FILLING = ("Four distinct full-coset terms per cell retain both CSS maxima twelve. Exclude the incumbent six atoms before selecting a new deck; twenty cards bind five coupling-profile pairings to four relative-doublet-gap prescriptions modulo <-1,35>. Repairs preserve these prescriptions while changing cross-cell labels. Derive C3-fixed and C3-even modules by summing VERIFIED component columns. Search the even module with all nonzero F4 coefficients and score EXPANDED binary weights, not coefficient popcounts; retain both module and unrestricted witnesses and test their cancellations. These are diagnostic restrictions, never submitted symmetry copies or distance lower bounds. Every finalist receives equal independent deep testing. Compare all complete pairings using exactly commuting, nontrivial sector-one representatives reduced modulo ORIGINAL stabilizers. Native component agreement, exact k, kernel-four exclusion, weights and CSS remain gates.")
HOW_AB = FILLING
FAMILY_NAME = "r020_m97r3s35_12x12_coupled_gap_dualC3_F4physical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 291

def custom_ops(q):
    # Same split-metacyclic construction rule, resized for factor 5.
    # 35^2=61 and 35^3=1 mod 97; the action has order three.
    if not valid_q(q):
        raise ValueError("this metacyclic bed requires order 291")
    m, r, s = 97, 3, 35
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
B_SHAPE = (1, 2)               # (rb, cb)


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
    """Binding new quartic cards with physically weighted dual-C3 searches."""
    q, width = G.q, 2 * G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (1, 2)
    support = [[4, 4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    trade = (card + cohort) % 4
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0xF49712B8 + 67867967 * (idx + 1)))

    # Cache native operations only. Every binary action block below
    # comes from the authoritative construction, not a handwritten lift.
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

    # Narrow tall galleries provide transpose monomials. Transpose
    # those EXACT bit blocks to obtain the horizontal monomials.
    monomials = {"A": [], "B": []}
    zero = [[[]]]
    for start in range(0, q, 12):
        ids = list(range(start, min(start + 12, q)))
        gallery = [[[g]] for g in ids]
        _, source_a, _ = lifted_matrices(probe_group, gallery, zero)
        source_b, _, _ = lifted_matrices(probe_group, zero, gallery)
        assert len(source_a) == len(source_b) == q
        for j in range(len(ids)):
            shift = (j + 1) * q
            for side, source in (("A", source_a), ("B", source_b)):
                rows = [0] * q
                for r, source_row in enumerate(source):
                    value = (source_row >> shift) & mask
                    assert value.bit_count() == 1
                    rows[value.bit_length() - 1] |= 1 << r
                monomials[side].append(rows)

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

    def component(grid, side):
        """GF2 superposition of scaffold-supplied horizontal blocks."""
        assert len(grid) == 1
        rows = [0] * q
        for j, cell in enumerate(grid[0]):
            for g in cell:
                for r, value in enumerate(monomials[side][G.idx(g)]):
                    rows[r] ^= value << (j * q)
        assert all(row >> (len(grid[0]) * q) == 0 for row in rows)
        return rows

    # A full-coset quartic has a useful normalized coupling parameter:
    # [e,x,y,x^t*y^2]. Swapping the two normal-coset terms replaces
    # t by -t-s after renormalization. Identify that elementary symmetry
    # before assigning cards; do not count its images as new designs.
    alphabet = []
    for t in range(97):
        if t > (-t - 35) % 97:
            continue
        cell = [(0, 0), (1, 0), (0, 1), (t, 2)]
        if not whole_sidon(cell):
            continue
        ranks = tuple(gf2_rank(component([[cell]], side))
                      for side in ("A", "B"))
        if min(ranks) >= q - 3:
            alphabet.append(t)
    if len(alphabet) < 6:
        raise ValueError("insufficient screened quartic coupling classes")

    # Exclude the incumbent deck, rather than restarting its archived
    # element model with a different random stream.
    start = (5 * cohort) % len(alphabet)
    incumbent_catalog = alphabet[start:] + alphabet[:start]
    incumbent_atoms = {
        incumbent_catalog[i * len(incumbent_catalog) // 6]
        for i in range(6)
    }
    catalog = [t for t in incumbent_catalog if t not in incumbent_atoms]
    if len(catalog) < 6:
        raise ValueError("new screened coupling deck is too small")
    shift = (9 + 11 * cohort) % len(catalog)
    catalog = catalog[shift:] + catalog[:shift]
    atom_types = tuple(catalog[i * len(catalog) // 6]
                       for i in range(6))
    assert len(set(atom_types)) == 6
    assert incumbent_atoms.isdisjoint(atom_types)
    profiles = (
        (0, 1), (0, 2), (1, 3),
        (2, 4), (3, 5), (4, 5),
    )
    # Five different pairings crossed with four binding gap relations.
    # These prescriptions do not claim complete Tanner inequivalence.
    plans = ((0, 3), (1, 4), (2, 5), (0, 5), (1, 3))
    plan, gap_policy = divmod(card, 4)
    pa, pb = plans[plan]
    signed_action = (1, 35, 61, 96, 62, 36)
    primitive = next(
        a for a in range(2, 97)
        if pow(a, 48, 97) != 1 and pow(a, 32, 97) != 1)
    gap_classes = {
        "A": ((0, 3, 5, 7)[gap_policy] + 5 * cohort) % 16,
        "B": ((2, 4, 6, 8)[gap_policy] + 5 * cohort) % 16,
    }
    ratios = {side: pow(primitive, exponent, 97)
              for side, exponent in gap_classes.items()}
    allowed_ratios = {
        side: {ratio * action % 97 for action in signed_action}
        for side, ratio in ratios.items()
    }

    def doublet_gap(cell):
        fibers = {}
        for a, b in cell:
            fibers.setdefault(b, []).append(a)
        if sorted(map(len, fibers.values())) != [1, 1, 2]:
            return None
        pair = next(v for v in fibers.values() if len(v) == 2)
        return (pair[1] - pair[0]) % 97

    def preserves_card(grid, side):
        if len(grid) != 1 or len(grid[0]) != 2:
            return False
        gaps = [doublet_gap(cell) for cell in grid[0]]
        if any(g is None or not g for g in gaps):
            return False
        return gaps[1] * pow(gaps[0], -1, 97) % 97 in allowed_ratios[side]

    def fresh_cell(atom, gap):
        """Preserve the atom and prescribed signed-action gap class."""
        assert gap % 97
        for _ in range(512):
            o = rng.randrange(97)
            d = gap * rng.choice(signed_action) % 97
            a = rng.randrange(97)
            phase = rng.randrange(3)
            c = (36 * a - 35 * o + atom * d) % 97
            cell = [
                (o, phase), ((o + d) % 97, phase),
                (a, (phase + 1) % 3), (c, (phase + 2) % 3),
            ]
            if whole_sidon(cell):
                return cell
        raise ValueError("binding coupled-quartic recipe exhausted")

    def fresh(profile, side):
        delta = rng.randrange(1, 97)
        grid = [[
            fresh_cell(atom_types[k],
                       delta * (1 if j == 0 else ratios[side]) % 97)
            for j, k in enumerate(profiles[profile])
        ]]
        assert preserves_card(grid, side)
        return grid

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
        return tuple(tuple(sorted(G.idx(g) for g in cell))
                     for cell in grid[0])

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
        stats = stats + (eight_cycles(rows, columns),)
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cell_ranks": ranks, "cycles": stats,
            "cycle_key": cycle_key(stats), "signature": signature(grid),
            "d": width + 1, "words": [],
        }

    def syndrome(columns, word):
        result = 0
        for c in _bits(word):
            result ^= columns[c]
        return result

    def kernel_probe(record, diagnostic_seed, trials):
        """Persistent genuine kernel witnesses; never certify distance."""
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        retained = sorted(
            {w for w in record["words"] if w},
            key=lambda w: (w.bit_count(), w))[:24]
        reading, witness = width + 1, None
        order = list(range(width))
        additions = []

        def note(word):
            nonlocal reading, witness
            if not word:
                return
            weight = word.bit_count()
            if weight < reading:
                reading, witness = weight, word
            if weight <= reading + 6:
                additions.append(word)
                if len(additions) > 192:
                    additions[:] = sorted(
                        set(additions),
                        key=lambda w: (w.bit_count(), w))[:32]

        for word in retained:
            assert syndrome(columns, word) == 0
            note(word)

        for turn in range(trials):
            diagnostic.shuffle(order)
            if retained and turn % 4 == 3:
                active = list(_bits(diagnostic.choice(retained[:6])))
                diagnostic.shuffle(active)
                preferred = set(active[:max(1, len(active) // 2)])
                order.sort(key=lambda c: c not in preferred)

            pivots, basis, independent = {}, [], []
            additions = list(retained)
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (value, word)
                        independent.append(c)
                        break
                    value ^= old[0]
                    word ^= old[1]
                if not value:
                    basis.append(word)
                    note(word)

            light = sorted(
                set(basis + retained),
                key=lambda w: (w.bit_count(), w))[:20]
            moves = basis + retained
            for word in light:
                for other in moves:
                    note(word ^ other)

            # Bounded hashed four-circuit cancellation. Every result is
            # a kernel word, but hash/list limits can miss lighter ones.
            if trials >= 32 and turn % 4 == 0 and independent:
                chosen = sorted(basis, key=int.bit_count)[:72]
                coords = diagnostic.sample(
                    independent, min(9, len(independent)))
                hash_mask = sum(1 << c for c in coords)
                buckets = {}
                for at, first in enumerate(chosen):
                    for second in chosen[:at]:
                        pair = first ^ second
                        bucket = buckets.setdefault(pair & hash_mask, [])
                        for previous in bucket:
                            note(pair ^ previous)
                        if len(bucket) < 12:
                            bucket.append(pair)

            diagnostic.shuffle(moves)
            for initial in light[:8]:
                word, weight = initial, initial.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    note(word)
                    if not changed:
                        break
                additions.append(word)
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    note(a ^ b ^ c)
            if witness is not None:
                additions.append(witness)
            retained = sorted(
                {w for w in retained + additions if w},
                key=lambda w: (w.bit_count(), w))[:24]
            if reading <= 24:
                break

        if witness is None:
            raise ValueError("horizontal component has no kernel witness")
        assert syndrome(columns, witness) == 0
        assert all((row & witness).bit_count() % 2 == 0
                   for row in record["rows"])
        record["words"] = sorted(
            set(retained + [witness]),
            key=lambda w: (w.bit_count(), w))[:24]
        return reading, witness

    def module_probe(record, side, diagnostic_seed, trials):
        """Genuine fixed/even kernel words, weighted AFTER binary expansion.

        C3 is semisimple over F2. Its even module has scalar alpha
        satisfying alpha^2=alpha+1. This scalar action is checked on
        actual syndrome columns; no representation supplies a new lift.
        """
        if "modules" not in record:
            subgroup = [G.idx((0, b)) for b in range(3)]
            remaining, orbits = set(range(q)), []
            while remaining:
                root = min(remaining)
                orbit = [
                    table[root][h] if side == "A" else table[h][root]
                    for h in subgroup
                ]
                assert len(set(orbit)) == 3
                assert set(orbit) <= remaining
                remaining.difference_update(orbit)
                orbits.append(tuple(orbit))
            assert len(orbits) == 97
            variable_orbits = [
                tuple(j * q + c for c in orbit)
                for j in range(2) for orbit in orbits
            ]
            count = len(variable_orbits)
            low_mask = sum(1 << (2 * c) for c in range(count))

            def alpha(value):
                low = value & low_mask
                high = value & (low_mask << 1)
                return (low << 1) ^ (high >> 1) ^ high

            def fixed_coordinates(value):
                result = 0
                for c, (a, b, d) in enumerate(orbits):
                    x = (value >> a) & 1
                    assert x == ((value >> b) & 1) == ((value >> d) & 1)
                    result |= x << c
                return result

            def even_coordinates(value):
                result = 0
                for c, (a, b, d) in enumerate(orbits):
                    x, z = (value >> a) & 1, (value >> d) & 1
                    assert ((value >> b) & 1) == (x ^ z)
                    result |= (x | (z << 1)) << (2 * c)
                return result

            columns = record["columns"]
            fixed, even = [], []
            for a, b, d in variable_orbits:
                fixed.append(fixed_coordinates(
                    columns[a] ^ columns[b] ^ columns[d]))
                first = even_coordinates(columns[a] ^ columns[b])
                second = even_coordinates(columns[b] ^ columns[d])
                # Regular translation sends (a+b) to (b+d).
                assert second == alpha(first)
                even.append(first)
            record["modules"] = {
                "orbits": variable_orbits, "low_mask": low_mask,
                "fixed": fixed, "even_F4": even,
            }
            record["module_words"] = {"fixed": [], "even_F4": []}
            record["module_readings"] = {
                "fixed": width + 1, "even_F4": width + 1,
            }

        modules = record["modules"]
        low_mask = modules["low_mask"]
        variable_orbits = modules["orbits"]

        def alpha(value):
            low = value & low_mask
            high = value & (low_mask << 1)
            return (low << 1) ^ (high >> 1) ^ high

        def multiply(value, scalar):
            if scalar == 1:
                return value
            image = alpha(value)
            return image if scalar == 2 else value ^ image

        def expand(word, kind):
            actual = 0
            if kind == "fixed":
                for c in _bits(word):
                    for v in variable_orbits[c]:
                        actual ^= 1 << v
            else:
                while word:
                    c = ((word & -word).bit_length() - 1) // 2
                    coefficient = (word >> (2 * c)) & 3
                    a, b, d = variable_orbits[c]
                    if coefficient & 1:
                        actual ^= (1 << a) | (1 << b)
                    if coefficient & 2:
                        actual ^= (1 << b) | (1 << d)
                    word &= ~(3 << (2 * c))
            return actual

        expanded = []
        for kind in ("fixed", "even_F4"):
            columns = modules[kind]
            stride = 1 if kind == "fixed" else 2
            scalars = (1,) if stride == 1 else (1, 2, 3)
            diagnostic = random.Random(
                diagnostic_seed + (0 if stride == 1 else 5000011))

            def physical_weight(word):
                if stride == 1:
                    return 3 * word.bit_count()
                # Each nonzero F4 symbol expands to TWO binary bits,
                # including coefficient 3, whose coefficient popcount is two.
                return 2 * ((word | (word >> 1)) & low_mask).bit_count()

            retained = sorted(
                set(record["module_words"][kind]),
                key=lambda w: (physical_weight(w), w))[:16]
            best = retained[0] if retained else None
            order = list(range(len(columns)))

            for turn in range(trials):
                diagnostic.shuffle(order)
                pivots, basis = {}, []
                for c in order:
                    value, word = columns[c], 1 << (stride * c)
                    while value:
                        lead = (value.bit_length() - 1) // stride
                        coefficient = (
                            1 if stride == 1 else (value >> (2 * lead)) & 3)
                        old = pivots.get(lead)
                        if old is None:
                            inverse = (
                                1 if stride == 1 else
                                {1: 1, 2: 3, 3: 2}[coefficient])
                            pivots[lead] = (
                                multiply(value, inverse),
                                multiply(word, inverse))
                            break
                        value ^= multiply(old[0], coefficient)
                        word ^= multiply(old[1], coefficient)
                    if not value:
                        basis.append(word)

                if not basis:
                    raise ValueError("verified C3 restriction has no kernel")
                grown = list(retained)

                def observe(word):
                    nonlocal best
                    if not word:
                        return
                    weight = physical_weight(word)
                    if best is None or weight < physical_weight(best):
                        best = word
                    if weight <= physical_weight(best) + 6:
                        grown.append(word)
                        if len(grown) > 192:
                            grown[:] = sorted(
                                set(grown),
                                key=lambda w: (physical_weight(w), w))[:24]

                for word in basis:
                    observe(word)
                light = sorted(
                    set(retained + basis),
                    key=lambda w: (physical_weight(w), w))[:12]
                moves = retained + basis
                for word in light:
                    for other in moves:
                        for scalar in scalars:
                            observe(word ^ multiply(other, scalar))

                diagnostic.shuffle(moves)
                for initial in light[:5]:
                    word = initial
                    for sweep in range(2):
                        before = physical_weight(word)
                        for other in moves:
                            winner = word
                            for scalar in scalars:
                                candidate = word ^ multiply(other, scalar)
                                if (candidate and physical_weight(candidate)
                                        < physical_weight(winner)):
                                    winner = candidate
                            word = winner
                        observe(word)
                        if physical_weight(word) == before:
                            break
                retained = sorted(
                    set(retained + grown + [best]),
                    key=lambda w: (physical_weight(w), w))[:16]
                if physical_weight(best) <= 24:
                    break

            assert best
            actual = expand(best, kind)
            assert actual.bit_count() == physical_weight(best)
            assert syndrome(record["columns"], actual) == 0
            record["module_words"][kind] = retained
            record["module_readings"][kind] = min(
                record["module_readings"][kind], actual.bit_count())
            expanded.extend(expand(word, kind)
                            for word in retained[:8] + [best])

        # Fixed + even cancellations challenge mixed-module words too.
        # This finite list is not an exhaustive decomposition search.
        words = sorted(
            set(record["words"] + expanded),
            key=lambda w: (w.bit_count(), w))
        assert all(w and syndrome(record["columns"], w) == 0 for w in words)
        additions = []
        for at, first in enumerate(words[:16]):
            for second in words[:at]:
                candidate = first ^ second
                if candidate:
                    additions.append(candidate)
        record["words"] = sorted(
            set(words + additions),
            key=lambda w: (w.bit_count(), w))[:24]
        assert record["words"]
        record["d"] = min(
            record["d"], record["words"][0].bit_count())

    def optimize(profile, side):
        bank = set()
        seen = set()
        records = []
        offset = 0 if side == "A" else 100003

        def probe(record, trials, turn):
            module_probe(
                record, side, seed + offset + 900001 * turn,
                max(4, trials // 3))
            d, word = kernel_probe(
                record, seed + offset + 700001 * turn, trials)
            record["d"] = min(record["d"], d)
            bank.update(record["words"][:4])
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
            # Reduce selection pressure on optimistic finite searches.
            # This cap is not a lower bound on component distance.
            return (min(76, record["d"]),
                    tuple(-x for x in record["cycle_key"]),
                    min(96, record["d"]), record["rank"])

        for trial in range(96 + 4 * min(cohort, 2)):
            record = inspect(fresh(profile, side), side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            records.append(record)
        records.sort(key=lambda record: record["cycle_key"])

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
            atom = atom_types[profiles[profile][j]]
            witnesses = sorted(
                set(current["words"]),
                key=lambda w: (w.bit_count(), w))[:6]
            proposals = []
            # Preserve both the assigned atoms and the binding gap card.
            # Periodic two-cell proposals change both gaps together;
            # other proposals replace the implicated cell independently.
            for attempt in range(5):
                if step % 6 == 0:
                    grid = fresh(profile, side)
                else:
                    grid = [[list(cell) for cell in current["grid"][0]]]
                    gap = doublet_gap(grid[0][j])
                    grid[0][j] = fresh_cell(atom, gap)
                assert preserves_card(grid, side)
                columns = _syndrome_columns(component(grid, side), width)
                boundaries = [
                    syndrome(columns, word).bit_count()
                    for word in witnesses
                ]
                key = (
                    sum(value > 0 for value in boundaries),
                    min(boundaries, default=0), sum(boundaries),
                )
                proposals.append((key, grid))
            proposals.sort(key=lambda item: item[0], reverse=True)
            grid = proposals[0][1]
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            if proposed["cycles"][0] > floor4 + allowance:
                continue
            # Transfer only words that remain genuine for the NEW matrix.
            proposed["words"] = [
                word for word in current["words"]
                if syndrome(proposed["columns"], word) == 0
            ]
            probe(proposed, 8, 1 + step)
            pk, ck = merit(proposed), merit(current)
            if pk > ck or (pk[0] == ck[0] and rng.random() < 0.15):
                current = proposed
                selected.append(proposed)

        selected.sort(key=merit, reverse=True)
        finalists, final_seen = [], set()
        for record in selected:
            if record["signature"] in final_seen:
                continue
            final_seen.add(record["signature"])
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 48, 101)
            finalists.append(record)
            if len(finalists) >= 4:
                break
        for record in finalists:
            # EVERY contender receives equal independent full/module
            # confirmation; no shallow record replaces a weakened winner.
            probe(record, 96 + 16 * min(cohort, 2), 211)
        finalists.sort(key=merit, reverse=True)
        return finalists[:3]

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
        """Actual logical upper readings in the ORIGINAL stabilizer quotient."""
        readings = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            embedded = set()
            for word in record["words"][:4]:
                assert syndrome(record["columns"], word) == 0
                for strip in range(2):
                    actual = (
                        sum(((word >> (j * q)) & mask)
                            << ((2 * j + strip) * q)
                            for j in range(2))
                        if kind == "Z" else word << (2 * strip * q))
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        embedded.add(actual)
            singles = sorted(embedded, key=lambda w: (w.bit_count(), w))
            candidates = set(singles)
            for at, first in enumerate(singles):
                for second in singles[:at]:
                    candidate = first ^ second
                    if candidate and outside(candidate, pivots):
                        candidates.add(candidate)

            reading = n + 1
            for actual in sorted(
                    candidates, key=lambda w: (w.bit_count(), w))[:8]:
                for sweep in range(3):
                    before = actual.bit_count()
                    for row in stabilizers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual = candidate
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:32]
                    winner = actual
                    for at, first in enumerate(nearby):
                        for second in nearby[:at]:
                            candidate = actual ^ first ^ second
                            if candidate.bit_count() < winner.bit_count():
                                winner = candidate
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            # n+1 means no nontrivial embedding was found, not exclusion.
            readings.append(reading)
        return tuple(readings)

    for restart in range(2):
        aa = optimize(pa, "A")
        bb = optimize(pb, "B")
        evaluated = []
        for a in aa:
            for b in bb:
                A, B = a["grid"], b["grid"]
                assert preserves_card(A, "A") and preserves_card(B, "B")
                hx, hz, n = lifted_matrices(G, A, B)
                assert n == 5 * q == 1455
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k == q + 4 and 7 * n <= 40 * k < 9 * n):
                    continue
                actual_a, actual_b = _verified_components(hx, hz, q)
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logicals = logical_readings(a, b, hx, hz, n)
                detected = min(a["d"], b["d"], *logicals)
                key = (
                    k * min(76, detected) ** 2,
                    min(96, detected),
                    tuple(-(x + y) for x, y in
                          zip(a["cycle_key"], b["cycle_key"])),
                    min(96, a["d"]) + min(96, b["d"]),
                )
                evaluated.append((key, a, b, hx, hz, n, k, logicals))
        evaluated.sort(key=lambda item: item[0], reverse=True)

        for key, a, b, hx, hz, n, k, logicals in evaluated:
            if not _no_kernel4(_syndrome_columns(hx, n)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, n)):
                continue
            assert max(row.bit_count() for row in hx + hz) == 12
            if not css_ok(hx, hz):
                raise ValueError("verified construction failed CSS")
            return {
                "A": a["grid"], "B": b["grid"], "idx": idx, "k": k,
                "card": (plan, gap_policy, trade, cohort),
                "coupling_types": atom_types,
                "excluded_incumbent_atoms": tuple(sorted(incumbent_atoms)),
                "coupling_profiles": (profiles[pa], profiles[pb]),
                "relative_gap_classes": (
                    gap_classes["A"], gap_classes["B"]),
                "allowed_normal_ratios": {
                    side: tuple(sorted(values))
                    for side, values in allowed_ratios.items()
                },
                "component_readings": (a["d"], b["d"]),
                "physical_C3_module_readings": (
                    dict(a["module_readings"]), dict(b["module_readings"])),
                "component_cycles": (a["cycles"], b["cycles"]),
                "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
                "verified_sector_one_logical_readings": logicals,
                "pairing_merit": key,
                "even_module_metric": "two binary bits per nonzero F4 symbol",
            }
    raise ValueError("no dual-C3 challenged coupling/gap pair passed full gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete screened cell and verify its binding atom/gap card."""
    cell = list(ctx[side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) == 4
    assert sorted(sum(g[1] == b for g in cell)
                  for b in range(3)) == [1, 1, 2]
    o, phase = cell[0]
    assert cell[1][1] == phase
    assert cell[2][1] == (phase + 1) % 3
    assert cell[3][1] == (phase + 2) % 3
    delta = (cell[1][0] - o) % 97
    atom = ((cell[3][0] - 36 * cell[2][0] + 35 * o)
            * pow(delta, -1, 97)) % 97
    profile = ctx["coupling_profiles"][0 if side == "A" else 1]
    assert atom == ctx["coupling_types"][profile[j]]
    assert atom not in ctx["excluded_incumbent_atoms"]
    gaps = [(c[1][0] - c[0][0]) % 97 for c in ctx[side][0]]
    relative = gaps[1] * pow(gaps[0], -1, 97) % 97
    assert relative in ctx["allowed_normal_ratios"][side]
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Coupled full-coset quartics with persistent unrestricted component challenges uniformly exclude sub-30 logical readings; incumbent horizontal slots 30 and 33 screened at 27 with k=295 and w=12.
# REFUTED: Translation closure or binding gap cards with stabilizer beams alone improve the 982.5226 horizontal record; the shown variants scored 943.8784 and 939.1340 respectively.
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
