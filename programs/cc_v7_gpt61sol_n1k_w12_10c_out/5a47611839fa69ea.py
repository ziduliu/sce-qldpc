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
CONCEPT = ("Hypothesis: on split-metacyclic Z97:Z3(s=35), order 291, A=B=(2,1), joint doublet-gap/quartic-coupling cards with persistent C3-fixed and unrestricted kernel challenges raise five-code fitness above 982.5226 in cell 0.200. Transposition alone is not credited with improvement; finite witness readings remain upper bounds.")
GROUP = ("Keep GTYPE, A0 and CUSTOM_GROUP byte-for-byte unchanged: Z97:Z3(s=35), q=291. Its horizontal and tall beds reached 982.5226 and 965.8564. Retain the faithful order-three action and prime normal subgroup, but generate independently new support relations. Those readings motivate this experiment, not certified distance transfer; symmetry images are diagnostics only.")
SHAPES = ("Change BOTH sides to (2,1). Factor five retains n=1455, k>=291 and guaranteed rate 0.200; q_max=299 and the unchanged group fixes q=291. Equal distance-bearing transpose components each have 582 variables and 291 checks. Parent minimum witnesses were sector-one strips: X commonly used one Acol across both Bcols, and Z one Bcol across both Acols. Analogous tall families occupy sector two. Exact full ranks enforce [0.175,0.225), rather than infer dimension from component ranks.")
FILLING = ("All four cells contain four distinct full-coset terms, giving BOTH CSS sums 4+8=12. Twenty unordered pairs of six NEW joint profiles prescribe both normalized quartic couplings and the ratio of the two normal-coset doublet gaps modulo <-1,35>. Whole-cell replacements preserve these relations; occasional two-cell replacements independently redraw origins and singleton labels, not a common scaling image. Authoritative tall monomial blocks supply transpose components. Diagnose every initial proposal in the C3-fixed word space, retain and expand genuine witnesses, seed unrestricted information-set and anchored OSD searches, and independently deepen finalists. Original-code sector-two logical retests, full ranks, component agreement, kernel-four exclusion, weights and CSS remain binding. This targets parent short readings 18..48 and selected drops from 74..78 to 65..71; neither fixed-space searches nor optimistic readings certify distance.")
HOW_AB = FILLING
FAMILY_NAME = "r020_m97r3s35_21x21_jointgap_coupled_C3binding_v3"

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

A_SHAPE = (2, 1)               # (ra, ca)
B_SHAPE = (2, 1)               # (rb, cb)


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
    """Twenty joint-gap tall cards with binding fixed/full kernel retests."""
    q, width = G.q, 2 * G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (2, 1)
    support = [[4], [4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    trade = (card + cohort) % 4
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0x29121C3B7 + 49979687 * (idx + 1)))

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

    # Keep the scaffold's TALL transpose blocks directly. Their sector
    # two starts after q columns; no independent action matrix is written.
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
                rows = [(row >> shift) & mask for row in source]
                assert all(row.bit_count() == 1 for row in rows)
                monomials[side].append(rows)

    # Coordinate restrictions of the ACTUAL components, not quotient
    # lift conventions. All expanded words are checked on native syndromes.
    complement = [G.idx((0, b)) for b in range(3)]
    fixed_masks = {}
    for side in ("A", "B"):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {
                table[root][h] if side == "A" else table[h][root]
                for h in complement
            }
            assert len(orbit) == 3 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(sum(1 << c for c in orbit))
        assert len(orbits) == 97
        fixed_masks[side] = [
            word << (j * q) for j in range(2) for word in orbits
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

    def component(grid, side):
        """Distance-bearing transpose component from verified tall blocks."""
        assert all(len(row) == 1 for row in grid)
        rows = [0] * q
        for j, base_row in enumerate(grid):
            for g in base_row[0]:
                for r, value in enumerate(monomials[side][G.idx(g)]):
                    rows[r] ^= value << (j * q)
        assert all(row >> (len(grid) * q) == 0 for row in rows)
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

    # Spread the atom classes across the legal catalog. Later cohorts
    # choose genuinely different class subsets, not affine label images.
    start = (11 + 7 * cohort) % len(alphabet)
    alphabet = alphabet[start:] + alphabet[:start]
    atom_types = tuple(alphabet[i * len(alphabet) // 6]
                       for i in range(6))
    assert len(set(atom_types)) == 6
    profiles = (
        (0, 0), (0, 2), (1, 4),
        (2, 5), (3, 3), (4, 5),
    )
    # F97*/<-1,35> has sixteen classes. Exponents 1..8 choose
    # representatives modulo inversion from swapping the two rows.
    primitive = next(
        a for a in range(2, 97)
        if pow(a, 48, 97) != 1 and pow(a, 32, 97) != 1)
    gap_classes = tuple(1 + (p + 3 * cohort) % 8 for p in range(6))
    gap_ratios = tuple(pow(primitive, p, 97) for p in gap_classes)
    signed_action = (1, 35, 61, 96, 62, 36)
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    assert len(plans) == 20
    pa, pb = plans[card]

    def fresh_cell(atom, gap):
        """Preserve both the assigned atom and prescribed doublet-gap class.

        Origins, singleton labels and quotient phases are independently
        fresh; different draws are not assumed to be inequivalent.
        """
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
        raise ValueError("coupled whole-cell Sidon recipe exhausted")

    def doublet_gap(cell):
        fibers = {}
        for a, b in cell:
            fibers.setdefault(b, []).append(a)
        assert sorted(map(len, fibers.values())) == [1, 1, 2]
        pair = next(values for values in fibers.values() if len(values) == 2)
        return (pair[1] - pair[0]) % 97

    def preserves_card(grid, profile):
        if len(grid) != 2 or any(len(row) != 1 for row in grid):
            return False
        first = doublet_gap(grid[0][0])
        second = doublet_gap(grid[1][0])
        relative = second * pow(first, -1, 97) % 97
        return relative in {
            gap_ratios[profile] * action % 97 for action in signed_action
        }

    def fresh(profile):
        delta = rng.randrange(1, 97)
        grid = [
            [fresh_cell(atom_types[k],
                        delta * (1 if j == 0 else gap_ratios[profile]) % 97)]
            for j, k in enumerate(profiles[profile])
        ]
        assert preserves_card(grid, profile)
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
        probe_width = len(columns)
        reading, witness = probe_width + 1, None
        order = list(range(probe_width))
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
            if reading * record.get("unit_weight", 1) <= 24:
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

    def complement_probe(record, side, diagnostic_seed, trials):
        """Persistent C3-fixed witnesses, expanded into the actual kernel.

        The restriction is a diagnostic subset, never a quantum-distance
        certificate. Keep every original syndrome coordinate.
        """
        masks = fixed_masks[side]
        fixed = record.get("fixed_record")
        if fixed is None:
            columns = [syndrome(record["columns"], word) for word in masks]
            # Transposing these syndrome vectors gives the exact binary
            # restriction matrix, without an assumed quotient convention.
            rows = _syndrome_columns(columns, q)
            assert gf2_rank(rows) <= 97
            fixed = {
                "columns": columns, "rows": rows, "words": [],
                "unit_weight": 3,
            }
            record["fixed_record"] = fixed
        reading, witness = kernel_probe(fixed, diagnostic_seed, trials)

        def expand(word):
            actual = 0
            for c in _bits(word):
                actual ^= masks[c]
            assert actual.bit_count() == 3 * word.bit_count()
            assert syndrome(record["columns"], actual) == 0
            return actual

        actuals = [expand(word) for word in fixed["words"][:12]]
        actual = expand(witness)
        assert all((row & actual).bit_count() % 2 == 0
                   for row in record["rows"])
        record["words"] = sorted(
            set(record["words"] + actuals + [actual]),
            key=lambda w: (w.bit_count(), w))[:24]
        record["fixed_d"] = min(
            record.get("fixed_d", width + 1), 3 * reading)
        record["d"] = min(record["d"], record["fixed_d"])
        return record["fixed_d"], actual

    def anchored_probe(record, side, diagnostic_seed, attempts):
        """Forced-anchor decoding/OSD returns genuine nonzero kernel words.

        Failure or an optimistic reading never excludes a lighter word.
        Circuit moves omit the anchor, so cancellation cannot produce zero.
        """
        local = random.Random(diagnostic_seed)
        columns, rows = record["columns"], record["rows"]
        retained = sorted(set(record["words"]),
                          key=lambda w: (w.bit_count(), w))[:24]
        assert retained
        best = retained[0]
        found = list(retained)

        def observe(word):
            nonlocal best
            assert word and syndrome(columns, word) == 0
            if word.bit_count() < best.bit_count():
                best = word
            if word.bit_count() <= best.bit_count() + 6:
                found.append(word)
                if len(found) > 96:
                    found[:] = sorted(set(found),
                                      key=lambda w: (w.bit_count(), w))[:24]

        for attempt in range(attempts):
            block = attempt % 2
            root = block * q + identity
            root_bit = 1 << root
            target = columns[root]
            edge_variables = []
            check_edges = [[] for _ in rows]
            for c, value in enumerate(columns):
                if c == root:
                    continue
                for r in _bits(value):
                    edge = len(edge_variables)
                    edge_variables.append(c)
                    check_edges[r].append(edge)

            channel = [local.uniform(1.0, 3.2) for _ in columns]
            nearby = sorted({
                edge_variables[e]
                for r in _bits(target) for e in check_edges[r]
            })
            warm = [word for word in retained
                    if (word >> (block * q)) & mask]
            if attempt % 3 == 1 and warm:
                source = local.choice(warm[:8])
                anchors = list(_bits((source >> (block * q)) & mask))
                multiplier = inverses[local.choice(anchors)]
                translated = 0
                for c in _bits(source):
                    j, coordinate = divmod(c, q)
                    coordinate = (
                        table[coordinate][multiplier] if side == "A"
                        else table[multiplier][coordinate])
                    translated |= 1 << (j * q + coordinate)
                assert translated & root_bit
                assert syndrome(columns, translated) == 0
                observe(translated)
                for c in _bits(translated ^ root_bit):
                    channel[c] = local.uniform(0.08, 0.8)
            elif attempt % 3 == 2 and nearby:
                for c in local.sample(nearby, min(3, len(nearby))):
                    channel[c] = -local.uniform(0.1, 1.0)
            else:
                for c in nearby:
                    channel[c] *= local.uniform(0.4, 1.0)

            variable_messages = [channel[c] for c in edge_variables]
            check_messages = [0.0] * len(edge_variables)
            posterior = list(channel)
            hard = 0
            for iteration in range(18):
                for r, incident in enumerate(check_edges):
                    sign_product = -1 if (target >> r) & 1 else 1
                    first = second = 24.0
                    smallest = -1
                    for e in incident:
                        value = variable_messages[e]
                        if value < 0:
                            sign_product = -sign_product
                        magnitude = abs(value)
                        if magnitude < first:
                            second, first = first, magnitude
                            smallest = e
                        elif magnitude < second:
                            second = magnitude
                    for e in incident:
                        sign = -1 if variable_messages[e] < 0 else 1
                        magnitude = second if e == smallest else first
                        proposed = 0.8 * sign_product * sign * magnitude
                        check_messages[e] = (
                            0.55 * check_messages[e] + 0.45 * proposed)

                posterior = list(channel)
                for e, c in enumerate(edge_variables):
                    posterior[c] += check_messages[e]
                hard = sum(1 << c for c in range(width)
                           if c != root and posterior[c] < 0)
                if syndrome(columns, hard) == target:
                    observe(hard ^ root_bit)
                    break
                for e, c in enumerate(edge_variables):
                    value = posterior[c] - check_messages[e]
                    variable_messages[e] = max(-24.0, min(
                        24.0, 0.4 * variable_messages[e] + 0.6 * value))

            # Low-reliability columns provide a syndrome-completion basis.
            # These are authoritative columns, not an assumed quotient lift.
            reliability = [abs(value) + local.uniform(0.0, 0.2)
                           for value in posterior]
            order = [c for c in range(width) if c != root]
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
            assert syndrome(columns, candidate) == 0
            for sweep in range(2):
                before = candidate.bit_count()
                local.shuffle(circuits)
                for move in circuits:
                    proposed = candidate ^ move
                    if proposed.bit_count() < candidate.bit_count():
                        candidate = proposed
                near = sorted(circuits,
                              key=lambda w: (candidate ^ w).bit_count())[:24]
                winner = candidate
                for at, first in enumerate(near):
                    partial = candidate ^ first
                    for second in near[:at]:
                        proposed = partial ^ second
                        if proposed.bit_count() < winner.bit_count():
                            winner = proposed
                candidate = winner
                if candidate.bit_count() == before:
                    break
            assert candidate & root_bit
            observe(candidate)
            if best.bit_count() <= 24:
                break

        record["words"] = sorted(set(found + retained + [best]),
                                 key=lambda w: (w.bit_count(), w))[:24]
        assert all(syndrome(columns, word) == 0
                   for word in record["words"])
        assert all((best & row).bit_count() % 2 == 0 for row in rows)
        record["anchored_d"] = min(
            record.get("anchored_d", width + 1), best.bit_count())
        return best.bit_count(), best

    def optimize(profile, side):
        bank = set()
        seen = set()
        records = []
        offset = 0 if side == "A" else 100003

        def probe(record, trials, turn):
            diagnostic_seed = seed + offset + 700001 * turn
            fixed_d, fixed_word = complement_probe(
                record, side, diagnostic_seed + 5300023, max(8, trials))
            # Expanded fixed words seed the unrestricted search; neither
            # a new information set nor OSD can erase an earlier witness.
            d, word = kernel_probe(record, diagnostic_seed, trials)
            attempts = 1 if trials < 32 else (4 if trials < 96 else 8)
            anchored_d, anchored_word = anchored_probe(
                record, side, diagnostic_seed + 2300009, attempts)
            record["d"] = min(record["d"], fixed_d, d, anchored_d)
            bank.update(record["words"][:4])
            bank.update((word, anchored_word, fixed_word))
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
            return (min(74, record["d"]),
                    min(84, record["fixed_d"]),
                    tuple(-x for x in record["cycle_key"]),
                    min(90, record["d"]), record["rank"])

        for trial in range(96 + 4 * min(cohort, 2)):
            record = inspect(fresh(profile), side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            complement_probe(record, side, seed + offset + 400009, 4)
            records.append(record)
        records.sort(key=lambda record: (
            -min(84, record["fixed_d"]), record["cycle_key"]))

        selected = []
        for record in records:
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 12, 0)
            selected.append(record)
            if len(selected) >= 12:
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
            # Whole-cell replacements preserve the assigned normalized
            # atom, but change its relation to the OTHER component cell.
            # Compare actual disruption of genuine retained witnesses.
            for attempt in range(5):
                if step % 6 == 5:
                    # Redraw BOTH supports with a new common gap scale,
                    # but independent origins/singletons, not an orbit copy.
                    grid = fresh(profile)
                else:
                    grid = [[list(row[0])] for row in current["grid"]]
                    grid[j][0] = fresh_cell(
                        atom, doublet_gap(grid[j][0]))
                assert preserves_card(grid, profile)
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
        finalists = []
        for record in selected:
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 48, 101)
            finalists.append(record)
            if len(finalists) >= 4:
                break
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            # Independent confirmation only lowers a retained reading.
            probe(record, 128 + 16 * min(cohort, 2), 211)
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

    def logical_retest(a, b, hx, hz, n):
        """Sector-two logical readings in the ORIGINAL stabilizer quotient.

        n+1 means no nontrivial embedding was found, not a distance bound.
        Adding stabilizers preserves exact commutation and nontriviality.
        """
        readings = []
        for kind, record, opposite, stabilizers in (
                ("X", a, hz, hx), ("Z", b, hx, hz)):
            pivots = span(stabilizers)
            words = sorted(set(record["words"]),
                           key=lambda w: (w.bit_count(), w))[:4]
            candidates = set()
            for word in words:
                assert syndrome(record["columns"], word) == 0
                for strip in range(2):
                    actual = (
                        sum(((word >> (j * q)) & mask)
                            << ((1 + 2 * j + strip) * q)
                            for j in range(2))
                        if kind == "X" else word << ((1 + 2 * strip) * q))
                    assert actual >> n == 0
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)

            reading = n + 1
            diagnostic = random.Random(
                seed + (4100009 if kind == "Z" else 5100011))
            reducers = list(stabilizers)
            diagnostic.shuffle(reducers)
            for actual in sorted(candidates,
                                 key=lambda w: (w.bit_count(), w))[:4]:
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
                    winner = actual
                    for at, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:at]:
                            proposed = partial ^ second
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

    for restart in range(2):
        aa = optimize(pa, "A")
        bb = optimize(pb, "B")
        evaluated = []
        for a in aa:
            for b in bb:
                hx, hz, n = lifted_matrices(G, a["grid"], b["grid"])
                assert n == 5 * q == 1455
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (k >= q and 7 * n <= 40 * k < 9 * n):
                    continue
                # Sector two is ordered (Arow, Brow), Brow varying first.
                # Confirm BOTH cached transpose components on this pairing.
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
                logicals = logical_retest(a, b, hx, hz, n)
                detected = min(a["d"], b["d"], *logicals)
                # Limit pressure on optimistic CPU readings. This is
                # a selection cap, never a distance lower bound.
                capped = min(74, detected)
                key = (
                    k * capped * capped,
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
            assert preserves_card(a["grid"], pa)
            assert preserves_card(b["grid"], pb)
            assert _grid_weights(support, support) == 12
            return {
                "A": a["grid"], "B": b["grid"], "idx": idx, "k": k,
                "card": (pa, pb, trade, cohort),
                "coupling_types": atom_types,
                "coupling_profiles": (profiles[pa], profiles[pb]),
                "relative_gap_classes": (gap_classes[pa], gap_classes[pb]),
                "component_readings": (a["d"], b["d"]),
                "C3_fixed_component_readings": (a["fixed_d"], b["fixed_d"]),
                "anchored_component_readings": (
                    a["anchored_d"], b["anchored_d"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
                "verified_sector_two_logical_readings": logicals,
                "pairing_merit": key,
                "selection_model": "joint-gap coupling, persistent C3/full/OSD challenges",
            }
    raise ValueError("no joint-gap tall pair passed full gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected joint-gap quartic cell of the tall shape."""
    cell = list(ctx[side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) == 4
    assert sorted(sum(g[1] == b for g in cell)
                  for b in range(3)) == [1, 1, 2]
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Coupled full-coset quartics with persistent component information-set challenges uniformly exclude sub-30 logical readings; current horizontal slots 30 and 33 instead screened at 27 with k=295 and w=12.
# REFUTED: Adding anchored min-sum/OSD and original-span retests to the horizontal coupled-quartic program raises five-code fitness above 982.5226; the measured child instead scored 927.4556, selected readings 65..71, and an accepted slot screened at 18.
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
