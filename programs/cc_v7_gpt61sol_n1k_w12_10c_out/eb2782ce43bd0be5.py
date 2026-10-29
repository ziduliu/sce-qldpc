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
CONCEPT = ("Hypothesis: on split-metacyclic Z37:Z3(s=10), order 111, A=(1,2), B=(2,5), balanced degree-three/four B columns and binding component/original-stabilizer challenges produce five NEW measured readings >=24 in cell 0.250. Since k/n>=0.250, that target scores at least 144, above the frontier's 124.6. Finite readings remain provisional upper bounds.")
GROUP = ("Keep the split-metacyclic construction RULE, resizing m,r,s to 37,3,10. Here 10^3=1 modulo 37 and the action is nontrivial. For factor twelve, q_max=124; 111 is the largest order with prime normal subgroup and faithful order-three action. Related Z37:Z3 beds reached 359.4 at lower rate, while this exact horizontal shape previously reached 103.0. These measurements motivate new fillings and screening, not certified distance transfer.")
SHAPES = ("Change A to (1,2) and B to (2,5): n=12*111=1332, k>=3*111=333 and guaranteed rate 0.250. Horizontal A has 222 variables/111 checks; B has 555 variables/222 checks, nominal rates one-half and three-fifths. Parent minima occupied sector-two strips spanning both Arows at one Brow or both Brows at one Arow. Replace those equal transpose components with unequal horizontal components, and enforce 0.225<=actual k/n<0.275 using authoritative full ranks.")
FILLING = ("A counts (4,4), B row totals eight and column totals (4,3,3,3,3) give BOTH CSS maxima 8+4=12. Twenty cards cross five structurally different B support topologies with four A occupancy pairs; later cohorts change higher-cell occupancy/affine geometry. Binomials use nonzero normal-subgroup differences of order 37, while higher cells satisfy bidirectional Sidon tests. All binary component blocks come from verified monomial lifts. Accumulate unrestricted and C3-orbit-restricted genuine kernel witnesses, repair implicated cells while preserving each card, and independently deepen finalists. Complete pairings challenge exactly commuting non-stabilizer sector-one strips with original-check uphill beam and two-row descent. Native assembly, full k, distinctness, caps and CSS remain binding. Neither finite searches nor card differences certify distance or Tanner inequivalence.")
HOW_AB = FILLING
FAMILY_NAME = "r025_split_m37r3s10_12x25_balanced34_fixed_joint_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 111

def custom_ops(q):
    # SAME split-metacyclic rule, resized for factor twelve.
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

A_SHAPE = (1, 2)               # (ra, ca)
B_SHAPE = (2, 5)               # (rb, cb)


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


def _retired_tall_orbit_ctx(rng, G, idx):
    """Inactive factor-five experiment; the new shape uses NEW_CTX below."""
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
    # Fresh supports, not archived supports followed by a coordinate gauge.
    rng = random.Random(seed + 32452843 * (idx + 1) + 86000021)
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

    # Cache exact transpose monomials from narrow scaffold-built galleries.
    # Subsequent component assembly uses only GF2 linearity of these blocks.
    monomials = {"A": [], "B": []}
    unit = [[[identity]]]
    for start in range(0, q, 12):
        ids = list(range(start, min(start + 12, q)))
        gallery = [[[g]] for g in ids]
        _, source_a, _ = lifted_matrices(probe_group, gallery, unit)
        source_b, _, _ = lifted_matrices(probe_group, unit, gallery)
        assert len(source_a) == len(source_b) == q
        for j in range(len(ids)):
            shift = (j + 1) * q
            for side, source in (("A", source_a), ("B", source_b)):
                block = [(row >> shift) & mask for row in source]
                assert all(row.bit_count() == 1 for row in block)
                monomials[side].append(block)
    assert all(len(blocks) == q for blocks in monomials.values())

    # Coordinate actions only, obtained from native multiplication.
    # A transpose components commute with right translation; B with left.
    # Every translated word is independently syndrome-verified below.
    actions = {
        "A": [[table[c][h] for c in range(q)] for h in range(q)],
        "B": [[table[h][c] for c in range(q)] for h in range(q)],
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

    def fresh(profile):
        return [[fresh_cell(occupancy)]
                for occupancy in profiles[profile]]

    def component(grid, side):
        """GF2 sums of authoritative transpose-monomial blocks."""
        rows = [0] * q
        for j, base_row in enumerate(grid):
            for g in base_row[0]:
                for r, value in enumerate(monomials[side][G.idx(g)]):
                    rows[r] ^= value << (j * q)
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
        stats = stats + (eight_cycles(rows, columns),)
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cell_ranks": ranks, "cycles": stats,
            "cycle_key": cycle_key(stats), "signature": signature(grid),
            "side": side, "d": width + 1, "words": [],
            "orbit_anchor": None, "orbit_moves": [],
            "orbit_cancellations": 0,
        }

    def syndrome(columns, word):
        result = 0
        for c in _bits(word):
            result ^= columns[c]
        return result

    def kernel_probe(record, diagnostic_seed, trials):
        """Orbit-expanded cancellation of genuine component kernel words.

        Translations enlarge the diagnostic move set only. Output cells
        are never obtained by translating an archived or another slot's code.
        All readings are witnessed upper readings, not exclusions.
        """
        diagnostic = random.Random(diagnostic_seed)
        columns = record["columns"]
        retained = sorted(
            {w for w in record["words"] if w},
            key=lambda w: (w.bit_count(), w))[:16]
        reading, witness = width + 1, None
        order = list(range(width))

        def note(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        for word in retained:
            assert syndrome(columns, word) == 0
            note(word)

        def orbit(word):
            positions = list(_bits(word))
            result = set()
            for permutation in actions[record["side"]]:
                actual = 0
                for c in positions:
                    block, local = divmod(c, q)
                    actual |= 1 << (block * q + permutation[local])
                assert actual.bit_count() == word.bit_count()
                assert syndrome(columns, actual) == 0
                result.add(actual)
            return sorted(result)

        for turn in range(trials):
            diagnostic.shuffle(order)
            if turn % 4 == 3 and retained:
                # Independent outside coordinates accompany a partial
                # support from a previously verified circuit.
                anchor = diagnostic.choice(retained[:6])
                active = list(_bits(anchor))
                diagnostic.shuffle(active)
                preferred = set(active[:max(1, len(active) // 2)])
                order.sort(key=lambda c: c not in preferred)

            values = [0] * q
            pivot_words = [0] * q
            basis = []
            for c in order:
                value, word = columns[c], 1 << c
                while value:
                    p = value.bit_length() - 1
                    if not values[p]:
                        values[p], pivot_words[p] = value, word
                        break
                    value ^= values[p]
                    word ^= pivot_words[p]
                if not value:
                    basis.append(word)
                    note(word)
            if not basis:
                raise ValueError("transpose component unexpectedly injective")

            light = sorted(
                set(basis + retained),
                key=lambda w: (w.bit_count(), w))[:16]
            grown = list(retained) + light

            # Refresh a full orbit when a strictly lighter anchor appears.
            # Reusing it avoids rebuilding hundreds of permutations on
            # every information set. Cross-orbit cancellations are genuine
            # circuits even when their summands came from different bases.
            old_anchor = record["orbit_anchor"]
            if ((turn % 6 == 0 or not record["orbit_moves"])
                    and witness is not None
                    and (old_anchor is None or
                         witness.bit_count() < old_anchor.bit_count())):
                record["orbit_anchor"] = witness
                record["orbit_moves"] = orbit(witness)
                before = reading
                for initial in light[:8]:
                    best = initial
                    for translated in record["orbit_moves"]:
                        candidate = initial ^ translated
                        note(candidate)
                        if (candidate and candidate.bit_count()
                                < best.bit_count()):
                            best = candidate
                    grown.append(best)
                record["orbit_cancellations"] += int(reading < before)

            translated_moves = record["orbit_moves"]
            if len(translated_moves) > 48:
                translated_moves = diagnostic.sample(translated_moves, 48)
            moves = basis + retained + translated_moves

            for initial in light:
                best = initial
                for other in moves:
                    candidate = initial ^ other
                    note(candidate)
                    if candidate and candidate.bit_count() < best.bit_count():
                        best = candidate
                grown.append(best)

            diagnostic.shuffle(moves)
            starts = sorted(set(grown), key=int.bit_count)[:6]
            for initial in starts:
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
                grown.append(word)

            # Retain newly shortened circuits across information sets.
            short = sorted(set(grown), key=int.bit_count)[:16]
            for i, first in enumerate(short):
                for second in short[:i]:
                    note(first ^ second)
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    note(a ^ b ^ c)
            if witness is not None:
                grown.append(witness)
            retained = sorted(
                {w for w in grown if w},
                key=lambda w: (w.bit_count(), w))[:16]
            if reading <= 24:
                break

        if witness is None:
            raise ValueError("transpose component has no kernel witness")
        assert syndrome(columns, witness) == 0
        assert all((row & witness).bit_count() % 2 == 0
                   for row in record["rows"])
        record["words"] = sorted(
            set(retained + [witness]),
            key=lambda w: (w.bit_count(), w))[:16]
        return reading, witness

    def optimize(profile, side):
        bank = set()
        seen = set()
        records = []
        offset = 0 if side == "A" else 100003

        def probe(record, trials, turn):
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
            return (record["d"],
                    tuple(-x for x in record["cycle_key"]),
                    record["rank"])

        for trial in range(96 + 8 * min(cohort, 2)):
            record = inspect(fresh(profile), side)
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
        for step in range(28):
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
                rng.shuffle(choices)
                witnesses = sorted(
                    set(current["words"]),
                    key=lambda w: (w.bit_count(), w))[:4]
                old_block = monomials[side][G.idx(old)]
                best = None

                # Compare actual label changes, not whole-cell gauges.
                # The syndrome of each retained word under a one-term
                # replacement is the exact old/new monomial difference.
                for replacement in choices[:7]:
                    proposed_cell = list(grid[j][0])
                    proposed_cell[t] = replacement
                    if not whole_sidon(proposed_cell):
                        continue
                    new_block = monomials[side][G.idx(replacement)]
                    boundaries = []
                    for word in witnesses:
                        local_word = (word >> (j * q)) & mask
                        changed_checks = sum(
                            ((local_word & (a ^ b)).bit_count() & 1)
                            for a, b in zip(old_block, new_block))
                        boundaries.append(changed_checks)
                    disruption = (
                        min(boundaries, default=0),
                        sum(boundaries),
                        sum(value > 0 for value in boundaries),
                    )
                    if best is None or disruption > best[0]:
                        best = (disruption, proposed_cell)
                if best is None:
                    continue
                grid[j][0] = best[1]
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
                "orbit_anchor_weights": tuple(
                    rec["orbit_anchor"].bit_count()
                    if rec["orbit_anchor"] is not None else None
                    for rec in (a, b)),
                "orbit_cancellation_improvements": (
                    a["orbit_cancellations"], b["orbit_cancellations"]),
                "repair_model": "verified witness-syndrome spread",
            }
    raise ValueError("no orbit-challenged tall-protograph pair passed full gates")


def NEW_CTX(rng, G, idx):
    """Twenty balanced-support cards with binding joint logical challenges."""
    q = G.q
    assert q == 111 and A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    topology, policy = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0x37251234 + 49979687 * (idx + 1)))

    # Each B topology has different concentration/zero-cell structure,
    # not merely a row/column permutation of the first topology.
    b_supports = (
        ((2, 2, 2, 1, 1), (2, 1, 1, 2, 2)),
        ((3, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((4, 1, 1, 1, 1), (0, 2, 2, 2, 2)),
        ((2, 3, 1, 1, 1), (2, 0, 2, 2, 2)),
        ((3, 3, 1, 1, 0), (1, 0, 2, 2, 3)),
    )
    supports = {
        "A": [[4, 4]],
        "B": [list(row) for row in b_supports[topology]],
    }
    assert [sum(row) for row in supports["B"]] == [8, 8]
    assert [sum(row[j] for row in supports["B"])
            for j in range(5)] == [4, 3, 3, 3, 3]
    assert _grid_weights(supports["A"], supports["B"]) == 12 <= W_MAX
    assert _no_dead_lines(supports["A"], supports["B"])

    a_profiles = (
        ((2, 1, 1), (2, 1, 1)),
        ((2, 1, 1), (3, 1, 0)),
        ((2, 1, 1), (2, 2, 0)),
        ((3, 1, 0), (3, 1, 0)),
    )
    quartics = ((2, 1, 1), (3, 1, 0), (2, 1, 1), (2, 2, 0))
    triangle = (3, 5, 7, 11, 13, 15)[(policy + 2 * cohort) % 6]

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    identity = G.idx(G.e)

    class Probe:
        # Native operation cache only; all binary action blocks use
        # lifted_matrices with the evaluator's conventions.
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
    zero = [[[]]]
    for start in range(0, q, 16):
        ids = list(range(start, min(start + 16, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(P, gallery, zero)
        _, zz, _ = lifted_matrices(P, zero, gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([(row >> (j * q)) & mask for row in xx])
            blocks["B"].append([(row >> (j * q)) & mask for row in zz])
    assert all(len(value) == q for value in blocks.values())

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def sidon(cell):
        if len(cell) != len(set(cell)):
            return False
        left = {table[inverse[a]][b]
                for a in cell for b in cell if a != b}
        right = {table[a][inverse[b]]
                 for a in cell for b in cell if a != b}
        target = len(cell) * (len(cell) - 1)
        return len(left) == len(right) == target

    def draw(side, i, j):
        count = supports[side][i][j]
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            # Avoid the short order-three ratio of complement binomials.
            b = rng.randrange(3)
            a = rng.randrange(37)
            d = rng.randrange(1, 37)
            return [G.idx((a, b)), G.idx(((a + d) % 37, b))]
        if count == 4:
            occupancy = (a_profiles[policy][j] if side == "A"
                         else quartics[(policy + cohort) % 4])
        else:
            occupancy = ((1, 1, 1) if (policy + cohort) % 2 == 0
                         else (2, 1, 0))
        for _ in range(512):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, size in zip(fibers, occupancy):
                if size == 3:
                    origin = rng.randrange(37)
                    scale = rng.randrange(1, 37)
                    aa = [origin, (origin + scale) % 37,
                          (origin + triangle * scale) % 37]
                else:
                    aa = rng.sample(range(37), size)
                cell.extend(G.idx((a, b)) for a in aa)
            if len(cell) == count and sidon(cell):
                return cell
        raise ValueError("balanced-card Sidon filling exhausted")

    def fresh(side):
        return [[draw(side, i, j) for j in range(len(row))]
                for i, row in enumerate(supports[side])]

    # Restrictions are actual coordinate masks, not separately lifted
    # quotient matrices. Their syndromes are sums of VERIFIED columns.
    restrictions = {}
    subgroup = [G.idx((0, b)) for b in range(3)]
    for side, shape in (("A", A_SHAPE), ("B", B_SHAPE)):
        remaining, orbits = set(range(q)), []
        while remaining:
            root = min(remaining)
            orbit = {table[root][h] if side == "A" else table[h][root]
                     for h in subgroup}
            assert len(orbit) == 3 and orbit <= remaining
            remaining.difference_update(orbit)
            orbits.append(sorted(orbit))
        restrictions[side] = [
            sum(1 << (j * q + c) for c in orbit)
            for j in range(shape[1]) for orbit in orbits
        ]

    def inspect(grid, side):
        nr, nc = len(grid), len(grid[0])
        width = nc * q
        rows = [0] * (nr * q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i * q + r] ^= value << (j * q)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        rank = gf2_rank(rows)
        if rank < nr * q - (3 if side == "A" else 6):
            return None
        fixed_columns = [syndrome(columns, part)
                         for part in restrictions[side]]
        signature = tuple(tuple(sorted(cell))
                          for row in grid for cell in row)
        return dict(
            grid=copy_grid(grid), rows=rows, columns=columns,
            fixed_columns=fixed_columns, side=side, rank=rank,
            cycles=cycles, signature=signature,
            d=width + 1, words=[], fixed_words=[])

    def search(columns, diagnostic_seed, trials, seeds=()):
        """Persistent genuine kernel words; misses certify no exclusion."""
        diagnostic = random.Random(diagnostic_seed)
        width = len(columns)
        order = list(range(width))
        retained = sorted({w for w in seeds if w},
                          key=lambda w: (w.bit_count(), w))[:24]
        best = None

        def observe(word, grown):
            nonlocal best
            if not word:
                return
            weight = word.bit_count()
            if best is None or weight < best.bit_count():
                best = word
            if weight <= best.bit_count() + 6:
                grown.append(word)
                if len(grown) > 192:
                    grown[:] = sorted(set(grown),
                                      key=lambda w: (w.bit_count(), w))[:32]

        for word in retained:
            assert syndrome(columns, word) == 0
            observe(word, [])

        for turn in range(trials):
            diagnostic.shuffle(order)
            if retained and turn % 4 == 3:
                anchor = diagnostic.choice(retained[:6])
                active = list(_bits(anchor))
                diagnostic.shuffle(active)
                preferred = set(active[:max(1, len(active) // 2)])
                order.sort(key=lambda c: c not in preferred)
            pivots, basis, independent = {}, [], []
            grown = list(retained)
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
                    observe(word, grown)
            if not basis and not retained:
                return []

            light = sorted(set(retained + basis),
                           key=lambda w: (w.bit_count(), w))[:14]
            moves = basis + retained
            for word in light:
                for other in moves:
                    observe(word ^ other, grown)

            if trials >= 16 and turn % 8 == 0 and independent:
                chosen = sorted(basis, key=int.bit_count)[:64]
                coordinates = diagnostic.sample(
                    independent, min(8, len(independent)))
                hash_mask = sum(1 << c for c in coordinates)
                buckets = {}
                for at, first in enumerate(chosen):
                    for second in chosen[:at]:
                        pair = first ^ second
                        bucket = buckets.setdefault(pair & hash_mask, [])
                        for previous in bucket:
                            observe(pair ^ previous, grown)
                        if len(bucket) < 8:
                            bucket.append(pair)

            diagnostic.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    observe(word, grown)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(96):
                    a, b, c = diagnostic.sample(basis, 3)
                    observe(a ^ b ^ c, grown)
            retained = sorted(
                set(retained + grown + ([best] if best else [])),
                key=lambda w: (w.bit_count(), w))[:24]
            if best is not None and best.bit_count() <= 4:
                break
        assert all(syndrome(columns, w) == 0 for w in retained)
        return retained

    def query(rec, full_trials, fixed_trials, offset):
        words = search(rec["columns"], seed + offset,
                       full_trials, rec["words"])
        fixed = search(rec["fixed_columns"], seed + offset + 5000011,
                       fixed_trials, rec["fixed_words"])
        rec["fixed_words"] = fixed
        expanded = []
        for word in fixed:
            actual = 0
            for c in _bits(word):
                actual ^= restrictions[rec["side"]][c]
            assert actual.bit_count() == 3 * word.bit_count()
            assert syndrome(rec["columns"], actual) == 0
            expanded.append(actual)
        rec["words"] = sorted(
            set(rec["words"] + words + expanded),
            key=lambda w: (w.bit_count(), w))[:24]
        if not rec["words"]:
            raise ValueError("balanced component yielded no kernel witness")
        assert all(all((row & w).bit_count() % 2 == 0
                       for row in rec["rows"]) for w in rec["words"])
        rec["d"] = min(rec["d"], rec["words"][0].bit_count())
        rec["fixed_d"] = min(
            rec.get("fixed_d", len(rec["columns"]) + 1),
            min(map(int.bit_count, expanded),
                default=len(rec["columns"]) + 1))

    def merit(rec):
        # Caps limit optimistic finite-search selection pressure.
        four, six = rec["cycles"]
        secondary = (four, six) if policy % 2 == 0 else (
            100 * four + six, four)
        return (min(36, rec["d"]), rec["d"],
                tuple(-v for v in secondary), rec["rank"])

    def pool(side, restart):
        offset = 100003 * int(side == "B") + 300007 * restart
        seen, records = set(), []
        for _ in range(64 + 4 * min(cohort, 2)):
            rec = inspect(fresh(side), side)
            if rec is None or rec["signature"] in seen:
                continue
            seen.add(rec["signature"])
            records.append(rec)
        records.sort(key=lambda rec: rec["cycles"])
        selected = []
        for rec in records:
            if not _no_kernel4(rec["columns"]):
                continue
            query(rec, 6, 6, offset)
            selected.append(rec)
            if len(selected) == 8:
                break
        if not selected:
            return []

        leaders = sorted(selected, key=merit, reverse=True)[:2]
        current = leaders[0]
        for step in range(18):
            if step == 9 and len(leaders) > 1:
                current = leaders[1]
            word = rng.choice(current["words"][:6])
            occupied = {c // q for c in _bits(word)}
            mutable = [(i, j) for i, row in enumerate(supports[side])
                       for j, count in enumerate(row) if count]
            targeted = [place for place in mutable if place[1] in occupied]
            i, j = rng.choice(targeted if targeted else mutable)
            grid = copy_grid(current["grid"])
            grid[i][j] = draw(side, i, j)
            if step % 6 == 5:
                other = rng.choice(mutable)
                grid[other[0]][other[1]] = draw(side, *other)
            proposed = inspect(grid, side)
            if proposed is None or proposed["signature"] in seen:
                continue
            seen.add(proposed["signature"])
            if not _no_kernel4(proposed["columns"]):
                continue
            proposed["words"] = [
                w for w in current["words"]
                if syndrome(proposed["columns"], w) == 0]
            query(proposed, 6, 6, offset + 700001 + 7919 * step)
            selected.append(proposed)
            if merit(proposed) > merit(current):
                current = proposed

        finalists = sorted(selected, key=merit, reverse=True)[:3]
        for rec in finalists:
            query(rec, 20, 20, offset + 1700021)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for rec in finalists:
            # Equal independent confirmation for every retained contender.
            query(rec, 64, 48, offset + 2700011)
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

    def logical_challenge(a, b, hx, hz, n):
        """Actual logical upper readings in the ORIGINAL stabilizer quotient."""
        readings = []
        sector_one = 10 * q
        for kind, rec, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            located = []
            for word in rec["words"][:3]:
                for strip in range(5 if kind == "Z" else 2):
                    actual = (
                        sum(((word >> (j * q)) & mask)
                            << ((5 * j + strip) * q) for j in range(2))
                        if kind == "Z" else word << (5 * strip * q))
                    assert actual >> n == 0
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        located.append((actual, strip))
            located.sort(key=lambda item: (
                item[0].bit_count(), item[1], item[0]))
            starts, covered = [], set()
            for actual, strip in located:
                if strip not in covered:
                    starts.append(actual)
                    covered.add(strip)
            reading = n + 1
            local = random.Random(
                seed + (4100011 if kind == "Z" else 5100011))
            reducers = list(stabilizers)
            local.shuffle(reducers)
            for initial in starts:
                best = initial
                ceiling = initial.bit_count() + 4
                frontier, visited = [initial], {initial}
                for _ in range(3):
                    layer = []
                    for actual in frontier:
                        nearby = sorted(
                            stabilizers,
                            key=lambda row: (row & actual).bit_count(),
                            reverse=True)[:32]
                        for row in nearby:
                            candidate = actual ^ row
                            if candidate in visited:
                                continue
                            visited.add(candidate)
                            weight = candidate.bit_count()
                            if weight > ceiling:
                                continue
                            if weight < best.bit_count():
                                best = candidate
                            layer.append((weight, local.random(), candidate))
                    layer.sort()
                    frontier, routes = [], set()
                    for weight, tie, candidate in layer:
                        route = candidate >> sector_one
                        if route not in routes:
                            routes.add(route)
                            frontier.append(candidate)
                            if len(frontier) == 6:
                                break
                    if not frontier:
                        break
                actual = best
                for _ in range(3):
                    before = actual.bit_count()
                    for row in reducers:
                        candidate = actual ^ row
                        if candidate.bit_count() < actual.bit_count():
                            actual = candidate
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:24]
                    winner = actual
                    for at, first in enumerate(nearby):
                        partial = actual ^ first
                        for second in nearby[:at]:
                            candidate = partial ^ second
                            if candidate.bit_count() < winner.bit_count():
                                winner = candidate
                    actual = winner
                    if actual.bit_count() == before:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            # n+1 denotes a missed embedding, never an exclusion.
            readings.append(reading)
        return tuple(readings)

    for restart in range(3):
        aa, bb = pool("A", restart), pool("B", restart)
        best = None
        for a in aa:
            for b in bb:
                hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 1332 and k >= 333 and
                        225 * n <= 1000 * k < 275 * n):
                    continue
                actual_a, actual_b = _verified_components(hx, hz, q)
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logicals = logical_challenge(a, b, hx, hz, n)
                detected = min(a["d"], b["d"], *logicals)
                quality = (
                    k * min(32, detected) ** 2, min(48, detected),
                    -a["cycles"][0] - b["cycles"][0],
                    -a["cycles"][1] - b["cycles"][1],
                    min(48, a["d"]) + min(48, b["d"]))
                if best is None or quality > best[0]:
                    best = (quality, a, b, hx, hz, k, logicals)
        if best is None:
            continue

        quality, a, b, cached_hx, cached_hz, k, logicals = best
        A = [[[G.elements[g] for g in cell] for cell in row]
             for row in a["grid"]]
        B = [[[G.elements[g] for g in cell] for cell in row]
             for row in b["grid"]]
        assert [[len(cell) for cell in row] for row in A] == supports["A"]
        assert [[len(cell) for cell in row] for row in B] == supports["B"]
        assert all(len(set(cell)) == len(cell)
                   for grid in (A, B) for row in grid for cell in row)
        hx, hz, n = lifted_matrices(G, A, B)
        assert hx == cached_hx and hz == cached_hz and n == 1332
        assert k == n - gf2_rank(hx) - gf2_rank(hz)
        assert max(row.bit_count() for row in hx + hz) <= 12
        assert css_ok(hx, hz)
        return dict(
            A=A, B=B, idx=idx, k=k, quality=quality,
            card=(topology, policy, cohort, triangle),
            support_counts=supports,
            component_ranks=(a["rank"], b["rank"]),
            component_readings=(a["d"], b["d"]),
            C3_restricted_readings=(a["fixed_d"], b["fixed_d"]),
            component_cycles=(a["cycles"], b["cycles"]),
            verified_sector_one_logical_readings=logicals,
            target_cell=0.250)
    raise ValueError("balanced component pairings missed cell 0.250")


def CELL(rng, G, side, i, j, ctx):
    """Return the entire selected cell, including deliberate zero blocks."""
    cell = list(ctx[side][i][j])
    assert len(cell) == ctx["support_counts"][side][i][j]
    assert 0 <= len(cell) <= 5
    assert len(cell) == len({G._key(g) for g in cell})
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Whole-cell Sidon supports plus transpose-kernel repair uniformly exclude sub-50 logical readings: current slots 21,43,54,58 screened at 30,39,45,36 despite four terms per cell and w=12.
# REFUTED: Verified C3-fixed screening before unrestricted repair improves the 965.8564 tall-bed record: the shown fixed-space program scored 932.7676 with selected readings 65..71.
# REFUTED: Full-coset gap cards with expansion-guided repair improve the 965.8564 tall-bed record: the shown expansion program scored 937.9176.
# REFUTED: Orbit-expanded component cancellation and syndrome-spread exponent repair raise five-code fitness above 965.8564 on the tall Z97:Z3 bed; the current program instead scored 922.3868, with measured readings 64..71 and a screened weight-21 outlier.
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
