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
CONCEPT = ("Hypothesis: on metacyclic Z97:Z3(s=35), order 291, A=B=(1,2), bidirectionally difference-disjoint four-term components challenged by C3-fixed kernels and stabilizer-reduced nontrivial logical witnesses raise five-code fitness above 965.8564 in cell 0.200. Orientation alone gives no claimed improvement; all sampled readings remain provisional.")
GROUP = ("Keep the exact split-metacyclic group, parameters and order: m=97,r=3,s=35,q=291. The wide and tall beds reached fitness 917.6920 and 965.8564. Factor five preserves n=1455 and the large order-97 normal subgroup; the order-three complement supplies a smaller restricted witness space. These measured beds motivate selection experiments, not certified distance transfer.")
SHAPES = ("Change both sides from (2,1) to (1,2). Equal components have 582 variables and 291 checks; n=1455 and k>=291 guarantee rate 0.200, with q=291 below q_max=299. Every reported parent measured minimum occupied sector 2, using one Arow across both Brows or one Brow across both Arows. The analogous families move to sector-1 Acol/Bcol strips. New constrained labels, not archived transposes, are generated; exact full ranks retain [0.175,0.225).")
FILLING = ("Four distinct terms per cell retain both check weights at twelve. Twenty cards cross five occupancy-profile pairings with four different A/B coupled normal-gap prescriptions; each side retains an all-three-coset cell. Require all 24 directed differences across BOTH cells to be distinct in both orientations. Screen authoritative wide components against unrestricted and expanded C3-fixed kernel words, retaining witnesses across bases and repairing implicated whole cells without erasing the card. Independently deepen finalists. Exact stabilizer-span membership identifies nontrivial localized quantum witnesses, and original-check single/pair/triple reductions challenge pairings. Native full ranks, kernel-four exclusion, weights and CSS gate submission; no sampled reading certifies distance.")
HOW_AB = FILLING
FAMILY_NAME = "r020_m97r3s35_12x12_gapcards_C3_quantum_retest_v5"

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


def _legacy_tall_ctx(rng, G, idx):
    """Unused historical tall optimizer; NEW_CTX below replaces it."""
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
    triangle = (3, 5, 7, 9, 11, 13, 15, 17)[
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
        stats = stats + (eight_cycles(rows, columns),)
        return {
            "grid": grid, "rows": rows, "columns": columns,
            "rank": rank, "cell_ranks": ranks, "cycles": stats,
            "cycle_key": cycle_key(stats), "signature": signature(grid),
            "d": width + 1,
        }

    def syndrome(columns, word):
        result = 0
        for c in _bits(word):
            result ^= columns[c]
        return result

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

            light = sorted(basis, key=int.bit_count)[:20]
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
            d, word = kernel_probe(
                record, seed + offset + 700001 * turn, trials)
            record["d"] = min(record["d"], d)
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
            records.append(record)
        records.sort(key=lambda record: record["cycle_key"])

        selected = []
        for record in records:
            if not _no_kernel4(record["columns"]):
                continue
            probe(record, 12, 0)
            selected.append(record)
            if len(selected) >= 6:
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
                "component_cycles": (a["cycles"], b["cycles"]),
                "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
            }
    raise ValueError("no repaired tall-protograph pair passed full gates")


def _r020_span(rows):
    """Exact GF2 span, in the original qubit coordinates."""
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


def _r020_nontrivial(word, pivots):
    """True exactly when word is outside this stabilizer span."""
    while word:
        lead = word.bit_length() - 1
        old = pivots.get(lead)
        if old is None:
            return True
        word ^= old
    return False


def NEW_CTX(rng, G, idx):
    """New wide cards, retained C3 witnesses and original-stabilizer tests."""
    q, width = G.q, 2 * G.q
    assert q == 291 and A_SHAPE == B_SHAPE == (1, 2)
    support = [[4, 4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)
    mask = (1 << q) - 1
    card, cohort = idx % 20, idx // 20
    topology, trade = card // 4, card % 4
    seed = rng.randrange(1 << 62)
    draw_rng = random.Random(seed + 32452843 + 104729 * (idx + 1))

    # Every component retains at least one all-three-coset cell.
    # Five different profile pairings, rather than a single design orbit.
    profiles = (
        ((2, 1, 1), (2, 1, 1)),
        ((2, 1, 1), (2, 2, 0)),
        ((2, 1, 1), (3, 1, 0)),
    )
    plans = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2))
    pa, pb = plans[topology]

    # Classes modulo <-1,35> distinguish normal gaps up to the
    # complement action, sign changes and common rotation scaling.
    primitive = next(
        a for a in range(2, 97)
        if pow(a, 48, 97) != 1 and pow(a, 32, 97) != 1)
    ea = 1 + trade + 4 * (cohort % 3)
    eb = (5 + 2 * trade + 3 * (cohort % 3)) % 16
    if eb == 0:
        eb = 13
    ratios = {"A": pow(primitive, ea, 97),
              "B": pow(primitive, eb, 97)}

    def affine_class(t):
        it = pow(t, 95, 97)
        im = pow((t - 1) % 97, 95, 97)
        return tuple(sorted({
            t, (1 - t) % 97, it, (-im) % 97,
            (t * im) % 97, ((t - 1) * it) % 97,
        }))

    old_classes = {
        affine_class(t) for t in (3, 5, 7, 9, 11, 13, 15, 17)
    }
    triangle_classes = sorted({
        affine_class(t) for t in range(2, 97)
        if len(affine_class(t)) == 6
        and affine_class(t) not in old_classes
    })
    assert triangle_classes

    # Cache ONLY native group operations. All component matrices are
    # returned by the authoritative scaffold constructor.
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

    P = Probe()

    def difference_free(grid):
        """All directed differences across both cells, both orientations."""
        left, right = set(), set()
        for cell in grid[0]:
            if len(cell) != 4 or len(set(cell)) != 4:
                return False
            for a in cell:
                for b in cell:
                    if a == b:
                        continue
                    l = table[inverses[a]][b]
                    r = table[a][inverses[b]]
                    if l in left or r in right:
                        return False
                    left.add(l)
                    right.add(r)
        return len(left) == len(right) == 24

    def fresh(profile, side, delta=None):
        """Preserve occupancy, coupled gaps and the card's triangle class."""
        ratio = ratios[side]
        triangles = triangle_classes[
            (trade + 2 * cohort + int(side == "B")) %
            len(triangle_classes)]
        for _ in range(4096):
            base = draw_rng.randrange(1, 97) if delta is None else delta
            cells = []
            for j, occupancy in enumerate(profiles[profile]):
                fibers = list(range(3))
                draw_rng.shuffle(fibers)
                cell, core = [], True
                for b, count in zip(fibers, occupancy):
                    if not count:
                        continue
                    origin = draw_rng.randrange(97)
                    if count == 1:
                        aa = [origin]
                    else:
                        if core:
                            gap = base * (1 if j == 0 else ratio) % 97
                            gap = gap * draw_rng.choice((1, 35, 61)) % 97
                            if draw_rng.randrange(2):
                                gap = (-gap) % 97
                        else:
                            gap = draw_rng.randrange(1, 97)
                        aa = [origin, (origin + gap) % 97]
                        if count == 3:
                            t = draw_rng.choice(triangles)
                            aa.append((origin + t * gap) % 97)
                        core = False
                    cell.extend(G.idx((a, b)) for a in aa)
                assert len(cell) == len(set(cell)) == 4
                cells.append(cell)
            grid = [cells]
            if difference_free(grid):
                return grid, base
        raise ValueError("new bidirectional coupled-gap card could not fill")

    # Restriction coordinates are genuine disjoint variable orbits.
    # Their syndromes are XORs of AUTHORITATIVE component columns.
    h = G.idx((0, 1))
    assert table[table[h][h]][h] == identity
    partitions = {}
    for side in ("A", "B"):
        remaining = set(range(q))
        base_orbits = []
        while remaining:
            root = min(remaining)
            orbit = [root]
            for _ in range(2):
                value = orbit[-1]
                orbit.append(table[value][h] if side == "A"
                             else table[h][value])
            end = (table[orbit[-1]][h] if side == "A"
                   else table[h][orbit[-1]])
            assert end == root and len(set(orbit)) == 3
            assert set(orbit) <= remaining
            remaining.difference_update(orbit)
            base_orbits.append(sum(1 << c for c in orbit))
        assert len(base_orbits) == 97
        partitions[side] = [
            word << (j * q)
            for j in range(2) for word in base_orbits
        ]

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def expand(word, groups):
        actual = 0
        for c in _bits(word):
            actual |= groups[c]
        return actual

    def signature(grid):
        return tuple(tuple(sorted(cell)) for cell in grid[0])

    def inspect(grid, side, delta):
        # Dead dummy lines are screening probes, never submitted cells.
        if side == "A":
            rows, _, _ = lifted_matrices(P, grid, [[[]]])
        else:
            _, rows, _ = lifted_matrices(P, [[[]]], grid)
        assert len(rows) == q
        assert all(row >> width == 0 for row in rows)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        if cycles[0]:
            return None
        rank = gf2_rank(rows)
        cell_ranks = (
            gf2_rank([row & mask for row in rows]),
            gf2_rank([row >> q for row in rows]),
        )
        if min((rank,) + cell_ranks) < q - 3:
            return None
        groups = partitions[side]
        fixed = [syndrome(columns, group) for group in groups]
        if not all(fixed) or len(set(fixed)) != len(fixed):
            return None
        return dict(
            grid=grid, delta=delta, rows=rows, columns=columns,
            cycles=cycles, rank=rank, cell_ranks=cell_ranks,
            signature=signature(grid), groups=groups, fixed=fixed,
            d=width + 1, fixed_d=width + 1,
            words=[], fixed_words=[],
        )

    def search(columns, diagnostic, trials, seeds=(), stop=0):
        """Retained cross-basis null-word search; never certify absence."""
        m = len(columns)
        beam = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:24]
        best, witness = m + 1, None
        for word in beam:
            assert word and syndrome(columns, word) == 0
            if word.bit_count() < best:
                best, witness = word.bit_count(), word
        order = list(range(m))

        for turn in range(trials):
            diagnostic.shuffle(order)
            if beam and turn % 4 == 3:
                anchor = diagnostic.choice(beam[:8])
                preferred = list(_bits(anchor))
                diagnostic.shuffle(preferred)
                chosen = set(preferred[:max(1, len(preferred) // 2)])
                order.sort(key=lambda c: c not in chosen)

            pivots, basis = {}, []
            candidates = list(beam)

            def note(word):
                nonlocal best, witness
                if not word:
                    return
                weight = word.bit_count()
                if weight < best:
                    best, witness = weight, word
                if weight <= best + 6:
                    candidates.append(word)

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
                    note(word)

            if not basis:
                raise ValueError("verified component restriction has no kernel")
            light = sorted(set(beam + basis),
                           key=lambda w: (w.bit_count(), w))[:20]

            # At restricted length 194, examine every fundamental pair.
            # At unrestricted length 582, use light-versus-all pairs.
            if m <= 250:
                for i, word in enumerate(basis):
                    for other in basis[:i]:
                        note(word ^ other)
                    if len(candidates) > 256:
                        candidates = sorted(
                            set(candidates), key=int.bit_count)[:48]
            else:
                for word in light:
                    for other in basis + beam:
                        note(word ^ other)
                    if len(candidates) > 256:
                        candidates = sorted(
                            set(candidates), key=int.bit_count)[:48]

            shortened = sorted(set(candidates + light),
                               key=lambda w: (w.bit_count(), w))[:24]
            for i, word in enumerate(shortened):
                for other in shortened[:i]:
                    note(word ^ other)

            moves = basis + beam
            diagnostic.shuffle(moves)
            for initial in shortened[:6]:
                word, weight = initial, initial.bit_count()
                for _ in range(3):
                    changed = False
                    for other in moves:
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

            if witness is not None:
                candidates.append(witness)
            beam = sorted(set(candidates),
                          key=lambda w: (w.bit_count(), w))[:24]
            if best <= stop:
                break

        if witness is None:
            raise ValueError("no genuine null-word reading")
        assert syndrome(columns, witness) == 0
        return best, witness, beam

    def optimize(profile, side):
        offset = 0 if side == "A" else 100003
        bank, seen = set(), set()

        def query(rec, full_trials, fixed_trials, turn):
            fd, fw, fixed_words = search(
                rec["fixed"],
                random.Random(seed + offset + 1700003 + 900001 * turn),
                fixed_trials, rec["fixed_words"], stop=8)
            rec["fixed_words"] = fixed_words
            actuals = [expand(w, rec["groups"]) for w in fixed_words[:8]]
            for original, actual in zip(fixed_words[:8], actuals):
                assert actual.bit_count() == 3 * original.bit_count()
                assert syndrome(rec["columns"], actual) == 0
            rec["fixed_d"] = min(rec["fixed_d"], 3 * fd)
            rec["d"] = min(rec["d"], 3 * fd)
            transferred = [
                word for word in bank
                if syndrome(rec["columns"], word) == 0
            ]
            words = sorted(set(rec["words"] + actuals + transferred),
                           key=int.bit_count)[:24]
            if full_trials:
                d, word, words = search(
                    rec["columns"],
                    random.Random(seed + offset + 700001 * turn),
                    full_trials, words, stop=24)
                rec["d"] = min(rec["d"], d)
                bank.add(word)
            rec["words"] = words
            bank.update(actuals)
            if len(bank) > 64:
                keep = sorted(bank, key=int.bit_count)[:48]
                bank.clear()
                bank.update(keep)
            assert all(syndrome(rec["columns"], w) == 0
                       for w in rec["words"][:3])

        def merit(rec):
            for word in bank:
                if (word.bit_count() < rec["d"]
                        and syndrome(rec["columns"], word) == 0):
                    rec["d"] = word.bit_count()
                    rec["words"] = sorted(
                        set(rec["words"] + [word]),
                        key=int.bit_count)[:24]
            return (rec["d"], rec["fixed_d"],
                    -rec["cycles"][1], rec["rank"])

        for restart in range(3):
            records = []
            for _ in range(48 + 4 * min(cohort, 2)):
                grid, delta = fresh(profile, side)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                rec = inspect(grid, side, delta)
                if rec is None:
                    continue
                # Challenge the restricted space BEFORE cycle shortlisting.
                query(rec, 0, 4, 1 + restart)
                records.append(rec)
            records.sort(key=lambda r: (-r["fixed_d"], r["cycles"]))

            selected = []
            for rec in records:
                if not _no_kernel4(rec["columns"]):
                    continue
                query(rec, 8, 24, 11 + restart)
                selected.append(rec)
                if len(selected) >= 6:
                    break
            if not selected:
                continue
            current = max(selected, key=merit)

            # Replace a whole implicated cell while retaining its gap card.
            for step in range(16):
                active = [w for w in current["words"]
                          if syndrome(current["columns"], w) == 0]
                word = min(active, key=int.bit_count) if active else None
                used = list(_bits(word)) if word is not None else []
                j = (draw_rng.choice(used) // q
                     if used else draw_rng.randrange(2))
                replacement, delta = fresh(
                    profile, side, current["delta"])
                grid = [[list(cell) for cell in current["grid"][0]]]
                if step % 5:
                    grid[0][j] = list(replacement[0][j])
                else:
                    grid = replacement
                if not difference_free(grid):
                    continue
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                proposed = inspect(grid, side, delta)
                if proposed is None:
                    continue
                if proposed["cycles"][1] > current["cycles"][1] + 8 * q:
                    continue
                if not _no_kernel4(proposed["columns"]):
                    continue
                query(proposed, 4, 16, 31 + step)
                selected.append(proposed)
                pk, ck = merit(proposed), merit(current)
                if pk > ck or (pk[0] == ck[0]
                               and draw_rng.random() < 0.10):
                    current = proposed

            unique = {r["signature"]: r for r in selected}
            finalists = sorted(unique.values(), key=merit,
                               reverse=True)[:3]
            for rec in finalists:
                query(rec, 32, 96, 101)
            finalists.sort(key=merit, reverse=True)
            finalists = finalists[:2]
            for rec in finalists:
                # Independent deeper orders retain all previous witnesses.
                query(rec, 96, 256, 211)
            finalists.sort(key=merit, reverse=True)
            return finalists
        raise ValueError("no new wide component survived witness challenges")

    aa, bb = optimize(pa, "A"), optimize(pb, "B")

    def localized(hx, hz, a, b, n):
        """Nontrivial quantum witnesses reduced by ORIGINAL stabilizers."""
        readings = []
        for kind, opposite, stabilizers, rec in (
                ("Z", hx, hz, a), ("X", hz, hx, b)):
            span = _r020_span(stabilizers)
            diagnostic = random.Random(
                seed + (3100009 if kind == "Z" else 4100011))
            candidates = []
            for word in rec["words"][:6]:
                for block in range(2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j * q)) & mask)
                            << ((2 * j + block) * q)
                            for j in range(2))
                    else:
                        actual = word << (2 * block * q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if _r020_nontrivial(actual, span):
                        candidates.append(actual)

            found = n + 1
            for actual in sorted(set(candidates),
                                 key=int.bit_count)[:8]:
                reducers = list(stabilizers)
                diagnostic.shuffle(reducers)
                for _ in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True

                    near = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:48]
                    best = actual
                    for i, first in enumerate(near):
                        for second in near[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < best.bit_count():
                                best = proposed
                    if best != actual:
                        actual, changed = best, True
                    for _ in range(160):
                        first, second, third = diagnostic.sample(near, 3)
                        proposed = actual ^ first ^ second ^ third
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break

                # Adding stabilizers preserves the original logical class.
                assert actual and _r020_nontrivial(actual, span)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                found = min(found, actual.bit_count())
            readings.append(found)
        return tuple(readings)

    proposals = []
    for a in aa:
        for b in bb:
            A = [[[G.elements[g] for g in cell] for cell in row]
                 for row in a["grid"]]
            B = [[[G.elements[g] for g in cell] for cell in row]
                 for row in b["grid"]]
            hx, hz, n = lifted_matrices(P, a["grid"], b["grid"])
            assert n == 5 * q == 1455
            ar, br = _verified_components(hx, hz, q)
            assert ar == a["rows"] and br == b["rows"]
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k >= q and 7 * n <= 40 * k < 9 * n):
                continue
            quantum = localized(hx, hz, a, b, n)
            # Component readings remain heuristics; quantum entries here
            # are weights of explicitly verified nontrivial logicals.
            detected = min(a["d"], b["d"], *quantum)
            key = (k * detected * detected, detected,
                   min(a["fixed_d"], b["fixed_d"]),
                   a["d"] + b["d"],
                   -a["cycles"][1] - b["cycles"][1])
            proposals.append((key, A, B, hx, hz, k, a, b, quantum))
    proposals.sort(key=lambda item: item[0], reverse=True)

    for key, A, B, cached_hx, cached_hz, k, a, b, quantum in proposals:
        if not _no_kernel4(_syndrome_columns(cached_hx, 1455)):
            continue
        if not _no_kernel4(_syndrome_columns(cached_hz, 1455)):
            continue
        hx, hz, n = lifted_matrices(G, A, B)
        assert hx == cached_hx and hz == cached_hz and n == 1455
        assert max(row.bit_count() for row in hx + hz) == 12
        assert css_ok(hx, hz)
        assert [[len(cell) for cell in row] for row in A] == support
        assert [[len(cell) for cell in row] for row in B] == support
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "card": (pa, pb, ea, eb, trade, cohort),
            "component_readings": (a["d"], b["d"]),
            "C3_fixed_readings": (a["fixed_d"], b["fixed_d"]),
            "verified_logical_readings": quantum,
            "component_cycles": (a["cycles"], b["cycles"]),
            "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
            "quality": key,
        }
    raise ValueError("new wide pairings missed native full validity gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the entire selected wide cell without subsequent draws."""
    cell = list(ctx[side][i][j])
    assert len(cell) == len({G._key(g) for g in cell}) == 4
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Whole-cell Sidon supports and transpose-component witness repair uniformly exclude sub-50 logical readings: accepted parent slots 21,43,54,58 instead screened at 30,39,45,36, despite four terms per cell and w=12.
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
