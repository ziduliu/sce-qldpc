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
CONCEPT = ("On dicyclic Dic9, order 36, A=(2,5), B=(3,7), balanced degree-three/four components and binding witness-directed label selection should produce five measured readings >=16 and fitness above 73.352 in cell 0.300. Failure of that five-code target refutes this experiment; finite searches are not distance certificates.")
GROUP = ("Jump to Dic9=<x,y | x^18=1, y^2=x^9, y*x*y^-1=x^-1>, order 36. Its order proves nonisomorphism to the order-291 parent. The order-18 cyclic subgroup, fourth-order elements outside it and unique involution motivate avoiding short binomial collisions. This bed is untried; nearby cyclic/dihedral (2,5)x(3,7) beds scored 49.5/42.7, so no distance transfer is assumed.")
SHAPES = ("A=(2,5), B=(3,7): factor 41 and q=36 give n=1476, k>=432 and guaranteed rate 12/41=0.292683 in cell 0.300. The size window permits q<=36. Components have 180 variables/72 checks and 252 variables/108 checks, nominal rates 3/5 and 4/7. Parent minima occupied sector-two Arow/Brow strips; the new wide components are challenged through sector-one logical embeddings. Actual full ranks enforce [0.275,0.325).")
FILLING = ("Twenty cards cross five A support patterns with four exact B zero budgets. Both sides have row totals eight and column totals three/four, giving HX<=8+4=12 and HZ<=8+4=12. Distinct cells contain zero to four terms, without forced identity anchors. Prefer bidirectional Sidon differences and relative orders above three where feasible. Scaffold-derived components supply cycles, retained kernel words, implicated-column replacement and independent retests. Complete pairings test commuting nontrivial logicals against original stabilizers, then verify full ranks, kernel-four exclusion, CSS and weights. All sampled readings remain provisional.")
HOW_AB = FILLING
FAMILY_NAME = "r030_Dic9_25x37_balanced34_binding_logicals_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 36

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("Dic9 requires order 36")
    # (a,b)=x^a*y^b; x^18=1, y^2=x^9, y*x*y^-1=x^-1.
    els = [(a, b) for b in range(2) for a in range(18)]

    def red(g):
        return (int(g[0]) % 18, int(g[1]) % 2)

    def mul(g, h):
        a, b = red(g)
        c, d = red(h)
        return ((a + (c if b == 0 else -c) + 9*b*d) % 18,
                (b+d) % 2)

    def inv(g):
        a, b = red(g)
        return ((-a if b == 0 else a+9) % 18, b)

    def idx(g):
        a, b = red(g)
        return 18*b+a

    return dict(elements=els, ident=(0, 0),
                mul=mul, inv=inv, idx=idx, red=red)
"""
A0 = None                      # only used by GTYPE = "abelian"

A_SHAPE = (2, 5)               # (ra, ca)
B_SHAPE = (3, 7)               # (rb, cb)


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


def _retired_tall_ctx(rng, G, idx):
    """Inactive historical tall-protograph experiment; never submitted."""
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


def NEW_CTX(rng, G, idx):
    """Balanced Dic9 cards; every binary screen uses scaffold-derived blocks."""
    from functools import lru_cache

    q = G.q
    assert q == 36 and A_SHAPE == (2, 5) and B_SHAPE == (3, 7)
    card, cohort = idx % 20, idx // 20
    model, zero_budget = divmod(card, 4)
    seed = rng.randrange(1 << 62)
    mask = (1 << q) - 1
    ident = G.idx(G.e)

    # Five different support prescriptions, not row/column relabelings
    # manufactured from one labeling. Every row total is eight.
    a_profiles = (
        ((2, 2, 2, 1, 1), (2, 1, 1, 2, 2)),
        ((3, 2, 1, 1, 1), (1, 1, 2, 2, 2)),
        ((4, 1, 1, 1, 1), (0, 2, 2, 2, 2)),
        ((3, 0, 2, 2, 1), (1, 3, 1, 1, 2)),
        ((2, 3, 0, 2, 1), (2, 0, 3, 1, 2)),
    )
    sa = [list(row) for row in a_profiles[model]]
    assert [sum(row) for row in sa] == [8, 8]
    assert [sum(row[j] for row in sa) for j in range(5)] == [4, 3, 3, 3, 3]

    degrees = (4, 4, 4, 3, 3, 3, 3)
    options = []
    for degree in degrees:
        opts = [
            (a, b, degree-a-b)
            for a in range(4) for b in range(4)
            if 0 <= degree-a-b <= 3
            and sum(x > 0 for x in (a, b, degree-a-b)) >= 2
        ]
        rng.shuffle(opts)
        options.append(opts)

    @lru_cache(None)
    def feasible(j, a, b, c, z):
        if min(a, b, c, z) < 0:
            return False
        if j == 7:
            return (a, b, c, z) == (0, 0, 0, 0)
        if a+b+c != sum(degrees[j:]):
            return False
        return any(
            feasible(j+1, a-v[0], b-v[1], c-v[2],
                     z-sum(x == 0 for x in v))
            for v in options[j]
        )

    assert feasible(0, 8, 8, 8, zero_budget)
    state = (8, 8, 8, zero_budget)
    sb = [[], [], []]
    for j in range(7):
        choices = []
        for v in options[j]:
            nxt = tuple(state[t]-v[t] for t in range(3))
            nxt += (state[3]-sum(x == 0 for x in v),)
            if feasible(j+1, *nxt):
                choices.append((v, nxt))
        v, state = rng.choice(choices)
        for i in range(3):
            sb[i].append(v[i])
    assert [sum(row) for row in sb] == [8, 8, 8]
    assert sum(x == 0 for row in sb for x in row) == zero_budget
    assert _grid_weights(sa, sb) == 12 <= W_MAX
    assert _no_dead_lines(sa, sb)

    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverses = [G.idx(G.inv(g)) for g in G.elements]
    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q+1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid dicyclic element order")

    # These zero partners are screening probes, never submitted codes.
    # Subsequent assembly uses GF2 linearity of authoritative monomials.
    mono = [[[g] for g in G.elements]]
    left, _, _ = lifted_matrices(G, mono, [[[]]])
    _, right, _ = lifted_matrices(G, [[[]]], mono)
    blocks = {
        "A": [[(row >> (g*q)) & mask for row in left]
              for g in range(q)],
        "B": [[(row >> (g*q)) & mask for row in right]
              for g in range(q)],
    }
    del left, right

    def copy_grid(grid):
        return [[list(cell) for cell in row] for row in grid]

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def fresh_cell(count):
        if count == 0:
            return []
        if count == 1:
            return [rng.randrange(q)]
        best = None
        for _ in range(192):
            ids = rng.sample(range(q), count)
            lefts, rights, short = set(), set(), 0
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    l = table[inverses[a]][b]
                    r = table[a][inverses[b]]
                    lefts.add(l)
                    rights.add(r)
                    short += orders[l] <= 3
            target = count*(count-1)
            key = (2*target-len(lefts)-len(rights), short)
            if best is None or key < best[0]:
                best = key, ids
            if key == (0, 0):
                break
        # Difference/order preferences never silently void a legal card.
        return list(best[1])

    def fresh(support):
        return [[fresh_cell(count) for count in row] for row in support]

    def inspect(grid, side):
        nr, nc = len(grid), len(grid[0])
        width = nc*q
        rows = [0]*(nr*q)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for g in cell:
                    for t, value in enumerate(blocks[side][g]):
                        rows[i*q+t] ^= value << (j*q)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        return {
            "grid": copy_grid(grid), "rows": rows, "columns": columns,
            "width": width, "cycles": cycles,
            "d": width+1, "words": [],
        }

    def geometry(record):
        four, excess = record["cycles"]
        penalty = (320, 480, 720, 960)[zero_budget]
        return penalty*four + excess

    def probe(record, diagnostic_seed, trials):
        """Binding actual kernel words; no search miss is an exclusion."""
        local = random.Random(diagnostic_seed)
        columns, width = record["columns"], record["width"]
        retained = sorted(set(record["words"]), key=int.bit_count)[:24]
        reading, witness = width+1, None

        def note(word):
            nonlocal reading, witness
            if word and word.bit_count() < reading:
                reading, witness = word.bit_count(), word

        for word in retained:
            assert syndrome(columns, word) == 0
            note(word)
        order = list(range(width))
        for turn in range(trials):
            local.shuffle(order)
            if turn % 5 == 4 and retained:
                anchor = local.choice(retained[:6])
                preferred = list(_bits(anchor))
                local.shuffle(preferred)
                preferred = set(preferred[:max(1, len(preferred)//2)])
                order.sort(key=lambda c: c not in preferred)

            pivots, basis = {}, []
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
                    basis.append(word)
                    note(word)

            light = sorted(set(basis+retained), key=int.bit_count)[:18]
            grown = list(light)+retained
            moves = basis+retained
            for a in light:
                for b in moves:
                    word = a ^ b
                    note(word)
                    if word and word.bit_count() <= min(30, reading+6):
                        grown.append(word)
                if len(grown) > 192:
                    grown = sorted(set(grown), key=int.bit_count)[:48]

            local.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for _ in range(3):
                    changed = False
                    for other in moves:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < word.bit_count():
                            word, changed = proposed, True
                    note(word)
                    if not changed:
                        break
                grown.append(word)
            if witness is not None:
                grown.append(witness)
            retained = sorted(set(grown), key=int.bit_count)[:24]
            if reading <= 6:
                break

        assert witness is not None and syndrome(columns, witness) == 0
        assert all((row & witness).bit_count() % 2 == 0
                   for row in record["rows"])
        record["d"] = min(record["d"], reading)
        record["words"] = sorted(set(retained+[witness]),
                                 key=int.bit_count)[:24]

    def optimize(side, support):
        offset = 100003*int(side == "B")
        seen, records = set(), []

        def remember(grid):
            sig = tuple(tuple(sorted(cell)) for row in grid for cell in row)
            if sig in seen:
                return None
            seen.add(sig)
            record = inspect(grid, side)
            if record is not None:
                records.append(record)
            return record

        for _ in range(56+4*min(cohort, 2)):
            remember(fresh(support))
        records.sort(key=geometry)

        # Refine actual labels, not row/column permutations or group images.
        mutable = [(i, j) for i, row in enumerate(support)
                   for j, count in enumerate(row) if count]
        for initial in list(records[:4]):
            current = initial
            for move in range(40):
                grid = copy_grid(current["grid"])
                i, j = rng.choice(mutable)
                grid[i][j] = fresh_cell(support[i][j])
                proposed = remember(grid)
                if proposed is None:
                    continue
                if (geometry(proposed) <= geometry(current)
                        or (move < 24 and rng.random() < 0.08)):
                    current = proposed

        records.sort(key=geometry)
        selected = []
        for record in records:
            rank = gf2_rank(record["rows"])
            if rank < len(record["rows"])-6:
                continue
            if not _no_kernel4(record["columns"]):
                continue
            record["rank"] = rank
            probe(record, seed+offset+400009, 10)
            selected.append(record)
            if len(selected) >= 8:
                break
        if not selected:
            return []

        def merit(record):
            return (min(22, record["d"]), -geometry(record),
                    record["d"], record["rank"])

        current = max(selected, key=merit)
        for step in range(24):
            word = min(current["words"], key=int.bit_count)
            occupied = {c//q for c in _bits(word)}
            targeted = [p for p in mutable if p[1] in occupied]
            i, j = rng.choice(targeted if targeted else mutable)
            grid = copy_grid(current["grid"])
            grid[i][j] = fresh_cell(support[i][j])
            proposed = remember(grid)
            if proposed is None:
                continue
            if geometry(proposed) > geometry(current)+4*q*720:
                continue
            if not _no_kernel4(proposed["columns"]):
                continue
            rank = gf2_rank(proposed["rows"])
            if rank < len(proposed["rows"])-6:
                continue
            proposed["rank"] = rank
            proposed["words"] = [
                w for w in current["words"]
                if syndrome(proposed["columns"], w) == 0
            ]
            probe(proposed, seed+offset+700001+1009*step, 8)
            selected.append(proposed)
            if merit(proposed) >= merit(current):
                current = proposed
            elif (proposed["d"] == current["d"]
                  and rng.random() < 0.08):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists = selected[:3]
        for record in finalists:
            probe(record, seed+offset+1900003, 32)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            # Independent information sets retain every earlier witness.
            probe(record, seed+offset+4100009, 64+8*min(cohort, 2))
        finalists.sort(key=merit, reverse=True)
        return finalists

    def span(rows):
        pivots = {}
        for word in rows:
            while word:
                lead = word.bit_length()-1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = word
                    break
                word ^= old
        return pivots

    def outside(word, pivots):
        while word:
            old = pivots.get(word.bit_length()-1)
            if old is None:
                return True
            word ^= old
        return False

    def logical_reading(a, b, hx, hz, n):
        """Exact commutation/nontriviality; bounded reductions give upper readings."""
        reading = n+1
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            words = sorted(set(record["words"]), key=int.bit_count)[:4]
            fixed_count = 7 if kind == "Z" else 5
            for word in words:
                assert syndrome(record["columns"], word) == 0
                for fixed in range(fixed_count):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask) << ((7*j+fixed)*q)
                            for j in range(5))
                    else:
                        actual = word << (fixed*7*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)

            local = random.Random(
                seed+(6100009 if kind == "Z" else 7100011))
            for actual in sorted(candidates, key=int.bit_count)[:6]:
                reducers = list(stabilizers)
                local.shuffle(reducers)
                for _ in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    if not changed:
                        break
                near = sorted(
                    stabilizers, key=lambda r: (r & actual).bit_count(),
                    reverse=True)[:32]
                for _ in range(96):
                    first, second = local.sample(near, 2)
                    proposed = actual ^ first ^ second
                    if proposed.bit_count() < actual.bit_count():
                        actual = proposed
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
        return reading

    for restart in range(3):
        aa, bb = optimize("A", sa), optimize("B", sb)
        proposals = []
        for a in aa:
            for b in bb:
                A = [[[G.elements[g] for g in cell] for cell in row]
                     for row in a["grid"]]
                B = [[[G.elements[g] for g in cell] for cell in row]
                     for row in b["grid"]]
                hx, hz, n = lifted_matrices(G, A, B)
                assert n == 41*q == 1476
                k = n-gf2_rank(hx)-gf2_rank(hz)
                if not (k >= 432 and 275*n <= 1000*k < 325*n):
                    continue
                ar, br = _verified_components(hx, hz, q)
                assert ar == a["rows"] and br == b["rows"]
                logical = logical_reading(a, b, hx, hz, n)
                detected = min(18, a["d"], b["d"], logical)
                key = (k*detected*detected, detected,
                       -geometry(a)-geometry(b), -(k-432))
                proposals.append((key, A, B, hx, hz, k, a, b, logical))
        proposals.sort(key=lambda item: item[0], reverse=True)

        for key, A, B, hx, hz, k, a, b, logical in proposals:
            if not _no_kernel4(_syndrome_columns(hx, 1476)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, 1476)):
                continue
            assert [[len(c) for c in row] for row in A] == sa
            assert [[len(c) for c in row] for row in B] == sb
            assert all(len(cell) == len(set(cell))
                       for grid in (A, B) for row in grid for cell in row)
            assert _grid_weights(sa, sb) == 12 <= W_MAX
            assert _no_dead_lines(sa, sb)
            assert max(row.bit_count() for row in hx+hz) == 12
            assert css_ok(hx, hz)
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "card": (model, zero_budget, cohort),
                "component_ranks": (a["rank"], b["rank"]),
                "component_readings": (a["d"], b["d"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "verified_localized_logical_reading": logical,
                "pairing_merit": key, "target_cell": 0.300,
                "column_degrees": ((4, 3, 3, 3, 3), degrees),
            }
    raise ValueError("balanced Dic9 pairings missed verified full gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete balanced card, without post-screen randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Whole-cell Sidon supports and transpose-kernel repair uniformly exclude sub-50 readings on the order-291 tall bed; current slots 21,43,54,58 instead screened at 30,39,45,36 despite four terms per cell and w=12.
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
