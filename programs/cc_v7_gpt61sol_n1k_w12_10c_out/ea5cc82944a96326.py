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
CONCEPT = ("Hypothesis: on A5 x C2, order 120, A=(1,2), B=(2,5), generating four-term A cells and subgroup-challenged degree-three/four B columns raise five-code fitness above 117.874 in cell 0.250. Measurement can refute this; sampled readings remain upper readings, not distance certificates.")
GROUP = ("Jump to the split direct product A5 x C2, order 120. Different order proves nonisomorphism to the order-291 parent. Its ten conjugacy classes distinguish it from SL2(F5), which has nine; the central extension splits. The SL2(F5) quarter-rate bed scored 117.9 and the S5 bed scored 86.2, motivating another nonsolvable order-120 experiment without assuming distance transfer. This group/shape bed is not listed in RUN STATE.")
SHAPES = ("A=(1,2), B=(2,5): n=1440, k>=360 and guaranteed rate 0.250; q_max=124 and the group fixes q=120. Unequal wide components have 240 variables/120 checks and 600 variables/240 checks. Parent minima occupied sector-2 strips spanning both Arows at one Brow or both Brows at one Arow. The new component families occupy sector-1 Acol/Bcol strips and are challenged modulo ORIGINAL stabilizer spans. Actual full ranks enforce [0.225,0.275).")
FILLING = ("A counts (4,4), and two positive B profiles with row totals eight and column degrees (4,3,3,3,3), give HX<=8+4=12 and HZ<=8+4=12. Twenty cards prescribe five binomial relative-order patterns, two B profiles and two geometry objectives. Require native bidirectional Sidon cells and full relative generation for cells of size three/four. Authoritative monomial blocks supply every screening matrix. Central-C2, noncentral-V4 and C3 fixed restrictions, manageable localized kernels, retained null words, witness-directed cell replacement and independent finalist retests challenge short families. Final pairings use genuine sector-1 logicals reduced by original stabilizers, actual k, full kernel-four exclusion, weights and CSS. Neither geometry nor sampled readings certify quantum distance.")
HOW_AB = FILLING
FAMILY_NAME = "r025_A5xC2_12x25_ordercards_subgroup_logical_v1"

GTYPE = "custom"
CUSTOM_GROUP = """
def valid_q(q):
    return q == 120

def custom_ops(q):
    if not valid_q(q):
        raise ValueError("A5 x C2 requires order 120")
    from itertools import permutations
    # (z,p0,p1,p2,p3,p4): central bit and an even permutation.
    even = [
        p for p in permutations(range(5))
        if sum(p[i] > p[j] for i in range(5)
               for j in range(i+1, 5)) % 2 == 0
    ]
    els = [(z,) + p for z in range(2) for p in even]
    positions = {g: i for i, g in enumerate(els)}

    def red(g):
        return (int(g[0]) % 2,) + tuple(int(x) % 5 for x in g[1:])

    def mul(g, h):
        return (g[0] ^ h[0],) + tuple(
            g[1 + h[1+i]] for i in range(5))

    def inv(g):
        p = [0] * 5
        for i in range(5):
            p[g[1+i]] = i
        return (g[0],) + tuple(p)

    def idx(g):
        return positions[red(g)]

    return dict(elements=els, ident=(0,0,1,2,3,4),
                mul=mul, inv=inv, idx=idx, red=red)
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


def _retired_metacyclic_ctx(rng, G, idx):
    """Inactive historical tall-bed experiment; not called by NEW_CTX."""
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
    """Twenty split-extension cards with binding subgroup/logical witnesses."""
    from math import exp

    q = G.q
    assert q == 120 and A_SHAPE == (1, 2) and B_SHAPE == (2, 5)
    card, cohort = idx % 20, idx // 20
    topology, model = divmod(card, 5)
    support_profile, geometry = topology % 2, topology // 2
    seed = rng.randrange(1 << 62)
    rng = random.Random(seed ^ (0xA5C22517 + 1000003 * idx))
    mask = (1 << q) - 1
    table = [[G.idx(G.mul(g, h)) for h in G.elements]
             for g in G.elements]
    inverse = [G.idx(G.inv(g)) for g in G.elements]
    ident = G.idx(G.e)

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
            return inverse[g]

    probe_group = Probe()
    orders = []
    for g in range(q):
        value = ident
        for order in range(1, q + 1):
            value = table[value][g]
            if value == ident:
                orders.append(order)
                break
        else:
            raise ValueError("invalid group order")

    # Cache only blocks returned by the authoritative constructor.
    # Chunked probes avoid constructing an unnecessarily wide gallery.
    blocks = {"A": [], "B": []}
    for start in range(0, q, 24):
        ids = list(range(start, min(start + 24, q)))
        gallery = [[[g] for g in ids]]
        xx, _, _ = lifted_matrices(probe_group, gallery, [[[]]])
        _, zz, _ = lifted_matrices(probe_group, [[[]]], gallery)
        assert len(xx) == len(zz) == q
        for j in range(len(ids)):
            blocks["A"].append([(row >> (j*q)) & mask for row in xx])
            blocks["B"].append([(row >> (j*q)) & mask for row in zz])

    supports = {
        "A": [[4, 4]],
        "B": [
            list(row) for row in (
                ((2,2,2,1,1), (2,1,1,2,2)),
                ((1,2,2,2,1), (3,1,1,1,2)),
            )[support_profile]
        ],
    }
    assert _grid_weights(supports["A"], supports["B"]) == 12 <= W_MAX
    assert _no_dead_lines(supports["A"], supports["B"])
    assert [sum(row) for row in supports["B"]] == [8, 8]
    assert [sum(row[j] for row in supports["B"])
            for j in range(5)] == [4, 3, 3, 3, 3]

    # These are actual order cards, not soft preferences or orbit images.
    recipes = ((5,), (6,), (10,), (5,10), (3,6,10))
    double_positions = [
        (i, j) for i, row in enumerate(supports["B"])
        for j, count in enumerate(row) if count == 2
    ]
    required_orders = {
        position: recipes[model][t % len(recipes[model])]
        for t, position in enumerate(double_positions)
    }
    order_pools = {
        order: [g for g in range(q) if orders[g] == order]
        for order in (3, 5, 6, 10)
    }

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
                    left.add(table[inverse[a]][b])
                    right.add(table[a][inverse[b]])
        wanted = len(cell) * (len(cell) - 1)
        return len(left) == len(right) == wanted

    def generating(cell):
        relative = [table[inverse[cell[0]]][g] for g in cell[1:]]
        moves = relative + [inverse[g] for g in relative]
        reached, pending = {ident}, [ident]
        for value in pending:
            for g in moves:
                image = table[value][g]
                if image not in reached:
                    reached.add(image)
                    pending.append(image)
        return len(reached) == q

    def fresh_cell(count, side, i, j):
        if count == 1:
            return [rng.randrange(q)]
        if count == 2:
            wanted = required_orders[(i, j)]
            root = rng.randrange(q)
            cell = [root, table[root][rng.choice(order_pools[wanted])]]
            assert sidon(cell)
            return cell
        assert count in (3, 4)
        preferred = ({5,10}, {3,6}, {5,10}, {3,5,10}, {3,6,10})[model]
        best = None
        for attempt in range(1024):
            cell = rng.sample(range(q), count)
            if not sidon(cell) or not generating(cell):
                continue
            cost = sum(
                orders[table[inverse[a]][b]] not in preferred
                for t, a in enumerate(cell) for b in cell[:t])
            if best is None or cost < best[0]:
                best = cost, cell
            if cost == 0 or attempt >= 63:
                break
        if best is None:
            raise ValueError("generating native Sidon cell unavailable")
        return best[1]

    def fresh(side):
        return [[fresh_cell(count, side, i, j)
                 for j, count in enumerate(row)]
                for i, row in enumerate(supports[side])]

    # Coordinate restrictions, not replacement lifts. All restricted
    # syndromes below are XORs of authoritative component columns.
    central = G.idx((1,0,1,2,3,4))
    v1 = G.idx((0,1,0,3,2,4))
    v2 = G.idx((0,2,3,0,1,4))
    v4 = [ident, v1, v2, table[v1][v2]]
    assert len(set(v4)) == 4
    assert all(table[a][b] in v4 for a in v4 for b in v4)
    assert table[central][central] == ident
    assert all(table[central][g] == table[g][central] for g in range(q))
    h3 = next(g for g in range(q) if orders[g] == 3)
    subgroups = (
        (2, [ident, central]),
        (4, v4),
        (3, [ident, h3, table[h3][h3]]),
    )
    partitions = {"A": {}, "B": {}}
    for side in ("A", "B"):
        for size, subgroup in subgroups:
            remaining, orbits = set(range(q)), []
            while remaining:
                root = min(remaining)
                orbit = {
                    table[root][h] if side == "A" else table[h][root]
                    for h in subgroup
                }
                assert len(orbit) == size and orbit <= remaining
                remaining.difference_update(orbit)
                orbits.append(sum(1 << c for c in orbit))
            partitions[side][size] = [
                word << (j*q)
                for j in range(len(supports[side][0]))
                for word in orbits
            ]

    def dependency_basis(columns, order):
        pivots, basis = {}, []
        for c in order:
            value, word = columns[c], 1 << c
            while value:
                lead = value.bit_length() - 1
                old = pivots.get(lead)
                if old is None:
                    pivots[lead] = value, word
                    break
                value ^= old[0]
                word ^= old[1]
            if not value:
                basis.append(word)
        return basis

    def search(columns, diagnostic, trials, seeds=()):
        """Genuine nonzero kernel words; no lower-bound interpretation."""
        width = len(columns)
        retained = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:24]
        order = list(range(width))
        best = min(retained, key=int.bit_count) if retained else None

        def note(word):
            nonlocal best
            if word and (best is None or word.bit_count() < best.bit_count()):
                best = word

        for word in retained:
            assert syndrome(columns, word) == 0
        for turn in range(trials):
            diagnostic.shuffle(order)
            if turn % 4 == 3 and retained:
                hint = list(_bits(diagnostic.choice(retained[:6])))
                diagnostic.shuffle(hint)
                front = set(hint[:max(1, len(hint)//2)])
                order.sort(key=lambda c: c not in front)
            basis = dependency_basis(columns, order)
            for word in basis:
                note(word)
            light = sorted(set(basis + retained),
                           key=lambda w: (w.bit_count(), w))[:16]
            additions = list(light)
            moves = basis + retained
            for word in light:
                for other in moves:
                    candidate = word ^ other
                    note(candidate)
                    if candidate and best is not None:
                        if candidate.bit_count() <= best.bit_count() + 4:
                            additions.append(candidate)
                if len(additions) > 192:
                    additions = sorted(set(additions),
                                       key=int.bit_count)[:40]
            diagnostic.shuffle(moves)
            for initial in light[:6]:
                word = initial
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < word.bit_count():
                            word, changed = candidate, True
                    note(word)
                    if not changed:
                        break
                additions.append(word)
            if best is not None:
                additions.append(best)
            retained = sorted(set(retained + additions),
                              key=lambda w: (w.bit_count(), w))[:24]
            if best is not None and best.bit_count() <= 4:
                break
        if best is not None:
            assert syndrome(columns, best) == 0
        return best, retained

    def inspect(grid, side):
        nr, nc = len(grid), len(grid[0])
        rows = [0] * (nr*q)
        for i, base_row in enumerate(grid):
            for j, cell in enumerate(base_row):
                for g in cell:
                    for r, value in enumerate(blocks[side][g]):
                        rows[i*q+r] ^= value << (j*q)
        width = nc*q
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        rank = gf2_rank(rows)
        if rank < nr*q - (8 if side == "A" else 14):
            return None
        cost = ((2000 + 500*geometry)*cycles[0]
                + (5 + model + 5*geometry)*cycles[1])
        return {
            "grid": copy_grid(grid), "rows": rows, "columns": columns,
            "rank": rank, "cycles": cycles, "cost": cost,
            "signature": signature(grid), "side": side,
            "words": [], "d": width + 1, "restrictions": {},
        }

    def localized(record):
        """Exact one-block minimum at nullity <=12; witnesses otherwise."""
        columns = record["columns"]
        found = []
        for start in range(0, len(columns), q):
            basis = dependency_basis(columns, range(start, start + q))
            if not basis:
                continue
            if len(basis) <= 12:
                word, best = 0, None
                for t in range(1, 1 << len(basis)):
                    word ^= basis[(t & -t).bit_length() - 1]
                    if best is None or word.bit_count() < best.bit_count():
                        best = word
                found.append(best)
            else:
                found.extend(sorted(basis, key=int.bit_count)[:4])
        for word in found:
            assert word and syndrome(columns, word) == 0
        return found

    def challenge(record, turn, trials):
        side = record["side"]
        offset = 100003 * int(side == "B")
        if not record.get("localized_checked"):
            record["words"].extend(localized(record))
            record["localized_checked"] = True
        for size in (2, 4, 3):
            masks = partitions[side][size]
            compressed = [syndrome(record["columns"], word) for word in masks]
            local = random.Random(seed + offset + 700001*turn + 1000003*size)
            word, _ = search(compressed, local, max(2, trials//3))
            if word is None:
                continue
            actual = 0
            for c in _bits(word):
                actual ^= masks[c]
            assert actual.bit_count() == size * word.bit_count()
            assert syndrome(record["columns"], actual) == 0
            record["words"].append(actual)
            record["restrictions"][size] = min(
                record["restrictions"].get(size, len(record["columns"]) + 1),
                actual.bit_count())
        local = random.Random(seed + offset + 900001 + 700001*turn)
        word, beam = search(record["columns"], local, trials, record["words"])
        if word is None:
            raise ValueError("positive-rate component yielded no witness")
        assert all((row & word).bit_count() % 2 == 0 for row in record["rows"])
        record["words"] = sorted(set(record["words"] + beam + [word]),
                                 key=lambda w: (w.bit_count(), w))[:24]
        record["d"] = min(record["d"],
                          min(w.bit_count() for w in record["words"]))

    def component_pool(side, restart):
        records, seen = [], set()
        for _ in range(40):
            record = inspect(fresh(side), side)
            if record is not None and record["signature"] not in seen:
                seen.add(record["signature"])
                records.append(record)
        records.sort(key=lambda r: (r["cost"], -r["rank"]))
        if not records:
            return []

        # Anneal actual cells, preserving support and exact binomial cards.
        mutable = [(i, j) for i, row in enumerate(supports[side])
                   for j, count in enumerate(row)]
        retained = records[:8]
        for initial in records[:2]:
            current = initial
            for step in range(48):
                grid = copy_grid(current["grid"])
                i, j = rng.choice(mutable)
                grid[i][j] = fresh_cell(supports[side][i][j], side, i, j)
                sig = signature(grid)
                if sig in seen:
                    continue
                seen.add(sig)
                proposed = inspect(grid, side)
                if proposed is None:
                    continue
                temperature = q * (12*(1-step/48.0)**2 + 0.2)
                delta = proposed["cost"] - current["cost"]
                if delta <= 0 or rng.random() < exp(min(0.0, -delta/temperature)):
                    current = proposed
                    retained.append(current)
                    retained.sort(key=lambda r: (r["cost"], -r["rank"]))
                    del retained[12:]

        selected, bank = [], set()
        for record in retained:
            if not _no_kernel4(record["columns"]):
                continue
            challenge(record, 1 + 10*restart, 12)
            bank.update(record["words"])
            selected.append(record)
            if len(selected) == 6:
                break
        if not selected:
            return []

        def merit(record):
            for word in bank:
                if word.bit_count() < record["d"]:
                    if syndrome(record["columns"], word) == 0:
                        record["words"].append(word)
                        record["d"] = word.bit_count()
            # Limit selection on unusually optimistic short searches.
            # This cap is not a lower bound on component distance.
            return (min(26, record["d"]), -record["cost"],
                    record["d"], record["rank"])

        current = max(selected, key=merit)
        for step in range(16):
            word = min(current["words"], key=int.bit_count)
            occupied = {c // q for c in _bits(word)}
            positions = [p for p in mutable if p[1] in occupied]
            i, j = rng.choice(positions if positions else mutable)
            grid = copy_grid(current["grid"])
            grid[i][j] = fresh_cell(supports[side][i][j], side, i, j)
            sig = signature(grid)
            if sig in seen:
                continue
            seen.add(sig)
            proposed = inspect(grid, side)
            if proposed is None:
                continue
            if not _no_kernel4(proposed["columns"]):
                continue
            challenge(proposed, 20 + step + 50*restart, 8)
            bank.update(proposed["words"])
            selected.append(proposed)
            if merit(proposed) > merit(current):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists, final_seen = [], set()
        for record in selected:
            if record["signature"] in final_seen:
                continue
            final_seen.add(record["signature"])
            challenge(record, 101 + 10*restart, 32)
            bank.update(record["words"])
            finalists.append(record)
            if len(finalists) == 3:
                break
        finalists.sort(key=merit, reverse=True)
        for record in finalists[:2]:
            # Retained earlier witnesses remain binding.
            challenge(record, 211 + 10*restart, 64 + 8*(cohort % 3))
            bank.update(record["words"])
        finalists.sort(key=merit, reverse=True)
        return finalists[:2]

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
        """Nontrivial commuting words modulo the ORIGINAL stabilizer spans."""
        readings = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            candidates = set()
            for word in sorted(set(record["words"]),
                               key=lambda w: (w.bit_count(), w))[:4]:
                for location in range(5 if kind == "Z" else 2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j*q)) & mask)
                            << ((5*j + location)*q) for j in range(2))
                    else:
                        actual = word << (5*location*q)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if outside(actual, pivots):
                        candidates.add(actual)
            reading = n + 1
            local = random.Random(seed + (3100009 if kind == "Z" else 4100011))
            for actual in sorted(candidates,
                                 key=lambda w: (w.bit_count(), w))[:6]:
                reducers = list(stabilizers)
                local.shuffle(reducers)
                for sweep in range(3):
                    changed = False
                    for row in reducers:
                        proposed = actual ^ row
                        if proposed.bit_count() < actual.bit_count():
                            actual, changed = proposed, True
                    nearby = sorted(
                        stabilizers,
                        key=lambda row: (row & actual).bit_count(),
                        reverse=True)[:20]
                    for i, first in enumerate(nearby):
                        for second in nearby[:i]:
                            proposed = actual ^ first ^ second
                            if proposed.bit_count() < actual.bit_count():
                                actual, changed = proposed, True
                    if not changed:
                        break
                assert actual and outside(actual, pivots)
                assert all((actual & row).bit_count() % 2 == 0
                           for row in opposite)
                reading = min(reading, actual.bit_count())
            # n+1 is a missed embedding, not an exclusion certificate.
            readings.append(reading)
        return tuple(readings)

    for restart in range(2):
        aa = component_pool("A", restart)
        bb = component_pool("B", restart)
        survivors = []
        for a in aa:
            for b in bb:
                A = [[[G.elements[g] for g in cell] for cell in row]
                     for row in a["grid"]]
                B = [[[G.elements[g] for g in cell] for cell in row]
                     for row in b["grid"]]
                hx, hz, n = lifted_matrices(G, A, B)
                k = n - gf2_rank(hx) - gf2_rank(hz)
                if not (n == 1440 and k >= 360 and 9*n <= 40*k < 11*n):
                    continue
                actual_a = [
                    sum(((hx[r] >> (5*j*q)) & mask) << (j*q)
                        for j in range(2)) for r in range(q)]
                actual_b = [row & ((1 << (5*q))-1) for row in hz[:2*q]]
                assert actual_a == a["rows"] and actual_b == b["rows"]
                logicals = logical_readings(a, b, hx, hz, n)
                detected = min(26, a["d"], b["d"], *logicals)
                key = (k*detected*detected, detected,
                       -a["cost"]-b["cost"], a["d"]+b["d"])
                survivors.append((key, A, B, hx, hz, k, a, b, logicals))
        survivors.sort(key=lambda item: item[0], reverse=True)
        for key, A, B, hx, hz, k, a, b, logicals in survivors:
            if not _no_kernel4(_syndrome_columns(hx, 1440)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, 1440)):
                continue
            assert css_ok(hx, hz)
            assert max(row.bit_count() for row in hx + hz) == 12
            assert _grid_weights(supports["A"], supports["B"]) == 12
            for position, wanted in required_orders.items():
                i, j = position
                cell = b["grid"][i][j]
                assert orders[table[inverse[cell[0]]][cell[1]]] == wanted
            return {
                "A": A, "B": B, "idx": idx, "k": k,
                "card": (support_profile, geometry, model, cohort),
                "binomial_order_card": tuple(
                    required_orders[p] for p in double_positions),
                "component_readings": (a["d"], b["d"]),
                "component_ranks": (a["rank"], b["rank"]),
                "component_cycles": (a["cycles"], b["cycles"]),
                "subgroup_readings": (
                    dict(a["restrictions"]), dict(b["restrictions"])),
                "verified_localized_logical_readings": logicals,
                "pairing_merit": key,
                "target_cell": 0.250,
            }
    raise ValueError("split-extension pairings missed full quarter-rate gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected native-group cell without further draws."""
    cell = list(ctx[side][i][j])
    assert 1 <= len(cell) <= 4 and len(set(cell)) == len(cell)
    return cell

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: All-four-term supports with within-coset Sidon selection and component descent reliably exclude sub-30 logicals; the measured parent screened slots 18, 26 and 48 at 27, 24 and 21 respectively.
# REFUTED: Whole-cell Sidon supports and repaired transpose kernels uniformly eliminate sub-50 readings on the order-291 tall bed; current slots 21,43,54,58 instead screened at 30,39,45,36 despite w=12.
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
