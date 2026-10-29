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
CONCEPT = ("Hypothesis: metacyclic Z97:Z3(s=35), order 291, A=B=(1,2), with coupled normal-gap cards, two-basis-word kernel exchanges and pairing-specific logical retests, raises five-code fitness above 965.8564 in cell 0.200. Transposition alone offers no gain; every distance reading remains a provisional upper bound.")
GROUP = ("Keep the split-metacyclic construction and m=97,r=3,s=35,q=291 unchanged. The measured wide and tall beds reached 917.6920 and 965.8564 respectively, substantially above related dihedral beds. Retain this order-three action and order-97 normal subgroup rather than halving the lift for a larger protograph. These measurements do not certify distance transfer.")
SHAPES = ("Change both sides to (1,2). The factor remains five: n=1455, k>=291 and guaranteed rate 0.200; q=291 remains below q_max=299. Equal components have 582 variables and 291 checks. All reported parent measured minima occupied sector-2 Arow/Brow strips; the analogous component families move to sector-1 Acol/Bcol strips. Exact full ranks retain [0.175,0.225). Only fresh independently optimized fillings are submitted, never archived transpose images.")
FILLING = ("Four distinct terms per cell give both CSS row sums 8+4=12. Twenty cards prescribe unordered pairs of cross-cell normal-gap classes modulo <-1,35>, with additional cohort classes and affine triple geometries. Require directed differences to be disjoint across both cells in both orientations. Preserve these cards during simultaneous primary-fiber mutations. Challenge authoritative unrestricted and C3-fixed kernels with retained-word descent and two-basis-word exchanges, independently deepen finalists, then challenge complete pairings using exactly commuting, nontrivial logicals and original stabilizers. Full ranks, kernel-four exclusion, weights and CSS gate submission. Capped search readings guide selection but never certify absence of lighter logicals.")
HOW_AB = FILLING
FAMILY_NAME = "r020_m97r3s35_12x12_coupledgap_pairexchange_logicalretest_v4"

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
    """Unused historical tall-bed policy; NEW_CTX below replaces it."""
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
    """Wide gap cards, pair-exchange searches and actual logical retests."""
    q, width = G.q, 2 * G.q
    mask = (1 << q) - 1
    assert q == 291 and A_SHAPE == B_SHAPE == (1, 2)
    support = [[4, 4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    card, cohort = idx % 20, idx // 20
    seed = rng.randrange(1 << 62)
    # New proposal streams, not the archived tall policy's draws.
    rng = random.Random(seed + 7919 * (idx + 1) + 62000017)

    profiles = (
        ((2, 1, 1), (2, 1, 1)),
        ((2, 1, 1), (3, 1, 0)),
        ((3, 1, 0), (3, 1, 0)),
        ((2, 1, 1), (2, 2, 0)),
        ((3, 1, 0), (2, 2, 0)),
    )
    profile_pairs = ((0, 0), (0, 1), (1, 2), (0, 3), (1, 4))
    pa, pb = profile_pairs[(card + 2 * cohort) % 5]

    primitive = next(
        a for a in range(2, 97)
        if pow(a, 48, 97) != 1 and pow(a, 32, 97) != 1)
    gap_pairs = [
        (a, b) for a in range(6) for b in range(a, 6)
        if (a, b) != (0, 0)
    ]
    assert len(gap_pairs) == 20
    ga, gb = gap_pairs[card]
    # <-1,35> has order six in F97*. Its quotient has sixteen classes.
    # Avoid its identity class, which can force coincident primary gaps.
    gap_classes = (
        1 + (ga + 6 * cohort) % 15,
        1 + (gb + 6 * cohort) % 15,
    )
    ratios = {
        "A": pow(primitive, gap_classes[0], 97),
        "B": pow(primitive, gap_classes[1], 97),
    }

    def affine_class(t):
        it = pow(t, -1, 97)
        im = pow((t - 1) % 97, -1, 97)
        return tuple(sorted({
            t, (1 - t) % 97, it, (-im) % 97,
            t * im % 97, (t - 1) * it % 97,
        }))

    triangle_classes = sorted({
        affine_class(t) for t in range(2, 97)
        if len(affine_class(t)) == 6
    })
    triangles = {
        side: [
            triangle_classes[
                (card + 3 * cohort + offset + 5 * j)
                % len(triangle_classes)][0]
            for j in range(2)
        ]
        for side, offset in (("A", 0), ("B", 7))
    }

    # Cache native operations, never a separately implemented lift.
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
    # Tall galleries keep the unused matrices small. Transpose only the
    # exact monomial blocks returned by the authoritative constructor.
    gallery, zero = [[[g]] for g in range(q)], [[[]]]
    _, at, _ = lifted_matrices(P, gallery, zero)
    bt, _, _ = lifted_matrices(P, zero, gallery)
    blocks = {}
    for side, source in (("A", at), ("B", bt)):
        bank = []
        for g in range(q):
            rows = [0] * q
            shift = (g + 1) * q
            for r, original in enumerate(source):
                value = (original >> shift) & mask
                assert value.bit_count() == 1
                rows[value.bit_length() - 1] |= 1 << r
            bank.append(rows)
        blocks[side] = bank
    del gallery, at, bt

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    # These are coordinate restrictions of authoritative columns, not
    # quotient lifts. Every expanded returned word is verified below.
    h = G.idx((0, 1))
    assert table[table[h][h]][h] == identity
    orbit_masks = {}
    for side in ("A", "B"):
        remaining, orbits = set(range(q)), []
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
        orbit_masks[side] = [
            sum(1 << (j * q + c) for c in orbit)
            for j in range(2) for orbit in orbits
        ]

    def difference_free(grid):
        left, right = set(), set()
        for cell in grid[0]:
            ids = [G.idx(g) for g in cell]
            if len(ids) != len(set(ids)) or len(ids) != 4:
                return False
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    l, r = table[inverse[a]][b], table[a][inverse[b]]
                    if l in left or r in right:
                        return False
                    left.add(l)
                    right.add(r)
        return len(left) == len(right) == 24

    def fiber_points(side, j, count, delta):
        gap = delta * (1 if j == 0 else ratios[side]) % 97
        gap = gap * rng.choice((1, 35, 61)) % 97
        if rng.randrange(2):
            gap = -gap % 97
        origin = rng.randrange(97)
        points = [origin, (origin + gap) % 97]
        if count == 3:
            points.append(
                (origin + triangles[side][j] * gap) % 97)
        return points

    def fresh(profile, side):
        for _ in range(4096):
            delta = rng.randrange(1, 97)
            cells = []
            for j, occupancy in enumerate(profiles[profile]):
                fibers = list(range(3))
                rng.shuffle(fibers)
                cell, primary = [], True
                for b, count in zip(fibers, occupancy):
                    if not count:
                        continue
                    if count >= 2 and primary:
                        points = fiber_points(side, j, count, delta)
                        primary = False
                    else:
                        points = rng.sample(range(97), count)
                    cell.extend((a, b) for a in points)
                assert len(cell) == len(set(cell)) == 4
                cells.append(cell)
            grid = [cells]
            if difference_free(grid):
                return grid, delta
        raise ValueError("coupled normal-gap card could not be filled")

    def coupled_mutation(record, side):
        """Change both primary fibers while preserving their ratio class."""
        grid = [[list(cell) for cell in record["grid"][0]]]
        delta = rng.randrange(1, 97)
        for j, cell in enumerate(grid[0]):
            fibers = {}
            for t, g in enumerate(cell):
                fibers.setdefault(g[1], []).append(t)
            primary = next(
                positions for positions in fibers.values()
                if len(positions) >= 2)
            b = cell[primary[0]][1]
            points = fiber_points(side, j, len(primary), delta)
            for t, a in zip(primary, points):
                cell[t] = (a, b)
            # Occasionally perturb a nonprimary term as well. This is
            # never a common group automorphism or a whole-code gauge.
            other = [t for t in range(4) if t not in primary]
            if other and rng.random() < 0.5:
                t = rng.choice(other)
                choices = [(a, cell[t][1]) for a in range(97)
                           if (a, cell[t][1]) not in cell]
                cell[t] = rng.choice(choices)
        return (grid, delta) if difference_free(grid) else (None, None)

    def inspect(grid, delta, side):
        ids = [[G.idx(g) for g in cell] for cell in grid[0]]
        rows = []
        for r in range(q):
            row = 0
            for j, cell in enumerate(ids):
                block = 0
                for g in cell:
                    block ^= blocks[side][g][r]
                row |= block << (j * q)
            rows.append(row)
        quality = _component_quality(rows, width)
        if quality is None or quality[0][0] != 0:
            return None
        cycles, columns = quality
        fixed = [
            syndrome(columns, word) for word in orbit_masks[side]
        ]
        return {
            "grid": grid, "delta": delta, "rows": rows,
            "columns": columns, "fixed": fixed, "cycles": cycles,
            "signature": tuple(tuple(sorted(cell)) for cell in ids),
            "beam": [], "fixed_beam": [], "words": [],
            "d": width + 1, "fixed_d": width + 1,
        }

    def rank_gate(record):
        if "rank" in record:
            return True
        rows = record["rows"]
        rank = gf2_rank(rows)
        ranks = (
            gf2_rank([row & mask for row in rows]),
            gf2_rank([row >> q for row in rows]),
        )
        if min((rank,) + ranks) < q - 3:
            return False
        if not _no_kernel4(record["columns"]):
            return False
        record["rank"], record["cell_ranks"] = rank, ranks
        return True

    def search(columns, diagnostic_seed, trials, seeds, restricted):
        """Actual kernel words with single- and two-basis-word exchanges."""
        local = random.Random(diagnostic_seed)
        m = len(columns)
        order = list(range(m))
        beam = sorted(set(seeds), key=lambda w: (w.bit_count(), w))[:24]
        best = None

        def note(word):
            nonlocal best
            if word and (best is None or
                         word.bit_count() < best.bit_count()):
                best = word

        for word in beam:
            assert syndrome(columns, word) == 0
            note(word)

        for turn in range(trials):
            local.shuffle(order)
            if turn % 5 == 4 and beam:
                anchor = local.choice(beam[:6])
                preferred = list(_bits(anchor))
                local.shuffle(preferred)
                preferred = set(preferred[:max(1, len(preferred) // 2)])
                order.sort(key=lambda c: c not in preferred)

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
                    note(word)
            if not basis:
                continue

            light = sorted(
                set(basis + beam), key=lambda w: (w.bit_count(), w))[:16]
            moves = basis + beam
            grown = list(light) + beam
            for a in light:
                for b in moves:
                    word = a ^ b
                    note(word)
                    if word and word.bit_count() <= best.bit_count() + 6:
                        grown.append(word)
                if len(grown) > 192:
                    grown = sorted(
                        set(grown), key=lambda w: (w.bit_count(), w))[:48]

            local.shuffle(moves)
            starts = sorted(
                set(grown), key=lambda w: (w.bit_count(), w))[:4]
            # Unlike a one-word descent, these moves can cross a local
            # minimum requiring simultaneous cancellation by two words.
            exchanges = []
            if turn % 6 == 0:
                chosen = sorted(
                    set(basis), key=lambda w: (w.bit_count(), w))[
                        :100 if restricted else 64]
                exchanges = [
                    a ^ b for i, a in enumerate(chosen)
                    for b in chosen[:i]
                ]
            for start in starts:
                word = start
                for sweep in range(3):
                    changed = False
                    for other in moves:
                        proposed = word ^ other
                        if proposed and proposed.bit_count() < word.bit_count():
                            word, changed = proposed, True
                    if exchanges:
                        candidate = word
                        for move in exchanges:
                            proposed = word ^ move
                            if proposed and proposed.bit_count() < candidate.bit_count():
                                candidate = proposed
                        if candidate != word:
                            word, changed = candidate, True
                    note(word)
                    if not changed:
                        break
                grown.append(word)
            grown.append(best)
            beam = sorted(
                set(grown), key=lambda w: (w.bit_count(), w))[:24]
            if best.bit_count() <= (10 if restricted else 24):
                break

        if best is None:
            raise ValueError("no kernel witness from pair-exchange search")
        assert syndrome(columns, best) == 0
        beam = sorted(
            set(beam + [best]), key=lambda w: (w.bit_count(), w))[:24]
        return best.bit_count(), best, beam

    def probe(record, side, turn, full_trials, fixed_trials):
        offset = 0 if side == "A" else 100003
        sd = seed + offset + 700001 * turn
        fd, fw, fixed_beam = search(
            record["fixed"], sd + 3100009, fixed_trials,
            record["fixed_beam"], True)
        record["fixed_beam"] = fixed_beam
        expanded = []
        for word in fixed_beam[:6]:
            actual = 0
            for c in _bits(word):
                actual ^= orbit_masks[side][c]
            assert actual.bit_count() == 3 * word.bit_count()
            assert syndrome(record["columns"], actual) == 0
            expanded.append(actual)
        record["fixed_d"] = min(record["fixed_d"], 3 * fd)

        d, word, beam = search(
            record["columns"], sd, full_trials,
            record["beam"] + expanded, False)
        record["beam"] = beam
        record["d"] = min(record["d"], d, 3 * fd)
        record["words"] = sorted(
            set(record["words"] + expanded + beam[:8] + [word]),
            key=lambda w: (w.bit_count(), w))[:16]
        assert all((row & word).bit_count() % 2 == 0
                   for row in record["rows"])

    def merit(record):
        # Cap unusually high sampled readings before geometry tie-breaking.
        # The cap is a selection choice, not a distance lower bound.
        return (
            min(78, record["d"]),
            -record["cycles"][0], -record["cycles"][1],
            min(84, record["fixed_d"]), record["rank"],
        )

    def optimize(profile, side, restart):
        records, seen = [], set()
        for _ in range(48 + 4 * min(cohort, 2)):
            grid, delta = fresh(profile, side)
            record = inspect(grid, delta, side)
            if record is None or record["signature"] in seen:
                continue
            seen.add(record["signature"])
            records.append(record)
        records.sort(key=lambda r: (r["cycles"], r["signature"]))

        selected = []
        for record in records:
            if not rank_gate(record):
                continue
            probe(record, side, 1 + 17 * restart, 8, 12)
            selected.append(record)
            if len(selected) >= 8:
                break
        if not selected:
            return []

        current = max(selected, key=merit)
        for step in range(18):
            if step % 6 == 0:
                grid, delta = fresh(profile, side)
            else:
                grid, delta = coupled_mutation(current, side)
            if grid is None:
                continue
            proposed = inspect(grid, delta, side)
            if proposed is None or proposed["signature"] in seen:
                continue
            seen.add(proposed["signature"])
            if proposed["cycles"][1] > current["cycles"][1] + 6 * q:
                continue
            if not rank_gate(proposed):
                continue
            probe(proposed, side, 31 + step + 17 * restart, 4, 8)
            selected.append(proposed)
            pk, ck = merit(proposed), merit(current)
            if pk > ck or (pk[0] == ck[0] and rng.random() < 0.10):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists = selected[:4]
        for record in finalists:
            probe(record, side, 101 + restart, 24, 40)
        finalists.sort(key=merit, reverse=True)
        finalists = finalists[:2]
        for record in finalists:
            # Independent challenges retain every earlier smaller reading.
            probe(record, side, 211 + restart, 48, 80)
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
        """Retest genuine logicals using the ORIGINAL quantum matrices."""
        reading = n + 1
        witnesses = []
        for kind, record, opposite, stabilizers in (
                ("Z", a, hx, hz), ("X", b, hz, hx)):
            pivots = span(stabilizers)
            local = random.Random(
                seed + (4100009 if kind == "Z" else 5100011))
            words = sorted(
                set(record["words"]), key=lambda w: (w.bit_count(), w))[:4]
            for word in words:
                assert syndrome(record["columns"], word) == 0
                for strip in range(2):
                    if kind == "Z":
                        actual = sum(
                            ((word >> (j * q)) & mask)
                            << ((2 * j + strip) * q)
                            for j in range(2))
                    else:
                        actual = word << (strip * width)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    if not outside(actual, pivots):
                        continue

                    reducers = list(stabilizers)
                    local.shuffle(reducers)
                    for sweep in range(4):
                        changed = False
                        for row in reducers:
                            candidate = actual ^ row
                            if candidate.bit_count() < actual.bit_count():
                                actual, changed = candidate, True
                        # Pair moves can cancel newly introduced sector-two
                        # support which a single-stabilizer move cannot.
                        near = sorted(
                            stabilizers,
                            key=lambda row: (
                                -(row & actual).bit_count(), row))[:24]
                        best = actual
                        for i, first in enumerate(near):
                            for second in near[:i]:
                                candidate = actual ^ first ^ second
                                if candidate.bit_count() < best.bit_count():
                                    best = candidate
                        if best != actual:
                            actual, changed = best, True
                        if not changed:
                            break
                    assert actual and outside(actual, pivots)
                    assert all((actual & row).bit_count() % 2 == 0
                               for row in opposite)
                    reading = min(reading, actual.bit_count())
                    witnesses.append((kind, actual.bit_count()))
        return reading, witnesses

    for restart in range(2):
        aa = optimize(pa, "A", restart)
        bb = optimize(pb, "B", restart)
        pairings = [(a, b) for a in aa for b in bb]
        pairings.sort(key=lambda ab: (
            min(78, ab[0]["d"], ab[1]["d"]),
            min(84, ab[0]["d"]) + min(84, ab[1]["d"]),
            -ab[0]["cycles"][1] - ab[1]["cycles"][1]),
            reverse=True)

        best, best_key = None, None
        for a, b in pairings:
            A, B = a["grid"], b["grid"]
            hx, hz, n = lifted_matrices(G, A, B)
            assert n == 5 * q == 1455
            ar, br = _verified_components(hx, hz, q)
            assert ar == a["rows"] and br == b["rows"]
            k = n - gf2_rank(hx) - gf2_rank(hz)
            if not (k >= q and 7 * n <= 40 * k < 9 * n):
                continue
            if not _no_kernel4(_syndrome_columns(hx, n)):
                continue
            if not _no_kernel4(_syndrome_columns(hz, n)):
                continue
            if not css_ok(hx, hz):
                raise ValueError("verified wide construction failed CSS")
            assert max(row.bit_count() for row in hx + hz) == 12

            logical_d, logical_words = logical_retest(a, b, hx, hz, n)
            conservative = min(a["d"], b["d"], logical_d)
            capped = min(72, conservative)
            key = (
                k * capped * capped, capped,
                -a["cycles"][1] - b["cycles"][1],
                min(78, a["d"]) + min(78, b["d"]),
                -(k - q),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "A": A, "B": B, "idx": idx, "k": k,
                    "card": (pa, pb, gap_classes, triangles),
                    "component_readings": (a["d"], b["d"]),
                    "fixed_readings": (a["fixed_d"], b["fixed_d"]),
                    "component_cycles": (a["cycles"], b["cycles"]),
                    "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
                    "logical_retest": logical_d,
                    "logical_witness_weights": logical_words,
                    "pairing_merit": key,
                }
        if best is not None:
            return best
    raise ValueError("new wide gap pairings missed full validity gates")


def CELL(rng, G, side, i, j, ctx):
    """Return the complete selected wide cell without post-screen draws."""
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
# REFUTED: Whole-cell Sidon supports and independently repaired transpose kernels uniformly exclude sub-50 logicals: accepted parent slots 21,43,54,58 instead screened at 30,39,45,36 despite four terms per cell and w=12.
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
