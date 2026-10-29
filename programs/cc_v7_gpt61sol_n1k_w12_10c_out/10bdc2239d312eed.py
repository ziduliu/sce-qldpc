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
CONCEPT = ("Hypothesis: metacyclic Z97:Z3(s=35), order 291, shapes (2,1)x(2,1), with coupled normal-gap classes and witness-directed replacement of verified transposed components, raises five-code fitness above 917.692 in cell 0.200. Search readings remain provisional upper bounds.")
GROUP = ("Keep Z97:Z3(s=35), q=291, unchanged. Its current bed scores 917.692 with five measured readings 65..70, whereas the dihedral order-298 bed scored 28.9. Retain the order-three action and order-97 normal subgroup; neither these readings nor component screening certifies distance.")
SHAPES = ("Change both sides to single-column (2,1). The factor remains five: n=1455, k>=291, guaranteed rate 0.200, and q<=299 under the size window. The relevant transposed components each have 582 variables and 291 checks. This orientation is equivalent as a shape family and alone adds no distance; entirely new labels are required. Parent minima used one Acol across both Bcols or one Bcol across both Acols, whose analogous locations move into sector-2 Arow/Brow blocks. Exact full ranks retain [0.175,0.225).")
FILLING = ("Four distinct terms in every cell give HX and HZ weights 4+8=12. Twenty cards cross four occupancy-profile pairs with five coupled normal-gap classes; extra cohorts vary within-fiber triangle classes and cycle priorities. Assemble screening components only from verified monomial lifts. Shortlist by actual cycles and ranks, search genuine kernel words, replace whole cells used by surviving witnesses while preserving each card, and independently deepen finalists. Exact full ranks, full kernel-4 exclusion and CSS gate submission. No archived filling is transposed, and no component reading is a quantum-distance certificate.")
HOW_AB = FILLING
FAMILY_NAME = "r020_m97r3s35_21x21_gapcoupled_witness_descent_v3"

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


def _legacy_fourterm_ctx(rng, G, idx):
    """Historical (1,2)x(1,2) reference; not called by the new NEW_CTX."""
    q = G.q
    width = 2 * q
    support = [[4, 4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    # Each profile specifies an unordered pair of within-cell occupancies.
    # Twenty unordered profile pairs, excluding the entirely even-fiber
    # pair, are explicit designs rather than permutation images.
    occupancies = ((2, 1, 1), (3, 1, 0), (2, 2, 0))
    profiles = ((0, 0), (0, 1), (1, 1),
                (0, 2), (1, 2), (2, 2))
    plans = [(a, b) for a in range(6) for b in range(a, 6)
             if (a, b) != (5, 5)]
    pa, pb = plans[idx % 20]
    phase = (idx // 20) % 3
    diagnostic_seed = rng.randrange(1 << 62)

    def fresh(profile):
        cells = []
        for kind in profiles[profile]:
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            for b, size in zip(fibers, occupancies[kind]):
                if not size:
                    continue
                if size == 3 and phase:
                    # Three-point affine classes are unchanged by common
                    # rotation scaling/translation. Classes 3 and 5 are
                    # distinct even after permuting the three points.
                    t = (3, 5)[phase - 1]
                    origin = rng.randrange(97)
                    scale = rng.randrange(1, 97)
                    rotations = [origin, (origin + scale) % 97,
                                 (origin + t * scale) % 97]
                else:
                    for _ in range(256):
                        rotations = rng.sample(range(97), size)
                        differences = [(a - c) % 97 for a in rotations
                                       for c in rotations if a != c]
                        if len(set(differences)) == len(differences):
                            break
                    else:
                        raise ValueError("Sidon fiber generation failed")
                cell.extend((a, b) for a in rotations)
            assert len(cell) == len(set(cell)) == 4
            cells.append(cell)
        return [cells]

    def kernel_probe(columns, diagnostic, trials):
        """Search genuine classical kernel words, never certify distance.

        Column elimination tracks words in original coordinates. Pair
        sums, triple sums and nonzero cancellation descent all remain
        in the kernel of the VERIFIED component matrix.
        """
        reading = width + 1
        order = list(range(width))
        for _ in range(trials):
            diagnostic.shuffle(order)
            pivots = {}
            basis = []
            for c in order:
                syndrome = columns[c]
                word = 1 << c
                while syndrome:
                    p = syndrome.bit_length() - 1
                    old = pivots.get(p)
                    if old is None:
                        pivots[p] = (syndrome, word)
                        break
                    syndrome ^= old[0]
                    word ^= old[1]
                if not syndrome:
                    basis.append(word)
                    reading = min(reading, word.bit_count())

            light = sorted(basis, key=int.bit_count)[:16]
            for word in light:
                for other in basis:
                    candidate = word ^ other
                    if candidate:
                        reading = min(reading, candidate.bit_count())

            # Descent explores combinations beyond the parent's single
            # systematic basis and its pair sums. Zero is never a witness.
            diagnostic.shuffle(basis)
            for word in light[:6]:
                weight = word.bit_count()
                for sweep in range(2):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if not candidate:
                            continue
                        w = candidate.bit_count()
                        if w < weight:
                            word, weight = candidate, w
                            changed = True
                    reading = min(reading, weight)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    candidate = a ^ b ^ c
                    if candidate:
                        reading = min(reading, candidate.bit_count())
            if reading <= 24:
                break
        return reading

    pools = [[], []]
    signatures = [set(), set()]
    # Each call supplies both components under the exact evaluator
    # conventions. Their original pairing does not constrain selection.
    for trial in range(40):
        A, B = fresh(pa), fresh(pb)
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 5 * q == 1455
        ar, br = _verified_components(hx, hz, q)
        for side, grid, rows in ((0, A, ar), (1, B, br)):
            signature = tuple(tuple(sorted(G.idx(g) for g in cell))
                              for cell in grid[0])
            if signature in signatures[side]:
                continue
            quality = _component_quality(rows, width)
            if quality is None:
                continue
            signatures[side].add(signature)
            cycles, columns = quality
            pools[side].append({
                "grid": grid, "rows": rows, "columns": columns,
                "cycles": cycles, "trial": trial,
            })

    def select_component(pool, side):
        if not pool:
            raise ValueError("empty verified component pool")
        floor = min(record["cycles"][0] for record in pool)
        # Extra slots explore fewer projection triangles while allowing
        # one additional orbit of four-cycles; this changes graph selection.
        allowance = q if phase == 1 else 0
        near = [record for record in pool
                if record["cycles"][0] <= floor + allowance]
        if phase == 1:
            near.sort(key=lambda record: (
                record["cycles"][1], record["cycles"][0],
                record["trial"]))
        else:
            near.sort(key=lambda record: (
                record["cycles"], record["trial"]))

        screened = []
        mask = (1 << q) - 1
        # Continue past a rejected record rather than wasting its slot.
        for record in near:
            rows = record["rows"]
            rank = gf2_rank(rows)
            if rank < q - 3:
                continue
            # Inspect individual blocks extracted from verified rows.
            # A loss beyond the three quotient dimensions is rejected;
            # no separate left/right action is implemented here.
            cell_ranks = (
                gf2_rank([row & mask for row in rows]),
                gf2_rank([row >> q for row in rows]),
            )
            if min(cell_ranks) < q - 3:
                continue
            if not _no_kernel4(record["columns"]):
                continue
            diagnostic = random.Random(
                diagnostic_seed + 100003 * side)
            d = kernel_probe(record["columns"], diagnostic,
                             16 if phase == 2 else 12)
            record["rank"] = rank
            record["cell_ranks"] = cell_ranks
            record["d"] = d
            screened.append(record)
            if len(screened) == 6:
                break
        if not screened:
            raise ValueError("no component passed rank/kernel screening")

        def merit(record):
            four, six = record["cycles"]
            return (record["d"], -four, -six, record["rank"])

        screened.sort(key=merit, reverse=True)
        finalists = screened[:2]
        for record in finalists:
            # A fresh, longer search can only lower the first reading.
            # Common diagnostic seeds reduce candidate-specific luck.
            diagnostic = random.Random(
                diagnostic_seed + 700001 + 100003 * side)
            record["d"] = min(
                record["d"],
                kernel_probe(record["columns"], diagnostic,
                             64 if phase == 2 else 48))
        finalists.sort(key=merit, reverse=True)
        return finalists

    aa = select_component(pools[0], 0)
    bb = select_component(pools[1], 1)

    def pair_merit(pair):
        a, b = pair
        ac, bc = a["cycles"], b["cycles"]
        return (min(a["d"], b["d"]), a["d"] + b["d"],
                -max(ac[0], bc[0]), -ac[0] - bc[0],
                -max(ac[1], bc[1]), -ac[1] - bc[1])

    pairs = sorted(((a, b) for a in aa for b in bb),
                   key=pair_merit, reverse=True)
    for a, b in pairs:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= q and 7 * n <= 40 * k < 9 * n):
            continue
        if not _no_kernel4(_syndrome_columns(hx, n)):
            continue
        if not _no_kernel4(_syndrome_columns(hz, n)):
            continue
        if not css_ok(hx, hz):
            raise ValueError("verified construction failed CSS")
        assert max(row.bit_count() for row in hx + hz) == 12
        return {
            "A": A, "B": B, "idx": idx, "k": k,
            "profiles": (pa, pb), "phase": phase,
            "component_readings": (a["d"], b["d"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
        }
    raise ValueError("no independently selected pair passed full screening")


def NEW_CTX(rng, G, idx):
    """New single-column cards; verified components and witness-directed search."""
    q = G.q
    width = 2 * q
    mask = (1 << q) - 1
    assert q == 291
    support = [[4], [4]]
    assert _grid_weights(support, support) == 12 <= W_MAX
    assert _no_dead_lines(support, support)

    card, cohort = idx % 20, idx // 20
    occupancies = ((2, 1, 1), (3, 1, 0), (2, 2, 0))
    profiles = ((0, 0), (0, 1), (1, 1),
                (0, 2), (1, 2), (2, 2))
    plans = ((0, 0), (0, 1), (0, 5), (1, 4))
    pa, pb = plans[card // 5]
    gap_ratio = (4, 7, 9, 11, 13)[card % 5]
    triangle = (4, 7, 9)[cohort % 3]
    seed = rng.randrange(1 << 62)

    # These screening-only galleries have q rows and one column.
    # Consequently their unused check matrices remain narrow.
    # Every permutation block comes from the authoritative constructor.
    gallery = [[[g]] for g in G.elements]
    zero = [[[]]]
    _, za, _ = lifted_matrices(G, gallery, zero)
    xb, _, _ = lifted_matrices(G, zero, gallery)
    blocks = (
        [[(row >> ((j + 1) * q)) & mask for row in za]
         for j in range(q)],
        [[(row >> ((j + 1) * q)) & mask for row in xb]
         for j in range(q)],
    )
    del za, xb, gallery

    def draw(profile, delta=None):
        if delta is None:
            delta = rng.randrange(1, 97)
        grid = []
        for j, kind in enumerate(profiles[profile]):
            fibers = list(range(3))
            rng.shuffle(fibers)
            cell = []
            core = True
            for b, size in zip(fibers, occupancies[kind]):
                if not size:
                    continue
                origin = rng.randrange(97)
                if size == 1:
                    rotations = [origin]
                else:
                    # The first nontrivial fiber in each cell has a
                    # prescribed coupled gap. Other fibers remain free.
                    gap = ((delta if j == 0 else delta * gap_ratio) % 97
                           if core else rng.randrange(1, 97))
                    rotations = [origin, (origin + gap) % 97]
                    if size == 3:
                        rotations.append((origin + triangle * gap) % 97)
                    core = False
                cell.extend((a, b) for a in rotations)
            assert len(cell) == len(set(cell)) == 4
            grid.append([cell])
        return grid, delta

    def assemble(grid, side):
        ids = [[G.idx(g) for g in row[0]] for row in grid]
        rows = []
        for t in range(q):
            row = 0
            for j, cell in enumerate(ids):
                block = 0
                for g in cell:
                    block ^= blocks[side][g][t]
                row |= block << (j * q)
            rows.append(row)
        return rows

    def short_cycles(rows, columns):
        """Exact simple Tanner C6/C8 counts using regular-action roots."""
        checks = [list(_bits(row)) for row in rows]
        neighbors = [list(_bits(column)) for column in columns]
        rooted6 = rooted8 = 0
        for root in (0, q):
            root_checks = set(neighbors[root])
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
                            rooted6 += sum(
                                c3 != c1 and c3 != c2
                                for c3 in root_checks.intersection(
                                    neighbors[end]))
                            previous = paths.setdefault(end, [])
                            for d1, u, d2 in previous:
                                if (u != v and c1 != d1 and c1 != d2
                                        and c2 != d1 and c2 != d2):
                                    rooted8 += 1
                            previous.append((c1, v, c2))
        assert (q * rooted6) % 6 == 0
        assert (q * rooted8) % 4 == 0
        return q * rooted6 // 6, q * rooted8 // 4

    def inspect(grid, side, delta, deep_geometry=False):
        rows = assemble(grid, side)
        quality = _component_quality(rows, width)
        if quality is None:
            return None
        cycles, columns = quality
        record = {
            "grid": grid, "delta": delta, "rows": rows,
            "columns": columns, "cycles": cycles,
        }
        if deep_geometry:
            c6, c8 = short_cycles(rows, columns)
            record["cycles"] = (cycles[0], c6, c8)
        return record

    def legal_rank(record):
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
        return True

    def kernel_search(record, diagnostic, trials):
        """Return an actual classical kernel witness, not a certificate."""
        columns = record["columns"]
        best, witness = width + 1, None
        order = list(range(width))

        def observe(word):
            nonlocal best, witness
            if word:
                weight = word.bit_count()
                if weight < best:
                    best, witness = weight, word

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
                    observe(word)

            light = sorted(basis, key=int.bit_count)[:20]
            for a in light:
                for b in basis:
                    observe(a ^ b)
            diagnostic.shuffle(basis)
            for initial in light[:8]:
                word = initial
                weight = word.bit_count()
                for sweep in range(3):
                    changed = False
                    for other in basis:
                        candidate = word ^ other
                        if candidate and candidate.bit_count() < weight:
                            word, weight = candidate, candidate.bit_count()
                            changed = True
                    observe(word)
                    if not changed:
                        break
            if len(basis) >= 3:
                for _ in range(128):
                    a, b, c = diagnostic.sample(basis, 3)
                    observe(a ^ b ^ c)
            if best <= 24:
                break
        if witness is None:
            raise ValueError("no transposed component kernel witness")
        assert all((witness & row).bit_count() % 2 == 0
                   for row in record["rows"])
        return best, witness

    def syndrome(columns, word):
        value = 0
        for c in _bits(word):
            value ^= columns[c]
        return value

    def component_pool(profile, side):
        bank = set()
        records = []
        signatures = set()

        def signature(grid):
            return tuple(tuple(sorted(G.idx(g) for g in row[0]))
                         for row in grid)

        def observe(record, trials, offset):
            diagnostic = random.Random(seed + 100003 * side + offset)
            reading, word = kernel_search(record, diagnostic, trials)
            record["d"] = min(record.get("d", width + 1), reading)
            record["word"] = word
            bank.add(word)
            if len(bank) > 48:
                keep = sorted(bank, key=int.bit_count)[:32]
                bank.clear()
                bank.update(keep)

        def effective(record):
            reading = record["d"]
            for word in bank:
                if word.bit_count() < reading:
                    if syndrome(record["columns"], word) == 0:
                        reading = word.bit_count()
            return reading

        def merit(record):
            c4, c6, c8 = record["cycles"]
            geometry = ((c4, c6, c8) if cohort % 2 == 0
                        else (c4, 8 * c6 + c8, c6))
            return (effective(record),
                    tuple(-x for x in geometry), record["rank"])

        for _ in range(40):
            grid, delta = draw(profile)
            sig = signature(grid)
            if sig in signatures:
                continue
            signatures.add(sig)
            record = inspect(grid, side, delta)
            if record is not None:
                records.append(record)
        records.sort(key=lambda r: r["cycles"])

        selected = []
        for record in records:
            if not legal_rank(record):
                continue
            c6, c8 = short_cycles(record["rows"], record["columns"])
            record["cycles"] = (record["cycles"][0], c6, c8)
            observe(record, 8, 17)
            selected.append(record)
            if len(selected) == 6:
                break
        if not selected:
            raise ValueError("no legal transposed component shortlist")
        selected.sort(key=merit, reverse=True)
        current = selected[0]

        # Replace an entire cell actually used by a surviving witness.
        # The normal-gap relation and occupancy card remain unchanged.
        for step in range(24):
            active = [w for w in bank
                      if syndrome(current["columns"], w) == 0]
            word = (min(active, key=int.bit_count)
                    if active else current["word"])
            used = sorted({c // q for c in _bits(word)})
            j = rng.choice(used) if used else rng.randrange(2)
            fresh, delta = draw(profile, current["delta"])
            grid = [[list(row[0])] for row in current["grid"]]
            grid[j] = [list(fresh[j][0])]
            sig = signature(grid)
            if sig in signatures:
                continue
            signatures.add(sig)
            proposed = inspect(grid, side, delta, True)
            if proposed is None or not legal_rank(proposed):
                continue
            observe(proposed, 3, 101 + step)
            selected.append(proposed)
            pk, ck = merit(proposed), merit(current)
            if pk > ck or (pk[0] == ck[0] and rng.random() < 0.08):
                current = proposed

        selected.sort(key=merit, reverse=True)
        finalists = selected[:3]
        for record in finalists:
            # Independent diagnostics can only lower the retained reading.
            observe(record, 48 + 8 * (cohort % 3), 700001)
            observe(record, 16, 1700003)
        finalists.sort(key=merit, reverse=True)
        for record in finalists:
            record["effective"] = effective(record)
        return finalists

    aa, bb = component_pool(pa, 0), component_pool(pb, 1)
    proposals = [(a, b) for a in aa for b in bb]
    proposals.sort(key=lambda ab: (
        min(ab[0]["effective"], ab[1]["effective"]),
        ab[0]["effective"] + ab[1]["effective"],
        tuple(-(x + y) for x, y in
              zip(ab[0]["cycles"], ab[1]["cycles"]))), reverse=True)

    for a, b in proposals:
        A, B = a["grid"], b["grid"]
        hx, hz, n = lifted_matrices(G, A, B)
        assert n == 5 * q == 1455
        # Confirm the cached linear assemblies against the final full lift.
        ar = [
            sum(((hz[t] >> (q + i * 2 * q)) & mask) << (i * q)
                for i in range(2))
            for t in range(q)
        ]
        br = [(hx[t] >> q) & ((1 << width) - 1)
              for t in range(q)]
        assert ar == a["rows"] and br == b["rows"]
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if not (k >= q and 7 * n <= 40 * k < 9 * n):
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
            "card": (pa, pb, gap_ratio, triangle),
            "component_readings": (a["effective"], b["effective"]),
            "component_cycles": (a["cycles"], b["cycles"]),
            "cell_ranks": (a["cell_ranks"], b["cell_ranks"]),
        }
    raise ValueError("new single-column pairs missed full validity gates")


def CELL(rng, G, side, i, j, ctx):
    """Return every selected single-column cell without further randomization."""
    return list(ctx[side][i][j])

# REFUTED LEDGER (append-only; one line per hypothesis killed by
# measured evidence - never delete or rewrite existing lines):
# (empty)
# REFUTED: Noncommuting binomials, one trinomial and exact kernel-4 exclusion suffice to suppress the parent's light logical family: all five measured Z27:Z3 (1,4)x(1,4) codes instead had distance reading 5 and hundreds of sector-1 minimum witnesses of each type.
# REFUTED: The parent's degree-three allocations match its all-four-term proxy performance on this bed: every slot with idx%4!=0 had d2s<=36, and all five measured winners used four terms in every cell.
# REFUTED: The parent's six-trial component selection reliably prevents sub-50 logicals: slot 48 fell from d2s=67 to d30s=42, exposing a sector-1 Z witness confined to Bcol0 across both Acols.
# REFUTED: Independent eight-cycle/component shortlists make readings above 70 stable on this group: shown Program 1 slot 36 fell from d2s=74 to d30s=30, with sector-1 Z witnesses at Bcol1 across both Acols.
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
